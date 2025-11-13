import time
import streamlit as st
import pyrebase
import firebase_admin
import requests
from dataclasses import dataclass
from typing import List
import math
import random
from geopy.geocoders import Nominatim
from firebase_admin import credentials, firestore
from firebase_admin import auth as admin_auth
from collections import deque
from datetime import datetime, timezone
from ollama import Client
from streamlit_extras.stylable_container import stylable_container

# ===================== MÔ-ĐUN THUẬT TOÁN GỢI Ý NƠI Ở =====================

@dataclass
class Accommodation:
    """
    Đại diện cho 1 nơi ở sau khi đã nạp từ API OpenStreetMap/Overpass.
    (price, rating hiện tại có thể là giá trị giả lập trong bản demo.)
    """
    id: str
    name: str
    city: str
    type: str           # hotel / hostel / apartment / ...
    price: float        # giá ước lượng VND/đêm
    stars: float        # 0–5
    rating: float       # 0–10
    capacity: int       # sức chứa tối đa
    amenities: List[str]
    address: str
    lon: float
    lat: float
    distance_km: float  # khoảng cách tới tâm thành phố (km)


@dataclass
class SearchQuery:
    """
    Gói toàn bộ input người dùng cho thuật toán gợi ý.
    Sau này ta sẽ build SearchQuery từ form trên web.
    """
    city: str                      # tên thành phố điểm đến
    group_size: int                # số người
    price_min: float               # ngân sách tối thiểu (cho 1 đêm)
    price_max: float               # ngân sách tối đa
    types: List[str]               # loại chỗ ở mong muốn: ["hotel","homestay",...]
    rating_min: float              # rating tối thiểu (0–10)
    amenities_required: List[str]  # tiện ích bắt buộc (phải có)
    amenities_preferred: List[str] # tiện ích ưu tiên (có thì cộng điểm)
    radius_km: float               # bán kính tìm kiếm quanh thành phố (km)

def filter_by_constraints(accommodations: List[Accommodation], q: SearchQuery) -> List[Accommodation]:
    """
    Lọc danh sách nơi ở theo các ràng buộc cứng:
    - Khoảng giá
    - Sức chứa
    - Loại chỗ ở
    - Rating tối thiểu
    - Tiện ích bắt buộc

    Nếu không thỏa một điều kiện nào thì nơi ở đó bị loại luôn.
    """
    filtered: List[Accommodation] = []

    for a in accommodations:
        # 1. Giá: nằm trong [price_min, price_max]
        if a.price < q.price_min or a.price > q.price_max:
            continue

        # 2. Sức chứa: phải đủ cho group_size
        if a.capacity < q.group_size:
            continue

        # 3. Loại chỗ ở: nếu user chọn types thì phải match 1 trong các loại đó
        if q.types and (a.type not in q.types):
            continue

        # 4. Rating tối thiểu (0–10)
        if a.rating < q.rating_min:
            continue

        # 5. Tiện ích bắt buộc: mỗi tiện ích required phải có trong a.amenities
        if any(req.lower() not in [am.lower() for am in a.amenities] for req in q.amenities_required):
            continue

        filtered.append(a)

    return filtered

def clamp01(x: float) -> float:
    """Giới hạn giá trị trong [0,1] để tránh <0 hoặc >1."""
    return max(0.0, min(1.0, x))

#mô-đun “Scoring & Ranking module”
def score_accommodation(a: Accommodation, q: SearchQuery) -> float:
    """
    Tính điểm xếp hạng cho 1 nơi ở theo nhiều tiêu chí.

    - S_price  : 1 nếu giá gần mức mong muốn, 0 nếu chênh lệch quá lớn.
    - S_stars  : sao / 5.
    - S_rating : rating / 10.
    - S_amen   : tỉ lệ tiện ích yêu cầu + ưu tiên được đáp ứng.
    - S_dist   : càng gần tâm city (so với bán kính radius_km) thì điểm càng cao.

    Tổng hợp: 
    Score = 0.25*S_price + 0.20*S_stars + 0.25*S_rating + 0.20*S_amen + 0.10*S_dist
    """

    # ----- 1. Điểm GIÁ -----
    Pmin, Pmax = q.price_min, q.price_max
    if Pmax > Pmin:
        Pc = (Pmin + Pmax) / 2.0                  # giá mục tiêu ở giữa khoảng
        denom = max(1.0, (Pmax - Pmin) / 2.0)     # "nửa khoảng" để chuẩn hoá
        S_price = 1.0 - min(abs(a.price - Pc) / denom, 1.0)
    else:
        # Nếu user không đặt khoảng giá rõ ràng, cho tất cả = 1
        S_price = 1.0

    # ----- 2. Điểm SAO & RATING -----
    S_stars = clamp01(a.stars / 5.0)       # 0–5 sao -> 0–1
    S_rating = clamp01(a.rating / 10.0)    # 0–10 rating -> 0–1

    # ----- 3. Điểm TIỆN ÍCH -----
    have = set(x.lower() for x in a.amenities)
    req = set(x.lower() for x in q.amenities_required)
    pref = set(x.lower() for x in q.amenities_preferred)

    if req or pref:
        match_req = len(have.intersection(req))
        match_pref = len(have.intersection(pref))

        # required trọng số 1.0, preferred trọng số 0.5
        matched_score = match_req + 0.5 * match_pref
        max_possible = max(1.0, len(req) + 0.5 * len(pref))
        S_amen = matched_score / max_possible
    else:
        S_amen = 1.0  # user không yêu cầu tiện ích gì đặc biệt

    # ----- 4. Điểm KHOẢNG CÁCH -----
    # distance_km: khoảng cách tới tâm thành phố; so với radius_km
    if q.radius_km > 0:
        S_dist = 1.0 - min(a.distance_km / q.radius_km, 1.0)
    else:
        S_dist = 1.0

    # ----- 5. Tổng hợp điểm (có thể chỉnh các trọng số này nếu cần) -----
    score = (
        0.25 * S_price +
        0.20 * S_stars +
        0.25 * S_rating +
        0.20 * S_amen +
        0.10 * S_dist
    )

    return score

def rank_accommodations(accommodations: List[Accommodation], q: SearchQuery, top_k: int = 5):
    """
    Thực hiện:
    - Lọc theo constraints (hard filter).
    - Tính score cho từng nơi ở.
    - Sắp xếp giảm dần theo score và lấy Top K.

    Trả về list các dict:
        { "score": float, "accommodation": Accommodation }
    để phần UI dễ render.
    """
    # 1. Lọc theo ràng buộc cứng
    filtered = filter_by_constraints(accommodations, q)

    if not filtered:
        return []

    # 2. Tính điểm cho từng nơi
    scored = []
    for a in filtered:
        s = score_accommodation(a, q)
        scored.append({
            "score": s,
            "accommodation": a,
        })

    # 3. Sắp xếp giảm dần theo score, nếu bằng nhau thì ưu tiên rating cao hơn
    scored.sort(
        key=lambda item: (item["score"], item["accommodation"].rating),
        reverse=True
    )

    # 4. Lấy Top-K
    return scored[:top_k]
def haversine_km(lon1, lat1, lon2, lat2):
    """
    Tính khoảng cách đường tròn lớn giữa 2 điểm (lat, lon) trên Trái đất, đơn vị km.
    Dùng công thức Haversine.
    """
    R = 6371.0  # bán kính Trái đất (km)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c

def geocode_city(city_name: str):
    """
    Dùng Nominatim để lấy toạ độ (lat, lon) của một thành phố.
    Trả về dict {"name", "lat", "lon"} hoặc None nếu lỗi.
    """
    geocoder = Nominatim(user_agent="smart_tourism_demo")
    try:
        loc = geocoder.geocode(city_name, exactly_one=True, addressdetails=True, language="en")
        if not loc:
            return None
        return {
            "name": loc.address,
            "lat": loc.latitude,
            "lon": loc.longitude,
        }
    except Exception:
        return None

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_osm_accommodations(city_name: str, radius_km: float = 5.0, max_results: int = 50):
    """
    Gọi OpenStreetMap (Overpass API) để lấy danh sách nơi ở quanh một thành phố.

    Bước:
    1) Geocode tên thành phố -> (lat_city, lon_city)
    2) Dùng Overpass query lấy các node/way/relation có tourism=hotel|hostel|guest_house|apartment
       trong bán kính radius_km quanh city.
    3) Convert về list[Accommodation], trong đó:
       - price, rating, capacity, amenities được GIẢ LẬP từ sao + một số tag.
    """

    # ----- 1. Geocode city -----
    city_geo = geocode_city(f"{city_name}, Vietnam")
    if not city_geo:
        return [], None  # không tìm được city

    city_lat = city_geo["lat"]
    city_lon = city_geo["lon"]
    radius_m = int(radius_km * 1000)

    # ----- 2. Overpass query -----
    # Lấy các đối tượng có tourism là hotel, hostel, guest_house hoặc apartment
    query = f"""
    [out:json][timeout:25];
    (
      node["tourism"~"hotel|hostel|guest_house|apartment"](around:{radius_m},{city_lat},{city_lon});
      way["tourism"~"hotel|hostel|guest_house|apartment"](around:{radius_m},{city_lat},{city_lon});
      relation["tourism"~"hotel|hostel|guest_house|apartment"](around:{radius_m},{city_lat},{city_lon});
    );
    out center {max_results};
    """

    resp = requests.post(OVERPASS_URL, data=query)
    resp.raise_for_status()
    data = resp.json()

    elements = data.get("elements", [])
    accommodations: list[Accommodation] = []

    # ----- 3. Duyệt kết quả Overpass & convert -> Accommodation -----
    for el in elements:
        tags = el.get("tags", {})

        # Lấy lat, lon: node có sẵn; way/relation dùng 'center'
        if el["type"] == "node":
            lat = el.get("lat")
            lon = el.get("lon")
        else:
            center = el.get("center") or {}
            lat = center.get("lat")
            lon = center.get("lon")

        if lat is None or lon is None:
            continue  # bỏ qua nếu không có toạ độ

        # Tên chỗ ở
        name = tags.get("name", "Chỗ ở không tên")

        # Thành phố: ưu tiên addr:city, fallback dùng city_name user nhập
        city = tags.get("addr:city", city_name)

        # Loại chỗ ở
        tourism_type = tags.get("tourism", "hotel")  # hotel / hostel / guest_house / apartment
        # Quy ước type đơn giản cho thuật toán
        if tourism_type == "guest_house":
            acc_type = "homestay"
        elif tourism_type == "apartment":
            acc_type = "apartment"
        elif tourism_type == "hostel":
            acc_type = "hostel"
        else:
            acc_type = "hotel"

        # Số sao (nếu OSM có tag 'stars'), mặc định 3
        try:
            stars = float(tags.get("stars", 3))
        except ValueError:
            stars = 3.0

        # GIẢ LẬP GIÁ dựa trên số sao (cho phù hợp thuật toán)
        base_by_star = {1: 300_000, 2: 450_000, 3: 700_000, 4: 1_000_000, 5: 1_500_000}
        base_price = base_by_star.get(int(stars), 700_000)
        # random nhẹ  ±10% cho giống thật
        price = base_price * (0.9 + 0.2 * random.random())

        # GIẢ LẬP RATING 7.0–10.0
        rating = 7.0 + 3.0 * random.random()

        # GIẢ LẬP SỨC CHỨA (cho đơn giản: 2–6 người)
        capacity = 2 + int(random.random() * 4)

        # Tiện ích: map từ một số tag OSM cơ bản
        amenities = []
        internet = tags.get("internet_access")
        if internet in ("wlan", "yes"):
            amenities.append("wifi")
        if tags.get("parking") == "yes":
            amenities.append("parking")
        if tags.get("breakfast") == "yes":
            amenities.append("breakfast")
        if tags.get("swimming_pool") == "yes":
            amenities.append("pool")

        # Địa chỉ hiển thị
        address = tags.get("addr:full") or tags.get("addr:street") or tags.get("addr:housenumber") or city

        # Khoảng cách tới tâm city (km)
        distance_km = haversine_km(city_lon, city_lat, lon, lat)

        acc = Accommodation(
            id=str(el.get("id")),
            name=name,
            city=city,
            type=acc_type,
            price=price,
            stars=stars,
            rating=rating,
            capacity=capacity,
            amenities=amenities,
            address=address,
            lon=lon,
            lat=lat,
            distance_km=distance_km,
        )
        accommodations.append(acc)

    return accommodations, (city_lon, city_lat)

def recommend_top5_from_api(q: SearchQuery):
    """
    Hàm tiện dụng:
    - Dùng city & radius trong SearchQuery để gọi Overpass lấy danh sách nơi ở.
    - Dùng rank_accommodations(...) để lọc + chấm điểm + lấy Top 5.

    Trả về:
      - danh sách top-5 (mỗi phần tử là dict {score, accommodation})
      - toạ độ tâm city (lon, lat) để sau này vẽ map
    """
    accommodations, city_center = fetch_osm_accommodations(
        city_name=q.city,
        radius_km=q.radius_km,
        max_results=50,
    )

    if not accommodations:
        return [], city_center

    top5 = rank_accommodations(accommodations, q, top_k=5)
    return top5, city_center


st.set_page_config(page_title="Tourism_Symstem", page_icon="💬")
MODEL = "llama3.2:1b"
client = Client(
    host='http://nrplz-34-187-131-164.a.free.pinggy.link'
)

def ollama_stream(history_messages: list[dict]):
    """
    Stream tokens from Ollama /api/chat. Yields string chunks suitable for st.write_stream.
    """
    print(history_messages)
    response = client.chat(
        model=MODEL,
        messages=history_messages
    )
    return response['message']['content']

def ollama_generate_itinerary(prompt: str):
    """
    Gửi một prompt tạo lịch trình đến Ollama và trả về kết quả.
    Sử dụng logic tương tự như ollama_stream nhưng chỉ với 1 prompt.
    """
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

def save_message(uid: str, role: str, content: str):
    doc = {
        "role": role,
        "content": content,
        "ts": datetime.now(timezone.utc)
    }
    db.collection("chats").document(uid).collection("messages").add(doc)

def load_last_messages(uid: str, limit: int = 8):
    q = (db.collection("chats").document(uid)
        .collection("messages")
        .order_by("ts", direction=firestore.Query.DESCENDING)
        .limit(limit))
    docs = list(q.stream())
    docs.reverse()
    out = []
    for d in docs:
        data = d.to_dict()
        out.append({"role": data.get("role", "assistant"),
                    "content": data.get("content", "")})
    return out

params = st.query_params
raw_token = params.get("id_token")
if isinstance(raw_token, list):
    id_token = raw_token[0]
else:
    id_token = raw_token
    
if id_token and not st.session_state.get("user"):
    id_token = params["id_token"][0]
    try:
        decoded = admin_auth.verify_id_token(id_token)
        st.session_state.user = {
            "email": decoded.get("email"),
            "uid": decoded.get("uid"),
            "idToken": id_token,
        }
        msgs = []
        try:
            msgs = load_last_messages(st.session_state.user["uid"], limit=8)
        except Exception:
            pass
        st.session_state.messages = deque(
            msgs if msgs else [{"role": "assistant", "content": "Xin chào Xin chào 👋! Tôi là Mika. Tôi có thể giúp gì cho bạn?"}],
            maxlen=8
        )
        st.experimental_set_query_params()
        st.success("Đăng nhập Google thành công!")
        st.rerun()
    except Exception as e:
        st.error(f"Xác thực Google thất bại: {e}")


@st.cache_resource
def get_firebase_clients():
    # Pyrebase (Auth)
    firebase_cfg = st.secrets["firebase_client"]
    firebase_app = pyrebase.initialize_app(firebase_cfg)
    auth = firebase_app.auth()

    # Admin (Firestore)
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    return auth, db

auth, db = get_firebase_clients()


if "user" not in st.session_state:
    st.session_state.user = None 

if "messages" not in st.session_state:
    st.session_state.messages = deque([
        {"role": "assistant", "content": "Xin chào Xin chào 👋! Tôi là Mika. Tôi có thể giúp gì cho bạn?"}
    ], maxlen=8)
else:
    if not isinstance(st.session_state.messages, deque):
        st.session_state.messages = deque(st.session_state.messages[-8:], maxlen=8)

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# Lưu kết quả gợi ý nơi ở (Top 5 + thông tin city center) để hiển thị sau
if "accommodation_results" not in st.session_state:
    st.session_state.accommodation_results = None


def login_form():
    st.markdown("<h3 style='text-align: center;'>Đăng nhập</h3>", unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", key="email_login")
        password = st.text_input("Mật khẩu", type="password", key="password_login")
        col1, _, col2 = st.columns([0.75, 0.75, 0.75])
        with col1:
            with stylable_container(
                "black",
                css_styles="""
                button {
                    background-color: #0DDEAA;
                    color: black;
                }""",
            ):
                login = st.form_submit_button("Đăng nhập")
        with col2:
            goto_signup = st.form_submit_button("Chưa có tài khoản? Đăng ký", type="primary")

    if goto_signup:
        st.session_state["show_signup"] = True
        st.session_state["show_login"] = False
        st.rerun()

    if login:
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            # user: dict có idToken, refreshToken, localId (uid), email
            st.session_state.user = {
                "email": email,
                "uid": user["localId"],
                "idToken": user["idToken"]
            }
            # tải lịch sử gần nhất từ Firestore
            msgs = load_last_messages(st.session_state.user["uid"], limit=8)
            if msgs:
                st.session_state.messages = deque(msgs, maxlen=8)
            else:
                st.session_state.messages = deque([
                    {"role": "assistant", "content": "Xin chào Xin chào 👋! Tôi là Mika. Tôi có thể giúp gì cho bạn?"}
                ], maxlen=8)
            st.success("Đăng nhập thành công!")
            st.rerun()
        except Exception as e:
            st.error(f"Đăng nhập thất bại: {e}")

def signup_form():
    st.subheader("Đăng ký")
    with st.form("signup_form", clear_on_submit=False):
        email = st.text_input("Email", key="email_signup")
        password = st.text_input("Mật khẩu (≥6 ký tự)", type="password", key="password_signup")
        col1, _, col2 = st.columns([0.75, 0.75, 0.75])
        with col1:
            with stylable_container(
                "black-1",
                css_styles="""
                button {
                    background-color: #0DD0DE;
                    color: black;
                }""",
            ):
                signup = st.form_submit_button("Tạo tài khoản")
        with col2:
                goto_login = st.form_submit_button("Đã có tài khoản? Đăng nhập", type="primary")

    if goto_login:
        st.session_state["show_signup"] = False
        st.session_state["show_login"] = True
        st.rerun()

    if signup:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success("Tạo tài khoản thành công! Vui lòng đăng nhập.")
            time.sleep(3)
            st.session_state["show_signup"] = False
            st.session_state["show_login"] = True
            st.rerun()
        except Exception as e:
            st.error(f"Đăng ký thất bại: {e}")

@st.dialog("Trợ lý Mika")
def chat_dialog():
    if not st.session_state.user:
        st.info("Bạn cần đăng nhập để chat và lưu lịch sử.")
        return
    
    chat_body = st.container(height=600, border=True)

    def render_history():
        chat_body.empty()
        with chat_body:
            for msg in list(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
    render_history()

    user_input = st.chat_input("Nhập tin nhắn...", key="dialog_input")
        
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_body:
            with st.chat_message("user"):
                st.markdown(user_input)
        save_message(st.session_state.user["uid"], "user", user_input)
        try:
            reply = ollama_stream(st.session_state.messages)
        except requests.RequestException as e:
            st.error(f"Ollama request failed: {e}")
            reply = ""
        st.session_state.messages.append({"role": "assistant", "content": reply})
        save_message(st.session_state.user["uid"], "assistant", reply)
        st.session_state.chat_open = True
        st.rerun()

st.markdown("<h1 style='text-align: center;'>Streamlit Chat + Firebase Login</h1>", unsafe_allow_html=True)

if "show_signup" not in st.session_state:
    st.session_state["show_signup"] = False
if "show_login" not in st.session_state:
    st.session_state["show_login"] = True

if st.session_state.user:
    st.success(f"Đang đăng nhập: {st.session_state.user['email']}")
    _, col2, _ = st.columns([1.3, 0.75, 1])
    with col2:
        if st.button("Đăng xuất", type="primary"):
            st.session_state.user = None
            st.session_state.chat_open = False
            st.rerun()

# --- Bắt đầu: Phần Gợi ý Nơi Ở ---

# Chỉ hiển thị giao diện gợi ý nơi ở khi người dùng đã đăng nhập
if st.session_state.user:
    st.markdown("## 🏨 Gợi ý Nơi Ở Phù Hợp")

    with st.form("accommodation_form"):
        st.write("Nhập nhu cầu nơi ở, hệ thống sẽ gợi ý Top 5 địa điểm phù hợp nhất xung quanh thành phố điểm đến (dữ liệu từ OpenStreetMap).")

        # 1. Thành phố điểm đến
        acc_city = st.text_input("Thành phố Điểm đến", value="Đà Nẵng")

        # 2. Số người
        group_size = st.number_input("Số người", min_value=1, max_value=20, value=2, step=1)

        # 3. Khoảng giá (tính theo 1 đêm, VND)
        col_price_1, col_price_2 = st.columns(2)
        with col_price_1:
            price_min = st.number_input(
                "Giá tối thiểu mỗi đêm (VND)",
                min_value=0,
                value=300_000,
                step=50_000
            )
        with col_price_2:
            price_max = st.number_input(
                "Giá tối đa mỗi đêm (VND)",
                min_value=0,
                value=1_500_000,
                step=50_000
            )

        # 4. Loại hình nơi ở
        types = st.multiselect(
            "Loại hình nơi ở",
            options=["hotel", "homestay", "hostel", "apartment"],
            default=["hotel", "homestay"]
        )

        # 5. Rating tối thiểu & Bán kính tìm kiếm
        col_rating, col_radius = st.columns(2)
        with col_rating:
            rating_min = st.slider("Rating tối thiểu", 0.0, 10.0, 7.5, 0.5)
        with col_radius:
            radius_km = st.slider("Bán kính tìm kiếm quanh thành phố (km)", 1.0, 20.0, 5.0, 1.0)

        # 6. Tiện ích bắt buộc & ưu tiên
        amenities_required = st.multiselect(
            "Tiện ích BẮT BUỘC phải có",
            options=["wifi", "breakfast", "pool", "parking"],
            default=["wifi"]
        )

        amenities_preferred = st.multiselect(
            "Tiện ích ƯU TIÊN (có thì tốt)",
            options=["wifi", "breakfast", "pool", "parking"],
            default=["breakfast", "pool"]
        )

        submit_acc = st.form_submit_button("🔍 Gợi ý Top 5 nơi ở")

        # ===== XỬ LÝ KHI NHẤN NÚT GỢI Ý =====
        if submit_acc:
            if not acc_city.strip():
                st.error("Vui lòng nhập Thành phố Điểm đến.")
            elif price_min > 0 and price_max > 0 and price_min > price_max:
                st.error("Giá tối thiểu phải nhỏ hơn hoặc bằng giá tối đa.")
            else:
                # Tạo SearchQuery từ input người dùng
                q = SearchQuery(
                    city=acc_city.strip(),
                    group_size=int(group_size),
                    price_min=float(price_min),
                    price_max=float(price_max),
                    types=types,
                    rating_min=float(rating_min),
                    amenities_required=amenities_required,
                    amenities_preferred=amenities_preferred,
                    radius_km=float(radius_km),
                )

                with st.spinner("Đang tìm kiếm và xếp hạng các nơi ở phù hợp..."):
                    try:
                        top5, city_center = recommend_top5_from_api(q)
                        st.session_state.accommodation_results = {
                            "query": q,
                            "city_center": city_center,
                            "results": top5
                        }
                    except requests.RequestException as e:
                        st.error(f"Lỗi khi gọi API OpenStreetMap/Overpass: {e}")
                        st.session_state.accommodation_results = None

                # Reload lại để phía dưới dùng session_state hiển thị kết quả
                st.rerun()

    # ===== KHU VỰC HIỂN THỊ KẾT QUẢ GỢI Ý NƠI Ở =====
    results_state = st.session_state.accommodation_results

    if results_state and results_state.get("results"):
        st.markdown("### 🔝 Top 5 nơi ở được đề xuất")

        for item in results_state["results"]:
            a = item["accommodation"]
            score = item["score"]

            st.markdown(f"#### {a.name} ({a.type})")
            st.write(
                f"- Thành phố: **{a.city}**  |  Cách trung tâm: ~**{a.distance_km:.2f} km**"
            )
            st.write(
                f"- Giá ước lượng/đêm: **{int(a.price):,} VND**  |  "
                f"Số sao: **{a.stars}⭐**  |  Rating: **{a.rating:.1f}/10**"
            )
            if a.amenities:
                st.write(f"- Tiện ích: {', '.join(a.amenities)}")
            else:
                st.write("- Tiện ích: (không rõ từ OSM)")
            st.write(f"- Điểm xếp hạng thuật toán: **{score:.3f}**")
            st.markdown("---")

    elif results_state is not None and results_state.get("results") == []:
        st.info("Không có nơi ở nào thỏa điều kiện tìm kiếm hiện tại. Hãy thử nới lỏng tiêu chí.")
else:
    # Nếu chưa đăng nhập thì vẫn giữ logic cũ: hiển thị form đăng ký / đăng nhập
    if st.session_state.get("show_signup", False):
        signup_form()
    elif st.session_state.get("show_login", True):
        login_form()

# --- Kết thúc: Phần Gợi ý Nơi Ở ---

st.markdown("<h5 style='text-align: center;'>Click 💬 để mở hộp thoại chat</h5>", unsafe_allow_html=True)

st.markdown('<div id="fab-anchor"></div>', unsafe_allow_html=True)
with stylable_container(
                "black-3",
                css_styles="""
                button {
                    background-color: #66c334;
                    color: black;
                    width: 704px !important; 
                    height: 30px; 
                }""",
            ):
    fab_clicked = st.button("💬", key="open_chat_fab", help="Mở chat")
    
if fab_clicked:
    st.session_state.chat_open = True
    st.rerun()

if st.session_state.chat_open:
    chat_dialog()


st.markdown("""
<style>
#fab-anchor + div button {
    position: fixed;
    bottom: 16px;
    right: 16px;
    width: 120px !important; 
    height: 60px; 
    border-radius: 50%;
    font-size: 26px; 
    line-height: 1; 
    padding: 0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    z-index: 10000;
}
#fab-anchor + div button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(250,206,175,0.28);
}
</style>
""", unsafe_allow_html=True)
