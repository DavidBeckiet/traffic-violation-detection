import streamlit as st
import tempfile
import os
import cv2
import threading
import time
import queue
import ctypes
from app.process_video import process_video
from app.ui_components import setup_page_style, show_header, show_violation_card, show_video_section

# ==========================
# ⚙️ Cấu hình trang
# ==========================
st.set_page_config(page_title="Traffic Violation Detection 🚦", page_icon="🚗", layout="wide")
setup_page_style()

# ==========================
# 🧭 Sidebar
# ==========================
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt hệ thống")
    st.markdown("Chọn video cần kiểm tra và bắt đầu nhận diện.")
    uploaded_video = st.file_uploader("🎞️ Tải video lên", type=["mp4", "avi", "mov"])
    st.divider()
    st.info("💡 Hệ thống nhận diện vượt đèn đỏ, biển số và trạng thái đèn tự động.")

# ==========================
# 🏁 Header
# ==========================
show_header()

col1, col2 = st.columns([3, 1], gap="large")

with col1:
    frame_placeholder = show_video_section()

with col2:
    st.subheader("🚨 Danh sách vi phạm")
    violation_list = st.empty()

# ==========================
# 📁 Thư mục output
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLATIONS_DIR = os.path.join(BASE_DIR, "..", "output", "violations")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# ==========================
# 🧠 Trạng thái toàn cục
# ==========================
frame_queue = queue.Queue(maxsize=3)
stop_flag = threading.Event()
processing_flag = threading.Event()
current_thread = None  # để quản lý thread xử lý video

# ==========================
# 🧩 Callback từ process_video
# ==========================
def update_frame(frame):
    """Nhận frame từ luồng xử lý video và đẩy vào hàng chờ"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not frame_queue.full():
        frame_queue.put(frame_rgb)

# ==========================
# 🚀 Luồng xử lý video chính
# ==========================
def run_detection(video_path):
    try:
        process_video(video_path, frame_callback=update_frame, display=False, stop_flag=stop_flag)
    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý: {e}")
    finally:
        processing_flag.clear()
        st.toast("🎯 Hoàn tất hoặc dừng xử lý!", icon="🚦")

# ==========================
# 💥 Hàm dừng cứng thread
# ==========================
def kill_thread(thread):
    """Dừng cứng một thread bằng cách ném SystemExit"""
    if not thread:
        return
    try:
        tid = thread.ident
        if tid is None:
            return
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(SystemExit))
        if res == 0:
            st.warning("⚠️ Không tìm thấy thread cần dừng.")
        elif res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), 0)
            st.error("⚠️ Lỗi dừng thread: nhiều thread bị ảnh hưởng.")
    except Exception as e:
        st.error(f"❌ Dừng thread thất bại: {e}")

# ==========================
# 🧭 Giao diện điều khiển
# ==========================
if uploaded_video:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    st.video(video_path)
    st.markdown("---")

    start, stop = st.columns(2)
    with start:
        start_btn = st.button("🚀 Bắt đầu nhận diện", use_container_width=True)
    with stop:
        stop_btn = st.button("🛑 Dừng lại", use_container_width=True)

    # --- Khi bấm Bắt đầu ---
    if start_btn and not processing_flag.is_set():
        stop_flag.clear()
        processing_flag.set()
        st.info("⏳ Hệ thống đang xử lý video...")
        current_thread = threading.Thread(target=run_detection, args=(video_path,), daemon=True)
        current_thread.start()

    # --- Khi bấm Dừng lại ---
    if stop_btn and processing_flag.is_set():
        st.warning("🛑 Đang dừng xử lý video...")
        stop_flag.set()
        processing_flag.clear()
        kill_thread(current_thread)
        current_thread = None

    # --- Hiển thị video và danh sách vi phạm ---
    while processing_flag.is_set():
        try:
            frame = frame_queue.get(timeout=0.2)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)
        except queue.Empty:
            pass

        # Cập nhật danh sách vi phạm
        files = sorted(
            [f for f in os.listdir(VIOLATIONS_DIR) if f.lower().endswith((".jpg", ".png"))],
            key=lambda x: os.path.getmtime(os.path.join(VIOLATIONS_DIR, x)),
            reverse=True
        )

        grouped = {}
        for f in files:
            base = f.split("_crop")[0] if "_crop" in f else f.split("_context")[0]
            if "_crop" in f:
                grouped.setdefault(base, {})["crop"] = os.path.join(VIOLATIONS_DIR, f)
            elif "_context" in f:
                grouped.setdefault(base, {})["context"] = os.path.join(VIOLATIONS_DIR, f)

        with violation_list.container():
            if grouped:
                st.markdown("### 📸 Các vi phạm gần đây:")
                for vid, imgs in list(grouped.items())[:5]:
                    show_violation_card(vid, imgs)
            else:
                st.success("✅ Chưa phát hiện vi phạm nào.")

        time.sleep(0.05)

else:
    st.warning("⬆️ Vui lòng tải video lên để bắt đầu quá trình nhận diện.")
