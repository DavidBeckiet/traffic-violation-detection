import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import threading
import time
import queue
import ctypes
import glob
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
    progress_placeholder = st.empty()

with col2:
    st.subheader("🚨 Danh sách vi phạm")
    violation_list = st.empty()

# ==========================
# 📁 Thư mục output
# ==========================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIOLATIONS_DIR = os.path.join(ROOT_DIR, "output", "violations")
UPLOADS_DIR = os.path.join(ROOT_DIR, "uploads")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ==========================
# 🧠 Trạng thái toàn cục
# ==========================
frame_queue = queue.Queue(maxsize=3)
stop_flag = threading.Event()
processing_flag = threading.Event()
current_thread = None
fps_display = st.empty()

# ==========================
# 🧩 Callback từ process_video
# ==========================
def update_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not frame_queue.full():
        frame_queue.put(frame_rgb)

# ==========================
# 🚀 Luồng xử lý video chính
# ==========================
def run_detection(video_path):
    try:
        result = process_video(video_path, frame_callback=update_frame, display=False, stop_flag=stop_flag)
        if result:
            st.session_state["last_video_result"] = result
    except Exception as e:
        st.session_state["error"] = str(e)
    finally:
        processing_flag.clear()

# ==========================
# 💥 Hàm dừng cứng thread
# ==========================
def kill_thread(thread):
    if not thread:
        return
    try:
        tid = thread.ident
        if tid is None:
            return
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), 0)
    except:
        pass

# ==========================
# 🧭 Giao diện điều khiển
# ==========================
if uploaded_video:

    # 🔒 Lưu video upload
    video_path = os.path.join(UPLOADS_DIR, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # ❌ BỎ video gốc — theo yêu cầu
    # st.video(video_path)
    # st.markdown("---")

    start, stop = st.columns(2)
    with start:
        start_btn = st.button("🚀 Bắt đầu nhận diện", use_container_width=True)
    with stop:
        stop_btn = st.button("🛑 Dừng lại", use_container_width=True)

    # --- Start ---
    if start_btn and not processing_flag.is_set():
        stop_flag.clear()
        processing_flag.set()
        st.info(f"⏳ Đang xử lý video: **{os.path.basename(video_path)}**")
        current_thread = threading.Thread(target=run_detection, args=(video_path,), daemon=True)
        current_thread.start()

    # --- Stop ---
    if stop_btn and processing_flag.is_set():
        st.warning("🛑 Đang dừng xử lý video...")
        stop_flag.set()
        processing_flag.clear()
        kill_thread(current_thread)
        current_thread = None

    # ==========================
    # 🔁 Cập nhật GUI realtime
    # ==========================
    last_time = time.time()
    frame_count = 0

    while processing_flag.is_set():

        # ====== Cập nhật frame ======
        try:
            frame = frame_queue.get(timeout=0.2)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            frame_count += 1
            now = time.time()
            if now - last_time >= 1:
                fps = frame_count / (now - last_time)
                progress_placeholder.info(f"🎞️ FPS: **{fps:.1f}**")
                frame_count = 0
                last_time = now

        except queue.Empty:
            pass

        # ====== Hiển thị vi phạm ======
        files = sorted(
            glob.glob(os.path.join(VIOLATIONS_DIR, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(VIOLATIONS_DIR, "**", "*.png"), recursive=True),
            key=lambda x: os.path.getmtime(x),
            reverse=True
        )

        grouped = {}

        for f in files:
            name = os.path.basename(f)

            if "_crop" in name:
                k = name.split("_crop")[0]
                grouped.setdefault(k, {})["crop"] = f

            elif "_ctx" in name:
                k = name.split("_ctx")[0]
                grouped.setdefault(k, {})["context"] = f

        with violation_list.container():
            if grouped:
                st.markdown("### 📸 Các vi phạm gần đây:")
                for vid, imgs in list(grouped.items())[:5]:
                    show_violation_card(vid, imgs)
            else:
                st.success("✅ Chưa phát hiện vi phạm nào.")

        time.sleep(0.05)

    # ==========================
    # Kết thúc xử lý
    # ==========================
    if "last_video_result" in st.session_state:
        result = st.session_state.pop("last_video_result")
        st.success(f"🎉 Hoàn tất xử lý video — phát hiện {len(result['violations'])} vi phạm.")
        st.write(f"📁 Video output: `{result['output_path']}`")

    if "error" in st.session_state:
        st.error(f"❌ Lỗi xử lý: {st.session_state.pop('error')}")

else:
    st.warning("⬆️ Vui lòng tải video lên để bắt đầu nhận diện.")
