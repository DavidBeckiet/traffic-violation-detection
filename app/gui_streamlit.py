import streamlit as st
import cv2
import os
import numpy as np
import threading
from app.process_video import process_video  # ⚠️ hoặc "from process_video import process_video" nếu bạn dùng cách 1 ở trên

# ===============================
# ⚙️ Cấu hình giao diện
# ===============================
st.set_page_config(page_title="Traffic Red Light Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚦 Traffic Red Light Violation Detection System</h1>", unsafe_allow_html=True)

# Tạo 2 cột: video bên trái, danh sách vi phạm bên phải
col1, col2 = st.columns([3, 1])

# Biến toàn cục để lưu frame hiện tại
frame_placeholder = col1.empty()
violation_placeholder = col2.empty()
stop_flag = threading.Event()
violations = []


# ===============================
# 🎞️ Hàm callback hiển thị video
# ===============================
def update_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)


# ===============================
# 🧠 Hàm chạy xử lý video
# ===============================
def process_thread(video_path):
    process_video(video_path, display=False, frame_callback=update_frame, save_output=True)
    st.toast("✅ Xử lý xong video!", icon="🎉")


# ===============================
# 🧩 Giao diện chính
# ===============================
with st.sidebar:
    st.header("📂 Input Video")
    uploaded_file = st.file_uploader("Upload video (mp4, avi)", type=["mp4", "avi"])
    start_button = st.button("▶️ Bắt đầu xử lý")

# Danh sách vi phạm
col2.markdown("### 📸 Danh sách vi phạm")

# Khi người dùng bấm nút
if start_button:
    if uploaded_file is None:
        st.warning("⚠️ Vui lòng upload video trước khi bắt đầu!")
        st.stop()

    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", uploaded_file.name)

    # Lưu video tạm
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"📥 Đã tải video: {uploaded_file.name}")
    stop_flag.clear()

    # Khởi chạy xử lý trong luồng riêng
    threading.Thread(target=process_thread, args=(video_path,), daemon=True).start()

# Hiển thị danh sách ảnh vi phạm (nếu có)
violation_dir = "output/violations"
if os.path.exists(violation_dir):
    images = [os.path.join(violation_dir, img) for img in os.listdir(violation_dir) if img.lower().endswith((".jpg", ".png"))]
    if len(images) > 0:
        with col2:
            for img_path in sorted(images, reverse=True):
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
