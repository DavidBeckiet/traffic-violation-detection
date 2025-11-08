import streamlit as st
import tempfile
import os
import cv2
import threading
from app.process_video import process_video

# ==========================
# ⚙️ Cấu hình giao diện chính
# ==========================
st.set_page_config(
    page_title="Traffic Violation Detection 🚗💡",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Traffic Violation Detection System")
st.markdown("""
### Hệ thống giám sát vượt đèn đỏ
Upload video, xem trực tiếp kết quả nhận diện **vượt đèn đỏ**, **biển số**, **vạch dừng** và **trạng thái đèn**.
""")

# ==========================
# 📤 Upload video đầu vào
# ==========================
uploaded_video = st.file_uploader("📤 Chọn video cần kiểm tra", type=["mp4", "avi", "mov"])

# ==========================
# 🧩 Cấu trúc giao diện
# ==========================
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🎥 Video Realtime")
    frame_placeholder = st.empty()

with col2:
    st.subheader("🚨 Danh sách vi phạm")
    violation_list = st.empty()
    detected_plates = []

# Tạo thư mục lưu vi phạm (nếu chưa có)
violations_dir = "output/violations"
os.makedirs(violations_dir, exist_ok=True)

# ==========================
# 🚀 Xử lý video
# ==========================
if uploaded_video:
    # Lưu tạm file video upload
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    # Hiển thị video gốc trước khi xử lý
    st.video(video_path)

    if st.button("🚀 Bắt đầu phát hiện vi phạm"):
        st.info("⏳ Đang xử lý video... Vui lòng chờ...")
        frame_count = 0
        violation_count = 0

        def update_frame(frame):
            """Callback được gọi liên tục từ process_video"""
            global frame_count, violation_count, detected_plates

            frame_count += 1

            # Hiển thị frame
            if frame_count % 2 == 0:  # giảm lag bằng cách hiển thị mỗi 2 frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb,
                    caption=f"Frame {frame_count}",
                    channels="RGB",
                    use_column_width=True
                )

            # Cập nhật danh sách vi phạm
            latest_violations = [
                os.path.basename(f)
                for f in sorted(os.listdir(violations_dir), reverse=True)
                if f.lower().endswith((".jpg", ".png"))
            ]
            detected_plates = latest_violations[:8]

            # Cập nhật danh sách hiển thị
            with violation_list.container():
                if detected_plates:
                    st.markdown("### 📸 Các vi phạm gần đây:")
                    for file in detected_plates:
                        img_path = os.path.join(violations_dir, file)
                        st.image(img_path, caption=file, use_column_width=True)
                    
                else:
                    st.success("✅ Chưa phát hiện vi phạm nào.")

        # Gọi pipeline xử lý video
        process_video(video_path, display=False, frame_callback=update_frame)
        st.success("✅ Hoàn tất phát hiện! Kết quả lưu tại `output/violations/`")

        # Hiển thị thông báo tổng kết
        st.toast(f"🎯 Hoàn tất! Tổng {len(detected_plates)} vi phạm được ghi nhận.", icon="🚦")

else:
    st.info("⬆️ Hãy upload 1 video để bắt đầu quá trình nhận diện.")
