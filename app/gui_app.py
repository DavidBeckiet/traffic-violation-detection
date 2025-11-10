import streamlit as st
import tempfile
import os
import cv2
from app.process_video import process_video
from app.ui_components import setup_page_style, show_header, show_violation_card, show_video_section

# ==========================
# ⚙️ Cấu hình trang
# ==========================
st.set_page_config(
    page_title="Traffic Violation Detection 🚦",
    page_icon="🚗",
    layout="wide"
)

# Giao diện nền + style
setup_page_style()

# ==========================
# 🧭 Sidebar điều hướng
# ==========================
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt hệ thống")
    st.markdown("Chọn video cần kiểm tra và bắt đầu nhận diện.")
    uploaded_video = st.file_uploader("🎞️ Tải video lên", type=["mp4", "avi", "mov"])
    st.divider()
    st.info("💡 Hệ thống nhận diện vượt đèn đỏ, biển số và trạng thái đèn tự động.")

# ==========================
# 🏁 Header chính
# ==========================
show_header()

# ==========================
# 🧩 Bố cục hiển thị
# ==========================
col1, col2 = st.columns([3, 1], gap="large")

with col1:
    frame_placeholder = show_video_section()

with col2:
    st.subheader("🚨 Danh sách vi phạm")
    violation_list = st.empty()
    detected_violations = []

# ==========================
# 📁 Thư mục output
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLATIONS_DIR = os.path.join(BASE_DIR, "..", "output", "violations")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# ==========================
# 🚀 Xử lý video
# ==========================
if uploaded_video:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    st.video(video_path)
    st.markdown("---")

    st.markdown("### 🚦 Sẵn sàng phân tích video của bạn!")

    # Giao diện nút trung tâm
    st.markdown(
        """
        <div style='text-align:center;'>
            <p style='color:#555;'>Nhấn nút bên dưới để bắt đầu quá trình nhận diện vi phạm.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    start_btn = st.button("🚀 Bắt đầu phát hiện vi phạm", use_container_width=True)

    if start_btn:
        st.info("⏳ Hệ thống đang xử lý video... Vui lòng chờ trong giây lát...")
        progress_bar = st.progress(0)
        frame_count = 0

        def update_frame(frame):
            global frame_count, detected_violations
            frame_count += 1

            # Hiển thị video frame (chậm 1 nhịp để nhẹ hơn)
            if frame_count % 3 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb,
                    caption=f"Khung hình {frame_count}",
                    channels="RGB",
                    use_container_width=True
                )

            # Cập nhật tiến trình giả lập
            progress_bar.progress(min(1.0, frame_count / 200))

            # Cập nhật danh sách vi phạm
            try:
                all_files = sorted(
                    [os.path.join(VIOLATIONS_DIR, f)
                     for f in os.listdir(VIOLATIONS_DIR)
                     if f.lower().endswith((".jpg", ".png"))],
                    key=os.path.getmtime, reverse=True
                )
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc file vi phạm: {e}")
                all_files = []

            grouped = {}
            for f in all_files:
                fname = os.path.basename(f)
                base = fname.split("_crop")[0] if "_crop" in fname else fname.split("_context")[0]
                if "_crop" in fname:
                    grouped.setdefault(base, {})["crop"] = f
                elif "_context" in fname:
                    grouped.setdefault(base, {})["context"] = f

            detected_violations = list(grouped.items())[:5]

            with violation_list.container():
                if detected_violations:
                    st.markdown("### 📸 Các vi phạm gần đây:")
                    for vid, imgs in detected_violations:
                        show_violation_card(vid, imgs)
                else:
                    st.success("✅ Chưa phát hiện vi phạm nào.")

        # Chạy pipeline
        process_video(video_path, display=False, frame_callback=update_frame)

        progress_bar.progress(1.0)
        st.success("✅ Quá trình phát hiện hoàn tất! Ảnh vi phạm được lưu trong `output/violations/`.")
        st.toast("🎯 Hoàn tất! Xem danh sách vi phạm bên phải 👉", icon="🚦")

else:
    st.warning("⬆️ Vui lòng tải video lên để bắt đầu quá trình nhận diện.")
