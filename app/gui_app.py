import streamlit as st
import tempfile
import os
import cv2
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
# 🧩 Bố cục giao diện
# ==========================
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🎥 Video Realtime")
    frame_placeholder = st.empty()

with col2:
    st.subheader("🚨 Danh sách vi phạm")
    violation_list = st.empty()
    detected_violations = []

# ==========================
# 🗂️ Chuẩn bị thư mục output (đường dẫn tuyệt đối)
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIOLATIONS_DIR = os.path.join(BASE_DIR, "..", "output", "violations")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

print(f"📂 Ảnh vi phạm sẽ được lưu tại: {os.path.abspath(VIOLATIONS_DIR)}")

# ==========================
# 🚀 Xử lý video
# ==========================
if uploaded_video:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    st.video(video_path)

    if st.button("🚀 Bắt đầu phát hiện vi phạm"):
        st.info("⏳ Đang xử lý video... Vui lòng chờ...")
        frame_count = 0

        def update_frame(frame):
            global frame_count, detected_violations
            frame_count += 1

            # Hiển thị frame (mỗi 2 frame)
            if frame_count % 2 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb,
                    caption=f"Frame {frame_count}",
                    channels="RGB",
                    use_container_width=True
                )

            # Lấy danh sách file vi phạm
            try:
                all_files = sorted(
                    [os.path.join(VIOLATIONS_DIR, f)
                     for f in os.listdir(VIOLATIONS_DIR)
                     if f.lower().endswith((".jpg", ".png"))],
                    key=os.path.getmtime,
                    reverse=True
                )
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc file vi phạm: {e}")
                all_files = []

            # Gom nhóm crop + context
            grouped = {}
            for f in all_files:
                fname = os.path.basename(f)
                if "_crop" in fname:
                    vid = fname.split("_crop")[0]
                    grouped.setdefault(vid, {})["crop"] = f
                elif "_context" in fname:
                    vid = fname.split("_context")[0]
                    grouped.setdefault(vid, {})["context"] = f

            detected_violations = list(grouped.items())[:5]  # hiển thị tối đa 5

            # Hiển thị danh sách vi phạm
            with violation_list.container():
                if detected_violations:
                    st.markdown("### 📸 Các vi phạm gần đây:")
                    for vid, imgs in detected_violations:
                        st.markdown(f"**🚗 {vid}**")
                        cols = st.columns(2)
                        if "crop" in imgs:
                            with cols[0]:
                                st.image(imgs["crop"], caption="📍 Xe vi phạm", use_container_width=True)
                        if "context" in imgs:
                            with cols[1]:
                                st.image(imgs["context"], caption="📷 Toàn cảnh", use_container_width=True)
                        st.divider()
                else:
                    st.success("✅ Chưa phát hiện vi phạm nào.")

        # Gọi pipeline xử lý video
        process_video(video_path, display=False, frame_callback=update_frame)

        st.success("✅ Hoàn tất phát hiện! Kết quả lưu tại `output/violations/`")
        st.toast("🎯 Hoàn tất! Kiểm tra danh sách vi phạm bên phải 👉", icon="🚦")

else:
    st.info("⬆️ Hãy upload 1 video để bắt đầu quá trình nhận diện.")
