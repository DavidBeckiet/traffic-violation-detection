import streamlit as st

# ==========================
# 🎨 CSS / Style setup
# ==========================
def setup_page_style():
    """Thêm CSS tùy chỉnh cho toàn bộ ứng dụng."""
    st.markdown("""
        <style>
            .main {
                background: linear-gradient(180deg, #f9fafc 0%, #eef3f8 100%);
            }

            h1 {
                text-align: center;
                color: #0f4c81;
                font-weight: 800;
                font-size: 2.5rem !important;
            }

            div.stButton > button {
                width: 100%;
                background: linear-gradient(90deg, #0f4c81, #1a73e8);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.75rem;
                font-size: 1.1rem;
                font-weight: 600;
                transition: 0.2s;
            }

            div.stButton > button:hover {
                background: linear-gradient(90deg, #1a73e8, #0f4c81);
                transform: scale(1.02);
            }

            .violation-card {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            }

            .violation-card:hover {
                box-shadow: 0 3px 12px rgba(0,0,0,0.12);
                transform: scale(1.01);
                transition: 0.2s ease;
            }
        </style>
    """, unsafe_allow_html=True)


# ==========================
# 🏁 Header
# ==========================
def show_header():
    """Hiển thị phần tiêu đề chính"""
    st.title("🚦 Traffic Violation Detection System")
    st.markdown("""
        <div style="text-align:center;">
            <h3>Hệ thống giám sát vượt đèn đỏ</h3>
            <p style="font-size:1.1rem; color:#555;">
                Upload video, theo dõi kết quả nhận diện <b>vượt đèn đỏ</b>,
                <b>biển số</b>, <b>vạch dừng</b> và <b>trạng thái đèn</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)


# ==========================
# 🎥 Video Section
# ==========================
def show_video_section():
    """Tạo khung hiển thị video realtime."""
    st.subheader("🎥 Video Realtime")
    return st.empty()


# ==========================
# 🚗 Violation Card
# ==========================
def show_violation_card(vid, imgs):
    """Hiển thị 1 card chứa thông tin xe vi phạm."""
    with st.container():
        st.markdown(f"<div class='violation-card'><b>🚗 {vid}</b>", unsafe_allow_html=True)
        cols = st.columns(2)
        if "crop" in imgs:
            with cols[0]:
                st.image(imgs["crop"], caption="📍 Xe vi phạm", use_container_width=True)
        if "context" in imgs:
            with cols[1]:
                st.image(imgs["context"], caption="📷 Toàn cảnh", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
