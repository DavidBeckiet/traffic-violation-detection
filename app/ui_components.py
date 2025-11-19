import streamlit as st

# ==========================
# 🎨 CSS / Style setup
# ==========================
def setup_page_style():
    """Thêm CSS tùy chỉnh cho toàn bộ ứng dụng."""
    st.markdown("""
        <style>

            /* ===== Background tổng thể ===== */
            .main {
                background: linear-gradient(180deg, #fdfdfd 0%, #eef3f8 100%);
            }

            /* ===== Header ===== */
            h1 {
                text-align: center;
                color: #0f4c81;
                font-weight: 900;
                font-size: 2.7rem !important;
                letter-spacing: 0.5px;
            }

            h3 {
                color: #0f4c81;
                font-weight: 700;
            }

            /* ===== Button ===== */
            div.stButton > button {
                width: 100%;
                background: linear-gradient(90deg, #0f4c81, #1a73e8);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.8rem;
                font-size: 1.1rem;
                font-weight: 600;
                transition: all 0.25s ease;
            }

            div.stButton > button:hover {
                background: linear-gradient(90deg, #1a73e8, #0f4c81);
                transform: scale(1.04);
            }

            /* ===== Card vi phạm ===== */
            .violation-card {
                background-color: #696969;
                border-radius: 14px;
                padding: 18px;
                margin-bottom: 25px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.18);
                border: 1px solid #8a8a8a;
                transition: 0.25s ease;
            }

            .violation-card:hover {
                box-shadow: 0 5px 20px rgba(0,0,0,0.25);
                transform: translateY(-3px);
                border-color: #0f4c81;
            }

            /* ===== Title trong card vi phạm ===== */
            .violation-title {
                font-weight: 700;
                font-size: 1.05rem;
                color: white;
                margin-bottom: 12px;
                padding: 10px 16px;

                background: linear-gradient(135deg, #4b4b4b 0%, #3a3a3a 100%);
                border-radius: 12px;

                border: 1px solid #8a8a8a;
                box-shadow: 0 3px 8px rgba(0,0,0,0.2);

                display: flex;
                align-items: center;
                gap: 8px;
            }

            .violation-title span {
                color: #ffffff !important;
                font-weight: 800;
            }

            /* ===== Ảnh trong card ===== */
            .violation-card img {
                border-radius: 10px;
                border: 1px solid #dcdcdc;
            }

            /* ===== Video frame ===== */
            .stImage > img {
                border-radius: 10px;
                box-shadow: 0 3px 12px rgba(0,0,0,0.10);
            }

            /* ===== Timeline ===== */
            .timeline-item img {
                border-radius: 10px;
                border: 1px solid #ccc;
                transition: 0.2s ease;
            }

            .timeline-item img:hover {
                transform: scale(1.05);
                border-color: #0f4c81;
            }

            /* ===== FPS block ===== */
            .fps-box {
                background: #ffffff;
                border-radius: 12px;
                padding: 12px 16px;
                margin-top: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                border-left: 5px solid #1a73e8;
                color: #0f4c81;
                font-weight: 700;
                font-size: 1.1rem;
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
            <h3>Hệ thống Giám Sát Vượt Đèn Đỏ – AI Realtime</h3>
            <p style="font-size:1.05rem; color:#444;">
                Theo dõi <b>xe vượt đèn đỏ</b>, <b>biển số</b>, <b>hướng di chuyển</b> 
                và <b>trạng thái đèn giao thông</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)


# ==========================
# 🎥 Video Section
# ==========================
def show_video_section():
    """Tạo khung hiển thị video realtime."""
    st.subheader("Video Realtime")
    return st.empty()


# ==========================
# 🚗 Violation Card
# ==========================
def show_violation_card(vid, imgs):
    """Hiển thị 1 card chứa thông tin xe vi phạm."""

    st.markdown(
        f"""
        <div class="violation-card">
            <div class="violation-title">
                 🚗 <span>{vid}</span>
            </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(2)

    if "crop" in imgs:
        with cols[0]:
            st.image(imgs["crop"], caption="📍 Xe vi phạm", use_container_width=True)

    if "context" in imgs:
        with cols[1]:
            st.image(imgs["context"], caption="📷 Toàn cảnh", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
