import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import threading
import time
import queue
import glob
import json
import pandas as pd

from app.process_video import process_video
from app.ui_components import setup_page_style, show_header, show_violation_card, show_video_section


# ==========================
# Cấu hình trang
# ==========================
st.set_page_config(page_title="Traffic Violation Detection 🚦",
                   page_icon="🚗",
                   layout="wide")
setup_page_style()

# ==========================
# Header
# ==========================
show_header()

# ==========================
# TAB điều hướng
# ==========================
tab_realtime, tab_history = st.tabs(["🔴 Realtime Detection", "📁 Lịch sử vi phạm"])

# ==========================
# Folder cấu hình
# ==========================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIOLATIONS_DIR = os.path.join(ROOT_DIR, "output", "violations")
UPLOADS_DIR = os.path.join(ROOT_DIR, "uploads")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ==========================
# Runtime states
# ==========================
frame_queue = queue.Queue(maxsize=3)
stop_flag = threading.Event()
processing_flag = threading.Event()

# Initialize session state
if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = None
if "current_video_folder" not in st.session_state:
    st.session_state["current_video_folder"] = None


# ==========================
# Nhận frame từ process_video
# ==========================
def update_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not frame_queue.full():
        frame_queue.put(frame_rgb)


# ==========================
# Load violations với cache (CHỈ video hiện tại)
# ==========================
def load_violations_cached():
    """Load violations với cache mỗi 0.5s để tránh block UI"""
    cache_key = "violations_cache"
    cache_time_key = "violations_cache_time"
    
    current_time = time.time()
    
    # Check cache (refresh mỗi 0.5s)
    if (cache_key in st.session_state and 
        cache_time_key in st.session_state and 
        current_time - st.session_state[cache_time_key] < 0.5):
        return st.session_state[cache_key]
    
    # Chỉ load ảnh từ folder video hiện tại
    current_folder = st.session_state.get("current_video_folder")
    if not current_folder or not os.path.exists(current_folder):
        st.session_state[cache_key] = {}
        st.session_state[cache_time_key] = current_time
        return {}
    
    # Load fresh data từ folder cụ thể
    files = sorted(
        glob.glob(os.path.join(current_folder, "*.jpg"))
        + glob.glob(os.path.join(current_folder, "*.png")),
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )

    grouped = {}
    for f in files:
        base = os.path.basename(f)

        id_base = (
            base.split('_crop')[0]
            if "_crop" in base else base.split('_context')[0]
        )

        if "_crop" in f:
            grouped.setdefault(id_base, {})["crop"] = f
        elif "_context" in f:
            grouped.setdefault(id_base, {})["context"] = f
    
    # Save to cache
    st.session_state[cache_key] = grouped
    st.session_state[cache_time_key] = current_time
    
    return grouped


# ==========================
# Luồng xử lý AI
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
        # Clear frame queue khi dừng
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                break


# ==========================================================
# ⭐ TAB 1 – REALTIME DETECTION
# ==========================================================
with tab_realtime:

    # Sidebar
    with st.sidebar:
        st.markdown("## Cài đặt hệ thống")
        uploaded_video = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov"])
        st.divider()
        st.info("Hệ thống nhận diện vượt đèn đỏ, biển số và trạng thái đèn tự động.")

    if uploaded_video:

        # Lưu video upload
        video_path = os.path.join(UPLOADS_DIR, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.markdown("---")

        start, stop = st.columns(2)
        with start:
            start_btn = st.button("Bắt đầu nhận diện", use_container_width=True)
        with stop:
            stop_btn = st.button("Dừng lại", use_container_width=True)

        # Start detection
        if start_btn and not processing_flag.is_set():
            stop_flag.clear()
            processing_flag.set()
            
            # Set folder cho video hiện tại
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            st.session_state["current_video_folder"] = os.path.join(VIOLATIONS_DIR, video_name)
            
            # Clear cache cũ
            if "violations_cache" in st.session_state:
                del st.session_state["violations_cache"]
            
            st.session_state["current_thread"] = threading.Thread(target=run_detection, args=(video_path,), daemon=True)
            st.session_state["current_thread"].start()
            st.info(f"Đang xử lý video: **{os.path.basename(video_path)}**")

        # Stop detection
        if stop_btn and processing_flag.is_set():
            st.warning("Đang dừng xử lý video...")
            
            # Set stop flag
            stop_flag.set()
            processing_flag.clear()
            
            # Clear frame queue để unblock thread
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except:
                    break
            
            # Đợi thread tối đa 5 giây
            if st.session_state["current_thread"] and st.session_state["current_thread"].is_alive():
                st.session_state["current_thread"].join(timeout=5)
                
                # Nếu vẫn còn sống sau 5s
                if st.session_state["current_thread"].is_alive():
                    st.error("⚠️ Thread không phản hồi - vui lòng reload trang (Ctrl+R)")
                else:
                    st.success("✅ Đã dừng thành công!")
            
            st.session_state["current_thread"] = None
            st.session_state["current_video_folder"] = None
            
            # Force rerun để clear UI
            st.rerun()

        # Layout 70/30
        left, right = st.columns([0.70, 0.30], gap="large")

        with left:
            frame_placeholder = show_video_section()
            fps_box = st.empty()

        with right:
            st.subheader("Danh sách vi phạm")
            violation_sidebar = st.empty()

        st.markdown("---")

        # Timeline
        st.markdown("### Violation Timeline (Gần đây nhất)")
        timeline_container = st.empty()

        # Loop realtime
        last_time = time.time()
        frame_count = 0
        last_violation_update = 0

        while processing_flag.is_set():

            # ======= FRAME =======
            try:
                frame = frame_queue.get(timeout=0.2)
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)
                frame_count += 1

                now = time.time()
                elapsed = now - last_time
                if elapsed >= 1:
                    fps = frame_count / elapsed

                    fps_box.markdown(
                        f'<div class="fps-box">FPS: <b>{fps:.1f}</b></div>',
                        unsafe_allow_html=True
                    )

                    last_time = now
                    frame_count = 0

            except queue.Empty:
                pass

            # ======= UPDATE VIOLATIONS (chỉ mỗi 1s) =======
            current_time = time.time()
            if current_time - last_violation_update >= 1.0:
                last_violation_update = current_time
                
                grouped = load_violations_cached()

                # ======= SIDEBAR (3 ảnh) =======
                with violation_sidebar.container():
                    if grouped:
                        for vid, imgs in list(grouped.items())[:3]:
                            show_violation_card(vid, imgs)
                    else:
                        st.success("Không có vi phạm nào.")

                # ======= TIMELINE (5 ảnh) =======
                with timeline_container.container():
                    if grouped:
                        timeline_items = list(grouped.items())[:5]
                        cols = st.columns(5)

                        for i, (vid, imgs) in enumerate(timeline_items):
                            with cols[i % 5]:
                                if "crop" in imgs:
                                    st.image(imgs["crop"], caption=f"#{vid}", use_container_width=True)
                    else:
                        st.info("Chưa có dữ liệu vi phạm.")

            time.sleep(0.05)

        # End detection
        if "last_video_result" in st.session_state:
            result = st.session_state.pop("last_video_result")
            st.success(f"Hoàn tất xử lý video. Ghi nhận {len(result['violations'])} vi phạm.")
            st.write(f"Kết quả lưu tại: `{result['output_path']}`")

        if "error" in st.session_state:
            st.error(f"Lỗi xử lý: {st.session_state.pop('error')}")

    else:
        st.warning("⬆ Vui lòng tải video lên để bắt đầu nhận diện.")

# ==========================================================
# ⭐ TAB 2 – HISTORY
# ==========================================================
with tab_history:

    st.subheader("📁 Lịch sử vi phạm đã lưu")

    records_file = os.path.join(VIOLATIONS_DIR, "violations.json")

    if not os.path.exists(records_file):
        st.info("Chưa có dữ liệu vi phạm nào.")
    else:
        with open(records_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        # Lọc chỉ các record có file ảnh tồn tại
        valid_records = []
        for r in records:
            crop_exists = os.path.exists(r.get("crop_image", ""))
            context_exists = os.path.exists(r.get("context_image", ""))
            if crop_exists and context_exists:
                valid_records.append(r)

        if not valid_records:
            st.info("Không có dữ liệu vi phạm hợp lệ (file ảnh đã bị xóa).")
        else:
            st.success(f"Tìm thấy {len(valid_records)}/{len(records)} vi phạm hợp lệ")

            # Bảng dữ liệu
            st.markdown("### 📋 Bảng dữ liệu")
            df = pd.DataFrame(valid_records)
            st.dataframe(df, use_container_width=True)

            # JSON raw
            with st.expander("📄 Xem JSON"):
                st.json(valid_records)

            # Ảnh vi phạm
            st.markdown("### 📸 Hình ảnh vi phạm")
            for r in valid_records:
                st.markdown("---")
                st.write(f"🚗 Loại xe: {r['vehicle_type']}")
                st.write(f"🔢 Biển số: **{r['license_plate']}**")
                if r.get('province') and r['province'] != 'Unknown':
                    st.write(f"📍 Tỉnh/TP: **{r['province']}**")
                st.write(f"🕒 Thời gian: {r['timestamp']}")

                cols = st.columns(2)
                with cols[0]:
                    st.image(r["crop_image"], caption="📍 Xe vi phạm", use_container_width=True)
                with cols[1]:
                    st.image(r["context_image"], caption="📷 Toàn cảnh", use_container_width=True)

            st.markdown("---")
            st.download_button(
                label="📥 Tải JSON",
                data=json.dumps(valid_records, indent=4, ensure_ascii=False),
                file_name="violations.json",
                mime="application/json"
            )
