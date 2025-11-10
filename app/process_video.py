import cv2
import os
import numpy as np
import threading
import queue
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.vehicle_detection import detect_vehicles
from core.traffic_light_detection import detect_traffic_light
from core.license_plate_recognition import detect_and_read_plate

# ==========================
# ⚙️ Cấu hình
# ==========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "violations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S"
)

CAMERA_DIRECTION_UP = True
FRAME_SKIP = 1
TEMPORAL_WINDOW = 1
RESIZE_WIDTH = 640
MAX_WORKERS = 3  # số thread OCR tối đa


# ==========================
# 📐 ROI động
# ==========================
def get_dynamic_roi(frame_width, frame_height):
    top_y = int(frame_height * 0.15)
    bottom_y = int(frame_height * 0.80)
    left_x = int(frame_width * 0.10)
    right_x = int(frame_width * 0.90)
    return np.array([
        (left_x, top_y),
        (right_x, top_y),
        (right_x, bottom_y),
        (left_x, bottom_y)
    ])


# ==========================
# 🚦 Kiểm tra ROI và vi phạm
# ==========================
def is_in_roi(box, roi_polygon):
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0


def check_violation(label, box, light_state, stopline_y, roi_polygon):
    x1, y1, x2, y2 = box
    if light_state != "red":
        return False
    if not is_in_roi(box, roi_polygon):
        return False
    tolerance = 15
    if CAMERA_DIRECTION_UP:
        return y2 <= stopline_y - tolerance
    else:
        return y1 >= stopline_y + tolerance


# ==========================
# 🎥 Xử lý video chính
# ==========================
def process_video(video_path, display=False, frame_callback=None, save_output=True, stop_flag=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"❌ Không thể mở video: {video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stopline_y = int(frame_height * 0.50)
    ROI_POLYGON = get_dynamic_roi(frame_width, frame_height)

    logging.info(f"🎞️ Xử lý video {os.path.basename(video_path)} ({frame_width}x{frame_height}, {fps}fps)")
    logging.info(f"🟩 Stopline tại y={stopline_y}, ROI={ROI_POLYGON.tolist()}")

    output_path = os.path.join(OUTPUT_DIR, "result.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_queue = queue.Queue(maxsize=5)
    violated_vehicles = set()
    violated_history = {}
    ocr_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    # ======================
    # 🧵 Thread đọc frame
    # ======================
    def read_frames():
        while cap.isOpened():
            if stop_flag and stop_flag.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                continue
        cap.release()
        frame_queue.put(None)

    threading.Thread(target=read_frames, daemon=True).start()

    frame_count = 0
    red_light_stable = None
    same_light_count = 0

    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Resize tăng tốc
        h, w = frame.shape[:2]
        scale_ratio = RESIZE_WIDTH / w
        resized_frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale_ratio)))

        # 🚦 Nhận diện đèn
        try:
            current_light = detect_traffic_light(resized_frame)
            if current_light == red_light_stable:
                same_light_count += 1
            else:
                same_light_count = 0
            light_state = current_light if same_light_count >= 3 else red_light_stable or current_light
            red_light_stable = current_light
        except Exception as e:
            logging.warning(f"Lỗi detect_traffic_light: {e}")
            light_state = "unknown"

        color_light = (0, 0, 255) if light_state == "red" else (
            (0, 255, 255) if light_state == "yellow" else (0, 255, 0)
        )
        cv2.putText(frame, f"Light: {light_state.upper()}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_light, 3)

        # 🚗 Nhận diện xe
        try:
            vehicles = detect_vehicles(resized_frame)
        except Exception as e:
            logging.warning(f"Lỗi detect_vehicles: {e}")
            vehicles = []

        # Gửi OCR song song
        futures = []
        for label, box, conf in vehicles:
            x1, y1, x2, y2 = [int(v / scale_ratio) for v in box]
            if y2 <= y1 or x2 <= x1:
                continue
            futures.append((label, (x1, y1, x2, y2),
                            ocr_executor.submit(detect_and_read_plate, frame, (x1, y1, x2, y2))))

        for label, (x1, y1, x2, y2), future in futures:
            try:
                plate = future.result(timeout=5) or "Unknown"
            except Exception:
                plate = "Unknown"

            vehicle_id = f"{plate}_{label}"
            violated = check_violation(label, (x1, y1, x2, y2), light_state, stopline_y, ROI_POLYGON)
            violated_history[vehicle_id] = violated_history.get(vehicle_id, 0) + 1 if violated else 0

            if violated_history[vehicle_id] >= TEMPORAL_WINDOW and vehicle_id not in violated_vehicles:
                violated_vehicles.add(vehicle_id)
                timestamp = datetime.now().strftime("%H%M%S")
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_path = os.path.join(OUTPUT_DIR, f"{vehicle_id}_{timestamp}_crop.jpg")
                    context_path = os.path.join(OUTPUT_DIR, f"{vehicle_id}_{timestamp}_context.jpg")
                    cv2.imwrite(crop_path, crop)
                    context = frame.copy()
                    cv2.rectangle(context, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(context, "VIOLATION", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imwrite(context_path, context)
                logging.info(f"🚨 Vi phạm: {vehicle_id}")

            # Vẽ realtime
            color = (0, 0, 255) if violated else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} [{plate}]", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.polylines(frame, [ROI_POLYGON], True, (255, 255, 0), 2)
        cv2.line(frame, (0, stopline_y), (frame_width, stopline_y), (0, 0, 255), 3)

        if frame_callback:
            try:
                frame_callback(frame)
            except Exception as e:
                logging.warning(f"Lỗi callback frame: {e}")

        if save_output:
            out.write(frame)

        if display:
            cv2.imshow("Traffic Violation Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if stop_flag and stop_flag.is_set():
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"✅ Hoàn tất xử lý. Kết quả lưu tại: {output_path}")

    return {
        "total_frames": frame_count,
        "violations": list(violated_vehicles),
        "output_path": output_path
    }
