import cv2
import os
import numpy as np
import threading
import queue
from core.vehicle_detection import detect_vehicles
from core.traffic_light_detection import detect_traffic_light
from core.license_plate_recognition import detect_and_read_plate



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output", "violations")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Lưu vi phạm vào: {os.path.abspath(OUTPUT_DIR)}")
# ==========================
# ⚙️ Cấu hình
# ==========================
OUTPUT_DIR = "output/violations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERA_DIRECTION_UP = True
FRAME_SKIP = 2           # bỏ qua 1 frame để tăng tốc
TEMPORAL_WINDOW = 1      # cần ≥3 frame liên tiếp để xác nhận vi phạm
RESIZE_WIDTH = 640       # giảm độ phân giải để tăng tốc YOLO

# ==========================
# 🚗 ROI động (Dynamic ROI)
# ==========================
def get_dynamic_roi(frame_width, frame_height):
    """Tính toán vùng ROI dựa trên tỉ lệ khung hình video"""
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
# 🧭 Kiểm tra xe trong ROI
# ==========================
def is_in_roi(box, roi_polygon):
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0

# ==========================
# 🚨 Kiểm tra vi phạm
# ==========================
def check_violation(label, box, light_state, stopline_y, roi_polygon):
    x1, y1, x2, y2 = box
    if light_state != "red":
        return False
    if not is_in_roi(box, roi_polygon):
        return False
    tolerance = 15
    if CAMERA_DIRECTION_UP:
        return y2 <= stopline_y - tolerance  # đuôi xe vượt vạch
    else:
        return y1 >= stopline_y + tolerance

# ==========================
# 🎥 Thread xử lý video chính
# ==========================
def process_video(video_path, display=False, frame_callback=None, save_output=True, stop_flag=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stopline_y = int(frame_height * 0.50)
    ROI_POLYGON = get_dynamic_roi(frame_width, frame_height)

    print(f"🟦 ROI động: {ROI_POLYGON.tolist()}")
    print(f"🟩 Vạch dừng (STOPLINE_Y) tại y = {stopline_y}px")

    output_path = "output/result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_queue = queue.Queue(maxsize=10)
    violated_history = {}
    violated_vehicles = set()

    # --- Thread đọc frame ---
    def read_frames():
        while cap.isOpened():
            if stop_flag and stop_flag.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
        cap.release()
        frame_queue.put(None)

    threading.Thread(target=read_frames, daemon=True).start()

    frame_count = 0
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        # 🔹 Resize để tăng tốc
        h, w = frame.shape[:2]
        scale_ratio = RESIZE_WIDTH / w
        resized_frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale_ratio)))

        # 🚦 Nhận diện đèn
        light_state = detect_traffic_light(resized_frame)
        color_light = (0, 0, 255) if light_state == "red" else (
            (0, 255, 255) if light_state == "yellow" else (0, 255, 0)
        )
        cv2.putText(frame, f"Light: {light_state.upper()}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_light, 3)

        # 🚘 Nhận diện phương tiện
        vehicles = detect_vehicles(resized_frame)

        for label, box, conf in vehicles:
            # Scale lại box theo khung gốc
            x1, y1, x2, y2 = [int(v / scale_ratio) for v in box]

            # Đảm bảo tọa độ nằm trong giới hạn frame
            h, w = frame.shape[:2]
            x1 = np.clip(x1, 0, w - 1)
            x2 = np.clip(x2, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            y2 = np.clip(y2, 0, h - 1)

            plate = detect_and_read_plate(frame, (x1, y1, x2, y2))
            violated = check_violation(label, (x1, y1, x2, y2), light_state, stopline_y, ROI_POLYGON)

            vehicle_id = plate or f"{label}_{x1}_{y1}"
            if violated:
                violated_history[vehicle_id] = violated_history.get(vehicle_id, 0) + 1
            else:
                violated_history[vehicle_id] = 0

            # 🔁 Lọc theo thời gian (≥3 frame liên tiếp)
            if violated_history[vehicle_id] >= TEMPORAL_WINDOW and vehicle_id not in violated_vehicles:
                violated_vehicles.add(vehicle_id)

                # ✅ Lưu ảnh crop xe (nếu vùng hợp lệ)
                if x2 > x1 and y2 > y1:
                    violation_crop = frame[y1:y2, x1:x2]
                    filename_crop = os.path.join(OUTPUT_DIR, f"{vehicle_id}_{frame_count}_crop.jpg")
                    cv2.imwrite(filename_crop, violation_crop)
                    print(f"✅ Lưu vi phạm (crop): {filename_crop}")
                else:
                    print(f"⚠️ Bỏ qua lưu crop cho {vehicle_id}, vùng rỗng hoặc sai tọa độ.")

                # 🟥 Lưu ảnh context (toàn cảnh có highlight)
                context = frame.copy()
                cv2.rectangle(context, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(context, "VIOLATION", (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                filename_context = os.path.join(OUTPUT_DIR, f"{vehicle_id}_{frame_count}_context.jpg")
                cv2.imwrite(filename_context, context)
                print(f"🚨 Vi phạm mới: {vehicle_id} tại frame {frame_count}")

            # Vẽ khung xe realtime
            color = (0, 0, 255) if violated else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {plate or ''}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Vẽ ROI & vạch dừng
        cv2.polylines(frame, [ROI_POLYGON], True, (255, 255, 0), 2)
        cv2.line(frame, (0, stopline_y), (frame_width, stopline_y), (0, 0, 255), 3)

        # Cập nhật GUI hoặc hiển thị
        if frame_callback:
            frame_callback(frame)
        if save_output:
            out.write(frame)
        if display:
            cv2.imshow("Traffic Violation Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video kết quả lưu tại: {output_path}")
