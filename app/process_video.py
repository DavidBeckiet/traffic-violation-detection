import cv2
import os
import numpy as np
from core.vehicle_detection import detect_vehicles
from core.traffic_light_detection import detect_traffic_light
from core.license_plate_recognition import detect_and_read_plate

# ==========================
# ⚙️ Cấu hình
# ==========================
OUTPUT_DIR = "output/violations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Xe di chuyển từ dưới lên
CAMERA_DIRECTION_UP = True

# ROI (Region of Interest) - vùng giám sát
ROI_POLYGON = np.array([
    [(450, 100), (1050, 100), (1450, 520), (100, 520)]
])

# ==========================
# 🧭 Kiểm tra xe trong ROI
# ==========================
def is_in_roi(box, roi_polygon):
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    return cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0


# ==========================
# 🚨 Kiểm tra vi phạm vượt đèn đỏ
# ==========================
def check_violation(label, box, plate, light_state, stopline_y, roi_polygon):
    x1, y1, x2, y2 = box

    # Chỉ xét khi đèn đỏ
    if light_state != "red":
        return False

    # Không nằm trong ROI thì bỏ qua
    if not is_in_roi(box, roi_polygon):
        return False

    # Giảm nhiễu - xe sát vạch không tính
    tolerance = 15
    if CAMERA_DIRECTION_UP:
        # ✅ Đuôi xe vượt vạch (xe chạy từ dưới lên)
        return y2 <= stopline_y - tolerance
    else:
        # ✅ Đuôi xe vượt vạch (xe chạy từ trên xuống)
        return y1 >= stopline_y + tolerance


# ==========================
# 🎥 Xử lý video chính
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
    print(f"🟦 Vạch dừng (STOPLINE_Y) tại y = {stopline_y}px")

    output_path = "output/result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    violated_vehicles = set()

    while True:
        if stop_flag and stop_flag.is_set():
            print("🛑 Dừng xử lý video theo yêu cầu người dùng.")
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Vẽ vạch dừng và ROI
        cv2.line(frame, (0, stopline_y), (frame_width, stopline_y), (0, 0, 255), 3)
        cv2.polylines(frame, [ROI_POLYGON], True, (255, 255, 0), 2)

        # Phát hiện trạng thái đèn giao thông
        light_state = detect_traffic_light(frame)
        color_light = (0, 0, 255) if light_state == "red" else (
            (0, 255, 255) if light_state == "yellow" else (0, 255, 0)
        )
        cv2.putText(frame, f"Light: {light_state.upper()}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_light, 3)

        # Phát hiện phương tiện
        vehicles = detect_vehicles(frame)

        for label, box, conf in vehicles:
            x1, y1, x2, y2 = map(int, box)
            plate = detect_and_read_plate(frame, box)
            violated = check_violation(label, box, plate, light_state, stopline_y, ROI_POLYGON[0])

            # Vẽ khung xe
            color = (0, 0, 255) if violated else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {plate or ''}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Xử lý khi phát hiện vi phạm ---
            if violated:
                vehicle_id = plate or f"{label}_{x1}_{y1}"
                if vehicle_id not in violated_vehicles:
                    violated_vehicles.add(vehicle_id)

                    # Lưu ảnh vi phạm
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    filename = os.path.join(OUTPUT_DIR, f"{vehicle_id}_{frame_count}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"🚨 Vi phạm mới: {vehicle_id} tại frame {frame_count}")

        # Cập nhật GUI hoặc lưu file
        if frame_callback:
            frame_callback(frame)
        if save_output:
            out.write(frame)
        if display:
            cv2.imshow("Traffic Violation Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video kết quả lưu tại: {output_path}")
