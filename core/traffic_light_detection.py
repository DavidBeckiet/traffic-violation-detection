import cv2
import numpy as np
from ultralytics import YOLO

# 🔧 Đường dẫn model YOLO
MODEL_PATH = "models/traffic_light/traffic_light.pt"
traffic_light_model = YOLO(MODEL_PATH)

# 🟦 Cấu hình vùng ROI đèn giao thông
# (cắt phía trên bên phải của khung hình)
def get_roi(frame):
    h, w, _ = frame.shape
    roi_width = int(w * 0.25)   # lấy 25% chiều rộng
    roi_height = int(h * 0.3)   # lấy 30% chiều cao
    x1 = w - roi_width          # bắt đầu từ bên phải
    y1 = 0
    x2 = w
    y2 = roi_height
    return frame[y1:y2, x1:x2]


def detect_traffic_light(frame):
    """
    🔦 Nhận diện đèn giao thông bằng YOLO + fallback HSV
    """
    roi = get_roi(frame)

    # Phát hiện bằng YOLO
    results = traffic_light_model(roi, verbose=False)
    if len(results) > 0 and len(results[0].boxes) > 0:
        classes = results[0].boxes.cls.cpu().numpy()
        # 0: green, 1: red, 2: yellow
        if 1 in classes:
            return "red"
        elif 2 in classes:
            return "yellow"
        elif 0 in classes:
            return "green"

    # 🟡 Nếu YOLO không phát hiện, fallback bằng phân tích màu HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # mask màu đỏ
    red_mask1 = cv2.inRange(hsv, (0, 80, 150), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 80, 150), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # mask màu vàng
    yellow_mask = cv2.inRange(hsv, (15, 100, 150), (35, 255, 255))

    # mask màu xanh
    green_mask = cv2.inRange(hsv, (40, 80, 120), (85, 255, 255))

    # Đếm pixel sáng
    red_pixels = np.sum(red_mask > 0)
    yellow_pixels = np.sum(yellow_mask > 0)
    green_pixels = np.sum(green_mask > 0)

    # Xác định đèn sáng nhất
    max_color = max(red_pixels, yellow_pixels, green_pixels)
    if max_color < 100:
        return "unknown"

    if max_color == red_pixels:
        return "red"
    elif max_color == yellow_pixels:
        return "yellow"
    else:
        return "green"
