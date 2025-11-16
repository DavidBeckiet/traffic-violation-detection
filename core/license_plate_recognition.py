import os
import cv2
import numpy as np
import re
from collections import defaultdict, Counter
from paddleocr import PaddleOCR

# ==========================
# ⚙️ PaddleOCR init 1 lần
# ==========================
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# ==========================
# 🧠 Voting lưu theo track_id
# ==========================
plate_votes = defaultdict(list)  # track_id → [(plate_text, conf)]

# ==========================
# 📏 Regex biển số VN chuẩn mở rộng
# ==========================
VN_PLATE_PATTERN = re.compile(
    r'^[0-9]{2}[A-Z][0-9]{4,5}$|'          # 30A12345
    r'^[0-9]{2}[A-Z][0-9]-[0-9]{3,4}$|'    # 30A1-2345
    r'^[0-9]{2}[A-Z][0-9]{2}\.[0-9]{3}$'   # 59H1.234.56
)

# ==========================
# 🧹 Chuẩn hóa text biển số
# ==========================
def normalize_plate(text: str) -> str:
    """Chuẩn hóa ký tự dễ nhầm lẫn."""
    if not text:
        return ""

    s = text.upper()
    s = s.replace(" ", "").replace("-", "").replace(".", "")

    # Thay ký tự dễ nhầm
    replace_map = {
        "O": "0", "Q": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }
    for k, v in replace_map.items():
        s = s.replace(k, v)

    return s


def is_valid_vn_plate(text: str) -> bool:
    return bool(VN_PLATE_PATTERN.match(text))


# ==========================
# 📸 Preprocess biển số
# ==========================
def preprocess_plate(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        35, 7
    )

    return [enhanced, denoised, thresh]


# ==========================
# 🔍 OCR biển số
# ==========================
def read_plate_ocr(img_bgr):
    variants = preprocess_plate(img_bgr)
    candidates = []

    for variant in variants:
        try:
            result = ocr.ocr(variant, cls=True)
        except:
            continue

        if not result:
            continue

        all_text = ""
        all_conf = []

        for line in result:
            for w in line:
                if len(w) < 2:
                    continue
                text, conf = w[1]
                if isinstance(text, str):
                    all_text += text
                if isinstance(conf, (int, float)):
                    all_conf.append(conf)

        if not all_text:
            continue

        plate = normalize_plate(all_text)
        conf = float(np.mean(all_conf)) if all_conf else 0.0

        # Lọc kết quả rác (quá ngắn / toàn chữ)
        if len(plate) < 5:
            continue

        candidates.append((plate, conf))

    if not candidates:
        return "Unknown", 0.0

    # Ưu tiên biển hợp lệ (VN format) và confidence cao
    candidates.sort(key=lambda x: (is_valid_vn_plate(x[0]), x[1]), reverse=True)
    return candidates[0]


# ==========================
# 📦 Heuristic crop (motor khác car)
# ==========================
def heuristic_crop_plate(vehicle_img, vehicle_label="car"):
    h, w = vehicle_img.shape[:2]

    if vehicle_label in ["motorcycle", "motorbike"]:
        y_top = int(h * 0.45)
        y_bottom = int(h * 0.85)
    else:
        y_top = int(h * 0.55)
        y_bottom = int(h * 0.90)

    y_top = max(0, min(y_top, h - 1))
    y_bottom = max(0, min(y_bottom, h))

    return vehicle_img[y_top:y_bottom, :]


# ==========================
# 🧾 Voting cho track_id
# ==========================
def vote_plate(track_id):
    votes = plate_votes[track_id]
    if not votes:
        return "Unknown"

    # Đếm theo confidence
    weighted = Counter()
    for plate, conf in votes:
        weighted[plate] += conf

    return weighted.most_common(1)[0][0]


# ==========================
# 🚀 Hàm chính
# ==========================
def detect_and_read_plate(frame, box, track_id=None, vehicle_label="car"):
    x1, y1, x2, y2 = map(int, box)
    crop_vehicle = frame[y1:y2, x1:x2]

    if crop_vehicle.size == 0:
        return "Unknown"

    plate_img = heuristic_crop_plate(crop_vehicle, vehicle_label)

    if plate_img.size == 0:
        return "Unknown"

    plate_text, conf = read_plate_ocr(plate_img)

    # Lưu voting nếu có tracking
    if track_id:
        plate_votes[track_id].append((plate_text, conf))
        return vote_plate(track_id)

    return plate_text
