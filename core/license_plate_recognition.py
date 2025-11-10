import os
import cv2
import numpy as np
import re
from collections import defaultdict, Counter
from paddleocr import PaddleOCR

# ==========================
# ⚙️ Khởi tạo OCR (chạy 1 lần)
# ==========================
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# ==========================
# 🧠 Bộ nhớ tạm cho voting
# ==========================
plate_votes = defaultdict(list)  # track_id -> [(plate_text, conf)]

# ==========================
# 📏 Regex kiểm tra biển số VN
# ==========================
VN_PLATE_PATTERN = re.compile(r'^[0-9]{2}[A-Z][0-9]{4,5}$')

# ==========================
# 🧹 Chuẩn hóa biển số
# ==========================
def normalize_plate(text: str) -> str:
    """Chuẩn hóa ký tự biển số (O→0, I→1, Z→2, bỏ dấu cách / gạch)."""
    s = text.upper().replace(' ', '').replace('-', '').replace('.', '')
    s = s.replace('O', '0').replace('I', '1').replace('Z', '2')
    return s

def is_valid_vn_plate(text: str) -> bool:
    """Kiểm tra định dạng biển số Việt Nam."""
    s = normalize_plate(text)
    return bool(VN_PLATE_PATTERN.match(s))

# ==========================
# 🧩 Xử lý ảnh biển số trước OCR
# ==========================
def preprocess_plate(img_bgr):
    """Tăng chất lượng ảnh biển số trước OCR."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Tăng tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Giảm nhiễu bằng bilateral
    denoised = cv2.bilateralFilter(enhanced, 7, 75, 75)

    # Ngưỡng hóa (threshold) để tách chữ số
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 5
    )

    return denoised, thresh

# ==========================
# 🔍 OCR biển số
# ==========================
def read_plate_ocr(img_bgr):
    """
    Nhận diện chữ trên biển số bằng PaddleOCR.
    Trả về (plate_text, confidence)
    """
    variants = preprocess_plate(img_bgr)
    candidates = []

    for variant in variants:
        result = ocr.ocr(variant, cls=True)
        if not result:
            continue

        # Gom ký tự từ các dòng OCR
        text = "".join([w[1][0] for line in result for w in line])
        conf = np.mean([w[1][1] for line in result for w in line]) if result else 0.0

        plate = normalize_plate(text)
        candidates.append((plate, conf))

    if not candidates:
        return "Unknown", 0.0

    # Ưu tiên biển hợp lệ có confidence cao
    candidates.sort(key=lambda x: (is_valid_vn_plate(x[0]), x[1]), reverse=True)
    return candidates[0]

# ==========================
# 📸 Crop biển số từ xe (nếu chưa có detector riêng)
# ==========================
def heuristic_crop_plate(vehicle_img, vehicle_label="car"):
    """Cắt vùng đáy xe để lấy biển số (heuristic)."""
    h, w = vehicle_img.shape[:2]
    if vehicle_label == "motorbike":
        y_top = int(h * 0.45)
        y_bottom = int(h * 0.80)
    else:  # car, truck
        y_top = int(h * 0.55)
        y_bottom = int(h * 0.85)

    y_top = np.clip(y_top, 0, h - 1)
    y_bottom = np.clip(y_bottom, 0, h)
    return vehicle_img[y_top:y_bottom, :]

# ==========================
# 🧾 Cơ chế voting
# ==========================
def vote_plate(track_id: str) -> str:
    """Trả về biển số xuất hiện nhiều nhất cho xe (nếu có tracking)."""
    votes = plate_votes[track_id]
    if not votes:
        return "Unknown"
    counts = Counter([p for p, _ in votes])
    return counts.most_common(1)[0][0]

# ==========================
# 🚀 Hàm chính cho pipeline
# ==========================
def detect_and_read_plate(frame, box, track_id=None, vehicle_label="car"):
    """
    Đầu vào:
        frame: khung hình gốc
        box: (x1, y1, x2, y2) của xe
        track_id: ID của xe (nếu có tracking)
    Trả về:
        plate_text: biển số tốt nhất hiện tại
    """
    x1, y1, x2, y2 = map(int, box)
    crop_vehicle = frame[y1:y2, x1:x2]

    # Heuristic crop biển số từ xe
    plate_img = heuristic_crop_plate(crop_vehicle, vehicle_label)
    if plate_img.size == 0:
        return "Unknown"

    # Đọc biển số bằng OCR
    plate_text, conf = read_plate_ocr(plate_img)

    if track_id:
        plate_votes[track_id].append((plate_text, conf))
        # Lấy biển số được vote nhiều nhất
        final_plate = vote_plate(track_id)
    else:
        final_plate = plate_text

    return final_plate
