import os
import cv2
import numpy as np
import re
from collections import defaultdict, Counter
from paddleocr import PaddleOCR
from ultralytics import YOLO

# ==========================
# ⚙️ LOAD MODELS
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "license_plate")

lp_detector = YOLO(os.path.join(MODEL_DIR, "license_plate_detection.pt"))
lp_ocr_yolo = YOLO(os.path.join(MODEL_DIR, "license_plate_ocr.pt"))

paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    show_log=False
)

# ==========================
# 🧠 VOTING
# track_id → [(text, conf)]
# ==========================
plate_votes = defaultdict(list)

# ==========================
# 📏 Vietnam plate regex
# ==========================
VN_PATTERN = re.compile(
    r'^[0-9]{2}[A-Z][0-9]{4,5}$|'
    r'^[0-9]{2}[A-Z][0-9]-[0-9]{3,4}$|'
    r'^[0-9]{2}[A-Z][0-9]{2}\.[0-9]{3}$'
)

# ==========================
# 🧹 Normalize plate
# ==========================
def normalize(text: str) -> str:
    if not text:
        return ""

    s = text.upper().replace(" ", "").replace("-", "").replace(".", "")

    map_table = {
        "O": "0", "Q": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }
    for k, v in map_table.items():
        s = s.replace(k, v)

    return s

def is_valid_vietnam_plate(text):
    return bool(VN_PATTERN.match(text))


# ==========================
# 🔍 OCR: YOLO + PaddleOCR
# ==========================
def ocr_paddle(img):
    """OCR bằng PaddleOCR"""
    try:
        result = paddle_ocr.ocr(img, cls=True)
    except:
        return None, 0.0

    text = ""
    confs = []

    if not result:
        return None, 0.0

    for line in result:
        for w in line:
            if len(w) < 2:
                continue
            t, c = w[1]
            text += str(t)
            confs.append(float(c))

    if not text:
        return None, 0.0

    return normalize(text), float(np.mean(confs))


def ocr_yolo_plate(img):
    """OCR bằng YOLO OCR model"""
    try:
        results = lp_ocr_yolo(img)
    except:
        return None, 0.0

    if len(results) == 0:
        return None, 0.0

    r = results[0]

    if not hasattr(r, "probs") or r.probs is None:
        return None, 0.0

    # YOLO OCR output: text classification
    text_raw = r.names[int(r.probs.top1)]
    conf = float(r.probs.top1conf)

    return normalize(text_raw), conf


# ==========================
# 🧠 BEST DECISION (YOLO + PaddleOCR)
# ==========================
def best_ocr_result(img):
    yolo_text, yolo_conf = ocr_yolo_plate(img)
    pad_text, pad_conf = ocr_paddle(img)

    candidates = []

    if yolo_text:
        candidates.append((yolo_text, yolo_conf, "YOLO"))

    if pad_text:
        candidates.append((pad_text, pad_conf, "Paddle"))

    if not candidates:
        return "Unknown", 0.0

    # Ưu tiên biển hợp lệ VN
    def score(item):
        plate, conf, model = item
        return (is_valid_vietnam_plate(plate), conf)

    candidates.sort(key=score, reverse=True)

    return candidates[0][0], candidates[0][1]


# ==========================
# 🚗 Detect + crop plate
# ==========================
def detect_plate_region(vehicle_img):
    """Trả về crop biển số từ YOLO detector"""
    results = lp_detector(vehicle_img)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    x1, y1, x2, y2 = boxes[0]  # lấy box đầu tiên (yolo đã sort by conf)

    crop = vehicle_img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


# ==========================
# 🎯 Main API
# ==========================
def detect_and_read_plate(frame, box, track_id=None, vehicle_label="car"):
    x1, y1, x2, y2 = map(int, box)
    vehicle_crop = frame[y1:y2, x1:x2]

    if vehicle_crop.size == 0:
        return "Unknown"

    # STEP 1 — Detect plate region
    lp_crop = detect_plate_region(vehicle_crop)

    if lp_crop is None:
        return "Unknown"

    # STEP 2 — OCR (YOLO + Paddle)
    plate_text, conf = best_ocr_result(lp_crop)

    # STEP 3 — Voting theo track_id
    if track_id is not None:
        plate_votes[track_id].append((plate_text, conf))

        # Weighted voting
        counter = Counter()
        for p, c in plate_votes[track_id]:
            counter[p] += c

        final = counter.most_common(1)[0][0]
        return final

    return plate_text
