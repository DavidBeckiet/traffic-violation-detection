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
# 🗺️ Vietnam Province Codes
# ==========================
PROVINCE_CODES = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh", "15": "Hải Phòng",
    "16": "Hải Phòng", "17": "Thái Bình", "18": "Nam Định", "19": "Phú Thọ",
    "20": "Thái Nguyên", "21": "Yên Bái", "22": "Tuyên Quang", "23": "Hà Giang",
    "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La", "27": "Điện Biên",
    "28": "Hòa Bình", "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội",
    "32": "Hà Nội", "33": "Hà Nội", "34": "Hải Dương", "35": "Ninh Bình",
    "36": "Thanh Hóa", "37": "Nghệ An", "38": "Hà Tĩnh", "39": "Hà Tĩnh",
    "40": "Hà Tĩnh", "41": "Quảng Bình", "42": "Quảng Trị", "43": "Thừa Thiên Huế",
    "47": "Đà Nẵng", "48": "Đà Nẵng", "49": "Quảng Nam", "50": "Quảng Ngãi",
    "51": "TP.HCM", "52": "Bình Định", "53": "Phú Yên", "54": "Phú Yên",
    "55": "Khánh Hòa", "56": "Khánh Hòa", "57": "Khánh Hòa", "58": "Ninh Thuận",
    "59": "TP.HCM", "60": "Đồng Nai", "61": "Bình Dương", "62": "Long An",
    "63": "Tiền Giang", "64": "Vĩnh Long", "65": "Cần Thơ", "66": "Đồng Tháp",
    "67": "An Giang", "68": "Kiên Giang", "69": "Cà Mau", "70": "Tây Ninh",
    "71": "Bến Tre", "72": "Bà Rịa - Vũng Tàu", "73": "Quảng Bình", "74": "Trà Vinh",
    "75": "Hậu Giang", "76": "Đắk Lắk", "77": "Quảng Trị", "78": "Quảng Trị",
    "79": "TP.HCM", "80": "Kon Tum", "81": "Gia Lai", "82": "Gia Lai",
    "83": "Bình Phước", "84": "Bình Phước", "85": "Lâm Đồng", "86": "Lâm Đồng",
    "88": "Vĩnh Phúc", "89": "Hưng Yên", "90": "Hà Nam", "92": "Quảng Ninh",
    "93": "Bắc Ninh", "94": "Hải Dương", "95": "Hải Phòng", "97": "Bắc Giang",
    "98": "Bắc Kạn", "99": "Bắc Kạn"
}

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


def extract_province(plate_text: str) -> str:
    """Extract tỉnh thành từ biển số VN"""
    if not plate_text or len(plate_text) < 2:
        return "Unknown"
    
    # Lấy 2 số đầu
    province_code = plate_text[:2]
    
    if not province_code.isdigit():
        return "Unknown"
    
    return PROVINCE_CODES.get(province_code, "Unknown")


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
        return {"plate": "Unknown", "province": "Unknown"}

    # STEP 1 — Detect plate region
    lp_crop = detect_plate_region(vehicle_crop)

    if lp_crop is None:
        return {"plate": "Unknown", "province": "Unknown"}

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
        province = extract_province(final)
        return {"plate": final, "province": province}

    province = extract_province(plate_text)
    return {"plate": plate_text, "province": province}
