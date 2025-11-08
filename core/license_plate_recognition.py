from ultralytics import YOLO
import cv2

detector = YOLO("models/license_plate/license_plate_detection.pt")
ocr = YOLO("models/license_plate/license_plate_ocr.pt")

def detect_and_read_plate(frame, box):
    x1, y1, x2, y2 = box
    lp_img = frame[y1:y2, x1:x2]
    results = ocr(lp_img)
    text = ""
    for r in results:
        for b in r.boxes:
            cls_id = int(b.cls[0])
            text += str(cls_id)  # hoặc mapping ký tự thực tế
    return text if text else None
