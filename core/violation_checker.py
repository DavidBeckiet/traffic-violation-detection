import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# =======================
# 🔧 CONFIGURATION
# =======================
VEHICLE_MODEL_PATH = "yolov8m.pt"
LICENSE_PLATE_MODEL_PATH = "models/license_plate/license_plate_detection.pt"
OCR_MODEL_PATH = "models/license_plate/license_plate_ocr.pt"
TRAFFIC_LIGHT_MODEL_PATH = "models/traffic_light/traffic_light.pt"

STOPLINE_Y = 500  # y-coordinate line stop
VIOLATION_DIR = "output/violations"
os.makedirs(VIOLATION_DIR, exist_ok=True)

# =======================
# 🔍 MODEL INITIALIZATION
# =======================
vehicle_detector = YOLO(VEHICLE_MODEL_PATH)
license_plate_detector = YOLO(LICENSE_PLATE_MODEL_PATH)
ocr_detector = YOLO(OCR_MODEL_PATH)
traffic_light_detector = YOLO(TRAFFIC_LIGHT_MODEL_PATH)


# =======================
# 🚦 TRAFFIC LIGHT DETECTION
# =======================
def get_traffic_light_state(frame):
    results = traffic_light_detector(frame)
    if len(results) == 0:
        return "unknown"

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for cls in classes:
        if cls == 1:
            return "red"
        elif cls == 0:
            return "green"
        elif cls == 2:
            return "yellow"
    return "unknown"


# =======================
# 🚗 RED LIGHT VIOLATION CHECK
# =======================
def check_red_light_violation(vehicle_box, traffic_light_state):
    if traffic_light_state == "red" and vehicle_box[3] > STOPLINE_Y:
        return True
    return False


# =======================
# 🔤 OCR LICENSE PLATE
# =======================
def split_characters(lp_image):
    gray = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
    boxes = sorted(boxes, key=lambda x: x[0])
    chars = [(lp_image[y:y + h, x:x + w], (x, y)) for (x, y, w, h) in boxes]
    return chars


def recognize_characters(chars):
    lp_text = ""
    y_coords = []
    for char_img, (x, y) in chars:
        results = ocr_detector(char_img)
        if len(results[0].boxes) > 0:
            cls = int(results[0].boxes.cls[0].cpu().numpy())
            lp_text += str(cls) if cls < 10 else chr(65 + (cls - 10))
            y_coords.append(y)
    return lp_text, y_coords


def check_license_plate_lines(y_coords):
    if not y_coords:
        return "unknown"
    y_diff = max(y_coords) - min(y_coords)
    return "2 lines" if y_diff > 20 else "1 line"


def normalize_license_plate(lp_text):
    if len(lp_text) >= 7 and lp_text[0].isdigit() and lp_text[2].isalpha():
        return lp_text[:2] + "-" + lp_text[2] + "-" + lp_text[3:]
    return lp_text


def process_license_plate(image, lp_box):
    x_min, y_min, x_max, y_max = map(int, lp_box)
    lp_image = image[y_min:y_max, x_min:x_max]

    chars = split_characters(lp_image)
    lp_text, y_coords = recognize_characters(chars)
    lp_type = check_license_plate_lines(y_coords)
    lp_text = normalize_license_plate(lp_text)

    return lp_text, lp_image, lp_type


# =======================
# 🎨 DRAWING UTILITIES
# =======================
def draw_annotations(image, box, text, color=(0, 0, 255)):
    x_min, y_min, x_max, y_max = map(int, box)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def save_violation(image, vehicle_id, license_plate_text, lp_type):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{VIOLATION_DIR}/{timestamp}_ID{vehicle_id}_{license_plate_text}_{lp_type}.jpg"
    cv2.imwrite(filename, image)
    print(f"💾 Saved violation: {filename}")


# =======================
# 🎥 MAIN VIDEO PIPELINE
# =======================
def process_video(video_path, display=True, frame_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = "output/result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.line(frame, (0, STOPLINE_Y), (frame_width, STOPLINE_Y), (255, 0, 0), 2)

        traffic_state = get_traffic_light_state(frame)
        cv2.putText(frame, f"Light: {traffic_state.upper()}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if traffic_state == "red" else (0, 255, 0), 3)

        vehicle_results = vehicle_detector(frame)
        vehicles = vehicle_results[0].boxes.xyxy.cpu().numpy()

        for i, vehicle_box in enumerate(vehicles):
            vehicle_id = i
            if check_red_light_violation(vehicle_box, traffic_state):
                lp_results = license_plate_detector(frame)
                lp_boxes = lp_results[0].boxes.xyxy.cpu().numpy()

                for lp_box in lp_boxes:
                    if (lp_box[0] >= vehicle_box[0] and lp_box[2] <= vehicle_box[2] and
                            lp_box[1] >= vehicle_box[1] and lp_box[3] <= vehicle_box[3]):
                        lp_text, lp_image, lp_type = process_license_plate(frame, lp_box)

                        draw_annotations(frame, vehicle_box, f"Vehicle {vehicle_id}")
                        draw_annotations(frame, lp_box, f"{lp_text} ({lp_type})")

                        save_violation(frame.copy(), vehicle_id, lp_text, lp_type)
                        break

        # GUI frame callback
        if frame_callback:
            frame_callback(frame)

        # OpenCV window (CLI)
        if display:
            cv2.imshow("Traffic Violation Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Video processed successfully.")
    print(f"📁 Saved to: {output_path}")


if __name__ == "__main__":
    video_path = "data_test/test_video_1.mp4"
    process_video(video_path)
