from ultralytics import YOLO

model = YOLO("yolov8m.pt")

def detect_vehicles(frame):
    results = model(frame, verbose=False)
    vehicles = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3]:  # car, motorcycle
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = "car" if cls_id == 2 else "motorcycle"
            vehicles.append((label, (x1, y1, x2, y2), conf))
    return vehicles
