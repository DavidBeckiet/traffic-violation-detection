import cv2
import os
import json
import numpy as np
import threading
import queue
import logging
from datetime import datetime

from core.vehicle_detection import detect_vehicles
from core.traffic_light_detection import detect_traffic_light
from core.license_plate_recognition import detect_and_read_plate
from utils.data_logger import save_violation_record

# =========================
# ⚙️ CONFIG
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "violations")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "video_zones.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S"
)

CAMERA_DIRECTION_UP = True
FRAME_SKIP = 1
RESIZE_WIDTH = 640


# =========================
# 📐 DEFAULT ROI
# =========================
def get_dynamic_roi(frame_width, frame_height):
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


# =========================
# UTILITIES
# =========================
def is_in_roi(box, roi_polygon):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2)//2, (y1 + y2)//2
    return cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0


def get_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5



# =========================
# 🎥 MAIN PROCESS
# =========================
def process_video(video_path, display=False, frame_callback=None, save_output=True, stop_flag=None):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("❌ Không thể mở video.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25


    # Load ROI
    video_name = os.path.basename(video_path)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            zones = json.load(f)
    else:
        zones = {}

    if video_name in zones:
        ROI_POLYGON = np.array(zones[video_name]["roi"], dtype=np.int32)
        stopline_y = zones[video_name]["stop_line_y"]
    else:
        ROI_POLYGON = get_dynamic_roi(frame_width, frame_height)
        stopline_y = int(frame_height * 0.5)
        zones[video_name] = {
            "roi": ROI_POLYGON.tolist(),
            "stop_line_y": stopline_y
        }
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(zones, f, indent=4)


    logging.info(f"🎞️ Start: {video_name}")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{os.path.splitext(video_name)[0]}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    )

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height)
    )

    # ===================
    # THREAD READ FRAMES
    # ===================
    frame_queue = queue.Queue(maxsize=5)

    def read_frames():
        while cap.isOpened():
            if stop_flag and stop_flag.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass
        cap.release()
        frame_queue.put(None)

    threading.Thread(target=read_frames, daemon=True).start()


    # ===================
    # TRACKING DATA
    # ===================
    tracks = {}
    track_id_counter = 0

    # Light smoothing
    stable_light = None
    same_light_counter = 0

    frame_count = 0


    # ===================
    # 🔁 MAIN LOOP
    # ===================
    while True:

        if stop_flag and stop_flag.is_set():
            break

        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        if frame is None:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # --- Resize for YOLO ---
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        resized = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

        # ======================
        # 🚦 TRAFFIC LIGHT
        # ======================
        try:
            cur = detect_traffic_light(resized)
            if cur == stable_light:
                same_light_counter += 1
            else:
                same_light_counter = 0

            light_state = cur if same_light_counter >= 3 else (stable_light or cur)
            stable_light = cur

        except:
            light_state = "unknown"

        color = (0,0,255) if light_state=="red" else ((0,255,255) if light_state=="yellow" else (0,255,0))
        cv2.putText(frame, f"Light: {light_state}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # ======================
        # 🚗 VEHICLE DETECTION
        # ======================
        try:
            detections = detect_vehicles(resized)
        except:
            detections = []

        # ============================================================
        # TRACKING + DIRECTION + STOPLINE VIOLATION LOGIC (FIX SIDE)
        # ============================================================
        for label, box, conf in detections:

            # Scale box về size gốc
            x1, y1, x2, y2 = [int(v / scale) for v in box]
            cx, cy = (x1+x2)//2, (y1+y2)//2

            # TRACK MATCH
            track_id = None
            best_dist = 9999
            for tid, t in tracks.items():
                dist = get_distance((cx, cy), t["pos"])
                if dist < 55 and dist < best_dist:
                    best_dist = dist
                    track_id = tid

            if track_id is None:
                track_id_counter += 1
                track_id = track_id_counter
                tracks[track_id] = {
                    "pos": (cx, cy),
                    "history": [],
                    "plate": None,
                    "label": label,
                    "violated": False,
                    "entered": False,
                    "crossed": False,
                    "last_pos": (cx, cy),
                    "direction": "unknown"
                }

            tr = tracks[track_id]

            # ---- Movement tracking ----
            last_x, last_y = tr["pos"]
            tr["pos"] = (cx, cy)

            dx = cx - last_x
            dy = cy - last_y

            abs_dx = abs(dx)
            abs_dy = abs(dy)

            bw = x2 - x1
            bh = y2 - y1

            horizontal_move = abs_dx > 4 and abs_dx > abs_dy * 2
            vertical_move = abs_dy > 4

            # ---- Direction rule ----
            if not horizontal_move and not vertical_move:
                direction = "idle"
            elif horizontal_move and bw > bh * 1.6:
                direction = "side"
            elif dy < -2:
                direction = "up"
            elif dy > 2:
                direction = "down"
            else:
                direction = "idle"

            tr["direction"] = direction

            # 🚫 SIDE → bỏ qua hoàn toàn
            if direction == "side":
                continue

            # =========================================
            # LICENSE PLATE RECOGNITION (WITH TRACK-ID)
            # =========================================
            if tr["plate"] is None:
                try:
                    tr["plate"] = detect_and_read_plate(
                        frame,
                        (x1, y1, x2, y2),
                        track_id=track_id,
                        vehicle_label=label
                    )
                except:
                    tr["plate"] = "Unknown"

            plate = tr["plate"]

            # ROI ENTER
            if is_in_roi((x1, y1, x2, y2), ROI_POLYGON):
                tr["entered"] = True

            # STOPLINE tolerance
            tol = max(10, int((y2-y1) * 0.20))

            violated_now = False

            if light_state == "red" and tr["entered"]:
                expected_dir = "up" if CAMERA_DIRECTION_UP else "down"

                if tr["direction"] == expected_dir:
                    if CAMERA_DIRECTION_UP:
                        if y2 <= stopline_y - tol:
                            violated_now = True
                    else:
                        if y1 >= stopline_y + tol:
                            violated_now = True

                if CAMERA_DIRECTION_UP:
                    if y2 < stopline_y:
                        tr["crossed"] = True
                else:
                    if y1 > stopline_y:
                        tr["crossed"] = True

            # SAVE VIOLATION
            if violated_now and not tr["violated"]:
                tr["violated"] = True

                ts = datetime.now().strftime("%H%M%S")
                folder = os.path.join(OUTPUT_DIR, os.path.splitext(video_name)[0])
                os.makedirs(folder, exist_ok=True)

                crop = frame[y1:y2, x1:x2]
                crop_path = os.path.join(folder, f"{track_id}_{ts}_crop.jpg")
                context_path = os.path.join(folder, f"{track_id}_{ts}_context.jpg")

                if crop.size > 0:
                    cv2.imwrite(crop_path, crop)
                    ctx = frame.copy()
                    cv2.rectangle(ctx, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(ctx, "VIOLATION", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.imwrite(context_path, ctx)

                record = {
                    "video": video_name,
                    "track_id": track_id,
                    "vehicle_type": tr["label"],
                    "license_plate": plate,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "crop_image": crop_path,
                    "context_image": context_path
                }
                save_violation_record(record)

            # DRAW BOX
            color = (0,0,255) if tr["violated"] else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                frame,
                f"{label} | {plate}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                color, 2
            )

        # Draw ROI + stopline
        cv2.polylines(frame, [ROI_POLYGON], True, (255,255,0), 2)
        cv2.line(frame, (0, stopline_y), (frame_width, stopline_y), (0,0,255), 3)

        if frame_callback:
            frame_callback(frame)

        if save_output:
            out.write(frame)

        if display:
            cv2.imshow("Traffic", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return {
        "total_frames": frame_count,
        "violations": [tid for tid, t in tracks.items() if t["violated"]],
        "output_path": output_path
    }
