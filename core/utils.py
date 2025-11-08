import cv2
import os
from datetime import datetime

def draw_stopline(frame, y=500):
    h, w, _ = frame.shape
    cv2.line(frame, (0, y), (w, y), (255, 0, 0), 2)

def save_violation(frame, plate):
    os.makedirs("output/violations", exist_ok=True)
    filename = f"output/violations/{plate or 'unknown'}_{datetime.now().strftime('%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
