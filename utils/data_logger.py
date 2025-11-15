import json
import os
import threading
from datetime import datetime

# ✅ Đảm bảo trỏ đúng tới output/violations
LOG_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "output", "violations", "violations.json"
))

# ✅ Tạo thư mục nếu chưa tồn tại
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ✅ Khóa thread-safe (tránh ghi chồng)
_lock = threading.Lock()

def save_violation_record(record: dict):
    """Lưu bản ghi vi phạm vào file JSON (thread-safe)."""
    with _lock:
        try:
            # Đọc file cũ (nếu có)
            if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            # Ghi thêm record mới
            record["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data.append(record)

            # Ghi lại file
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"[save_violation_record] ❌ Ghi JSON lỗi: {e}")
