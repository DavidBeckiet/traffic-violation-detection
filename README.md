🚦 Red-Light Violation Detection System
YOLOv8 + PaddleOCR + OpenCV – Real-time Traffic Surveillance

Hệ thống phát hiện vượt đèn đỏ thời gian thực, sử dụng các công nghệ thị giác máy tính hiện đại:

🚗 Nhận diện phương tiện bằng YOLOv8

🚥 Nhận diện trạng thái đèn giao thông (đỏ – vàng – xanh)

📍 Tracking thông minh để xác định hướng di chuyển

🔴 Phát hiện vượt đèn đỏ theo ROI + stop-line

🔎 Nhận diện biển số bằng PaddleOCR

🖼 Lưu ảnh crop biển số + ảnh toàn cảnh

📄 Xuất log JSON và video kết quả

⚡ Chạy real-time phù hợp triển khai tại giao lộ

📚 Table of Contents

Giới thiệu

Cấu trúc thư mục

Cài đặt môi trường

Chạy chương trình

Pipeline xử lý

Cấu trúc log JSON

Kết quả kiểm thử

Hạn chế

Hướng phát triển

📌 Giới thiệu

Dự án được xây dựng nhằm tự động giám sát giao thông và phát hiện các trường hợp vượt đèn đỏ, từ đó hỗ trợ hệ thống phạt nguội hoặc quản lý giao thông đô thị.

Hệ thống hoạt động theo thời gian thực, hỗ trợ FullHD và có thể triển khai tại các nút giao thông.

🚀 Cài đặt môi trường
1️⃣ Tạo môi trường Python (khuyến nghị 3.12.6)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

2️⃣ Cài đặt requirements
pip install -r requirements.txt

▶️ Chạy chương trình
py -m streamlit run app/gui_app.py

🔍 Pipeline xử lý
1️⃣ Phát hiện phương tiện (Vehicle Detection)

Sử dụng YOLOv8m

Lọc các lớp: car, motorcycle

Chuyển bounding box về kích thước gốc

Gán track_id theo chuyển động

2️⃣ Nhận diện đèn giao thông (Traffic Light Detection)

Kết hợp 2 phương pháp:

Phương pháp	Vai trò
YOLO	Phát hiện vị trí đèn
HSV Color Detection	Dự phòng khi YOLO bị miss detection

Đèn được ổn định bằng light smoothing để tránh nhấp nháy.

3️⃣ Tracking

Dựa trên:

Tâm bounding box

Khoảng cách Euclid giữa các frame

Xác định hướng di chuyển: up / down / side / idle

Xe đi ngang → loại bỏ (tránh false positive).

4️⃣ Logic vượt đèn đỏ (Red-Light Violation)

Phát hiện vi phạm khi:

Xe ∈ ROI  AND  đèn đỏ  AND đúng hướng  AND vượt qua stop-line


ROI tải từ video_zones.json

stopline_y xác định cho từng video

Có tolerance theo kích thước xe

5️⃣ Nhận diện biển số (License Plate OCR)

Sử dụng PaddleOCR:

Pipeline:

Cắt vùng biển số heuristic (dưới bounding box xe)

Tiền xử lý ảnh:

gray

enhance

threshold

OCR

Normalize biển số Việt Nam

Retry tối đa 5 lần / track_id

6️⃣ Lưu log vi phạm

Lưu trữ:

output/violations/<video_name>/
│-- <track_id>_crop.jpg
│-- <track_id>_context.jpg
│-- violations.json

📄 Cấu trúc log JSON
{
  "video": "sample.mp4",
  "track_id": 3,
  "vehicle_type": "motorcycle",
  "license_plate": "59B123456",
  "province": "HCM",
  "timestamp": "2025-01-20T10:15:23",
  "crop_image": "output/violations/sample/3_101523_crop.jpg",
  "context_image": "output/violations/sample/3_101523_context.jpg"
}

🧪 Kết quả kiểm thử
Điều kiện	Kết quả
Ban ngày	✔ Tốt
Nhiều xe trong frame	✔ Tracking ổn
Biển số rõ	✔ OCR 85–90%
Xe đi ngang	✘ Bỏ qua chính xác
Xe đi lùi	✘ Bỏ qua
📈 Hạn chế

OCR chưa tốt với biển số mờ / quá nhỏ

Cần GPU để real-time FullHD

Hiệu suất ban đêm / mưa chưa tối ưu

YOLO đôi khi miss detection → ảnh hưởng tracking

🔮 Hướng phát triển

Huấn luyện mô hình LP Detection riêng

Áp dụng Super Resolution cho biển số nhỏ

Dùng DeepSORT thay thuật toán tracking thủ công

Xây dựng dashboard giám sát real-time

Tích hợp API phạt nguội hoặc VNeID