🎮 Hướng Dẫn Sử Dụng (User Guide)
1️⃣ Chạy giao diện người dùng (Streamlit GUI)

Nếu bạn sử dụng giao diện trực quan để thao tác:

python -m streamlit run app/gui_app.py


Giao diện sẽ bao gồm:

Chọn video đầu vào

Nút chạy phân tích

Cửa sổ hiển thị video và cảnh báo vi phạm


2️⃣ Chọn video cần phân tích

Trong giao diện GUI:

Nhấn Browse video

Chọn file video (.mp4, .avi, .mov)


3️⃣ Thiết lập vùng giám sát (ROI)
Trong config/video_zones.json
Khi chạy 1 video bất kì có thể chỉnh sửa Roi và stopline thông qua file json
Hệ thống chưa tối ưu được Roi tự động và Stopline tự động chuẩn do còn nhiều hạn chế


4️⃣ Chạy phát hiện vi phạm

Để bắt đầu:

Nhấn bắt đầu 

Hệ thống sẽ hiển thị:

Bounding box phương tiện

Trạng thái đèn giao thông (đỏ / vàng / xanh) góc trái trên video

Cảnh báo khi vi phạm vượt đèn đỏ


6️⃣ Kết quả và xuất file

Khi phát hiện vi phạm, hệ thống tự động lưu kết quả vào thư mục:

output/violations/<video_name>/


Gồm:

File	Ý nghĩa
crop.jpg	Ảnh crop biển số vi phạm
context.jpg	Ảnh toàn cảnh chứa xe
violations.json	Log vi phạm dạng JSON

Ví dụ structure:

output/violations/sample_video/
   ├── 3_101523_crop.jpg
   ├── 3_101523_context.jpg
   └── violations.json

7️⃣ Đọc log JSON

Ví dụ log JSON:

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

8️⃣ Lưu ý khi sử dụng

Video nên có góc nhìn cố định, camera không rung

Cần ánh sáng đủ rõ để OCR nhận diện biển số

Video ban đêm nên tăng sáng hoặc chạy qua module Enhance

Nếu ROI sai → kết quả vi phạm có thể sai

Nên thiết lập stop-line tương ứng với từng video khác nhau