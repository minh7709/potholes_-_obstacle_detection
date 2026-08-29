# Yêu cầu xây dựng ứng dụng web nhận diện ổ gà (Pothole Detection) với FastAPI lại từ đầu

Chào bạn, tôi đang phát triển một hệ thống nhận diện ổ gà sử dụng AI. Trước đây tôi sử dụng Streamlit, nhưng hiện tại tôi muốn đập đi xây lại với kiến trúc backend FastAPI và frontend dùng HTML, CSS, JS thuần. Dưới đây là cấu trúc thư mục hiện tại của project và các yêu cầu chi tiết.

## 1. Cấu trúc thư mục tham khảo
Dựa trên kiến trúc hiện tại, code của bạn sẽ cần fill vào các file sau:
```text
├── app/
│   ├── api/
│   │   └── routes.py           # Định nghĩa các API endpoints
│   ├── database/
│   │   ├── connection.py       # Kết nối database SQLite
│   │   └── models.py           # Định nghĩa các table (SQLAlchemy)
│   ├── services/
│   │   ├── detector.py         # Logic chạy model YOLO xử lý video và trích xuất frame
│   │   └── video_upload_service.py # Logic lưu file upload
│   └── static/                 # Chứa HTML, CSS, JS và thư mục lưu file upload
│       └── index.html
├── assets/
│   └── best.pt                 # File weights của mô hình AI
├── main.py                     # File entry point của FastAPI
└── potholes.db                 # SQLite database
```

## 2. Yêu cầu chi tiết về Backend (FastAPI)
- **Reset Database:** Viết một đoạn logic hoặc script để xóa toàn bộ dữ liệu cũ có trong database (`potholes.db`) mỗi lần chạy lệnh reset hoặc khi hệ thống khởi tạo lại theo yêu cầu.
- **Upload và Lưu trữ Video:** Xây dựng API cho phép upload video từ client. Video phải được lưu trữ vào thư mục public tĩnh (ví dụ `app/static/videos/`) để client có thể fetch và xem lại trực tiếp trên trình duyệt.
- **Xử lý AI (YOLO / OpenCV):** Xây dựng logic trong `detector.py` sử dụng mô hình AI (file `assets/best.onnx`) để đọc video vừa upload theo từng luồng/khung hình. 
- **Lưu trữ Kết quả (Database & File):**
  - Trong quá trình model infer, nếu phát hiện có ổ gà trong một frame, hệ thống phải crop hoặc lưu toàn bộ frame đó thành một file ảnh (ví dụ lưu vào `app/static/frames/`).
  - Thiết kế database schema (`models.py`) để lưu trữ metadata bao gồm: thông tin video gốc, đường dẫn tới file ảnh của khung hình chứa ổ gà, và các thông số độ tin cậy (confidence score) của từng ổ gà xuất hiện trong khung hình đó.

## 3. Yêu cầu chi tiết về Frontend (HTML/CSS/JS thuần)
- Giao diện đơn giản, code trực tiếp vào thư mục `app/static/`.
- Cung cấp một form upload video.
- Sau khi upload thành công, giao diện có một thẻ `<video>` để phát lại video vừa được tải lên.
- Sử dụng Javascript (Fetch API) để giao tiếp với backend, kích hoạt quá trình nhận diện AI.
- Sau khi AI xử lý xong, hiển thị kết quả phân tích bên dưới: Bao gồm danh sách các khung hình (dạng hình ảnh) chứa ổ gà, kèm theo danh sách độ tin cậy (confidence) của các ổ gà được phát hiện trong bức ảnh đó.

## 4. Yêu cầu về Output
Hãy cung cấp mã nguồn đầy đủ cho các file sau để hệ thống có thể hoạt động được ngay:
1. `main.py`
2. `app/database/models.py` & `app/database/connection.py`
3. `app/api/routes.py`
4. `app/services/detector.py`
5. `app/static/index.html` (chứa luôn css và js hoặc tách file tùy ý)