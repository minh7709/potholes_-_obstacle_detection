# 📋 PROJECT IMPLEMENTATION PLAN (AI-GENERATION READY)

## 1. System Overview
* **Project Name:** Pothole Detection & Streaming System
* **Goal:** Dựng hệ thống nhận diện ổ gà real-time từ luồng Camera RTSP / Webcam, hiển thị kết quả lên Web Dashboard và lưu lịch sử vào Database.
* **Tech Stack Target:**
  * **Backend:** Python 3.10+, FastAPI, OpenCV, Ultralytics YOLOv8
  * **Database:** SQLite (SQLAlchemy ORM)
  * **Frontend:** Streamlit (hoặc HTML5/JS WebSockets tùy chọn)
  * **Model:** YOLOv8 ONNX (`best.pt` sẽ được chuyển thành best.onnx sau)

Hãy tạo môi trường ảo và cài đặt các thư viện cần thiết
# # --- Compatibility & Utilities ---
# numpy==1.26.4
# opencv-python-headless==4.9.0.80

# --- Web Server ---
fastapi
uvicorn
python-multipart

# --- AI Inference (Siêu nhẹ, không dùng PyTorch/Paddle) ---
onnxruntime
rapidocr-onnxruntime

# --- Utilities ---
Pillow==10.2.0
numpy==1.26.4
opencv-python-headless==4.9.0.80


---

## 2. Architecture & Directory Structure
Yêu cầu AI tuân thủ chính xác cấu trúc thư mục sau:

```text
pothole-system/
├── assets/
│   └── best.pt               # File YOLO model
├── app/
│   ├── __init__.py
│   ├── config.py             # Quản lý cấu hình (ENV, path, threshold)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py     # SQLAlchemy engine & session
│   │   └── models.py         # DB Models (PotholeLog)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detector.py       # Core YOLOv8 Inference Logic
│   │   └── stream_handler.py # Read RTSP/Webcam stream via OpenCV
│   └── api/
│       ├── __init__.py
│       └── routes.py         # FastAPI endpoints
├── main.py                   # Entry point
├── requirements.txt
└── README.md

