# Pothole Detection & Streaming System - Codebase Summary

This document consolidates all text-based source files in the project to make it easier for AI models to read the codebase in a single context window, saving tokens on repeated file reads.

---

## Directory Structure

```text
d:/CNTT/NCKH/project/potholes_-_obstacle_detection/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── Plan.md
├── main.py
├── export_model.py
├── verify_system.py
├── app_streamlit.py
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── stream_handler.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── styles.css
└── assets/ (binary files)
    ├── best.pt
    └── best.onnx
```

---

## File Contents

### 1. Project Configuration & Settings

#### File: `.gitignore`
````progressbar
.venv/
.vscode/
__pycache__/
venv/
.env
app/static/potholes/
````

#### File: `.streamlit/config.toml`
````toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
````

#### File: `requirements.txt`
````text
fastapi
uvicorn
python-multipart
onnxruntime
sqlalchemy
numpy==1.26.4
opencv-python==4.9.0.80
pillow==10.2.0
ultralytics
streamlit
````

---

### 2. Main Entry Points & Verification

#### File: `main.py`
````python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router, init_stream_handler
from app.database.connection import init_db

# Use lifespan context manager for FastAPI startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables if not exist
    print("Initializing database...")
    init_db()
    
    # Startup: Initialize camera and streaming background thread
    print("Starting camera stream and detector...")
    init_stream_handler()
    
    yield
    
    # Shutdown: Stop camera capture and release resources
    print("Stopping streaming resources...")
    from app.api.routes import stream_handler
    if stream_handler:
        stream_handler.stop()

app = FastAPI(
    title="Pothole Detection & Streaming System",
    description="FastAPI + ONNX Runtime + OpenCV Pothole detection dashboard server.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for standard web client communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "app", "static")
potholes_dir = os.path.join(static_dir, "potholes")
os.makedirs(potholes_dir, exist_ok=True)

# Mount static folder under /static URL prefix
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include the API router
app.include_router(api_router, prefix="/api")

# Serve dashboard index.html at root url
@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    # Start uvicorn server locally on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
````

#### File: `export_model.py`
````python
import os
import sys

def main():
    print("Converting best.pt to ONNX format...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found. Make sure the virtual environment is activated and requirements are installed.")
        sys.exit(1)

    model_path = os.path.join("assets", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Load YOLO model
    print(f"Loading PyTorch model from {model_path}...")
    model = YOLO(model_path)
    
    # Export to ONNX
    print("Exporting model to ONNX...")
    onnx_path = model.export(format="onnx", imgsz=640)
    print(f"Model exported successfully to: {onnx_path}")

if __name__ == "__main__":
    main()
````

#### File: `verify_system.py`
````python
import sys
import numpy as np

def test_detector():
    print("Testing PotholeDetector loading and inference...")
    try:
        from app.services.detector import PotholeDetector
        detector = PotholeDetector()
        
        # Create a dummy frame (640x480 black image)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("Running detect on dummy frame...")
        boxes, confs, class_ids = detector.detect(dummy_frame)
        
        print(f"Success! Detected boxes: {boxes}, confidences: {confs}, class_ids: {class_ids}")
        return True
    except Exception as e:
        print(f"Error testing detector: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detector()
    if success:
        print("System verification PASSED.")
        sys.exit(0)
    else:
        print("System verification FAILED.")
        sys.exit(1)
````

#### File: `Plan.md`
````markdown
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
```
````

---

### 3. Application Source Code (`app/` directory)

#### File: `app/__init__.py`
````python
# Pothole Detection & Streaming System Package
````

#### File: `app/config.py`
````python
import os

# Root Directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "assets")
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "best.onnx")

# Database Configuration
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'potholes.db')}")

# Detection Parameters
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.25))
NMS_THRESHOLD = float(os.environ.get("NMS_THRESHOLD", 0.45))

# Video Source (0 for webcam, RTSP url string, or local video file path)
# Default is 0 (webcam). If webcam is not available, we will try to handle it.
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "0")

# Logging Configuration
DETECT_COOLDOWN = float(os.environ.get("DETECT_COOLDOWN", 2.0))  # seconds
SAVE_IMAGE_DIR = os.path.join(BASE_DIR, "app", "static", "potholes")

# Ensure image directory exists
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
````

#### File: `app/api/__init__.py`
````python
# API Module Init
from .routes import router, init_stream_handler
````

#### File: `app/api/routes.py`
````python
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta, timezone

from app.database.connection import get_db
from app.database.models import PotholeLog
from app.config import CONFIDENCE_THRESHOLD, NMS_THRESHOLD, VIDEO_SOURCE

router = APIRouter()

# Global StreamHandler instance reference
stream_handler = None

def init_stream_handler(detector=None):
    """Initializes and starts the stream handler."""
    global stream_handler
    from app.services.stream_handler import StreamHandler
    stream_handler = StreamHandler(detector=detector)
    stream_handler.start()
    return stream_handler

@router.get("/video_feed")
def video_feed():
    """Endpoint serving the MJPEG video stream."""
    global stream_handler
    if stream_handler is None:
        init_stream_handler()
        
    return StreamingResponse(
        stream_handler.get_latest_frame_bytes(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/detections")
def get_detections(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    severity: Optional[str] = None
):
    """Fetches the latest pothole detection logs from database."""
    query = db.query(PotholeLog)
    if severity:
        query = query.filter(PotholeLog.severity == severity)
        
    logs = query.order_by(PotholeLog.timestamp.desc()).limit(limit).all()
    return [log.to_dict() for log in logs]

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Computes real-time statistics for the dashboard dashboard."""
    # Total detection counts
    total = db.query(func.count(PotholeLog.id)).scalar() or 0
    
    # Severity breakdown
    severity_query = db.query(PotholeLog.severity, func.count(PotholeLog.id)).group_by(PotholeLog.severity).all()
    severity_breakdown = {"Low": 0, "Medium": 0, "High": 0}
    for sev, count in severity_query:
        if sev in severity_breakdown:
            severity_breakdown[sev] = count
            
    # Counts grouped by hour in local/UTC time for the past 24 hours
    # SQLite strftime('%H', timestamp) is simple and works on UTC dates
    since_time = datetime.now(timezone.utc) - timedelta(hours=24)
    hourly_query = db.query(
        func.strftime('%H', PotholeLog.timestamp).label('hour'),
        func.count(PotholeLog.id)
    ).filter(PotholeLog.timestamp >= since_time)\
     .group_by('hour')\
     .order_by('hour').all()
     
    # Fill in all 24 hours with default zeros
    current_hour = datetime.now().hour
    hourly_stats = {}
    for i in range(24):
        h = (current_hour - i) % 24
        hourly_stats[f"{h:02d}"] = 0
        
    for hour, count in hourly_query:
        # Match keys in hourly_stats
        hourly_stats[hour] = count
        
    # Sort hourly_stats chronologically
    sorted_hours = sorted(hourly_stats.keys(), key=lambda x: (int(x) - current_hour - 1) % 24)
    chronological_hourly = {h: hourly_stats[h] for h in sorted_hours}

    # Recent list (last 5 detections)
    recent_detections = db.query(PotholeLog).order_by(PotholeLog.timestamp.desc()).limit(5).all()
    recent_list = [log.to_dict() for log in recent_detections]

    return {
        "total": total,
        "severity": severity_breakdown,
        "hourly": chronological_hourly,
        "recent": recent_list
    }

@router.get("/config")
def get_config():
    """Gets current configuration values."""
    global stream_handler
    if stream_handler and stream_handler.detector:
        detector = stream_handler.detector
        return {
            "confidence_threshold": detector.conf_threshold,
            "nms_threshold": detector.nms_threshold,
            "video_source": str(stream_handler.source)
        }
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "video_source": str(VIDEO_SOURCE)
    }

@router.post("/config")
def update_config(
    confidence_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    video_source: Optional[str] = None
):
    """Updates configuration parameters on the fly."""
    global stream_handler
    if stream_handler is None:
        raise HTTPException(status_code=503, detail="Streaming service not running")
        
    if confidence_threshold is not None:
        stream_handler.detector.conf_threshold = confidence_threshold
        
    if nms_threshold is not None:
        stream_handler.detector.nms_threshold = nms_threshold
        
    if video_source is not None:
        from app.services.stream_handler import clean_video_source
        new_source = clean_video_source(video_source)
        if stream_handler.source != new_source:
            stream_handler.stop()
            stream_handler.source = new_source
            stream_handler.start()
            
    return {"status": "success", "message": "Configuration updated successfully"}
````

#### File: `app/database/__init__.py`
````python
# Database module init
from .connection import engine, SessionLocal, Base, init_db, get_db
from .models import PotholeLog
````

#### File: `app/database/connection.py`
````python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL

# SQLite specific argument connect_args={"check_same_thread": False} is required
# because FastAPI can handle requests in multiple threads.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """Initializes the database schema."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for getting database sessions in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
````

#### File: `app/database/models.py`
````python
from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime, timezone
from app.database.connection import Base

class PotholeLog(Base):
    __tablename__ = "pothole_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    severity = Column(String, nullable=False)  # "Low", "Medium", "High"
    image_path = Column(String, nullable=True)   # Relative path to static folder for web viewing

    def to_dict(self):
        """Converts the object fields to a dictionary for JSON responses."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "severity": self.severity,
            "image_path": self.image_path
        }
````

#### File: `app/services/__init__.py`
````python
# Services module init
from .detector import PotholeDetector
from .stream_handler import StreamHandler
````

#### File: `app/services/detector.py`
````python
import os
import time
import json
import cv2
import numpy as np
import onnxruntime as ort
from app.config import ONNX_MODEL_PATH, CONFIDENCE_THRESHOLD, NMS_THRESHOLD

class PotholeDetector:
    def __init__(self, model_path=ONNX_MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.class_names = {0: "pothole"}  # Default fallback class name
        
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"ONNX model file not found at: {self.model_path}. Detector will load once ONNX model becomes available.")
            return
            
        print(f"Loading ONNX Model from {self.model_path}...")
        try:
            # Use CPU execution provider as standard for compatibility
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get model inputs/outputs details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            # Try to load class names from custom model metadata
            meta = self.session.get_modelmeta().custom_metadata_map
            if 'names' in meta:
                try:
                    import ast
                    self.class_names = ast.literal_eval(meta['names'])
                    self.class_names = {int(k): v for k, v in self.class_names.items()}
                    print(f"Loaded class names from model metadata: {self.class_names}")
                except Exception as e:
                    print(f"Error parsing model metadata names: {e}, using default.")
            else:
                print("Model metadata 'names' not found, using default {0: 'pothole'}.")
        except Exception as e:
            print(f"Failed to load ONNX session: {e}")

    def preprocess(self, frame):
        """Preprocesses the image for YOLOv8 model inference.
        Uses letterbox padding to maintain aspect ratio.
        """
        h, w = frame.shape[:2]
        net_h, net_w = 640, 640
        
        # Calculate aspect ratio scaling
        r = min(net_w / w, net_h / h)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = net_w - new_unpad[0], net_h - new_unpad[1]
        
        # Divide padding equally between sides
        dw /= 2
        dh /= 2
        
        if (w, h) != new_unpad:
            im = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            im = frame.copy()
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add gray borders (value 114)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert BGR to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # Convert HWC to CHW format, normalize to [0, 1] range, and add batch dimension
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        im = im.astype(np.float32) / 255.0
        
        return im, r, (dw, dh)

    def detect(self, frame):
        """Runs object detection on the input frame.
        
        Returns:
            boxes (list): Bounding boxes [[x1, y1, x2, y2], ...]
            confidences (list): Confidence values [conf1, ...]
            class_ids (list): Class ids [id1, ...]
        """
        if self.session is None:
            self._load_model()
            if self.session is None:
                return [], [], []
                
        h, w = frame.shape[:2]
        
        # Preprocess frame
        blob, ratio, (dw, dh) = self.preprocess(frame)
        
        try:
            # Run model inference
            outputs = self.session.run(self.output_names, {self.input_name: blob})
            output = outputs[0]  # Shape: [1, 4 + num_classes, 8400]
            
            # Remove batch dimension -> [4 + num_classes, 8400]
            output = np.squeeze(output)
            # Transpose to shape [8400, 4 + num_classes] (rows representing predictions)
            output = output.T
            
            boxes = []
            confidences = []
            class_ids = []
            
            # Postprocess: extract box coords and class scores
            for row in output:
                scores = row[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence >= self.conf_threshold:
                    cx, cy, bw, bh = row[:4]
                    
                    # Convert center x, y, w, h to coordinates on preprocessed image
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    
                    # Revert padding and scaling to get original frame coordinates
                    x1 = int((x1 - dw) / ratio)
                    y1 = int((y1 - dh) / ratio)
                    x2 = int((x2 - dw) / ratio)
                    y2 = int((y2 - dh) / ratio)
                    
                    # Ensure coordinates stay within the frame boundaries
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    if box_w > 0 and box_h > 0:
                        boxes.append([x1, y1, box_w, box_h])
                        confidences.append(float(confidence))
                        class_ids.append(int(class_id))
                        
            if not boxes:
                return [], [], []
                
            # Perform Non-Maximum Suppression to remove overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            
            final_boxes = []
            final_confidences = []
            final_class_ids = []
            
            if len(indices) > 0:
                flat_indices = np.array(indices).flatten()
                for i in flat_indices:
                    x, y, bw_val, bh_val = boxes[i]
                    final_boxes.append([x, y, x + bw_val, y + bh_val])
                    final_confidences.append(confidences[i])
                    final_class_ids.append(class_ids[i])
                    
            return final_boxes, final_confidences, final_class_ids
        except Exception as e:
            print(f"Error during model execution or postprocessing: {e}")
            return [], [], []
````

#### File: `app/services/stream_handler.py`
````python
import os
import cv2
import time
import uuid
import re
import numpy as np
from datetime import datetime, timezone
import threading
from app.config import VIDEO_SOURCE, SAVE_IMAGE_DIR, DETECT_COOLDOWN
from app.services.detector import PotholeDetector
from app.database.connection import SessionLocal
from app.database.models import PotholeLog

def clean_video_source(source):
    """Sanitizes and normalizes the camera source (int for webcam, string for URLs)."""
    if isinstance(source, int):
        return source
    if isinstance(source, str):
        cleaned = source.strip()
        
        # If numeric (like "0"), convert to int
        if cleaned.isdigit():
            return int(cleaned)
            
        # Fix common typo patterns like http;// or http:/ or spaces
        if cleaned.lower().startswith("http;"):
            cleaned = "http:" + cleaned[5:]
        elif cleaned.lower().startswith("https;"):
            cleaned = "https:" + cleaned[6:]
            
        # Normalize slashes (e.g. http:// instead of http:/ or http:///)
        cleaned = re.sub(r'^(https?):[/\s;]+', r'\1://', cleaned, flags=re.IGNORECASE)
        # Remove all spaces inside the URL
        cleaned = cleaned.replace(" ", "")
        
        return cleaned
    return source

class StreamHandler:
    def __init__(self, source=VIDEO_SOURCE, detector=None):
        self.source = clean_video_source(source)

        self.detector = detector or PotholeDetector()
        self.cap = None
        self.running = False
        self.last_detection_time = 0
        self.lock = threading.Lock()
        self.latest_frame = None

    def _create_capture(self):
        """Creates a cv2.VideoCapture with the optimal backend on Windows."""
        if isinstance(self.source, int):
            # DirectShow works much better and avoids MSMF errors for webcams
            return cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        elif isinstance(self.source, str) and self.source.startswith(("http://", "https://", "rtsp://", "rtmp://")):
            # Force FFmpeg for network streams to prevent MSMF from attempting to open them
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            return cv2.VideoCapture(self.source)

    def start(self):
        """Starts the capture and processing thread."""
        with self.lock:
            if self.running:
                return
            self.running = True
            
        self.cap = self._create_capture()
        if not self.cap.isOpened():
            print(f"Warning: Could not open video source {self.source}")
            
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("Video capture thread started.")

    def stop(self):
        """Stops the capture and processing thread."""
        with self.lock:
            self.running = False
        if self.cap:
            self.cap.release()
        print("Video capture stopped.")

    def _capture_loop(self):
        """Internal loop to continuously read and process frames from the camera."""
        while True:
            with self.lock:
                if not self.running:
                    break
                    
            if self.cap is None or not self.cap.isOpened():
                # Attempt reconnection
                time.sleep(2.0)
                self.cap = self._create_capture()
                continue

            ret, frame = self.cap.read()
            if not ret:
                # End of video file or camera disconnected
                time.sleep(0.03)
                continue

            # Run detection on the raw frame
            boxes, confidences, class_ids = self.detector.detect(frame)
            
            # Process detections (log and draw boxes)
            processed_frame = self._process_detections(frame, boxes, confidences)
            
            with self.lock:
                self.latest_frame = processed_frame
                
            # Control frame rate roughly
            time.sleep(0.01)

    def _process_detections(self, frame, boxes, confidences):
        """Analyzes bounding boxes, draws them, and logs them if outside cooldown."""
        draw_frame = frame.copy()
        h, w = frame.shape[:2]
        frame_area = w * h
        
        detected_potholes = []
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            # Severity based on bounding box size relative to the frame
            ratio = box_area / frame_area
            if ratio > 0.04:
                severity = "High"
                color = (0, 0, 255)       # Red for High
            elif ratio > 0.012:
                severity = "Medium"
                color = (0, 165, 255)     # Orange for Medium
            else:
                severity = "Low"
                color = (0, 255, 0)       # Green for Low
                
            detected_potholes.append({
                "box": box,
                "confidence": conf,
                "severity": severity,
                "color": color
            })
            
            # Draw bbox
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 3)
            
            # Label text
            label = f"Pothole {conf:.2f} ({severity})"
            
            # Text styling
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(draw_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(draw_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Log to Database if cooldown has passed
        current_time = time.time()
        if detected_potholes and (current_time - self.last_detection_time >= DETECT_COOLDOWN):
            # Log the highest confidence or largest detection in this frame
            best_det = max(detected_potholes, key=lambda x: x["confidence"])
            x1, y1, x2, y2 = best_det["box"]
            conf = best_det["confidence"]
            severity = best_det["severity"]
            
            # Save the detection frame to disk
            filename = f"pothole_{int(current_time)}_{uuid.uuid4().hex[:6]}.jpg"
            file_path = os.path.join(SAVE_IMAGE_DIR, filename)
            
            # Save frame
            cv2.imwrite(file_path, frame)
            
            # Database record insertion
            db = SessionLocal()
            try:
                db_log = PotholeLog(
                    confidence=conf,
                    bbox_x1=int(x1),
                    bbox_y1=int(y1),
                    bbox_x2=int(x2),
                    bbox_y2=int(y2),
                    severity=severity,
                    image_path=f"/static/potholes/{filename}"
                )
                db.add(db_log)
                db.commit()
                self.last_detection_time = current_time
                print(f"Logged detection to DB: {severity} severity, confidence {conf:.2f}")
            except Exception as e:
                print(f"Error logging detection to DB: {e}")
                db.rollback()
            finally:
                db.close()
                
        return draw_frame

    def get_latest_frame_bytes(self):
        """Yields MJPEG stream frame bytes."""
        # Initialize camera if not running
        if not self.running:
            self.start()
            
        placeholder_sent = False
        
        while True:
            with self.lock:
                frame = self.latest_frame
                
            if frame is None:
                if not placeholder_sent:
                    # Create offline placeholder frame
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, "Camera Starting / Offline", (120, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    ret, jpeg = cv2.imencode('.jpg', img)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                        placeholder_sent = True
                time.sleep(0.1)
                continue
                
            placeholder_sent = False
            
            # Encode frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                time.sleep(0.03)
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # Throttle stream output to ~30 FPS
            time.sleep(0.033)
````

---

### 4. User Interfaces & Frontend Components

#### File: `app_streamlit.py`
````python
import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

# Backend API server URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RoadVision - Pothole Detection & Telemetry System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Theme Custom styling
st.markdown("""
<style>
    /* Dark premium canvas background */
    .stApp {
        background-color: #080a15;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.06) 0px, transparent 50%);
    }
    
    /* Title text styling */
    h1, h2, h3, h4, h5 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }
    
    /* System Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34d399;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }
    
    /* Glassmorphic Metrics Card */
    .metric-card {
        background: rgba(16, 20, 38, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(16px);
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.25rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if backend server is online
def is_backend_online():
    try:
        response = requests.get(f"{API_URL}/api/config", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Setup Sidebar System Control Panel
st.sidebar.markdown('<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2rem;">'
                    '<div style="width: 32px; height: 32px; background: linear-gradient(135deg, #6366f1, #a855f7); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(99,102,241,0.3);">'
                    '<span style="color: white; font-weight: bold; font-size: 1.1rem;">R</span></div>'
                    '<span style="font-size: 1.5rem; font-weight: 800; font-family: \'Outfit\', sans-serif; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">RoadVision</span>'
                    '</div>', unsafe_allow_html=True)

backend_online = is_backend_online()

if backend_online:
    st.sidebar.markdown('<div class="status-badge"><div class="status-dot"></div>Analyzing Live Stream</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge" style="background: rgba(244, 63, 94, 0.12); border-color: rgba(244, 63, 94, 0.25); color: #fda4af;">'
                        '<div class="status-dot" style="background-color: #f43f5e; box-shadow: 0 0 8px #f43f5e;"></div>Offline</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Detector Settings")

# Fetch configuration values from API
config_data = {"confidence_threshold": 0.25, "nms_threshold": 0.45, "video_source": "0"}
if backend_online:
    try:
        config_response = requests.get(f"{API_URL}/api/config")
        if config_response.status_code == 200:
            config_data = config_response.json()
    except Exception as e:
        st.sidebar.error(f"Error fetching config: {e}")

# Form to hold slider states
with st.sidebar.form("config_form"):
    conf_val = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=float(config_data["confidence_threshold"]), step=0.05)
    nms_val = st.slider("NMS Threshold (Overlap)", min_value=0.1, max_value=0.9, value=float(config_data["nms_threshold"]), step=0.05)
    source_val = st.text_input(
        "Stream Video Source", 
        value=str(config_data["video_source"]),
        help="Nhập '0' cho webcam máy tính, hoặc link MJPEG từ điện thoại (ví dụ: http://192.168.1.5:8080/stream.mjpg)."
    )
    submit_config = st.form_submit_button("Apply Configuration")

    if submit_config and backend_online:
        try:
            post_url = f"{API_URL}/api/config?confidence_threshold={conf_val}&nms_threshold={nms_val}&video_source={source_val}"
            res = requests.post(post_url)
            if res.status_code == 200:
                st.sidebar.success("Configuration updated!")
                # Give a short sleep for settings to apply
                time.sleep(0.5)
                st.rerun()
            else:
                st.sidebar.error(f"Failed to update config: {res.text}")
        except Exception as e:
            st.sidebar.error(f"Communication error: {e}")

st.sidebar.markdown("---")
live_reload = st.sidebar.checkbox("🔄 Live Autorefresh", value=True)
if live_reload:
    st.sidebar.caption("Auto-refreshing dashboard metrics every 3 seconds.")

# Main Page Layout
st.markdown("<h1 style='margin-bottom: 5px;'>Pothole Detection & Telemetry</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1rem; margin-bottom: 2rem;'>Real-time AI roadway inspection camera and statistics logs</p>", unsafe_allow_html=True)

if not backend_online:
    st.error("⚠️ Backend API Server is Offline. Please start the backend server by running `uvicorn main:app --reload` first.")
    st.info("💡 Make sure you are running the command in the virtual environment `.venv\\Scripts\\activate`.")
    st.stop()

# Load Statistics from Backend API
stats = {"total": 0, "severity": {"Low": 0, "Medium": 0, "High": 0}, "hourly": {}, "recent": []}
try:
    stats_response = requests.get(f"{API_URL}/api/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
except Exception as e:
    st.error(f"Error connecting to backend API: {e}")

# Grid Layout: Stream Feed & Metrics
col_stream, col_telemetry = st.columns([2, 1])

with col_stream:
    st.markdown("### 📹 Live Analysis Stream Feed")
    # Stream embedded MJPEG from backend
    st.markdown(
        f'<div style="display: flex; width: 100%; justify-content: center; background: #000; border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.07); aspect-ratio: 16/9;">'
        f'  <img src="{API_URL}/api/video_feed" style="width: 100%; object-fit: contain;">'
        f'</div>',
        unsafe_allow_html=True
    )

with col_telemetry:
    st.markdown("### 📊 Live Telemetry")
    
    # Render custom metric card for Total potholes
    st.markdown(
        f'<div class="metric-card">'
        f'  <div class="metric-label">Total Potholes Detected</div>'
        f'  <div class="metric-value">{stats["total"]}</div>'
        f'  <div class="metric-sub">Across all inspection streams</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Render severity metrics breakdown
    st.markdown("#### ⚠️ Severity Distribution")
    sev_data = stats["severity"]
    
    # Create simple data table
    sev_df = pd.DataFrame({
        "Severity": ["Low", "Medium", "High"],
        "Count": [sev_data.get("Low", 0), sev_data.get("Medium", 0), sev_data.get("High", 0)]
    })
    
    # Color-coded charts using Streamlit Native Bar Chart
    st.bar_chart(sev_df.set_index("Severity"), color="#818cf8", height=200)

# Lower Section: Timeline & Logs Table
st.markdown("---")
col_timeline, col_logs = st.columns([1, 1])

with col_timeline:
    st.markdown("### 📈 Detection Timeline (Past 24 Hours)")
    if stats["hourly"]:
        hourly_df = pd.DataFrame({
            "Hour": [f"{h}:00" for h in stats["hourly"].keys()],
            "Detections": list(stats["hourly"].values())
        })
        st.line_chart(hourly_df.set_index("Hour"), color="#a855f7", height=260)
    else:
        st.info("No logs in the past 24 hours.")

with col_logs:
    st.markdown("### 📝 Recent Pothole Records")
    recent_list = stats["recent"]
    
    if not recent_list:
        st.info("Waiting for first pothole detection from live stream...")
    else:
        # Create a display list of logs
        log_records = []
        for log in recent_list:
            # Parse ISO timestamp
            dt = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))
            local_time = dt.strftime("%H:%M:%S")
            log_records.append({
                "Record ID": log["id"],
                "Time": local_time,
                "Confidence": f"{int(log['confidence'] * 100)}%",
                "Severity": log["severity"],
                "Image Path": log["image_path"]
            })
            
        logs_df = pd.DataFrame(log_records)
        
        # Display the logs table (excluding Image Path for display size)
        st.dataframe(logs_df[["Time", "Confidence", "Severity"]], use_container_width=True, hide_index=True)

        # Snapshot Preview functionality
        st.markdown("#### 🔍 Snapshot Inspector")
        # Let user choose a record ID to preview
        selected_record = st.selectbox(
            "Select detection to preview snapshot:",
            options=[r["Record ID"] for r in log_records],
            format_func=lambda x: f"Record #{x} - {next(item for item in log_records if item['Record ID'] == x)['Severity']} Severity"
        )
        
        if selected_record:
            item = next(i for i in log_records if i["Record ID"] == selected_record)
            img_url = f"{API_URL}{item['Image Path']}"
            st.image(img_url, caption=f"Pothole Snapshot (Record #{item['Record ID']} - Detected at {item['Time']})", use_container_width=True)

# Handle live autorefresh mechanism
if live_reload:
    time.sleep(3.0)
    st.rerun()
````

#### File: `app/static/index.html`
````html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pothole Detection Dashboard</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Stylesheet -->
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="app-container">
        <!-- Sidebar / Navigation -->
        <aside class="sidebar">
            <div class="logo-area">
                <div class="logo-icon">
                    <i data-lucide="scan-eye"></i>
                </div>
                <span>RoadVision</span>
            </div>
            
            <nav class="nav-menu">
                <a href="#" class="nav-item active">
                    <i data-lucide="layout-dashboard"></i>
                    <span>Dashboard</span>
                </a>
                <a href="#history" class="nav-item" id="nav-history">
                    <i data-lucide="history"></i>
                    <span>Log History</span>
                </a>
                <a href="#settings" class="nav-item" id="nav-settings">
                    <i data-lucide="settings"></i>
                    <span>Settings</span>
                </a>
            </nav>
            
            <div class="system-status">
                <div class="status-indicator online"></div>
                <div class="status-details">
                    <p class="status-title">System Status</p>
                    <p class="status-sub">Online & Analyzing</p>
                </div>
            </div>
        </aside>

        <!-- Main Content Area -->
        <main class="main-content">
            <!-- Header -->
            <header class="app-header">
                <div class="header-title">
                    <h1>Pothole Detection Dashboard</h1>
                    <p>Real-time roadway inspection and telemetry analysis</p>
                </div>
                <div class="header-actions">
                    <span class="time-badge" id="current-time">00:00:00 AM</span>
                </div>
            </header>

            <!-- Grid Layout -->
            <div class="dashboard-grid">
                <!-- Live Video Feed (Left Column) -->
                <div class="card video-card">
                    <div class="card-header">
                        <div class="card-title">
                            <i data-lucide="video" class="icon-indigo"></i>
                            <h2>Live Stream Feed</h2>
                        </div>
                        <span class="badge badge-danger blink">LIVE</span>
                    </div>
                    <div class="video-container">
                        <img id="video-stream" src="/api/video_feed" alt="Video stream feed offline">
                    </div>
                </div>

                <!-- Telemetry Cards (Right Column) -->
                <div class="telemetry-column">
                    <!-- Total Potholes Card -->
                    <div class="card stat-card pulse-blue">
                        <div class="stat-icon icon-rose">
                            <i data-lucide="alert-triangle"></i>
                        </div>
                        <div class="stat-info">
                            <h3>Total Potholes</h3>
                            <p class="stat-value" id="stat-total">0</p>
                            <span class="stat-desc">Detected overall</span>
                        </div>
                    </div>

                    <!-- Severity breakdown card -->
                    <div class="card chart-card">
                        <div class="card-header">
                            <div class="card-title">
                                <i data-lucide="pie-chart" class="icon-purple"></i>
                                <h2>Severity Distribution</h2>
                            </div>
                        </div>
                        <div class="chart-container-donut">
                            <canvas id="severity-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Second Grid: Hourly Detection & Real-time Logs -->
            <div class="dashboard-grid-lower">
                <!-- Hourly Timeline -->
                <div class="card chart-card-lower">
                    <div class="card-header">
                        <div class="card-title">
                            <i data-lucide="trending-up" class="icon-emerald"></i>
                            <h2>Detection History (Past 24 Hours)</h2>
                        </div>
                    </div>
                    <div class="chart-container-line">
                        <canvas id="timeline-chart"></canvas>
                    </div>
                </div>

                <!-- Recent Logs Table -->
                <div class="card table-card">
                    <div class="card-header">
                        <div class="card-title">
                            <i data-lucide="list" class="icon-amber"></i>
                            <h2>Recent Detection Logs</h2>
                        </div>
                        <button class="btn btn-secondary btn-sm" id="btn-refresh-logs">
                            <i data-lucide="refresh-cw"></i> Refresh
                        </button>
                    </div>
                    <div class="table-container">
                        <table class="logs-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Confidence</th>
                                    <th>Severity</th>
                                    <th>Preview</th>
                                </tr>
                            </thead>
                            <tbody id="logs-tbody">
                                <!-- Dynamically filled -->
                                <tr>
                                    <td colspan="4" class="empty-state">No detections yet... Waiting for stream</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Configuration Modal / Section -->
    <div class="modal" id="settings-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3><i data-lucide="settings" style="vertical-align: middle; margin-right: 8px;"></i> System Settings</h3>
                <button class="close-btn" id="close-settings"><i data-lucide="x"></i></button>
            </div>
            <div class="modal-body">
                <form id="settings-form">
                    <div class="form-group">
                        <label for="conf-slider">Confidence Threshold: <span id="conf-val">0.25</span></label>
                        <input type="range" id="conf-slider" min="0.1" max="0.9" step="0.05" value="0.25">
                        <p class="form-help">Minimum confidence required to classify an object as a pothole.</p>
                    </div>
                    <div class="form-group">
                        <label for="nms-slider">NMS Threshold: <span id="nms-val">0.45</span></label>
                        <input type="range" id="nms-slider" min="0.1" max="0.9" step="0.05" value="0.45">
                        <p class="form-help">Non-Maximum Suppression threshold (reduces overlapping boxes).</p>
                    </div>
                    <div class="form-group">
                        <label for="source-input">Camera / Video Source</label>
                        <input type="text" id="source-input" value="0" placeholder="Camera index e.g. 0, RTSP stream URL or Video Path">
                        <p class="form-help">0 for local webcam, or use RTSP stream address/video file path.</p>
                    </div>
                    <button type="submit" class="btn btn-primary btn-block">Apply Changes</button>
                </form>
            </div>
        </div>
    </div>

    <!-- Image Preview Modal -->
    <div class="modal" id="preview-modal">
        <div class="modal-content preview-content">
            <div class="modal-header">
                <h3><i data-lucide="image" style="vertical-align: middle; margin-right: 8px;"></i> Detection Snapshot</h3>
                <button class="close-btn" id="close-preview"><i data-lucide="x"></i></button>
            </div>
            <div class="modal-body preview-body">
                <img id="preview-img" src="" alt="Snapshot of detected pothole">
                <div class="preview-details">
                    <span class="badge" id="preview-severity">HIGH</span>
                    <span class="preview-meta" id="preview-time">Time: 2026-07-25 16:30:15 UTC</span>
                    <span class="preview-meta" id="preview-conf">Confidence: 89%</span>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>
````

#### File: `app/static/styles.css`
````css
:root {
    --bg-dark: #070913;
    --bg-main: #0c0f1d;
    --bg-glass: rgba(16, 20, 38, 0.65);
    --bg-glass-hover: rgba(23, 28, 51, 0.75);
    --border-glass: rgba(255, 255, 255, 0.07);
    --border-glass-focus: rgba(99, 102, 241, 0.4);
    
    --primary: #6366f1;
    --primary-light: #818cf8;
    --primary-grad: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    
    --rose: #f43f5e;
    --rose-grad: linear-gradient(135deg, #f43f5e 0%, #fb7185 100%);
    --emerald: #10b981;
    --emerald-grad: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    --amber: #f59e0b;
    --amber-grad: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
    
    --shadow-main: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
    --font-outfit: 'Outfit', sans-serif;
    --font-inter: 'Inter', sans-serif;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
}

/* Scrollbar Customization */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: var(--font-inter);
    min-height: 100vh;
    overflow-x: hidden;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.08) 0px, transparent 50%);
}

.app-container {
    display: flex;
    min-height: 100vh;
}

/* Sidebar Styling */
.sidebar {
    width: 260px;
    background: var(--bg-glass);
    border-right: 1px solid var(--border-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    padding: 2rem 1.5rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: fixed;
    height: 100vh;
    z-index: 10;
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 3rem;
}

.logo-icon {
    width: 40px;
    height: 40px;
    background: var(--primary-grad);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}

.logo-icon i {
    color: white;
    width: 20px;
    height: 20px;
}

.logo-area span {
    font-family: var(--font-outfit);
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: var(--primary-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nav-menu {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    flex-grow: 1;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.85rem 1rem;
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: 10px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-weight: 500;
    font-size: 0.95rem;
}

.nav-item i {
    width: 20px;
    height: 20px;
    transition: color 0.3s;
}

.nav-item:hover {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
}

.nav-item.active {
    background: var(--primary-grad);
    color: white;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
}

.nav-item.active i {
    color: white;
}

.system-status {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.status-indicator.online {
    background-color: var(--emerald);
    box-shadow: 0 0 10px var(--emerald);
}

.status-details {
    display: flex;
    flex-direction: column;
}

.status-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-primary);
}

.status-sub {
    font-size: 0.7rem;
    color: var(--text-muted);
}

/* Main Content Area */
.main-content {
    flex-grow: 1;
    margin-left: 260px;
    padding: 2rem 2.5rem;
    min-height: 100vh;
}

/* Header Styling */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border-glass);
    padding-bottom: 1.5rem;
}

.header-title h1 {
    font-family: var(--font-outfit);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0.25rem;
}

.header-title p {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

.time-badge {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-family: monospace;
    font-size: 0.9rem;
    color: var(--primary-light);
}

/* Grid Layouts */
.dashboard-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}

.dashboard-grid-lower {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

/* Card General Design */
.card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: var(--shadow-main);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, transparent 100%);
    pointer-events: none;
}

.card:hover {
    border-color: rgba(255, 255, 255, 0.12);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
}

.card-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-title i {
    width: 20px;
    height: 20px;
}

.card-title h2 {
    font-family: var(--font-outfit);
    font-size: 1.2rem;
    font-weight: 600;
}

/* Colors for Icons */
.icon-indigo { color: var(--primary); }
.icon-purple { color: #a855f7; }
.icon-rose { color: var(--rose); }
.icon-emerald { color: var(--emerald); }
.icon-amber { color: var(--amber); }

/* Video Feed Container */
.video-card {
    padding: 1.5rem;
}

.video-container {
    width: 100%;
    aspect-ratio: 16/9;
    background: #000;
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(255,255,255,0.05);
}

.video-container img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* Telemetry Column */
.telemetry-column {
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

/* Stat Cards */
.stat-card {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.75rem;
}

.stat-icon {
    width: 60px;
    height: 60px;
    border-radius: 15px;
    background: rgba(244, 63, 94, 0.1);
    border: 1px solid rgba(244, 63, 94, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
}

.stat-icon i {
    width: 28px;
    height: 28px;
}

.stat-info h3 {
    font-size: 0.9rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stat-value {
    font-family: var(--font-outfit);
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0.15rem 0;
    background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-desc {
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* Chart Card Donut */
.chart-container-donut {
    position: relative;
    height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Lower Grid Containers */
.chart-card-lower {
    grid-column: span 1;
}

.chart-container-line {
    position: relative;
    height: 250px;
    width: 100%;
}

/* Table Design */
.table-card {
    padding: 1.5rem;
}

.table-container {
    overflow-y: auto;
    max-height: 250px;
    margin-top: 0.5rem;
}

.logs-table {
    width: 100%;
    border-collapse: collapse;
    text-align: left;
}

.logs-table th, .logs-table td {
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
}

.logs-table th {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
}

.logs-table tbody tr {
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    transition: background 0.2s;
}

.logs-table tbody tr:hover {
    background: rgba(255, 255, 255, 0.02);
}

.empty-state {
    text-align: center;
    color: var(--text-muted);
    padding: 3rem !important;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-danger {
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.3);
    color: #fda4af;
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fde047;
}

.badge-success {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

/* Buttons */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.6rem 1.2rem;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
}

.btn-sm {
    padding: 0.4rem 0.8rem;
    font-size: 0.8rem;
    border-radius: 8px;
}

.btn-primary {
    background: var(--primary-grad);
    color: white;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
}

.btn-primary:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
    transform: translateY(-1px);
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
    border: 1px solid var(--border-glass);
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1);
}

.btn-block {
    width: 100%;
}

.btn-link {
    background: none;
    border: none;
    color: var(--primary-light);
    cursor: pointer;
    font-weight: 600;
    padding: 0;
    font-size: 0.85rem;
    text-decoration: underline;
}

.btn-link:hover {
    color: white;
}

/* Modals */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(7, 9, 19, 0.8);
    backdrop-filter: blur(8px);
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
    z-index: 100;
}

.modal.active {
    opacity: 1;
    pointer-events: auto;
}

.modal-content {
    background: #0d1127;
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    width: 90%;
    max-width: 500px;
    padding: 2rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    transform: scale(0.9);
    transition: transform 0.3s ease;
}

.modal.active .modal-content {
    transform: scale(1);
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
}

.modal-header h3 {
    font-family: var(--font-outfit);
    font-size: 1.25rem;
    font-weight: 600;
}

.close-btn {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
}

.close-btn:hover {
    color: white;
}

/* Image Preview Modal Specifics */
.preview-content {
    max-width: 650px;
    padding: 1.5rem;
}

.preview-body img {
    width: 100%;
    border-radius: 12px;
    object-fit: cover;
    aspect-ratio: 16/9;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1rem;
}

.preview-details {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.preview-meta {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* Settings Form elements */
.form-group {
    margin-bottom: 1.25rem;
}

.form-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-secondary);
}

.form-group input[type="range"] {
    width: 100%;
    accent-color: var(--primary);
}

.form-group input[type="text"] {
    width: 100%;
    padding: 0.6rem 0.8rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border-glass);
    border-radius: 8px;
    color: white;
    font-size: 0.9rem;
    transition: border-color 0.2s;
}

.form-group input[type="text"]:focus {
    border-color: var(--primary-light);
    outline: none;
}

.form-help {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* Animations */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.blink {
    animation: blink 1.5s infinite;
}

@keyframes pulse-blue {
    0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.1); }
    70% { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
    100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
}

.pulse-blue:hover {
    animation: pulse-blue 1.5s infinite;
}

/* Responsive Grid Adaptation */
@media (max-width: 1024px) {
    .dashboard-grid {
        grid-template-columns: 1fr;
    }
    .dashboard-grid-lower {
        grid-template-columns: 1fr;
    }
}
````

---

## 5. How to Run (Single Command)

### File: `run_all.sh` (đã tạo trong project)

Script này khởi động đồng thời FastAPI backend (port 8000) và Streamlit dashboard (port 8501) trong cùng một terminal.

**Cách dùng:**
```bash
chmod +x run_all.sh
./run_all.sh
```
Nhấn `Ctrl+C` để dừng tất cả.

### Services & Ports

| Service | Port | URL | Command |
|---------|------|-----|---------|
| FastAPI Backend | 8000 | http://localhost:8000 | `python main.py` |
| Streamlit Dashboard | 8501 | http://localhost:8501 | `streamlit run app_streamlit.py` |
| Web Dashboard (HTML/JS) | 8000 | http://localhost:8000 | (do FastAPI serve) |

### API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | Web Dashboard |
| `/api/video_feed` | GET | MJPEG live stream |
| `/api/detections?limit=N&severity=` | GET | Danh sách phát hiện |
| `/api/stats` | GET | Thống kê real-time |
| `/api/config` | GET | Xem cấu hình hiện tại |
| `/api/config?confidence_threshold=&nms_threshold=&video_source=` | POST | Cập nhật cấu hình |

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│              run_all.sh                      │
│  (khởi động cả 2 service cùng lúc)           │
└──────────────────┬──────────────────┬────────┘
                   │                  │
         ┌─────────▼──────┐  ┌───────▼──────────┐
         │  main.py       │  │ app_streamlit.py │
         │  (FastAPI)     │  │ (Streamlit)      │
         │  Port 8000     │  │ Port 8501        │
         └────────┬───────┘  └───────┬──────────┘
                  │                  │
         ┌────────▼───────┐         │
         │  app/           │◄────────┘ (HTTP requests)
         │  ├── api/       │
         │  ├── database/  │
         │  ├── services/  │
         │  └── static/    │
         └────────────────┘
```

### Cấu trúc thư mục & Chức năng

| File/Folder | Chức năng |
|-------------|-----------|
| `main.py` | Entry point FastAPI server. Khởi tạo DB, camera stream, API routes. |
| `app_streamlit.py` | Streamlit dashboard UI. Hiển thị live video, thống kê, logs. |
| `app/config.py` | Quản lý cấu hình (model paths, thresholds, video source). |
| `app/database/` | SQLAlchemy ORM + SQLite. Model `PotholeLog`. |
| `app/services/detector.py` | YOLOv8 ONNX inference engine. |
| `app/services/stream_handler.py` | Multi-threaded camera capture + inference pipeline. |
| `app/api/routes.py` | REST API endpoints. |
| `app/static/` | Web dashboard frontend (HTML/CSS/JS + Chart.js). |
| `assets/` | Model files (`best.pt`, `best.onnx`). |
| `export_model.py` | Convert `best.pt` → `best.onnx`. |
| `verify_system.py` | Kiểm tra hệ thống (detector loading). |
| `run_all.sh` | Script chạy tất cả services. |

### Hướng dẫn cài đặt

```bash
# 1. Tạo virtual environment
python -m venv .venv

# 2. Kích hoạt
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Export ONNX model (nếu chưa có best.onnx)
python export_model.py

# 5. Chạy tất cả
./run_all.sh
```

### Kiểm tra hệ thống

```bash
python verify_system.py
```
