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
