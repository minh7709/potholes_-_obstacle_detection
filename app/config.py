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

# Static storage directories
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
VIDEOS_DIR = os.path.join(STATIC_DIR, "videos")
FRAMES_DIR = os.path.join(STATIC_DIR, "frames")

# Ensure static directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
