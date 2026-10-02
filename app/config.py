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


# ─────────────────────────────────────────────────────────────────────
# Forward Collision Warning (FCW) System Configuration
# ─────────────────────────────────────────────────────────────────────

# --- Device / Performance ---
# "auto" uses CUDA when available, otherwise falls back to CPU.
FCW_DEVICE = os.environ.get("FCW_DEVICE", "auto")
# Depth Anything V2 is the heaviest component; it runs at its own resolution.
FCW_DEPTH_PROCESSING_RESOLUTION = int(os.environ.get("FCW_DEPTH_RESOLUTION", 518))
FCW_YOLO_IMG_SIZE = int(os.environ.get("FCW_YOLO_IMG_SIZE", 640))

# --- Models ---
FCW_YOLO_MODEL = os.environ.get("FCW_YOLO_MODEL", "yolov8n.pt")
# Save a JPEG every Nth warning frame when processing a video file for the web
# app. 1 = save every warning frame; raise it for long videos to limit storage.
FCW_SAVE_FRAME_EVERY = int(os.environ.get("FCW_SAVE_FRAME_EVERY", 1))
# Metric variant: outputs depth in metres (NOT arbitrary relative depth).
# "outdoor" is trained on driving/road scenes, which is what a dashcam sees.
FCW_DEPTH_MODEL_ID = os.environ.get(
    "FCW_DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
)

# --- Detection ---
FCW_TARGET_CLASSES = os.environ.get("FCW_TARGET_CLASSES", "motorcycle,car,bus,truck,bicycle,person").split(",")
# Only "motorcycle" drives the warning logic in v1; others are drawn for context.
FCW_PRIMARY_CLASSES = os.environ.get("FCW_PRIMARY_CLASSES", "motorcycle").split(",")
FCW_CONF_THRESHOLD = float(os.environ.get("FCW_CONF_THRESHOLD", 0.45))
FCW_IOU_THRESHOLD = float(os.environ.get("FCW_IOU_THRESHOLD", 0.5))
FCW_MIN_BOX_AREA_RATIO = float(os.environ.get("FCW_MIN_BOX_AREA_RATIO", 0.0008))

# --- Tracking (ByteTrack) ---
FCW_TRACKER = os.environ.get("FCW_TRACKER", "bytetrack.yaml")
FCW_TRACK_BUFFER = int(os.environ.get("FCW_TRACK_BUFFER", 30))
FCW_MIN_TRACK_HITS = int(os.environ.get("FCW_MIN_TRACK_HITS", 3))
# A track not seen for this many frames is considered lost and its history is dropped.
FCW_TRACK_MAX_AGE = int(os.environ.get("FCW_TRACK_MAX_AGE", 45))

# --- Depth calibration ---
# Depth Anything V2 Metric is metric in principle but not calibrated to YOUR camera.
# We measure one known reference distance once to derive the correction factors below.
FCW_DEPTH_CALIBRATION_ENABLED = os.environ.get("FCW_DEPTH_CALIBRATION", "1") == "1"
# Depth at the calibration reference distance (metres) and that reference distance (metres).
FCW_DEPTH_CALIB_REF_DEPTH_M = float(os.environ.get("FCW_DEPTH_CALIB_REF_DEPTH_M", 10.0))
FCW_DEPTH_CALIB_REF_TRUE_M = float(os.environ.get("FCW_DEPTH_CALIB_REF_TRUE_M", 10.0))
# Multiplicative gain applied after the reference fix: d_corrected = gain * d_raw.
FCW_DEPTH_GAIN = float(os.environ.get("FCW_DEPTH_GAIN", 1.0))
# Robust outlier rejection when sampling depth inside a box (fraction of pixels kept).
FCW_DEPTH_SAMPLE_TOP_RATIO = float(os.environ.get("FCW_DEPTH_SAMPLE_TOP_RATIO", 0.55))
FCW_DEPTH_MIN_SAMPLES = int(os.environ.get("FCW_DEPTH_MIN_SAMPLES", 24))
# Ignore samples that disagree with the box median by more than this ratio.
FCW_DEPTH_OUTLIER_TOLERANCE = float(os.environ.get("FCW_DEPTH_OUTLIER_TOLERANCE", 0.25))
FCW_DEPTH_VALID_MIN_M = float(os.environ.get("FCW_DEPTH_VALID_MIN_M", 1.0))
FCW_DEPTH_VALID_MAX_M = float(os.environ.get("FCW_DEPTH_VALID_MAX_M", 120.0))

# --- Distance / speed estimation ---
FCW_HISTORY_LENGTH = int(os.environ.get("FCW_HISTORY_LENGTH", 20))
# Exponential moving average factor for depth (0 = frozen, 1 = no smoothing).
FCW_DEPTH_EMA_ALPHA = float(os.environ.get("FCW_DEPTH_EMA_ALPHA", 0.4))
# Minimum frames of history before a closing speed is considered trustworthy.
FCW_MIN_SPEED_SAMPLES = int(os.environ.get("FCW_MIN_SPEED_SAMPLES", 4))
# Slope is fitted over at least this many seconds to avoid frame-to-frame noise.
FCW_SPEED_FIT_MIN_SECONDS = float(os.environ.get("FCW_SPEED_FIT_MIN_SECONDS", 0.6))
# Clamp implausible closing speeds (m/s).
FCW_MAX_CLOSING_SPEED = float(os.environ.get("FCW_MAX_CLOSING_SPEED", 25.0))

# --- Ego-motion compensation ---
# When the ego vehicle accelerates, EVERY object appears to shrink in distance.
# We estimate ego speed from optical flow of the background and subtract it.
FCW_EGO_COMPENSATION = os.environ.get("FCW_EGO_COMPENSATION", "1") == "1"
FCW_EGO_FLOW_POINTS = int(os.environ.get("FCW_EGO_FLOW_POINTS", 400))
# Depth range (m) treated as "far background" used to convert flow into ego speed.
FCW_EGO_BACKGROUND_DEPTH_M = float(os.environ.get("FCW_EGO_BACKGROUND_DEPTH_M", 15.0))
# Max plausible ego speed (m/s) accepted from optical flow.
FCW_EGO_SPEED_MAX_MPS = float(os.environ.get("FCW_EGO_SPEED_MAX_MPS", 33.0))
# Physical acceleration limit (m/s^2) used to rate-limit ego-speed changes.
FCW_EGO_MAX_ACCEL_MPS = float(os.environ.get("FCW_EGO_MAX_ACCEL_MPS", 8.0))
# Exponential smoothing factor for ego speed (higher = more responsive).
FCW_EGO_SMOOTHING = float(os.environ.get("FCW_EGO_SMOOTHING", 0.6))
# Minimum |row - horizon| (as a fraction of frame height) for a flow point to be
# usable. Near the horizon the radial flow vanishes and inverting it explodes.
FCW_EGO_MIN_YR_RATIO = float(os.environ.get("FCW_EGO_MIN_YR_RATIO", 0.12))
# Scale from measured ego speed to depth-rate bias (depth shrinks ~1/d^2 with range).
FCW_EGO_BIAS_GAIN = float(os.environ.get("FCW_EGO_BIAS_GAIN", 1.0))

# --- Time To Collision ---
# TTC = distance / closing_speed. Levels are (name, upper_bound_seconds).
FCW_TTC_WARNING_SECONDS = float(os.environ.get("FCW_TTC_WARNING_SECONDS", 4.0))
FCW_TTC_DANGER_SECONDS = float(os.environ.get("FCW_TTC_DANGER_SECONDS", 2.0))
# Ignore TTC below this distance: we are already too close to react usefully.
FCW_TTC_MIN_DISTANCE_M = float(os.environ.get("FCW_TTC_MIN_DISTANCE_M", 0.5))

# --- Warning confirmation & audio ---
# Require this many consecutive frames at a level before it is confirmed.
FCW_CONFIRM_FRAMES = int(os.environ.get("FCW_CONFIRM_FRAMES", 3))
FCW_AUDIO_ENABLED = os.environ.get("FCW_AUDIO_ENABLED", "1") == "1"
FCW_AUDIO_WARNING_COOLDOWN = float(os.environ.get("FCW_AUDIO_WARNING_COOLDOWN", 1.5))
FCW_AUDIO_DANGER_COOLDOWN = float(os.environ.get("FCW_AUDIO_DANGER_COOLDOWN", 1.0))
FCW_AUDIO_VOLUME = float(os.environ.get("FCW_AUDIO_VOLUME", 0.8))
FCW_AUDIO_DIR = os.path.join(BASE_DIR, "assets", "audio")
FCW_AUDIO_WARNING_FILE = os.path.join(FCW_AUDIO_DIR, "warning.wav")
FCW_AUDIO_DANGER_FILE = os.path.join(FCW_AUDIO_DIR, "danger.wav")

# --- Road ROI (fractions of frame width/height, 0..1) ---
# Mirrors a forward-facing dashcam: the lower-centre wedge where traffic actually is.
FCW_ROI_X_MIN = float(os.environ.get("FCW_ROI_X_MIN", 0.20))
FCW_ROI_X_MAX = float(os.environ.get("FCW_ROI_X_MAX", 0.80))
FCW_ROI_Y_MIN = float(os.environ.get("FCW_ROI_Y_MIN", 0.30))
FCW_ROI_Y_MAX = float(os.environ.get("FCW_ROI_Y_MAX", 0.97))
# How close to the horizontal centre an object's box centre must be (0..1, 1 = full width).
FCW_ROI_CENTER_TOLERANCE = float(os.environ.get("FCW_ROI_CENTER_TOLERANCE", 0.35))

# --- Display ---
FCW_DISPLAY_WIDTH = int(os.environ.get("FCW_DISPLAY_WIDTH", 1280))
FCW_SAVE_VIDEO_ENABLED = os.environ.get("FCW_SAVE_VIDEO", "0") == "1"
FCW_SAVE_VIDEO_PATH = os.environ.get("FCW_SAVE_VIDEO_PATH", os.path.join(BASE_DIR, "output", "fcw_output.mp4"))

# Ensure audio directory exists
os.makedirs(FCW_AUDIO_DIR, exist_ok=True)
