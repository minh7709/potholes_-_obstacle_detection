import os

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


def is_image_path(path):
    return os.path.splitext(str(path))[1].lower() in IMAGE_EXTENSIONS


def is_video_path(path):
    return os.path.splitext(str(path))[1].lower() in VIDEO_EXTENSIONS


def parse_source(source):
    """Turn a CLI --source string into an int (camera index) or a path/URL string.

    '0' -> 0 (webcam), 'video.mp4' -> path, 'rtsp://...' -> URL.
    """
    if source is None:
        return 0
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text


def open_source(source):
    """Open a camera index, video file or stream URL. Returns a VideoCapture or None."""
    if isinstance(source, int):
        # CAP_DSHOW on Windows avoids a common 300 ms black-frame delay; elsewhere
        # V4L2 is the right backend. Fall back to plain open if either is refused.
        for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    path = str(source)
    if path.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        # OpenCV's FFmpeg backend is extremely verbose about stream reconnects.
        cv2.setLogLevel(3)
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    if not os.path.exists(path):
        print(f"[FCW] Source not found: {path}")
        return None

    cap = cv2.VideoCapture(path)
    return cap if cap.isOpened() else None


def load_image(path):
    """Load a still image as BGR, or None if it cannot be read."""
    if not os.path.exists(path):
        print(f"[FCW] Image not found: {path}")
        return None
    image = cv2.imread(path)
    if image is None:
        print(f"[FCW] Could not decode image: {path}")
    return image


def source_fps(cap, source, default=25.0):
    """Best-effort FPS for a source, with sane fallbacks.

    For video files we need the real FPS because frame timestamps drive the
    closing-speed regression; a wrong value here silently rescales every speed.
    """
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        fps = 0.0

    if fps is None or not (1.0 < fps < 240.0):
        if isinstance(source, int):
            fps = 30.0
        else:
            fps = default
    return fps
