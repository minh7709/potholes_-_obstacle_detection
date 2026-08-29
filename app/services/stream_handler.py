import os
import sys
import cv2
import time
import uuid
import re
import queue
import numpy as np
from datetime import datetime, timezone
import threading
from app.config import VIDEO_SOURCE, SAVE_IMAGE_DIR, DETECT_COOLDOWN
from app.services.detector import PotholeDetector
from app.database.connection import SessionLocal
from app.database.models import PotholeLog

_IS_WINDOWS = sys.platform.startswith("win")

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

        # Auto-append /video for bare IP Webcam URLs that have no path.
        # IP Webcam app serves MJPEG stream at /video, not at root /.
        # e.g. "http://192.168.1.3:8080" → "http://192.168.1.3:8080/video"
        if re.match(r'^https?://[^/]+$', cleaned) or re.match(r'^https?://[^/]+/$', cleaned):
            cleaned = cleaned.rstrip('/') + '/video'
            print(f"[clean_video_source] Auto-appended /video → {cleaned}")

        return cleaned
    return source


class StreamHandler:
    # Frames waiting in inference queue; drop stale ones when full
    _INFER_QUEUE_SIZE = 2
    # Limit capture rate (fps) so inference thread isn't flooded
    _CAPTURE_TARGET_FPS = 15
    # Force webcam to capture at this resolution to lower CPU load
    _CAPTURE_WIDTH = 640
    _CAPTURE_HEIGHT = 480
    # JPEG quality for MJPEG output (lower = faster, less bandwidth)
    _JPEG_QUALITY = 75
    # Max seconds to wait for a thread to stop before warning
    _THREAD_JOIN_TIMEOUT = 5.0

    def __init__(self, source=VIDEO_SOURCE, detector=None):
        self.source = clean_video_source(source)
        self.detector = detector or PotholeDetector()
        self.cap = None
        self.running = False
        self.last_detection_time = 0

        self.lock = threading.Lock()

        # Latest raw frame from camera (always fresh, no inference delay)
        self.latest_raw_frame = None
        # Latest analyzed detections [{box, confidence, severity, color}, ...]
        self.latest_detections = []
        # Timestamp of last frame written (for FPS display)
        self._last_stream_ts = time.monotonic()
        self._stream_fps = 0.0

        # Queue connecting capture thread → inference thread
        self._frame_queue = queue.Queue(maxsize=self._INFER_QUEUE_SIZE)

        self._capture_thread = None
        self._inference_thread = None

    # ------------------------------------------------------------------
    def _create_capture(self):
        """Creates a cv2.VideoCapture with optimal backend."""
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_V4L2
            cap = cv2.VideoCapture(self.source, backend)
            # Lower resolution & FPS so laptop CPU isn't overwhelmed
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, self._CAPTURE_TARGET_FPS)
            return cap
        elif isinstance(self.source, str) and self.source.startswith(
            ("http://", "https://", "rtsp://", "rtmp://")
        ):
            # Suppress verbose FFmpeg log spam ("Stream ends prematurely", etc.)
            cv2.setLogLevel(3)   # 3 = ERROR only (suppress WARNING/INFO)
            # Use FFMPEG; set buffer to 1 to always get the newest frame
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        else:
            return cv2.VideoCapture(self.source)

    # ------------------------------------------------------------------
    def _is_http_stream(self):
        """True when source is an HTTP/HTTPS URL (IP Webcam MJPEG stream)."""
        return isinstance(self.source, str) and self.source.startswith(("http://", "https://"))

    def start(self):
        """Starts the capture thread and inference thread."""
        with self.lock:
            if self.running:
                return
            self.running = True

        # Drain any leftover frames from a previous run
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        if self._is_http_stream():
            # Direct MJPEG reading via requests — no FFMPEG buffer, minimal latency
            self.cap = None   # not used for HTTP streams
            capture_target = self._capture_loop_mjpeg
            print(f"StreamHandler: using direct MJPEG reader for {self.source}")
        else:
            # Webcam / RTSP / file — use OpenCV VideoCapture
            self.cap = self._create_capture()
            if not self.cap.isOpened():
                print(f"Warning: Could not open video source '{self.source}'. "
                      "Will retry automatically.")
            capture_target = self._capture_loop

        self._capture_thread = threading.Thread(
            target=capture_target, daemon=True, name="CaptureThread"
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="InferenceThread"
        )
        self._capture_thread.start()
        self._inference_thread.start()
        print(f"StreamHandler started — source: {self.source}")

    def stop(self):
        """Stops both threads gracefully, then releases the camera."""
        with self.lock:
            self.running = False

        # Unblock inference thread that may be waiting on queue.get()
        try:
            self._frame_queue.put_nowait(None)   # sentinel
        except queue.Full:
            pass

        # Wait for threads to finish BEFORE releasing camera (fixes race condition)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=self._THREAD_JOIN_TIMEOUT)
            if self._capture_thread.is_alive():
                print("Warning: CaptureThread did not stop in time.")

        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=self._THREAD_JOIN_TIMEOUT)
            if self._inference_thread.is_alive():
                print("Warning: InferenceThread did not stop in time.")

        if self.cap:
            self.cap.release()
            self.cap = None
        # (MJPEG streams use requests — no cap to release)

        with self.lock:
            self.latest_raw_frame = None
            self.latest_detections = []

        print("StreamHandler stopped.")

    # ------------------------------------------------------------------
    # Thread 1a: Capture via OpenCV — for webcam / RTSP / local file
    # ------------------------------------------------------------------
    def _capture_loop(self):
        """Reads frames from camera and pushes to the inference queue."""
        frame_interval = 1.0 / self._CAPTURE_TARGET_FPS
        reconnect_delay = 1.0   # exponential backoff starting value

        while True:
            with self.lock:
                if not self.running:
                    break

            # Auto-reconnect if camera becomes unavailable
            if self.cap is None or not self.cap.isOpened():
                print(f"Reconnecting to source '{self.source}' (retry in {reconnect_delay:.0f}s)...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 16.0)  # cap at 16s
                self.cap = self._create_capture()
                continue

            t_start = time.monotonic()
            ret, frame = self.cap.read()

            if not ret:
                if isinstance(self.source, str):
                    # IP stream disconnected — reconnect with backoff
                    print(f"IP stream read failed, reconnecting in {reconnect_delay:.0f}s...")
                    self.cap.release()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 16.0)
                    self.cap = self._create_capture()
                else:
                    time.sleep(0.03)
                continue

            # Successful read — reset backoff
            reconnect_delay = 1.0

            # Store raw frame for immediate streaming (no inference lag)
            with self.lock:
                self.latest_raw_frame = frame

            # Push frame to inference queue (drop stale if busy)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # inference too slow, skip this frame

            # Pace capture to target FPS
            elapsed = time.monotonic() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Thread 1b: Capture via requests — for HTTP/HTTPS IP Webcam MJPEG
    # Bypasses FFMPEG buffer completely for near-zero latency
    # ------------------------------------------------------------------
    def _capture_loop_mjpeg(self):
        """Reads MJPEG stream directly via requests, no FFMPEG buffering.

        IP Webcam app streams multipart JPEG over HTTP. We parse boundaries
        ourselves, decode each JPEG with cv2.imdecode, and store the latest
        frame immediately — no internal buffering, minimal latency.
        """
        import requests as req
        reconnect_delay = 1.0

        while True:
            with self.lock:
                if not self.running:
                    break

            try:
                print(f"[MJPEG] Connecting to {self.source}...")
                response = req.get(
                    self.source,
                    stream=True,
                    timeout=(5, 10),   # (connect timeout, read timeout)
                    headers={"Connection": "keep-alive"}
                )
                response.raise_for_status()
                print(f"[MJPEG] Connected. Content-Type: {response.headers.get('Content-Type', '?')}")
                reconnect_delay = 1.0   # reset backoff on successful connect

                buf = b''
                for chunk in response.iter_content(chunk_size=8192):
                    with self.lock:
                        if not self.running:
                            break

                    buf += chunk

                    # Parse all complete JPEG frames in current buffer
                    while True:
                        jpg_start = buf.find(b'\xff\xd8')
                        jpg_end   = buf.find(b'\xff\xd9', jpg_start + 2 if jpg_start >= 0 else 0)

                        if jpg_start == -1 or jpg_end == -1:
                            # Keep last part — may contain partial JPEG
                            if len(buf) > 131072:   # 128 KB safety cap
                                buf = buf[-8192:]
                            break

                        jpg_bytes = buf[jpg_start: jpg_end + 2]
                        buf = buf[jpg_end + 2:]     # advance past this JPEG

                        frame = cv2.imdecode(
                            np.frombuffer(jpg_bytes, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        if frame is None:
                            continue

                        # Resize to inference resolution to reduce CPU load
                        if frame.shape[1] > self._CAPTURE_WIDTH or frame.shape[0] > self._CAPTURE_HEIGHT:
                            frame = cv2.resize(
                                frame,
                                (self._CAPTURE_WIDTH, self._CAPTURE_HEIGHT),
                                interpolation=cv2.INTER_LINEAR
                            )

                        # Store raw frame immediately — stream generator picks it up
                        with self.lock:
                            self.latest_raw_frame = frame

                        # Push to inference queue, drop stale frame if busy
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            self._frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass

            except Exception as e:
                with self.lock:
                    if not self.running:
                        break
                print(f"[MJPEG] Stream error: {e}. Reconnecting in {reconnect_delay:.0f}s...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 16.0)

    # ------------------------------------------------------------------
    # Thread 2: Inference — pops frames, runs ONNX, stores detections only
    # ------------------------------------------------------------------
    def _inference_loop(self):
        """Pops frames from queue, runs detection, stores results.
        Decoupled from streaming: detections are cached and overlaid on
        the latest raw frame by get_latest_frame_bytes().
        """
        while True:
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                with self.lock:
                    if not self.running:
                        break
                continue

            if frame is None:   # sentinel — time to stop
                break

            boxes, confidences, class_ids = self.detector.detect(frame)
            h, w = frame.shape[:2]
            detections = self._analyze_detections(boxes, confidences, w * h)

            # Log to DB if new detections and cooldown passed
            self._log_detections(frame, detections)

            # Store detections; stream generator overlays them on fresh raw frames
            with self.lock:
                self.latest_detections = detections

    # ------------------------------------------------------------------
    def _analyze_detections(self, boxes, confidences, frame_area):
        """Convert raw boxes+confs into a structured detection list with severity."""
        detections = []
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / frame_area if frame_area > 0 else 0

            if ratio > 0.04:
                severity, color = "High",   (0, 0, 255)
            elif ratio > 0.012:
                severity, color = "Medium", (0, 165, 255)
            else:
                severity, color = "Low",    (0, 255, 0)

            detections.append({
                "box": box,
                "confidence": conf,
                "severity": severity,
                "color": color,
            })
        return detections

    def _draw_detections(self, frame, detections):
        """Overlay detection bounding boxes onto a frame (in-place copy)."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = det["color"]
            conf  = det["confidence"]
            severity = det["severity"]

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            label = f"Pothole {conf:.2f} ({severity})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _log_detections(self, frame, detections):
        """Save snapshot and write DB record when new potholes are found."""
        if not detections:
            return
        current_time = time.time()
        if current_time - self.last_detection_time < DETECT_COOLDOWN:
            return

        best = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["box"]
        conf     = best["confidence"]
        severity = best["severity"]

        filename  = f"pothole_{int(current_time)}_{uuid.uuid4().hex[:6]}.jpg"
        file_path = os.path.join(SAVE_IMAGE_DIR, filename)
        cv2.imwrite(file_path, frame)

        db = SessionLocal()
        try:
            db_log = PotholeLog(
                confidence=conf,
                bbox_x1=int(x1), bbox_y1=int(y1),
                bbox_x2=int(x2), bbox_y2=int(y2),
                severity=severity,
                image_path=f"/static/potholes/{filename}"
            )
            db.add(db_log)
            db.commit()
            self.last_detection_time = current_time
            print(f"Logged detection: {severity} | conf={conf:.2f}")
        except Exception as e:
            print(f"DB logging error: {e}")
            db.rollback()
        finally:
            db.close()

    # ------------------------------------------------------------------
    # MJPEG stream generator
    # ------------------------------------------------------------------
    def get_latest_frame_bytes(self):
        """Yields MJPEG frame bytes.

        Key design: grabs the latest RAW frame (always fresh from camera),
        overlays the cached detections from the last inference run, then
        encodes. This means video is smooth at camera FPS regardless of
        how slow ONNX inference is.
        """
        if not self.running:
            self.start()

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._JPEG_QUALITY]
        placeholder_sent = False
        prev_ts = time.monotonic()

        while True:
            with self.lock:
                raw   = self.latest_raw_frame
                dets  = list(self.latest_detections)   # shallow copy, safe outside lock

            if raw is None:
                if not placeholder_sent:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        img, "Camera Starting / Offline", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA
                    )
                    ret, jpeg = cv2.imencode('.jpg', img, encode_params)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                        placeholder_sent = True
                time.sleep(0.05)
                continue

            placeholder_sent = False

            # Overlay detections on the freshest raw frame
            display = self._draw_detections(raw, dets)

            # FPS counter overlay (top-left corner)
            now = time.monotonic()
            dt  = now - prev_ts
            prev_ts = now
            fps = 1.0 / dt if dt > 0 else 0
            cv2.putText(
                display, f"Stream {fps:.1f} FPS", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 120), 2, cv2.LINE_AA
            )

            ret, jpeg = cv2.imencode('.jpg', display, encode_params)
            if not ret:
                time.sleep(0.03)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            # Aim for ~25 FPS stream output
            time.sleep(0.04)
