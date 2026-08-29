"""
video_upload_service.py
-----------------------
Manages uploaded video files through a per-session inference pipeline.

Each upload creates an isolated `VideoUploadSession` that:
  - Reads frames from the video file via OpenCV
  - Runs ONNX pothole inference on every frame (decoupled thread)
  - Stores the latest annotated frame for MJPEG streaming
  - Tracks progress (current / total frames) and detection counts
  - Logs detections to DB (reuses the same DB logic as StreamHandler)
  - Cleans up the temp file when done or when explicitly stopped

`VideoUploadManager` is a global singleton that creates/stores/retires sessions.
Old sessions (> SESSION_TTL_SECONDS idle) are swept automatically.
"""

import os
import uuid
import cv2
import time
import queue
import threading
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from app.services.detector import PotholeDetector
from app.database.connection import SessionLocal
from app.database.models import PotholeLog
from app.config import SAVE_IMAGE_DIR, DETECT_COOLDOWN, CONFIDENCE_THRESHOLD, NMS_THRESHOLD

# ──────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────
JPEG_QUALITY        = 80
STREAM_FPS_TARGET   = 25         # max MJPEG output FPS
INFER_QUEUE_SIZE    = 2          # frames buffered between reader and inference
SESSION_TTL_SECONDS = 600        # auto-expire idle sessions after 10 min


# ──────────────────────────────────────────────
# Per-upload session
# ──────────────────────────────────────────────
class VideoUploadSession:
    """Handles one uploaded video file end-to-end."""

    def __init__(self, video_path: str, session_id: str, detector: PotholeDetector):
        self.video_path   = video_path
        self.session_id   = session_id
        self.detector     = detector

        # Progress tracking
        self.total_frames   = 0
        self.current_frame  = 0
        self.detections_count = 0
        self.severity_counts  = {"Low": 0, "Medium": 0, "High": 0}
        self.finished       = False
        self.error: Optional[str] = None

        # Stream state
        self._lock              = threading.Lock()
        self._latest_annotated  = None   # JPEG bytes of the annotated frame
        self._last_stream_ts    = time.monotonic()
        self._stream_fps_val    = 0.0

        self._running       = False
        self._frame_queue   = queue.Queue(maxsize=INFER_QUEUE_SIZE)

        self._reader_thread    = None
        self._inference_thread = None

        # Cooldown for DB logging
        self._last_log_time = 0.0
        # TTL tracking
        self.last_accessed = time.monotonic()

    # ── Lifecycle ──────────────────────────────
    def start(self):
        """Open the video and start reader + inference threads."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error = f"Cannot open video: {self.video_path}"
            self.finished = True
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()

        self._running = True

        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"VidReader-{self.session_id[:8]}"
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name=f"VidInfer-{self.session_id[:8]}"
        )
        self._reader_thread.start()
        self._inference_thread.start()
        print(f"[VideoUpload] Session {self.session_id[:8]} started — {self.total_frames} frames")

    def stop(self):
        """Signal threads to stop and wait briefly."""
        self._running = False
        # Unblock inference thread
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3.0)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=3.0)

    def cleanup(self):
        """Stop threads and delete the temp video file."""
        self.stop()
        if os.path.exists(self.video_path):
            try:
                os.remove(self.video_path)
                print(f"[VideoUpload] Cleaned up {self.video_path}")
            except OSError as e:
                print(f"[VideoUpload] Could not delete {self.video_path}: {e}")

    # ── Reader thread ──────────────────────────
    def _reader_loop(self):
        """Read frames sequentially and push to inference queue."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._running = False
            self.finished = True
            return

        frame_idx = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break  # EOF

            frame_idx += 1
            with self._lock:
                self.current_frame = frame_idx

            # Drop stale queued frame to keep inference fresh
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        cap.release()

        # Send sentinel to inference thread
        try:
            self._frame_queue.put(None, timeout=2.0)
        except queue.Full:
            pass

    # ── Inference thread ────────────────────────
    def _inference_loop(self):
        """Pop frames, run detection, annotate, and store JPEG bytes."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        while True:
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if frame is None:       # sentinel — EOF reached
                break

            boxes, confidences, class_ids = self.detector.detect(frame)
            h, w = frame.shape[:2]
            detections = self._analyze_detections(boxes, confidences, w * h)

            # Update counts
            with self._lock:
                self.detections_count += len(detections)
                for det in detections:
                    sev = det["severity"]
                    if sev in self.severity_counts:
                        self.severity_counts[sev] += 1

            # Log to DB (rate-limited)
            self._log_detections(frame, detections)

            # Annotate frame
            annotated = self._draw_detections(frame, detections)

            # FPS overlay
            now = time.monotonic()
            dt  = now - self._last_stream_ts
            self._last_stream_ts = now
            fps = 1.0 / dt if dt > 0 else 0
            self._stream_fps_val = fps

            # Progress overlay
            with self._lock:
                cf = self.current_frame
                tf = self.total_frames
            pct = (cf / tf * 100) if tf > 0 else 0
            cv2.putText(
                annotated, f"Video {fps:.1f} FPS  |  {cf}/{tf} ({pct:.0f}%)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 2, cv2.LINE_AA
            )

            ret, jpeg = cv2.imencode('.jpg', annotated, encode_params)
            if ret:
                with self._lock:
                    self._latest_annotated = jpeg.tobytes()

        with self._lock:
            self.finished = True
        print(f"[VideoUpload] Session {self.session_id[:8]} finished — "
              f"{self.detections_count} detections")

    # ── Helpers ────────────────────────────────
    def _analyze_detections(self, boxes, confidences, frame_area):
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
                "box": box, "confidence": conf,
                "severity": severity, "color": color,
            })
        return detections

    def _draw_detections(self, frame, detections):
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color    = det["color"]
            conf     = det["confidence"]
            severity = det["severity"]

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            label = f"Pothole {conf:.2f} ({severity})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _log_detections(self, frame, detections):
        if not detections:
            return
        now = time.time()
        if now - self._last_log_time < DETECT_COOLDOWN:
            return

        best = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["box"]
        conf     = best["confidence"]
        severity = best["severity"]

        filename  = f"pothole_vid_{int(now)}_{uuid.uuid4().hex[:6]}.jpg"
        file_path = os.path.join(SAVE_IMAGE_DIR, filename)
        cv2.imwrite(file_path, frame)

        db = SessionLocal()
        try:
            db_log = PotholeLog(
                confidence=conf,
                bbox_x1=int(x1), bbox_y1=int(y1),
                bbox_x2=int(x2), bbox_y2=int(y2),
                severity=severity,
                image_path=f"/static/potholes/{filename}",
            )
            db.add(db_log)
            db.commit()
            self._last_log_time = now
        except Exception as e:
            print(f"[VideoUpload] DB logging error: {e}")
            db.rollback()
        finally:
            db.close()

    # ── MJPEG generator ─────────────────────────
    def get_frame_bytes(self):
        """Generator that yields MJPEG frames for StreamingResponse."""
        self.last_accessed = time.monotonic()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        while True:
            self.last_accessed = time.monotonic()

            with self._lock:
                jpeg_bytes = self._latest_annotated
                finished   = self.finished

            if jpeg_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            elif not finished:
                # Not started yet — send placeholder
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "Processing video...", (140, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
                ret, jpeg = cv2.imencode('.jpg', img, encode_params)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            if finished and jpeg_bytes:
                # Send a final "Done" overlay once
                with self._lock:
                    last = self._latest_annotated
                if last:
                    img = cv2.imdecode(np.frombuffer(last, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        overlay = img.copy()
                        cv2.putText(overlay, "Analysis Complete",
                                    (int(overlay.shape[1] * 0.25), int(overlay.shape[0] * 0.5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 120), 3, cv2.LINE_AA)
                        ret2, jpeg2 = cv2.imencode('.jpg', overlay, encode_params)
                        if ret2:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg2.tobytes() + b'\r\n')
                break

            time.sleep(1.0 / STREAM_FPS_TARGET)

    # ── Progress snapshot ───────────────────────
    def get_progress(self) -> dict:
        with self._lock:
            cf       = self.current_frame
            tf       = self.total_frames
            finished = self.finished
            det_cnt  = self.detections_count
            sev      = dict(self.severity_counts)
            err      = self.error

        pct = round(cf / tf * 100, 1) if tf > 0 else (100.0 if finished else 0.0)
        return {
            "session_id":       self.session_id,
            "current_frame":    cf,
            "total_frames":     tf,
            "percent":          pct,
            "detections_count": det_cnt,
            "severity":         sev,
            "finished":         finished,
            "error":            err,
        }


# ──────────────────────────────────────────────
# Global session manager (singleton)
# ──────────────────────────────────────────────
class VideoUploadManager:
    """Thread-safe registry of active upload sessions."""

    def __init__(self):
        self._sessions: dict[str, VideoUploadSession] = {}
        self._lock     = threading.Lock()
        self._detector = PotholeDetector()   # shared across sessions

        # Background sweep thread
        self._sweep_thread = threading.Thread(
            target=self._sweep_loop, daemon=True, name="SessionSweep"
        )
        self._sweep_thread.start()

    def create_session(self, video_path: str) -> str:
        session_id = uuid.uuid4().hex
        session    = VideoUploadSession(video_path, session_id, self._detector)
        session.start()
        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[VideoUploadSession]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session:
            session.last_accessed = time.monotonic()
        return session

    def delete_session(self, session_id: str):
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.cleanup()

    def _sweep_loop(self):
        """Periodically remove sessions that have been idle or finished+old."""
        while True:
            time.sleep(60)
            now = time.monotonic()
            to_remove = []
            with self._lock:
                for sid, session in self._sessions.items():
                    idle = now - session.last_accessed
                    if idle > SESSION_TTL_SECONDS:
                        to_remove.append(sid)
            for sid in to_remove:
                print(f"[VideoUpload] Auto-expiring session {sid[:8]}")
                self.delete_session(sid)


# Module-level singleton (imported by routes.py)
upload_manager = VideoUploadManager()
