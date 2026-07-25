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
