import os
import json
import cv2
import numpy as np
import onnxruntime as ort
from app.config import ONNX_MODEL_PATH, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, FRAMES_DIR

class PotholeDetector:
    def __init__(self, model_path=ONNX_MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.class_names = {0: "pothole"}
        
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"ONNX model file not found at: {self.model_path}")
            return
            
        print(f"Loading ONNX Model from {self.model_path}...")
        try:
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            meta = self.session.get_modelmeta().custom_metadata_map
            if 'names' in meta:
                try:
                    import ast
                    self.class_names = ast.literal_eval(meta['names'])
                    self.class_names = {int(k): v for k, v in self.class_names.items()}
                except Exception:
                    pass
            print(f"ONNX Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load ONNX session: {e}")

    def preprocess(self, frame):
        """Preprocesses the image for YOLOv8 model inference with letterbox padding."""
        h, w = frame.shape[:2]
        net_h, net_w = 640, 640
        
        r = min(net_w / w, net_h / h)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = net_w - new_unpad[0], net_h - new_unpad[1]
        
        dw /= 2
        dh /= 2
        
        if (w, h) != new_unpad:
            im = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            im = frame.copy()
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        im = im.astype(np.float32) / 255.0
        
        return im, r, (dw, dh)

    def detect(self, frame):
        """
        Runs object detection on the input frame.
        Returns:
            final_boxes: [[x1, y1, x2, y2], ...]
            final_confidences: [conf1, ...]
            final_class_ids: [id1, ...]
        """
        if self.session is None:
            self._load_model()
            if self.session is None:
                return [], [], []
                
        h, w = frame.shape[:2]
        blob, ratio, (dw, dh) = self.preprocess(frame)
        
        try:
            outputs = self.session.run(self.output_names, {self.input_name: blob})
            output = outputs[0]  # Shape: [1, 4 + num_classes, 8400]
            output = np.squeeze(output)
            output = output.T
            
            boxes = []
            confidences = []
            class_ids = []
            
            for row in output:
                scores = row[4:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                
                if confidence >= self.conf_threshold:
                    cx, cy, bw, bh = row[:4]
                    
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    
                    x1 = int((x1 - dw) / ratio)
                    y1 = int((y1 - dh) / ratio)
                    x2 = int((x2 - dw) / ratio)
                    y2 = int((y2 - dh) / ratio)
                    
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    if box_w > 0 and box_h > 0:
                        boxes.append([x1, y1, box_w, box_h])
                        confidences.append(confidence)
                        class_ids.append(int(class_id))
                        
            if not boxes:
                return [], [], []
                
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
            print(f"Error during detection: {e}")
            return [], [], []

    def analyze_detections(self, boxes, confidences, frame_area):
        """Enriches raw boxes and confidences with severity and colors."""
        detections = []
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / frame_area if frame_area > 0 else 0

            if ratio > 0.04:
                severity, color = "High", (0, 0, 255)       # Red
            elif ratio > 0.012:
                severity, color = "Medium", (0, 165, 255)   # Orange
            else:
                severity, color = "Low", (0, 255, 0)        # Green

            detections.append({
                "box": box,
                "confidence": round(float(conf), 4),
                "severity": severity,
                "color": color
            })
        return detections

    def draw_detections(self, frame, detections):
        """Draws bounding boxes, labels, and confidence tags on the frame."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = det["color"]
            conf = det["confidence"]
            severity = det["severity"]
            conf_pct = int(conf * 100)

            # Draw outer rectangle
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            # Label banner
            label = f"Pothole {conf_pct}% ({severity})"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(out, label, (x1 + 4, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def process_video(self, video_path: str, video_id: int):
        """
        Processes an uploaded video file frame-by-frame:
        - Detects potholes in each frame
        - Saves detected frames as annotated images in app/static/frames/
        - Returns full video telemetry & list of detected frame records.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        detected_frames_data = []
        total_potholes_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            h, w = frame.shape[:2]
            boxes, confidences, class_ids = self.detect(frame)

            if boxes:
                detections = self.analyze_detections(boxes, confidences, w * h)
                total_potholes_count += len(detections)

                # Draw bounding boxes on frame
                annotated = self.draw_detections(frame, detections)

                # Save image to frames directory
                filename = f"video_{video_id}_frame_{frame_idx:06d}.jpg"
                save_path = os.path.join(FRAMES_DIR, filename)
                cv2.imwrite(save_path, annotated, encode_params)

                # Relative public URL
                public_image_url = f"/static/frames/{filename}"

                # Calculate stats for this frame
                conf_list = [d["confidence"] for d in detections]
                max_conf = max(conf_list)
                
                # Determine overall severity for frame
                if any(d["severity"] == "High" for d in detections):
                    frame_sev = "High"
                elif any(d["severity"] == "Medium" for d in detections):
                    frame_sev = "Medium"
                else:
                    frame_sev = "Low"

                timestamp_sec = frame_idx / fps if fps > 0 else 0.0

                detected_frames_data.append({
                    "video_id": video_id,
                    "frame_number": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "image_path": public_image_url,
                    "pothole_count": len(detections),
                    "max_confidence": max_conf,
                    "severity": frame_sev,
                    "confidences_json": json.dumps(conf_list),
                    "detections_json": json.dumps([
                        {
                            "box": d["box"],
                            "confidence": d["confidence"],
                            "severity": d["severity"]
                        }
                        for d in detections
                    ])
                })

        cap.release()

        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": duration_sec,
            "total_potholes_count": total_potholes_count,
            "detected_frames": detected_frames_data
        }

# Global detector instance
detector = PotholeDetector()
