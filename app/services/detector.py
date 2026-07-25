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
