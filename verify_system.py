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
