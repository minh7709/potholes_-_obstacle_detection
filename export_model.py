import os
import sys

def main():
    print("Converting best.pt to ONNX format...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found. Make sure the virtual environment is activated and requirements are installed.")
        sys.exit(1)

    model_path = os.path.join("assets", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Load YOLO model
    print(f"Loading PyTorch model from {model_path}...")
    model = YOLO(model_path)
    
    # Export to ONNX
    print("Exporting model to ONNX...")
    onnx_path = model.export(format="onnx", imgsz=640)
    print(f"Model exported successfully to: {onnx_path}")

if __name__ == "__main__":
    main()
