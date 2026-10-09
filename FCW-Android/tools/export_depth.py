#!/usr/bin/env python3
"""
Xuất Depth Anything V2 Metric ra ONNX cố định kích thước, rồi đo sai số.

Vì sao không dùng bản cộng đồng (77ukhtar): nó cố định 518 và chạy ~840ms trên
CPU điện thoại. Ngân sách của cả pipeline đã là 280ms cho YOLO, nên depth ở 518
là không dùng được. Bản cũng có LayerNormalization + Resize, hai phép toán NNAPI
hay bị từ chối.

Vì sao phải tự export: input động không được (DPT position embedding phụ thuộc
kích thước), nên kích thước phải chọn trước và đo thay vì đoán.

Dùng:  ../.venv/bin/python tools/export_depth.py [kích thước ...]
"""

import pathlib
import sys
import time

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
OUT = ROOT / "FCW-Android/app/src/main/assets/models"

SIZES = [int(a) for a in sys.argv[1:]] or [350]


def main():
    print(f"nạp {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()

    # Một tensor đồ hình. Input tĩnh 3xHxW: DPT dùng position embedding theo kích
    # thước thật, nên dynamic_axes sẽ hỏng lúc export.
    dummy = torch.randn(1, 3, SIZES[0], SIZES[0])

    with torch.no_grad():
        out = model(pixel_values=dummy)
    ref = out.predicted_depth
    print(f"output PyTorch: {tuple(ref.shape)}  "
          f"min={ref.min():.2f} max={ref.max():.2f}")

    OUT.mkdir(parents=True, exist_ok=True)

    for size in SIZES:
        if size != SIZES[0]:
            dummy = torch.randn(1, 3, size, size)
            with torch.no_grad():
                ref = out.predicted_depth if size == SIZES[0] else \
                    model(pixel_values=dummy).predicted_depth

        path = OUT / f"depth_{size}.onnx"
        print(f"\nxuất {size}x{size} -> {path.name}")
        t0 = time.time()
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, str(path),
                input_names=["pixel_values"], output_names=["predicted_depth"],
                opset_version=17, do_constant_folding=True,
            )
        print(f"  fp32 {path.stat().st_size / 1048576:.1f} MB "
              f"({time.time() - t0:.1f}s)")

        # int8: dynamic quantization cho conv. Với depth, dynamic lấy thang theo
        # tensor lúc chạy nên giữ được độ dối hơn static trên một mô hình chưa
        # calibrate cho đúng phân bố ảnh xe.
        from onnxruntime.quantization import QuantType, quantize_dynamic
        qpath = OUT / f"depth_{size}_int8.onnx"
        quantize_dynamic(str(path), str(qpath), weight_type=QuantType.QInt8)
        print(f"  int8 {qpath.stat().st_size / 1048576:.1f} MB")

        # Kiểm chứng ONNX khớp PyTorch, và thời gian chạy CPU.
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"pixel_values": dummy.numpy()})[0]
        if got.shape != ref.shape:
            print(f"  ⚠ shape lệch: onnx {got.shape} vs torch {tuple(ref.shape)}")
        else:
            a = got.reshape(-1).astype(np.float64)
            b = ref.reshape(-1).numpy().astype(np.float64)
            denom = np.abs(b).mean()
            err = np.abs(a - b).mean() / max(denom, 1e-9)
            corr = float(np.corrcoef(a, b)[0, 1])
            print(f"  sai số TB {err * 100:.2f}%  tương quan {corr:.5f}")

        xs = sess.run(None, {"pixel_values": dummy.numpy()})[0]
        t0 = time.time()
        for _ in range(5):
            sess.run(None, {"pixel_values": dummy.numpy()})
        print(f"  CPU trên máy này: {(time.time() - t0) / 5 * 1000:.0f} ms/lần")


if __name__ == "__main__":
    main()