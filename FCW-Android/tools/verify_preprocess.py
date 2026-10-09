#!/usr/bin/env python3
"""
Đối chiếu tiền xử lý Kotlin với model gốc, ngoài thiết bị.

Mục đích: tách hai nguyên nhân gây "nhận diện sai" mà không cần điện thoại.

  (A) Preprocessor của app sai  -> sai số pixel vào model
  (B) Quantize int8 làm nhoè    -> model đúng, đầu vào đúng, kết quả vẫn kém

Cách kiểm: dựng lại đúng thứ YoloPreprocessor.kt làm trong Python, chạy
best_int8.onnx với tensor đó, rồi so với best.pt gốc trên cùng ảnh. Nếu hai bên
khớp thì preprocessor đúng và vấn đề nằm ở model/quantization.

Dùng:  ../.venv/bin/python tools/verify_preprocess.py <ảnh> [<ảnh> ...]
"""

import sys
import pathlib
import numpy as np
import cv2
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ONNX = ROOT / "FCW-Android/app/src/main/assets/models/best.onnx"
WEIGHTS = ROOT / "assets/best.pt"

INPUT = 640
CONF = 0.45
IOU = 0.5

# YoloDetector.DEFAULT_WANTED_CLASSES
WANTED = {
    "motorbike", "motorcycle", "car", "bus", "truck", "bicycle", "person",
    "pothole", "reflective_cone", "warning_column", "spherical_roadblock",
    "ashcan", "stop_sign", "tricycle", "fire_hydrant", "dog",
}

ALIASES = {
    "motorcycle": "motorbike", "motor bike": "motorbike",
    "potholes": "pothole", "manhole": "pothole",
    "trash can": "ashcan", "garbage can": "ashcan",
    "cone": "reflective cone", "bollard": "warning column",
    "hydrant": "fire hydrant",
}


def normalise(name):
    base = name.strip().lower().replace("_", " ").replace("-", " ")
    squashed = " ".join(w for w in base.split(" ") if w)
    return ALIASES.get(squashed, squashed)


# --------------------------------------------------------------------------
# Bản port nguyên văn từ YoloPreprocessor.kt
# --------------------------------------------------------------------------

def kotlin_letterbox(bgr):
    """Trả về (tensor NCHW float32, scale, padX, padY)."""
    src_h, src_w = bgr.shape[:2]
    s = min(INPUT / src_w, INPUT / src_h)
    new_w = min(max(int(src_w * s), 1), INPUT)
    new_h = min(max(int(src_h * s), 1), INPUT)
    pad_x = (INPUT - new_w) // 2
    pad_y = (INPUT - new_h) // 2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = np.zeros((3, INPUT, INPUT), dtype=np.float32)

    # nearest neighbour, cùng công thức lấy chỉ số như bản Kotlin
    for y in range(new_h):
        sy = src_h * y // new_h
        for x in range(new_w):
            sx = src_w * x // new_w
            p = rgb[sy, sx]
            tensor[0, y + pad_y, x + pad_x] = p[0] / 255.0
            tensor[1, y + pad_y, x + pad_x] = p[1] / 255.0
            tensor[2, y + pad_y, x + pad_x] = p[2] / 255.0

    tensor[:, :pad_y, :] = 114 / 255.0
    tensor[:, pad_y + new_h:, :] = 114 / 255.0
    tensor[:, :, :pad_x] = 114 / 255.0
    tensor[:, :, pad_x + new_w:] = 114 / 255.0
    return tensor, s, pad_x, pad_y


def to_source(x1, y1, x2, y2, s, pad_x, pad_y, src_w, src_h):
    sx = np.clip((x1 - pad_x) / s, 0, src_w)
    ex = np.clip((x2 - pad_x) / s, 0, src_w)
    sy = np.clip((y1 - pad_y) / s, 0, src_h)
    ey = np.clip((y2 - pad_y) / s, 0, src_h)
    return (min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))


def nms(boxes, scores, classes, iou_thr):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        # class-aware: khác lớp thì không chặn nhau
        same = classes[rest] == classes[i]
        order = rest[~(iou > iou_thr) | ~same]
    return keep


def decode(head, names, scale, pad_x, pad_y, src_w, src_h):
    anchors = head.shape[2]
    n_cls = len(names)
    flat = head.reshape(head.shape[1], anchors)

    cls = flat[4:4 + n_cls, :]
    best_cls = cls.argmax(axis=0)
    best_score = cls.max(axis=0)
    keep_mask = best_score >= CONF
    idx = np.where(keep_mask)[0]
    if idx.size == 0:
        return []

    cx, cy, w, h = flat[0, idx], flat[1, idx], flat[2, idx], flat[3, idx]
    model_boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    src = np.array([to_source(*model_boxes[i], scale, pad_x, pad_y, src_w, src_h)
                    for i in range(idx.size)])

    wanted_ids = {i for i, n in enumerate(names) if normalise(n) in WANTED}
    sel = np.array([best_cls[idx[i]] in wanted_ids for i in range(idx.size)])
    src = src[sel]
    scores = best_score[idx][sel]
    cls_ids = best_cls[idx][sel]

    keep = nms(src, scores, cls_ids, IOU)
    out = []
    for k in keep:
        out.append({
            "cls": names[cls_ids[k]],
            "score": float(scores[k]),
            "box": [float(v) for v in src[k]],
        })
    out.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    return out


def main():
    images = sys.argv[1:] or [str(ROOT / "output/fcw_image.jpg")]
    names = None
    sess = None

    for path in images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"Không đọc được {path}")
            continue
        src_h, src_w = bgr.shape[:2]

        if sess is None:
            sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
            raw_names = eval(sess.get_modelmeta().custom_metadata_map["names"])
            # Metadata là dict {id: name}; cần list theo id, y hệt cách Kotlin dựng mảng.
            names = [raw_names[i] for i in range(max(raw_names) + 1)]
            print(f"Model: {len(names)} lớp — {names}\n")

        tensor, s, px, py = kotlin_letterbox(bgr)
        head = sess.run(None, {sess.get_inputs()[0].name: tensor[None]})[0]
        mine = decode(head, names, s, px, py, src_w, src_h)

        from ultralytics import YOLO
        ref = YOLO(str(WEIGHTS)).predict(
            path, imgsz=INPUT, conf=CONF, iou=IOU, verbose=False
        )[0]
        ref_names = ref.names
        theirs = []
        for b in ref.boxes:
            cls = ref_names[int(b.cls)]
            if normalise(cls) not in WANTED:
                continue
            theirs.append({
                "cls": cls,
                "score": float(b.conf),
                "box": [float(v) for v in b.xyxy[0]],
            })
        theirs.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))

        print(f"=== {path}  {src_w}x{src_h}  scale={s:.4f} pad=({px},{py}) ===")
        print(f"  preprocessor của app : {len(mine)} box")
        for d in mine:
            b = d["box"]
            print(f"     {d['cls']:<16} {d['score']:.3f}  "
                  f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
        print(f"  best.pt gốc          : {len(theirs)} box")
        for d in theirs:
            b = d["box"]
            print(f"     {d['cls']:<16} {d['score']:.3f}  "
                  f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")

        mine_cls = sorted(d["cls"] for d in mine)
        ref_cls = sorted(d["cls"] for d in theirs)
        verdict = "KHỚP" if mine_cls == ref_cls else "LỆCH"
        print(f"  => {verdict}  (app={mine_cls} pt={ref_cls})\n")


if __name__ == "__main__":
    main()