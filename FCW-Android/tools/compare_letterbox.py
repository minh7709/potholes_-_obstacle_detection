#!/usr/bin/env python3
"""
Tách hai nguyên nhân của "nhận diện sai" bằng cách đo, không đoán.

Ba cách chạy cùng một ảnh:

  A  int8 ONNX + letterbox Ultralytics (bilinear)   -> chuẩn của nhà huấn luyện
  B  int8 ONNX + letterbox của app (nearest)       -> đúng thứ Kotlin làm
  C  best.pt PyTorch + letterbox Ultralytics        -> model gốc, chưa quantize

  A vs B  = ảnh hưởng của preprocessor (bilinear so với nearest)
  C vs A  = ảnh hưởng của quantize int8

Chỉ khi biết phần nào lớn mới biết nên sửa gì. Nếu B khớp C thì Kotlin đúng và
vấn đề nằm ở quantize. Nếu B lệch A nhiều thì preprocessor cần đổi.

Dùng:  ../.venv/bin/python tools/compare_letterbox.py <ảnh> [...]
"""

import os
import sys
import pathlib
import numpy as np
import cv2
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ONNX = pathlib.Path(os.environ.get("FCW_ONNX", str(
    ROOT / "FCW-Android/app/src/main/assets/models/best.onnx")))
WEIGHTS = ROOT / "assets/best.pt"

INPUT = 640
CONF = 0.45
IOU = 0.5

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

_names = None
_sess = None
_ort_in = None


def normalise(name):
    base = name.strip().lower().replace("_", " ").replace("-", " ")
    squashed = " ".join(w for w in base.split(" ") if w)
    return ALIASES.get(squashed, squashed)


def lazy_init():
    global _names, _sess, _ort_in
    if _sess is not None:
        return
    _sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    _ort_in = _sess.get_inputs()[0].name
    raw = eval(_sess.get_modelmeta().custom_metadata_map["names"])
    _names = [raw[i] for i in range(max(raw) + 1)]


def geometry(src_w, src_h):
    s = min(INPUT / src_w, INPUT / src_h)
    new_w = min(max(int(src_w * s), 1), INPUT)
    new_h = min(max(int(src_h * s), 1), INPUT)
    return s, new_w, new_h, (INPUT - new_w) // 2, (INPUT - new_h) // 2


def ultralytics_letterbox(bgr):
    """Giống LetterBox của Ultralytics: resize bilinear rồi đệm 114, canh giữa."""
    src_h, src_w = bgr.shape[:2]
    s, new_w, new_h, pad_x, pad_y = geometry(src_w, src_h)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT, INPUT, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32).transpose(2, 0, 1) / 255.0, s, pad_x, pad_y


def kotlin_letterbox(bgr):
    """Port nguyên văn YoloPreprocessor.kt: nearest neighbour, đệm 114, NCHW."""
    src_h, src_w = bgr.shape[:2]
    s, new_w, new_h, pad_x, pad_y = geometry(src_w, src_h)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = np.zeros((3, INPUT, INPUT), dtype=np.float32)
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


def kotlin_bilinear_letterbox(bgr):
    """Port nguyên văn phần bilinear mới của YoloPreprocessor.kt."""
    src_h, src_w = bgr.shape[:2]
    s, new_w, new_h, pad_x, pad_y = geometry(src_w, src_h)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.zeros((3, INPUT, INPUT), dtype=np.float32)

    step_x = src_w / new_w
    step_y = src_h / new_h
    max_y, max_x = src_h - 1, src_w - 1

    for y in range(new_h):
        sy = min(max((y + 0.5) * step_y - 0.5, 0.0), max_y)
        y0 = int(sy)
        y1 = min(y0 + 1, max_y)
        wy = sy - y0
        wy1 = 1.0 - wy
        for x in range(new_w):
            sx = min(max((x + 0.5) * step_x - 0.5, 0.0), max_x)
            x0 = int(sx)
            x1 = min(x0 + 1, max_x)
            wx = sx - x0
            wx1 = 1.0 - wx
            p00 = rgb[y0, x0]
            p01 = rgb[y0, x1]
            p10 = rgb[y1, x0]
            p11 = rgb[y1, x1]
            top = p00 * wx1 + p01 * wx
            bottom = p10 * wx1 + p11 * wx
            val = (top * wy1 + bottom * wy) / 255.0
            col = x + pad_x
            row = y + pad_y
            tensor[0, row, col] = val[0]
            tensor[1, row, col] = val[1]
            tensor[2, row, col] = val[2]

    tensor[:, :pad_y, :] = 114 / 255.0
    tensor[:, pad_y + new_h:, :] = 114 / 255.0
    tensor[:, :, :pad_x] = 114 / 255.0
    tensor[:, :, pad_x + new_w:] = 114 / 255.0
    return tensor, s, pad_x, pad_y


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
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(a_i + a_r - inter, 1e-9)
        order = rest[~((iou > iou_thr) & (classes[rest] == classes[i]))]
    return keep


def decode(head, s, pad_x, pad_y, src_w, src_h, conf=CONF):
    names = _names
    flat = head.reshape(head.shape[0], head.shape[1])
    anchors = flat.shape[1]
    n_cls = len(names)

    cls = flat[4:4 + n_cls, :]
    best_cls = cls.argmax(axis=0)
    best_score = cls.max(axis=0)
    idx = np.where(best_score >= conf)[0]
    if idx.size == 0:
        return [], best_score, best_cls

    cx, cy, w, h = flat[0, idx], flat[1, idx], flat[2, idx], flat[3, idx]
    mb = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    src = np.empty_like(mb)
    for i in range(idx.size):
        x1, y1, x2, y2 = mb[i]
        src[i, 0] = np.clip((x1 - pad_x) / s, 0, src_w)
        src[i, 2] = np.clip((x2 - pad_x) / s, 0, src_w)
        src[i, 1] = np.clip((y1 - pad_y) / s, 0, src_h)
        src[i, 3] = np.clip((y2 - pad_y) / s, 0, src_h)
    src = np.stack([np.minimum(src[:, 0], src[:, 2]), np.minimum(src[:, 1], src[:, 3]),
                    np.maximum(src[:, 0], src[:, 2]), np.maximum(src[:, 1], src[:, 3])], axis=1)

    wanted_ids = {i for i, n in enumerate(names) if normalise(n) in WANTED}
    sel = np.array([best_cls[idx[i]] in wanted_ids for i in range(idx.size)])
    src, scores, ids = src[sel], best_score[idx][sel], best_cls[idx][sel]

    out = []
    for k in nms(src, scores, ids, IOU):
        out.append({"cls": names[ids[k]], "score": float(scores[k]),
                    "box": [float(v) for v in src[k]]})
    out.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    return out, best_score, best_cls


def run(path, verbose=True):
    lazy_init()
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    src_h, src_w = bgr.shape[:2]

    tu, su, pxu, pyu = ultralytics_letterbox(bgr)
    head_u = _sess.run(None, {_ort_in: tu[None]})[0][0]  # [1,19,8400] -> [19,8400]
    a, score_u, cls_u = decode(head_u, su, pxu, pyu, src_w, src_h)

    tk, sk, pxk, pyk = kotlin_bilinear_letterbox(bgr)
    head_k = _sess.run(None, {_ort_in: tk[None]})[0][0]
    b, score_k, cls_k = decode(head_k, sk, pxk, pyk, src_w, src_h)

    from ultralytics import YOLO
    ref = YOLO(str(WEIGHTS)).predict(path, imgsz=INPUT, conf=CONF, iou=IOU, verbose=False)[0]
    rn = ref.names
    c = []
    for bx in ref.boxes:
        cn = rn[int(bx.cls)]
        if normalise(cn) in WANTED:
            c.append({"cls": cn, "score": float(bx.conf), "box": [float(v) for v in bx.xyxy[0]]})
    c.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))

    if verbose:
        print(f"=== {pathlib.Path(path).name}  {src_w}x{src_h} ===")
        for tag, res in (("A int8+ultralytics-LB", a), ("B letterbox-của-app (bilinear)", b),
                         ("C best.pt fp32", c)):
            print(f"  {tag:<28} {len(res)} box: "
                  + ", ".join(f"{d['cls']} {d['score']:.2f}" for d in res))

    # Box nào C thấy mà B bỏ sót -> so điểm để biết vì sao mất
    lost = []
    for d in c:
        d_box = np.array(d["box"])
        hit = False
        for e in b:
            e_box = np.array(e["box"])
            xx1 = max(d_box[0], e_box[0]); yy1 = max(d_box[1], e_box[1])
            xx2 = min(d_box[2], e_box[2]); yy2 = min(d_box[3], e_box[3])
            inter = max(xx2 - xx1, 0) * max(yy2 - yy1, 0)
            ai = np.prod(d_box[2:] - d_box[:2]); ae = np.prod(e_box[2:] - e_box[:2])
            if inter / max(ai + ae - inter, 1e-9) > 0.5:
                hit = True
                break
        if not hit:
            cx = (d_box[0] + d_box[2]) / 2 / INPUT
            lost.append((d["cls"], d["score"]))
    if verbose and lost:
        print(f"  B bỏ sót so với C: {lost}")
    return {"A": a, "B": b, "C": c, "lost": lost}


def main():
    images = sys.argv[1:] or [str(ROOT / "output/fcw_image.jpg")]
    totals = {"A": 0, "B": 0, "C": 0, "lost": 0}
    for p in images:
        r = run(p)
        if r:
            totals["A"] += len(r["A"]); totals["B"] += len(r["B"])
            totals["C"] += len(r["C"]); totals["lost"] += len(r["lost"])
    print(f"\nTỔNG  A={totals['A']}  B={totals['B']}  C={totals['C']}  "
          f"B-bỏ-sót={totals['lost']}")


if __name__ == "__main__":
    main()