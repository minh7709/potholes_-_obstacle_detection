#!/usr/bin/env python3
"""
Đo chất lượng nhận diện của đúng những gì app đang chạy.

Vấn đề đang tranh luận: "app nhận diện kém hơn lúc làm pipeline upload video".
Hai khác biệt có thể là:
  (a) quantize  -> app chạy best_int8.onnx, pipeline cũ chạy best.pt fp32
  (b) tiền xử lý -> app resize bilinear tự viết, Ultralytics dùng cv2.INTER_LINEAR

Cách đo: chạy app pipeline (bilinear của chính nó) trên N frame 1280x960, so với
best.pt fp32 của Ultralytics làm chuẩn. Báo precision / recall / IoU trung bình.

Chuẩn cũng là một mô hình, nên không phải ground-truth tuyệt đối; nhưng nó là chuẩn
đúng cho câu hỏi "quantize làm hỏng bao nhiêu", vì cả hai cùng họ weights.

Dùng:  ../.venv/bin/python tools/evaluate_models.py [số frame]
"""

import os
import pathlib
import sys
import cv2
import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "FCW-Android/app/src/main/assets/models"
VIDEO = ROOT / "output/annotated_video.mp4"
WEIGHTS = ROOT / "assets/best.pt"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compare_letterbox as cl  # noqa: E402

INPUT = 640
CONF = 0.45
REF_CONF = 0.45
IOU_MATCH = 0.5


def make_frames(n):
    """Lấy n frame ở 1280x960, crop 4:3 ở giữa rồi resize, đúng như khung hình máy."""
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        nw = int(w * 3 / 4)
        x0 = (w - nw) // 2
        crop = frame[:, x0:x0 + nw]
        out.append(cv2.resize(crop, (1280, 960), interpolation=cv2.INTER_LINEAR))
    cap.release()
    return out


def load(path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    raw = eval(s.get_modelmeta().custom_metadata_map["names"])
    names = [raw[i] for i in range(max(raw) + 1)]
    # decode() đọc danh sách lớp từ module-level cache; đặt vào đó để dùng chung.
    cl._names = names
    cl._sess = s
    cl._ort_in = s.get_inputs()[0].name
    return s, s.get_inputs()[0].name, names


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = (np.array([b0[k] for b0 in a]) for k in range(4))
    bx1, by1, bx2, by2 = (np.array([b0[k] for b0 in b]) for k in range(4))
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    aa = ((ax2 - ax1) * (ay2 - ay1))[:, None]
    ba = ((bx2 - bx1) * (by2 - by1))[None]
    return inter / np.maximum(aa + ba - inter, 1e-9)


def evaluate(sess, in_name, names, frames, reference, preproc, conf=CONF):
    tp = fp = fn = 0
    ious = []
    per_class = {}
    for img, ref in zip(frames, reference):
        h, w = img.shape[:2]
        tensor, s, px, py = preproc(img)
        head = sess.run(None, {in_name: tensor[None]})[0][0]
        dets, _, _ = cl.decode(head, s, px, py, w, h, conf=conf)

        mine = [d["box"] + [d["cls"]] for d in dets]
        gold = [d["box"] + [d["cls"]] for d in ref]

        m = iou_matrix(mine, gold)
        used_gold = set()
        used_mine = set()
        pairs = sorted(
            [(m[i, j], i, j) for i in range(len(mine)) for j in range(len(gold)) if m[i, j] >= IOU_MATCH],
            key=lambda t: -t[0],
        )
        for score, i, j in pairs:
            if i in used_mine or j in used_gold:
                continue
            if mine[i][4] != gold[j][4]:
                continue
            used_mine.add(i)
            used_gold.add(j)
            tp += 1
            ious.append(score)
            per_class.setdefault(mine[i][4], [0, 0, 0])
            per_class[mine[i][4]][0] += 1

        for i, d in enumerate(mine):
            if i not in used_mine:
                fp += 1
                per_class.setdefault(d[4], [0, 0, 0])
                per_class[d[4]][1] += 1
        for j, d in enumerate(gold):
            if j not in used_gold:
                fn += 1
                per_class.setdefault(d[4], [0, 0, 0])
                per_class[d[4]][2] += 1

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec,
            "f1": f1, "iou": mean_iou, "per_class": per_class}


def main():
    global REF_CONF, CONF
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    frames = make_frames(n)
    if len(sys.argv) > 2:
        REF_CONF = float(sys.argv[2])
    if len(sys.argv) > 3:
        CONF = float(sys.argv[3])
    print(f"{len(frames)} frame 1280x960 | app conf={CONF} | chuẩn best.pt fp32 conf={REF_CONF}\n")

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    reference = []
    for img in frames:
        tmp = ROOT / "output/_eval_tmp.jpg"
        cv2.imwrite(str(tmp), img)
        res = model.predict(str(tmp), imgsz=INPUT, conf=REF_CONF, iou=0.5, verbose=False)[0]
        rn = res.names
        got = []
        for b in res.boxes:
            got.append({"cls": rn[int(b.cls)],
                        "box": [float(v) for v in b.xyxy[0]]})
        reference.append(got)
    total_ref = sum(len(r) for r in reference)
    print(f"Chuẩn có {total_ref} box trên {len(frames)} frame\n")
    cl.CONF = CONF

    variants = [
        ("bilinear (của app)", cl.kotlin_bilinear_letterbox),
        ("INTER_LINEAR (ultralytics)", cl.ultralytics_letterbox),
    ]
    # fp32 không nằm trong assets: đo cho thấy nó không tốt hơn int8, nên không
    # xuất vào APK. Vẫn giữ bản offline để kiểm chứng lại khi cần.
    models = [("int8  best.onnx", MODELS / "best.onnx"),
              ("fp16  best_fp16.onnx", MODELS / "best_fp16.onnx"),
              ("fp32  best_fp32.onnx", ROOT / "FCW-Android/tools/offline_models/best_fp32.onnx")]

    for pre_name, pre in variants:
        print(f"--- tiền xử lý: {pre_name} ---")
        for label, fname in models:
            path = pathlib.Path(fname)
            if not path.exists():
                print(f"  {label:<22} (thiếu file)")
                continue
            sess, in_name, names = load(path)
            r = evaluate(sess, in_name, names, frames, reference, pre, conf=CONF)
            size = path.stat().st_size / 1048576
            print(f"  {label:<22} P={r['prec']:.3f}  R={r['rec']:.3f}  "
                  f"F1={r['f1']:.3f}  IoU={r['iou']:.3f}  "
                  f"tp={r['tp']} fp={r['fp']} fn={r['fn']}  {size:.1f}MB")
            worst = sorted(r["per_class"].items(), key=lambda kv: kv[1][2], reverse=True)[:3]
            for cname, (t, f, nmiss) in worst:
                if f or nmiss:
                    print(f"        {cname:<16} trúng={t} báo-sai={f} bỏ-sót={nmiss}")
        print()


if __name__ == "__main__":
    main()