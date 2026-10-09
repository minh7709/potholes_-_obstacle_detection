#!/usr/bin/env python3
"""
Chạy ĐÚNG logic của app Android trên một file video, để biết cảnh báo lên lúc nào.

Không phải chạy lại pipeline Python cũ. Ở đây mô phỏng từng khâu của app:
  - tiền xử lý bilinear tự viết (giống YoloPreprocessor.kt)
  - cắt vùng nhìn thấy về 4:3 trước khi letterbox (modelFittedRect)
  - best_int8.onnx — model app thật sự dùng
  - depth_224_int8.onnx — model depth app thật sự dùng
  - ByteTracker 2 giai đoạn, TrackHistory, RiskEngine: port nguyên văn từ Kotlin

Chỗ phải nói thẳng: ego-motion dùng cv2.calcOpticalFlowPyrLK thay cho bản
Lucas-Kanade tự viết trong Kotlin (Android không có OpenCV). Hai thuật toán
cùng ý tưởng nhưng không giống hệt nhau.

Dùng:  ../.venv/bin/python tools/run_android_logic.py <video> [--hz 3.5]
"""

import argparse
import collections
import pathlib
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "FCW-Android/app/src/main/assets/models"

INPUT = 640
CONF = 0.45
IOU_NMS = 0.5
TRAINING_ASPECT = 4 / 3

# Ngưỡng lấy từ FcwConfig.kt, tương ứng config.yaml
WARN_TTC, DANGER_TTC = 4.0, 2.0
WARN_DIST, DANGER_DIST, MIN_DIST = 8.0, 3.5, 0.5
CONFIRM_FRAMES, MIN_HITS = 3, 3
ROI = (0.20, 0.30, 0.80, 0.97)
CORRIDOR = (0.33, 0.32, 0.67, 0.97)
PRIMARY = {"motorbike", "motorcycle"}
DEPTH_INTERVAL = 5
DEPTH_CENTER_CROP, DEPTH_MIN_SAMPLES = 0.5, 16
DEPTH_OUTLIER_TOL, DEPTH_VALID_MIN, DEPTH_VALID_MAX = 0.25, 1.0, 120.0
DEPTH_SCALE = 0.602   # cùng hệ số với FcwConfig.depthScale5
HIST_LEN, DIST_EMA = 20, 0.3
MIN_SPEED_SAMPLES, MIN_WINDOW_S, MAX_SPEED, SPEED_EMA = 4, 0.6, 25.0, 0.5
EGO_BG_DEPTH, EGO_MIN_ROW, EGO_SMOOTH = 15.0, 0.12, 0.6
EGO_MAX_ACCEL, EGO_MAX_SPEED = 8.0, 33.0
TRACK_BUFFER = 30

ALIASES = {"motorcycle": "motorbike", "motor bike": "motorbike",
           "potholes": "pothole", "manhole": "pothole",
           "trash can": "ashcan", "garbage can": "ashcan",
           "cone": "reflective cone", "bollard": "warning column",
           "hydrant": "fire hydrant"}
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalise(n):
    s = " ".join(w for w in n.strip().lower().replace("_", " ").replace("-", " ").split() if w)
    return ALIASES.get(s, s)


# ── Tiền xử lý (port YoloPreprocessor.kt) ──────────────────────────────────

def kotlin_bilinear_letterbox(rgb, crop_rect):
    """Port YoloPreprocessor.fill với rect con. rgb uint8 HxWx3."""
    cx0, cy0, cw, ch = crop_rect
    s = min(INPUT / cw, INPUT / ch)
    nw = min(max(int(cw * s), 1), INPUT)
    nh = min(max(int(ch * s), 1), INPUT)
    px, py = (INPUT - nw) // 2, (INPUT - nh) // 2

    step_x, step_y = cw / nw, ch / nh
    max_x, max_y = cw - 1, ch - 1
    t = np.zeros((3, INPUT, INPUT), np.float32)
    for y in range(nh):
        sy = min(max((y + 0.5) * step_y - 0.5, 0.0), float(max_y))
        y0 = int(sy); y1 = min(y0 + 1, max_y); wy = sy - y0
        for x in range(nw):
            sx = min(max((x + 0.5) * step_x - 0.5, 0.0), float(max_x))
            x0 = int(sx); x1 = min(x0 + 1, max_x); wx = sx - x0
            p = rgb[cy0 + y0, cx0 + x0], rgb[cy0 + y0, cx0 + x1], rgb[cy0 + y1, cx0 + x0], rgb[cy0 + y1, cx0 + x1]
            top = p[0] * (1 - wx) + p[1] * wx
            bot = p[2] * (1 - wx) + p[3] * wx
            v = (top * (1 - wy) + bot * wy) / 255.0
            t[0, y + py, x + px] = v[0]
            t[1, y + py, x + px] = v[1]
            t[2, y + py, x + px] = v[2]
    t[:, :py, :] = 114 / 255.0
    t[:, py + nh:, :] = 114 / 255.0
    t[:, :, :px] = 114 / 255.0
    t[:, :, px + nw:] = 114 / 255.0
    return t, s, px, py


def model_fitted_rect(w, h):
    """Port modelFittedRect: cắt vùng nhìn thấy về đúng tỉ lệ huấn luyện."""
    a = w / h
    if a > TRAINING_ASPECT:
        cw = int(h * TRAINING_ASPECT)
        return (w - cw) // 2, 0, min(cw, w), h
    ch = int(w / TRAINING_ASPECT)
    return 0, (h - ch) // 2, w, min(ch, h)


DEPTH_H, DEPTH_W = 224, 299


def depth_input(bgr):
    """Đưa ảnh vào depth đúng như app vừa sửa: resize thẳng về 299x224, KHÔNG đệm.

    Khung nguồn 4:3 giữ nguyên tỉ lệ: 1280x960 -> 299x224 khớp chính xác
    (1280 * 224/960 = 298.7). Bản cũ đệm vào 224x224, mất 75px mỗi bên và đo
    ra khoảng cách lớn gấp 2.55 lần.
    """
    x = cv2.resize(bgr, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
    t = np.transpose((x / 255.0 - IMNET_MEAN) / IMNET_STD, (2, 0, 1))[None]
    return t, DEPTH_W, DEPTH_H, 0, 0


def decode(head, names, scale, px, py, cw, ch, ox, oy):
    """Port của YoloPostprocessor.kt."""
    flat = head.reshape(head.shape[0], head.shape[1])
    cls = flat[4:4 + len(names), :]
    best_cls = cls.argmax(axis=0)
    best = cls.max(axis=0)
    idx = np.where(best >= CONF)[0]
    if idx.size == 0:
        return []
    cx, cy, w, h = flat[0, idx], flat[1, idx], flat[2, idx], flat[3, idx]
    mx1, my1 = cx - w / 2, cy - h / 2
    mx2, my2 = cx + w / 2, cy + h / 2
    sx = np.clip((mx1 - px) / scale, 0, cw) + ox
    ex = np.clip((mx2 - px) / scale, 0, cw) + ox
    sy = np.clip((my1 - py) / scale, 0, ch) + oy
    ey = np.clip((my2 - py) / scale, 0, ch) + oy
    boxes = np.stack([np.minimum(sx, ex), np.minimum(sy, ey),
                      np.maximum(sx, ex), np.maximum(sy, ey)], axis=1)
    scores, ids = best[idx], best_cls[idx]
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[r, 0]); y1 = np.maximum(boxes[i, 1], boxes[r, 1])
        x2 = np.minimum(boxes[i, 2], boxes[r, 2]); y2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ar = (boxes[r, 2] - boxes[r, 0]) * (boxes[r, 3] - boxes[r, 1])
        iou = inter / np.maximum(ai + ar - inter, 1e-9)
        order = r[~((iou > IOU_NMS) & (ids[r] == ids[i]))]
    return [{"id": int(ids[k]), "cls": names[ids[k]], "score": float(scores[k]),
             "box": boxes[k]} for k in keep]


def sample_box(depth, box, frame_w=None, frame_h=None,
               content_w=None, content_h=None, content_x=None, content_y=None):
    """Port của depth.py::sample_box.

    `box` nằm trong không gian khung hình, còn `depth` là bản 224x224. Phải ánh xạ
    box sang không gian của bản depth TRƯỚC khi lấy mẫu, nếu không mọi box đều bị
    cắt về góc trên-trái và hàm luôn trả None — đúng lỗi đã làm harness báo 0 cảnh
    báo dù logic còn nguyên.
    """
    H, W = depth.shape
    x1, y1, x2, y2 = box
    if frame_w and frame_h:
        # Ánh xạ vào VÙNG NỘI DUNG của letterbox, rồi cộng phần đệm. Ánh xạ vào
        # cả 224x224 như bản nén méo sẽ lệch ngang theo đúng tỉ lệ đệm.
        cw = content_w if content_w else W
        chh = content_h if content_h else H
        x1 = x1 / frame_w * cw + (content_x if content_w else 0)
        x2 = x2 / frame_w * cw + (content_x if content_w else 0)
        y1 = y1 / frame_h * chh + (content_y if content_h else 0)
        y2 = y2 / frame_h * chh + (content_y if content_h else 0)
    x1 = int(np.clip(x1, 0, W - 1)); x2 = int(np.clip(x2, x1 + 1, W))
    y1 = int(np.clip(y1, 0, H - 1)); y2 = int(np.clip(y2, y1 + 1, H))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    crop = min(max(DEPTH_CENTER_CROP, 0.05), 1.0)
    ccx, ccy = (x1 + x2) // 2, (y1 + y2) // 2
    rx1 = max(x1, int(ccx - bw * 0.5 * crop)); rx2 = min(x2, int(ccx + bw * 0.5 * crop))
    ry1 = max(y1, int(ccy - bh * 0.5 * crop)); ry2 = min(y2, int(ccy + bh * 0.5 * crop))
    if rx2 - rx1 < 2 or ry2 - ry1 < 2:
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
    vals = np.asarray(depth[ry1:ry2, rx1:rx2], np.float32).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if vals.size < DEPTH_MIN_SAMPLES:
        half = max(2, int(bw * 0.30))
        bx1 = max(x1, ccx - half); bx2 = min(x2, ccx + half)
        by1 = max(y1, y1 + int(bh * 0.5)); by2 = y2
        if by2 <= by1:
            by1 = max(0, by2 - 2)
        vals = np.asarray(depth[by1:by2, bx1:bx2], np.float32).ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None

    med = float(np.median(vals))
    if med <= 0:
        return None
    tol = max(0.05, DEPTH_OUTLIER_TOL)
    inl = vals[np.abs(vals - med) <= tol * med]
    dist = float(np.median(inl)) if inl.size >= DEPTH_MIN_SAMPLES else med
    dist *= DEPTH_SCALE
    if not np.isfinite(dist) or not (DEPTH_VALID_MIN <= dist <= DEPTH_VALID_MAX):
        return None
    return dist


# ── Ego motion (thay bản LK của Kotlin bằng OpenCV) ───────────────────────

class Ego:
    def __init__(self):
        self.prev = None; self.prev_t = None
        self.speed = 0.0

    def update(self, gray, t, depth):
        prev, pt = self.prev, self.prev_t
        self.prev, self.prev_t = gray, t
        if prev is None or pt is None or prev.shape != gray.shape:
            self.speed = 0.0; return 0.0
        dt = t - pt
        if dt <= 1e-3:
            self.speed = 0.0; return 0.0

        H, W = gray.shape
        cy = H * 0.5
        gx = np.linspace(int(W * 0.12), int(W * 0.88), 20)
        gy = np.linspace(int(H * 0.15), int(H * 0.95), 12)
        p0 = np.array([[x, y] for y in gy for x in gx], np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev, gray, p0, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        ok = st.reshape(-1).astype(bool)
        if ok.sum() < 20:
            self.speed = 0.0; return 0.0

        p0f, p1f = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        dy = p1f[:, 1] - p0f[:, 1]
        yr = p0f[:, 1] - cy
        ds = np.array([sample_point(depth, x, y, 5) for x, y in p0f])
        valid = np.isfinite(ds) & (ds > 0) & (ds >= EGO_BG_DEPTH) & (np.abs(yr) > H * EGO_MIN_ROW)
        if valid.sum() < 12:
            self.speed = 0.0; return 0.0

        coeff = (yr[valid] * dt) / ds[valid]
        dv = dy[valid]
        den = float(np.sum(coeff * coeff))
        if den < 1e-12:
            self.speed = 0.0; return 0.0
        speed = float(np.sum(coeff * dv) / den)

        resid = dv - coeff * speed
        c = float(np.median(resid))
        mad = float(np.median(np.abs(resid - c)))
        if mad > 1e-9:
            keep = np.abs(resid - c) <= 3.0 * mad
            if keep.sum() >= 12:
                c2, d2 = coeff[keep], dv[keep]
                d2s = float(np.sum(c2 * c2))
                if d2s > 1e-12:
                    speed = float(np.sum(c2 * d2) / d2s)

        speed = float(np.clip(speed, 0.0, EGO_MAX_SPEED))
        prev_speed = self.speed
        if prev_speed > 0.0:
            md = EGO_MAX_ACCEL * dt + 0.5
            speed = float(np.clip(speed, prev_speed - md, prev_speed + md))
            self.speed = EGO_SMOOTH * speed + (1 - EGO_SMOOTH) * prev_speed
        else:
            self.speed = speed
        return self.speed


def sample_point(depth, x, y, radius):
    H, W = depth.shape
    xi, yi = int(round(x)), int(round(y))
    r = max(1, int(radius))
    x1, x2 = max(0, xi - r), min(W, xi + r + 1)
    y1, y2 = max(0, yi - r), min(H, yi + r + 1)
    if x2 <= x1 or y2 <= y1:
        return np.nan
    v = np.asarray(depth[y1:y2, x1:x2], np.float32).ravel()
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if v.size else np.nan


# ── TrackHistory + RiskEngine (port từ Kotlin) ────────────────────────────

class History:
    def __init__(self, tid, cls):
        self.tid = tid; self.cls = cls
        self.samples = collections.deque(maxlen=HIST_LEN)
        self.distance = None
        self.fit = None
        self.speed = 0.0
        self.valid = False
        self.frames = 0
        self.source = "none"

    def update(self, t, d, source="measured"):
        self.frames += 1
        if d is None or not np.isfinite(d) or d <= 0:
            return False
        self.source = source
        self.distance = d if self.distance is None else DIST_EMA * d + (1 - DIST_EMA) * self.distance
        if self.samples and t <= self.samples[-1][0]:
            t = self.samples[-1][0] + 1e-3
        self.samples.append((t, d))
        return True

    def interpolated(self, now, max_rate):
        """Port của interpolated_distance."""
        if len(self.samples) < 2:
            if len(self.samples) == 1:
                return float(self.samples[-1][1]), "held"
            return None, "none"
        (t1, d1), (t2, d2) = self.samples[-2], self.samples[-1]
        dt = t2 - t1
        if dt <= 1e-6:
            return float(d2), "held"
        rate = (d2 - d1) / dt
        if abs(rate) > max_rate:
            return float(d2), "held"
        pred = d2 + rate * (now - t2)
        if not np.isfinite(pred) or pred <= 0:
            return float(d2), "held"
        return float(pred), "interpolated"

    def estimate(self, bias):
        self.valid = False
        if len(self.samples) < MIN_SPEED_SAMPLES:
            self.speed = 0.0
            return
        ts = np.array([s[0] for s in self.samples])
        ds = np.array([s[1] for s in self.samples])
        span = ts[-1] - ts[0]
        if span < MIN_WINDOW_S:
            self.speed = 0.0
            return
        tr = ts - ts[-1]
        var = float(np.var(tr))
        if var < 1e-9:
            self.speed = 0.0
            return
        slope = float(np.sum((tr - tr.mean()) * (ds - ds.mean())) / (var * len(ts)))
        speed = -slope - bias
        speed = float(np.clip(speed, -MAX_SPEED, MAX_SPEED))
        self.fit = float(ds.mean() - slope * tr.mean())
        self.speed = speed
        self.valid = True

    def ttc(self):
        if not self.valid:
            return float("inf"), False
        d = self.fit if self.fit is not None else self.distance
        if d is None or not np.isfinite(d) or d <= 0:
            return float("inf"), False
        if self.speed <= 1e-3:
            return float("inf"), False
        return float(d / self.speed), True


def classify(ttc, dist):
    lv = 0
    if np.isfinite(ttc) and ttc < float("inf"):
        lv = 2 if ttc <= DANGER_TTC else (1 if ttc <= WARN_TTC else 0)
    ld = 0
    if dist is not None and np.isfinite(dist) and dist > 0:
        ld = 2 if dist <= DANGER_DIST else (1 if dist <= WARN_DIST else 0)
    raw = max(lv, ld)
    return raw, lv, ld


# ── ByteTracker 2 giai đoạn (port ByteTracker.kt) ─────────────────────────

class Tracker:
    def __init__(self):
        self.tracks = []
        self.hist = {}
        self.next_id = 1

    def update(self, dets):
        high = [d for d in dets if d["score"] >= 0.5]
        low = [d for d in dets if 0.1 <= d["score"] < 0.5]
        matched_t, matched_d = set(), set()

        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(high):
                iou = iou_of(t["box"], d["box"])
                if iou >= IOU_NMS:
                    pairs.append((iou, ti, di))
        for iou, ti, di in sorted(pairs, key=lambda z: -z[0]):
            t = self.tracks[ti]
            if t["id"] in matched_t or di in matched_d:
                continue
            matched_t.add(t["id"]); matched_d.add(di)
            t["box"] = high[di]["box"]; t["det"] = high[di]; t["missed"] = 0
            t["hits"] += 1

        pairs = []
        for ti, t in enumerate(self.tracks):
            if t["id"] in matched_t:
                continue
            for di, d in enumerate(low):
                iou = iou_of(t["box"], d["box"])
                if iou >= IOU_NMS:
                    pairs.append((iou, ti, di))
        for iou, ti, di in sorted(pairs, key=lambda z: -z[0]):
            t = self.tracks[ti]
            if t["id"] in matched_t or di in matched_d:
                continue
            matched_t.add(t["id"]); matched_d.add(di)
            t["box"] = low[di]["box"]; t["det"] = low[di]; t["missed"] = 0
            t["hits"] += 1

        for di, d in enumerate(dets):
            if di in matched_d or d["score"] < CONF:
                continue
            self.tracks.append({"id": self.next_id, "box": d["box"], "det": d,
                                "missed": 0, "hits": 1})
            matched_t.add(self.next_id)
            matched_d.add(di)
            self.next_id += 1

        out = []
        alive = []
        for t in self.tracks:
            if t["id"] in matched_t:
                t["missed"] = 0
                alive.append(t)
                out.append(t)
            else:
                t["missed"] += 1
                if t["missed"] <= TRACK_BUFFER:
                    alive.append(t)
        self.tracks = alive

        live = set()
        for t in out:
            live.add(t["id"])
            if t["id"] not in self.hist:
                self.hist[t["id"]] = History(t["id"], t["det"]["cls"])
            self.hist[t["id"]].cls = t["det"]["cls"]
        for tid in list(self.hist):
            if tid not in live:
                del self.hist[tid]
        return out


def iou_of(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw, ih = x2 - x1, y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ai = (a[2] - a[0]) * (a[3] - a[1])
    bi = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(ai + bi - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--hz", type=float, default=30.0,
                    help="tần suất xử lý mô phỏng; 3.5 là nhịp thật của máy")
    ap.add_argument("--trace", action="store_true",
                    help="in từng frame xử lý: khoảng cách, hành lang, mức, streak")
    ap.add_argument("--until", type=float, default=None,
                    help="dừng sau N giây video, để chạy trace nhanh")
    args = ap.parse_args()

    yolo = ort.InferenceSession(str(MODELS / "best.onnx"), providers=["CPUExecutionProvider"])
    dep = ort.InferenceSession(str(MODELS / "depth_299x224_int8.onnx"), providers=["CPUExecutionProvider"])
    yn = yolo.get_inputs()[0].name
    dn = dep.get_inputs()[0].name
    raw = eval(yolo.get_modelmeta().custom_metadata_map["names"])
    names = [raw[i] for i in range(max(raw) + 1)]

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / max(args.hz, 0.01))))
    print(f"{args.video}\n{fps:.0f} fps, {total} frame, xử lý mỗi {step} frame "
          f"(~{args.hz:.1f} Hz)\n")

    tracker, ego = Tracker(), Ego()
    streaks = {}
    last_depth = None
    last_depth_frame = -999
    depth_runs = 0
    depth_marks = 0
    depth_geom = None
    first_warning = first_danger = None
    timeline = []
    infer_count = 0

    t0 = time.time()
    for f in range(total):
        now = f / fps
        if args.until is not None and now > args.until:
            break
        if f % step != 0:
            continue
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        frame_h = H
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ox, oy, cw, ch = model_fitted_rect(W, H)

        # YOLO does not depend on depth. Guarding it on `last_depth is not None`
        # meant the first frame produced no detections, so the depth pass had nothing
        # to sample, and its counter then blocked every later depth run.
        tensor, s, px, py = kotlin_bilinear_letterbox(rgb, (ox, oy, cw, ch))
        head = yolo.run(None, {yn: tensor[None]})[0][0]
        dets = decode(head, names, s, px, py, cw, ch, ox, oy)
        infer_count += 1

        ego_speed = ego.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), now, last_depth) \
            if last_depth is not None else 0.0

        tracked = tracker.update(dets)

        if last_depth is None or (infer_count - depth_marks) >= DEPTH_INTERVAL:
            din, dnw, dnh, dpx, dpy = depth_input(frame)
            dm = dep.run(None, {dn: din})[0]
            last_depth = np.squeeze(dm)
            depth_geom = (dnw, dnh, dpx, dpy)
            depth_marks = infer_count
            depth_runs += 1
            for tr in tracked:
                # `h` is the frame height here; naming the history `h` shadowed it and
                # sample_box received a History object instead of a pixel count.
                hist = tracker.hist[tr["id"]]
                d = sample_box(last_depth, tr["box"], frame_h, W,
                               *depth_geom) if depth_geom else None
                if d is not None:
                    hist.update(now, d, "measured")
                else:
                    est, src = hist.interpolated(now, 12.0)
                    if est is not None:
                        hist.update(now, est, src)

        for t in tracked:
            tracker.hist[t["id"]].estimate(ego_speed)

        live = {t["id"] for t in tracked}
        best = None
        for t in tracked:
            h = tracker.hist[t["id"]]
            box = t["box"]
            ccx = (box[0] + box[2]) / 2
            ccy = (box[1] + box[3]) / 2
            ttc, ttc_ok = h.ttc()
            raw_lv, lv_ttc, lv_dist = classify(ttc, h.distance)
            in_roi = (W * ROI[0] <= ccx <= W * ROI[2]) and (H * ROI[1] <= ccy <= H * ROI[3])
            in_corr = (W * CORRIDOR[0] <= ccx <= W * CORRIDOR[2]) and (H * CORRIDOR[1] <= ccy <= H * CORRIDOR[3])
            level = raw_lv if in_corr else 0
            if normalise(h.cls) not in PRIMARY or not in_roi:
                level = 0
            streak = streaks.get(t["id"], 0) + 1 if streaks.get((t["id"], level)) == level else 1
            streaks[(t["id"], level)] = level
            streaks[t["id"]] = streak
            confirmed = level != 0 and h.frames >= MIN_HITS and streak >= CONFIRM_FRAMES
            if args.trace:
                d_txt = "None" if h.distance is None else f"{h.distance:.2f}"
                print(f"TRACE t={now:5.2f} id={t['id']:2d} {t['det']['cls']:<10}"
                      f" score={t['det']['score']:.2f} dist={d_txt:>6}"
                      f" sValid={int(h.valid)} close={h.speed:6.2f}"
                      f" ttc={'inf' if not ttc_ok else f'{ttc:.2f}'}"
                      f" roi={int(in_roi)} corr={int(in_corr)}"
                      f" raw={raw_lv} level={level} streak={streak}"
                      f" ego={ego_speed:.2f} frames={h.frames}", flush=True)
            if confirmed and (best is None or level > best[0] or
                              (level == best[0] and (ttc if ttc_ok else 1e6) < best[2])):
                best = (level, h, ttc if ttc_ok else float("inf"))
        for k in [k for k in list(streaks) if isinstance(k, tuple) and k[0] not in live]:
            del streaks[k]

        if best:
            level, h, ttc = best
            label = "DANGER" if level == 2 else "WARNING"
            if level == 2 and first_danger is None:
                first_danger = now
            if first_warning is None:
                first_warning = now
            timeline.append((now, label, h.cls, h.distance, ttc))

    print(f"đọc xong {total} frame trong {time.time() - t0:.0f}s, "
          f"{infer_count} lần nhận diện, depth chạy {depth_runs} lần\n")
    if timeline:
        print(f"CẢNH BÁO ĐẦU TIÊN: giây {timeline[0][0]:.2f}")
        print(f"  {timeline[0][1]}  {timeline[0][2]}  "
              f"khoảng cách {timeline[0][3]:.1f}m  TTC {timeline[0][4]:.2f}s")
    else:
        print("KHÔNG có cảnh báo nào trong video này.")
    if first_danger is not None:
        print(f"DANGER ĐẦU TIÊN: giây {first_danger:.2f}")
    print(f"\ntổng cảnh báo: {len(timeline)} lần")
    for i, (t, lab, cls, d, ttc) in enumerate(timeline[:25]):
        print(f"  {t:6.2f}s  {lab:<8}{cls:<10}{d if d is None else f'{d:.1f}m':>8}  "
              f"TTC {'inf' if not np.isfinite(ttc) else f'{ttc:.2f}s'}")
    if len(timeline) > 25:
        print(f"  ... còn {len(timeline) - 25} lần")


if __name__ == "__main__":
    main()