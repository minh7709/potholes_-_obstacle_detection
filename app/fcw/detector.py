"""YOLO detection + ByteTrack tracking.

The tracker is what makes the rest of the system possible: it is the only thing
that turns "a motorcycle-shaped blob in frame 100" and "the same motorcycle in
frame 101" into one object with a stable identity and therefore one distance
history.

Model and class names
---------------------
The default detector is ``assets/best.pt``, the 15-class road-obstacle model
trained on local footage:

    stop_sign, person, bicycle, bus, truck, car, motorbike, reflective_cone,
    ashcan, warning_column, spherical_roadblock, pothole, dog, tricycle,
    fire_hydrant

A class-naming trap
-------------------
That model calls a motorcycle ``motorbike``, while the COCO weights call it
``motorcycle``. Filtering by literal name meant the single most important class
for collision warning matched NOTHING, and it failed silently: the pipeline kept
running, just never reported a motorcycle.

So names are normalised before matching (see ``_normalise``) and every requested
class that the model cannot provide is reported loudly at startup. A silently
missing class is worse than a crash, because it looks like a working system.
"""

import time

import numpy as np


def _normalise(name):
    """Canonical form of a class name, so naming variants still match.

    'motorbike' and 'motorcycle' are the same real-world class, and 'stop_sign'
    versus 'stop sign' differ only by the separator. Comparing normalised keys
    removes that whole class of silent failure.
    """
    text = str(name).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


# Extra spellings that should resolve to an existing class name.
_ALIASES = {
    "motorcycle": "motorbike",
    "motor bike": "motorbike",
    "motorbike": "motorbike",
    "potholes": "pothole",
    "manhole": "pothole",
    "traffic signal": "stop sign",
    "traffic light": "stop sign",
    "trash can": "ashcan",
    "garbage can": "ashcan",
    "cone": "reflective cone",
    "bollard": "warning column",
    "hydrant": "fire hydrant",
}

# Default class list, expressed with the vocabulary a user is likely to type.
# Resolution runs both ways, so both 'motorcycle' and 'motorbike' select id 6.
DEFAULT_TARGET_CLASSES = [
    "motorcycle", "car", "bus", "truck", "bicycle", "person",
]


def resolve_device(preferred="auto"):
    """Return a validated torch device string, preferring CUDA when available."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if preferred == "auto":
        try:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if str(preferred).startswith("cuda"):
        try:
            if torch.cuda.is_available():
                return preferred
        except Exception:
            pass
        print(f"[FCW] Device '{preferred}' unavailable, falling back to CPU.")
    return "cpu"


class VehicleDetector:
    """YOLO + tracker wrapper exposing a single `detect(frame)` call."""

    def __init__(self, model_name="assets/best.pt", device="auto", conf=0.45, iou=0.5,
                 classes=None, img_size=640, tracker="bytetrack.yaml",
                 track_buffer=30, min_area_ratio=0.0008):
        from ultralytics import YOLO

        self.device = resolve_device(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.img_size = int(img_size)
        self.tracker_config = tracker
        self.track_buffer = int(track_buffer)
        self.min_area_ratio = float(min_area_ratio)

        self._wanted = [c.strip() for c in (classes or DEFAULT_TARGET_CLASSES) if str(c).strip()]

        print(f"[FCW] Loading YOLO '{model_name}' on {self.device}...")
        self.model = YOLO(model_name)
        try:
            self.model.to(self.device)
        except Exception as exc:
            print(f"[FCW] Could not move model to {self.device}: {exc}")

        names = getattr(self.model, "names", None) or {}
        if names and isinstance(next(iter(names)), str):
            names = {int(v): k for k, v in names.items()}
        self.class_names = names

        # Map normalised class name -> model class id, so 'motorcycle' finds
        # 'motorbike' and 'stop sign' finds 'stop_sign'.
        available = {}
        for cid, name in names.items():
            available[_normalise(name)] = cid
            key = _normalise(_ALIASES.get(_normalise(name), name))
            available.setdefault(key, cid)

        self._class_arg = None
        resolved, missing = [], []
        for wanted in self._wanted:
            key = _normalise(wanted)
            cid = available.get(key) or available.get(_normalise(_ALIASES.get(key, key)))
            if cid is None:
                missing.append(wanted)
            else:
                resolved.append(cid)

        # Loud on purpose. A requested class the model cannot detect would
        # otherwise silently never appear in the output, which is exactly the
        # failure that hid 'motorcycle' vs 'motorbike'.
        if missing:
            print(f"[FCW] WARNING: model '{model_name}' has no class for "
                  f"{missing}. Available: {sorted(names.values())}")
            print(f"[FCW] These objects will never be detected. Check "
                  f"classes.targets in config.yaml.")
        if not resolved:
            raise ValueError(
                f"None of the requested classes {self._wanted} exist in "
                f"'{model_name}'. Available classes: {sorted(names.values())}")

        self._class_arg = sorted(set(resolved))
        self.requested_classes = list(self._wanted)
        self.unmatched_classes = missing
        print(f"[FCW] Watching {len(self._class_arg)} classes "
              f"(from {self._wanted}): "
              f"{sorted(names[c] for c in self._class_arg)}")

        self._torch_device = 0 if self.device.startswith("cuda") else "cpu"
        self.inference_ms = 0.0
        self.inference_count = 0
        self._loaded = True

    def detect(self, frame):
        """Detect and track objects in one frame.

        Returns a list of dicts sorted largest-first (nearest/most prominent
        first). Fields per detection:
            track_id, class_id, class_name, confidence, box (x1,y1,x2,y2),
            area_ratio, center_x, center_y
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        start = time.perf_counter()
        try:
            results = self.model.track(
                frame,
                persist=True,          # keeps track state across calls
                tracker=self.tracker_config,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.img_size,
                classes=self._class_arg,
                verbose=False,
                device=self._torch_device,
            )
        except Exception as exc:
            print(f"[FCW] Detection failed: {exc}")
            return []
        finally:
            self.inference_ms = (time.perf_counter() - start) * 1000.0
            self.inference_count += 1

        if not results:
            return []
        return self._parse(results[0], w, h, frame_area)

    def _parse(self, result, frame_w, frame_h, frame_area):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        cls_raw = boxes.cls
        conf_raw = boxes.conf
        cls_list = cls_raw.tolist() if hasattr(cls_raw, "tolist") else list(cls_raw)
        conf_list = conf_raw.tolist() if hasattr(conf_raw, "tolist") else list(conf_raw)
        id_obj = boxes.id
        id_list = (id_obj.tolist() if hasattr(id_obj, "tolist") else list(id_obj)) \
            if id_obj is not None else [None] * len(cls_list)

        xyxy = boxes.xyxy.tolist()
        detections = []
        max_x = frame_w - 1.0
        max_y = frame_h - 1.0
        min_area = self.min_area_ratio

        for i, box in enumerate(xyxy):
            x1 = min(max(float(box[0]), 0.0), max_x)
            y1 = min(max(float(box[1]), 0.0), max_y)
            x2 = min(max(float(box[2]), 0.0), max_x)
            y2 = min(max(float(box[3]), 0.0), max_y)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1.0 or bh <= 1.0:
                continue

            area_ratio = (bw * bh) / frame_area if frame_area > 0 else 0.0
            if area_ratio < min_area:
                continue

            raw_id = id_list[i]
            track_id = None
            if raw_id is not None:
                try:
                    tid = int(raw_id)
                    # ByteTrack uses -1 for not-yet-confirmed tracks.
                    track_id = tid if tid >= 0 else None
                except (TypeError, ValueError):
                    track_id = None

            class_id = int(cls_list[i])
            detections.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": self.class_names.get(class_id, str(class_id)),
                "confidence": float(conf_list[i]),
                "box": (x1, y1, x2, y2),
                "area_ratio": area_ratio,
                "area_px": float(bw * bh),
                "center_x": (x1 + x2) * 0.5,
                "center_y": (y1 + y2) * 0.5,
            })

        # Largest first: nearer, more prominent objects sort to the front, which
        # is the natural priority order for collision analysis.
        detections.sort(key=lambda d: d["area_ratio"], reverse=True)
        return detections

    def reset(self):
        """Force the tracker to forget all tracks.

        Needed between two videos: leftover associations from a previous clip
        would otherwise seed a new clip's distance histories with wrong range.
        """
        try:
            self.model.predictor = None
        except Exception:
            pass


def _norm_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [v.strip().lower() for v in values.split(",") if v.strip()]
    return [str(v).strip().lower() for v in values if str(v).strip()]