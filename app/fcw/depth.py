"""Monocular depth via Depth Anything V2 (Metric Outdoor).

WHAT THIS MODEL RETURNS
-----------------------
Depth Anything V2 ships two families of weights:

* Relative (``Depth-Anything-V2-*-hf``): affine-invariant disparity. The
  numbers are NOT metres.
* Metric (``Depth-Anything-V2-Metric-{Indoor,Outdoor}-*``): trained against
  absolute ground truth, so the output is intended to be metres.

We use Metric Outdoor, which is the variant trained on driving scenes.
Verified on real footage: road surface ~3 m, sky ~37 m, mid-ground ~56 m.

Even so, metric is not the same as calibrated:

* Accuracy depends on FOV, mounting height and lens distortion.
* Real-world error is roughly 10-25%, worse for small distant objects,
  partial occlusions, night and heavy rain.

So the output is treated as an ESTIMATE everywhere. It is never presented as
a measurement, and the TTC chain is built to absorb error rather than hide it.
See ``DepthCalibration`` for the one-point correction, and FCW_README for the
full discussion.

CPU NOTES
---------
Depth is by far the most expensive stage. Measured on an i5-8250U (4c/8t,
no GPU) for a 1280x720 frame:

    height 518 -> 2580 ms      height 350 -> 841 ms      height 224 -> 285 ms

Resolution is close to quadratic in cost, so ``processing_height`` and
``depth.interval`` dominate throughput far more than any YOLO tuning does.
There is no official Metric-Tiny checkpoint, and exporting a dynamic-resolution
ONNX fails because DPT interpolates fixed-size position embeddings, so the
224 px PyTorch path is the practical CPU optimum for this stack.
"""

import time

import numpy as np

from app.config import FCW_DEPTH_MODEL_ID, FCW_DEVICE


class DepthEstimator:
    """Full-frame depth inference with a cheap, cached depth map.

    Two caches matter for speed:
      * ``_last_raw`` — the depth map from the most recent real inference.
        ``sample_box`` reads from it, so sampling N objects costs nothing
        beyond the box arithmetic.
      * ``_last_source_shape`` — the frame size the map corresponds to, so a
        mismatched map is detected instead of returning silently wrong metres.
    """

    def __init__(self, model_id=FCW_DEPTH_MODEL_ID, device=FCW_DEVICE,
                 processing_height=224, cfg=None):
        self.cfg = cfg
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.processing_height = max(154, int(processing_height))

        self._model = None
        self._processor = None
        self._loaded = False

        self._last_raw = None
        self._last_source_shape = None
        self.inference_ms = 0.0
        self.inference_count = 0
        self._net_hw = (self.processing_height, self.processing_height)

    # ── Lazy model loading ──────────────────────────────────────────
    def _ensure_model(self):
        """Load YOLO's sibling model on first use.

        Deferred rather than eager because importing torch plus loading two
        networks costs seconds, and browsing the web UI should not pay that.
        """
        if self._loaded:
            return self._model is not None

        self._loaded = True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self._torch = torch
            print(f"[FCW] Loading depth model '{self.model_id}' on {self.device} "
                  f"(H={self.processing_height})...")
            t0 = time.perf_counter()
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            self._model.eval()
            try:
                self._model.to(self.device)
            except Exception as exc:
                print(f"[FCW] Could not move depth model to {self.device}: {exc}")
            print(f"[FCW] Depth model ready in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            print(f"[FCW] Depth model unavailable: {exc}")
            self._model = None
        return self._model is not None

    # ── Inference ───────────────────────────────────────────────────
    def infer(self, frame):
        """Run depth on a whole frame. Returns (H, W) float32 metres, or None.

        `inference_mode` rather than `no_grad`: it skips autograd bookkeeping
        entirely, which measurably helps on CPU where every op is memory bound.
        """
        if frame is None or frame.size == 0:
            return None
        if not self._ensure_model():
            return None

        torch = self._torch
        start = time.perf_counter()
        try:
            from PIL import Image

            h, w = frame.shape[:2]
            rgb = frame[:, :, ::-1].copy() if frame.ndim == 3 else frame
            image = Image.fromarray(rgb)

            # Preserve aspect ratio: stretching the input distorts the geometry
            # the network infers depth from.
            scale = self.processing_height / float(h)
            net_w = max(32, int(round(w * scale)))
            net_h = self.processing_height

            inputs = self._processor(
                images=image, return_tensors="pt",
                size={"height": net_h, "width": net_w},
            )
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self._model(**inputs)

            post = self._processor.post_process_depth_estimation(
                outputs, target_sizes=[(h, w)]
            )
            if not post:
                return None
            depth = post[0]["predicted_depth"]
            if torch.is_tensor(depth):
                depth = depth.detach().cpu().numpy()
            depth = np.asarray(depth, dtype=np.float32)
        except Exception as exc:
            print(f"[FCW] Depth inference failed: {exc}")
            return None
        finally:
            self.inference_ms = (time.perf_counter() - start) * 1000.0
            self.inference_count += 1

        if depth.ndim != 2:
            return None

        self._last_raw = depth
        self._last_source_shape = frame.shape[:2]
        self._net_hw = (net_h, net_w)
        return depth

    @property
    def last_depth_map(self):
        return self._last_raw

    def has_fresh_map(self, frame_shape):
        """True when the cached map matches this frame's size.

        Guards a subtle bug class: if the source resolution changes mid-run, a
        stale map would be sampled with the wrong coordinates and report
        plausible but wrong distances.
        """
        return (self._last_raw is not None
                and tuple(self._last_raw.shape[:2]) == tuple(frame_shape[:2]))

    # ── Sampling ────────────────────────────────────────────────────
    def sample_box(self, box, depth_map=None, cfg=None, center_crop=None,
                   min_samples=None, outlier_tolerance=None,
                   valid_min=None, valid_max=None):
        """Estimate distance to one object from its bounding box.

        Steps, in order:
        1. Sample many pixels from a CENTRE CROP of the box, not a single pixel
           and not the whole box. The whole box of a motorcycle is mostly sky
           and road; the centre is vehicle body.
        2. Discard non-finite values and non-positive depths.
        3. Reject pixels inconsistent with the median (background bleeding
           through gaps between wheels and handlebars).
        4. Take the MEDIAN of the survivors, which is robust to residual noise.
        5. Reject anything outside a plausible metric range.

        Returns (distance_metres, sample_count); (None, 0) when unusable. Never
        raises, because a bad frame must not kill the pipeline.
        """
        depth_map = self._last_raw if depth_map is None else depth_map
        if depth_map is None:
            return None, 0

        cfg = self.cfg
        if center_crop is None:
            center_crop = _cfg_float(cfg, "depth.center_crop", 0.5)
        if min_samples is None:
            min_samples = _cfg_int(cfg, "depth.min_valid_samples", 16)
        if outlier_tolerance is None:
            outlier_tolerance = _cfg_float(cfg, "depth.outlier_tolerance", 0.25)
        if valid_min is None:
            valid_min = _cfg_float(cfg, "depth.valid_min_m", 1.0)
        if valid_max is None:
            valid_max = _cfg_float(cfg, "depth.valid_max_m", 120.0)

        H, W = depth_map.shape[:2]
        x1, y1, x2, y2 = _clip_box(box, W, H)
        bw, bh = x2 - x1, y2 - y1
        if bw < 3 or bh < 3:
            return None, 0

        # Centre crop.
        crop = min(max(0.05, float(center_crop)), 1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        hw = bw * 0.5 * crop
        hh = bh * 0.5 * crop
        sx1 = int(max(0, round(cx - hw)))
        sx2 = int(min(W, round(cx + hw)))
        sy1 = int(max(0, round(cy - hh)))
        sy2 = int(min(H, round(cy + hh)))

        if sx2 - sx1 < 2 or sy2 - sy1 < 2:
            sx1, sy1, sx2, sy2 = x1, y1, x2, y2

        patch = depth_map[sy1:sy2, sx1:sx2]
        values = _clean_depth(patch)
        if values.size < min_samples:
            # Centre crop too small: fall back to the lower-centre band, which is
            # where the wheels and body are for a ground vehicle.
            values = _clean_depth(_lower_center_band(depth_map, x1, y1, x2, y2))
        if values.size < 8:
            return None, 0

        median = float(np.median(values))
        if median <= 0.0:
            return None, 0

        tol = max(0.05, float(outlier_tolerance))
        inliers = values[np.abs(values - median) <= tol * median]
        distance = float(np.median(inliers)) if inliers.size >= min_samples else median

        distance = _apply_calibration(distance, cfg)
        if not np.isfinite(distance):
            return None, 0
        if not (valid_min <= distance <= valid_max):
            return None, 0

        return distance, int(values.size)

    def sample_point(self, x, y, radius=6, depth_map=None):
        """Median depth at one point. Used for ego-motion background sampling."""
        depth_map = self._last_raw if depth_map is None else depth_map
        if depth_map is None:
            return None
        H, W = depth_map.shape[:2]
        xi, yi = int(round(x)), int(round(y))
        r = max(1, int(radius))
        x1, x2 = max(0, xi - r), min(W, xi + r + 1)
        y1, y2 = max(0, yi - r), min(H, yi + r + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        values = _clean_depth(depth_map[y1:y2, x1:x2])
        if values.size == 0:
            return None
        return float(np.median(values))

    # ── Interpolation support ───────────────────────────────────────
    def clear_cache(self):
        self._last_raw = None
        self._last_source_shape = None


# ── Calibration ───────────────────────────────────────────────────────
class DepthCalibration:
    """Corrects the systematic scale error of the metric depth model.

    Metric depth is unbiased on average but systematically off for a given
    camera. Supplying ONE known reference distance ("a motorcycle parked
    exactly 10 m away") yields a single scale factor:

        scale = true_distance / model_reading_at_that_distance

    This cannot fix per-object error, but it removes the dominant systematic
    bias, which is precisely what makes uncorrected output unusable.
    """

    def __init__(self, scale=None, enabled=True):
        self.enabled = enabled
        self.scale = 1.0 if scale is None else float(scale)
        self.calibrated = abs(self.scale - 1.0) > 1e-6

    @classmethod
    def from_reference(cls, ref_depth_m, ref_true_m, gain=1.0, enabled=True):
        ref_depth_m = float(ref_depth_m or 0.0)
        ref_true_m = float(ref_true_m or 0.0)
        scale = (ref_true_m / ref_depth_m) if ref_depth_m > 0 else 1.0
        return cls(scale=scale * float(gain or 1.0), enabled=enabled)

    def apply(self, depth_m):
        if depth_m is None:
            return None
        return float(depth_m) * self.scale

    def describe(self):
        return (f"scale={self.scale:.3f} "
                f"({'calibrated' if self.calibrated else 'uncorrected'})")


# ── Helpers ───────────────────────────────────────────────────────────
def _resolve_device(preferred):
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


def _clean_depth(patch):
    """Return finite, strictly positive depths from a patch as a flat array."""
    if patch is None or patch.size == 0:
        return np.empty(0, dtype=np.float32)
    values = np.asarray(patch, dtype=np.float32).ravel()
    # NaN and +/-inf are the failure modes a monocular model actually produces;
    # they must never reach the median.
    values = values[np.isfinite(values)]
    values = values[values > 0.0]
    return values


def _lower_center_band(depth_map, x1, y1, x2, y2):
    """Horizontal band from the vertical middle to the bottom, centred in x."""
    H, W = depth_map.shape[:2]
    bw = max(1, x2 - x1)
    cx = (x1 + x2) // 2
    half = max(2, int(bw * 0.30))
    sx1 = max(x1, cx - half)
    sx2 = min(x2, cx + half)
    sy1 = max(y1, y1 + int((y2 - y1) * 0.5))
    sy2 = y2
    if sy2 <= sy1:
        sy1 = max(0, sy2 - 2)
    return depth_map[sy1:sy2, sx1:sx2]


def _clip_box(box, W, H):
    """Clamp a bbox to the frame and return integer coordinates."""
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return 0, 0, 0, 0
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _apply_calibration(distance, cfg):
    scale = _cfg_float(cfg, "calibration.depth_scale", 1.0) if cfg else 1.0
    return float(distance) * (scale if scale and scale > 0 else 1.0)


def _cfg_float(cfg, key, default):
    if cfg is None:
        return default
    try:
        value = cfg.get(key, default, cast=float)
        return default if value is None else value
    except Exception:
        return default


def _cfg_int(cfg, key, default):
    if cfg is None:
        return default
    try:
        value = cfg.get(key, default, cast=int)
        return default if value is None else value
    except Exception:
        return default