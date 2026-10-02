"""Per-track distance history, interpolation, closing speed and ego-motion.

Two independent problems live here.

1. Per-track distance history (TrackHistory)
   Turns a stream of noisy depth estimates for one tracked object into a
   smoothed distance and a stable closing speed.

2. Ego-motion estimation (EgoMotionEstimator)
   Removes the velocity of the camera from the apparent closing rate.

Why ego-motion cannot be skipped
--------------------------------
When your car drives at 15 m/s, a parked motorcycle appears to close at 15 m/s.
Without compensation the system reports a huge closing speed for everything and
alarms continuously, making it useless.

What "closing speed" actually means
-----------------------------------
From a single camera we can only measure RELATIVE closing speed: the rate at
which the gap to the target shrinks. It is the sum of the target's own motion
and the ego vehicle's motion. The pipeline subtracts an ESTIMATED ego speed,
which reduces the bias, but the residual error is real:

  * ego speed from optical flow is noisy and assumes a static scene
  * the target may be moving laterally or turning, which a single camera
    cannot disentangle from range change
  * monocular depth error of 10-25% feeds straight into the rate

So `closing_speed` is a RELATIVE quantity. It is not the motorcycle's true
speed, and TTC derived from it inherits that limitation. `FCW_EGO_BIAS_GAIN`
exists so a real ego speed from GPS/OBD can replace the estimate later.
"""

from collections import deque

import numpy as np


# ── Configuration, sourced from config.yaml via small helpers ─────────
def _f(cfg, key, default):
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default, cast=float)
        return default if v is None else float(v)
    except Exception:
        return default


def _i(cfg, key, default):
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default, cast=int)
        return default if v is None else int(v)
    except Exception:
        return default


class TrackHistory:
    """Distance history and kinematics for one tracking ID.

    `source` on each sample records how the value was obtained, and is
    propagated into the CSV as `distance_source`. This matters for honesty:
    an interpolated distance is not a measurement, and the output should say so.
    """

    __slots__ = (
        "track_id", "class_name", "frames_seen", "last_seen",
        "box", "confidence", "samples",
        "raw_distance", "distance", "fit_distance",
        "closing_speed", "speed_valid", "speed_ema",
        "distance_source", "ego_compensated",
        "_last_measured", "_last_measured_t",
        "_smoothing", "_alpha", "_window",
    )

    def __init__(self, track_id, class_name="unknown", history_len=20, smoothing="ema",
                 alpha=0.3, window=5):
        self.track_id = track_id
        self.class_name = class_name
        self.frames_seen = 0
        self.last_seen = 0.0
        self.box = (0.0, 0.0, 0.0, 0.0)
        self.confidence = 0.0

        # (timestamp, distance) — raw samples, no smoothing applied.
        self.samples = deque(maxlen=max(4, int(history_len)))

        self.raw_distance = None       # latest depth reading (may be interpolated)
        self.distance = None           # smoothed distance
        self.fit_distance = None       # regression-implied distance at newest t
        self.closing_speed = 0.0
        self.speed_valid = False
        self.speed_ema = None
        self.distance_source = "none"  # measured | interpolated | held | none
        self.ego_compensated = 0.0

        self._smoothing = smoothing
        self._alpha = max(0.01, min(1.0, float(alpha)))
        self._window = max(2, int(window))
        self._last_measured = None
        self._last_measured_t = None

    # ── Updates ─────────────────────────────────────────────────────
    def update(self, timestamp, distance, box=None, confidence=None,
               class_name=None, source="measured"):
        """Feed one frame's reading.

        `distance` may be None (no usable depth this frame), in which case the
        history is left intact and the caller falls back to interpolation.
        """
        self.frames_seen += 1
        self.last_seen = timestamp
        if box is not None:
            self.box = box
        if confidence is not None:
            self.confidence = confidence
        if class_name:
            self.class_name = class_name

        if distance is None or not np.isfinite(distance) or distance <= 0:
            return False

        if source == "measured":
            self._last_measured = float(distance)
            self._last_measured_t = timestamp

        self.raw_distance = float(distance)

        # Smoothing: EMA by default; moving average as an alternative.
        if self.distance is None:
            self.distance = float(distance)
        elif self._smoothing == "moving_average":
            window = list(self.samples)[-self._window:]
            values = [s[1] for s in window] + [float(distance)]
            self.distance = float(np.mean(values[-self._window:]))
        else:
            a = self._alpha
            self.distance = a * float(distance) + (1.0 - a) * self.distance

        # Strictly increasing timestamps keep the regression window well formed.
        if self.samples and timestamp <= self.samples[-1][0]:
            timestamp = self.samples[-1][0] + 1e-3
        self.samples.append((timestamp, float(distance)))
        self.distance_source = source
        return True

    # ── Kinematics ──────────────────────────────────────────────────
    def estimate_speed(self, ego_bias_mps=0.0, min_samples=None, min_window_sec=None,
                       max_speed=None, ema_alpha=None):
        """Estimate RELATIVE closing speed (m/s) by least-squares fit.

        Why a regression instead of a two-frame difference:

        Monocular depth error of 10% on a 10 m object is ~1 m. At 30 fps that is
        a 0.033 s baseline, so a naive per-frame difference has an error of
        roughly +/-30 m/s. Fitting a line across the whole recent window uses
        every sample coherently and suppresses that noise by an order of
        magnitude.

        Positive result means the gap is shrinking.
        """
        self.speed_valid = False

        if min_samples is None:
            min_samples = 4
        if min_window_sec is None:
            min_window_sec = 0.6
        if max_speed is None:
            max_speed = 25.0

        if len(self.samples) < int(min_samples):
            self.closing_speed = 0.0
            return 0.0, False

        ts = np.fromiter((s[0] for s in self.samples), dtype=np.float64, count=len(self.samples))
        ds = np.fromiter((s[1] for s in self.samples), dtype=np.float64, count=len(self.samples))

        span = float(ts[-1] - ts[0])
        if span < float(min_window_sec):
            self.closing_speed = 0.0
            return 0.0, False

        # Recentre time for numerical stability: ts values grow without bound on
        # a long video and the variance computation loses precision.
        t_rel = ts - ts[-1]
        var = float(np.var(t_rel))
        if var < 1e-9:
            self.closing_speed = 0.0
            return 0.0, False

        slope = float(np.sum((t_rel - t_rel.mean()) * (ds - ds.mean())) / (var * len(ts)))
        intercept = float(ds.mean() - slope * t_rel.mean())

        speed = -slope                      # slope is d(distance)/dt
        speed -= float(ego_bias_mps or 0.0) # remove camera motion
        speed = float(np.clip(speed, -float(max_speed), float(max_speed)))

        # Distance implied by the SAME fitted line, evaluated at the newest sample.
        # Using this for TTC keeps range and rate internally consistent: the EMA
        # lags a ramp, so pairing a lagged distance with an unlagged slope would
        # report a systematically too-long TTC.
        fitted = intercept
        if ego_bias_mps:
            fitted = intercept - float(ego_bias_mps) * (span * 0.5)
        self.fit_distance = fitted

        # Extra EMA on the speed to stop TTC flickering frame to frame.
        if ema_alpha is None:
            ema_alpha = 0.5
        if self.speed_ema is None:
            self.speed_ema = speed
        else:
            a = max(0.0, min(1.0, float(ema_alpha)))
            self.speed_ema = a * speed + (1.0 - a) * self.speed_ema

        self.closing_speed = float(self.speed_ema)
        self.ego_compensated = float(ego_bias_mps or 0.0)
        self.speed_valid = True
        return self.closing_speed, True

    def time_to_collision(self):
        """TTC in seconds, or infinity.

        TTC = distance / closing_speed, defined only for a genuinely closing
        target. A non-positive closing speed means the gap is not shrinking, so
        TTC is infinite rather than zero or negative.
        """
        if not self.speed_valid:
            return float("inf"), False
        distance = self.fit_distance if self.fit_distance is not None else self.distance
        if distance is None or not np.isfinite(distance) or distance <= 0:
            return float("inf"), False
        if self.closing_speed <= 1e-3:
            return float("inf"), False
        return float(distance / self.closing_speed), True

    def interpolated_distance(self, timestamp, mode="linear", max_rate_mps=None):
        """Estimate distance for a frame that had no depth measurement.

        Linear extrapolation from the last two samples fills the gaps between
        depth runs (interval > 1), which keeps closing-speed windows dense
        enough to fit. The result is labelled "interpolated" in the output,
        never as a measurement.

        `max_rate_mps` guards the failure mode: two noisy samples 5 frames apart
        can imply an absurdly steep ramp. If the implied rate exceeds it, return
        None so the caller falls back to holding the last good value, which is
        more honest than a confident wrong number.
        """
        if len(self.samples) < 2:
            if self.samples:
                return float(self.samples[-1][1]), "held"
            return None, "none"

        (t1, d1), (t2, d2) = self.samples[-2], self.samples[-1]
        dt = t2 - t1
        if dt <= 1e-6:
            return float(d2), "held"

        if mode == "hold":
            return float(d2), "held"

        rate = (d2 - d1) / dt
        if max_rate_mps is not None and abs(rate) > float(max_rate_mps):
            return float(d2), "held"

        predicted = d2 + rate * (timestamp - t2)
        if not np.isfinite(predicted) or predicted <= 0:
            return float(d2), "held"
        return float(predicted), "interpolated"


class EgoMotionEstimator:
    """Estimates ego-vehicle speed from optical flow of static background.

    Model
    -----
    For pure forward translation of a static scene, a pixel at row offset
    ``yr = y - cy`` shifts by

        dy = yr * v_ego * dt / Z

    so  v_ego = Z * dy / (yr * dt).

    What is deliberately NOT in that formula: a focal length. The pinhole
    projection u = f*X/Z already folded f into the pixel coordinate. Multiplying
    by f again double-counts it and inflates the estimate by roughly 1300x on a
    1080p frame. This is an easy mistake to make and looks plausible because it
    still produces numbers.

    Robustness
    ----------
    * Only points well below the horizon are used. Near the principal point
      yr -> 0, so inverting it explodes.
    * Only distant background (depth above a threshold): near road pixels have
      huge flow that swamps the estimate, and moving vehicles break the
      static-scene assumption.
    * One least-squares fit across all usable points, then one outlier rejection
      pass (MAD), then a refit. More stable than a median of per-point ratios.
    * Depth is read per point, never as one median for the whole frame.
    """

    def __init__(self, enabled=True, background_depth_m=15.0, min_row_offset_ratio=0.12,
                 smoothing=0.6, max_accel_mps=8.0, max_speed_mps=33.0):
        self.enabled = enabled
        self.background_depth_m = float(background_depth_m)
        self.min_row_offset_ratio = float(min_row_offset_ratio)
        self.smoothing = max(0.0, min(1.0, float(smoothing)))
        self.max_accel_mps = float(max_accel_mps)
        self.max_speed_mps = float(max_speed_mps)

        self.prev_gray = None
        self.last_time = None
        self.speed_mps = 0.0
        self.bias_mps = 0.0
        self.quality = 0.0

    def reset(self):
        self.prev_gray = None
        self.last_time = None
        self.speed_mps = 0.0
        self.bias_mps = 0.0
        self.quality = 0.0

    def update(self, frame, timestamp, depth_estimator, depth_map, bias_gain=1.0):
        """Estimate ego speed in m/s. Positive means moving forward."""
        if not self.enabled or frame is None or depth_map is None:
            self.speed_mps = 0.0
            self.bias_mps = 0.0
            return 0.0

        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray, prev_time = self.prev_gray, self.last_time
        self.prev_gray = gray
        self.last_time = timestamp

        if prev_gray is None or prev_time is None or prev_gray.shape != gray.shape:
            self.speed_mps = 0.0
            self.bias_mps = 0.0
            return 0.0

        dt = timestamp - prev_time
        if dt <= 1e-3:
            self.speed_mps = 0.0
            self.bias_mps = 0.0
            return 0.0

        H, W = gray.shape
        cy = H * 0.5

        grid_x = np.linspace(int(W * 0.12), int(W * 0.88), 20)
        grid_y = np.linspace(int(H * 0.15), int(H * 0.95), 12)
        p0 = np.array([[x, y] for y in grid_y for x in grid_x],
                      dtype=np.float32).reshape(-1, 1, 2)

        try:
            p1, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, p0, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
        except Exception:
            self.speed_mps = 0.0
            self.bias_mps = 0.0
            return 0.0

        if p1 is None or status is None:
            self.speed_mps = 0.0
            self.bias_mps = 0.0
            return 0.0

        ok = status.reshape(-1).astype(bool)
        if int(ok.sum()) < 20:
            self.speed_mps = 0.0
            self.bias_mps = 0.0
            return 0.0

        p0f = p0.reshape(-1, 2)[ok]
        p1f = p1.reshape(-1, 2)[ok]
        dy = p1f[:, 1] - p0f[:, 1]
        yr = p0f[:, 1] - cy

        depths = np.array(
            [depth_estimator.sample_point(px, py, radius=5, depth_map=depth_map)
             for px, py in p0f],
            dtype=np.float64,
        )

        valid = np.isfinite(depths) & (depths > 0.0)
        valid &= depths >= self.background_depth_m
        valid &= np.abs(yr) > (H * self.min_row_offset_ratio)

        self.quality = float(valid.mean())
        if int(valid.sum()) < 12:
            self.speed_mps = 0.0
            self.bias_mps = 0.0
            return 0.0

        Z = depths[valid]
        yv = yr[valid]
        dv = dy[valid]

        coeff = (yv * dt) / Z
        denom = float(np.sum(coeff * coeff))
        if denom < 1e-12:
            self.speed_mps = 0.0
            self.bias_mps = 0.0
            return 0.0

        speed = float(np.sum(coeff * dv) / denom)

        resid = dv - coeff * speed
        centre = float(np.median(resid))
        mad = float(np.median(np.abs(resid - centre)))
        if mad > 1e-9:
            keep = np.abs(resid - centre) <= 3.0 * mad
            if int(keep.sum()) >= 12:
                c2, d2 = coeff[keep], dv[keep]
                denom2 = float(np.sum(c2 * c2))
                if denom2 > 1e-12:
                    speed = float(np.sum(c2 * d2) / denom2)

        speed = float(np.clip(speed, 0.0, self.max_speed_mps))

        prev_speed = self.speed_mps
        if prev_speed > 0.0:
            max_delta = self.max_accel_mps * dt + 0.5
            speed = float(np.clip(speed, prev_speed - max_delta, prev_speed + max_delta))
            self.speed_mps = self.smoothing * speed + (1.0 - self.smoothing) * prev_speed
        else:
            self.speed_mps = speed

        # A static object shows apparent closing equal to ego speed, first order.
        self.bias_mps = self.speed_mps * float(bias_gain or 1.0)
        return self.speed_mps


def build_tracker(cfg, enabled=True, **kwargs):
    """Construct an EgoMotionEstimator from config."""
    return EgoMotionEstimator(
        enabled=enabled,
        background_depth_m=_f(cfg, "ego_motion.background_depth_m", 15.0),
        min_row_offset_ratio=_f(cfg, "ego_motion.min_row_offset_ratio", 0.12),
        smoothing=_f(cfg, "ego_motion.smoothing", 0.6),
        max_accel_mps=_f(cfg, "ego_motion.max_accel_mps", 8.0),
        max_speed_mps=_f(cfg, "ego_motion.max_speed_mps", 33.0),
        **kwargs,
    )