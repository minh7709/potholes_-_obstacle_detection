"""Pipeline orchestration: the frame loop that ties every stage together.

Per-frame cost control
----------------------
Two independent switches, because they buy different things:

* Frame decimation (``processing.target_fps``) reduces how OFTEN the whole
  pipeline runs. Cheap on accuracy for the distance estimate: the tracker's
  history still spans real time, so the closing-speed fit is unaffected.
* ``depth.interval`` reduces how OFTEN the expensive depth model runs. This is
  where nearly all the time goes (285 ms vs 55 ms on an i5-8250U), so this is
  the first lever to turn.

Between real depth inferences the pipeline fills frames by linear interpolation
and marks the value as ``interpolated`` so the CSV never passes it off as a
measurement.

Ego-motion, and what "closing speed" means
-----------------------------------------
``ego_motion.enabled`` subtracts an ESTIMATED ego speed from the apparent
closing rate. The residual ``closing_speed`` is therefore a RELATIVE quantity
approximating the target's own motion, not its true absolute speed. A single
camera cannot separate the two. TTC inherits that limitation.

Road safety relevance
---------------------
Sound is tuned to be distinctive under road noise, and TTS-free tones avoid a
dependency in a safety-relevant path. But a prototype on CPU at a few FPS is
NOT a production ADAS: latency is far too high to react to. Treat it as
algorithm validation, not a safety device.
"""

import os
import time

import numpy as np

from app.fcw import risk as risk_mod
from app.fcw.audio import AudioAlerter
from app.fcw.config_loader import (
    depth_height,
    depth_interval,
    depth_interpolation_mode,
    primary_classes,
    target_classes,
    target_fps,
)
from app.fcw.depth import DepthCalibration, DepthEstimator
from app.fcw.detector import VehicleDetector
from app.fcw.frame_sampler import FrameSampler
from app.fcw.monitor import PerformanceMonitor
from app.fcw.tracking import EgoMotionEstimator, TrackHistory
from app.fcw.visualizer import Visualizer

NORMAL = risk_mod.NORMAL
WARNING = risk_mod.WARNING
DANGER = risk_mod.DANGER


def _f(cfg, key, default):
    try:
        v = cfg.get(key, default, cast=float)
        return default if v is None else float(v)
    except Exception:
        return default


def _i(cfg, key, default):
    try:
        v = cfg.get(key, default, cast=int)
        return default if v is None else int(v)
    except Exception:
        return default


class CollisionWarningSystem:
    """The whole per-frame pipeline.

    ``process_frame`` advances exactly one frame and returns the annotated image
    plus a result dict. Keeping it frame-at-a-time (rather than looping
    internally) lets the same class serve a video file, a live camera and the
    web API unchanged.
    """

    def __init__(self, cfg, detector=None, depth_estimator=None,
                 alerter=None, visualizer=None):
        self.cfg = cfg
        self.source_label = ""

        self.history_len = _i(cfg, "distance.history_length", 20)
        self.smoothing = cfg.get("distance.smoothing", "ema")
        self.alpha = _f(cfg, "distance.alpha", 0.3)

        self.depth_interval = depth_interval(cfg)
        self.interp_mode = depth_interpolation_mode(cfg)
        self.max_interp_rate = _f(cfg, "depth.max_interpolated_speed_mps", 12.0)

        self.warning_ttc = _f(cfg, "collision.warning_ttc", 4.0)
        self.danger_ttc = _f(cfg, "collision.danger_ttc", 2.0)
        self.confirm_frames = _i(cfg, "collision.confirm_frames", 3)
        self.min_hits = _i(cfg, "tracking.min_hits", 3)
        self.max_age = _i(cfg, "tracking.max_age", 30)

        self.min_speed_samples = _i(cfg, "speed.min_samples", 4)
        self.min_speed_window = _f(cfg, "speed.min_window_sec", 0.6)
        self.max_closing_speed = _f(cfg, "speed.max_closing_speed_mps", 25.0)
        self.speed_ema_alpha = _f(cfg, "speed.ema_alpha", 0.5)
        self.ego_interval = max(1, _i(cfg, "ego_motion.interval", 1))
        self.render_enabled = bool(cfg.get("visual.render", True))

        self.primary_classes = primary_classes(cfg)

        self.detector = detector or VehicleDetector(
            model_name=cfg.get("classes.yolo_model", "assets/best.pt"),
            device=cfg.get("performance.device", "auto"),
            conf=_f(cfg, "classes.conf_threshold", 0.45),
            iou=_f(cfg, "classes.iou_threshold", 0.5),
            classes=target_classes(cfg),
            tracker=cfg.get("tracking.tracker", "bytetrack.yaml"),
            track_buffer=_i(cfg, "tracking.buffer", 30),
            min_area_ratio=_f(cfg, "classes.min_area_ratio", 0.0008),
        )

        self.depth = depth_estimator or DepthEstimator(
            model_id=cfg.get("depth.model"),
            device=cfg.get("performance.device", "auto"),
            processing_height=depth_height(cfg),
            cfg=cfg,
        )
        self.calibration = DepthCalibration.from_reference(
            ref_depth_m=_f(cfg, "calibration.calib_model_m", 10.0),
            ref_true_m=_f(cfg, "calibration.calib_true_m", 10.0),
            gain=_f(cfg, "calibration.depth_scale", 1.0),
        )

        audio_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "assets", "audio")
        self.alerter = alerter if alerter is not None else AudioAlerter(
            enabled=bool(cfg.get("warning.audio_enabled", True)),
            volume=_f(cfg, "warning.volume", 0.8),
            warning_path=os.path.join(audio_dir, "warning.wav"),
            danger_path=os.path.join(audio_dir, "danger.wav"),
            warning_cooldown=_f(cfg, "warning.warning_cooldown_sec", 1.5),
            danger_cooldown=_f(cfg, "warning.danger_cooldown_sec", 1.0),
        )

        self.visualizer = visualizer if visualizer is not None else Visualizer(cfg)

        self.histories = {}
        self.ego = EgoMotionEstimator(
            enabled=bool(cfg.get("ego_motion.enabled", True)),
            background_depth_m=_f(cfg, "ego_motion.background_depth_m", 15.0),
            min_row_offset_ratio=_f(cfg, "ego_motion.min_row_offset_ratio", 0.12),
            smoothing=_f(cfg, "ego_motion.smoothing", 0.6),
            max_accel_mps=_f(cfg, "ego_motion.max_accel_mps", 8.0),
            max_speed_mps=_f(cfg, "ego_motion.max_speed_mps", 33.0),
        )

        self.frame_count = 0
        self.depth_runs = 0
        self._frames_since_depth = 0
        self._frames_since_ego = 0
        self._confirm_state = {}
        self._last_depth_time = None
        self._synthetic_id = 0

    # ── Core loop ────────────────────────────────────────────────────
    def process_frame(self, frame, timestamp, source_frame=None,
                      force_depth=False, depth_map=None):
        """Run one frame through the pipeline.

        `timestamp` MUST be the true capture time of the frame. Depth runs only
        when its interval has elapsed (or `force_depth` is set), and on those
        frames every object gets a real measurement; otherwise objects fall back
        to interpolation from the previous two measurements.
        """
        if frame is None or frame.size == 0:
            return frame, self._empty_result()

        now = float(timestamp)
        self.frame_count += 1
        source_frame = self.frame_count if source_frame is None else source_frame

        # 1. Detection + tracking (cheap, runs every processed frame)
        detections = self.detector.detect(frame)

        # 2. Depth, on its own slower cadence.
        #
        # The gate counts FRAMES SINCE THE LAST DEPTH RUN. Counting
        # `depth_runs % interval` instead would deadlock after the first run,
        # because the counter could never reach a second multiple of the
        # interval -- the symptom being distance that silently freezes.
        due = (force_depth
               or self.depth_runs == 0
               or (self._frames_since_depth + 1) >= self.depth_interval)
        depth_computed = False
        self._frames_since_depth += 1
        if due and (force_depth or self._has_measurable_content(detections)):
            depth_map = self.depth.infer(frame)
            if depth_map is not None:
                self.depth_runs += 1
                self._frames_since_depth = 0
                depth_computed = True
                self._last_depth_time = now
        elif depth_map is None:
            depth_map = self.depth.last_depth_map

        # 3. Ego motion from background optical flow.
        #
        # Optical flow is the third most expensive stage, and ego speed is
        # heavily smoothed and changes slowly, so it runs on its own (slower)
        # cadence and holds its value in between. The returned bias stays
        # constant between updates, which is exactly what the speed regression
        # subtracts, so a stale value only biases the correction, never the
        # measurement itself.
        if (depth_computed
                or self.ego.last_time is None
                or (self._frames_since_ego + 1) >= self.ego_interval):
            ego_speed = self.ego.update(
                frame, now, self.depth, depth_map,
                bias_gain=_f(self.cfg, "ego_motion.bias_gain", 1.0))
            self._frames_since_ego = 0
        else:
            ego_speed = self.ego.speed_mps
        self._frames_since_ego += 1

        # 4. Per-object distance, kinematics and risk
        height, width = frame.shape[:2]
        roi = risk_mod.road_roi(frame.shape, self.cfg)
        # Narrower band where a warning may actually be raised. The wide ROI
        # still gates which objects are tracked at all.
        corridor = risk_mod.warning_corridor(frame.shape, self.cfg)
        records = []
        seen_ids = set()

        for det in detections:
            history = self._history_for(det)
            distance, source = self._distance_for(det, history, now, depth_map, depth_computed)

            history.update(now, distance, det["box"], det["confidence"],
                           det["class_name"], source=source)
            history.estimate_speed(
                ego_bias_mps=self.ego.bias_mps,
                min_samples=self.min_speed_samples,
                min_window_sec=self.min_speed_window,
                max_speed=self.max_closing_speed,
                ema_alpha=self.speed_ema_alpha,
            )

            risk = risk_mod.compute_risk(det, history, roi, width, self.cfg,
                                          corridor=corridor)
            records.append({"detection": det, "history": history, "risk": risk})

            if det["track_id"] is not None:
                seen_ids.add(det["track_id"])

        self._prune(now, seen_ids)

        # 5. One target drives the audio
        primary = risk_mod.select_primary_target(
            records, frame.shape, self.cfg, self.primary_classes)
        primary = self._confirm(primary)

        if primary is not None:
            self.alerter.play(primary["risk"]["level"], now)

        # 6. Render (skippable for pure telemetry runs)
        if self.render_enabled:
            annotated = self.visualizer.render(
                frame, records, primary,
                ego_speed=ego_speed,
                detector=self.detector,
                depth=self.depth,
                depth_computed=depth_computed,
                source_label=self.source_label,
                level=primary["risk"]["level"] if primary else NORMAL,
                corridor=corridor,
            )
        else:
            annotated = frame

        return annotated, self._result(records, primary, ego_speed, now,
                                      source_frame, depth_computed)

    # ── Distance acquisition ─────────────────────────────────────────
    def _has_measurable_content(self, detections):
        """Skip a depth run when nothing is worth measuring.

        Ego-motion also needs depth, but only when there are still background
        points to track, so an entirely empty frame is not worth 285 ms.
        """
        return bool(detections)

    def _distance_for(self, det, history, now, depth_map, depth_computed):
        """Measured distance when depth ran, else interpolated/held."""
        if depth_computed and depth_map is not None:
            distance, _samples = self.depth.sample_box(det["box"], depth_map, cfg=self.cfg)
            if distance is not None:
                return distance, "measured"

        estimated, source = history.interpolated_distance(
            now, self.interp_mode, self.max_interp_rate)
        if estimated is None and history.distance is not None:
            return history.distance, "held"
        return estimated, source

    # ── History bookkeeping ──────────────────────────────────────────
    def _history_for(self, det):
        track_id = det.get("track_id")
        if track_id is None:
            # Unconfirmed track: give it a short-lived synthetic id so it still
            # gets smoothing, without polluting real tracker ids.
            self._synthetic_id += 1
            track_id = -self._synthetic_id
        history = self.histories.get(track_id)
        if history is None:
            history = TrackHistory(
                track_id=track_id, class_name=det["class_name"],
                history_len=self.history_len, smoothing=self.smoothing,
                alpha=self.alpha)
            self.histories[track_id] = history
        return history

    def _prune(self, now, seen_ids):
        """Drop histories for tracks that have gone stale.

        This is what bounds memory: without it, histories would accumulate for
        every object ever seen, which on a long video is thousands of entries
        holding deques.
        """
        stale = [tid for tid, hist in self.histories.items()
                 if (now - hist.last_seen) > (self.max_age * 0.5)
                 or (tid >= 0 and tid not in seen_ids)]
        for tid in stale:
            self.histories.pop(tid, None)
            self._confirm_state.pop(tid, None)

    def _confirm(self, primary):
        """Require N consecutive frames at the same level before alarming."""
        if primary is None:
            self._confirm_state.clear()
            return None

        level = primary["risk"]["level"]
        history = primary["history"]
        if history.frames_seen < max(self.min_hits, 1):
            return None

        key = primary["detection"]["track_id"] or history.track_id
        previous, count = self._confirm_state.get(key, (None, 0))
        count = count + 1 if previous == level else 1
        self._confirm_state[key] = (level, count)
        return primary if count >= self.confirm_frames else None

    # ── Result assembly ──────────────────────────────────────────────
    def _result(self, records, primary, ego_speed, now, source_frame, depth_computed):
        objects = [self._object_summary(r) for r in records]
        return {
            "frame": source_frame,
            "timestamp": now,
            "detections": len(records),
            "tracks": len([r for r in records if r["detection"].get("track_id") is not None]),
            "ego_speed_mps": ego_speed,
            "objects": objects,
            "primary": self._object_summary(primary) if primary else None,
            "level": primary["risk"]["level"] if primary else NORMAL,
            "depth_computed": depth_computed,
            "depth_interval": self.depth_interval,
            "depth_runs": self.depth_runs,
            "calibration": self.calibration.describe(),
            "yolo_ms": self.detector.inference_ms,
            "depth_ms": self.depth.inference_ms if depth_computed else 0.0,
        }

    def _object_summary(self, record):
        if record is None:
            return None
        det, hist, risk = record["detection"], record["history"], record["risk"]
        ttc = risk["ttc"]
        return {
            "track_id": det.get("track_id"),
            "class": det.get("class_name"),
            "confidence": round(det.get("confidence", 0.0), 3),
            "bbox": [round(v, 1) for v in det.get("box", (0, 0, 0, 0))],
            "distance_m": _round(hist.distance, 2),
            "raw_distance_m": _round(hist.raw_distance, 2),
            "distance_source": risk.get("distance_source", "none"),
            "closing_speed_mps": (round(hist.closing_speed, 2)
                                  if hist.speed_valid else None),
            "ttc_s": (round(ttc, 2) if ttc is not None and np.isfinite(ttc) else None),
            "level": risk["level"],
            # Why it warned. Range and TTC are independent triggers, and
            # "TTC inf but DANGER" is only explicable once you know the distance
            # rule fired.
            "level_from_ttc": risk.get("level_from_ttc"),
            "level_from_distance": risk.get("level_from_distance"),
            "proximity_trigger": bool(risk.get("proximity_trigger")),
            "in_roi": risk["in_roi"],
            "in_warning_corridor": risk.get("in_warning_corridor"),
            "level_unrestricted": risk.get("level_unrestricted"),
            "frames_seen": hist.frames_seen,
        }

    def _empty_result(self):
        return {
            "frame": self.frame_count, "timestamp": 0.0, "detections": 0, "tracks": 0,
            "ego_speed_mps": 0.0, "objects": [], "primary": None, "level": NORMAL,
            "depth_computed": False, "depth_interval": self.depth_interval,
            "calibration": self.calibration.describe(),
            "yolo_ms": 0.0, "depth_ms": 0.0,
        }

    def set_performance(self, target_fps_value, depth_interval_value):
        """Apply an adaptive step. Called by the runner, never by the loop."""
        if target_fps_value is not None:
            self.target_fps_override = float(target_fps_value)
        if depth_interval_value:
            self.depth_interval = max(1, int(depth_interval_value))

    def reset(self):
        """Clear all temporal state. Required between videos."""
        self.histories.clear()
        self._confirm_state.clear()
        self.ego.reset()
        self.depth.clear_cache()
        self.frame_count = 0
        self.depth_runs = 0
        self._frames_since_depth = 0
        self._frames_since_ego = 0
        self._last_depth_time = None
        self._synthetic_id = 0
        if self.detector is not None and hasattr(self.detector, "reset"):
            self.detector.reset()
        if self.alerter is not None and hasattr(self.alerter, "reset"):
            self.alerter.reset()

    def close(self):
        if self.alerter is not None:
            self.alerter.close()


def _round(value, digits):
    if value is None:
        return None
    try:
        if not np.isfinite(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None