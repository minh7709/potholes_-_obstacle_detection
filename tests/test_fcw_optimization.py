"""Tests for the optimisation layer: decimation, depth interval, interpolation,
smoothing, output and adaptive control.

These verify the things that are easy to get subtly wrong and hard to notice:
timestamp correctness under decimation, memory bounds on track history, and
that adaptive control never touches safety thresholds.
"""

import csv
import json
import os

import numpy as np
import pytest

from app.fcw.config_loader import Config, load_config, parse_set_overrides
from app.fcw.frame_sampler import FrameSampler, LiveFrameSampler
from app.fcw.monitor import AdaptiveController, PerformanceMonitor
from app.fcw.output import CSV_COLUMNS, ResultWriter
from app.fcw.risk import DANGER, NORMAL, WARNING
from app.fcw.tracking import TrackHistory


# ── Config ───────────────────────────────────────────────────────────
def test_config_yaml_loads_with_expected_defaults():
    cfg = load_config()
    assert cfg.get("processing.target_fps", cast=float) == 10
    assert cfg.get("depth.interval", cast=int) == 5
    assert cfg.get("depth.processing_height", cast=int) == 224
    assert cfg.get("collision.warning_ttc", cast=float) == 4.0
    assert cfg.get("collision.danger_ttc", cast=float) == 2.0
    assert cfg.get("input.max_width", cast=int) == 1280


def test_config_dotted_access_and_defaults():
    cfg = Config({"depth": {"interval": 3}})
    assert cfg.get("depth.interval") == 3
    assert cfg.get("depth.missing", 99) == 99
    assert cfg.get("nothing.at.all", None) is None


def test_config_set_creates_nested_path():
    cfg = Config({})
    cfg.set("a.b.c", 5)
    assert cfg.get("a.b.c") == 5


def test_parse_set_overrides_types():
    out = parse_set_overrides(["depth.interval=3", "collision.warning_ttc=5.5",
                               "warning.audio_enabled=false", "classes.targets=motorcycle,car"])
    assert out["depth"]["interval"] == 3
    assert isinstance(out["depth"]["interval"], int)
    assert out["collision"]["warning_ttc"] == 5.5
    assert out["warning"]["audio_enabled"] is False
    assert out["classes"]["targets"] == ["motorcycle", "car"]


def test_env_override_beats_yaml(monkeypatch):
    monkeypatch.setenv("FCW_DEPTH_INTERVAL", "9")
    cfg = load_config()
    assert cfg.get("depth.interval", cast=int) == 9


# ── Frame decimation ─────────────────────────────────────────────────
def test_60fps_to_10fps_uses_every_sixth_frame():
    """60 -> 10 FPS must sample original frames 0, 6, 12 ..."""
    sampler = FrameSampler(input_fps=60, target_fps=10)
    assert sampler.frame_step == 6
    processed = [i for i in range(24) if sampler.should_process()]
    assert processed == [0, 6, 12, 18]


def test_30fps_to_10fps_uses_every_third_frame():
    sampler = FrameSampler(input_fps=30, target_fps=10)
    assert sampler.frame_step == 3
    processed = [i for i in range(12) if sampler.should_process()]
    assert processed == [0, 3, 6, 9]


def test_no_decimation_when_target_exceeds_input():
    sampler = FrameSampler(input_fps=30, target_fps=60)
    assert sampler.frame_step == 1
    assert all(sampler.should_process() for _ in range(10))


def test_null_target_fps_processes_every_frame():
    sampler = FrameSampler(input_fps=60, target_fps=None)
    assert sampler.frame_step == 1


def test_timestamps_are_preserved_under_decimation():
    """TTC divides by dt, so timestamps must follow the ORIGINAL frame index.

    This is the core invariant: decimating to 10 FPS must not re-time the
    samples as though the source were 10 FPS.
    """
    sampler = FrameSampler(input_fps=60, target_fps=10)
    times = []
    for i in range(60):
        if sampler.should_process():
            times.append(sampler.timestamp_for(i))
    # Original frames 0,6,12,... at 60 FPS => 0.0, 0.1, 0.2 s
    assert times[:3] == pytest.approx([0.0, 0.1, 0.2])
    assert times[-1] == pytest.approx(0.9, abs=1e-6)


def test_timestamps_recover_correct_duration():
    sampler = FrameSampler(input_fps=60, target_fps=10)
    # 5400 frames at 60 FPS is 90 s. Frame index 5400 is the start of the next
    # second, so its timestamp is exactly 90.0.
    assert sampler.timestamp_for(5400) == pytest.approx(90.0)
    assert sampler.timestamp_for(5394) == pytest.approx(89.9, abs=0.01)


def test_resize_preserves_aspect_ratio():
    import cv2
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    sampler = FrameSampler(30, 10, max_width=1280, max_height=720)
    out = sampler.resize(frame)
    h, w = out.shape[:2]
    assert w <= 1280 and h <= 720
    assert abs((w / h) - (3840 / 2160)) < 0.02


def test_resize_skips_when_already_small():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sampler = FrameSampler(30, 10, max_width=1280, max_height=720)
    assert sampler.resize(frame) is frame


def test_scale_box_maps_between_resolutions():
    sampler = FrameSampler(30, 10)
    box = (100, 200, 300, 400)
    scaled = sampler.scale_box(box, (720, 1280), (360, 640))
    assert scaled == pytest.approx((50, 100, 150, 200))
    back = sampler.scale_box(scaled, (360, 640), (720, 1280))
    assert back == pytest.approx(box)


def test_sampler_stats_add_up():
    sampler = FrameSampler(input_fps=30, target_fps=10)
    for _ in range(30):
        sampler.should_process()
    stats = sampler.stats()
    assert stats["source_frames"] == 30
    assert stats["processed_frames"] == 10
    assert stats["skipped_frames"] == 20
    assert stats["frame_step"] == 3


def test_live_sampler_is_time_based():
    sampler = LiveFrameSampler(input_fps=30, target_fps=10, max_width=1280, max_height=720)
    # interval should be 0.1 s
    assert sampler.should_process(0.0) is True
    assert sampler.should_process(0.01) is False
    assert sampler.should_process(0.05) is False
    assert sampler.should_process(0.11) is True


# ── Depth sampling robustness ────────────────────────────────────────
def _estimator():
    from app.fcw.depth import DepthEstimator
    est = DepthEstimator.__new__(DepthEstimator)
    est.cfg = None
    est._last_raw = None
    return est


def test_depth_sampling_rejects_nan_and_inf():
    """NaN/inf must never reach the median, or the distance becomes garbage."""
    est = _estimator()
    depth = np.full((480, 640), 10.0, dtype=np.float32)
    # Corrupt a minority of pixels inside the box, not all of them: a fully
    # invalid region must return None (covered by a separate test).
    depth[200:215, 280:300] = np.nan
    depth[265:275, 330:355] = np.inf
    depth[230:240, 300:315] = -np.inf
    est._last_raw = depth
    distance, samples = est.sample_box((280, 200, 360, 280), depth)
    assert distance is not None
    assert 8.0 < distance < 12.0
    assert samples > 0


def test_depth_centre_crop_ignores_background():
    """A mostly-near centre with far corners must report the near value."""
    est = _estimator()
    depth = np.full((480, 640), 90.0, dtype=np.float32)
    depth[180:300, 200:440] = 12.0   # central object, far background around it
    est._last_raw = depth
    distance, _ = est.sample_box((160, 120, 480, 320), depth, center_crop=0.4)
    assert distance is not None
    assert 9.0 < distance < 16.0, f"expected ~12 m, got {distance}"


def test_depth_returns_none_for_out_of_range():
    est = _estimator()
    depth = np.full((480, 640), 500.0, dtype=np.float32)
    est._last_raw = depth
    distance, _ = est.sample_box((100, 100, 300, 300), depth)
    assert distance is None


def test_depth_returns_none_for_tiny_box():
    est = _estimator()
    est._last_raw = np.full((480, 640), 10.0, dtype=np.float32)
    distance, _ = est.sample_box((100, 100, 101, 101), est._last_raw)
    assert distance is None


def test_depth_all_nan_patch_returns_none():
    est = _estimator()
    est._last_raw = np.full((480, 640), np.nan, dtype=np.float32)
    distance, _ = est.sample_box((100, 100, 300, 300), est._last_raw)
    assert distance is None


def test_has_fresh_map_detects_resolution_change():
    """A stale map at the wrong resolution would report plausible wrong metres."""
    est = _estimator()
    est._last_raw = np.zeros((720, 1280), dtype=np.float32)
    est._last_source_shape = (720, 1280)
    assert est.has_fresh_map((720, 1280, 3)) is True
    assert est.has_fresh_map((2160, 3840, 3)) is False


# ── Interpolation ────────────────────────────────────────────────────
def _history_with(samples):
    h = TrackHistory(track_id=1, history_len=20)
    for t, d in samples:
        h.update(t, d)
    return h


def test_linear_interpolation_between_depth_samples():
    """Fills frames between depth runs so the speed window stays dense."""
    h = _history_with([(0.0, 10.0), (0.5, 9.0)])
    distance, source = h.interpolated_distance(0.25, mode="linear")
    assert distance == pytest.approx(9.5, abs=0.01)
    assert source == "interpolated"


def test_hold_mode_returns_last_value():
    h = _history_with([(0.0, 10.0), (0.5, 9.0)])
    distance, source = h.interpolated_distance(0.25, mode="hold")
    assert distance == pytest.approx(9.0)
    assert source == "held"


def test_interpolation_falls_back_to_hold_when_rate_implausible():
    """Two noisy samples must not produce a confident absurd ramp."""
    h = _history_with([(0.0, 30.0), (0.5, 4.0)])  # 52 m/s implied
    distance, source = h.interpolated_distance(0.25, mode="linear",
                                               max_rate_mps=12.0)
    assert source == "held"
    assert distance == pytest.approx(4.0)


def test_interpolation_with_no_history():
    h = TrackHistory(track_id=1)
    distance, source = h.interpolated_distance(1.0)
    assert distance is None
    assert source == "none"


def test_distance_source_is_tracked():
    h = _history_with([(0.0, 10.0)])
    assert h.distance_source == "measured"
    h.update(0.1, 9.5, source="interpolated")
    assert h.distance_source == "interpolated"


# ── Smoothing ────────────────────────────────────────────────────────
def test_ema_smooths_noise():
    """Raw readings jitter; EMA must reduce the spread of the output."""
    noisy = [8.1, 7.7, 8.0, 7.4, 7.8]
    h = TrackHistory(track_id=1, alpha=0.3)
    smoothed = []
    for i, d in enumerate(noisy):
        h.update(i * 0.2, d)
        smoothed.append(h.distance)
    raw_spread = max(noisy) - min(noisy)
    smooth_spread = max(smoothed) - min(smoothed)
    assert smooth_spread < raw_spread, "EMA did not reduce spread"
    # First sample seeds the EMA directly, so smoothing starts without a jump.
    assert smoothed[0] == pytest.approx(noisy[0])
    assert smoothed[-1] == pytest.approx(7.7, abs=0.35)


def test_moving_average_alternative():
    h = TrackHistory(track_id=1, smoothing="moving_average", alpha=0.3, window=3)
    for i, d in enumerate([8.1, 7.7, 8.0, 7.4]):
        h.update(i * 0.2, d)
    assert h.distance is not None


def test_speed_ema_reduces_flicker():
    """TTC flicker is the symptom this guards against."""
    h = TrackHistory(track_id=1, alpha=0.5)
    t = 0.0
    speeds = []
    # Alternate steep and shallow ramps to create a noisy slope sequence.
    for i in range(24):
        d = 10.0 - 0.30 * i - (0.25 if i % 2 else 0.0)
        h.update(t, d)
        h.estimate_speed(ema_alpha=0.35)
        # speed_ema stays None during the warm-up (fewer than min_samples).
        if h.speed_ema is not None:
            speeds.append(h.speed_ema)
        t += 0.25
    assert len(speeds) >= 15, f"too few speed samples: {len(speeds)}"
    deltas = [abs(b - a) for a, b in zip(speeds, speeds[1:])]
    assert max(deltas) < 2.0, f"speed EMA still flickers: {deltas}"


# ── History memory bound ─────────────────────────────────────────────
def test_history_deque_is_bounded():
    """RAM must not grow with video length."""
    h = TrackHistory(track_id=1, history_len=20)
    for i in range(5000):
        h.update(i * 0.1, 10.0)
    assert len(h.samples) <= 20


def test_history_rejects_invalid_distance():
    h = TrackHistory(track_id=1)
    assert h.update(0.0, None) is False
    assert h.update(0.0, float("nan")) is False
    assert h.update(0.0, -5.0) is False
    assert h.update(0.0, 8.0) is True


# ── Output writer ────────────────────────────────────────────────────
def test_csv_and_summary_written(tmp_path):
    writer = ResultWriter(str(tmp_path), save_video=False, save_csv=True,
                          save_summary=True).open()
    objs = [{"track_id": 5, "class": "motorcycle", "confidence": 0.94,
             "distance_m": 8.2, "distance_source": "measured",
             "closing_speed_mps": 1.4, "ttc_s": 5.86, "level": "NORMAL",
             "in_roi": True},
            {"track_id": 5, "class": "motorcycle", "confidence": 0.95,
             "distance_m": 1.9, "distance_source": "measured",
             "closing_speed_mps": 3.1, "ttc_s": 0.61, "level": "DANGER",
             "in_roi": True}]
    writer.write_frame(np.zeros((10, 10, 3), np.uint8), objs, 12.40, 372,
                       ego_speed=9.0, depth_computed=True)
    writer.write_frame(np.zeros((10, 10, 3), np.uint8), objs, 13.80, 414,
                       ego_speed=9.0, depth_computed=True)
    summary = writer.close()

    csv_path = os.path.join(str(tmp_path), "events.csv")
    assert os.path.exists(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert rows[0]["timestamp"] == "12.40"
    assert rows[0]["track_id"] == "5"
    assert rows[0]["class"] == "motorcycle"
    assert rows[0]["distance_m"] == "8.20"
    assert rows[0]["status"] == "NORMAL"
    assert rows[0]["in_roi"] == "1"
    assert rows[0]["depth_computed"] == "1"
    assert set(rows[0].keys()) == set(CSV_COLUMNS)

    with open(os.path.join(str(tmp_path), "summary.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["danger_frames"] == 2
    assert data["min_ttc_sec"] == 0.61
    assert data["min_distance_m"] == 1.9
    assert data["tracks"]["5"]["min_ttc_sec"] == 0.61


def test_csv_distinguishes_measured_from_interpolated(tmp_path):
    """Downstream must be able to tell an estimate from a measurement."""
    writer = ResultWriter(str(tmp_path), save_video=False, save_csv=True,
                          save_summary=False).open()
    objs = [{"track_id": 1, "class": "motorcycle", "confidence": 0.9,
             "distance_m": 7.0, "distance_source": "interpolated",
             "closing_speed_mps": 0.5, "ttc_s": 14.0, "level": "NORMAL",
             "in_roi": True}]
    writer.write_frame(None, objs, 1.0, 10)
    writer.close()
    with open(os.path.join(str(tmp_path), "events.csv"), newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["distance_source"] == "interpolated"


def test_writer_survives_empty_run(tmp_path):
    writer = ResultWriter(str(tmp_path), save_video=False, save_csv=True,
                          save_summary=True).open()
    summary = writer.close()
    assert summary["frames_processed"] == 0


# ── Performance monitor ──────────────────────────────────────────────
def test_speed_ratio_uses_analysed_span_not_full_duration():
    """A truncated run must not report inflated throughput.

    Analysing 10 s of a 100 s video and reporting >1x real time would be wrong:
    the remaining 90 s were never processed.
    """
    monitor = PerformanceMonitor().start()
    monitor.source_duration = 100.0          # full video
    monitor.mark_video_time(0.0)
    monitor.mark_video_time(10.0)            # only 10 s analysed
    monitor.elapsed = 4.0

    assert monitor.processed_duration() == pytest.approx(10.0)
    assert monitor.speed_ratio() == pytest.approx(2.5)   # 10 s in 4 s
    summary = monitor.summary()
    assert summary["truncated"] is True
    assert summary["video_duration_sec"] == 100.0
    assert summary["processed_duration_sec"] == 10.0

    monitor.elapsed = 40.0
    assert monitor.speed_ratio() == pytest.approx(0.25)


def test_speed_ratio_full_run():
    monitor = PerformanceMonitor().start()
    monitor.source_duration = 60.0
    monitor.mark_video_time(0.0)
    monitor.mark_video_time(60.0)
    monitor.elapsed = 30.0
    assert monitor.speed_ratio() == pytest.approx(2.0)
    assert monitor.summary()["truncated"] is False


def test_summary_reports_stage_metrics():
    monitor = PerformanceMonitor()
    monitor.source_duration = 60.0
    monitor.elapsed = 120.0
    monitor._processed = 60
    monitor.mark_video_time(0.0)
    monitor.mark_video_time(60.0)
    monitor.add_yolo(50.0)
    monitor.add_depth(285.0)
    summary = monitor.summary(source_fps=30, target_fps=10, frame_step=3,
                              input_size=(1280, 720))
    assert summary["yolo_ms"] == 50.0
    assert summary["yolo_fps"] == pytest.approx(20.0)
    assert summary["speed_ratio"] == pytest.approx(0.5)
    assert summary["input_size"] == "1280x720"


# ── Adaptive control ─────────────────────────────────────────────────
STEPS = [{"target_fps": 10, "depth_interval": 5},
         {"target_fps": 8, "depth_interval": 6},
         {"target_fps": 6, "depth_interval": 8},
         {"target_fps": 4, "depth_interval": 10}]


def test_adaptive_requires_sustained_low_fps():
    """One or two slow frames must not degrade quality.

    The first inference of a run, and any GC pause, is slow for reasons that
    have nothing to do with sustained load. Degrading on those would needlessly
    lower output quality.
    """
    ctrl = AdaptiveController(STEPS, min_fps=5.0, enabled=True, window=5,
                              cooldown_sec=0.0)
    # The controller enforces a minimum window of 3.
    assert ctrl.window == 5
    assert ctrl.maybe_degrade(2.0, now=0.0) is None       # 1st slow sample
    assert ctrl.maybe_degrade(2.0, now=1.0) is None       # still below window
    assert ctrl.maybe_degrade(2.0, now=2.0) is None       # still below window
    assert ctrl.maybe_degrade(2.0, now=3.0) is None       # still below window
    assert ctrl.level == 0, "degraded before the window was satisfied"
    step = ctrl.maybe_degrade(2.0, now=4.0)              # 5th sample
    assert step is not None
    assert ctrl.level == 1
    assert step["depth_interval"] == 6


def test_adaptive_does_not_trigger_above_floor():
    ctrl = AdaptiveController(STEPS, min_fps=5.0, enabled=True, window=5,
                              cooldown_sec=0.0)
    for i in range(20):
        assert ctrl.maybe_degrade(8.0, now=i) is None
    assert ctrl.level == 0


def test_adaptive_resets_streak_on_recovery():
    ctrl = AdaptiveController(STEPS, min_fps=5.0, enabled=True, window=5,
                              cooldown_sec=0.0)
    for _ in range(4):
        ctrl.maybe_degrade(2.0, now=0.0)
    ctrl.maybe_degrade(9.0, now=0.0)   # one good sample resets the streak
    assert ctrl._slow_samples == 0
    for _ in range(4):
        ctrl.maybe_degrade(2.0, now=0.0)
    assert ctrl.level == 0, "streak should have reset"


def test_adaptive_respects_cooldown():
    """After a step down, a second change is refused until the cooldown expires."""
    ctrl = AdaptiveController(STEPS, min_fps=5.0, enabled=True, window=3,
                              cooldown_sec=30.0)

    # 3 samples of sustained slowness -> level 1.
    ctrl.maybe_degrade(2.0, now=0.0)
    ctrl.maybe_degrade(2.0, now=0.1)
    assert ctrl.maybe_degrade(2.0, now=0.2) is not None
    assert ctrl.level == 1

    # The window is satisfied again but the cooldown is still open.
    ctrl.maybe_degrade(2.0, now=1.0)
    ctrl.maybe_degrade(2.0, now=2.0)
    assert ctrl.maybe_degrade(2.0, now=3.0) is None, "cooldown not enforced"
    assert ctrl.level == 1

    # A cooldown-blocked call does NOT reset the streak: those samples really
    # were slow. So the first call after the cooldown expires steps down.
    assert ctrl.maybe_degrade(2.0, now=39.0) is not None
    assert ctrl.level == 2
    assert ctrl.changes[-1]["depth_interval"] == 8


def test_adaptive_never_touches_ttc_thresholds():
    """Relaxing a safety threshold to gain FPS is the wrong trade."""
    ctrl = AdaptiveController(STEPS, min_fps=5.0, enabled=True, window=3,
                              cooldown_sec=0.0)
    for i in range(20):
        step = ctrl.maybe_degrade(1.0, now=float(i))
        if step:
            assert "warning_ttc" not in step
            assert "danger_ttc" not in step


def test_adaptive_stops_at_last_step():
    ctrl = AdaptiveController(STEPS, min_fps=5.0, enabled=True, window=3,
                              cooldown_sec=0.0)
    for i in range(100):
        ctrl.maybe_degrade(1.0, now=float(i))
    assert ctrl.level == len(STEPS) - 1


def test_adaptive_disabled_never_fires():
    ctrl = AdaptiveController(STEPS, min_fps=5.0, enabled=False, window=3,
                              cooldown_sec=0.0)
    for i in range(50):
        assert ctrl.maybe_degrade(0.5, now=float(i)) is None
    assert ctrl.level == 0


# ── Camera calibration scaffold ──────────────────────────────────────
def test_calibration_absent_is_not_fatal():
    from app.fcw.calibration import CameraCalibration
    calib = CameraCalibration.load("/nonexistent/calibration.yaml")
    assert calib.is_valid is False
    assert calib.intrinsics() is None


def test_calibration_roundtrip(tmp_path):
    from app.fcw.calibration import CameraCalibration
    matrix = [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    calib = CameraCalibration(camera_matrix=matrix, dist_coeffs=[0.1, -0.05, 0, 0, 0],
                              image_size=(1280, 720), reprojection_error=0.21)
    path = calib.save(str(tmp_path / "calibration.yaml"))
    loaded = CameraCalibration.load(path)
    assert loaded.is_valid
    assert loaded.focal_length_px() == pytest.approx(800.0)
    assert loaded.horizontal_fov_deg() == pytest.approx(77.3, abs=0.5)
    assert loaded.has_distortion


def test_undistort_passthrough_without_calibration():
    from app.fcw.calibration import CameraCalibration
    frame = np.zeros((100, 100, 3), np.uint8)
    assert CameraCalibration().undistort(frame) is frame

# ── Close-range (proximity) warning ─────────────────────────────────
# TTC alone is blind to the worst case: a vehicle stopped 3 m ahead has
# closing_speed = 0, so TTC is infinite and pure TTC logic reports NORMAL.
# These tests pin that down so the gap cannot be reintroduced silently.

def test_classify_distance_thresholds():
    from app.fcw.risk import classify_distance

    assert classify_distance(1.0, 8.0, 3.5) == DANGER
    assert classify_distance(3.5, 8.0, 3.5) == DANGER      # boundary inclusive
    assert classify_distance(3.6, 8.0, 3.5) == WARNING
    assert classify_distance(8.0, 8.0, 3.5) == WARNING
    assert classify_distance(8.1, 8.0, 3.5) == NORMAL
    assert classify_distance(30.0, 8.0, 3.5) == NORMAL
    assert classify_distance(None, 8.0, 3.5) == NORMAL
    assert classify_distance(float("nan"), 8.0, 3.5) == NORMAL


def test_worse_level_picks_most_severe():
    from app.fcw.risk import worse_level

    assert worse_level(NORMAL, WARNING) == WARNING
    assert worse_level(DANGER, WARNING) == DANGER
    assert worse_level(NORMAL, NORMAL) == NORMAL
    assert worse_level() == NORMAL


def _make_history(distance=None, ttc=None, valid=False):
    from app.fcw.tracking import TrackHistory
    h = TrackHistory(track_id=1, class_name="motorbike")
    h.frames_seen = 10
    h.distance = distance
    h.fit_distance = distance
    h.closing_speed = 0.0 if ttc is None else distance / ttc if ttc > 0 else 0.0
    h.speed_valid = valid
    if valid and ttc is not None:
        for i in range(5):
            h.samples.append((float(i), distance))
    return h


def test_stationary_close_object_is_danger_despite_infinite_ttc():
    """The core safety case: gap stops shrinking, TTC explodes, target is 3 m away."""
    from app.fcw.risk import compute_risk, road_roi

    detection = {"track_id": 1, "class_name": "motorbike", "confidence": 0.95,
                 "box": (600, 300, 700, 420), "area_ratio": 0.01,
                 "center_x": 650.0, "center_y": 360.0}
    roi = road_roi((720, 1280), None)
    risk = compute_risk(detection, _make_history(distance=3.0, ttc=None, valid=False),
                        roi, 1280, None)

    assert not risk["ttc_valid"]
    assert risk["level"] == DANGER, "a stationary target 3 m ahead must warn"
    assert risk["proximity_trigger"] is True
    assert risk["level_from_ttc"] == NORMAL
    assert risk["level_from_distance"] == DANGER


def test_close_object_warning_at_warning_distance():
    from app.fcw.risk import compute_risk, road_roi

    detection = {"track_id": 1, "class_name": "motorbike", "confidence": 0.9,
                 "box": (600, 300, 700, 420), "area_ratio": 0.01,
                 "center_x": 650.0, "center_y": 360.0}
    roi = road_roi((720, 1280), None)
    risk = compute_risk(detection, _make_history(distance=6.0), roi, 1280, None)
    assert risk["level"] == WARNING


def test_far_stationary_object_stays_normal():
    from app.fcw.risk import compute_risk, road_roi

    detection = {"track_id": 1, "class_name": "motorbike", "confidence": 0.9,
                 "box": (600, 300, 700, 420), "area_ratio": 0.01,
                 "center_x": 650.0, "center_y": 360.0}
    roi = road_roi((720, 1280), None)
    risk = compute_risk(detection, _make_history(distance=25.0), roi, 1280, None)
    assert risk["level"] == NORMAL


def test_distance_and_ttc_take_the_worse_level():
    """Far away but closing fast: TTC must still escalate."""
    from app.fcw.risk import compute_risk, road_roi

    detection = {"track_id": 1, "class_name": "motorbike", "confidence": 0.9,
                 "box": (600, 300, 700, 420), "area_ratio": 0.01,
                 "center_x": 650.0, "center_y": 360.0}
    roi = road_roi((720, 1280), None)
    # 20 m out (no proximity warning) but TTC 1.5 s.
    risk = compute_risk(detection, _make_history(distance=20.0, ttc=1.5, valid=True),
                        roi, 1280, None)
    assert risk["level_from_distance"] == NORMAL
    assert risk["level_from_ttc"] == DANGER
    assert risk["level"] == DANGER


# ── Central-corridor gate ───────────────────────────────────────────
# A windshield camera approximates the driver's eyeline, so only objects in the
# middle of the frame may raise an alarm. A motorcycle 2 m away in the next lane
# is not a collision risk for this vehicle, and alarming on it trains the driver
# to ignore the warning.

def _detect(cx_frac, cy_frac=0.72):
    return {"track_id": 1, "class_name": "motorbike", "confidence": 0.95,
            "box": (0, 0, 10, 10), "area_ratio": 0.05,
            "center_x": 1280 * cx_frac, "center_y": 720 * cy_frac}


def _risk_at(cx_frac, distance=3.0, cfg=None):
    from app.fcw.risk import compute_risk, road_roi, warning_corridor
    shape = (720, 1280)
    det = _detect(cx_frac)
    return compute_risk(det, _make_history(distance=distance),
                        road_roi(shape, cfg), 1280, cfg,
                        corridor=warning_corridor(shape, cfg))


def test_warning_corridor_is_narrower_than_roi():
    from app.fcw.config_loader import load_config
    from app.fcw.risk import road_roi, warning_corridor

    cfg = load_config()
    shape = (720, 1280)
    rx1, _, rx2, _ = road_roi(shape, cfg)
    cx1, _, cx2, _ = warning_corridor(shape, cfg)
    assert rx1 < cx1 and cx2 < rx2, "corridor must be tighter than the ROI"


def test_object_at_centre_warns():
    risk = _risk_at(0.50)
    assert risk["in_warning_corridor"] is True
    assert risk["level"] == DANGER


def test_object_at_frame_edge_is_suppressed():
    """3 m away but beside the ego lane: measured and drawn, but never alarmed.

    22% of width is inside the wide ROI yet outside the corridor, which isolates
    the corridor gate from the coarser ROI filter.
    """
    risk = _risk_at(0.22)
    assert risk["in_roi"] is True, "should still count as a road object"
    assert risk["in_warning_corridor"] is False
    assert risk["level"] == NORMAL
    # The ungated level is retained so the UI can explain the suppression.
    assert risk["level_unrestricted"] == DANGER


def test_gate_boundary_is_inclusive():
    from app.fcw.config_loader import load_config
    from app.fcw.risk import warning_corridor

    cfg = load_config()
    x_min, _, x_max, _ = warning_corridor((720, 1280), cfg)
    inside_lo = (x_min + 1) / 1280
    inside_hi = (x_max - 1) / 1280
    assert _risk_at(inside_lo, cfg=cfg)["in_warning_corridor"] is True
    assert _risk_at(inside_hi, cfg=cfg)["in_warning_corridor"] is True
    assert _risk_at(x_min / 1280 - 0.02, cfg=cfg)["in_warning_corridor"] is False


def test_low_object_below_corridor_is_suppressed():
    """Bottom edge: outside the vertical band, so no alarm."""
    risk = _risk_at(0.50, distance=3.0)
    det = _detect(0.50, cy_frac=0.995)
    from app.fcw.risk import compute_risk, road_roi, warning_corridor
    shape = (720, 1280)
    r2 = compute_risk(det, _make_history(distance=3.0), road_roi(shape, None),
                      1280, None, corridor=warning_corridor(shape, None))
    assert r2["in_warning_corridor"] is False
    assert r2["level"] == NORMAL
    assert risk["in_warning_corridor"] is True   # sanity: the centred one warns


def test_corridor_disabled_when_none_falls_back_to_roi():
    """Passing no corridor keeps the old ROI-only behaviour."""
    from app.fcw.risk import compute_risk, road_roi

    det = _detect(0.22)   # inside the wide ROI
    risk = compute_risk(det, _make_history(distance=3.0), road_roi((720, 1280), None),
                        1280, None, corridor=None)
    assert risk["in_roi"] is True
    assert risk["in_warning_corridor"] is True   # falls back to in_roi
    assert risk["level"] == DANGER


def test_object_outside_wide_roi_is_not_tracked_as_road():
    """Well off to the side is excluded earlier, by the ROI rather than the gate."""
    risk = _risk_at(0.08)
    assert risk["in_roi"] is False
    assert risk["in_warning_corridor"] is False
    assert risk["level"] == NORMAL
