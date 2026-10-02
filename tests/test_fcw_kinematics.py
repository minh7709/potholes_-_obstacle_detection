"""Unit tests for the FCW kinematics: closing speed, ego compensation, TTC, levels.

Run with:  .venv/bin/python -m pytest tests/test_fcw_kinematics.py -v
"""

import numpy as np
import pytest

from app.fcw.audio import ensure_audio_assets
from app.fcw.depth import DepthCalibration
from app.fcw.risk import (
    DANGER,
    NORMAL,
    WARNING,
    classify_ttc,
    compute_risk,
    road_roi,
    select_primary_target,
)
from app.fcw.tracking import TrackHistory


def feed(history, distances, rate=10.0, ego_bias=0.0, class_name="motorcycle"):
    """Push a synthetic distance ramp through a TrackHistory and return (speed, ttc)."""
    t = 0.0
    for d in distances:
        history.update(t, float(d), (0, 0, 10, 10), 0.9, class_name)
        history.estimate_speed(ego_bias)
        t += 1.0 / rate
    speed, valid = history.estimate_speed(ego_bias)
    ttc, ttc_valid = history.time_to_collision()
    return speed, valid, ttc, ttc_valid


def ramp(d0, d1, seconds, rate=10.0):
    n = int(seconds * rate) + 1
    return np.linspace(d0, d1, n)


# ── Closing speed ─────────────────────────────────────────────────────
def test_constant_closing_speed_recovered():
    """A target closing at 1 m/s must be measured as ~1 m/s."""
    history = TrackHistory(track_id=1)
    speed, valid, _, _ = feed(history, ramp(10.0, 8.0, 2.0))
    assert valid
    assert speed == pytest.approx(1.0, abs=0.05)


def test_fast_closing_speed_recovered():
    history = TrackHistory(track_id=1)
    speed, valid, _, _ = feed(history, ramp(10.0, 4.0, 2.0))
    assert valid
    assert speed == pytest.approx(3.0, abs=0.15)


def test_receding_target_yields_negative_speed():
    history = TrackHistory(track_id=1)
    speed, valid, _, _ = feed(history, ramp(10.0, 14.0, 2.0))
    assert valid
    assert speed == pytest.approx(-2.0, abs=0.15)


# ── Ego-motion compensation ───────────────────────────────────────────
def test_world_static_target_under_ego_motion_reads_zero_closing():
    """The core false-alarm case.

    A parked motorcycle with the ego car doing 2 m/s appears to shrink at 2 m/s.
    Without compensation the system reports a 2 m/s closing speed and warns.
    After compensation it must read ~0.
    """
    history = TrackHistory(track_id=1)
    # Apparent range shrinks 10 -> 6 m over 2 s, purely from ego motion.
    speed, valid, ttc, ttc_valid = feed(history, ramp(10.0, 6.0, 2.0), ego_bias=2.0)
    assert valid
    assert speed == pytest.approx(0.0, abs=0.35)
    assert not ttc_valid, "a world-static target must never produce a finite TTC"


def test_ego_compensation_preserves_true_relative_speed():
    """Compensation must subtract ego motion without destroying real closing."""
    history = TrackHistory(track_id=1)
    # Apparent 2 m/s closing while ego does 3 m/s => target is receding at 1 m/s.
    speed, valid, _, _ = feed(history, ramp(8.0, 4.0, 2.0), ego_bias=3.0)
    assert valid
    assert speed == pytest.approx(-1.0, abs=0.25)


def test_zero_ego_bias_leaves_speed_unchanged():
    history_a, history_b = TrackHistory(track_id=1), TrackHistory(track_id=2)
    speed_a, _, _, _ = feed(history_a, ramp(12.0, 9.0, 2.0), ego_bias=0.0)
    speed_b, _, _, _ = feed(history_b, ramp(12.0, 9.0, 2.0), ego_bias=0.0)
    assert speed_a == pytest.approx(speed_b)


# ── Sample sufficiency ────────────────────────────────────────────────
def test_insufficient_history_is_invalid():
    history = TrackHistory(track_id=1)
    history.update(0.0, 10.0, (0, 0, 10, 10), 0.9, "motorcycle")
    speed, valid = history.estimate_speed(0.0)
    assert not valid
    assert speed == 0.0


def test_short_time_window_is_rejected():
    """Few frames spanning too little time cannot produce a trustworthy slope."""
    history = TrackHistory(track_id=1)
    speed, valid, _, _ = feed(history, ramp(10.0, 9.5, 0.2))  # 0.2 s < 0.6 s minimum
    assert not valid


# ── TTC ───────────────────────────────────────────────────────────────
def test_ttc_definition():
    history = TrackHistory(track_id=1)
    _, _, ttc, valid = feed(history, ramp(10.0, 8.0, 2.0))
    assert valid
    # distance / closing_speed = 8.0 / 1.0
    assert ttc == pytest.approx(8.0, abs=0.4)


def test_ttc_infinite_when_not_closing():
    history = TrackHistory(track_id=1)
    _, _, ttc, valid = feed(history, ramp(10.0, 14.0, 2.0))
    assert not valid
    assert not np.isfinite(ttc)


def test_ttc_is_infinite_for_exactly_zero_speed():
    history = TrackHistory(track_id=1)
    _, _, ttc, valid = feed(history, ramp(10.0, 10.0, 2.0))
    assert not valid
    assert not np.isfinite(ttc)


def test_ttc_uses_regression_distance_not_ema():
    """TTC must not inherit the EMA lag, or range and rate disagree.

    On a constant ramp the EMA lags behind, so dividing the lagged distance by the
    instantaneous rate would report a TTC that is systematically too long.
    """
    history = TrackHistory(track_id=1)
    _, _, ttc, _ = feed(history, ramp(10.0, 4.0, 2.0))
    assert history.fit_distance is not None
    assert ttc == pytest.approx(history.fit_distance / history.closing_speed, rel=1e-6)
    assert history.fit_distance == pytest.approx(4.0, abs=0.2)


# ── Warning levels ────────────────────────────────────────────────────
def test_level_thresholds():
    assert classify_ttc(10.0) == NORMAL
    assert classify_ttc(4.0) == WARNING      # boundary inclusive
    assert classify_ttc(3.0) == WARNING
    assert classify_ttc(2.0) == DANGER       # boundary inclusive
    assert classify_ttc(0.5) == DANGER
    assert classify_ttc(float("inf")) == NORMAL
    assert classify_ttc(None) == NORMAL


# ── ROI ───────────────────────────────────────────────────────────────
def test_roi_excludes_edges_and_sky():
    roi = road_roi((720, 1280))
    x1, y1, x2, y2 = roi
    assert x1 > 0 and x2 < 1280
    assert y1 > 0
    assert y2 <= 720


def make_detection(cx, cy, cls="motorcycle", conf=0.9):
    return {
        "track_id": 1, "class_id": 3, "class_name": cls, "confidence": conf,
        "box": (cx - 40, cy - 60, cx + 40, cy + 60), "area_ratio": 0.02,
        "norm_box": None, "area_px": 9600.0, "center_x": cx, "center_y": cy,
    }


def make_history(track_id=1, distance=None, cls="motorcycle", speed=0.0, valid=False,
                 frames_seen=10):
    history = TrackHistory(track_id=track_id, class_name=cls)
    history.frames_seen = frames_seen
    history.distance = distance
    history.closing_speed = speed
    history.speed_valid = valid
    history.fit_distance = distance
    if valid and distance:
        history.samples.append((0.0, distance))
        history.samples.append((1.0, distance - speed))
        history.samples.append((2.0, distance - 2 * speed))
        history.samples.append((3.0, distance - 3 * speed))
    return history


def test_only_forward_roi_objects_are_flagged_in_roi():
    roi = road_roi((720, 1280))
    inside = compute_risk(make_detection(640, 500), make_history(distance=10.0, speed=0.0, valid=True), roi, 1280)
    outside = compute_risk(make_detection(60, 500), make_history(distance=10.0, speed=0.0, valid=True), roi, 1280)
    assert inside["in_roi"]
    assert not outside["in_roi"]


def test_non_motorcycle_never_becomes_primary_target():
    roi = road_roi((720, 1280))
    records = [{
        "detection": make_detection(640, 500, cls="car"),
        "history": make_history(distance=3.0, speed=3.0, valid=True, cls="car"),
        "risk": compute_risk(make_detection(640, 500, cls="car"),
                             make_history(distance=3.0, speed=3.0, valid=True, cls="car"), roi, 1280),
    }]
    assert select_primary_target(records, (720, 1280)) is None


def test_primary_target_prefers_danger_over_warning():
    roi = road_roi((720, 1280))

    def record(cx, distance, speed, cls="motorcycle"):
        det = make_detection(cx, 500, cls=cls)
        hist = make_history(distance=distance, speed=speed, valid=True, cls=cls)
        return {"detection": det, "history": hist, "risk": compute_risk(det, hist, roi, 1280)}

    # Left one is WARNING (low speed), centre one is DANGER (fast).
    records = [record(500, 10.0, 3.0), record(640, 5.0, 4.0)]
    chosen = select_primary_target(records, (720, 1280))
    assert chosen is not None
    assert chosen["risk"]["level"] == DANGER


def test_no_targets_when_everything_is_normal():
    roi = road_roi((720, 1280))
    det = make_detection(640, 500)
    hist = make_history(distance=25.0, speed=0.0, valid=False)
    records = [{"detection": det, "history": hist, "risk": compute_risk(det, hist, roi, 1280)}]
    assert select_primary_target(records, (720, 1280)) is None


# ── Depth sampling ────────────────────────────────────────────────────
def test_box_depth_median_rejects_background_outliers():
    """A mostly-near box with far background must report the near median.

    NaN / inf / out-of-range pixels must never survive into the median, which is
    exactly the failure mode a real monocular model exhibits on thin structures
    (handlebars, the gap between wheels).
    """
    from app.fcw.depth import DepthEstimator

    estimator = DepthEstimator.__new__(DepthEstimator)  # bypass model loading
    estimator._last_raw = None
    estimator.cfg = None

    depth = np.full((480, 640), 50.0, dtype=np.float32)
    depth[300:420, 280:360] = 8.0   # the "vehicle"
    depth[300:420, 360:400] = 90.0  # background bleeding in from the right
    depth[305:315, 285:295] = np.nan
    depth[400:410, 340:355] = np.inf
    estimator._last_raw = depth

    distance, samples = estimator.sample_box((280, 300, 400, 420), depth)
    assert distance is not None
    assert samples > 0
    assert 7.0 < distance < 12.0, f"expected ~8 m, got {distance}"


def test_depth_sampling_returns_none_on_out_of_range():
    from app.fcw.depth import DepthEstimator

    estimator = DepthEstimator.__new__(DepthEstimator)
    estimator._last_raw = None
    estimator.cfg = None
    depth = np.full((480, 640), 500.0, dtype=np.float32)  # beyond valid_max_m
    distance, samples = estimator.sample_box((200, 200, 300, 300), depth)
    assert distance is None


def test_depth_sampling_handles_tiny_box():
    from app.fcw.depth import DepthEstimator

    estimator = DepthEstimator.__new__(DepthEstimator)
    estimator._last_raw = np.full((480, 640), 10.0, dtype=np.float32)
    estimator.cfg = None
    distance, _ = estimator.sample_box((100, 100, 101, 101), estimator._last_raw)
    assert distance is None


# ── Calibration ───────────────────────────────────────────────────────
def test_calibration_scales_depth():
    calibration = DepthCalibration.from_reference(ref_depth_m=10.0, ref_true_m=12.0)
    # Model reads 10 m where truth is 12 m => multiply by 1.2
    assert calibration.scale == pytest.approx(1.2)
    assert calibration.apply(10.0) == pytest.approx(12.0)


def test_calibration_is_identity_when_unset():
    calibration = DepthCalibration.from_reference(ref_depth_m=10.0, ref_true_m=10.0)
    assert calibration.scale == pytest.approx(1.0)
    assert calibration.calibrated is False
    assert calibration.apply(7.5) == pytest.approx(7.5)


def test_calibration_handles_none():
    calibration = DepthCalibration()
    assert calibration.apply(None) is None


# ── Audio ─────────────────────────────────────────────────────────────
def test_audio_cooldown_suppresses_repeats():
    from app.fcw.audio import AudioAlerter

    alerter = AudioAlerter(enabled=False)
    assert alerter.play("DANGER", now=100.0) is False  # disabled
    alerter.enabled = True
    assert alerter.should_play("DANGER", 100.0)
    assert not alerter.should_play("DANGER", 100.1)
    assert not alerter.should_play("DANGER", 100.5)   # still inside the 1.0 s window
    assert alerter.should_play("DANGER", 101.5)
    alerter.close()


def test_audio_wav_files_are_generated(tmp_path):
    import os
    import wave

    warning = str(tmp_path / "warning.wav")
    danger = str(tmp_path / "danger.wav")
    created = ensure_audio_assets(warning, danger)

    assert len(created) == 2
    for path in (warning, danger):
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000
        # Must be a real, decodable PCM WAV, not a stub.
        with wave.open(path) as fh:
            assert fh.getnchannels() == 1
            assert fh.getsampwidth() == 2
            assert fh.getframerate() == 44100
            assert fh.getnframes() > 1000

    # Calling again must not overwrite existing assets.
    assert ensure_audio_assets(warning, danger) == []
