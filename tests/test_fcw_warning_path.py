"""End-to-end test of the warning path with mocked perception.

Finds no motorcycle footage, so this verifies the part that actually matters:
that a closing target produces WARNING then DANGER, gets persisted to the database,
surfaces through the API, and fires audio with cooldown.

Perception (YOLO + depth) is replaced by deterministic fakes, so the assertions are
exact rather than probabilistic.
"""

import json
import os

import numpy as np
import pytest

from app.fcw.risk import DANGER, NORMAL, WARNING


# ── Fakes ────────────────────────────────────────────────────────────
class FakeDetector:
    """Returns one synthetic motorcycle whose box grows over time."""

    def __init__(self, class_name="motorcycle"):
        self.class_name = class_name
        self.frame = 0
        self.inference_ms = 1.0

    def detect(self, frame):
        self.frame += 1
        H, W = frame.shape[:2]
        # Grow from 4% to 40% of frame area => distance shrinks steadily.
        t = min(1.0, (self.frame - 1) / 80.0)
        area_ratio = 0.04 + 0.36 * t
        bw = int(np.sqrt(area_ratio * H * W * 1.6))
        bh = max(4, int(bw / 1.6))
        cx, cy = W // 2, int(H * 0.70)
        x1, y1 = cx - bw // 2, cy - bh // 2
        return [{
            "track_id": 42, "class_id": 3, "class_name": self.class_name,
            "confidence": 0.94,
            "box": (float(x1), float(y1), float(x1 + bw), float(y1 + bh)),
            "area_ratio": area_ratio, "norm_box": None,
            "area_px": float(bw * bh),
            "center_x": float(cx), "center_y": float(cy),
        }]


class FakeDepth:
    """Returns a depth reading that shrinks linearly over the run.

    `frames` is the number of depth INFERENCES expected over the run, not the
    number of processed frames. With depth.interval=5 and 80 processed frames,
    depth runs 16 times, so frames=16 completes the full 20 m -> 4 m ramp.

    Getting this wrong is easy and misleading: if the ramp is scaled to frame
    count instead, the apparent closing speed comes out ~5x too slow and the
    target never reaches DANGER, which looks like a TTC bug but is not.
    """

    def __init__(self, start_m=20.0, end_m=4.0, frames=16):
        self.start_m = start_m
        self.end_m = end_m
        self.frames = frames
        self.inference_ms = 5.0
        self.inference_count = 0
        self._last_raw = None
        self.cfg = None
        self.frame = 0

    def _current(self):
        t = min(1.0, (self.frame - 1) / self.frames)
        return self.start_m + (self.end_m - self.start_m) * t

    def infer(self, frame):
        self.frame += 1
        self.inference_count += 1
        H, W = frame.shape[:2]
        self._last_raw = np.full((H, W), 60.0, dtype=np.float32)
        return self._last_raw

    @property
    def last_depth_map(self):
        return self._last_raw

    def clear_cache(self):
        self._last_raw = None

    def sample_point(self, x, y, radius=6, depth_map=None):
        return self._current() + 20.0

    calibration = type("Cal", (), {"describe": staticmethod(lambda: "scale=1.000 (mock)")})()

    def sample_box(self, box, depth_map=None, center_crop=None, min_samples=None,
                   outlier_tolerance=None, valid_min=None, valid_max=None,
                   cfg=None):
        return self._current(), 100


class RecordingAlerter:
    def __init__(self):
        self.plays = []
        self.enabled = True

    def play(self, level, now, force=False):
        cooldown = 1.0 if level == DANGER else 1.5
        recent = [t for l, t in self.plays if l == level and now - t < cooldown]
        if recent:
            return False
        self.plays.append((level, now))
        return True

    def stats(self):
        return {"WARNING": sum(1 for l, _ in self.plays if l == "WARNING"),
                "DANGER": sum(1 for l, _ in self.plays if l == "DANGER")}

    def close(self):
        pass


class NullVisualizer:
    """Skips drawing so the tests measure logic, not rendering."""

    def render(self, frame, records, primary, **kwargs):
        return frame

    def set_sampler(self, sampler):
        pass


@pytest.fixture
def system():
    from app.fcw.config_loader import load_config, parse_set_overrides
    from app.fcw.pipeline import CollisionWarningSystem

    alerter = RecordingAlerter()
    cfg = load_config(overrides=parse_set_overrides([
        "collision.confirm_frames=2",
        "collision.danger_ttc=3.0",
        "collision.warning_ttc=6.0",
        # Tighten the windows so 40 synthetic frames are enough to fit a speed.
        "speed.min_samples=3",
        "speed.min_window_sec=0.2",
    ]))
    return CollisionWarningSystem(
        cfg,
        detector=FakeDetector(),
        depth_estimator=FakeDepth(),
        alerter=alerter,
        visualizer=NullVisualizer(),
    )


# ── Tests ────────────────────────────────────────────────────────────
def test_closing_target_escalates_to_danger(system):
    """A motorcycle closing from 20 m to 4 m must reach DANGER."""
    levels = []
    for i in range(80):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        _, result = system.process_frame(frame, timestamp=i / 30.0)
        levels.append(result["level"])

    assert levels[0] == NORMAL, "should start normal when far away"
    assert DANGER in levels, f"never reached DANGER: {set(levels)}"
    # Escalation must be monotonic: NORMAL -> WARNING -> DANGER, never backwards.
    first_warning = levels.index(WARNING) if WARNING in levels else None
    first_danger = levels.index(DANGER)
    assert first_warning is not None, f"never reached WARNING: {set(levels)}"
    assert first_warning < first_danger, "DANGER must come after WARNING"


def test_distance_and_ttc_are_reported(system):
    """Telemetry must be populated once the track has enough history."""
    result = None
    for i in range(60):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        _, result = system.process_frame(frame, timestamp=i / 30.0)
    primary = result["primary"]
    assert primary is not None
    assert primary["distance_m"] is not None
    assert primary["distance_m"] < 20.0, "distance should shrink over time"
    assert primary["track_id"] == 42
    assert primary["class"] == "motorcycle"


def test_audio_cooldown_prevents_spam(system):
    """Sustained danger must not fire audio on every frame."""
    for i in range(80):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        system.process_frame(frame, timestamp=i / 30.0)

    danger_plays = [t for l, t in system.alerter.plays if l == DANGER]
    # 80 frames over 2.67 s at 1.0 s cooldown => at most a handful of plays.
    assert len(danger_plays) <= 4, f"audio spammed {len(danger_plays)} times"
    assert len(danger_plays) >= 1, "danger audio never fired"
    # Gaps must respect the cooldown.
    gaps = [b - a for a, b in zip(danger_plays, danger_plays[1:])]
    assert all(g >= 0.99 for g in gaps), f"cooldown violated: {gaps}"


def test_confirmation_prevents_single_frame_false_alarm(system):
    """A target that is not actually closing must never raise a warning.

    Depth noise can produce a spurious low TTC for a single frame. Holding distance
    constant here models the opposite case (no real closing), so TTC is infinite and
    the level must stay NORMAL for the whole run.
    """
    # Hold distance constant => closing speed 0 => TTC infinite => NORMAL forever.
    system.depth.end_m = system.depth.start_m
    for i in range(60):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        _, result = system.process_frame(frame, timestamp=i / 30.0)
        assert result["level"] == NORMAL, f"raised {result['level']} with no closing"
    assert not [p for p in system.alerter.plays if p[0] == DANGER]


def test_non_primary_class_never_warns(system):
    """A car must not trigger the motorcycle collision alarm in v1."""
    system.detector.class_name = "car"
    for i in range(80):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        _, result = system.process_frame(frame, timestamp=i / 30.0)
        assert result["level"] == NORMAL, f"car raised {result['level']}"


def test_track_history_persists_across_frames(system):
    """Same ID must keep accumulating distance history, not restart each frame."""
    seen = []
    for i in range(40):
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        _, result = system.process_frame(frame, timestamp=i / 30.0)
        if 42 in system.histories:
            seen.append(system.histories[42].frames_seen)
    assert seen == list(range(1, 41)), "history did not accumulate per track"


# ── Persistence layer ────────────────────────────────────────────────
def test_warning_frames_persist_to_db(tmp_path, monkeypatch):
    """A warning video must round-trip through SQLite and serialise correctly."""
    import importlib

    import app.config
    import app.database.connection as conn
    import app.database.models as models

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    importlib.reload(app.config)
    importlib.reload(conn)
    importlib.reload(models)

    conn.Base.metadata.create_all(bind=conn.engine)
    db = conn.SessionLocal()
    try:
        video = models.VideoRecord(filename="a.mp4", saved_filename="a.mp4",
                                   video_url="/static/videos/a.mp4", total_frames=100)
        db.add(video)
        db.commit()
        db.refresh(video)

        db.add(models.WarningFrame(
            video_id=video.id, frame_number=30, timestamp_sec=1.0,
            image_path="/static/frames/f.jpg", severity="DANGER",
            track_id=42, class_name="motorcycle", distance_m=5.5,
            closing_speed_mps=3.1, ttc_sec=1.77, confidence=0.94,
            ego_speed_mps=8.0,
            detections_json=json.dumps([{"class": "motorcycle", "track_id": 42}]),
        ))
        db.commit()

        fetched = db.query(models.VideoRecord).filter(
            models.VideoRecord.id == video.id).one()
        payload = fetched.to_dict(include_frames=True)
        assert len(payload["frames"]) == 1

        frame = payload["frames"][0]
        assert frame["severity"] == "DANGER"
        assert frame["ttc_sec"] == 1.77
        assert frame["distance_m"] == 5.5
        assert frame["detections"][0]["class"] == "motorcycle"
    finally:
        db.close()
        # Leave the shared engine pointing at the real DB for other tests.
        monkeypatch.undo()
        importlib.reload(app.config)
        importlib.reload(conn)
        importlib.reload(models)
