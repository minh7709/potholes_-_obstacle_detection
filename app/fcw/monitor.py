"""Throughput measurement and adaptive degradation.

What is measured
----------------
    input_fps        true frame rate of the source
    processing_fps   rate at which frames completed the pipeline
    yolo_fps         rate of the detection stage alone
    depth_fps        rate of real depth inferences (much lower when interval > 1)
    speed_ratio      video_duration / processing_time

`speed_ratio` above 1.0 means faster than real time; 0.3 means a 10-minute
video takes 33 minutes.

Adaptive policy
---------------
When throughput falls below a floor, reduce load in a fixed ladder
(target_fps 10 -> 8 -> 6 -> 4, depth_interval 5 -> 6 -> 8 -> 10).

Two deliberate constraints:

* TTC thresholds are NEVER adjusted. Loosening a safety threshold to gain FPS
  would trade a visible warning for an invisible failure, which is the wrong
  direction for a collision-warning system.
* Degradation is one-way within a run and logged. Bouncing between quality
  levels would make the output harder to interpret than simply running slower.

A single slow frame (first inference, GC pause) must not trigger it, so the
controller waits for a sustained run of slow samples.
"""

import time


class PerformanceMonitor:
    """Rolling FPS estimates plus a stage-timing breakdown."""

    def __init__(self, window=30):
        self.window = max(5, int(window))
        self._times = []
        self._source_frames = 0
        self._processed = 0

        self.yolo_count = 0
        self.depth_count = 0
        self.yolo_ms = 0.0
        self.depth_ms = 0.0
        self.render_ms = 0.0

        self.started_at = None
        self.last_frame_at = None
        self.elapsed = 0.0
        self.source_duration = None
        # Video time actually analysed. Differs from source_duration whenever a
        # run is truncated (max_frames), which must not inflate speed_ratio.
        self._first_video_time = None
        self._last_video_time = None

    def mark_video_time(self, timestamp):
        if self._first_video_time is None:
            self._first_video_time = timestamp
        self._last_video_time = timestamp

    def start(self):
        self.started_at = time.perf_counter()
        self.last_frame_at = self.started_at
        return self

    def mark_frame(self):
        """Call once per processed frame."""
        now = time.perf_counter()
        if self.started_at is not None:
            self.elapsed = now - self.started_at
        if self.last_frame_at is not None:
            self._times.append(now - self.last_frame_at)
            if len(self._times) > self.window:
                self._times.pop(0)
        self.last_frame_at = now
        self._processed += 1

    def mark_source_frame(self):
        self._source_frames += 1

    def add_yolo(self, ms, count=1):
        self.yolo_ms = float(ms)
        self.yolo_count += count

    def add_depth(self, ms, count=1):
        self.depth_ms = float(ms)
        self.depth_count += count

    def add_render(self, ms):
        self.render_ms = float(ms)

    # ── Derived metrics ──────────────────────────────────────────────
    def processing_fps(self):
        """Rate over the recent window (responsive, used for adaptation).

        `_times` holds the wall-clock delta BETWEEN consecutive frames, so the
        total elapsed time is their SUM. Using only first+last would badly
        underestimate the elapsed span and overstate the rate.
        """
        if len(self._times) < 2:
            return 0.0
        elapsed = float(sum(self._times))
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed

    def average_processing_fps(self):
        """Rate over the whole run (used for the final report)."""
        if self.elapsed <= 0 or self._processed <= 0:
            return 0.0
        return self._processed / self.elapsed

    def yolo_fps(self):
        if self.yolo_ms <= 0:
            return 0.0
        return 1000.0 / self.yolo_ms

    def depth_fps(self):
        if self.depth_ms <= 0:
            return 0.0
        return 1000.0 / self.depth_ms

    def processed_duration(self):
        """Video time actually analysed, in seconds."""
        if self._first_video_time is None or self._last_video_time is None:
            return None
        return max(0.0, self._last_video_time - self._first_video_time)

    def speed_ratio(self):
        """video_duration / processing_time. > 1 means faster than real time.

        Uses the video span ACTUALLY processed, not the source metadata. With a
        truncated run the metadata duration covers frames that were never
        analysed, which would overstate throughput (a 150-frame slice of a
        900-frame video would otherwise report >1x).
        """
        span = self.processed_duration()
        if span and self.elapsed > 0:
            return span / self.elapsed
        return None

    def summary(self, source_fps=None, target_fps=None, frame_step=None,
                input_size=None):
        return {
            "input_fps": round(source_fps, 2) if source_fps else None,
            "target_fps": target_fps,
            "frame_step": frame_step,
            "processing_fps": round(self.average_processing_fps(), 2),
            "processing_fps_recent": round(self.processing_fps(), 2),
            "yolo_fps": round(self.yolo_fps(), 2),
            "depth_fps": round(self.depth_fps(), 2),
            "yolo_ms": round(self.yolo_ms, 1),
            "depth_ms": round(self.depth_ms, 1),
            "frames_processed": self._processed,
            "source_frames": self._source_frames,
            "processing_time_sec": round(self.elapsed, 2),
            "video_duration_sec": (round(self.source_duration, 2)
                                   if self.source_duration else None),
            "processed_duration_sec": (round(self.processed_duration(), 2)
                                       if self.processed_duration() else None),
            "truncated": bool(self.source_duration and self.processed_duration()
                              and self.processed_duration() < self.source_duration - 1.0),
            "speed_ratio": (round(self.speed_ratio(), 3)
                            if self.speed_ratio() is not None else None),
            "input_size": f"{input_size[0]}x{input_size[1]}" if input_size else None,
        }


class AdaptiveController:
    """Steps down a performance ladder when throughput is sustained low."""

    def __init__(self, steps, min_fps=5.0, enabled=True, window=25, cooldown_sec=8.0):
        self.steps = steps or [{"target_fps": 10, "depth_interval": 5}]
        self.min_fps = float(min_fps)
        self.enabled = bool(enabled)
        self.window = max(3, int(window))
        self.cooldown_sec = float(cooldown_sec)

        self.level = 0
        self._slow_samples = 0
        # None, not 0.0: a 0.0 sentinel would look like "a change just happened at
        # the epoch", blocking the first degradation for the whole cooldown.
        self._last_change = None
        self.changes = []

    def current(self):
        return self.steps[self.level]

    def maybe_degrade(self, measured_fps, now):
        """Return the new step if it degraded, else None.

        TTC thresholds are never touched; only sample rates change.
        """
        if not self.enabled:
            return None
        if self.level >= len(self.steps) - 1:
            return None

        if measured_fps <= 0 or measured_fps >= self.min_fps:
            # Recover the streak so a brief recovery resets the requirement.
            self._slow_samples = 0
            return None

        self._slow_samples += 1
        if self._slow_samples < self.window:
            return None
        if (self._last_change is not None
                and (now - self._last_change) < self.cooldown_sec):
            return None

        self.level += 1
        self._slow_samples = 0
        self._last_change = now
        step = self.steps[self.level]
        entry = dict(step)
        entry["level"] = self.level
        entry["reason"] = f"processing_fps {measured_fps:.1f} < {self.min_fps:.1f}"
        self.changes.append(entry)
        return entry


class NullController(AdaptiveController):
    """Always-disabled controller, for benchmarking without adaptation."""

    def __init__(self):
        super().__init__([{"target_fps": None, "depth_interval": 1}],
                         enabled=False)