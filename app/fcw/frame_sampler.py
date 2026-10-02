"""Frame decimation and resizing for the inference pipeline.

This is the single most important optimisation after depth_interval, and it is
where the correctness traps live, so the invariants are spelled out here.

Invariant 1: timestamps come from the ORIGINAL frame index
------------------------------------------------------------
TTC is a division by a time delta, so a wrong timestamp silently corrupts every
speed in the output while still looking plausible. We therefore never derive
time from wall-clock or from "how many frames have we processed". For a file at
`input_fps`, original frame `i` is at time `i / input_fps`. Decimating to 10 FPS
keeps that exact time; only the spacing between samples changes.

Invariant 2: decimation drops frames, it never re-times them
-------------------------------------------------------------
At 60 FPS with target 10 FPS we process original frames 0, 6, 12, 18 ... The
timestamps are 0.0, 0.1, 0.2, 0.3 s. We do NOT pretend the processed stream is
60 FPS.

Invariant 3: aspect ratio is preserved on resize
------------------------------------------------
Stretching changes object shape, which degrades both YOLO and depth accuracy.
Size reduction keeps the aspect ratio and only shrinks when the frame exceeds
the configured maximum.
"""

import math

import numpy as np


class FrameSampler:
    """Decides whether a given frame should be processed, and at what size.

    `input_fps` is the true frame rate of the source. `target_fps` is the rate we
    want to run inference at; None processes every frame.
    """

    def __init__(self, input_fps, target_fps, max_width=None, max_height=None):
        self.input_fps = float(input_fps) if input_fps and input_fps > 0 else 30.0
        self.target_fps = float(target_fps) if target_fps and target_fps > 0 else None

        # Frame interval in source frames. 60 -> 10 FPS means every 6th frame.
        self.frame_step = 1
        if self.target_fps and self.target_fps < self.input_fps:
            self.frame_step = max(1, int(round(self.input_fps / self.target_fps)))

        self.max_width = max_width
        self.max_height = max_height

        self.source_frames_seen = 0     # every frame pulled from the source
        self.processed_frames = 0      # frames that actually went through inference
        self.skipped_frames = 0

    # ── Decimation ──────────────────────────────────────────────────
    def should_process(self):
        """True when the current source frame is due for processing.

        Consumes exactly one source frame per call, so the caller must invoke
        this once per frame read from the capture, in order.
        """
        index = self.source_frames_seen
        self.source_frames_seen += 1

        if index % self.frame_step == 0:
            self.processed_frames += 1
            return True

        self.skipped_frames += 1
        return False

    def timestamp_for(self, source_frame_index):
        """Wall-clock position of a source frame, in seconds.

        Derived from the frame index so it is reproducible and independent of
        how fast the machine processes frames.
        """
        return source_frame_index / float(self.input_fps)

    @property
    def effective_fps(self):
        """Rate at which frames actually reach the pipeline."""
        return self.input_fps / self.frame_step if self.frame_step else self.input_fps

    # ── Resize ──────────────────────────────────────────────────────
    def resize(self, frame):
        """Downscale to the configured maximum, preserving aspect ratio.

        Returns the frame unchanged when no limit is set or the frame is
        already small enough. `cv2.INTER_AREA` is used for downscaling because
        it averages source pixels, which avoids aliasing on fine structures
        (a distant motorcycle is mostly thin edges).
        """
        if frame is None or not self.max_width or not self.max_height:
            return frame

        h, w = frame.shape[:2]
        if w <= self.max_width and h <= self.max_height:
            return frame

        scale = min(self.max_width / float(w), self.max_height / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        import cv2
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ── Box coordinate mapping ──────────────────────────────────────
    def scale_box(self, box, from_shape, to_shape):
        """Map a bbox from one resolution to another.

        Detection runs on the resized frame, but boxes are often needed in
        source coordinates (or vice versa) for display or logging. Both axes
        must scale independently since the ratio differs per axis.
        """
        fh, fw = from_shape[:2]
        th, tw = to_shape[:2]
        if fh <= 0 or fw <= 0 or fh == th and fw == tw:
            return tuple(box)
        sx = tw / float(fw)
        sy = th / float(fh)
        x1, y1, x2, y2 = box
        return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

    def stats(self):
        total = self.source_frames_seen
        return {
            "input_fps": round(self.input_fps, 2),
            "target_fps": round(self.target_fps, 2) if self.target_fps else None,
            "frame_step": self.frame_step,
            "effective_fps": round(self.effective_fps, 2),
            "source_frames": total,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
        }


class LiveFrameSampler(FrameSampler):
    """FrameSampler variant for cameras, where wall-clock time is authoritative.

    For a live camera there is no frame index to trust: dropped frames and driver
    buffering make indices unreliable. So the decision is time-based against the
    clock, and the timestamp returned is the real capture time. That keeps dt
    honest, which matters because closing speed divides by it.
    """

    def __init__(self, input_fps, target_fps, max_width=None, max_height=None):
        super().__init__(input_fps, target_fps, max_width, max_height)
        self._interval = 1.0 / self.effective_fps if self.effective_fps else 0.0
        self._last_time = None

    def should_process(self, now):
        if not self._interval:
            return True
        if self._last_time is None or (now - self._last_time) >= self._interval:
            self._last_time = now
            return True
        return False

    def timestamp_for(self, source_frame_index, now=None):
        """For live sources prefer the real clock; fall back to the index."""
        if now is not None:
            return now
        return source_frame_index / float(self.input_fps)


def estimate_processing_fps(measured_fps, steps):
    """Pick the next performance step for the adaptive controller.

    `steps` is the ordered ladder from config (fastest first). We step DOWN one
    level at a time and never back up, because oscillating between quality
    levels is more distracting than running slightly slower than ideal.

    Returns the index of the chosen step.
    """
    if not steps:
        return 0
    if measured_fps is None or measured_fps <= 0:
        return 0
    for index, step in enumerate(steps):
        if measured_fps >= (step.get("target_fps") or 0):
            return index
    return len(steps) - 1