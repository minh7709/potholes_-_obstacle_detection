"""FCW video processing for the web app.

Replaces the old ONNX pothole detector while keeping the public surface the API
routes expect:

    results = detector.process_video(video_path=..., video_id=...)
    results["total_frames"], ["fps"], ["duration_sec"], ["warning_frames"]

Only the image processing changed: YOLO + ByteTrack + Depth Anything V2 instead of
the pothole ONNX model.

Performance
-----------
This reuses the same FrameSampler as the CLI, so the web path gets the same
optimisations:

  * frames decimated to ``processing.target_fps`` (a 60 FPS phone video is
    sampled at 10 FPS, so it costs about the same as a 30 FPS one)
  * frames downscaled to ``input.max_width/height`` before inference
  * Depth Anything V2 run every ``depth.interval`` frames

Timestamps come from the ORIGINAL frame index, never from a processed-frame
counter. TTC divides by delta time, so re-timing the samples would silently
corrupt every speed in the output.

Lazy model loading
------------------
The pipeline is built on first use, not at import. Browsing history or opening
the page should not pay the YOLO + depth load cost.
"""

import json
import os
import time

import cv2

from app.config import FRAMES_DIR
from app.fcw.config_loader import (
    depth_interval,
    input_max_size,
    load_config,
    target_fps,
)
from app.fcw.frame_sampler import FrameSampler
from app.fcw.monitor import PerformanceMonitor
from app.fcw.pipeline import CollisionWarningSystem
from app.fcw.risk import DANGER, WARNING


class FCWVideoProcessor:
    """Runs the FCW pipeline over an uploaded video and saves warning frames."""

    def __init__(self, config_path=None):
        self._config_path = config_path
        self._system = None
        self._cfg = None

    # ── Pipeline construction ────────────────────────────────────────
    def _get_system(self):
        """Build the pipeline on first use, reusing it for later uploads."""
        if self._system is None:
            # Same config.yaml as the CLI, so tuning one tunes both.
            self._cfg = load_config(self._config_path)
            # A browser has no audio device, and a failed alert should never
            # take down an upload.
            self._cfg.set("warning.audio_enabled", False)
            self._system = CollisionWarningSystem(self._cfg)
            self._system.source_label = "web"
        return self._system

    @property
    def config(self):
        if self._cfg is None:
            self._cfg = load_config(self._config_path)
        return self._cfg

    # ── Video analysis ───────────────────────────────────────────────
    def process_video(self, video_path: str, video_id: int, progress_callback=None):
        """Analyse a video file frame by frame.

        Saves a JPEG for each frame that raised a WARNING or DANGER, then
        returns the telemetry the routes persist to SQLite.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or not (1.0 < src_fps < 240.0):
            src_fps = 25.0
        total_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (total_source / src_fps) if total_source > 0 else 0.0

        cfg = self.config
        system = self._get_system()
        system.source_label = f"video #{video_id}"

        max_w, max_h = input_max_size(cfg)
        sampler = FrameSampler(src_fps, target_fps(cfg), max_w, max_h)
        interval = depth_interval(cfg)

        monitor = PerformanceMonitor()
        monitor.source_duration = duration_sec
        monitor.start()

        print(f"[FCW] Video #{video_id}: {total_source} frames @ {src_fps:.2f} FPS "
              f"({duration_sec:.1f}s) -> processing {sampler.effective_fps:.1f} FPS "
              f"(step {sampler.frame_step}), depth every {interval}")

        warning_frames = []
        danger_count = 0
        min_ttc = None
        min_distance = None
        processed = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                source_index = sampler.source_frames_seen
                if not sampler.should_process():
                    continue

                monitor.mark_source_frame()

                # Timestamp from the ORIGINAL index. This is what keeps TTC right.
                timestamp = sampler.timestamp_for(source_index)

                small = sampler.resize(frame)
                annotated, result = system.process_frame(
                    small, timestamp, source_frame=source_index)
                processed += 1

                monitor.mark_video_time(timestamp)
                monitor.add_yolo(result.get("yolo_ms", 0.0))
                if result.get("depth_computed") and result.get("depth_ms"):
                    monitor.add_depth(result["depth_ms"])
                monitor.mark_frame()

                level = result["level"]
                primary = result["primary"]

                if primary is not None:
                    ttc = primary.get("ttc_s")
                    if ttc is not None and ttc > 0:
                        min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
                    dist = primary.get("distance_m")
                    if dist is not None and dist > 0:
                        min_distance = dist if min_distance is None else min(min_distance, dist)

                # Persist only frames that actually warned. Saving every frame of a
                # 90 s video would produce thousands of images for no benefit.
                if level in (DANGER, WARNING) and primary is not None:
                    if level == DANGER:
                        danger_count += 1
                    image_path = self._save_frame(annotated, video_id, source_index)
                    if image_path:
                        warning_frames.append({
                            "frame_number": source_index,
                            "timestamp_sec": round(timestamp, 3),
                            "image_path": image_path,
                            "severity": level,
                            "track_id": primary["track_id"],
                            "class_name": primary["class"],
                            "distance_m": primary["distance_m"],
                            "closing_speed_mps": primary["closing_speed_mps"],
                            "ttc_sec": primary["ttc_s"],
                            "confidence": primary["confidence"],
                            "ego_speed_mps": round(result["ego_speed_mps"], 2),
                            "detections_json": json.dumps(result["objects"]),
                        })

                if progress_callback:
                    progress_callback(processed, total_source)
        finally:
            cap.release()

        sampler_stats = sampler.stats()
        perf = monitor.summary(source_fps=sampler_stats["input_fps"],
                               target_fps=sampler_stats["target_fps"],
                               frame_step=sampler_stats["frame_step"],
                               input_size=(max_w, max_h))

        print(f"[FCW] Video #{video_id} done: {processed}/{total_source} frames "
              f"in {perf['processing_time_sec']}s "
              f"(speed_ratio {perf['speed_ratio']}), "
              f"{len(warning_frames)} warning frames")

        return {
            "total_frames": processed,
            "source_frames": total_source,
            "frames_skipped": sampler_stats["skipped_frames"],
            "fps": round(src_fps, 2),
            "duration_sec": round(duration_sec, 2),
            "warning_frames": warning_frames,
            "warning_frames_count": len(warning_frames),
            "danger_frames_count": danger_count,
            "min_ttc_sec": min_ttc,
            "min_distance_m": min_distance,
            "performance": perf,
        }

    def _save_frame(self, frame, video_id, source_frame):
        """Write one annotated frame to the static frames directory."""
        try:
            filename = f"fcw_{video_id}_frame_{source_frame:06d}.jpg"
            filepath = os.path.join(FRAMES_DIR, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return f"/static/frames/{filename}"
        except Exception as exc:
            print(f"[FCW] Could not save frame {source_frame}: {exc}")
            return None

    def reset(self):
        """Drop all temporal state so the next upload starts clean.

        Essential: without this, one video's tracking histories and ego-motion
        history would seed the next video's first frames with wrong range and
        speed.
        """
        if self._system is not None:
            self._system.reset()


# Module-level singleton, matching the pattern used across the codebase.
detector = FCWVideoProcessor()