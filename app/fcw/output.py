"""Result output: annotated video, events CSV and summary JSON.

Memory discipline
-----------------
Rows are streamed to the CSV as they are produced and the video is encoded
frame by frame. Nothing accumulates a full-video-sized structure in RAM, so a
1-2 minute clip and a 30 minute clip cost the same working memory. The only
in-memory lists are the warning events themselves, which are bounded by the
warning rate rather than by the frame count.

CSV honesty
-----------
The CSV carries a `distance_source` column with one of:

    measured     a real Depth Anything V2 inference at this timestamp
    interpolated linearly extrapolated between two real inferences
    held         last good value reused because the frame had no depth

A downstream reader can therefore tell an estimate from a measurement, which
matters because interpolated rows drive the closing-speed fit.
"""

import csv
import json
import os
import time

CSV_COLUMNS = [
    "timestamp",
    "frame",
    "track_id",
    "class",
    "confidence",
    "distance_m",
    "distance_source",
    "closing_speed_mps",
    "ttc_seconds",
    "status",
    "in_roi",
    "ego_speed_mps",
    "depth_computed",
    "proximity_trigger",
]


class ResultWriter:
    """Writes the annotated video, the events CSV and the summary JSON.

    All three outputs are optional so the same class serves both the CLI (which
    usually wants the video) and the web app (which persists to SQLite instead).
    """

    def __init__(self, output_dir="output", video_name="annotated_video.mp4",
                 csv_name="events.csv", summary_name="summary.json",
                 save_video=True, save_csv=True, save_summary=True,
                 video_fps=30.0, video_size=None, csv_every_frame=True,
                 fourcc="mp4v"):
        self.output_dir = output_dir
        self.save_video = save_video
        self.save_csv = save_csv
        self.save_summary = save_summary
        self.csv_every_frame = csv_every_frame

        self.video_path = os.path.join(output_dir, video_name) if save_video else None
        self.csv_path = os.path.join(output_dir, csv_name) if save_csv else None
        self.summary_path = os.path.join(output_dir, summary_name) if save_summary else None

        self._writer = None
        self._csv_file = None
        self._csv_writer = None
        self._video_writer = None
        self._video_fps = float(video_fps) if video_fps and video_fps > 0 else 30.0
        self._video_size = video_size
        self._fourcc = fourcc

        self._events = []          # warning-level rows only, for the summary
        self._row_count = 0
        self._frame_count = 0
        self._warning_frames = 0
        self._danger_frames = 0
        self._min_ttc = None
        self._min_distance = None
        self._started_at = None
        self._elapsed = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────
    def set_video_fps(self, fps):
        """Set the output frame rate.

        Must be called before the first `write_frame`. The encoder is created
        lazily precisely so this can happen after construction: the runner only
        learns the real processing rate once it has decoded the source.
        """
        if fps and fps > 0:
            self._video_fps = float(fps)

    def _ensure_video_writer(self, frame):
        """Create the encoder on the first frame.

        Lazy so that fps and size can still be corrected after construction but
        before any frame is encoded.
        """
        if self._video_writer is not None or not self.save_video:
            return
        if frame is None:
            return
        import cv2
        h, w = frame.shape[:2]
        size = self._video_size or (w, h)
        try:
            fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
            writer = cv2.VideoWriter(self.video_path, fourcc, self._video_fps, size)
        except Exception as exc:
            print(f"[FCW] Could not create video writer: {exc}")
            self.save_video = False
            return
        if not writer.isOpened():
            print(f"[FCW] Could not open video writer for {self.video_path}; "
                  f"video output disabled.")
            self.save_video = False
            try:
                writer.release()
            except Exception:
                pass
            return
        self._video_writer = writer
        self._video_size = size

    def open(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self._started_at = time.perf_counter()

        if self.save_csv:
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
            self._csv_writer.writeheader()

        return self

    def close(self):
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None

        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

        if self._started_at is not None:
            self._elapsed = time.perf_counter() - self._started_at

        if self.save_summary and self.summary_path:
            self._write_summary()
        return self.summary()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    # ── Per-frame writes ─────────────────────────────────────────────
    def write_frame(self, annotated, objects, timestamp, source_frame,
                    ego_speed=0.0, depth_computed=False, fps=None):
        """Record one processed frame.

        `objects` is the list of per-object telemetry dicts returned by the
        pipeline. Rows are written for every tracked object when
        csv_every_frame is set, so the CSV reconstructs the full trajectory;
        warning-level rows are additionally retained for the summary.
        """
        self._frame_count += 1

        if self.save_video and annotated is not None:
            self._ensure_video_writer(annotated)
            if self._video_writer is not None:
                try:
                    self._video_writer.write(annotated)
                except Exception as exc:
                    print(f"[FCW] Video write failed: {exc}")
                    self._video_writer = None

        primary_level = None
        if objects:
            for obj in objects:
                level = obj.get("level")
                if level in ("DANGER", "WARNING"):
                    if primary_level is None or level == "DANGER":
                        primary_level = level
                    self._track_minima(obj)
                    self._events.append(dict(obj))

        if primary_level == "DANGER":
            self._danger_frames += 1
            self._warning_frames += 1
        elif primary_level == "WARNING":
            self._warning_frames += 1

        if self._csv_writer is not None and objects and self.csv_every_frame:
            for obj in objects:
                self._write_row(obj, timestamp, source_frame, ego_speed, depth_computed)

        return primary_level

    def _track_minima(self, obj):
        ttc = obj.get("ttc_s")
        if ttc is not None and ttc > 0:
            self._min_ttc = ttc if self._min_ttc is None else min(self._min_ttc, ttc)
        dist = obj.get("distance_m")
        if dist is not None and dist > 0:
            self._min_distance = dist if self._min_distance is None else min(self._min_distance, dist)

    def _write_row(self, obj, timestamp, source_frame, ego_speed, depth_computed):
        try:
            self._csv_writer.writerow({
                "timestamp": f"{timestamp:.2f}",
                "frame": int(source_frame),
                "track_id": obj.get("track_id") if obj.get("track_id") is not None else "",
                "class": obj.get("class", ""),
                "confidence": f"{obj.get('confidence', 0.0):.4f}",
                "distance_m": _num(obj.get("distance_m"), 2),
                "distance_source": obj.get("distance_source", ""),
                "closing_speed_mps": _num(obj.get("closing_speed_mps"), 2),
                "ttc_seconds": _num(obj.get("ttc_s"), 2),
                "status": obj.get("level", ""),
                "in_roi": int(bool(obj.get("in_roi"))),
                "ego_speed_mps": f"{float(ego_speed or 0.0):.2f}",
                "depth_computed": int(bool(depth_computed)),
                # 1 when range alone triggered the warning (TTC not applicable).
                "proximity_trigger": int(bool(obj.get("proximity_trigger"))),
            })
            self._row_count += 1
        except Exception as exc:
            print(f"[FCW] CSV write failed: {exc}")

    # ── Summary ──────────────────────────────────────────────────────
    def summary(self):
        """Aggregate statistics. Safe to call before close() for partial runs."""
        return {
            "frames_processed": self._frame_count,
            "csv_rows": self._row_count,
            "warning_events": len(self._events),
            "warning_frames": self._warning_frames,
            "danger_frames": self._danger_frames,
            "min_ttc_sec": round(self._min_ttc, 2) if self._min_ttc is not None else None,
            "min_distance_m": (round(self._min_distance, 2)
                               if self._min_distance is not None else None),
        }

    def _write_summary(self):
        data = dict(self.summary())
        data["processing_time_sec"] = round(self._elapsed, 2)
        data["outputs"] = {
            "annotated_video": self.video_path if self.save_video else None,
            "events_csv": self.csv_path if self.save_csv else None,
        }
        # Per-track summary: how long each object was tracked and how close it got.
        data["tracks"] = self._track_summary()
        try:
            with open(self.summary_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[FCW] Could not write summary: {exc}")

    def _track_summary(self):
        """Aggregate the warning events per track ID."""
        tracks = {}
        for event in self._events:
            tid = event.get("track_id")
            key = str(tid) if tid is not None else "untracked"
            entry = tracks.setdefault(key, {
                "class": event.get("class"),
                "events": 0,
                "min_distance_m": None,
                "min_ttc_sec": None,
                "max_closing_mps": None,
            })
            entry["events"] += 1

            def _maybe_min(field, value):
                if value is None:
                    return
                current = entry[field]
                entry[field] = value if current is None else min(current, value)

            def _maybe_max(field, value):
                if value is None:
                    return
                current = entry[field]
                entry[field] = value if current is None else max(current, value)

            _maybe_min("min_distance_m", event.get("distance_m"))
            _maybe_min("min_ttc_sec", event.get("ttc_s"))
            _maybe_max("max_closing_mps", event.get("closing_speed_mps"))

        for entry in tracks.values():
            for field in ("min_distance_m", "min_ttc_sec", "max_closing_mps"):
                if entry[field] is not None:
                    entry[field] = round(entry[field], 2)
        return tracks


def _num(value, digits):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""