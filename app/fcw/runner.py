"""Runners for offline video files and live cameras.

Offline runner
--------------
Reads metadata, decimates, runs the pipeline, writes the video/CSV/summary, and
prints the throughput report. Timestamps come from the ORIGINAL frame index, so
TTC is correct even though only every Nth frame is processed.

Live runner
-----------
A camera on a slow CPU produces frames faster than we can analyse them. The
naive approach, one queue per frame, grows without bound: latency climbs without
limit and the alert describes a scene from tens of seconds ago, which is worse
than no alert at all.

So the live runner uses a single-slot "latest frame wins" mailbox: the capture
thread overwrites the previous frame, and the processing thread always works on
the freshest one. Old frames are discarded rather than queued. On a phone
recording 4K over USB this is the difference between a usable warning and one
arriving far too late.
"""

import os
import queue
import threading
import time

import numpy as np

from app.fcw.config_loader import depth_interval, input_max_size, target_fps
from app.fcw.frame_sampler import FrameSampler, LiveFrameSampler
from app.fcw.monitor import AdaptiveController, PerformanceMonitor
from app.fcw.output import ResultWriter
from app.fcw.source import open_source, source_fps


class BaseRunner:
    """Shared timing, adaptation and reporting for both runners."""

    def __init__(self, system, cfg, output_writer=None, verbose=True):
        self.system = system
        self.cfg = cfg
        self.verbose = verbose

        self.target_fps = target_fps(cfg)
        self.depth_interval = depth_interval(cfg)
        self.max_w, self.max_h = input_max_size(cfg)

        steps = _adaptive_steps(cfg)
        self.controller = AdaptiveController(
            steps=steps,
            min_fps=_cfg_f(cfg, "performance.adaptive_min_fps", 5.0),
            enabled=bool(cfg.get("performance.adaptive_enabled", True)),
            window=_cfg_i(cfg, "performance.adaptive_window", 25),
            cooldown_sec=_cfg_f(cfg, "performance.adaptive_cooldown_sec", 8.0),
        )
        self.monitor = PerformanceMonitor()
        self.writer = output_writer
        self._last_report = 0.0

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _maybe_adapt(self):
        """Apply a performance step if sustained throughput is too low.

        Only sample rates change. TTC thresholds are never relaxed to buy speed.
        """
        step = self.controller.maybe_degrade(self.monitor.processing_fps(), time.time())
        if step is None:
            return
        self.depth_interval = max(1, int(step.get("depth_interval", self.depth_interval)))
        new_fps = step.get("target_fps")
        if new_fps and self.sampler is not None:
            self.sampler.target_fps = float(new_fps)
            if self.sampler.frame_step < 1:
                self.sampler.frame_step = 1
            ratio = self.sampler.input_fps / float(new_fps)
            self.sampler.frame_step = max(1, int(round(ratio)))
        self.system.depth_interval = self.depth_interval
        self._log(
            f"[FCW] ADAPTIVE: throughput below floor, stepping down to level "
            f"{step['level']} (target_fps={new_fps}, depth_interval={self.depth_interval}) "
            f"because {step['reason']}. TTC thresholds unchanged.")

    def _tick_monitor(self, result, depth_computed):
        self.monitor.add_yolo(result.get("yolo_ms", 0.0))
        if depth_computed and result.get("depth_ms"):
            self.monitor.add_depth(result["depth_ms"])
        self.monitor.mark_frame()

    def _report(self, force=False):
        now = time.time()
        if not force and (now - self._last_report) < 2.0:
            return
        self._last_report = now
        stats = self.sampler.stats() if self.sampler else {}
        fps = self.monitor.processing_fps()
        self._log(
            f"[FCW] {stats.get('processed_frames', 0)} processed | "
            f"proc {fps:4.1f} FPS | depth every {self.depth_interval} | "
            f"tracks {result_tracks(self._last_result)}")

    def performance_summary(self):
        stats = self.sampler.stats() if self.sampler else {}
        return self.monitor.summary(
            source_fps=stats.get("input_fps"),
            target_fps=stats.get("target_fps"),
            frame_step=stats.get("frame_step"),
            input_size=(self.max_w, self.max_h),
        )


class VideoRunner(BaseRunner):
    """Offline analysis of a video file."""

    def __init__(self, source, system, cfg, output_writer=None, verbose=True,
                 max_frames=None, pause_every=0):
        super().__init__(system, cfg, output_writer, verbose)
        self.source = source
        self.max_frames = max_frames
        self.pause_every = pause_every
        self.sampler = None
        self._last_result = None

    def run(self):
        import cv2

        cap = open_source(self.source)
        if cap is None:
            raise RuntimeError(f"Could not open source: {self.source}")

        src_fps = source_fps(cap, self.source)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        self.sampler = FrameSampler(src_fps, self.target_fps, self.max_w, self.max_h)
        self.system.visualizer.set_sampler(self.sampler)
        self.system.reset()

        duration = (total / src_fps) if total > 0 else None
        self.monitor.source_duration = duration
        self.monitor.start()

        self._log(f"[FCW] Input: {src_w}x{src_h} @ {src_fps:.2f} FPS, "
                  f"{total} frames ({duration:.1f}s)" if duration else
                  f"[FCW] Input: {src_w}x{src_h} @ {src_fps:.2f} FPS, {total} frames")
        self._log(f"[FCW] Processing {self.sampler.effective_fps:.1f} FPS "
                  f"(step {self.sampler.frame_step}), depth every {self.depth_interval}")

        if self.writer is not None:
            # The annotated video is written at the PROCESSING rate. Only every
            # Nth frame is analysed, so claiming the original FPS would make a
            # decimated video play back several times too fast. Skipped frames
            # cannot be reconstructed, so this is a record of what was analysed,
            # not a faithful re-encode of the input.
            self.writer.set_video_fps(self.sampler.effective_fps)

        processed = 0
        analysed = 0
        try:
            while True:
                if self.max_frames and processed >= self.max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                self.monitor.mark_source_frame()
                source_index = self.sampler.source_frames_seen

                if not self.sampler.should_process():
                    continue

                # Timestamp from the ORIGINAL index: this is what keeps TTC right.
                timestamp = self.sampler.timestamp_for(source_index)

                small = self.sampler.resize(frame)
                if self.pause_every and processed % self.pause_every == 0:
                    self._report(force=True)

                annotated, result = self.system.process_frame(
                    small, timestamp, source_frame=source_index)

                processed += 1
                analysed += 1
                self.monitor.mark_video_time(timestamp)
                self._last_result = result
                self._tick_monitor(result, result.get("depth_computed"))
                self._maybe_adapt()

                if self.writer is not None:
                    self.writer.write_frame(
                        annotated, result["objects"], timestamp, source_index,
                        ego_speed=result["ego_speed_mps"],
                        depth_computed=result.get("depth_computed"))

                if processed % 50 == 0:
                    self._report()
        finally:
            cap.release()

        self._report(force=True)
        perf = self.performance_summary()
        sampler_stats = self.sampler.stats()
        writer_summary = self.writer.summary() if self.writer else {}

        return {
            "frames_source": sampler_stats["source_frames"],
            "frames_processed": processed,
            "frames_skipped": sampler_stats["skipped_frames"],
            "performance": perf,
            "output": writer_summary,
            "adaptive_changes": self.controller.changes,
            "detector": {"yolo_ms": round(self.system.detector.inference_ms, 1),
                         "depth_ms": round(self.system.depth.inference_ms, 1),
                         "yolo_calls": self.system.detector.inference_count,
                         "depth_calls": self.system.depth.inference_count},
        }


class LiveRunner(BaseRunner):
    """Live camera analysis with latest-frame-wins buffering.

    A single-slot mailbox instead of a queue. When processing is slower than
    capture, the pending frame is replaced rather than appended, so latency
    stays bounded at roughly one processing interval no matter how long we fall
    behind.
    """

    def __init__(self, source, system, cfg, output_writer=None, verbose=True,
                 window_name="RoadGuard - FCW"):
        super().__init__(system, cfg, output_writer, verbose)
        self.source = source
        self.window_name = window_name
        self.sampler = None
        self.paused = False
        self.audio_muted = False
        self._last_result = None

        self._cap = None
        self._frame_box = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self.display = True

    # ── Capture thread ───────────────────────────────────────────────
    def _capture_loop(self):
        import cv2

        while not self._stop.is_set():
            if self.paused:
                time.sleep(0.03)
                continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            frame = self.sampler.resize(frame)
            self.monitor.mark_source_frame()
            payload = (time.time(), self.monitor._source_frames, frame)

            # Latest frame wins: overwrite whatever is pending. Queuing instead
            # would let latency grow without bound whenever processing loses to
            # capture, and a stale alert is worse than a missing one.
            try:
                self._frame_box.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_box.put_nowait(payload)
            except queue.Full:
                pass

    # ── Main loop ────────────────────────────────────────────────────
    def run(self):
        import cv2

        self._cap = open_source(self.source)
        if self._cap is None:
            raise RuntimeError(f"Could not open camera source: {self.source}")

        src_fps = source_fps(self._cap, self.source, default=30.0)
        self.sampler = LiveFrameSampler(src_fps, self.target_fps,
                                        self.max_w, self.max_h)
        self.system.visualizer.set_sampler(self.sampler)
        self.system.reset()
        self.monitor.source_duration = None
        self.monitor.start()

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception:
            self.display = False
            self._log("[FCW] No GUI available. Headless mode (no display).")

        self._log(f"[FCW] Live source {self.source} @ {src_fps:.1f} FPS, "
                  f"processing {self.sampler.effective_fps:.1f} FPS, "
                  f"depth every {self.depth_interval}")
        self._log("[FCW] Keys: Q/ESC quit, P pause, M mute/unmute")

        self._thread = threading.Thread(target=self._capture_loop,
                                        name="FCWCapture", daemon=True)
        self._thread.start()

        processed = 0
        try:
            while not self._stop.is_set():
                try:
                    timestamp, source_index, frame = self._frame_box.get(timeout=0.2)
                except queue.Empty:
                    continue

                if self.paused:
                    continue
                if not self.sampler.should_process(timestamp):
                    continue

                annotated, result = self.system.process_frame(
                    frame, timestamp, source_frame=source_index)

                processed += 1
                self.monitor.mark_video_time(timestamp)
                self._last_result = result
                self._tick_monitor(result, result.get("depth_computed"))
                self._maybe_adapt()

                if self.writer is not None:
                    self.writer.write_frame(
                        annotated, result["objects"], timestamp, source_index,
                        ego_speed=result["ego_speed_mps"],
                        depth_computed=result.get("depth_computed"))

                if self.display:
                    cv2.imshow(self.window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        self._stop.set()
                        break
                    if key == ord("p"):
                        self.paused = not self.paused
                        self._log(f"[FCW] Paused: {self.paused}")
                        self._frame_box = queue.Queue(maxsize=1)
                    if key == ord("m"):
                        self.audio_muted = not self.audio_muted
                        self.system.alerter.enabled = not self.audio_muted
                        self._log(f"[FCW] Audio "
                                  f"{'muted' if self.audio_muted else 'on'}")

                if processed % 50 == 0:
                    self._report()
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            try:
                self._cap.release()
            except Exception:
                pass
            if self.display:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        self._report(force=True)
        return {
            "frames_processed": processed,
            "frames_skipped": self.sampler.stats()["skipped_frames"],
            "performance": self.performance_summary(),
            "output": self.writer.summary() if self.writer else {},
            "adaptive_changes": self.controller.changes,
        }

    def stop(self):
        self._stop.set()


def result_tracks(result):
    return len(result["objects"]) if result else 0


def _adaptive_steps(cfg):
    steps = cfg.get("performance.adaptive_steps", None)
    if isinstance(steps, list) and steps:
        return [s for s in steps if isinstance(s, dict)]
    return [{"target_fps": 10, "depth_interval": 5}]


def _cfg_f(cfg, key, default):
    try:
        v = cfg.get(key, default, cast=float)
        return default if v is None else float(v)
    except Exception:
        return default


def _cfg_i(cfg, key, default):
    try:
        v = cfg.get(key, default, cast=int)
        return default if v is None else int(v)
    except Exception:
        return default