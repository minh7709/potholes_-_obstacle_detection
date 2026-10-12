"""Road Obstacle & FCW Demo — live camera feed.

Receives frames from a phone camera (e.g. IP Webcam) over the local network,
processes them through the existing app/fcw pipeline, and displays an annotated
window with detection, tracking, depth, risk assessment and performance metrics.

Usage:
    python tools/live.py --url "http://192.168.1.42:8080/video"
    python tools/live.py --url "http://192.168.1.42:8080/video" --width 960
    python tools/live.py --url path/to/video.mp4          # fallback video file
    python tools/live.py --help

The phone only supplies frames. Detection, tracking, depth and risk are the
existing app/fcw pipeline; nothing about it is reimplemented here.

Press q or Esc to quit.
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

# ── Project root setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.fcw.config_loader import load_config          # noqa: E402
from app.fcw.pipeline import CollisionWarningSystem    # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────
WINDOW_NAME = "Road Obstacle & FCW Demo"
MAX_RECONNECT_ATTEMPTS = 5
DEFAULT_RECONNECT_DELAY = 2.0  # seconds
FPS_WINDOW_SIZE = 30
BENCHMARK_DIR = "runs"


# ── CLI ───────────────────────────────────────────────────────────────
def build_parser():
    """Build argparse parser. Does NOT load model or open camera."""
    p = argparse.ArgumentParser(
        prog="live.py",
        description="Road Obstacle & FCW Demo: live camera or video file processing.",
        epilog=(
            "Example:\n"
            "  python tools/live.py --url http://192.168.1.42:8080/video --width 960\n"
            "  python tools/live.py --url dashcam.mp4\n"
            "\n"
            "Replace the IP with YOUR phone's IP. Do NOT use a public IP.\n"
            "On your phone: open IP Webcam -> Start Server."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--url", required=True,
        help="Camera stream URL (e.g. http://192.168.1.42:8080/video) or path to a video file.",
    )
    p.add_argument(
        "--width", type=int, default=0,
        help="Target output width in pixels. Height scales proportionally. 0 = no resize.",
    )
    p.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml at project root).",
    )
    p.add_argument(
        "--reconnect", type=int, default=MAX_RECONNECT_ATTEMPTS,
        help=f"Max reconnect attempts when stream drops (default: {MAX_RECONNECT_ATTEMPTS}). 0 = no reconnect.",
    )
    p.add_argument(
        "--reconnect-delay", type=float, default=DEFAULT_RECONNECT_DELAY,
        help=f"Seconds between reconnect attempts (default: {DEFAULT_RECONNECT_DELAY}).",
    )
    p.add_argument(
        "--benchmark", action="store_true",
        help="Write per-frame benchmark CSV to runs/ directory.",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Run headless (no OpenCV window). Useful for pure benchmark.",
    )
    return p


def validate_args(args):
    """Validate parsed arguments; return error message or None."""
    if not args.url:
        return "URL cannot be empty."
    if args.width < 0:
        return f"Width must be >= 0, got {args.width}."
    if args.reconnect < 0:
        return f"Reconnect attempts must be >= 0, got {args.reconnect}."
    if args.reconnect_delay < 0:
        return f"Reconnect delay must be >= 0, got {args.reconnect_delay}."
    return None


# ── FPS / Metrics ─────────────────────────────────────────────────────
class FrameMetrics:
    """Sliding-window FPS and per-frame timing using monotonic clock."""

    def __init__(self, window_size=FPS_WINDOW_SIZE):
        self._times = deque(maxlen=window_size)
        self._pipeline_ms = deque(maxlen=window_size)
        self._yolo_ms = deque(maxlen=window_size)
        self._depth_ms = deque(maxlen=window_size)
        self.total_frames = 0
        self._start_time = time.perf_counter()

    def tick(self, pipeline_ms, yolo_ms=None, depth_ms=None):
        """Record one processed frame."""
        now = time.perf_counter()
        self._times.append(now)
        self._pipeline_ms.append(pipeline_ms)
        if isinstance(yolo_ms, (int, float)):
            self._yolo_ms.append(float(yolo_ms))
        if isinstance(depth_ms, (int, float)) and depth_ms > 0:
            self._depth_ms.append(float(depth_ms))
        self.total_frames += 1

    @property
    def sliding_fps(self):
        """FPS over the sliding window: (N-1) / (t_last - t_first), N >= 2."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span

    @property
    def overall_fps(self):
        """Total frames / total elapsed time."""
        elapsed = time.perf_counter() - self._start_time
        return self.total_frames / elapsed if elapsed > 0 else 0.0

    @staticmethod
    def _avg(d):
        return sum(d) / len(d) if d else 0.0

    @property
    def avg_pipeline_ms(self):
        return self._avg(self._pipeline_ms)

    @property
    def avg_yolo_ms(self):
        return self._avg(self._yolo_ms)

    @property
    def avg_depth_ms(self):
        return self._avg(self._depth_ms)

    @property
    def p50_pipeline_ms(self):
        return self._percentile(self._pipeline_ms, 50)

    @property
    def p95_pipeline_ms(self):
        return self._percentile(self._pipeline_ms, 95)

    @staticmethod
    def _percentile(d, pct):
        if not d:
            return 0.0
        import numpy as np
        return float(np.percentile(list(d), pct))


# ── Benchmark CSV ─────────────────────────────────────────────────────
class BenchmarkWriter:
    """Writes per-frame stats to a CSV file."""

    COLUMNS = [
        "frame", "timestamp", "src_w", "src_h", "proc_w", "proc_h",
        "loop_fps", "pipeline_ms", "yolo_ms", "depth_ms",
        "detections", "tracks", "level", "primary_class",
        "primary_distance_m", "primary_ttc_s",
    ]

    def __init__(self, directory=BENCHMARK_DIR):
        os.makedirs(directory, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"live_benchmark_{tag}.csv")
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.COLUMNS)
        self._writer.writeheader()

    def write(self, row_dict):
        self._writer.writerow(row_dict)

    def close(self):
        self._fh.close()


# ── Camera connection ─────────────────────────────────────────────────
def open_camera(url, timeout_sec=15):
    """Try to open a video source. Returns (cap, True) or (None, False)."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None, False
    # Try to read a test frame to confirm the stream is alive.
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None, False
    return cap, True


def get_frame_size(cap, frame):
    """Get reliable frame dimensions from the actual frame, not cap.get()."""
    if frame is not None:
        h, w = frame.shape[:2]
        return w, h
    # Fallback to cap properties (may be 0 for network streams)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h


# ── Overlay drawing ──────────────────────────────────────────────────
def draw_overlay(frame, metrics, res, src_w, src_h, proc_w, proc_h, status=""):
    """Draw multi-line performance and status overlay on frame."""
    level = res.get("level", "NORMAL") if isinstance(res, dict) else "NORMAL"
    detections = res.get("detections", 0) if isinstance(res, dict) else 0
    tracks = res.get("tracks", 0) if isinstance(res, dict) else 0
    primary = res.get("primary") if isinstance(res, dict) else None

    lines = [
        f"Loop FPS: {metrics.sliding_fps:5.1f}  |  Overall: {metrics.overall_fps:5.1f}",
        f"Pipeline: {metrics.avg_pipeline_ms:5.1f}ms  |  YOLO: {metrics.avg_yolo_ms:4.0f}ms  |  Depth: {metrics.avg_depth_ms:4.0f}ms",
        f"Source: {src_w}x{src_h} -> {proc_w}x{proc_h}  |  Det: {detections}  Tracks: {tracks}",
    ]

    # Warning/risk status from pipeline
    if primary is not None:
        p_class = primary.get("class", "?")
        p_dist = primary.get("distance_m")
        p_ttc = primary.get("ttc_s")
        dist_str = f"{p_dist:.1f}m" if p_dist is not None else "--"
        ttc_str = f"{p_ttc:.2f}s" if p_ttc is not None else "inf"
        lines.append(f"Primary: {p_class}  Dist:{dist_str}  TTC:{ttc_str}  [{level}]")
    else:
        lines.append(f"Status: {level}")

    if status:
        lines.append(status)

    # Draw with dark background for readability
    y_start = 10
    line_height = 22
    max_width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        max_width = max(max_width, tw)

    panel_h = y_start + line_height * len(lines) + 8
    panel_w = max_width + 24
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (80, 80, 80), 1)

    for i, line in enumerate(lines):
        y = y_start + (i + 1) * line_height
        # Color the warning line
        color = (230, 230, 230)
        if level == "DANGER" and "Primary:" in line:
            color = (0, 0, 255)
        elif level == "WARNING" and "Primary:" in line:
            color = (0, 165, 255)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def draw_reconnect_status(frame, attempt, max_attempts, delay):
    """Show reconnect status on a black frame."""
    h, w = frame.shape[:2]
    text = f"Reconnecting... ({attempt}/{max_attempts})  Next in {delay:.0f}s"
    cv2.putText(frame, text, (20, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press q/Esc to quit", (20, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    err = validate_args(args)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 1

    url = args.url
    target_w = args.width
    max_reconnect = args.reconnect
    reconnect_delay = min(args.reconnect_delay, 30.0)  # cap at 30s

    # ── Load pipeline (before opening camera so --help does not load model) ──
    print(f"Loading config: {args.config}", flush=True)
    cfg = load_config(args.config)
    print("Loading models...", flush=True)
    pipe = CollisionWarningSystem(cfg)

    # ── Open camera ──
    print(f"Connecting to: {url}", flush=True)
    cap, ok = open_camera(url)
    if not ok:
        print(f"\n[ERROR] Cannot open camera/video: {url}", file=sys.stderr)
        print("Troubleshooting:", file=sys.stderr)
        print("  1. Is IP Webcam running and 'Start Server' pressed?", file=sys.stderr)
        print("  2. Is the IP/port/path correct?", file=sys.stderr)
        print("  3. Are both devices on the same Wi-Fi network?", file=sys.stderr)
        print("  4. Is there a firewall blocking the connection?", file=sys.stderr)
        if os.path.exists(url):
            print(f"  5. Video file exists but could not be read: {url}", file=sys.stderr)
        return 1

    # Get true frame size from actual frame
    test_ok, test_frame = cap.read()
    if not test_ok or test_frame is None:
        print("[ERROR] Got camera handle but cannot read frames.", file=sys.stderr)
        cap.release()
        return 1

    src_w, src_h = get_frame_size(cap, test_frame)

    # Calculate scale
    scale = 1.0
    if target_w > 0 and src_w > target_w:
        scale = target_w / src_w
    proc_w = int(src_w * scale)
    proc_h = int(src_h * scale)

    print(f"Connected: source {src_w}x{src_h}"
          + (f" -> processing {proc_w}x{proc_h}" if scale < 1 else ""),
          flush=True)
    print(f"Press q or Esc to quit.", flush=True)

    # ── Benchmark setup ──
    benchmark = BenchmarkWriter() if args.benchmark else None
    if benchmark:
        print(f"Benchmark CSV: {benchmark.path}", flush=True)

    # ── Metrics ──
    metrics = FrameMetrics()
    shown_keys = False

    # Process the test frame we already read
    first_frame = test_frame
    if scale < 1.0:
        first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

    # ── Main loop with try/finally ──
    cap_released = False
    try:
        # Process first frame
        t0 = time.perf_counter()
        now = time.time()
        result = pipe.process_frame(first_frame, now)
        t1 = time.perf_counter()
        drawn, res = result[0], result[1] if len(result) > 1 else {}
        pipeline_ms = (t1 - t0) * 1000.0

        if isinstance(res, dict):
            if not shown_keys:
                shown_keys = True
                print(f"Pipeline result keys: {sorted(res)}", flush=True)
            metrics.tick(pipeline_ms, res.get("yolo_ms"), res.get("depth_ms"))
        else:
            metrics.tick(pipeline_ms)

        if not args.no_display:
            draw_overlay(drawn, metrics, res, src_w, src_h, proc_w, proc_h)
            cv2.imshow(WINDOW_NAME, drawn)
            cv2.waitKey(1)

        # Main frame loop
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                # ── Reconnect logic ──
                if max_reconnect <= 0:
                    print("Stream ended. No reconnect configured.", flush=True)
                    break

                reconnected = False
                for attempt in range(1, max_reconnect + 1):
                    print(f"Stream lost. Reconnect attempt {attempt}/{max_reconnect}...",
                          flush=True)

                    # Show reconnect status on screen
                    if not args.no_display:
                        status_frame = (drawn if drawn is not None
                                        else 128 * __import__("numpy").ones((proc_h, proc_w, 3),
                                                                           dtype="uint8"))
                        draw_reconnect_status(status_frame, attempt, max_reconnect,
                                              reconnect_delay)
                        cv2.imshow(WINDOW_NAME, status_frame)
                        key = cv2.waitKey(int(reconnect_delay * 1000))
                        if key in (27, ord("q")):
                            print("Quit during reconnect.", flush=True)
                            break
                    else:
                        time.sleep(reconnect_delay)

                    cap.release()
                    cap, reopen_ok = open_camera(url)
                    if reopen_ok:
                        print(f"Reconnected on attempt {attempt}.", flush=True)
                        reconnected = True
                        break

                if not reconnected:
                    print(f"Failed to reconnect after {max_reconnect} attempts.", flush=True)
                    break
                continue

            # Resize if needed
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

            # ── Pipeline ──
            t0 = time.perf_counter()
            now = time.time()
            result = pipe.process_frame(frame, now)
            t1 = time.perf_counter()
            drawn, res = result[0], result[1] if len(result) > 1 else {}
            pipeline_ms = (t1 - t0) * 1000.0

            yolo_ms_val = None
            depth_ms_val = None
            if isinstance(res, dict):
                if not shown_keys:
                    shown_keys = True
                    print(f"Pipeline result keys: {sorted(res)}", flush=True)
                yolo_ms_val = res.get("yolo_ms")
                depth_ms_val = res.get("depth_ms")
            metrics.tick(pipeline_ms, yolo_ms_val, depth_ms_val)

            # ── Overlay ──
            if not args.no_display:
                draw_overlay(drawn, metrics, res, src_w, src_h, proc_w, proc_h)
                cv2.imshow(WINDOW_NAME, drawn)
                key = cv2.waitKey(1)
                if key in (27, ord("q")):
                    print("Quit by user.", flush=True)
                    break

            # ── Benchmark row ──
            if benchmark and isinstance(res, dict):
                primary = res.get("primary")
                benchmark.write({
                    "frame": metrics.total_frames,
                    "timestamp": now,
                    "src_w": src_w, "src_h": src_h,
                    "proc_w": proc_w, "proc_h": proc_h,
                    "loop_fps": round(metrics.sliding_fps, 2),
                    "pipeline_ms": round(pipeline_ms, 2),
                    "yolo_ms": round(yolo_ms_val, 2) if yolo_ms_val else "",
                    "depth_ms": round(depth_ms_val, 2) if depth_ms_val else "",
                    "detections": res.get("detections", 0),
                    "tracks": res.get("tracks", 0),
                    "level": res.get("level", "NORMAL"),
                    "primary_class": primary.get("class", "") if primary else "",
                    "primary_distance_m": (round(primary["distance_m"], 2)
                                           if primary and primary.get("distance_m") is not None
                                           else ""),
                    "primary_ttc_s": (round(primary["ttc_s"], 2)
                                     if primary and primary.get("ttc_s") is not None
                                     else ""),
                })

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C.", flush=True)
    except Exception as exc:
        print(f"\n[ERROR] Pipeline exception: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up resources
        if cap is not None and not cap_released:
            cap.release()
            cap_released = True
        cv2.destroyAllWindows()
        if pipe is not None:
            pipe.close()
        if benchmark:
            benchmark.close()
            print(f"Benchmark saved: {benchmark.path}", flush=True)

    elapsed = time.perf_counter() - metrics._start_time
    print(f"Done: {metrics.total_frames} frames in {elapsed:.1f}s "
          f"({metrics.overall_fps:.1f} FPS overall)", flush=True)
    if metrics.total_frames > 20:
        print(f"Pipeline p50: {metrics.p50_pipeline_ms:.1f}ms  "
              f"p95: {metrics.p95_pipeline_ms:.1f}ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())