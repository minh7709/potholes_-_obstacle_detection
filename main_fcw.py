"""RoadGuard — Forward Collision Warning. Command line entry point.

    python main_fcw.py --source 0                  # webcam
    python main_fcw.py --source video.mp4          # offline analysis
    python main_fcw.py --source image.jpg          # single image (distance only)
    python main_fcw.py --source rtsp://cam/stream  # network camera

Live keys:  Q/ESC quit   P pause   M mute/unmute

Every tunable lives in config.yaml; see --set for one-off overrides.
"""

import argparse
import contextlib
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.fcw.config_loader import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    parse_set_overrides,
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main_fcw.py",
        description="Real-time forward collision warning: YOLO + ByteTrack + "
                    "Depth Anything V2 + TTC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="0",
                        help="Camera index (0), video file, image file, or stream URL.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML configuration file.")

    # ── Performance ────────────────────────────────────────────────
    parser.add_argument("--target-fps", type=float, default=None,
                        help="Processing FPS. Lower is faster. Default 10.")
    parser.add_argument("--depth-interval", type=int, default=None,
                        help="Run depth every N processed frames. Default 5.")
    parser.add_argument("--depth-resolution", type=int, default=None,
                        help="Depth network input height. 224 fast, 350 balanced, 518 slow.")
    parser.add_argument("--max-width", type=int, default=None,
                        help="Downscale frames wider than this before inference.")
    parser.add_argument("--max-height", type=int, default=None,
                        help="Downscale frames taller than this before inference.")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable automatic quality reduction when too slow.")
    parser.add_argument("--threads", type=int, default=None,
                        help="Torch CPU thread count.")

    # ── Models ─────────────────────────────────────────────────────
    parser.add_argument("--yolo-model", default=None,
                        help="YOLO weights. Default assets/best.pt (15-class road-obstacle model).")
    parser.add_argument("--depth-model", default=None, help="Depth Anything V2 model id.")
    parser.add_argument("--tracker", default=None,
                        help="bytetrack.yaml or botsort.yaml.")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated classes to detect.")
    parser.add_argument("--conf", type=float, default=None, help="Detection threshold.")

    # ── Safety thresholds (never auto-adjusted) ────────────────────
    parser.add_argument("--warning-ttc", type=float, default=None,
                        help="TTC (s) at or below which WARNING is raised.")
    parser.add_argument("--danger-ttc", type=float, default=None,
                        help="TTC (s) at or below which DANGER is raised.")

    # ── Calibration ────────────────────────────────────────────────
    parser.add_argument("--depth-scale", type=float, default=None,
                        help="Depth correction factor. Overrides calib-* when set.")
    parser.add_argument("--calib-true-m", type=float, default=None,
                        help="True distance (m) of the calibration reference object.")
    parser.add_argument("--calib-model-m", type=float, default=None,
                        help="Distance (m) the depth model reports for that object.")
    parser.add_argument("--calibrate", type=float, default=None, metavar="TRUE_M",
                        help="Interactive depth calibration against a known distance.")
    parser.add_argument("--calibrate-cam", type=int, default=None, metavar="CAMERA",
                        help="Capture chessboard frames and solve camera intrinsics.")
    parser.add_argument("--calib-pattern", default="9x6",
                        help="Chessboard INNER corners, e.g. 9x6.")
    parser.add_argument("--calib-samples", type=int, default=15,
                        help="Chessboard images to capture.")

    # ── Output ─────────────────────────────────────────────────────
    parser.add_argument("--output-dir", default=None, help="Directory for outputs.")
    parser.add_argument("--save-video", action="store_true", help="Write annotated video.")
    parser.add_argument("--no-video", action="store_true", help="Skip annotated video.")
    parser.add_argument("--save-csv", action="store_true", help="Write events.csv.")
    parser.add_argument("--save-summary", action="store_true", help="Write summary.json.")
    parser.add_argument("--report-json", action="store_true",
                        help="Print the full run report as JSON on stdout.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run without adaptation to measure the true ceiling.")

    # ── Runtime ────────────────────────────────────────────────────
    parser.add_argument("--no-audio", action="store_true", help="Disable audio alerts.")
    parser.add_argument("--no-ego", action="store_true",
                        help="Disable ego-motion compensation (diagnostic only).")
    parser.add_argument("--ego-interval", type=int, default=None,
                        help="Run optical-flow ego estimation every N frames.")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip overlay drawing (telemetry-only runs).")
    parser.add_argument("--no-display", action="store_true", help="Headless: no window.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N processed frames (testing).")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=None,
                        help="Override any config key, e.g. --set depth.interval=3. "
                             "Repeatable.")

    return parser


def _apply_cli(cfg, args):
    """Fold CLI flags into the config tree. CLI wins over YAML."""
    if args.target_fps is not None:
        cfg.set("processing.target_fps", args.target_fps)
    if args.depth_interval is not None:
        cfg.set("depth.interval", max(1, args.depth_interval))
    if args.depth_resolution is not None:
        cfg.set("depth.processing_height", args.depth_resolution)
    if args.max_width is not None:
        cfg.set("input.max_width", args.max_width)
    if args.max_height is not None:
        cfg.set("input.max_height", args.max_height)
    if args.threads is not None:
        cfg.set("processing.torch_threads", args.threads)
    if args.no_adaptive or args.benchmark:
        cfg.set("performance.adaptive_enabled", False)
    if args.yolo_model:
        cfg.set("classes.yolo_model", args.yolo_model)
    if args.depth_model:
        cfg.set("depth.model", args.depth_model)
    if args.tracker:
        cfg.set("tracking.tracker", args.tracker)
    if args.classes:
        cfg.set("classes.targets", [c.strip() for c in args.classes.split(",") if c.strip()])
    if args.conf is not None:
        cfg.set("classes.conf_threshold", args.conf)
    if args.warning_ttc is not None:
        cfg.set("collision.warning_ttc", args.warning_ttc)
    if args.danger_ttc is not None:
        cfg.set("collision.danger_ttc", args.danger_ttc)
    if args.no_audio:
        cfg.set("warning.audio_enabled", False)
    if args.no_ego:
        cfg.set("ego_motion.enabled", False)
    if args.ego_interval is not None:
        cfg.set("ego_motion.interval", max(1, args.ego_interval))
    if args.no_render:
        cfg.set("visual.render", False)
    if args.output_dir:
        cfg.set("output.directory", args.output_dir)
    if args.save_video:
        cfg.set("output.save_annotated_video", True)
    if args.no_video:
        cfg.set("output.save_annotated_video", False)
    if args.save_csv:
        cfg.set("output.save_csv", True)
    if args.save_summary:
        cfg.set("output.save_summary", True)
    if args.depth_scale is not None:
        cfg.set("calibration.depth_scale", args.depth_scale)
    if args.calib_true_m is not None:
        cfg.set("calibration.calib_true_m", args.calib_true_m)
    if args.calib_model_m is not None:
        cfg.set("calibration.calib_model_m", args.calib_model_m)
    return cfg


def _configure_threads(cfg):
    threads = cfg.get("processing.torch_threads", 0, cast=int)
    if threads and threads > 0:
        try:
            import torch
            torch.set_num_threads(int(threads))
            print(f"[FCW] Torch threads: {threads}")
        except Exception:
            pass


def _print_banner(cfg, args):
    from app.fcw.config_loader import depth_interval, depth_height, input_max_size, target_fps

    print()
    print("=" * 66)
    print("  RoadGuard — Forward Collision Warning")
    print("=" * 66)
    print(f"  Source          : {args.source}")
    print(f"  YOLO            : {cfg.get('classes.yolo_model', 'assets/best.pt')}")
    print(f"  Depth model     : {cfg.get('depth.model')}")
    print(f"  Depth height    : {depth_height(cfg)} px   every {depth_interval(cfg)} frames")
    fps = target_fps(cfg)
    print(f"  Processing FPS  : {fps if fps else 'all frames'}")
    mw, mh = input_max_size(cfg)
    print(f"  Max input size  : {mw}x{mh}")
    print(f"  Tracker         : {cfg.get('tracking.tracker', 'bytetrack.yaml')}")
    print(f"  TTC thresholds  : WARNING<={cfg.get('collision.warning_ttc')}s, "
          f"DANGER<={cfg.get('collision.danger_ttc')}s (never auto-adjusted)")
    print(f"  Ego compensation: {cfg.get('ego_motion.enabled', True)}")
    print("=" * 66)


# ── Calibration modes ────────────────────────────────────────────────
def _run_depth_calibration(cfg, true_distance, source_desc):
    """One-point depth calibration.

    Measures the model's reading against one known distance and prints the
    scale factor. A single monocular model cannot be trusted without this: the
    metric checkpoints are close to correct on average but carry a systematic
    per-camera offset.
    """
    import cv2
    import numpy as np

    from app.fcw.depth import DepthEstimator
    from app.fcw.source import is_image_path, load_image, open_source, parse_source

    estimator = DepthEstimator(
        model_id=cfg.get("depth.model"),
        device=cfg.get("performance.device", "auto"),
        processing_height=cfg.get("depth.processing_height", 224, cast=int),
        cfg=cfg)

    print("\n" + "=" * 66)
    print("  DEPTH CALIBRATION")
    print("=" * 66)
    print(f"  Place ONE clear object at exactly {true_distance:.2f} m from the camera,")
    print("  centred in frame.")
    print("=" * 66)

    frame = None
    if is_image_path(str(source_desc)):
        frame = load_image(str(source_desc))
    else:
        cap = open_source(parse_source(source_desc))
        if cap is None:
            print("  Could not open source.")
            return None
        window = "FCW Calibration"
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except Exception:
            cv2.setLogLevel(3)
        print("  SPACE = capture    ESC/Q = abort")
        while True:
            ok, candidate = cap.read()
            if not ok:
                break
            preview = candidate.copy()
            cv2.putText(preview, f"Reference object at {true_distance:.2f} m",
                        (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            try:
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
            except Exception:
                key = 32
            if key == 32:
                frame = candidate.copy()
                break
            if key in (27, ord("q")):
                break
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        cap.release()

    if frame is None:
        print("  Aborted.")
        return None

    h, w = frame.shape[:2]
    readings = []
    for _ in range(5):
        depth_map = estimator.infer(frame)
        if depth_map is None:
            continue
        patch = depth_map[int(h * 0.35):int(h * 0.75), int(w * 0.35):int(w * 0.65)]
        values = patch[np.isfinite(patch) & (patch > 0)]
        if values.size:
            readings.append(float(np.median(values)))

    if not readings:
        print("  Could not obtain a depth reading.")
        return None

    model_distance = float(np.median(readings))
    scale = true_distance / model_distance if model_distance > 0 else 1.0
    spread = float(np.std(readings))

    print("\n  Result")
    print(f"    Model reported : {model_distance:.2f} m")
    print(f"    True distance  : {true_distance:.2f} m")
    print(f"    Scale factor   : {scale:.4f}")
    print(f"    Sample spread  : {spread:.3f} m (higher = less reliable)")
    print(f"\n  Apply with:  --depth-scale {scale:.4f}\n")
    return scale


def _run_camera_calibration(cfg, camera_index, pattern, samples, display=True):
    """Capture chessboard frames and solve pinhole intrinsics."""
    import glob

    import cv2

    from app.fcw.calibration import CALIB_DIR, CameraCalibration

    try:
        cols, rows = (int(v) for v in pattern.lower().split("x"))
    except ValueError:
        print(f"Invalid pattern '{pattern}'. Expected e.g. 9x6.")
        return 1

    image_dir = os.path.join(CALIB_DIR, "images")
    os.makedirs(image_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        return 1

    corners = (cols, rows)
    window = "Camera Calibration"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except Exception:
        display = False

    print("\n" + "=" * 66)
    print("  CAMERA CALIBRATION")
    print("=" * 66)
    print(f"  Pattern: {cols}x{rows} INNER corners, {samples} samples needed.")
    print("  Hold the board flat and fill the frame: centre, then all four edges,")
    print("  then tilted views. Keep every corner visible in each shot.")
    print("  SPACE = capture    ESC/Q = finish")
    print("=" * 66)

    captured = []
    try:
        while len(captured) < samples:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, found_corners = cv2.findChessboardCorners(gray, corners, None)
            preview = frame.copy()
            if found:
                preview = cv2.drawChessboardCorners(preview, corners, found_corners, found)
                cv2.putText(preview, f"Detected — SPACE to capture ({len(captured)}/{samples})",
                            (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(preview, "Chessboard not found",
                            (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            if display:
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 32 if found else -1
                time.sleep(0.03)
            if key in (27, ord("q")):
                break
            if key == 32 and found:
                path = os.path.join(image_dir, f"calib_{len(captured):02d}.jpg")
                cv2.imwrite(path, frame)
                captured.append(path)
                print(f"  Captured {len(captured)}/{samples}")
    finally:
        cap.release()
        if display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if len(captured) < 4:
        print(f"\nOnly {len(captured)} samples captured; need at least 4.")
        return 1

    calib = CameraCalibration.calibrate_from_images(captured, pattern_size=corners)
    path = calib.save()
    print(f"\n  Saved to {path}")
    return 0


# ── Main ─────────────────────────────────────────────────────────────
def main(argv=None):
    args = build_parser().parse_args(argv)

    # With --report-json, stdout must stay machine-readable, so every
    # diagnostic (banner, model loading, progress) is diverted to stderr and
    # only the JSON document reaches stdout. The real stdout is captured first
    # and handed to _main, since the redirected one points at stderr.
    if args.report_json:
        real_stdout = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):
            exit_code = _main(args, json_stream=real_stdout)
        return exit_code
    return _main(args)


def _main(args, json_stream=None):

    cfg = load_config(args.config, parse_set_overrides(args.set))
    cfg = _apply_cli(cfg, args)

    _print_banner(cfg, args)

    if args.calibrate_cam is not None:
        return _run_camera_calibration(
            cfg, args.calibrate_cam, args.calib_pattern,
            args.calib_samples, display=not args.no_display)

    if args.calibrate is not None:
        scale = _run_depth_calibration(cfg, args.calibrate, args.source)
        if scale is None:
            return 1
        print(f"  Then run with:  --depth-scale {scale:.4f}\n")
        return 0

    _configure_threads(cfg)

    from app.fcw.pipeline import CollisionWarningSystem
    from app.fcw.source import is_image_path, parse_source

    source = parse_source(args.source)

    # Load camera calibration when present; it is optional and purely additive.
    from app.fcw.calibration import CameraCalibration
    calib = CameraCalibration.load()
    if calib.is_valid and calib.has_distortion:
        print(f"[FCW] Camera calibration active: FOV "
              f"{calib.horizontal_fov_deg():.1f} deg, "
              f"distortion {'applied' if True else 'none'}")

    system = CollisionWarningSystem(cfg)
    system.source_label = str(args.source)

    # ── Single image ───────────────────────────────────────────────
    if is_image_path(str(args.source)):
        return _run_single_image(cfg, system, args)

    from app.fcw.runner import LiveRunner, VideoRunner

    is_live = isinstance(source, int) or _is_stream(source)

    try:
        if is_live:
            runner = LiveRunner(source, system, cfg,
                                verbose=not args.report_json)
            runner.display = not args.no_display
            report = runner.run()
        else:
            runner = VideoRunner(source, system, cfg, verbose=not args.report_json,
                                 max_frames=args.max_frames)
            runner.writer = _build_writer(cfg, args, source)
            report = runner.run()
            if runner.writer is not None:
                runner.writer.close()
    except KeyboardInterrupt:
        print("\n[FCW] Interrupted.")
        report = {"interrupted": True}
    finally:
        system.close()

    if args.report_json:
        stream = json_stream if json_stream is not None else sys.stdout
        print(json.dumps(report, indent=2, default=str), file=stream)
    else:
        _print_report(report, cfg)
    return 0


def _run_single_image(cfg, system, args):
    """Distance-only mode.

    A single frame has no time dimension, so closing speed and TTC are undefined.
    The pipeline runs a few passes over the same frame purely so the tracker and
    the depth statistics settle enough to display, and the output is labelled
    accordingly.
    """
    import cv2

    from app.fcw.source import load_image

    frame = load_image(str(args.source))
    if frame is None:
        return 1

    print("[FCW] Single image: distance only. TTC requires a video or camera "
          "(multiple frames over time).")

    annotated, result = system.process_frame(frame, 0.0, source_frame=0)
    for i in range(3):
        annotated, result = system.process_frame(frame, 0.05 * (i + 1), source_frame=i + 1)

    os.makedirs(cfg.get("output.directory", "output"), exist_ok=True)
    out_path = os.path.join(cfg.get("output.directory", "output"), "fcw_image.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"[FCW] Wrote {out_path}")
    _print_objects(result)
    return 0


def _build_writer(cfg, args, source):
    import cv2

    from app.fcw.output import ResultWriter

    directory = cfg.get("output.directory", "output")
    save_video = bool(cfg.get("output.save_annotated_video", True))
    if args.report_json:
        save_video = False

    if not save_video:
        return ResultWriter(directory, save_video=False,
                            save_csv=bool(cfg.get("output.save_csv", True)),
                            save_summary=bool(cfg.get("output.save_summary", True))).open()

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap = None
    if cap is not None:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280,
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720)
        cap.release()
    else:
        size = (1280, 720)

    mw = cfg.get("input.max_width", 1280, cast=int)
    mh = cfg.get("input.max_height", 720, cast=int)
    if size[0] > mw or size[1] > mh:
        scale = min(mw / size[0], mh / size[1])
        size = (int(size[0] * scale), int(size[1] * scale))

    return ResultWriter(
        directory,
        video_name=cfg.get("output.video_name", "annotated_video.mp4"),
        csv_name=cfg.get("output.csv_name", "events.csv"),
        summary_name=cfg.get("output.summary_name", "summary.json"),
        save_video=True,
        save_csv=bool(cfg.get("output.save_csv", True)),
        save_summary=bool(cfg.get("output.save_summary", True)),
        video_size=size,
        csv_every_frame=bool(cfg.get("output.csv_every_frame", True)),
    ).open()


def _print_report(report, cfg):
    print()
    print("=" * 66)
    print("  RUN REPORT")
    print("=" * 66)

    perf = report.get("performance") or {}
    if perf:
        print("  Throughput")
        print(f"    Input           : {perf.get('input_fps')} FPS "
              f"(frame step {perf.get('frame_step')})")
        print(f"    Processing      : {perf.get('processing_fps')} FPS average, "
              f"{perf.get('processing_fps_recent')} FPS recent")
        print(f"    YOLO            : {perf.get('yolo_fps')} FPS ({perf.get('yolo_ms')} ms)")
        print(f"    Depth           : {perf.get('depth_fps')} FPS ({perf.get('depth_ms')} ms)")
        print(f"    Frames processed: {perf.get('frames_processed')} of {perf.get('source_frames')}")
        duration = perf.get("video_duration_sec")
        analysed = perf.get("processed_duration_sec")
        elapsed = perf.get("processing_time_sec")
        if duration:
            print(f"    Video duration  : {duration} s")
        if analysed is not None:
            note = "  (run truncated, ratio uses analysed span)" if perf.get("truncated") else ""
            print(f"    Analysed span   : {analysed} s{note}")
        print(f"    Processing time : {elapsed} s")
        ratio = perf.get("speed_ratio")
        if ratio is not None:
            verdict = "FASTER than real time" if ratio >= 1.0 else "slower than real time"
            print(f"    Speed ratio     : {ratio}x  ({verdict})")

    detector = report.get("detector")
    if detector:
        print("  Model calls")
        print(f"    YOLO            : {detector.get('yolo_calls')}")
        print(f"    Depth           : {detector.get('depth_calls')}")

    output = report.get("output") or {}
    if output:
        print("  Results")
        print(f"    Warning frames  : {output.get('warning_frames')}")
        print(f"    Danger frames   : {output.get('danger_frames')}")
        print(f"    Warning events  : {output.get('warning_events')}")
        print(f"    Min TTC         : {output.get('min_ttc_sec')} s")
        print(f"    Min distance    : {output.get('min_distance_m')} m")

    changes = report.get("adaptive_changes") or []
    if changes:
        print("  Adaptive adjustments")
        for change in changes:
            print(f"    level {change['level']}: target_fps={change.get('target_fps')}, "
                  f"depth_interval={change.get('depth_interval')} "
                  f"({change.get('reason')})")

    print("  Outputs")
    print(f"    Directory       : {os.path.abspath(cfg.get('output.directory', 'output'))}")
    print("=" * 66)
    print("  Note: distance and TTC are ESTIMATES from monocular depth")
    print("  (typical error 10-25%). closing_speed is RELATIVE, not the")
    print("  target's true absolute speed. Calibrate with --calibrate before")
    print("  trusting absolute numbers. Prototype only — not a safety device.")
    print()


def _print_objects(result):
    print(f"\n  Detections: {result['detections']}")
    objects = result.get("objects") or []
    if not objects:
        print("  (none)")
        return
    print(f"  {'ID':>5} {'class':<11} {'dist':>7} {'src':<13} {'close':>7} {'TTC':>7}  status")
    for obj in objects:
        tid = obj["track_id"] if obj["track_id"] is not None else "-"
        dist = "  --" if obj["distance_m"] is None else f"{obj['distance_m']:5.2f}"
        speed = "  --" if obj["closing_speed_mps"] is None else f"{obj['closing_speed_mps']:5.2f}"
        ttc = "  inf" if obj["ttc_s"] is None else f"{obj['ttc_s']:5.2f}"
        print(f"  {str(tid):>5} {obj['class']:<11} {dist} "
              f"{obj['distance_source']:<13} {speed} {ttc}  {obj['level']}"
              f"{'' if obj['in_roi'] else '  (outside ROI)'}")


def _is_stream(source):
    return str(source).lower().startswith(
        ("http://", "https://", "rtsp://", "rtmp://"))


if __name__ == "__main__":
    sys.exit(main())