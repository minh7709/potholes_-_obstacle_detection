"""Phone camera over the network, processed on this machine, drawn in one window.

    python tools/live.py http://5.53.241.254:8080/video

The phone only supplies frames. Detection, tracking, depth and risk are the existing
app/fcw pipeline; nothing about it is reimplemented here.

Press q or Esc to quit.
"""
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.fcw.config_loader import load_config          # noqa: E402
from app.fcw.pipeline import CollisionWarningSystem    # noqa: E402


def parse_width(tokens):
    """Second argument, if numeric, is the target width. Height follows the source
    aspect. 1080p costs about 2.3x the pixels of 720p and the window rarely needs
    more, so shrinking first is the cheapest speedup available."""
    for t in tokens:
        try:
            return int(t)
        except ValueError:
            continue
    return 0


def main():
    url = "http://192.168.1.42:8080/video"
    target_w = parse_width(sys.argv[1:])
    if not sys.argv[1:]:
        url = "http://192.168.1.42:8080/video"
    elif sys.argv[1] not in ("-h", "--help"):
        url = sys.argv[1]

    print("Nap model...", flush=True)
    pipe = CollisionWarningSystem(load_config("config.yaml"))

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Không mở được camera: {url}", flush=True)
        print("IP Webcam trên điện thoại đã bấm Start Server chưa? IP trong URL đúng chưa?", flush=True)
        return 1
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1.0
    if target_w > 0 and src_w > target_w:
        scale = target_w / src_w
    print(f"Đã nối {url}  nguồn {src_w}x{src_h}"
          + (f"  -> xuống còn {int(src_w*scale)}x{int(src_h*scale)}" if scale < 1 else ""),
          flush=True)

    times, yolo, depth = deque(maxlen=30), deque(maxlen=30), deque(maxlen=30)
    shown_keys = False
    started = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        now = time.time()
        result = pipe.process_frame(frame, now)
        drawn, res = result[0], result[1] if len(result) > 1 else {}
        frames += 1
        times.append(now)

        if isinstance(res, dict):
            if not shown_keys:
                shown_keys = True
                print("Khoá trong res:", sorted(res), flush=True)
            for key, acc in (("yolo_ms", yolo), ("depth_ms", depth)):
                v = res.get(key)
                if isinstance(v, (int, float)):
                    acc.append(float(v))

        fps = (len(times) / (times[-1] - times[0])) if len(times) > 1 else 0.0
        def avg(a):
            return sum(a) / len(a) if a else 0.0

        lines = [
            f"fps {fps:4.1f}   YOLO {avg(yolo):3.0f}ms   depth {avg(depth):3.0f}ms"
            f"   {frames/(time.time()-started):.1f} FPS tong",
        ]
        for i, line in enumerate(lines):
            cv2.putText(drawn, line, (12, 30 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(drawn, line, (12, 30 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("FCW", drawn)
        if cv2.waitKey(1) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Ket thuc: {frames} khung, {(time.time()-started):.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())