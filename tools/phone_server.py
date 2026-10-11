"""Phone camera in, annotated frames out. Processing runs here, not on the phone.

The phone only supplies pictures. Everything downstream — detection, tracking,
depth, risk — is the existing app/fcw pipeline, which was written frame-at-a-time
exactly so it could serve a video file, a live camera and this API unchanged.

    python tools/phone_server.py --port 8000

Then open http://<this machine's LAN ip>:8000 on the phone. The page asks for the
rear camera, posts frames here, and shows the annotated result. Open /view on the
laptop to watch the same stream full size.

Nothing has to be installed on the phone beyond a browser.
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.fcw.config_loader import load_config          # noqa: E402
from app.fcw.pipeline import CollisionWarningSystem    # noqa: E402

PAGE = """<!doctype html><meta charset=utf-8>
<title>FCW phone</title>
<style>body{background:#111;color:#eee;font:14px system-ui;margin:0;padding:8px}
img{width:100%;max-width:760px;display:block;background:#000}
#s{white-space:pre-wrap;padding:6px 0}</style>
<h3>Camera goc</h3>
<img id=v autoplay muted playsinline>
<h3>Ket qua nhanh (may tinh xu ly)</h3>
<img id=a>
<div id=s>...</div>
<canvas id=c hidden></canvas>
<script>
const v=document.getElementById('v'),c=document.getElementById('c'),s=document.getElementById('s'),a=document.getElementById('a');
const x=c.getContext('2d');
(async()=>{
  try{
    // exact:'environment' fails outright when the browser does not label a rear
    // camera that way, and it was rejecting the whole request on this phone.
    // Try the strict form first, then relax: ideal facingMode, then no preference.
    const tries=[
      {video:{facingMode:{exact:'environment'}}},
      {video:{facingMode:'environment'}},
      {video:{facingMode:{ideal:'environment'}}},
      {video:true}
    ];
    let st=null,err=null;
    for(const c of tries){
      try{ st=await navigator.mediaDevices.getUserMedia(
             Object.assign({},c,{audio:false})); break; }
      catch(e){ err=e; }
    }
    if(!st){ s.textContent='Khong mo duoc camera: '+(err&&err.name)+' - '+(err&&err.message);
             console.error(err); return; }
    v.srcObject=st;
    const tr=v.srcObject.getVideoTracks()[0];
    const set=tr.getSettings?tr.getSettings():{};
    s.textContent='Camera: '+(set.facingMode||'?')+' '+(set.width||'?')+'x'+(set.height||'?');
  }catch(e){s.textContent='Camera bi tu choi: '+e.message;return;}
  await v.play();
  c.width=960;c.height=720;
  setInterval(async()=>{
    if(!v.videoWidth)return;
    x.drawImage(v,0,0,c.width,c.height);
    const b=await new Promise(r=>c.toBlob(r,'image/jpeg',0.7));
    try{
      const r=await fetch('/frame',{method:'POST',body:b});
      const j=await r.json();
      s.textContent='fps '+j.fps+'  YOLO '+j.yolo_ms+'ms  depth '+j.depth_ms+
        'ms  xe '+j.vehicles+'  canh bao '+j.alert+'  drop '+j.dropped;
      // The phone only sends frames; the annotated result comes back as a separate
      // image, so it has to be fetched on its own or the phone shows raw video only.
      a.src='/latest.jpg?'+Date.now();
    }catch(e){s.textContent='';}
  },120);
})();
</script>"""

VIEW = """<!doctype html><meta charset=utf-8><title>FCW laptop</title>
<style>body{background:#111;margin:0}img{width:100%;display:block}</style>
<img src=/latest.jpg>
<script>setInterval(()=>{document.querySelector('img').src='/latest.jpg?'+Date.now()},150)</script>"""

latest_jpeg = {"data": None, "t": 0.0}
lock = threading.Lock()
counters = {"done": 0, "dropped": 0, "t0": time.time(), "fps": 0.0, "yolo_ms": 0.0,
            "depth_ms": 0.0, "vehicles": 0, "alert": "-"}


def count_vehicles(res):
    """The key has been a count and a list at different points; accept both."""
    for key in ("vehicles", "detections", "objects"):
        v = res.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, (list, tuple)):
            return len(v)
    return 0


def build(cfg_path):
    cfg = load_config(cfg_path)
    return CollisionWarningSystem(cfg)


class Handler(BaseHTTPRequestHandler):
    system = None

    def log_message(self, *_):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/latest.jpg"):
            with lock:
                data = latest_jpeg["data"]
            if data is None:
                return self._send(404, b"", "image/jpeg")
            return self._send(200, data, "image/jpeg")
        if self.path.startswith("/view"):
            return self._send(200, VIEW.encode(), "text/html; charset=utf-8")
        return self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        if not self.path.startswith("/frame"):
            return self._send(404, b"", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return self._send(400, b"{}", "application/json")
        raw = self.rfile.read(n)
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            counters["dropped"] += 1
            return self._send(400, b'{"error":"jpeg"}', "application/json")

        try:
            out, res = self.system.process_frame(frame, time.time())
        except TypeError:
            out = self.system.process_frame(frame, time.time())
            res = {}
        except Exception as e:                       # keep the stream alive
            counters["dropped"] += 1
            return self._send(500, json.dumps({"error": str(e)}).encode(),
                              "application/json")

        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with lock:
                latest_jpeg["data"] = enc.tobytes()
                latest_jpeg["t"] = time.time()

        counters["done"] += 1
        elapsed = time.time() - counters["t0"]
        counters["fps"] = counters["done"] / elapsed if elapsed > 0 else 0.0
        if isinstance(res, dict):
            for key in ("yolo_ms", "depth_ms"):
                v = res.get(key)
                if isinstance(v, (int, float)):
                    counters[key] = v
            counters["vehicles"] = count_vehicles(res)
            counters["alert"] = str(res.get("level", res.get("status", "-")))

        self._send(200, json.dumps(counters).encode(), "application/json")


def stream_loop(system, url, view_only):
    """Pull frames from a network camera and run them through the pipeline.

    This path needs no browser and no secure context: the camera sends JPEG over the
    LAN and OpenCV reads it directly, which is what makes it work without a cable
    where getUserMedia cannot.
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Không mở được stream: {url}", flush=True)
        print("Kiểm tra IP trong URL, và app IP Webcam đã bấm Start Server chưa.", flush=True)
        return
    print(f"Đang đọc {url}  ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})", flush=True)

    win = None
    if view_only:
        win = cv2.namedWindow("FCW", cv2.WINDOW_NORMAL)

    last = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = time.time()
        if t - last < 1e-3:
            continue
        last = t

        try:
            out, res = system.process_frame(frame, t)[:2]
        except Exception as e:
            print(f"Lỗi pipeline: {e}", flush=True)
            continue

        counters["done"] += 1
        elapsed = time.time() - counters["t0"]
        counters["fps"] = counters["done"] / elapsed if elapsed > 0 else 0.0
        if isinstance(res, dict):
            for key in ("yolo_ms", "depth_ms"):
                if isinstance(res.get(key), (int, float)):
                    counters[key] = res[key]
            counters["vehicles"] = count_vehicles(res)
            counters["alert"] = str(res.get("level", res.get("status", "-")))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with lock:
                latest_jpeg["data"] = enc.tobytes()
                latest_jpeg["t"] = t
        if win:
            cv2.imshow("FCW", out)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            print("\r fps %.1f  YOLO %.0fms  depth %.0fms  xe %d  %s  " % (
                counters["fps"], counters["yolo_ms"], counters["depth_ms"],
                counters["vehicles"], counters["alert"]), end="", flush=True)
    if win:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--video", default=None,
                    help="URL của IP camera, ví dụ http://192.168.1.42:8080/video")
    ap.add_argument("--window", action="store_true",
                    help="hiện cửa sổ kết quả ngay trên máy tính")
    args = ap.parse_args()

    # config.yaml names assets relatively ("assets/best.pt"), so the pipeline only
    # resolves them from the repository root. Chdir makes the working directory
    # irrelevant instead of forcing the caller to cd first.
    os.chdir(ROOT)

    print("Nap model...", flush=True)
    Handler.system = build(args.config)
    print("San sang. May tinh: http://0.0.0.0:%d  (dien thoai dung IP may tinh)" % args.port, flush=True)
    if args.video:
        stream_loop(Handler.system, args.video, not args.window)
        return
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()