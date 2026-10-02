"""On-screen rendering: ROI, boxes, telemetry and performance HUD."""

import numpy as np

from app.fcw import risk as risk_mod
from app.fcw.risk import DANGER, NORMAL, WARNING, road_roi

# BGR
COLORS = {
    NORMAL: (0, 200, 0),
    WARNING: (0, 165, 255),
    DANGER: (0, 0, 255),
}
PRIMARY_COLOR = (255, 220, 0)

FONT = 0.5
THICK = 2


class Visualizer:
    """Draws the overlay for one frame.

    Kept free of inference concerns: it reads telemetry that the pipeline has
    already computed, and holds only drawing state (the FPS sample window).
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.show_perf = bool(_get(cfg, "visual.show_performance", True))
        self._fps_times = []
        self._sampler = None

    def set_sampler(self, sampler):
        """Attach a FrameSampler so the HUD can report decimation honestly."""
        self._sampler = sampler

    def _update_fps(self, now):
        self._fps_times.append(now)
        if len(self._fps_times) > 30:
            self._fps_times.pop(0)
        if len(self._fps_times) < 2:
            return 0.0
        span = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / span if span > 0 else 0.0

    # ── Main entry point ─────────────────────────────────────────────
    def render(self, frame, records, primary, ego_speed=0.0, detector=None,
               depth=None, depth_computed=False, source_label="", level=NORMAL,
               corridor=None):
        import cv2

        canvas = frame.copy()
        h, w = canvas.shape[:2]
        now = _now()

        # ROI first so boxes draw on top of it.
        rx1, ry1, rx2, ry2 = road_roi(canvas.shape, self.cfg)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 200, 0), 1)
        cv2.putText(canvas, "ROAD ROI", (rx1 + 6, ry1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        # Warning corridor: the only band that may raise an alarm. Drawn so it is
        # obvious why an object was or was not announced.
        if corridor is not None:
            cx1, cy1, cx2, cy2 = corridor
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 215, 255), 2)
            cv2.putText(canvas, "CANH BAO", (cx1 + 6, cy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        for record in records:
            self._draw_object(canvas, record, record is primary)

        if primary is not None and primary["risk"]["level"] == DANGER:
            bar = max(6, int(h * 0.014))
            cv2.rectangle(canvas, (0, h - bar), (w, h), (0, 0, 255), -1)

        self._draw_hud(canvas, records, primary, now, ego_speed,
                       detector, depth, depth_computed, source_label)

        # Big status banner, so the state is readable at a glance.
        if level in (WARNING, DANGER):
            self._draw_banner(canvas, level, primary)

        return canvas

    # ── Objects ──────────────────────────────────────────────────────
    def _draw_object(self, canvas, record, is_primary):
        import cv2

        det = record["detection"]
        hist = record["history"]
        risk = record["risk"]
        level = risk["level"]

        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = PRIMARY_COLOR if is_primary else COLORS.get(level, COLORS[NORMAL])
        thickness = THICK + 2 if is_primary else THICK
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        lines = self._label_lines(det, hist, risk, is_primary)
        self._draw_label(canvas, lines, x1, y1, color)

    def _label_lines(self, det, hist, risk, is_primary):
        ttc = risk.get("ttc")
        ttc_text = "inf" if (ttc is None or not np.isfinite(ttc)) else f"{ttc:.2f}s"
        dist = "--" if hist.distance is None else f"{hist.distance:.1f}m"
        speed = f"{hist.closing_speed:.1f}m/s" if hist.speed_valid else "--m/s"
        tid = det.get("track_id")
        tid_text = tid if tid is not None else "?"
        level = risk["level"]

        source = hist.distance_source
        marker = {"measured": "M", "interpolated": "I", "held": "H"}.get(source, "")

        lines = [
            f"ID:{tid_text} {det.get('class_name')} {det.get('confidence', 0):.2f}",
            f"Dist:{dist}{(' ' + marker) if marker else ''} Closing:{speed}",
            f"TTC:{ttc_text} {level}",
        ]
        # Say WHY it is warning when range, not TTC, is the reason. "TTC inf
        # DANGER" looks like a contradiction unless explained on screen.
        if risk.get("proximity_trigger"):
            lines.append("!! QUÁ GẦN")
        # Would have warned but sits outside the corridor: say so, otherwise a
        # red-looking box reading NORMAL looks like a bug.
        if (not risk.get("in_warning_corridor", True)
                and risk.get("level_unrestricted") not in (None, NORMAL)):
            lines.append("ngoai CANH BAO")
        if is_primary:
            lines.append("<< MỤC TIÊU >>")
        return lines

    def _draw_label(self, canvas, lines, x1, y1, color):
        """Dark panel with level-coloured text.

        An earlier version filled the panel with the full level colour, which
        hid the very object the box was pointing at. Contrast now comes from the
        text and the box outline, so the vehicle stays visible.
        """
        import cv2

        line_h = 16
        width = 0
        for text in lines:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT, 1)
            width = max(width, tw)
        height = line_h * len(lines) + 6

        h, w = canvas.shape[:2]
        bx1 = max(0, min(x1, w - 2))
        # Prefer above the box; drop below when there is no room at the top.
        by1 = y1 - height
        if by1 < 0:
            by1 = min(h - 2, y1 + 4)   # no room above: drop below the box
        by1 = max(0, by1)
        bx2 = min(w - 2, bx1 + width + 10)
        by2 = min(h - 1, by1 + height)
        if by2 <= by1:
            by2 = min(h - 1, by1 + 2)

        overlay = canvas
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 1)
        for i, text in enumerate(lines):
            text_color = (255, 255, 255)
            if text.startswith("!!") or "MỤC TIÊU" in text:
                text_color = color if color != COLORS[NORMAL] else (255, 255, 255)
            cv2.putText(canvas, text, (bx1 + 5, by1 + line_h * (i + 1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT, text_color, 1, cv2.LINE_AA)

    # ── HUD ──────────────────────────────────────────────────────────
    def _draw_hud(self, canvas, records, primary, now, ego_speed, detector,
                  depth, depth_computed, source_label):
        import cv2

        h, w = canvas.shape[:2]
        fps = self._update_fps(now)

        sampler = self._sampler
        stats = sampler.stats() if sampler is not None else {}

        tracked = sum(1 for r in records if r["detection"].get("track_id") is not None)
        danger = sum(1 for r in records if r["risk"]["level"] == DANGER)
        warn = sum(1 for r in records if r["risk"]["level"] == WARNING)

        lines = [
            f"Processing: {fps:4.1f} FPS   Input: {stats.get('input_fps', 0):.0f} FPS "
            f"(step {stats.get('frame_step', 1)})   Tracks: {len(records)}/{tracked}",
            f"Ego: {ego_speed:5.1f} m/s   YOLO: {_ms(detector)}   "
            f"Depth: {_ms(depth)}{' *' if depth_computed else ''}",
            f"Depth interval: {getattr(depth, 'depth_interval', '?')}   "
            f"WARN:{warn}  DANGER:{danger}",
        ]
        if source_label:
            lines.append(f"Source: {source_label[:40]}")

        panel_w = min(430, w)
        panel_h = 20 + 19 * len(lines)
        cv2.rectangle(canvas, (0, 0), (panel_w, panel_h), (25, 25, 25), -1)
        cv2.rectangle(canvas, (0, 0), (panel_w, panel_h), (95, 95, 95), 1)

        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (10, 18 + i * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)

        # Telemetry for the primary target, in a second panel.
        if primary is not None:
            self._draw_primary_panel(canvas, primary, ego_speed)

    def _draw_primary_panel(self, canvas, primary, ego_speed):
        import cv2

        hist = primary["history"]
        risk = primary["risk"]
        det = primary["detection"]
        ttc = risk["ttc"]

        h, w = canvas.shape[:2]
        line_h = 22
        height = line_h * 5 + 12
        top = h - height - 10
        left = max(0, w - 300)

        cv2.rectangle(canvas, (left, top), (left + 290, top + height), (25, 25, 25), -1)
        cv2.rectangle(canvas, (left, top), (left + 290, top + height), PRIMARY_COLOR, 1)

        rows = [
            (f"Distance: {_m(hist.distance)}", (255, 255, 255)),
            (f"Closing: {_v(hist.closing_speed)}", (255, 255, 255)),
            (f"TTC: {'inf' if (ttc is None or not np.isfinite(ttc)) else f'{ttc:.2f}s'}",
             COLORS.get(risk["level"], COLORS[NORMAL])),
            (f"STATUS: {risk['level']}", COLORS.get(risk["level"], COLORS[NORMAL])),
            (f"Source: {hist.distance_source}   ID:{det.get('track_id')}", (200, 200, 200)),
        ]
        for i, (text, color) in enumerate(rows):
            cv2.putText(canvas, text, (left + 10, top + 24 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _draw_banner(self, canvas, level, primary):
        import cv2

        h, w = canvas.shape[:2]
        color = COLORS.get(level, COLORS[WARNING])
        text = level
        if level == WARNING:
            text = "  WARNING "
        elif level == DANGER:
            text = "  DANGER  "

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        x = max(0, (w - tw) // 2)
        y = max(th + 10, int(h * 0.12))
        cv2.rectangle(canvas, (x - 12, y - th - 10), (x + tw + 12, y + 10), color, -1)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3, cv2.LINE_AA)


def _get(cfg, key, default):
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default)
        return default if v is None else v
    except Exception:
        return default


def _ms(component):
    value = getattr(component, "inference_ms", 0.0) if component else 0.0
    return f"{value:5.0f}ms"


def _m(value):
    return "--" if value is None else f"{value:.1f}m"


def _v(value):
    return "--" if value is None else f"{value:.2f}m/s"


def _now():
    import time
    return time.time()