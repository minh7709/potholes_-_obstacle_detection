"""TTC computation, warning levels, forward-road ROI and target selection.

Warning levels
--------------
    TTC  > warning_ttc (4.0 s)  ->  NORMAL
    TTC  in (danger_ttc, warning_ttc]  ->  WARNING
    TTC  <= danger_ttc (2.0 s)  ->  DANGER

Both thresholds live in config.yaml, never hard-coded here.

TTC is only defined for a genuinely closing target. When closing_speed <= 0 the
gap is not shrinking, so TTC is infinity. That is a deliberate distinction from
"no data": an object that is moving away has a *known safe* state, whereas a
target with too little history is *unknown*, and the two must not be conflated.
"""

import numpy as np

from app.fcw.detector import _ALIASES, _normalise

NORMAL = "NORMAL"
WARNING = "WARNING"
DANGER = "DANGER"


def _class_matches(name, wanted):
    """Compare class names across model vocabularies.

    The detection model reports whatever IT calls the class ('motorbike'),
    while config and the UI speak the common name ('motorcycle'). Comparing raw
    strings made the collision logic silently never fire. Normalising both
    sides through the same alias table removes that failure.
    """
    if not name:
        return False
    actual = _normalise(name)
    actual = _normalise(_ALIASES.get(actual, actual))
    for candidate in wanted:
        key = _normalise(candidate)
        key = _normalise(_ALIASES.get(key, key))
        if actual == key:
            return True
    return False


def _f(cfg, key, default):
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default, cast=float)
        return default if v is None else float(v)
    except Exception:
        return default


def _i(cfg, key, default):
    if cfg is None:
        return default
    try:
        v = cfg.get(key, default, cast=int)
        return default if v is None else int(v)
    except Exception:
        return default


def classify_ttc(ttc, warning_ttc=None, danger_ttc=None):
    """Map a TTC (seconds, or inf) to a level. Boundaries are inclusive."""
    warning_ttc = 4.0 if warning_ttc is None else float(warning_ttc)
    danger_ttc = 2.0 if danger_ttc is None else float(danger_ttc)

    if ttc is None or not np.isfinite(ttc):
        return NORMAL
    if ttc <= danger_ttc:
        return DANGER
    if ttc <= warning_ttc:
        return WARNING
    return NORMAL


def classify_distance(distance, warning_distance_m=None, danger_distance_m=None):
    """Map a range (metres) to a level, independently of closing speed.

    Why this exists
    ---------------
    TTC is the right metric while you are closing on a target. It fails in
    exactly the situation that matters most: when the gap stops shrinking,
    closing_speed goes to zero, TTC becomes infinite, and the system reports
    NORMAL.

    A car stopped 3 m in front of you is the single most dangerous state there
    is, and pure TTC logic calls it NORMAL. Range therefore raises a warning on
    its own, and the two levels are combined by taking the worse of the two.
    """
    if distance is None or not np.isfinite(distance) or distance <= 0:
        return NORMAL
    if danger_distance_m is None:
        danger_distance_m = 3.5
    if warning_distance_m is None:
        warning_distance_m = 8.0

    if distance <= float(danger_distance_m):
        return DANGER
    if distance <= float(warning_distance_m):
        return WARNING
    return NORMAL


LEVEL_RANK = {NORMAL: 0, WARNING: 1, DANGER: 2}


def worse_level(*levels):
    """Return the most severe level given."""
    best = NORMAL
    for level in levels:
        if level and LEVEL_RANK.get(level, 0) > LEVEL_RANK.get(best, 0):
            best = level
    return best


def road_roi(frame_shape, cfg=None):
    """Forward-road ROI as (x1, y1, x2, y2) in pixels.

    For a forward-facing dashcam the drivable area is a wedge in the lower-centre
    of the frame. Vehicles at the far margins are typically oncoming traffic,
    on the shoulder, or roadside parked, which is not a collision risk for the
    ego vehicle. Expressed as fractions so it adapts to any resolution.
    """
    h, w = frame_shape[:2]
    return (
        int(w * _f(cfg, "roi.x_min", 0.20)),
        int(h * _f(cfg, "roi.y_min", 0.30)),
        int(w * _f(cfg, "roi.x_max", 0.80)),
        int(h * _f(cfg, "roi.y_max", 0.97)),
    )


def is_in_roi(detection, roi):
    """True when the box centre lies inside the ROI."""
    x1, y1, x2, y2 = roi
    cx = detection.get("center_x")
    cy = detection.get("center_y")
    if cx is None or cy is None:
        x1b, y1b, x2b, y2b = detection.get("box", (0, 0, 0, 0))
        cx = (x1b + x2b) * 0.5
        cy = (y1b + y2b) * 0.5
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def center_score(detection, frame_width, cfg=None):
    """0..1 score for proximity to the horizontal centre of the frame.

    Objects directly ahead matter much more than objects at the image edge,
    where they are usually in an adjacent lane or off the road.
    """
    if not frame_width:
        return 0.0
    cx = detection.get("center_x")
    if cx is None:
        return 0.0
    offset = abs(cx - frame_width * 0.5) / (frame_width * 0.5)
    tol = max(1e-3, _f(cfg, "roi.center_tolerance", 0.35))
    return float(max(0.0, 1.0 - offset / tol))


def warning_corridor(frame_shape, cfg=None):
    """The narrow central band where a warning may be raised, as (x1,y1,x2,y2).

    A windshield camera is an approximation of the driver's eyeline, so the frame
    centre approximates where the driver is looking. The wide ROI above says "this
    is a road object worth tracking"; this corridor says "this is directly ahead,
    so it is worth interrupting the driver for".

    Anything outside it is still detected, still distance-measured and still drawn
    with its telemetry, it just cannot escalate. That matters because alarming on
    roadside cones or oncoming traffic teaches a driver to ignore the alarm.
    """
    h, w = frame_shape[:2]
    return (
        int(w * _f(cfg, "roi.warn_x_min", 0.33)),
        int(h * _f(cfg, "roi.warn_y_min", 0.32)),
        int(w * _f(cfg, "roi.warn_x_max", 0.67)),
        int(h * _f(cfg, "roi.warn_y_max", 0.97)),
    )


def in_corridor(detection, corridor):
    """True when the object's centre is inside the warning corridor."""
    x1, y1, x2, y2 = corridor
    cx = detection.get("center_x")
    cy = detection.get("center_y")
    if cx is None or cy is None:
        bx1, by1, bx2, by2 = detection.get("box", (0, 0, 0, 0))
        cx = (bx1 + bx2) * 0.5
        cy = (by1 + by2) * 0.5
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def compute_risk(detection, history, roi, frame_width, cfg=None, corridor=None):
    """Full risk assessment for one tracked object.

    Combines the geometric signals: ROI membership, proximity to centre,
    distance, and TTC. Returns a dict consumed by the renderer, the CSV writer
    and the audio trigger.
    """
    warning_ttc = _f(cfg, "collision.warning_ttc", 4.0)
    danger_ttc = _f(cfg, "collision.danger_ttc", 2.0)
    min_distance = _f(cfg, "collision.min_distance_m", 0.5)
    warn_distance = _f(cfg, "collision.warning_distance_m", 8.0)
    danger_distance = _f(cfg, "collision.danger_distance_m", 3.5)

    ttc, ttc_valid = history.time_to_collision()
    distance = history.distance
    level_ttc = classify_ttc(ttc, warning_ttc, danger_ttc)
    level_dist = classify_distance(distance, warn_distance, danger_distance)
    # The worse of the two wins: range catches the stationary case that TTC
    # cannot express, TTC catches the fast-approach case that range cannot.
    raw_level = worse_level(level_ttc, level_dist)

    center = center_score(detection, frame_width, cfg)
    in_roi = is_in_roi(detection, roi)

    # Central-corridor gate. An object beside the ego vehicle or in an adjacent
    # lane must not sound the alarm, however close it is: a motorcycle stopped
    # 2 m away in the next lane is not a collision risk for this vehicle.
    if corridor is not None:
        centered = in_corridor(detection, corridor)
    else:
        centered = in_roi
    level = raw_level if centered else NORMAL

    if distance is None or not np.isfinite(distance):
        proximity = 0.0
    else:
        proximity = float(np.clip((30.0 - distance) / 30.0, 0.0, 1.0))

    if ttc_valid and np.isfinite(ttc):
        ttc_risk = float(np.clip(1.0 - (ttc / max(warning_ttc, 1e-3)), 0.0, 1.0))
    else:
        ttc_risk = 0.0

    return {
        "level": level,
        # The level this object WOULD have had without the central-corridor gate.
        # Kept so the UI can explain "TTC 1.8s but NORMAL" instead of looking broken.
        "level_unrestricted": raw_level,
        "in_warning_corridor": centered,
        "level_from_ttc": level_ttc,
        "level_from_distance": level_dist,
        # True when range alone triggered the warning, i.e. TTC is infinite.
        # Worth surfacing: it means the target is close but not closing, which is
        # a different failure mode than a fast approach.
        "proximity_trigger": (level == DANGER and level_dist == DANGER
                              and level_ttc != DANGER),
        "ttc": ttc,
        "ttc_valid": ttc_valid,
        "score": 0.65 * ttc_risk + 0.25 * proximity + 0.10 * center,
        "proximity": proximity,
        "center_score": center,
        "in_roi": in_roi,
        "imminent": bool(distance is not None and np.isfinite(distance)
                         and 0 < distance <= min_distance),
        # Whether the distance is measured or inferred; surfaced in the CSV.
        "distance_source": history.distance_source,
    }


def select_primary_target(risk_records, frame_shape, cfg=None, primary_classes=None):
    """Choose the ONE object that drives the audio and the DANGER banner.

    Only one target is ever announced. Overlapping alarms make a driving aid
    unusable, so priority is explicit:
      1. DANGER before WARNING
      2. within a level: lower TTC first
      3. then closer to frame centre
      4. then closer to the camera
    """
    if not risk_records:
        return None

    if primary_classes is None:
        primary_classes = ["motorcycle"]

    frame_width = frame_shape[1]
    rank = {DANGER: 0, WARNING: 1, NORMAL: 2}

    candidates = [
        r for r in risk_records
        if r["risk"]["level"] in (DANGER, WARNING)
        and _class_matches(r["history"].class_name, primary_classes)
        and r["risk"]["in_roi"]
    ]

    # A very close object counts even if TTC could not be computed yet: it is a
    # real hazard regardless of the speed estimate being unavailable.
    if not candidates:
        candidates = [
            r for r in risk_records
            if r["risk"]["imminent"]
            and _class_matches(r["history"].class_name, primary_classes)
            and r["risk"]["in_roi"]
        ]

    if not candidates:
        return None

    def sort_key(r):
        ttc = r["risk"]["ttc"]
        ttc_key = ttc if (ttc is not None and np.isfinite(ttc)) else 1e6
        dist = r["history"].distance
        dist_key = dist if (dist is not None and np.isfinite(dist)) else 1e6
        return (rank[r["risk"]["level"]], ttc_key,
                -r["risk"]["center_score"], dist_key)

    candidates.sort(key=sort_key)
    return candidates[0]


def is_confirmed(level, consecutive_count, confirmed_frames, history=None, min_hits=3):
    """Require sustained evidence before escalating a warning.

    Depth noise produces a spurious very low TTC for single frames. Requiring N
    consecutive frames at the same level removes almost all of those at
    10-20 FPS, adding only ~0.1-0.2 s of latency.
    """
    if level == NORMAL:
        return False
    if history is not None and history.frames_seen < max(int(min_hits), 1):
        return False
    return consecutive_count >= int(confirmed_frames)