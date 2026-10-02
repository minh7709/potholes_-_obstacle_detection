"""Loads config.yaml and exposes it as a flat, dotted-key namespace.

Design
------
Why a YAML file at all, when app/config.py already has constants?

Because the numbers that matter for performance (depth interval, target FPS,
input size) are exactly the ones you want to sweep while tuning on a real
video. Editing a YAML file is safer than editing code, and it keeps the
optimisation settings in one reviewable place.

Precedence (highest first):
    1. explicit overrides passed to load_config()
    2. environment variables (FCW_*)
    3. config.yaml
    4. the DEFAULT_CONFIG dict below

The file is deliberately optional: if it is missing the built-in defaults are
used, so the pipeline still runs on a fresh checkout.
"""

import copy
import os

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is in requirements
    yaml = None

# Absolute path resolution is anchored on the project root, not the CWD, so
# the CLI works from anywhere.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


class Config:
    """Dotted-key read access over a nested dict, with type coercion.

    `cfg.get("depth.interval", 5)` returns 5 whether the YAML nests it under
    `depth:` or it is missing entirely. Keeping lookups tolerant means a typo
    in the YAML degrades to a sensible default instead of crashing mid-run.
    """

    def __init__(self, data=None):
        self._data = data or {}

    def get(self, dotted_key, default=None, cast=None):
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        if cast is None or node is None:
            return node
        try:
            if cast is bool and isinstance(node, str):
                return node.strip().lower() in ("1", "true", "yes", "on")
            if cast is bool:
                return bool(node)
            return cast(node)
        except (TypeError, ValueError):
            return default

    def section(self, name):
        value = self._data.get(name)
        return value if isinstance(value, dict) else {}

    def set(self, dotted_key, value):
        """Write a value, creating intermediate dicts as needed."""
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                return
        node[parts[-1]] = value

    def as_dict(self):
        return copy.deepcopy(self._data)

    def __contains__(self, dotted_key):
        sentinel = object()
        return self.get(dotted_key, sentinel) is not sentinel


# Environment overrides. The key here is the YAML path; the value is the
# env var name. Only these are honoured, so an unrelated FCW_* variable
# cannot silently alter behaviour.
_ENV_MAP = {
    "input.max_width": ("FCW_INPUT_MAX_WIDTH", int),
    "input.max_height": ("FCW_INPUT_MAX_HEIGHT", int),
    "processing.target_fps": ("FCW_TARGET_FPS", float),
    "processing.torch_threads": ("FCW_TORCH_THREADS", int),
    "depth.interval": ("FCW_DEPTH_INTERVAL", int),
    "depth.processing_height": ("FCW_DEPTH_RESOLUTION", int),
    "depth.model": ("FCW_DEPTH_MODEL_ID", str),
    "depth.interpolation": ("FCW_DEPTH_INTERPOLATION", str),
    "classes.conf_threshold": ("FCW_CONF_THRESHOLD", float),
    "classes.iou_threshold": ("FCW_IOU_THRESHOLD", float),
    "tracking.tracker": ("FCW_TRACKER", str),
    "tracking.max_age": ("FCW_TRACK_MAX_AGE", int),
    "distance.alpha": ("FCW_DEPTH_EMA_ALPHA", float),
    "speed.ema_alpha": ("FCW_SPEED_EMA_ALPHA", float),
    "collision.warning_ttc": ("FCW_TTC_WARNING_SECONDS", float),
    "collision.danger_ttc": ("FCW_TTC_DANGER_SECONDS", float),
    "collision.confirm_frames": ("FCW_CONFIRM_FRAMES", int),
    "ego_motion.enabled": ("FCW_EGO_COMPENSATION", bool),
    "warning.audio_enabled": ("FCW_AUDIO_ENABLED", bool),
    "warning.warning_cooldown_sec": ("FCW_AUDIO_WARNING_COOLDOWN", float),
    "warning.danger_cooldown_sec": ("FCW_AUDIO_DANGER_COOLDOWN", float),
    "calibration.depth_scale": ("FCW_DEPTH_GAIN", float),
    "performance.adaptive_enabled": ("FCW_ADAPTIVE_ENABLED", bool),
}


def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base`."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path=None, overrides=None):
    """Load YAML config, apply env vars and explicit overrides. Returns Config."""
    path = path or os.environ.get("FCW_CONFIG_PATH") or DEFAULT_CONFIG_PATH

    data = {}
    if yaml is not None and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                data = loaded
        except Exception as exc:
            print(f"[FCW] Could not read {path}: {exc}. Using defaults.")

    # A bare --set key=value pair targets the dotted path directly.
    if overrides:
        data = _deep_merge(data, overrides)

    cfg = Config(data)

    for dotted, (env_name, caster) in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        value = cfg.get(dotted, None)
        try:
            cfg.set(dotted, caster(raw))
        except (TypeError, ValueError):
            print(f"[FCW] Ignoring invalid {env_name}={raw!r}")

    return cfg


def parse_set_overrides(pairs):
    """Turn ['a.b=1', 'c=foo'] into a nested dict."""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            print(f"[FCW] Ignoring malformed --set {pair!r} (expected key=value)")
            continue
        key, _, raw = pair.partition("=")
        key = key.strip()
        raw = raw.strip()
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce_scalar(raw)
    return out


def _coerce_scalar(text):
    """Infer int/float/bool/str so YAML overrides do not stay strings."""
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    return text


# ── Frequently used accessors ─────────────────────────────────────────
# The pipeline reads these constantly; helpers keep call sites readable and
# guarantee the cast, instead of every caller repeating get(..., cast=int).

def target_fps(cfg):
    """Requested processing FPS. None means 'process every frame'."""
    value = cfg.get("processing.target_fps", 10, cast=float)
    if value is None or value <= 0:
        return None
    return float(value)


def depth_interval(cfg):
    return max(1, cfg.get("depth.interval", 5, cast=int))


def depth_height(cfg):
    return max(154, cfg.get("depth.processing_height", 224, cast=int))


def input_max_size(cfg):
    return (cfg.get("input.max_width", 1280, cast=int),
            cfg.get("input.max_height", 720, cast=int))


def primary_classes(cfg):
    values = cfg.get("classes.primary", ["motorbike"])
    if isinstance(values, str):
        values = [v.strip() for v in values.split(",")]
    return [v for v in values if v]


def target_classes(cfg):
    values = cfg.get("classes.targets", ["motorbike", "car", "bus", "truck", "bicycle", "person"])
    if isinstance(values, str):
        values = [v.strip() for v in values.split(",")]
    return [v for v in values if v]


def depth_interpolation_mode(cfg):
    mode = cfg.get("depth.interpolation", "linear")
    return "hold" if str(mode).strip().lower() == "hold" else "linear"