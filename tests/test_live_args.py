"""Tests for tools/live.py CLI argument parsing and validation."""

import sys
import os
from pathlib import Path

# Add project root so tools.live can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
from tools.live import build_parser, validate_args, FrameMetrics


# ── Argument parsing ─────────────────────────────────────────────────

class TestBuildParser:
    """T01, T05: --help and argument validation."""

    def test_help_exits_cleanly(self):
        """T01: --help exits without loading model or camera."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_url_required(self):
        """Missing --url should error."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    def test_valid_args(self):
        """T02: valid args parse correctly."""
        parser = build_parser()
        args = parser.parse_args(["--url", "http://192.168.1.42:8080/video", "--width", "960"])
        assert args.url == "http://192.168.1.42:8080/video"
        assert args.width == 960

    def test_default_width_zero(self):
        parser = build_parser()
        args = parser.parse_args(["--url", "http://example.com/video"])
        assert args.width == 0

    def test_config_default(self):
        parser = build_parser()
        args = parser.parse_args(["--url", "test.mp4"])
        assert args.config == "config.yaml"

    def test_benchmark_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--url", "test.mp4", "--benchmark"])
        assert args.benchmark is True

    def test_no_display_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--url", "test.mp4", "--no-display"])
        assert args.no_display is True


class TestValidateArgs:
    """T05: invalid arguments are caught."""

    def _make_args(self, **kwargs):
        parser = build_parser()
        defaults = ["--url", "http://192.168.1.1:8080/video"]
        args = parser.parse_args(defaults)
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    def test_valid_args_return_none(self):
        args = self._make_args()
        assert validate_args(args) is None

    def test_negative_width(self):
        args = self._make_args(width=-100)
        err = validate_args(args)
        assert err is not None
        assert "Width" in err

    def test_zero_width_ok(self):
        args = self._make_args(width=0)
        assert validate_args(args) is None

    def test_negative_reconnect(self):
        args = self._make_args(reconnect=-1)
        err = validate_args(args)
        assert err is not None
        assert "Reconnect" in err

    def test_negative_reconnect_delay(self):
        args = self._make_args(reconnect_delay=-5.0)
        err = validate_args(args)
        assert err is not None
        assert "delay" in err


# ── FPS metrics ──────────────────────────────────────────────────────

class TestFrameMetrics:
    """T14: verify FPS formula and percentile computation."""

    def test_fps_formula_n_minus_1(self):
        """Sliding FPS must use (N-1)/(t_last - t_first), not N/(...)."""
        m = FrameMetrics(window_size=10)
        # Simulate 5 frames at 1-second intervals
        import time as _time
        base = _time.perf_counter()
        for i in range(5):
            m._times.append(base + i * 1.0)
            m.total_frames += 1
            m._pipeline_ms.append(100.0)

        # 4 intervals over 4 seconds = 1.0 FPS exactly
        fps = m.sliding_fps
        assert abs(fps - 1.0) < 0.01, f"Expected ~1.0 FPS, got {fps}"

    def test_fps_single_frame_zero(self):
        """With only 1 frame, FPS should be 0."""
        m = FrameMetrics()
        m._times.append(1000.0)
        assert m.sliding_fps == 0.0

    def test_fps_two_frames(self):
        """With 2 frames 0.5s apart, FPS should be ~2.0."""
        m = FrameMetrics()
        m._times.append(100.0)
        m._times.append(100.5)
        fps = m.sliding_fps
        assert abs(fps - 2.0) < 0.01

    def test_percentiles(self):
        """p50 and p95 should be computed correctly."""
        m = FrameMetrics(window_size=100)
        for i in range(100):
            m._pipeline_ms.append(float(i))
        p50 = m.p50_pipeline_ms
        p95 = m.p95_pipeline_ms
        assert 45 < p50 < 55, f"p50 = {p50}"
        assert 90 < p95 < 100, f"p95 = {p95}"

    def test_empty_percentiles(self):
        m = FrameMetrics()
        assert m.p50_pipeline_ms == 0.0
        assert m.p95_pipeline_ms == 0.0

    def test_avg_pipeline(self):
        m = FrameMetrics()
        m._pipeline_ms.extend([10.0, 20.0, 30.0])
        assert abs(m.avg_pipeline_ms - 20.0) < 0.01

    def test_tick_records_all(self):
        m = FrameMetrics()
        m.tick(100.0, yolo_ms=50.0, depth_ms=200.0)
        assert m.total_frames == 1
        assert len(m._pipeline_ms) == 1
        assert len(m._yolo_ms) == 1
        assert len(m._depth_ms) == 1

    def test_tick_skips_none_depth(self):
        """depth_ms=0 should not be recorded."""
        m = FrameMetrics()
        m.tick(100.0, yolo_ms=50.0, depth_ms=0.0)
        assert len(m._depth_ms) == 0

    def test_overall_fps(self):
        """Overall FPS = total_frames / elapsed."""
        m = FrameMetrics()
        m.total_frames = 100
        # Simulate some elapsed time
        import time
        time.sleep(0.05)
        fps = m.overall_fps
        assert fps > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
