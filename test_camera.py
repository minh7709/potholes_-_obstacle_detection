import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock


class TestCleanVideoSource:
    def test_int_source_passthrough(self):
        from app.services.stream_handler import clean_video_source
        assert clean_video_source(0) == 0
        assert clean_video_source(1) == 1

    def test_numeric_string_converts_to_int(self):
        from app.services.stream_handler import clean_video_source
        assert clean_video_source("0") == 0
        assert clean_video_source("1") == 1

    def test_http_typo_fix(self):
        from app.services.stream_handler import clean_video_source
        result = clean_video_source("http;//192.168.1.5:8080")
        assert result.startswith("http://")

    def test_https_typo_fix(self):
        from app.services.stream_handler import clean_video_source
        result = clean_video_source("https;//192.168.1.5:8080")
        assert result.startswith("https://")

    def test_url_auto_append_video(self):
        from app.services.stream_handler import clean_video_source
        result = clean_video_source("http://192.168.1.5:8080")
        assert result == "http://192.168.1.5:8080/video"

    def test_url_with_trailing_slash_auto_append_video(self):
        from app.services.stream_handler import clean_video_source
        result = clean_video_source("http://192.168.1.5:8080/")
        assert result == "http://192.168.1.5:8080/video"

    def test_url_with_existing_path_unchanged(self):
        from app.services.stream_handler import clean_video_source
        url = "http://192.168.1.5:8080/stream.mjpg"
        assert clean_video_source(url) == url

    def test_url_with_spaces_removed(self):
        from app.services.stream_handler import clean_video_source
        result = clean_video_source("http://192.168.1.5:8080 /video")
        assert " " not in result

    def test_non_numeric_string_passthrough(self):
        from app.services.stream_handler import clean_video_source
        url = "rtsp://192.168.1.5:554/stream"
        assert clean_video_source(url) == url


class TestStreamHandlerInit:
    def test_default_source_uses_video_source_config(self):
        with patch("app.services.stream_handler.VIDEO_SOURCE", "0"):
            from app.services.stream_handler import StreamHandler
            handler = StreamHandler()
            assert handler.source == 0
            assert handler.running is False
            assert handler.latest_raw_frame is None
            assert handler.latest_detections == []
            assert handler._stream_fps == 0.0

    def test_custom_source(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler(source="http://example.com/video")
        assert handler.source == "http://example.com/video"

    def test_detector_initialized_by_default(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        assert handler.detector is not None

    def test_internal_queue_created(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        assert handler._frame_queue.maxsize == 2

    def test_http_stream_property(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler(source="http://example.com/video")
        assert handler._is_http_stream() is True

    def test_non_http_stream_property(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler(source="0")
        assert handler._is_http_stream() is False


class TestStreamHandlerStartStop:
    def test_start_creates_threads(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler(source="0")
        handler.start()
        assert handler.running is True
        assert handler._capture_thread is not None
        assert handler._inference_thread is not None
        assert handler._capture_thread.name == "CaptureThread"
        assert handler._inference_thread.name == "InferenceThread"
        handler.stop()

    def test_double_start_is_idempotent(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler(source="0")
        handler.start()
        t1 = handler._capture_thread
        handler.start()
        assert handler._capture_thread is t1
        handler.stop()

    def test_stop_clears_state(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler(source="0")
        handler.start()
        handler.stop()
        assert handler.running is False
        assert handler.latest_raw_frame is None
        assert handler.latest_detections == []

    def test_stop_releases_capture(self):
        mock_cap = MagicMock()
        with patch("app.services.stream_handler.StreamHandler._create_capture", return_value=mock_cap):
            from app.services.stream_handler import StreamHandler
            handler = StreamHandler(source="0")
            handler.start()
            handler.stop()
            mock_cap.release.assert_called_once()

    def test_create_capture_webcam_sets_properties(self):
        from app.services.stream_handler import StreamHandler
        with patch("app.services.stream_handler.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_vc.return_value = mock_cap
            handler = StreamHandler(source="0")
            cap = handler._create_capture()
            mock_cap.set.assert_any_call(3, 640)
            mock_cap.set.assert_any_call(4, 480)
            mock_cap.set.assert_any_call(5, 15)

    def test_create_capture_http_uses_ffmpeg(self):
        from app.services.stream_handler import StreamHandler
        with patch("app.services.stream_handler.cv2.VideoCapture") as mock_vc:
            with patch("app.services.stream_handler.cv2.setLogLevel", create=True):
                handler = StreamHandler(source="http://example.com/video")
                cap = handler._create_capture()
                mock_vc.assert_called_with("http://example.com/video", mock_vc.call_args[0][1])


class TestAnalyzeDetections:
    def test_low_severity_small_box(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        frame_area = 640 * 480
        boxes = [(10, 10, 50, 50)]
        confidences = [0.5]
        dets = handler._analyze_detections(boxes, confidences, frame_area)
        assert dets[0]["severity"] == "Low"
        assert dets[0]["color"] == (0, 255, 0)

    def test_medium_severity_medium_box(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        frame_area = 640 * 480
        box_w = int((0.03 * frame_area) ** 0.5) + 1
        boxes = [(10, 10, 10 + box_w, 10 + box_w)]
        confidences = [0.6]
        dets = handler._analyze_detections(boxes, confidences, frame_area)
        assert dets[0]["severity"] == "Medium"
        assert dets[0]["color"] == (0, 165, 255)

    def test_high_severity_large_box(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        frame_area = 640 * 480
        box_w = int((0.05 * frame_area) ** 0.5) + 1
        boxes = [(10, 10, 10 + box_w, 10 + box_w)]
        confidences = [0.7]
        dets = handler._analyze_detections(boxes, confidences, frame_area)
        assert dets[0]["severity"] == "High"
        assert dets[0]["color"] == (0, 0, 255)

    def test_multiple_detections(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        frame_area = 640 * 480
        # Low: ~0.003 < 0.012, Medium: ~0.025 between 0.012-0.04, High: ~0.083 > 0.04
        boxes = [(10, 10, 40, 40), (50, 50, 150, 150), (100, 100, 350, 350)]
        confidences = [0.4, 0.6, 0.9]
        dets = handler._analyze_detections(boxes, confidences, frame_area)
        assert len(dets) == 3
        assert dets[0]["severity"] == "Low"
        assert dets[1]["severity"] == "Medium"
        assert dets[2]["severity"] == "High"


class TestDrawDetections:
    def test_draw_returns_copy_with_overlays(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"box": (50, 50, 150, 150), "confidence": 0.85, "severity": "High", "color": (0, 0, 255)}
        ]
        out = handler._draw_detections(frame, detections)
        assert out.shape == frame.shape
        assert out.dtype == frame.dtype
        assert out is not frame

    def test_draw_empty_detections_returns_plain_copy(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = handler._draw_detections(frame, [])
        assert out.shape == frame.shape


class TestGetLatestFrameBytes:
    def test_placeholder_when_no_frame(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        handler.running = True
        gen = handler.get_latest_frame_bytes()
        chunk = next(gen)
        assert b'--frame' in chunk
        assert b'Content-Type: image/jpeg' in chunk

    def test_yields_jpeg_when_frame_available(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler()
        handler.running = True
        handler.latest_raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gen = handler.get_latest_frame_bytes()
        chunk = next(gen)
        assert chunk.startswith(b'--frame')
        assert b'Content-Type: image/jpeg' in chunk


class TestVideoFeedEndpoint:
    def test_video_feed_returns_streaming_response(self):
        import app.api.routes
        from fastapi.responses import StreamingResponse
        mock_handler = MagicMock()
        mock_handler.get_latest_frame_bytes.return_value = iter([b'test'])
        app.api.routes.stream_handler = MagicMock()
        app.api.routes.stream_handler.get_latest_frame_bytes.return_value = iter([b'test'])
        with patch("app.api.routes.init_stream_handler") as mock_init:
            response = app.api.routes.video_feed()
            mock_init.assert_not_called()
            assert isinstance(response, StreamingResponse)
            assert response.media_type == "multipart/x-mixed-replace; boundary=frame"

    def test_video_feed_uses_existing_handler(self):
        from app.api.routes import video_feed, stream_handler
        from fastapi.responses import StreamingResponse
        mock_handler = MagicMock()
        mock_handler.get_latest_frame_bytes.return_value = iter([b'test'])
        with patch("app.api.routes.stream_handler", mock_handler):
            with patch("app.api.routes.init_stream_handler") as mock_init:
                response = video_feed()
                mock_init.assert_not_called()
                assert isinstance(response, StreamingResponse)


class TestStreamHandlerIntegration:
    def test_full_lifecycle(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler(source="0")
        handler.start()
        assert handler.running is True
        handler.stop()
        assert handler.running is False
        assert handler._capture_thread is not None
        assert handler._inference_thread is not None

    def test_capture_and_inference_threads_start_and_stop(self):
        from app.services.stream_handler import StreamHandler
        handler = StreamHandler(source="0")
        handler.start()
        assert handler._capture_thread.is_alive() is True
        assert handler._inference_thread.is_alive() is True
        handler.stop()
        handler._capture_thread.join(timeout=3)
        handler._inference_thread.join(timeout=3)
        assert handler._capture_thread.is_alive() is False
        assert handler._inference_thread.is_alive() is False


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
