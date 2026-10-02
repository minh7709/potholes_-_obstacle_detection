import json
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.database.connection import Base


class VideoRecord(Base):
    """Một video đã được phân tích bởi hệ thống FCW."""

    __tablename__ = "fcw_videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    saved_filename = Column(String, nullable=False)
    video_url = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    total_frames = Column(Integer, default=0)
    fps = Column(Float, default=0.0)
    duration_sec = Column(Float, default=0.0)

    # Số frame có phát sinh cảnh báo (WARNING hoặc DANGER).
    warning_frames_count = Column(Integer, default=0)
    # Số frame ở mức DANGER.
    danger_frames_count = Column(Integer, default=0)
    # TTC nhỏ nhất ghi nhận trong toàn bộ video (None nếu không có cảnh báo).
    min_ttc_sec = Column(Float, nullable=True)
    # Khoảng cách gần nhất tới vật thể trong ROI (mét).
    min_distance_m = Column(Float, nullable=True)

    status = Column(String, default="completed")  # "processing", "completed", "failed"
    error_message = Column(String, nullable=True)

    frames = relationship(
        "WarningFrame", back_populates="video", cascade="all, delete-orphan",
        order_by="WarningFrame.frame_number",
    )

    def to_dict(self, include_frames=False):
        data = {
            "id": self.id,
            "filename": self.filename,
            "saved_filename": self.saved_filename,
            "video_url": self.video_url,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2) if self.fps else 0,
            "duration_sec": round(self.duration_sec, 2) if self.duration_sec else 0,
            "warning_frames_count": self.warning_frames_count,
            "danger_frames_count": self.danger_frames_count,
            "min_ttc_sec": round(self.min_ttc_sec, 2) if self.min_ttc_sec is not None else None,
            "min_distance_m": round(self.min_distance_m, 2) if self.min_distance_m is not None else None,
            "status": self.status,
            "error_message": self.error_message,
        }
        if include_frames:
            data["frames"] = [frame.to_dict() for frame in self.frames]
        return data


class WarningFrame(Base):
    """Một khung hình có phát sinh cảnh báo va chạm.

    Chỉ lưu frame có WARNING/DANGER để database không phình theo số frame.
    Mọi số đo đều là *ước lượng* từ monocular depth, không phải đo đạc.
    """

    __tablename__ = "fcw_warning_frames"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("fcw_videos.id", ondelete="CASCADE"),
                      nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp_sec = Column(Float, default=0.0)
    image_path = Column(String, nullable=False)

    # Mức cảnh báo của frame: "WARNING" hoặc "DANGER".
    severity = Column(String, default="WARNING")
    # Mục tiêu chính trong frame này.
    track_id = Column(Integer, nullable=True)
    class_name = Column(String, default="motorcycle")
    distance_m = Column(Float, nullable=True)
    closing_speed_mps = Column(Float, nullable=True)
    ttc_sec = Column(Float, nullable=True)
    confidence = Column(Float, default=0.0)
    # Vận tốc ước lượng của xe (m/s), để giải thích closing speed.
    ego_speed_mps = Column(Float, default=0.0)

    # JSON string of full per-object telemetry for this frame.
    detections_json = Column(Text, default="[]")
    detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoRecord", back_populates="frames")

    def to_dict(self):
        try:
            det_list = json.loads(self.detections_json) if self.detections_json else []
        except Exception:
            det_list = []

        return {
            "id": self.id,
            "video_id": self.video_id,
            "frame_number": self.frame_number,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "image_path": self.image_path,
            "severity": self.severity,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "closing_speed_mps": round(self.closing_speed_mps, 2) if self.closing_speed_mps is not None else None,
            "ttc_sec": round(self.ttc_sec, 2) if self.ttc_sec is not None else None,
            "confidence": round(self.confidence, 4),
            "ego_speed_mps": round(self.ego_speed_mps, 2),
            "detections": det_list,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
        }
