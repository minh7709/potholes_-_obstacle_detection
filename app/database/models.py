import json
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database.connection import Base

class VideoRecord(Base):
    __tablename__ = "video_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    saved_filename = Column(String, nullable=False)
    video_url = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    total_frames = Column(Integer, default=0)
    fps = Column(Float, default=0.0)
    duration_sec = Column(Float, default=0.0)
    potholes_count = Column(Integer, default=0)
    detected_frames_count = Column(Integer, default=0)
    status = Column(String, default="completed")  # "processing", "completed", "failed"
    error_message = Column(String, nullable=True)

    # Relationship to detected frames
    frames = relationship("DetectedFrame", back_populates="video", cascade="all, delete-orphan", order_by="DetectedFrame.frame_number")

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
            "potholes_count": self.potholes_count,
            "detected_frames_count": self.detected_frames_count,
            "status": self.status,
            "error_message": self.error_message
        }
        if include_frames:
            data["frames"] = [frame.to_dict() for frame in self.frames]
        return data


class DetectedFrame(Base):
    __tablename__ = "detected_frames"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("video_records.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp_sec = Column(Float, default=0.0)
    image_path = Column(String, nullable=False)
    pothole_count = Column(Integer, default=1)
    max_confidence = Column(Float, nullable=False)
    severity = Column(String, default="Medium")  # "Low", "Medium", "High"
    confidences_json = Column(Text, default="[]")  # JSON string of list of float confidences
    detections_json = Column(Text, default="[]")   # JSON string of list of box & conf dicts
    detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    video = relationship("VideoRecord", back_populates="frames")

    def to_dict(self):
        try:
            conf_list = json.loads(self.confidences_json) if self.confidences_json else []
        except Exception:
            conf_list = []
            
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
            "pothole_count": self.pothole_count,
            "max_confidence": round(self.max_confidence, 4),
            "severity": self.severity,
            "confidences": conf_list,
            "detections": det_list,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None
        }
