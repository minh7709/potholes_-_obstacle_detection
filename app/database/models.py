from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime, timezone
from app.database.connection import Base

class PotholeLog(Base):
    __tablename__ = "pothole_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    severity = Column(String, nullable=False)  # "Low", "Medium", "High"
    image_path = Column(String, nullable=True)   # Relative path to static folder for web viewing

    def to_dict(self):
        """Converts the object fields to a dictionary for JSON responses."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
            "bbox": [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2],
            "severity": self.severity,
            "image_path": self.image_path
        }
