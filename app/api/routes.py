from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta, timezone

from app.database.connection import get_db
from app.database.models import PotholeLog
from app.config import CONFIDENCE_THRESHOLD, NMS_THRESHOLD, VIDEO_SOURCE

router = APIRouter()

# Global StreamHandler instance reference
stream_handler = None

def init_stream_handler(detector=None):
    """Initializes and starts the stream handler."""
    global stream_handler
    from app.services.stream_handler import StreamHandler
    stream_handler = StreamHandler(detector=detector)
    stream_handler.start()
    return stream_handler

@router.get("/video_feed")
def video_feed():
    """Endpoint serving the MJPEG video stream."""
    global stream_handler
    if stream_handler is None:
        init_stream_handler()
        
    return StreamingResponse(
        stream_handler.get_latest_frame_bytes(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/detections")
def get_detections(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    severity: Optional[str] = None
):
    """Fetches the latest pothole detection logs from database."""
    query = db.query(PotholeLog)
    if severity:
        query = query.filter(PotholeLog.severity == severity)
        
    logs = query.order_by(PotholeLog.timestamp.desc()).limit(limit).all()
    return [log.to_dict() for log in logs]

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Computes real-time statistics for the dashboard dashboard."""
    # Total detection counts
    total = db.query(func.count(PotholeLog.id)).scalar() or 0
    
    # Severity breakdown
    severity_query = db.query(PotholeLog.severity, func.count(PotholeLog.id)).group_by(PotholeLog.severity).all()
    severity_breakdown = {"Low": 0, "Medium": 0, "High": 0}
    for sev, count in severity_query:
        if sev in severity_breakdown:
            severity_breakdown[sev] = count
            
    # Counts grouped by hour in local/UTC time for the past 24 hours
    # SQLite strftime('%H', timestamp) is simple and works on UTC dates
    since_time = datetime.now(timezone.utc) - timedelta(hours=24)
    hourly_query = db.query(
        func.strftime('%H', PotholeLog.timestamp).label('hour'),
        func.count(PotholeLog.id)
    ).filter(PotholeLog.timestamp >= since_time)\
     .group_by('hour')\
     .order_by('hour').all()
     
    # Fill in all 24 hours with default zeros
    current_hour = datetime.now().hour
    hourly_stats = {}
    for i in range(24):
        h = (current_hour - i) % 24
        hourly_stats[f"{h:02d}"] = 0
        
    for hour, count in hourly_query:
        # Match keys in hourly_stats
        hourly_stats[hour] = count
        
    # Sort hourly_stats chronologically
    sorted_hours = sorted(hourly_stats.keys(), key=lambda x: (int(x) - current_hour - 1) % 24)
    chronological_hourly = {h: hourly_stats[h] for h in sorted_hours}

    # Recent list (last 5 detections)
    recent_detections = db.query(PotholeLog).order_by(PotholeLog.timestamp.desc()).limit(5).all()
    recent_list = [log.to_dict() for log in recent_detections]

    return {
        "total": total,
        "severity": severity_breakdown,
        "hourly": chronological_hourly,
        "recent": recent_list
    }

@router.get("/config")
def get_config():
    """Gets current configuration values."""
    global stream_handler
    if stream_handler and stream_handler.detector:
        detector = stream_handler.detector
        return {
            "confidence_threshold": detector.conf_threshold,
            "nms_threshold": detector.nms_threshold,
            "video_source": str(stream_handler.source)
        }
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "video_source": str(VIDEO_SOURCE)
    }

@router.post("/config")
def update_config(
    confidence_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    video_source: Optional[str] = None
):
    """Updates configuration parameters on the fly."""
    global stream_handler
    if stream_handler is None:
        raise HTTPException(status_code=503, detail="Streaming service not running")
        
    if confidence_threshold is not None:
        stream_handler.detector.conf_threshold = confidence_threshold
        
    if nms_threshold is not None:
        stream_handler.detector.nms_threshold = nms_threshold
        
    if video_source is not None:
        from app.services.stream_handler import clean_video_source
        new_source = clean_video_source(video_source)
        if stream_handler.source != new_source:
            stream_handler.stop()
            stream_handler.source = new_source
            stream_handler.start()
            
    return {"status": "success", "message": "Configuration updated successfully"}
