import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional

from app.database.connection import get_db, reset_db
from app.database.models import VideoRecord, DetectedFrame
from app.services.detector import detector
from app.services.video_upload_service import save_uploaded_video
from app.config import VIDEOS_DIR, FRAMES_DIR

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Video Upload & AI Processing
# ─────────────────────────────────────────────────────────────

@router.post("/upload-video")
async def upload_and_process_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    1. Uploads video file to server.
    2. Runs YOLO/ONNX AI pothole detection frame-by-frame.
    3. Saves frames with detected potholes to static/frames/.
    4. Saves video metadata and detection details to SQLite database.
    5. Returns full analysis results immediately to client.
    """
    # 1. Save uploaded video to app/static/videos/
    saved_filepath, saved_filename, public_url = save_uploaded_video(file)

    # 2. Create initial video record in DB
    video_record = VideoRecord(
        filename=file.filename or "video.mp4",
        saved_filename=saved_filename,
        video_url=public_url,
        status="processing"
    )
    db.add(video_record)
    db.commit()
    db.refresh(video_record)

    try:
        # 3. Process video through AI Detector
        results = detector.process_video(
            video_path=saved_filepath,
            video_id=video_record.id
        )

        # 4. Save detected frames to DB
        detected_frames_objs = []
        for f_data in results["detected_frames"]:
            df = DetectedFrame(
                video_id=video_record.id,
                frame_number=f_data["frame_number"],
                timestamp_sec=f_data["timestamp_sec"],
                image_path=f_data["image_path"],
                pothole_count=f_data["pothole_count"],
                max_confidence=f_data["max_confidence"],
                severity=f_data["severity"],
                confidences_json=f_data["confidences_json"],
                detections_json=f_data["detections_json"]
            )
            detected_frames_objs.append(df)

        if detected_frames_objs:
            db.bulk_save_objects(detected_frames_objs)

        # 5. Update video record with final telemetry
        video_record.total_frames = results["total_frames"]
        video_record.fps = results["fps"]
        video_record.duration_sec = results["duration_sec"]
        video_record.potholes_count = results["total_potholes_count"]
        video_record.detected_frames_count = len(detected_frames_objs)
        video_record.status = "completed"
        db.commit()
        db.refresh(video_record)

        return {
            "status": "success",
            "message": "Video processed successfully",
            "data": video_record.to_dict(include_frames=True)
        }

    except Exception as e:
        video_record.status = "failed"
        video_record.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý AI cho video: {e}")


# ─────────────────────────────────────────────────────────────
# Video Retrieval & History
# ─────────────────────────────────────────────────────────────

@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    """Fetches list of all uploaded videos and their summary statistics."""
    videos = db.query(VideoRecord).order_by(VideoRecord.uploaded_at.desc()).all()
    return [v.to_dict(include_frames=False) for v in videos]


@router.get("/videos/{video_id}")
def get_video_details(video_id: int, db: Session = Depends(get_db)):
    """Fetches detailed analysis of a specific video including all detected pothole frames."""
    video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Không tìm thấy video với ID này.")
    return video.to_dict(include_frames=True)


@router.delete("/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    """Deletes a video record and removes associated video and frame image files."""
    video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Không tìm thấy video.")

    # Remove physical video file
    video_path = os.path.join(VIDEOS_DIR, video.saved_filename)
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass

    # Remove physical frame images
    for frame in video.frames:
        frame_filename = os.path.basename(frame.image_path)
        frame_path = os.path.join(FRAMES_DIR, frame_filename)
        if os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except Exception:
                pass

    db.delete(video)
    db.commit()
    return {"status": "success", "message": f"Đã xóa video #{video_id}"}


# ─────────────────────────────────────────────────────────────
# Database Reset & Overall Statistics
# ─────────────────────────────────────────────────────────────

@router.post("/reset-db")
def reset_database():
    """
    Clears all records in database and cleans up saved videos & frame images.
    """
    try:
        reset_db(clear_files=True)
        return {"status": "success", "message": "Database và dữ liệu lưu trữ đã được reset thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi reset database: {e}")


@router.get("/stats")
def get_overall_stats(db: Session = Depends(get_db)):
    """Computes overall statistics across all processed videos."""
    total_videos = db.query(func.count(VideoRecord.id)).scalar() or 0
    total_potholes = db.query(func.sum(VideoRecord.potholes_count)).scalar() or 0
    total_detected_frames = db.query(func.count(DetectedFrame.id)).scalar() or 0

    severity_counts = {
        "High": db.query(func.count(DetectedFrame.id)).filter(DetectedFrame.severity == "High").scalar() or 0,
        "Medium": db.query(func.count(DetectedFrame.id)).filter(DetectedFrame.severity == "Medium").scalar() or 0,
        "Low": db.query(func.count(DetectedFrame.id)).filter(DetectedFrame.severity == "Low").scalar() or 0,
    }

    return {
        "total_videos": total_videos,
        "total_potholes": int(total_potholes),
        "total_detected_frames": total_detected_frames,
        "severity": severity_counts
    }
