import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.connection import get_db, reset_db
from app.database.models import VideoRecord, WarningFrame
from app.services.detector import detector
from app.services.video_upload_service import save_uploaded_video
from app.config import VIDEOS_DIR, FRAMES_DIR

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Video Upload & FCW Analysis
# ─────────────────────────────────────────────────────────────

@router.post("/upload-video")
async def upload_and_process_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    1. Uploads video file to server.
    2. Runs the FCW pipeline (YOLO + ByteTrack + Depth Anything V2) frame by frame.
    3. Estimates distance, closing speed and TTC per tracked vehicle.
    4. Saves frames that raised a WARNING/DANGER to static/frames/.
    5. Saves telemetry to SQLite and returns the full result.
    """
    saved_filepath, saved_filename, public_url = save_uploaded_video(file)

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
        # Tracking state must not leak between uploads, otherwise one video's
        # distance history would seed the next video's first frames.
        detector.reset()

        results = detector.process_video(
            video_path=saved_filepath,
            video_id=video_record.id
        )

        frame_objs = []
        for f in results["warning_frames"]:
            frame_objs.append(WarningFrame(
                video_id=video_record.id,
                frame_number=f["frame_number"],
                timestamp_sec=f["timestamp_sec"],
                image_path=f["image_path"],
                severity=f["severity"],
                track_id=f["track_id"],
                class_name=f["class_name"],
                distance_m=f["distance_m"],
                closing_speed_mps=f["closing_speed_mps"],
                ttc_sec=f["ttc_sec"],
                confidence=f["confidence"],
                ego_speed_mps=f["ego_speed_mps"],
                detections_json=f["detections_json"]
            ))

        if frame_objs:
            db.bulk_save_objects(frame_objs)

        video_record.total_frames = results["total_frames"]
        video_record.fps = results["fps"]
        video_record.duration_sec = results["duration_sec"]
        video_record.warning_frames_count = results["warning_frames_count"]
        video_record.danger_frames_count = results["danger_frames_count"]
        video_record.min_ttc_sec = results["min_ttc_sec"]
        video_record.min_distance_m = results["min_distance_m"]
        video_record.status = "completed"
        db.commit()
        db.refresh(video_record)

        # Surface throughput so the user knows a long video took real compute
        # time, instead of the request simply appearing to hang.
        perf = results.get("performance") or {}
        ratio = perf.get("speed_ratio")
        elapsed = perf.get("processing_time_sec")
        detail = ""
        if elapsed and ratio:
            speed = "nhanh hơn" if ratio >= 1.0 else "chậm hơn"
            detail = (f" Phân tích {results['total_frames']}/{results['source_frames']} frame "
                      f"trong {elapsed}s ({ratio}x {speed} thời gian thực).")

        return {
            "status": "success",
            "message": f"Video phân tích FCW thành công.{detail}",
            "data": video_record.to_dict(include_frames=True),
            "performance": perf,
        }

    except Exception as e:
        video_record.status = "failed"
        video_record.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý FCW cho video: {e}")


# ─────────────────────────────────────────────────────────────
# Video Retrieval & History
# ─────────────────────────────────────────────────────────────

@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    """Fetches list of all analysed videos and their summary statistics."""
    videos = db.query(VideoRecord).order_by(VideoRecord.uploaded_at.desc()).all()
    return [v.to_dict(include_frames=False) for v in videos]


@router.get("/videos/{video_id}")
def get_video_details(video_id: int, db: Session = Depends(get_db)):
    """Fetches detailed FCW analysis of a specific video."""
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

    video_path = os.path.join(VIDEOS_DIR, video.saved_filename)
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass

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
    """Clears all records in database and cleans up saved videos & frame images."""
    try:
        reset_db(clear_files=True)
        detector.reset()
        return {"status": "success", "message": "Database và dữ liệu lưu trữ đã được reset thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi reset database: {e}")


@router.get("/stats")
def get_overall_stats(db: Session = Depends(get_db)):
    """Computes overall statistics across all analysed videos."""
    total_videos = db.query(func.count(VideoRecord.id)).scalar() or 0
    total_warning_frames = db.query(func.count(WarningFrame.id)).scalar() or 0

    danger_count = (db.query(func.count(WarningFrame.id))
                    .filter(WarningFrame.severity == "DANGER").scalar() or 0)
    warning_count = (db.query(func.count(WarningFrame.id))
                     .filter(WarningFrame.severity == "WARNING").scalar() or 0)

    # Average TTC across all DANGER frames, rounded for display.
    danger_ttc_avg = (db.query(func.avg(WarningFrame.ttc_sec))
                      .filter(WarningFrame.severity == "DANGER").scalar())
    danger_ttc_avg = round(danger_ttc_avg, 2) if danger_ttc_avg is not None else None

    return {
        "total_videos": total_videos,
        "total_warning_frames": total_warning_frames,
        "severity": {
            "DANGER": danger_count,
            "WARNING": warning_count,
            "NORMAL": 0,
        },
        "danger_frames": danger_count,
        "warning_frames": warning_count,
        "avg_danger_ttc_sec": danger_ttc_avg,
    }
