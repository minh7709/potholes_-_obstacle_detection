import os
import uuid
import shutil
from fastapi import UploadFile, HTTPException
from app.config import VIDEOS_DIR

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_MAX_FILE_SIZE_MB = 500

def save_uploaded_video(file: UploadFile) -> tuple[str, str, str]:
    """
    Saves an uploaded video file to the static videos directory.
    
    Returns:
        (saved_filepath, saved_filename, public_url)
    """
    original_filename = file.filename or "video.mp4"
    ext = os.path.splitext(original_filename)[1].lower()
    
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không hỗ trợ: '{ext}'. Các định dạng hợp lệ: {', '.join(_ALLOWED_EXTENSIONS)}"
        )
        
    unique_id = uuid.uuid4().hex[:10]
    safe_base = "".join(c for c in os.path.splitext(original_filename)[0] if c.isalnum() or c in ('_', '-'))[:25]
    saved_filename = f"{safe_base}_{unique_id}{ext}"
    saved_filepath = os.path.join(VIDEOS_DIR, saved_filename)
    
    try:
        with open(saved_filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu video lên server: {e}")
        
    size_mb = os.path.getsize(saved_filepath) / (1024 * 1024)
    if size_mb > _MAX_FILE_SIZE_MB:
        try:
            os.remove(saved_filepath)
        except Exception:
            pass
        raise HTTPException(
            status_code=413,
            detail=f"Dung lượng video quá lớn ({size_mb:.1f} MB). Giới hạn là {_MAX_FILE_SIZE_MB} MB."
        )
        
    public_url = f"/static/videos/{saved_filename}"
    return saved_filepath, saved_filename, public_url
