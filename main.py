import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.database.connection import init_db
from app.config import STATIC_DIR, VIDEOS_DIR, FRAMES_DIR

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure static directories and database tables are initialized
    print("[RoadVision] Initializing system & database...")
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    init_db()
    print("[RoadVision] System initialized successfully. Ready to process videos.")
    yield

app = FastAPI(
    title="RoadVision - Pothole Detection AI System",
    description="FastAPI + YOLO ONNX Pothole Detection & Telemetry Platform.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin client requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include the API routes
app.include_router(api_router, prefix="/api")

# Serve main application index.html at root route
@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
