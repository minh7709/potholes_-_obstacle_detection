import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router, init_stream_handler
from app.database.connection import init_db

# Use lifespan context manager for FastAPI startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables if not exist
    print("Initializing database...")
    init_db()
    
    # Startup: Initialize camera and streaming background thread
    print("Starting camera stream and detector...")
    init_stream_handler()
    
    yield
    
    # Shutdown: Stop camera capture and release resources
    print("Stopping streaming resources...")
    from app.api.routes import stream_handler
    if stream_handler:
        stream_handler.stop()

app = FastAPI(
    title="Pothole Detection & Streaming System",
    description="FastAPI + ONNX Runtime + OpenCV Pothole detection dashboard server.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for standard web client communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "app", "static")
potholes_dir = os.path.join(static_dir, "potholes")
os.makedirs(potholes_dir, exist_ok=True)

# Mount static folder under /static URL prefix
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include the API router
app.include_router(api_router, prefix="/api")

# Serve dashboard index.html at root url
@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    # Start uvicorn server locally on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
