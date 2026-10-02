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
    print("[RoadGuard] Initializing system & database...")
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    init_db()
    print("[RoadGuard] Ready. Upload a video to run FCW analysis.")
    yield

app = FastAPI(
    title="RoadGuard - Forward Collision Warning System",
    description="FastAPI + YOLO + ByteTrack + Depth Anything V2 collision warning platform.",
    version="3.0.0",
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

# Disable heuristic caching on static assets.
#
# Without an explicit Cache-Control, browsers apply heuristic freshness from
# Last-Modified and keep serving a stale app.js after a code change. That
# produced a real failure: the renamed KPI element `kpiTotalPotholes` was gone
# from the HTML while the browser ran the previous script, and the page threw
# "can't access property textContent, kpiTotalPotholes is null".
#
# `no-cache` does not mean "do not cache": the browser may store the response
# but must revalidate it, which is a cheap 304 against the ETag. Correctness
# wins here, and this is a local prototype app where file sizes are small.
@app.middleware("http")
async def disable_static_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Include the API routes
app.include_router(api_router, prefix="/api")

# Serve index.html with content-hashed asset URLs.
#
# `no-cache` alone was not enough. The browser had ALREADY stored app.js under
# headers that granted long heuristic freshness, so it never re-fetched and
# therefore never saw the new Cache-Control. Meanwhile `/` was re-fetched and
# returned the renamed HTML. Result: new HTML (no #kpi-total-potholes) paired
# with old JS (expecting it) -> "kpiTotalPotholes is null", every load.
#
# Appending `?v=<hash of the assets>` makes the URL itself change whenever the
# asset changes, so a content change is always a cache MISS by definition. Old
# cached HTML keeps pointing at the old JS, so the HTML/JS pair is always
# internally consistent instead of half-updated.
_ASSET_VERSION_CACHE = {"key": None, "html": None}


def _asset_version():
    """Short digest of the asset files, recomputed only when they change."""
    import hashlib

    digest = hashlib.sha1()
    for name in ("app.js", "styles.css"):
        path = os.path.join(STATIC_DIR, name)
        try:
            with open(path, "rb") as fh:
                digest.update(fh.read())
        except OSError:
            pass
    return digest.hexdigest()[:10]


@app.get("/")
def read_root():
    from fastapi.responses import HTMLResponse

    version = _asset_version()
    cached = _ASSET_VERSION_CACHE
    if cached["key"] != version:
        path = os.path.join(STATIC_DIR, "index.html")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                html = fh.read()
        except OSError:
            return FileResponse(path)
        html = html.replace("/static/app.js", f"/static/app.js?v={version}")
        html = html.replace("/static/styles.css", f"/static/styles.css?v={version}")
        cached["key"] = version
        cached["html"] = html
    return HTMLResponse(cached["html"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)