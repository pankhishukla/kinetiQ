"""
web/backend/main.py
====================
FastAPI application entry point.

HOW TO RUN:
    From the project root directory:
        uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000

    Then open your browser at:
        http://localhost:8000

WHY FastAPI?
    - Native async/await: handles WebSocket frames without blocking
    - Automatic JSON serialisation for endpoint responses
    - Built-in static file serving (serves frontend/index.html)
    - Lightweight — similar overhead to Flask but properly async

WHY serve the frontend from the same server?
    Cross-Origin Resource Sharing (CORS) issues arise when the frontend
    is on a different origin (port) than the WebSocket.  Serving both
    from port 8000 avoids this entirely.  No CORS headers needed.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src/` is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.backend.routes.ws_routes import router as ws_router

# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Kinetiq",
    description = "Real-time pose analysis via YOLOv8 + WebSocket",
    version     = "1.0.0",
)

# Register the WebSocket route
app.include_router(ws_router)

# ---------------------------------------------------------------------------
# STATIC FILES  — serve frontend/
# ---------------------------------------------------------------------------
# Mount the frontend directory so the browser can load CSS and JS files.
# FastAPI will serve any file under /static/... directly.

FRONTEND_DIR = ROOT / "web" / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# ROOT ROUTE  — serve index.html
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """
    Serve the main HTML page when the user navigates to http://localhost:8000
    WHY not just use StaticFiles for this?
        StaticFiles serves files at /static/index.html — not at /.
        This explicit route maps / → index.html cleanly.
    """
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ---------------------------------------------------------------------------
# HEALTH CHECK  — useful for verifying the server is alive
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
