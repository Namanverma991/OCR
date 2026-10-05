import os
import sys
from pathlib import Path

# Configure CPU execution flags before importing paddle or dependent engines
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.api.routes import router as ocr_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Industrial OCR Engine based on PaddleOCR / PP-OCR algorithms."
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Endpoints
app.include_router(ocr_router)

@app.get("/health", include_in_schema=False)
async def root_health_check():
    return {"status": "healthy", "service": settings.APP_NAME, "version": settings.APP_VERSION}

# Mount Web Dashboard
web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        """Serve interactive OCR Web UI."""
        return FileResponse(str(web_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
