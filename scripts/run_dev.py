"""
Development Server Runner
Launches FastAPI backend with hot reloading and static web studio.
"""

import uvicorn
from app.core.config import settings


def main():
    print("=" * 60)
    print("  🚀 PADDLEOCR STUDIO & REST API SERVER")
    print(f"  📡 Running on: http://localhost:{settings.PORT}")
    print(f"  📖 API Docs:   http://localhost:{settings.PORT}/docs")
    print(f"  🎨 Web Studio: http://localhost:{settings.PORT}/")
    print("=" * 60)
    
    uvicorn.run(
        "app.api.main:app",
        host="127.0.0.1",
        port=settings.PORT,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
