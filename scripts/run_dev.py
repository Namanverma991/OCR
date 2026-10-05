import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
