"""
Core Configuration Module
Manages environment settings, model parameterization, device discovery, and directory paths.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="allow")

    # Application Info
    APP_NAME: str = "Enterprise PaddleOCR Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    
    # PaddleOCR Engine Settings
    OCR_VERSION: str = "PP-OCRv4"  # PP-OCRv4 or PP-OCRv3
    USE_GPU: bool = False  # Auto-detected if CUDA available
    USE_ANGLE_CLS: bool = True  # Enable 0/180/90/270 degree text orientation classifier
    DEFAULT_LANG: str = "en"  # Default OCR language
    SUPPORTED_LANGS: List[str] = [
        "en", "ch", "hi", "fr", "german", "es", "japan", "korean", 
        "arabic", "latin", "cyrillic", "devanagari"
    ]
    
    # Detection & Recognition Hyperparameters
    DET_DB_THRESH: float = 0.3  # DBNet binarization threshold
    DET_DB_BOX_THRESH: float = 0.6  # DBNet bounding box score threshold
    DET_DB_UNCLIP_RATIO: float = 1.5  # DBNet Vatti unclip ratio
    REC_CONFIDENCE_THRESH: float = 0.5  # Recognition confidence cutoff
    
    # Preprocessing
    MAX_IMAGE_SIDE_LEN: int = 2048  # Maximum image side dimension
    MIN_IMAGE_SIDE_LEN: int = 32
    AUTO_DESKEW: bool = True
    ENHANCE_CONTRAST: bool = True
    
    # Storage & Cache Directories
    MODEL_DIR: Path = BASE_DIR / "models"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    STATIC_DIR: Path = BASE_DIR / "app" / "web"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"


# Instantiate global settings
settings = Settings()

# Ensure required directories exist
for path in [settings.MODEL_DIR, settings.UPLOAD_DIR, settings.OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
