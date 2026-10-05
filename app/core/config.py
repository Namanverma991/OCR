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
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    
    # PaddleOCR Engine Settings
    OCR_ENABLED: bool = True
    OCR_VERSION: str = "PP-OCRv4"  # Production lightweight mobile/tiny model for CPU
    OCR_DEVICE: str = "cpu"        # Production default: CPU inference (integrated GPU / laptop friendly)
    USE_GPU: bool = False          # Explicitly False for CPU environments
    USE_ANGLE_CLS: bool = False    # Disable heavy orientation models for fast CPU inference
    DEFAULT_LANG: str = "en"       # Default OCR language
    SUPPORTED_LANGS: List[str] = [
        "en", "ch", "hi", "fr", "german", "es", "japan", "korean", 
        "arabic", "latin", "cyrillic", "devanagari"
    ]
    
    # Detection & Recognition Hyperparameters
    DET_DB_THRESH: float = 0.2  # DBNet binarization threshold
    DET_DB_BOX_THRESH: float = 0.5  # DBNet bounding box score threshold
    DET_DB_UNCLIP_RATIO: float = 1.5  # DBNet Vatti unclip ratio
    REC_CONFIDENCE_THRESH: float = 0.3  # Recognition confidence cutoff
    
    # Preprocessing
    MAX_IMAGE_SIDE_LEN: int = 2048  # Maximum image side dimension
    MIN_IMAGE_SIDE_LEN: int = 32
    AUTO_DESKEW: bool = False
    ENHANCE_CONTRAST: bool = False
    
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
