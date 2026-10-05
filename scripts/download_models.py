"""
PaddleOCR Model Pre-fetching and Asset Manager
Downloads and verifies model weights for PP-OCRv4 / PP-OCRv3 detection, recognition, and classification.
"""

import sys
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_downloader")


def prefetch_models():
    """Trigger PaddleOCR weight pre-caching for specified languages."""
    logger.info("Checking environment for PaddleOCR model weights...")
    try:
        from paddleocr import PaddleOCR
        for lang in ["en"]:
            logger.info(f"Pre-caching models for language: '{lang}'...")
            PaddleOCR(use_angle_cls=True, lang=lang, ocr_version=settings.OCR_VERSION)
        logger.info("✅ Model weights verified and ready.")
    except Exception as e:
        logger.warning(f"PaddleOCR automatic downloader notice: {e}")


if __name__ == "__main__":
    prefetch_models()
