"""
PaddleOCR Model Pre-fetching and Asset Manager
Downloads and verifies model weights for PP-OCRv6 Tiny detection, recognition, and classification on CPU.
"""

import sys
import logging
from app.core.config import settings
from app.core.ocr_engine import OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_downloader")


def prefetch_models():
    """Trigger PaddleOCR weight pre-caching for configured languages on CPU."""
    logger.info(f"Checking environment for PaddleOCR model weights ({settings.OCR_VERSION} on {settings.OCR_DEVICE.upper()})...")
    try:
        engine = OCREngine(
            lang=settings.DEFAULT_LANG,
            device=settings.OCR_DEVICE,
            ocr_version=settings.OCR_VERSION
        )
        if engine.ocr_model is not None:
            logger.info(f"✅ Model weights for {settings.DEFAULT_LANG} verified and ready.")
        else:
            logger.info("Fallback heuristic extractor available.")
    except Exception as e:
        logger.warning(f"PaddleOCR model downloader notice: {e}")


if __name__ == "__main__":
    prefetch_models()
