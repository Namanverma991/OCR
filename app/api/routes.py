"""
REST API Endpoints for OCR Processing
"""

import time
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium

from app.core.config import settings
from app.core.preprocessor import ImagePreprocessor
from app.core.ocr_engine import OCREngine, OCRResultItem
from app.core.postprocessor import PostProcessor
from app.core.layout_engine import LayoutEngine
from app.api.schemas import (
    OCRImageResponse, TableOCRResponse, StructuredOCRResponse, HealthResponse,
    OCRResultModel, TableDataModel
)

router = APIRouter(prefix="/api/v1", tags=["OCR Services"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and diagnostic inspection."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        paddle_installed=OCREngine.check_paddle_installed(),
        default_engine=settings.OCR_VERSION,
        gpu_available=settings.USE_GPU
    )


@router.post("/ocr/image", response_model=OCRImageResponse)
async def process_image_ocr(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    preprocess: bool = Form(True),
    use_angle_cls: bool = Form(True)
):
    """
    Perform deep-learning OCR extraction on an uploaded image file.
    Returns extracted text lines, polygon coordinates, confidence, and full reading-order text.
    """
    try:
        content = await file.read()
        img = ImagePreprocessor.bytes_to_cv2(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    engine = OCREngine(lang=lang, use_gpu=settings.USE_GPU, use_angle_cls=use_angle_cls)
    items, meta = engine.process_image(img, preprocess=preprocess)

    # Sort in reading order
    sorted_items = PostProcessor.sort_reading_order(items)
    full_text = PostProcessor.assemble_full_text(items)

    result_models = [
        OCRResultModel(
            text=it.text,
            confidence=it.confidence,
            box=it.box,
            center=it.center,
            width=it.width,
            height=it.height,
            angle=it.angle
        )
        for it in sorted_items
    ]

    return OCRImageResponse(
        success=True,
        filename=file.filename or "uploaded_image.png",
        processing_time_ms=meta.get("processing_time_ms", 0.0),
        total_lines=len(result_models),
        full_text=full_text,
        results=result_models,
        meta=meta
    )


@router.post("/ocr/table", response_model=TableOCRResponse)
async def process_table_ocr(
    file: UploadFile = File(...),
    lang: str = Form("en")
):
    """
    Detect and extract tabular structures from images.
    Returns 2D matrix, CSV data, and Markdown table representation.
    """
    try:
        content = await file.read()
        img = ImagePreprocessor.bytes_to_cv2(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    engine = OCREngine(lang=lang, use_gpu=settings.USE_GPU)
    items, meta = engine.process_image(img, preprocess=True)
    table_data = PostProcessor.reconstruct_table(items)

    table_model = TableDataModel(
        rows=table_data["rows"],
        cols=table_data["cols"],
        headers=table_data["headers"],
        matrix=table_data["matrix"],
        csv=table_data["csv"],
        markdown=table_data["markdown"]
    )

    return TableOCRResponse(
        success=True,
        filename=file.filename or "table.png",
        processing_time_ms=meta.get("processing_time_ms", 0.0),
        table_count=1 if table_data["rows"] > 0 else 0,
        tables=[table_model]
    )


@router.post("/ocr/structured", response_model=StructuredOCRResponse)
async def process_structured_ocr(
    file: UploadFile = File(...),
    lang: str = Form("en")
):
    """
    Extract key business entities (Invoice #, Dates, Currency, Total Amount, Tax ID, Email, Phone).
    """
    try:
        content = await file.read()
        img = ImagePreprocessor.bytes_to_cv2(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    engine = OCREngine(lang=lang, use_gpu=settings.USE_GPU)
    items, meta = engine.process_image(img, preprocess=True)
    full_text = PostProcessor.assemble_full_text(items)
    entities = PostProcessor.extract_structured_entities(full_text)

    return StructuredOCRResponse(
        success=True,
        filename=file.filename or "document.png",
        processing_time_ms=meta.get("processing_time_ms", 0.0),
        entities=entities,
        full_text=full_text,
        total_detected_blocks=len(items)
    )
