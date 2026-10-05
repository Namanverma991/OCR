"""
REST API Endpoints for Document Ingestion and OCR Processing
"""

import time
import io
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from app.core.config import settings
from app.core.preprocessor import ImagePreprocessor
from app.core.ocr_engine import OCREngine, OCRResultItem
from app.core.document_processor import DocumentProcessor
from app.core.postprocessor import PostProcessor
from app.api.schemas import (
    OCRImageResponse, OCRDocumentResponse, TableOCRResponse, StructuredOCRResponse,
    HealthResponse, OCRResultModel, TableDataModel, PageOCRResultModel
)

router = APIRouter(prefix="/api/v1", tags=["Document & OCR Services"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check and diagnostic inspection."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        paddle_installed=OCREngine.check_paddle_installed(),
        default_engine=settings.OCR_VERSION,
        device=settings.OCR_DEVICE,
        gpu_available=settings.USE_GPU
    )


@router.post("/ocr/upload", response_model=OCRDocumentResponse)
@router.post("/ocr/document", response_model=OCRDocumentResponse)
async def process_document_upload(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    preprocess: bool = Form(True),
    use_angle_cls: bool = Form(True)
):
    """
    Unified production document ingestion endpoint.
    Accepts PDF, DOCX, and images (PNG, JPG, WEBP, TIFF, BMP).
    Performs digital native text extraction when available,
    and invokes PaddleOCR PP-OCRv6 Tiny CPU inference for scanned pages and images.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    filename = file.filename or "uploaded_document"
    processor = DocumentProcessor(
        lang=lang,
        device=settings.OCR_DEVICE,
        use_angle_cls=use_angle_cls,
        ocr_version=settings.OCR_VERSION
    )

    try:
        result = processor.process(file_bytes=content, filename=filename, preprocess=preprocess)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal document processing error: {str(e)}")

    pages_models = [
        PageOCRResultModel(
            page_number=p["page_number"],
            extraction_method=p["extraction_method"],
            text=p["text"],
            lines_count=p["lines_count"],
            confidence=p["confidence"],
            results=p.get("results", [])
        )
        for p in result.get("pages", [])
    ]

    table_models = [
        TableDataModel(
            rows=t.get("rows", 0),
            cols=t.get("cols", 0),
            headers=t.get("headers", []),
            matrix=t.get("matrix", []),
            csv=t.get("csv", ""),
            markdown=t.get("markdown", "")
        )
        for t in result.get("tables", [])
    ]

    results_models = None
    if "results" in result and result["results"]:
        results_models = [
            OCRResultModel(
                text=it["text"],
                confidence=it["confidence"],
                box=it["box"],
                center=it["center"],
                width=it["width"],
                height=it["height"],
                angle=it.get("angle", 0.0)
            )
            for it in result["results"]
        ]

    return OCRDocumentResponse(
        success=True,
        filename=result["filename"],
        file_type=result["file_type"],
        total_pages=result["total_pages"],
        processing_time_ms=result["processing_time_ms"],
        full_text=result["full_text"],
        pages=pages_models,
        results=results_models,
        entities=result.get("entities", {}),
        tables=table_models,
        meta=result.get("meta", {})
    )


@router.post("/ocr/image", response_model=OCRImageResponse)
async def process_image_ocr(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    preprocess: bool = Form(True),
    use_angle_cls: bool = Form(True)
):
    """
    Perform deep-learning OCR extraction on an uploaded image file using PP-OCRv6 Tiny.
    Returns extracted text lines, polygon coordinates, confidence, and full reading-order text.
    """
    try:
        content = await file.read()
        img = ImagePreprocessor.bytes_to_cv2(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    engine = OCREngine(
        lang=lang,
        device=settings.OCR_DEVICE,
        use_angle_cls=use_angle_cls,
        ocr_version=settings.OCR_VERSION
    )
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


@router.post("/ocr/pdf", response_model=OCRDocumentResponse)
async def process_pdf_ocr(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    preprocess: bool = Form(True)
):
    """
    Process multi-page PDF documents page-by-page.
    """
    return await process_document_upload(file=file, lang=lang, preprocess=preprocess)


@router.post("/ocr/table", response_model=TableOCRResponse)
async def process_table_ocr(
    file: UploadFile = File(...),
    lang: str = Form("en")
):
    """
    Detect and extract tabular structures from documents/images.
    Returns 2D matrix, CSV data, and Markdown table representation.
    """
    try:
        content = await file.read()
        img = ImagePreprocessor.bytes_to_cv2(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    engine = OCREngine(lang=lang, device=settings.OCR_DEVICE, ocr_version=settings.OCR_VERSION)
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

    engine = OCREngine(lang=lang, device=settings.OCR_DEVICE, ocr_version=settings.OCR_VERSION)
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
