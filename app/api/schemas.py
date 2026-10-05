"""
Pydantic Schemas for Request and Response Payloads
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class BoxCoordinate(BaseModel):
    points: List[List[float]] = Field(..., description="4 corner vertices [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")


class OCRResultModel(BaseModel):
    text: str
    confidence: float
    box: List[List[float]]
    center: List[float]
    width: float
    height: float
    angle: float = 0.0


class OCRImageResponse(BaseModel):
    success: bool = True
    filename: str
    processing_time_ms: float
    total_lines: int
    full_text: str
    results: List[OCRResultModel]
    meta: Dict[str, Any]


class TableDataModel(BaseModel):
    rows: int
    cols: int
    headers: List[str]
    matrix: List[List[str]]
    csv: str
    markdown: str


class TableOCRResponse(BaseModel):
    success: bool = True
    filename: str
    processing_time_ms: float
    table_count: int
    tables: List[TableDataModel]


class StructuredOCRResponse(BaseModel):
    success: bool = True
    filename: str
    processing_time_ms: float
    entities: Dict[str, Any]
    full_text: str
    total_detected_blocks: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    paddle_installed: bool
    default_engine: str
    gpu_available: bool
