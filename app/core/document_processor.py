"""
Unified Document Ingestion and Processing Engine
Handles PDF, DOCX, and multi-format image files.
Employs smart routing: preserves native text extraction when available,
and invokes PaddleOCR PP-OCRv6 Tiny on CPU for scanned documents and images.
"""

import io
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.core.ocr_engine import OCREngine, OCRResultItem
from app.core.preprocessor import ImagePreprocessor
from app.core.postprocessor import PostProcessor

logger = logging.getLogger("document.processor")
logging.basicConfig(level=logging.INFO)


class DocumentProcessor:
    """Production document ingestion pipeline with hybrid native/OCR extraction."""

    def __init__(
        self,
        lang: Optional[str] = None,
        device: Optional[str] = None,
        use_angle_cls: Optional[bool] = None,
        ocr_version: Optional[str] = None
    ):
        self.lang = lang or settings.DEFAULT_LANG
        self.device = device or settings.OCR_DEVICE
        self.ocr_version = ocr_version or settings.OCR_VERSION
        self.engine = OCREngine(
            lang=self.lang,
            device=self.device,
            use_angle_cls=use_angle_cls,
            ocr_version=self.ocr_version
        )

    def process(
        self,
        file_bytes: bytes,
        filename: str,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Main entrypoint for processing any uploaded document (PDF, DOCX, Image).
        """
        start_time = time.time()
        ext = Path(filename).suffix.lower()
        
        logger.info(f"Processing document: '{filename}' (Extension: {ext}, Size: {len(file_bytes)} bytes)")

        if ext == ".pdf":
            result = self._process_pdf(file_bytes, filename, preprocess=preprocess)
        elif ext in [".docx", ".doc"]:
            result = self._process_docx(file_bytes, filename, preprocess=preprocess)
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"]:
            result = self._process_image(file_bytes, filename, preprocess=preprocess)
        else:
            # Try interpreting as image fallback
            try:
                result = self._process_image(file_bytes, filename, preprocess=preprocess)
            except Exception as e:
                raise ValueError(f"Unsupported or unparseable file format '{ext}': {str(e)}")

        elapsed_ms = round((time.time() - start_time) * 1000.0, 2)
        result["processing_time_ms"] = elapsed_ms
        return result

    def _process_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Process multi-page PDF documents.
        Checks each page for embedded native text. If present, extracts directly.
        If the page is scanned / image-based, renders and applies PP-OCRv6 Tiny.
        """
        import pypdfium2 as pdfium
        import pdfplumber

        pages_data: List[Dict[str, Any]] = []
        all_ocr_items: List[OCRResultItem] = []
        all_text_segments: List[str] = []

        # Open with pypdfium2 for rendering and pdfplumber for native text
        try:
            pdf_doc = pdfium.PdfDocument(file_bytes)
            num_pages = len(pdf_doc)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid PDF file: {str(e)}")

        # Open pdfplumber in parallel for native text checks
        plumber_doc = None
        try:
            plumber_doc = pdfplumber.open(io.BytesIO(file_bytes))
        except Exception:
            pass

        for page_idx in range(num_pages):
            page_num = page_idx + 1
            native_text = ""
            
            # 1. Attempt native text extraction
            if plumber_doc and page_idx < len(plumber_doc.pages):
                try:
                    p_page = plumber_doc.pages[page_idx]
                    native_text = (p_page.extract_text() or "").strip()
                except Exception as ex:
                    logger.debug(f"Native PDF text extract error on page {page_num}: {ex}")

            # Heuristic: If native text contains at least 30 characters, prefer native extraction
            if len(native_text) >= 30:
                logger.info(f"Page {page_num}/{num_pages}: Native digital text found ({len(native_text)} chars). Skipping OCR.")
                pages_data.append({
                    "page_number": page_num,
                    "extraction_method": "native_text",
                    "text": native_text,
                    "lines_count": len(native_text.splitlines()),
                    "confidence": 1.0,
                    "results": []
                })
                all_text_segments.append(native_text)
            else:
                # 2. Fallback to PaddleOCR PP-OCRv6 Tiny CPU inference
                logger.info(f"Page {page_num}/{num_pages}: Scanned or image-based page. Running PP-OCRv6 Tiny OCR.")
                try:
                    page = pdf_doc.get_page(page_idx)
                    # Render page at 150 DPI for fast low-memory CPU processing
                    bitmap = page.render(scale=2.0)
                    pil_image = bitmap.to_pil()
                    img_np = np.array(pil_image)
                    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                    elif len(img_np.shape) == 3:
                        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    else:
                        img_cv = img_np

                    items, _ = self.engine.process_image(img_cv, preprocess=preprocess)
                    sorted_items = PostProcessor.sort_reading_order(items)
                    page_text = PostProcessor.assemble_full_text(items)
                    avg_conf = (
                        round(sum(it.confidence for it in sorted_items) / len(sorted_items), 4)
                        if sorted_items else 0.0
                    )

                    all_ocr_items.extend(items)
                    all_text_segments.append(page_text)

                    pages_data.append({
                        "page_number": page_num,
                        "extraction_method": "ocr_pp_ocrv6",
                        "text": page_text,
                        "lines_count": len(sorted_items),
                        "confidence": avg_conf,
                        "results": [it.to_dict() for it in sorted_items]
                    })
                except Exception as e:
                    logger.error(f"Error during OCR on page {page_num}: {e}")
                    pages_data.append({
                        "page_number": page_num,
                        "extraction_method": "error",
                        "text": native_text,
                        "lines_count": 0,
                        "confidence": 0.0,
                        "results": []
                    })
                    if native_text:
                        all_text_segments.append(native_text)

        if plumber_doc:
            try:
                plumber_doc.close()
            except Exception:
                pass

        full_text = "\n\n--- Page Break ---\n\n".join(seg for seg in all_text_segments if seg)
        entities = PostProcessor.extract_structured_entities(full_text)
        table_data = PostProcessor.reconstruct_table(all_ocr_items) if all_ocr_items else {"rows": 0, "cols": 0, "matrix": [], "csv": "", "markdown": ""}

        return {
            "success": True,
            "filename": filename,
            "file_type": "pdf",
            "total_pages": num_pages,
            "full_text": full_text,
            "pages": pages_data,
            "entities": entities,
            "tables": [table_data] if table_data["rows"] > 0 else [],
            "meta": {
                "engine": self.ocr_version,
                "device": self.device.upper(),
                "lang": self.lang
            }
        }

    def _process_docx(
        self,
        file_bytes: bytes,
        filename: str,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Process DOCX files. Extracts native paragraphs and tables.
        If the document is purely image-based, OCRs embedded pictures.
        """
        import docx

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Invalid DOCX document: {str(e)}")

        paragraphs_text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        
        # Extract tables natively
        tables_extracted = []
        for table in doc.tables:
            matrix = []
            for row in table.rows:
                matrix.append([cell.text.strip() for cell in row.cells])
            if matrix:
                csv_lines = [",".join(['"' + c.replace('"', '""') + '"' for c in r]) for r in matrix]
                md_lines = ["| " + " | ".join(matrix[0]) + " |", "| " + " | ".join(["---"] * len(matrix[0])) + " |"]
                for r in matrix[1:]:
                    md_lines.append("| " + " | ".join(r) + " |")
                tables_extracted.append({
                    "rows": len(matrix),
                    "cols": len(matrix[0]) if matrix else 0,
                    "headers": matrix[0] if matrix else [],
                    "matrix": matrix,
                    "csv": "\n".join(csv_lines),
                    "markdown": "\n".join(md_lines)
                })

        native_text = "\n".join(paragraphs_text)
        
        # Check for embedded images if text is minimal
        ocr_text_segments = []
        all_ocr_items: List[OCRResultItem] = []
        if len(native_text) < 30:
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        img_bytes = rel.target_part.blob
                        img_cv = ImagePreprocessor.bytes_to_cv2(img_bytes)
                        items, _ = self.engine.process_image(img_cv, preprocess=preprocess)
                        all_ocr_items.extend(items)
                        txt = PostProcessor.assemble_full_text(items)
                        if txt:
                            ocr_text_segments.append(txt)
                    except Exception as e:
                        logger.debug(f"Failed to OCR embedded image in DOCX: {e}")

        full_text = native_text
        if ocr_text_segments:
            full_text = (full_text + "\n\n" + "\n\n".join(ocr_text_segments)).strip()

        entities = PostProcessor.extract_structured_entities(full_text)

        return {
            "success": True,
            "filename": filename,
            "file_type": "docx",
            "total_pages": 1,
            "full_text": full_text,
            "pages": [{
                "page_number": 1,
                "extraction_method": "native_docx" if not ocr_text_segments else "hybrid_docx_ocr",
                "text": full_text,
                "lines_count": len(full_text.splitlines()),
                "confidence": 1.0,
                "results": [it.to_dict() for it in all_ocr_items]
            }],
            "entities": entities,
            "tables": tables_extracted,
            "meta": {
                "engine": self.ocr_version,
                "device": self.device.upper(),
                "lang": self.lang
            }
        }

    def _process_image(
        self,
        file_bytes: bytes,
        filename: str,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Process single image uploads with PaddleOCR PP-OCRv6 Tiny on CPU.
        """
        img = ImagePreprocessor.bytes_to_cv2(file_bytes)
        items, meta = self.engine.process_image(img, preprocess=preprocess)
        
        sorted_items = PostProcessor.sort_reading_order(items)
        full_text = PostProcessor.assemble_full_text(items)
        entities = PostProcessor.extract_structured_entities(full_text)
        table_data = PostProcessor.reconstruct_table(items)

        avg_conf = (
            round(sum(it.confidence for it in sorted_items) / len(sorted_items), 4)
            if sorted_items else 0.0
        )

        return {
            "success": True,
            "filename": filename,
            "file_type": "image",
            "total_pages": 1,
            "full_text": full_text,
            "pages": [{
                "page_number": 1,
                "extraction_method": "ocr_pp_ocrv6",
                "text": full_text,
                "lines_count": len(sorted_items),
                "confidence": avg_conf,
                "results": [it.to_dict() for it in sorted_items]
            }],
            "results": [it.to_dict() for it in sorted_items],
            "entities": entities,
            "tables": [table_data] if table_data["rows"] > 0 else [],
            "meta": meta
        }
