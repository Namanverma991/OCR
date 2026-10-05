"""
Core OCR Engine Module
Wraps PaddleOCR (PP-OCRv6 Tiny / PP-OCRv4 Mobile CPU) with unified batching, fallback logic,
confidence normalization, and multi-language routing.
"""

import os
# Configure CPU execution flags before paddle import to prevent OneDNN/PIR runtime attribute conflicts
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from app.core.config import settings
from app.core.preprocessor import ImagePreprocessor

logger = logging.getLogger("ocr.engine")
logging.basicConfig(level=logging.INFO)

# Apply runtime flags to paddle if available
try:
    import paddle
    paddle.set_flags({
        "FLAGS_enable_pir_api": 0,
        "FLAGS_use_mkldnn": 0,
        "FLAGS_enable_onednn": 0
    })
except Exception:
    pass


class OCRResultItem:
    """Represents an individual recognized text block with geometry and score."""
    def __init__(
        self,
        text: str,
        confidence: float,
        box: List[List[float]],
        angle: float = 0.0,
        cls_confidence: float = 1.0,
    ):
        self.text = str(text).strip()
        self.confidence = float(confidence)
        # 4 points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        self.box = [[float(p[0]), float(p[1])] for p in box] if box else [[0.0, 0.0]] * 4
        self.angle = float(angle)
        self.cls_confidence = float(cls_confidence)
        
        # Calculate center & approximate bounding rectangle
        if self.box:
            xs = [p[0] for p in self.box]
            ys = [p[1] for p in self.box]
            self.min_x = min(xs)
            self.max_x = max(xs)
            self.min_y = min(ys)
            self.max_y = max(ys)
        else:
            self.min_x = self.max_x = self.min_y = self.max_y = 0.0
            
        self.center = [(self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0]
        self.width = max(0.0, self.max_x - self.min_x)
        self.height = max(0.0, self.max_y - self.min_y)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "box": self.box,
            "center": [round(self.center[0], 1), round(self.center[1], 1)],
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "angle": self.angle,
        }


class OCREngine:
    """
    PP-OCR Production Engine Manager.
    Optimized for CPU inference with PaddleOCR PP-OCRv6 Tiny / PP-OCRv4 Mobile.
    Handles model caching, multi-language switching, and graceful inference execution.
    """

    _instances: Dict[str, Any] = {}
    _is_paddle_available: Optional[bool] = None

    def __init__(
        self,
        lang: Optional[str] = None,
        device: Optional[str] = None,
        use_angle_cls: Optional[bool] = None,
        ocr_version: Optional[str] = None
    ):
        self.lang = lang or settings.DEFAULT_LANG
        self.device = device or settings.OCR_DEVICE
        self.use_gpu = (self.device.lower() == "gpu") or settings.USE_GPU
        self.use_angle_cls = settings.USE_ANGLE_CLS if use_angle_cls is None else use_angle_cls
        self.ocr_version = ocr_version or settings.OCR_VERSION
        self.ocr_model = self._get_or_create_model(
            lang=self.lang,
            device=self.device,
            use_angle_cls=self.use_angle_cls,
            ocr_version=self.ocr_version
        )

    @classmethod
    def check_paddle_installed(cls) -> bool:
        """Check if paddlepaddle and paddleocr are installed in the environment."""
        if cls._is_paddle_available is not None:
            return cls._is_paddle_available
        try:
            import paddleocr
            cls._is_paddle_available = True
            logger.info("PaddleOCR verified successfully.")
        except ImportError as e:
            cls._is_paddle_available = False
            logger.warning(f"PaddleOCR native packages not found: {e}.")
        return cls._is_paddle_available

    def _get_or_create_model(
        self,
        lang: str,
        device: str,
        use_angle_cls: bool,
        ocr_version: str
    ):
        """Retrieve cached model instance or initialize a new PaddleOCR pipeline."""
        cache_key = f"{lang}_{device}_{use_angle_cls}_{ocr_version}"
        if cache_key in self._instances:
            return self._instances[cache_key]

        if not settings.OCR_ENABLED:
            logger.info("OCR is disabled via configuration.")
            return None

        if self.check_paddle_installed():
            try:
                from paddleocr import PaddleOCR
                
                use_gpu_flag = (device.lower() == "gpu") or settings.USE_GPU
                model = PaddleOCR(
                    use_angle_cls=use_angle_cls,
                    lang=lang,
                    use_gpu=use_gpu_flag,
                    show_log=False,
                    ocr_version=ocr_version or "PP-OCRv4"
                )
                logger.info(f"Initialized PaddleOCR engine: lang='{lang}', version='{ocr_version}', gpu={use_gpu_flag}")
                self._instances[cache_key] = model
                return model
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR model: {e}", exc_info=True)
                return None
        return None

    def _parse_paddle_results(self, raw_results: Any) -> List[OCRResultItem]:
        """Normalize results from PaddleOCR pipelines into OCRResultItem list."""
        items: List[OCRResultItem] = []
        if not raw_results:
            return items

        # Case 1: PaddleOCR v3 dictionary structure
        if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], dict):
            for res_dict in raw_results:
                rec_texts = res_dict.get("rec_texts", [])
                rec_scores = res_dict.get("rec_scores", [])
                rec_polys = res_dict.get("rec_polys")
                if rec_polys is None:
                    rec_polys = res_dict.get("dt_polys", [])
                angles = res_dict.get("textline_orientation_angles", [])

                for i, text in enumerate(rec_texts):
                    score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                    if score < settings.REC_CONFIDENCE_THRESH:
                        continue
                    poly = rec_polys[i] if i < len(rec_polys) else []
                    box = poly.tolist() if isinstance(poly, np.ndarray) else poly
                    angle = float(angles[i]) if i < len(angles) else 0.0
                    items.append(OCRResultItem(text=text, confidence=score, box=box, angle=angle))
            return items

        # Case 2: PaddleOCR v2 / PP-OCR standard list: [[[points], (text, score)], ...]
        if isinstance(raw_results, list):
            lines = raw_results[0] if (len(raw_results) == 1 and isinstance(raw_results[0], list)) else raw_results
            if lines is None:
                return items
            for line in lines:
                if not line or not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                box = line[0]
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text, conf = text_info[0], float(text_info[1])
                else:
                    text, conf = str(text_info), 1.0

                if conf >= settings.REC_CONFIDENCE_THRESH:
                    poly_box = box.tolist() if isinstance(box, np.ndarray) else box
                    items.append(OCRResultItem(text=text, confidence=conf, box=poly_box))

        return items

    def process_image(
        self, 
        img: np.ndarray, 
        preprocess: bool = False
    ) -> Tuple[List[OCRResultItem], Dict[str, Any]]:
        """
        Execute full OCR pipeline on an OpenCV image.
        Returns list of recognized text items and execution metadata.
        """
        start_time = time.time()
        meta: Dict[str, Any] = {
            "lang": self.lang,
            "engine": self.ocr_version,
            "device": self.device.upper()
        }
        
        # 1. Preprocessing
        if preprocess:
            processed_img, prep_meta = ImagePreprocessor.preprocess_pipeline(
                img, 
                auto_deskew=settings.AUTO_DESKEW, 
                enhance_contrast=settings.ENHANCE_CONTRAST,
                max_side=settings.MAX_IMAGE_SIDE_LEN
            )
            meta["preprocessing"] = prep_meta
        else:
            processed_img = img
            meta["preprocessing"] = {"bypassed": True}

        # 2. Model Inference
        items: List[OCRResultItem] = []
        if self.ocr_model is not None:
            try:
                raw_results = self.ocr_model.ocr(processed_img, cls=self.use_angle_cls)
                items = self._parse_paddle_results(raw_results)
                logger.info(f"OCR inference completed: {len(items)} text blocks detected.")
            except Exception as e:
                logger.error(f"Inference error in PaddleOCR: {e}", exc_info=True)
                items = []
        else:
            logger.error("PaddleOCR model is not initialized (ocr_model is None).")

        # 3. Compute elapsed time
        elapsed_ms = (time.time() - start_time) * 1000.0
        meta["processing_time_ms"] = round(elapsed_ms, 2)
        meta["total_detected_blocks"] = len(items)
        
        return items, meta
