"""
Image Preprocessing Pipeline
Provides robust computer vision operations for deskewing, noise filtering,
contrast normalization, and aspect-ratio preserving resizing.
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
import io
from typing import Tuple, Union, Optional
import math


class ImagePreprocessor:
    """Production-grade image enhancement and normalization pipeline for OCR."""

    @staticmethod
    def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
        """
        Convert raw binary bytes into an OpenCV BGR numpy array.
        Automatically applies EXIF orientation transpose so mobile/WhatsApp photos match visual browser orientation.
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Handle mobile & WhatsApp camera EXIF orientation tags
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception:
            # Fallback direct OpenCV decoding
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid image format or corrupted bytes.")
            return img

    @staticmethod
    def cv2_to_bytes(img: np.ndarray, ext: str = ".jpg") -> bytes:
        """Convert OpenCV image back to bytes."""
        success, encoded_img = cv2.imencode(ext, img)
        if not success:
            raise ValueError("Failed to encode image to bytes.")
        return encoded_img.tobytes()

    @classmethod
    def enhance_contrast(cls, img: np.ndarray) -> np.ndarray:
        """
        Apply gentle Contrast Limited Adaptive Histogram Equalization (CLAHE)
        on the Luminance channel (LAB color space).
        """
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            return clahe.apply(img)
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @classmethod
    def estimate_skew_angle(cls, gray: np.ndarray) -> float:
        """
        Estimate document skew angle using Hough Line Transform on morphological gradients.
        Only applied on sufficiently large pages/documents.
        """
        h, w = gray.shape[:2]
        if h < 400 or w < 400:
            return 0.0

        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        lines = cv2.HoughLinesP(
            dilated, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20
        )
        
        if lines is None or len(lines) == 0:
            return 0.0

        angles = []
        for line in lines:
            coords = line.ravel()
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:
                    angles.append(angle)

        if not angles:
            return 0.0

        return float(np.median(angles))

    @classmethod
    def rotate_image(cls, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image around its center by the specified angle with background padding."""
        if abs(angle) < 0.5:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(
            img, rot_mat, (new_w, new_h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    @classmethod
    def deskew(cls, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Automatically detect document skew and correct it."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        angle = cls.estimate_skew_angle(gray)
        if abs(angle) >= 0.8:
            return cls.rotate_image(img, angle), angle
        return img, 0.0

    @classmethod
    def resize_max_side(cls, img: np.ndarray, max_side: int = 2048) -> np.ndarray:
        """Resize image keeping aspect ratio such that max(h, w) <= max_side."""
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return img
        
        scale = max_side / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @classmethod
    def preprocess_pipeline(
        cls, 
        img: np.ndarray, 
        auto_deskew: bool = False, 
        enhance_contrast: bool = False,
        max_side: int = 2048
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute end-to-end preprocessing pipeline.
        Returns preprocessed OpenCV image and metadata dict.
        """
        metadata = {
            "original_shape": img.shape[:2],
            "deskew_angle": 0.0,
            "contrast_enhanced": False,
        }
        
        # 1. Resize if excessively large
        processed = cls.resize_max_side(img, max_side=max_side)
        
        # 2. Deskew if large document
        if auto_deskew:
            processed, angle = cls.deskew(processed)
            metadata["deskew_angle"] = round(angle, 2)
            
        # 3. Contrast amplification if enabled
        if enhance_contrast:
            processed = cls.enhance_contrast(processed)
            metadata["contrast_enhanced"] = True
            
        metadata["processed_shape"] = processed.shape[:2]
        return processed, metadata
