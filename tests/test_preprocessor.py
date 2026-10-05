import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from app.core.preprocessor import ImagePreprocessor


def test_contrast_enhancement():
    img = np.random.randint(50, 150, (200, 200, 3), dtype=np.uint8)
    enhanced = ImagePreprocessor.enhance_contrast(img)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8


def test_resize_max_side():
    img = np.ones((3000, 1500, 3), dtype=np.uint8)
    resized = ImagePreprocessor.resize_max_side(img, max_side=2048)
    assert max(resized.shape[:2]) <= 2048


def test_deskew_angle_estimation():
    img = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST OCR LINE SAMPLE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    processed, angle = ImagePreprocessor.deskew(img)
    assert isinstance(angle, float)
    assert len(processed.shape) == 3
    assert processed.shape[2] == 3


if __name__ == "__main__":
    test_contrast_enhancement()
    test_resize_max_side()
    test_deskew_angle_estimation()
    print("All Preprocessor Tests Passed!")
