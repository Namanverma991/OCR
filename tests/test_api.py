import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.api.main import app
import cv2
import numpy as np
import io

client = TestClient(app)


def get_test_image_bytes():
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST OCR INVOICE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Invoice: INV-2026-001", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Total: $500.00", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_ocr_image_endpoint():
    img_bytes = get_test_image_bytes()
    response = client.post(
        "/api/v1/ocr/image",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"lang": "en", "preprocess": "true"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "results" in data
    assert "full_text" in data


def test_ocr_structured_endpoint():
    img_bytes = get_test_image_bytes()
    response = client.post(
        "/api/v1/ocr/structured",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"lang": "en"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "entities" in data


if __name__ == "__main__":
    test_health_endpoint()
    test_ocr_image_endpoint()
    test_ocr_structured_endpoint()
    print("All API Integration Tests Passed!")
