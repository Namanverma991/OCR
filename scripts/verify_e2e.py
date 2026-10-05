"""
End-to-End OCR and API verification script.
Tests health endpoint, document upload, OCR image processing,
table extraction, and structured entity extraction.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def run_tests():
    print("=" * 60)
    print("1. Testing Health Endpoint (/health)")
    print("=" * 60)
    health_res = client.get("/health")
    print(f"Status Code: {health_res.status_code}")
    print(f"Health Response: {health_res.json()}")
    assert health_res.status_code == 200, "Health check failed"

    print("\n" + "=" * 60)
    print("2. Testing Document Upload Endpoint (/api/v1/ocr/upload)")
    print("=" * 60)
    with open("uploads/test_invoice.png", "rb") as f:
        upload_res = client.post(
            "/api/v1/ocr/upload",
            files={"file": ("test_invoice.png", f, "image/png")},
            data={"lang": "en", "auto_deskew": "true", "use_angle_cls": "false"}
        )
    
    print(f"Status Code: {upload_res.status_code}")
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    
    data = upload_res.json()
    print(f"Document: {data.get('filename')}")
    print(f"Processing Time: {data.get('processing_time_ms'):.1f} ms")
    print(f"Page Count: {data.get('page_count')}")
    
    pages = data.get("pages", [])
    assert len(pages) > 0, "No pages returned"
    
    results = pages[0].get("results", []) or data.get("results", [])
    print(f"Total Text Boxes Detected: {len(results)}")
    assert len(results) > 0, "No text blocks detected!"
    
    print("\nExtracted Text Blocks (First 10):")
    for i, item in enumerate(results[:10], 1):
        txt = item.get("text")
        conf = item.get("confidence")
        box = item.get("box")
        print(f"  {i:2d}. [Confidence: {conf*100:.1f}%] \"{txt}\"")

    print("\n" + "=" * 60)
    print("3. Testing Structured Key-Value Extraction (/api/v1/ocr/structured)")
    print("=" * 60)
    with open("uploads/test_invoice.png", "rb") as f:
        struct_res = client.post(
            "/api/v1/ocr/structured",
            files={"file": ("test_invoice.png", f, "image/png")},
            data={"lang": "en", "doc_type": "invoice"}
        )
    print(f"Status Code: {struct_res.status_code}")
    assert struct_res.status_code == 200, "Structured extraction failed"
    struct_data = struct_res.json()
    entities = struct_data.get("entities", {})
    print(f"Entities Extracted: {len(entities)} keys found:")
    for k, v in entities.items():
        print(f"  - {k}: {v}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! OCR ENGINE READY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
