import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from app.core.ocr_engine import OCREngine, OCRResultItem
from app.core.postprocessor import PostProcessor


def test_postprocessor_spatial_reading_order():
    # Item 1: Top-Right
    item1 = OCRResultItem("World", 0.95, [[200, 50], [300, 50], [300, 70], [200, 70]])
    # Item 2: Top-Left
    item2 = OCRResultItem("Hello", 0.98, [[50, 50], [150, 50], [150, 70], [50, 70]])
    # Item 3: Bottom Line
    item3 = OCRResultItem("PaddleOCR Engine", 0.99, [[50, 120], [350, 120], [350, 140], [50, 140]])

    sorted_items = PostProcessor.sort_reading_order([item1, item3, item2])
    
    assert sorted_items[0].text == "Hello"
    assert sorted_items[1].text == "World"
    assert sorted_items[2].text == "PaddleOCR Engine"


def test_postprocessor_entity_extraction():
    sample_text = """
    TAX INVOICE
    Invoice Number: INV-2026-9081
    Date: 2026-10-05
    Customer Support: billing@enterprise.ai
    Total Amount: $1,950.00
    Tax ID: US-991024-X
    """
    entities = PostProcessor.extract_structured_entities(sample_text)
    assert entities["invoice_number"] == "INV-2026-9081"
    assert "2026-10-05" in entities["dates"]
    assert "billing@enterprise.ai" in entities["emails"]
    assert entities["total_amount"] == "$1,950.00"
    assert entities["tax_id"] == "US-991024-X"


def test_table_reconstruction():
    item_h1 = OCRResultItem("Product", 0.99, [[50, 50], [150, 50], [150, 70], [50, 70]])
    item_h2 = OCRResultItem("Price", 0.99, [[200, 50], [300, 50], [300, 70], [200, 70]])
    item_r1 = OCRResultItem("OCR License", 0.95, [[50, 100], [150, 100], [150, 120], [50, 120]])
    item_r2 = OCRResultItem("$500", 0.97, [[200, 100], [300, 100], [300, 120], [200, 120]])

    table_data = PostProcessor.reconstruct_table([item_h1, item_h2, item_r1, item_r2])
    assert table_data["rows"] == 2
    assert table_data["cols"] == 2
    assert "Product" in table_data["csv"]
    assert "$500" in table_data["markdown"]


if __name__ == "__main__":
    test_postprocessor_spatial_reading_order()
    test_postprocessor_entity_extraction()
    test_table_reconstruction()
    print("All Engine & PostProcessor Tests Passed!")
