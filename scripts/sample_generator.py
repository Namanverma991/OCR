"""
Sample Test Document Generator
Creates synthetic invoices, tabular reports, and noisy rotated documents for testing.
"""

import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_sample_invoice(filename: str = "test_invoice.png") -> Path:
    """Generate high-contrast sample tax invoice with tabular items."""
    img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Header
    cv2.putText(img, "GLOBAL TECH ENTERPRISES", (60, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (20, 20, 20), 2)
    cv2.putText(img, "500 Silicon Boulevard, Tech City, CA 95054", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(img, "Email: billing@globaltech.com | Phone: +1-800-555-0199", (60, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    cv2.line(img, (60, 160), (740, 160), (200, 200, 200), 2)
    
    # Metadata
    cv2.putText(img, "COMMERCIAL INVOICE", (60, 200), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(img, "Invoice Number: INV-2026-8832", (60, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.putText(img, "Invoice Date: 2026-10-05", (60, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.putText(img, "Tax ID: US-449102-TX", (60, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    
    # Table Header
    cv2.rectangle(img, (60, 340), (740, 375), (240, 240, 240), -1)
    cv2.rectangle(img, (60, 340), (740, 375), (180, 180, 180), 1)
    cv2.putText(img, "Item Description", (75, 365), cv2.FONT_HERSHEY_DUPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(img, "Qty", (420, 365), cv2.FONT_HERSHEY_DUPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(img, "Rate", (520, 365), cv2.FONT_HERSHEY_DUPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(img, "Amount", (640, 365), cv2.FONT_HERSHEY_DUPLEX, 0.55, (20, 20, 20), 1)
    
    # Rows
    rows = [
        ("PaddleOCR Production License", "1", "1,500.00", "1,500.00"),
        ("DBNet Text Detection Module", "2", "300.00", "600.00"),
        ("SVTR Sequence Recognizer", "2", "350.00", "700.00"),
        ("PP-Structure Table Parser", "1", "450.00", "450.00")
    ]
    
    y = 415
    for desc, qty, rate, amt in rows:
        cv2.putText(img, desc, (75, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(img, qty, (430, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(img, rate, (520, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(img, amt, (640, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.line(img, (60, y + 15), (740, y + 15), (230, 230, 230), 1)
        y += 45
        
    # Totals
    cv2.putText(img, "Subtotal: $3,250.00", (520, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.putText(img, "Tax (8.5%): $276.25", (520, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.putText(img, "Total Amount: $3,526.25", (480, 690), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)
    
    # Save
    target_file = OUTPUT_DIR / filename
    cv2.imwrite(str(target_file), img)
    print(f"Generated sample invoice at: {target_file}")
    return target_file


if __name__ == "__main__":
    generate_sample_invoice()
