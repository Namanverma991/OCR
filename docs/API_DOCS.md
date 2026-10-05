# 🔌 REST API Specification & Usage Guide

The OCR API provides high-performance asynchronous endpoints for image and document text recognition, layout analysis, tabular extraction, and key-value pair parsing.

Base URL: `http://localhost:8000/api/v1`

---

## Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/ocr/image` | Process single or multiple images (returns text, coordinates, confidence). |
| `POST` | `/ocr/pdf` | Process multi-page PDF documents page-by-page. |
| `POST` | `/ocr/table` | Extract structured tables as JSON matrix, CSV, and Markdown. |
| `POST` | `/ocr/structured` | Extract structured key-value entities (Invoices, Receipts, IDs). |
| `GET` | `/health` | System health check and device/hardware statistics. |
| `GET` | `/models` | Information on loaded models (PP-OCRv4, Det, Rec, Cls). |

---

## 1. POST `/ocr/image`

### Request
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: (Binary) Image file (PNG, JPG, WEBP, TIFF, BMP).
  - `lang`: (Optional, string, default: `"en"`) Language code (`"en"`, `"ch"`, `"hi"`, `"fr"`, `"de"`, `"es"`, etc.).
  - `use_angle_cls`: (Optional, boolean, default: `true`) Enable text orientation rectification.
  - `preprocess`: (Optional, boolean, default: `true`) Apply auto-deskew and contrast enhancement.

### Example cURL
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_invoice.png" \
  -F "lang=en" \
  -F "preprocess=true"
```

### Response Example
```json
{
  "success": true,
  "filename": "sample_invoice.png",
  "processing_time_ms": 124.5,
  "total_lines": 3,
  "full_text": "TAX INVOICE\nInvoice Number: INV-2026-0042\nTotal Amount: $1,450.00",
  "results": [
    {
      "text": "TAX INVOICE",
      "confidence": 0.994,
      "box": [[120, 45], [380, 45], [380, 85], [120, 85]],
      "center": [250, 65],
      "angle": 0.0
    },
    {
      "text": "Invoice Number: INV-2026-0042",
      "confidence": 0.988,
      "box": [[120, 105], [520, 105], [520, 135], [120, 135]],
      "center": [320, 120],
      "angle": 0.0
    },
    {
      "text": "Total Amount: $1,450.00",
      "confidence": 0.991,
      "box": [[120, 155], [440, 155], [440, 185], [120, 185]],
      "center": [280, 170],
      "angle": 0.0
    }
  ],
  "meta": {
    "engine": "PP-OCRv4",
    "device": "CPU"
  }
}
```

---

## 2. POST `/ocr/table`

### Request
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file`: Image containing tabular data.

### Response Example
```json
{
  "success": true,
  "filename": "financial_report.png",
  "table_count": 1,
  "tables": [
    {
      "rows": 3,
      "cols": 3,
      "headers": ["Item Description", "Quantity", "Price ($)"],
      "matrix": [
        ["Cloud Server Instance (c6i.4xlarge)", "2", "380.00"],
        ["Managed Postgres Database", "1", "240.00"],
        ["SSL Wildcard Certificate", "1", "99.00"]
      ],
      "csv": "Item Description,Quantity,Price ($)\nCloud Server Instance (c6i.4xlarge),2,380.00\nManaged Postgres Database,1,240.00\nSSL Wildcard Certificate,1,99.00\n",
      "markdown": "| Item Description | Quantity | Price ($) |\n| --- | --- | --- |\n| Cloud Server Instance (c6i.4xlarge) | 2 | 380.00 |\n| Managed Postgres Database | 1 | 240.00 |\n| SSL Wildcard Certificate | 1 | 99.00 |"
    }
  ]
}
```

---

## 3. POST `/ocr/structured`

### Request
- Extracts structured key-value entities such as dates, invoice numbers, tax codes, and totals.

### Response Example
```json
{
  "success": true,
  "entities": {
    "invoice_number": "INV-2026-0042",
    "invoice_date": "2026-10-05",
    "due_date": "2026-10-25",
    "total_amount": "1,450.00",
    "currency": "USD",
    "vendor_name": "ACME Cloud Solutions Inc.",
    "tax_id": "US-8829104-X"
  },
  "raw_lines_count": 28
}
```

---

## 4. Python SDK Quickstart Example

```python
import requests

url = "http://localhost:8000/api/v1/ocr/image"
with open("document.jpg", "rb") as f:
    files = {"file": f}
    data = {"lang": "en", "preprocess": "true"}
    response = requests.post(url, files=files, data=data)

result = response.json()
print("Extracted Text:")
print(result["full_text"])
```
