# 🚀 How to Run PaddleOCR Studio & Enterprise REST API

A complete, step-by-step guide for setting up and running the Industrial OCR Platform on **Windows**, **Linux**, and **macOS**.

---

## 📋 Prerequisites

- **Python 3.10 or 3.11** (Recommended: Python 3.11)
- **Git** installed on your system
- **Internet connection** on first run (to download standard PP-OCR model weights ~15MB)

---

## 🛠️ Step-by-Step Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Namanverma991/OCR.git
cd OCR
```

---

### 2. Create and Activate a Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell gives a script execution policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate).*

#### On Windows (Command Prompt `cmd`):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### On Linux / macOS (Bash / Zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install All Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 **Note**: All stable dependency versions (including `paddlepaddle<3.0.0`, `paddleocr 2.7.3`, and `numpy<2.0.0`) are locked inside `requirements.txt` to guarantee 100% out-of-the-box CPU/GPU compatibility without conflicts.

---

### 4. Run the Application

#### Option A: Run the Interactive Web Studio & API Server (Recommended)

```bash
# Windows
.\.venv\Scripts\python.exe scripts/run_dev.py

# Linux / macOS
python scripts/run_dev.py
```

Once started, open your browser and visit:
- 🌐 **Interactive Web Dashboard:** [http://localhost:8000](http://localhost:8000)
- 📚 **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- 🩺 **Health Check:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

#### Option B: Run via CLI Batch Processing

Process an image or PDF directly from your terminal:

```bash
# Basic single image extraction
python app/cli/cli.py ocr uploads/test_invoice.png

# Extract structured key-value entities to JSON
python app/cli/cli.py structured uploads/test_invoice.png --doc-type invoice --output results.json

# Extract tables to CSV
python app/cli/cli.py table uploads/test_invoice.png --output-csv table.csv
```

---

#### Option C: Run with Docker (Optional)

If you prefer running inside Docker:

```bash
# Build and run with Docker Compose
docker-compose up --build
```
Access at [http://localhost:8000](http://localhost:8000).

---

## 🧪 Verify the Installation

To verify that the OCR engine and all API endpoints are working properly on your machine, run the automated verification script:

```bash
python scripts/verify_e2e.py
```

You should see:
```text
============================================================
ALL TESTS PASSED SUCCESSFULLY! OCR ENGINE READY!
============================================================
```

---

## 📁 Supported File Formats

| Document Type | Extensions | Processing Method |
| :--- | :--- | :--- |
| **Images** | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp` | PaddleOCR DBNet + SVTR-LCNet |
| **PDF Documents** | `.pdf` | Native Digital Text + PP-OCR fallback for scanned pages |
| **Word Documents** | `.docx`, `.doc` | Native XML text extraction + OCR on embedded images |

---

## ❓ Troubleshooting & FAQs

### Q1: Bounding boxes are empty or 0 lines detected?
- Ensure `numpy<2.0.0` is installed. Run `pip install "numpy<2.0.0"`.

### Q2: Port 8000 is already in use?
- You can change the port in `.env` or pass `--port 8080`:
  ```bash
  uvicorn app.api.main:app --host 0.0.0.0 --port 8080 --reload
  ```

---
⭐ *Happy Text Extracting!*
