# 🏗️ System Architecture & Engineering Design

## 1. Overview
The OCR System is designed as a modular, cloud-native microservice with high-throughput batch processing capabilities, interactive visualization, and flexible integration surfaces.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Client Tier                                   │
│  ┌────────────────────────┐  ┌───────────────────┐  ┌───────────────┐  │
│  │ Web Interactive UI     │  │ Python / REST API │  │ CLI Runner    │  │
│  │ (Canvas Box Inspector) │  │ Clients / SDK     │  │ (Batch / Dir) │  │
│  └───────────┬────────────┘  └─────────┬─────────┘  └───────┬───────┘  │
└──────────────┼─────────────────────────┼────────────────────┼──────────┘
               │                         │                    │
               ▼                         ▼                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       FastAPI Gateway Layer                            │
│  • Endpoint Routing (/ocr/image, /ocr/pdf, /ocr/table, /ocr/structured) │
│  • CORS, Security, Rate Limiting, File Validation, Error Handling      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Core Processing Engine                          │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Image Preprocessing Pipeline (Deskew, CLAHE, Adaptive Filter)  │   │
│   └──────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                     │
│   ┌──────────────────────────────▼─────────────────────────────────┐   │
│   │ PaddleOCR / PP-OCR Multi-Stage Inference Engine                │   │
│   │  ├── Stage 1: Text Detection (DBNet++)                         │   │
│   │  ├── Stage 2: Direction / Angle Classifier (PPLCNet)           │   │
│   │  └── Stage 3: Text Recognition (SVTR-LCNet / CTC)              │   │
│   └──────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                     │
│   ┌──────────────────────────────▼─────────────────────────────────┐   │
│   │ Post-Processing & Layout Reconstructor                         │   │
│   │  ├── Bounding Box Sorting & Line-Grouping Algorithm            │   │
│   │  ├── Table Structure Matrix Builder & HTML/CSV Exporter        │   │
│   │  └── Key-Value & Regex Entity Extractor                        │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
e:\OCR\
├── app\
│   ├── __init__.py
│   ├── core\
│   │   ├── __init__.py
│   │   ├── config.py             # Settings, model parameters, device detection (CPU/CUDA)
│   │   ├── preprocessor.py       # Auto-rotation, contrast enhancement, noise removal, deskewing
│   │   ├── ocr_engine.py         # Complete PP-OCR pipeline wrapper (Detection, Orientation, Recognition)
│   │   ├── postprocessor.py      # Bounding box spatial sorting, line grouping, table reconstruction
│   │   └── layout_engine.py      # PP-Structure layout analysis, table cells, key-value extraction
│   ├── api\
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application with CORS, static mounting, lifespan events
│   │   ├── routes.py             # /ocr/image, /ocr/pdf, /ocr/table, /ocr/structured, /health, /metrics
│   │   └── schemas.py            # Pydantic schemas (OCRRequest, OCRResponse, TableResponse, etc.)
│   ├── cli\
│   │   ├── __init__.py
│   │   └── cli.py                # Rich CLI for processing files/folders/PDFs with progress bars
│   └── web\
│       ├── index.html            # Ultra-sleek, glassmorphic modern dashboard
│       ├── css\
│       │   └── style.css         # Modern typography, dark mode, smooth animations, interactive highlights
│       └── js\
│           └── app.js            # Live interactive bounding box canvas, JSON viewer, CSV/MD downloader
├── docs\
│   ├── IMPLEMENTATION_PLAN.md    # Step-by-step roadmap from scratch to full enterprise deployment
│   ├── ALGORITHM_DEEP_DIVE.md    # In-depth math, loss functions, DBNet, SVTR, CTC, Distillation
│   ├── ARCHITECTURE.md           # High-level architecture, module breakdown & data flow
│   ├── API_DOCS.md               # Complete REST API reference with curl/Python examples
│   └── PP_OCR_Architecture_and_Algorithm_Guide.pdf # Professional compiled PDF Guide
├── docker\
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── scripts\
│   ├── generate_pdf_guide.py     # ReportLab script generating the comprehensive PDF document
│   ├── download_models.py        # Model artifact pre-fetcher
│   ├── sample_generator.py       # Generates sample documents for test runs
│   └── run_dev.py                # Single command launch script
├── tests\
│   ├── __init__.py
│   ├── test_engine.py
│   ├── test_api.py
│   └── test_preprocessor.py
├── requirements.txt              # Production python dependencies
├── requirements-gpu.txt          # GPU-accelerated dependencies
├── environment.yml               # Conda environment definition
└── README.md                     # Main repository README
```

---

## 3. Core Engine Components

### 1. `app.core.config.Settings`
- Controls model selection (`PP-OCRv4`, `PP-OCRv3`), device mode (`cpu`, `gpu`, `mps`), detection/recognition thresholds, and cache directories.

### 2. `app.core.preprocessor.ImagePreprocessor`
- **Deskewing**: Calculates orientation angle via Radon transform / Hough line transform and rotates back to $0^\circ$.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Amplifies faint ink, watermark-obscured text, and carbon copies.
- **Adaptive Gaussian Thresholding & Morphological Operations**: Separates foreground character strokes from textured document backgrounds.

### 3. `app.core.ocr_engine.OCREngine`
- Orchestrates PaddleOCR inference.
- Provides fallback inference pipeline for environments with CPU constraints or standalone testing.
- Computes character and word-level confidence metrics and calculates bounding polygon coordinates.

### 4. `app.core.postprocessor.PostProcessor`
- **Reading Order Sorting Algorithm**: Sorts polygons top-to-bottom, left-to-right with dynamic line-threshold clustering.
- **Table Matrix Reconstructor**: Reconstructs multi-row, multi-column tables with aligned header/cell associations.
- **Key-Value Extractor**: Applies semantic regex patterns (Invoice No, Dates, Currency, Tax ID, Total Amounts).

---

## 4. Resilience & Scalability
- **Horizontal Scaling**: Stateless FastAPI worker instances running behind NGINX / Envoy load balancers.
- **Batch Inference Optimization**: Tensor batching with Paddle inference runtime for high-throughput queues (Celery/Kafka).
- **Graceful Degradation**: Dual engine pipeline ensuring that even if GPU drivers restart or deep learning models encounter out-of-memory spikes, requests fall back smoothly.
