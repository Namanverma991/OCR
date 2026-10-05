import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from tqdm import tqdm

from app.core.config import settings
from app.core.document_processor import DocumentProcessor


def run_cli():
    parser = argparse.ArgumentParser(description="Industrial PaddleOCR Document & Image CLI Tool")
    parser.add_argument("--input", "-i", required=True, help="Path to input image, PDF, DOCX, or directory")
    parser.add_argument("--output", "-o", default="./outputs", help="Output directory for results")
    parser.add_argument("--lang", "-l", default=settings.DEFAULT_LANG, help="Language code (en, ch, hi, fr, etc.)")
    parser.add_argument("--format", "-f", choices=["json", "txt", "csv", "all"], default="all", help="Output format")
    parser.add_argument("--deskew", action="store_true", default=True, help="Auto-deskew images")
    parser.add_argument("--device", "-d", choices=["cpu", "gpu"], default=settings.OCR_DEVICE, help="Compute device (default: cpu)")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[Error] Input path '{input_path}' does not exist.")
        sys.exit(1)

    # Collect files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".pdf", ".docx", ".doc"}
    files = []
    if input_path.is_file():
        files.append(input_path)
    else:
        for ext in valid_exts:
            files.extend(list(input_path.glob(f"*{ext}")))
            files.extend(list(input_path.glob(f"*{ext.upper()}")))

    if not files:
        print(f"[Error] No supported document or image files found in {input_path}")
        sys.exit(1)

    print(f"[Info] Initializing PaddleOCR Engine (Engine: {settings.OCR_VERSION}, Lang: {args.lang}, Device: {args.device.upper()})...")
    processor = DocumentProcessor(
        lang=args.lang,
        device=args.device,
        ocr_version=settings.OCR_VERSION
    )

    print(f"[Info] Processing {len(files)} file(s)...")
    for file_path in tqdm(files, desc="Document Processing"):
        try:
            content = file_path.read_bytes()
            result = processor.process(file_bytes=content, filename=file_path.name, preprocess=args.deskew)
            stem = file_path.stem

            if args.format in ["txt", "all"]:
                (output_dir / f"{stem}.txt").write_text(result.get("full_text", ""), encoding="utf-8")

            if args.format in ["json", "all"]:
                (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

            if args.format in ["csv", "all"]:
                tables = result.get("tables", [])
                if tables and tables[0].get("csv"):
                    combined_csv = "\n\n".join(t["csv"] for t in tables if t.get("csv"))
                    (output_dir / f"{stem}.csv").write_text(combined_csv, encoding="utf-8")

        except Exception as e:
            print(f"\n[Error] Error processing {file_path.name}: {e}")

    print(f"\n[Success] Processing complete! Results exported to: {output_dir.resolve()}")


if __name__ == "__main__":
    run_cli()
