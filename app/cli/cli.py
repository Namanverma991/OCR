import sys
import os
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import cv2
from tqdm import tqdm

from app.core.config import settings
from app.core.ocr_engine import OCREngine
from app.core.postprocessor import PostProcessor
from app.core.preprocessor import ImagePreprocessor


def run_cli():
    parser = argparse.ArgumentParser(description="Industrial PaddleOCR CLI Tool")
    parser.add_argument("--input", "-i", required=True, help="Path to input image, PDF, or directory")
    parser.add_argument("--output", "-o", default="./outputs", help="Output directory for results")
    parser.add_argument("--lang", "-l", default="en", help="Language code (en, ch, hi, fr, etc.)")
    parser.add_argument("--format", "-f", choices=["json", "txt", "csv", "all"], default="all", help="Output format")
    parser.add_argument("--deskew", action="store_true", default=True, help="Auto-deskew images")
    parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU acceleration")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[Error] Input path '{input_path}' does not exist.")
        sys.exit(1)

    # Collect files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    files = []
    if input_path.is_file():
        files.append(input_path)
    else:
        for ext in valid_exts:
            files.extend(list(input_path.glob(f"*{ext}")))
            files.extend(list(input_path.glob(f"*{ext.upper()}")))

    if not files:
        print(f"[Error] No supported image files found in {input_path}")
        sys.exit(1)

    print(f"[Info] Initializing PP-OCR Engine (Lang: {args.lang}, GPU: {args.gpu})...")
    engine = OCREngine(lang=args.lang, use_gpu=args.gpu)

    print(f"[Info] Processing {len(files)} file(s)...")
    for file_path in tqdm(files, desc="OCR Processing"):
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                continue
            
            items, meta = engine.process_image(img, preprocess=args.deskew)
            sorted_items = PostProcessor.sort_reading_order(items)
            full_text = PostProcessor.assemble_full_text(items)
            stem = file_path.stem

            if args.format in ["txt", "all"]:
                (output_dir / f"{stem}.txt").write_text(full_text, encoding="utf-8")

            if args.format in ["json", "all"]:
                json_data = {
                    "filename": file_path.name,
                    "meta": meta,
                    "full_text": full_text,
                    "results": [it.to_dict() for it in sorted_items]
                }
                (output_dir / f"{stem}.json").write_text(json.dumps(json_data, indent=2), encoding="utf-8")

            if args.format in ["csv", "all"]:
                table_data = PostProcessor.reconstruct_table(items)
                if table_data["csv"]:
                    (output_dir / f"{stem}.csv").write_text(table_data["csv"], encoding="utf-8")

        except Exception as e:
            print(f"\n[Error] Error processing {file_path.name}: {e}")

    print(f"\n[Success] Processing complete! Results exported to: {output_dir.resolve()}")


if __name__ == "__main__":
    run_cli()
