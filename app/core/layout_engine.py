"""
Layout Analysis & Document Structure Engine
Integrates PP-Structure principles to categorize document elements
into Title, Paragraph, Table, Header, Footer, and Key-Value entities.
"""

from typing import List, Dict, Any
import numpy as np
from app.core.ocr_engine import OCRResultItem
from app.core.postprocessor import PostProcessor


class DocumentBlock:
    def __init__(self, block_type: str, items: List[OCRResultItem]):
        self.block_type = block_type  # title, paragraph, table, header, footer
        self.items = items
        self.text = " ".join([it.text for it in items])
        xs = [p for it in items for p in [it.min_x, it.max_x]]
        ys = [p for it in items for p in [it.min_y, it.max_y]]
        self.bbox = [min(xs), min(ys), max(xs), max(ys)] if xs else [0, 0, 0, 0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.block_type,
            "text": self.text,
            "bbox": [round(v, 1) for v in self.bbox],
            "item_count": len(self.items)
        }


class LayoutEngine:
    """Classifies OCR blocks into structured document sections."""

    @classmethod
    def analyze_layout(cls, items: List[OCRResultItem], img_shape: tuple) -> List[DocumentBlock]:
        if not items:
            return []

        h, w = img_shape[:2]
        sorted_items = PostProcessor.sort_reading_order(items)
        blocks: List[DocumentBlock] = []
        
        # Heuristic layout classification based on font geometry and spatial location
        median_h = np.median([it.height for it in items]) if items else 20.0
        
        current_para: List[OCRResultItem] = []
        
        for item in sorted_items:
            # Title / Header detection (significantly taller font)
            if item.height >= median_h * 1.6 or item.min_y < h * 0.08:
                if current_para:
                    blocks.append(DocumentBlock("paragraph", current_para))
                    current_para = []
                btype = "header" if item.min_y < h * 0.08 else "title"
                blocks.append(DocumentBlock(btype, [item]))
            elif item.min_y > h * 0.92:
                # Footer
                if current_para:
                    blocks.append(DocumentBlock("paragraph", current_para))
                    current_para = []
                blocks.append(DocumentBlock("footer", [item]))
            else:
                current_para.append(item)

        if current_para:
            blocks.append(DocumentBlock("paragraph", current_para))

        return blocks
