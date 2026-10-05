"""
OCR Post-Processing & Spatial Reconstruction Engine
Performs line-clustering, reading order sorting, tabular matrix alignment,
and regex-based key-value entity extraction.
"""

import re
from typing import List, Dict, Any, Tuple
from app.core.ocr_engine import OCRResultItem


class PostProcessor:
    """Post-inference geometry and semantic analysis."""

    @staticmethod
    def sort_reading_order(items: List[OCRResultItem], y_threshold: float = 12.0) -> List[OCRResultItem]:
        """
        Sort bounding boxes in natural human reading order:
        Top-to-Bottom, Left-to-Right, grouping elements within `y_threshold` pixels into lines.
        """
        if not items:
            return []

        # Sort primarily by vertical coordinate
        sorted_by_y = sorted(items, key=lambda it: it.min_y)
        lines: List[List[OCRResultItem]] = []
        current_line: List[OCRResultItem] = [sorted_by_y[0]]

        for item in sorted_by_y[1:]:
            # Check if within same line vertical range
            prev_y_center = (current_line[-1].min_y + current_line[-1].max_y) / 2.0
            curr_y_center = (item.min_y + item.max_y) / 2.0
            
            if abs(curr_y_center - prev_y_center) <= max(y_threshold, item.height * 0.5):
                current_line.append(item)
            else:
                # Sort line left-to-right
                current_line.sort(key=lambda it: it.min_x)
                lines.append(current_line)
                current_line = [item]

        if current_line:
            current_line.sort(key=lambda it: it.min_x)
            lines.append(current_line)

        # Flatten sorted lines
        sorted_items = [item for line in lines for item in line]
        return sorted_items

    @classmethod
    def assemble_full_text(cls, items: List[OCRResultItem]) -> str:
        """Join items in natural reading order with newlines between spatial lines."""
        if not items:
            return ""
        
        sorted_items = cls.sort_reading_order(items)
        lines_text = []
        current_line = [sorted_items[0].text]
        prev_y = sorted_items[0].center[1]

        for item in sorted_items[1:]:
            curr_y = item.center[1]
            if abs(curr_y - prev_y) <= max(15.0, item.height * 0.6):
                current_line.append(item.text)
            else:
                lines_text.append(" ".join(current_line))
                current_line = [item.text]
            prev_y = curr_y

        if current_line:
            lines_text.append(" ".join(current_line))

        return "\n".join(lines_text)

    @staticmethod
    def extract_structured_entities(text: str) -> Dict[str, Any]:
        """
        Extract common business document entities using compiled regex patterns.
        (Invoices, Receipts, Dates, Totals, Tax IDs, Email, Phone, Currency).
        """
        entities: Dict[str, Any] = {
            "invoice_number": None,
            "dates": [],
            "total_amount": None,
            "emails": [],
            "phone_numbers": [],
            "tax_id": None
        }

        # Date pattern (YYYY-MM-DD, DD/MM/YYYY, Month DD, YYYY)
        date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        if dates:
            entities["dates"] = list(set(dates))

        # Invoice number pattern
        inv_match = re.search(r'(?:invoice|inv|bill|receipt)\s*(?:number|no|#|\:|\-)[\s.:#-]*([A-Z0-9_-]{4,25})', text, re.IGNORECASE)
        if not inv_match:
            inv_match = re.search(r'(?:invoice|inv|bill)[\s#.:-]*([A-Z0-9_-]{4,25})', text, re.IGNORECASE)
        if inv_match:
            val = inv_match.group(1).strip()
            if val.lower() not in ["number", "invoice", "date", "total"]:
                entities["invoice_number"] = val

        # Email pattern
        emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
        if emails:
            entities["emails"] = list(set(emails))

        # Phone numbers
        phones = re.findall(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phones:
            entities["phone_numbers"] = list(set(phones))

        # Total Amount
        total_match = re.search(r'(?:total\s*amount|grand\s*total|amount\s*due|balance\s*due|net\s*amount|subtotal|total)[\s:=]*([$€£¥₹]?\s*[\d,]+\.\d{2})', text, re.IGNORECASE)
        if total_match:
            entities["total_amount"] = total_match.group(1).strip()

        # Tax ID / GSTIN / VAT
        tax_match = re.search(r'(?:tax\s*id|gstin|vat|ein)[\s#.:-]*([A-Z0-9_-]{5,25})', text, re.IGNORECASE)
        if tax_match:
            entities["tax_id"] = tax_match.group(1).strip()

        return entities

    @classmethod
    def reconstruct_table(cls, items: List[OCRResultItem]) -> Dict[str, Any]:
        """
        Group OCR blocks into a 2D spatial grid (table matrix) and convert to CSV/Markdown.
        """
        if not items:
            return {"rows": 0, "cols": 0, "matrix": [], "csv": "", "markdown": ""}

        # 1. Group into vertical rows
        sorted_items = sorted(items, key=lambda it: it.min_y)
        rows: List[List[OCRResultItem]] = []
        current_row: List[OCRResultItem] = [sorted_items[0]]

        for item in sorted_items[1:]:
            prev_y_center = (current_row[-1].min_y + current_row[-1].max_y) / 2.0
            curr_y_center = (item.min_y + item.max_y) / 2.0
            if abs(curr_y_center - prev_y_center) <= max(14.0, item.height * 0.5):
                current_row.append(item)
            else:
                current_row.sort(key=lambda it: it.min_x)
                rows.append(current_row)
                current_row = [item]

        if current_row:
            current_row.sort(key=lambda it: it.min_x)
            rows.append(current_row)

        # 2. Build string matrix
        matrix: List[List[str]] = []
        max_cols = 0
        for row in rows:
            row_texts = [cell.text for cell in row]
            max_cols = max(max_cols, len(row_texts))
            matrix.append(row_texts)

        # Pad columns
        for row in matrix:
            while len(row) < max_cols:
                row.append("")

        # 3. Generate CSV
        csv_lines = []
        for row in matrix:
            escaped = ['"' + cell.replace('"', '""') + '"' for cell in row]
            csv_lines.append(",".join(escaped))
        csv_str = "\n".join(csv_lines)

        # 4. Generate Markdown table
        md_lines = []
        if matrix:
            header = "| " + " | ".join(matrix[0]) + " |"
            separator = "| " + " | ".join(["---"] * max_cols) + " |"
            md_lines.extend([header, separator])
            for row in matrix[1:]:
                md_lines.append("| " + " | ".join(row) + " |")
        md_str = "\n".join(md_lines)

        return {
            "rows": len(matrix),
            "cols": max_cols,
            "headers": matrix[0] if matrix else [],
            "matrix": matrix,
            "csv": csv_str,
            "markdown": md_str
        }
