"""
Extract AJCC clinical parameters from uploaded clinical reports.

Supports: PDF, TXT, DOCX, and image OCR when Tesseract is available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger("clinical_report")

ALLOWED_REPORT_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires the pypdf package. Run: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_docx_text(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "DOCX support requires python-docx. Run: pip install python-docx"
        ) from exc
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_image_text(path: Path) -> str:
    """OCR via pytesseract if installed; otherwise empty."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("OCR libraries not installed; cannot read image reports.")
        return ""
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img) or ""
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", path, exc)
        return ""


def extract_text_from_report(path: Path) -> str:
    """Load text content from a clinical report file."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf_text(path)
    if ext in {".txt", ".text", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext in {".docx", ".doc"}:
        return _read_docx_text(path)
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
        return _read_image_text(path)
    raise ValueError(f"Unsupported report type: {ext}")


def _first_float(patterns: list[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                continue
    return None


def _yes_no(patterns_yes: list[str], patterns_no: list[str], text: str) -> Optional[str]:
    for pat in patterns_yes:
        if re.search(pat, text, flags=re.IGNORECASE):
            return "yes"
    for pat in patterns_no:
        if re.search(pat, text, flags=re.IGNORECASE):
            return "no"
    return None


def parse_clinical_text(text: str) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """
    Parse clinical fields from free text.

    Returns:
        (values, found_flags) where found_flags marks which keys were detected.
    """
    cleaned = re.sub(r"[ \t]+", " ", text or "")
    values: Dict[str, Any] = {
        "breslow_thickness_mm": 0.0,
        "ulceration": "no",
        "mitotic_rate": 0.0,
        "lymph_node_involvement": "no",
        "metastasis": "no",
        "num_positive_nodes": 0,
    }
    found = {k: False for k in values}

    breslow = _first_float(
        [
            r"breslow(?:\s+thickness)?(?:\s*\(mm\))?\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)",
            r"tumor\s+thickness\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)\s*mm",
            r"thickness\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)\s*mm",
            r"([0-9]+(?:[.,][0-9]+)?)\s*mm\s+(?:breslow|thickness)",
        ],
        cleaned,
    )
    if breslow is not None:
        values["breslow_thickness_mm"] = breslow
        found["breslow_thickness_mm"] = True

    mitotic = _first_float(
        [
            r"mitotic\s+rate\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)",
            r"mitoses?\s*(?:per|/)\s*mm(?:\u00b2|2)?\s*[:\-]?\s*([0-9]+(?:[.,][0-9]+)?)",
            r"([0-9]+(?:[.,][0-9]+)?)\s*mitoses?\s*/\s*mm",
        ],
        cleaned,
    )
    if mitotic is not None:
        values["mitotic_rate"] = mitotic
        found["mitotic_rate"] = True

    ulcer = _yes_no(
        [
            r"ulceration\s*[:\-]?\s*(yes|present|positive|\+|identified)",
            r"with\s+ulceration",
            r"ulcerated\s*[:=]?\s*yes",
        ],
        [
            r"ulceration\s*[:\-]?\s*(no|absent|negative|not\s+identified|none)",
            r"without\s+ulceration",
            r"non[- ]ulcerated",
        ],
        cleaned,
    )
    if ulcer is not None:
        values["ulceration"] = ulcer
        found["ulceration"] = True

    nodes = _yes_no(
        [
            r"lymph\s+nodes?\s*[:\-]?\s*(yes|positive|involved|metastasis|present)",
            r"nodal\s+involvement\s*[:\-]?\s*(yes|positive|present)",
            r"sentinel\s+node\s+positive",
            r"regional\s+lymph\s+node\s+metastasis",
        ],
        [
            r"lymph\s+nodes?\s*[:\-]?\s*(no|negative|not\s+involved|absent|clear)",
            r"no\s+lymph\s+node\s+involvement",
            r"node[- ]negative",
            r"sentinel\s+node\s+negative",
        ],
        cleaned,
    )
    if nodes is not None:
        values["lymph_node_involvement"] = nodes
        found["lymph_node_involvement"] = True

    mets = _yes_no(
        [
            r"distant\s+metastasis\s*[:\-]?\s*(yes|present|positive|identified)",
            r"metastasis\s*[:\-]?\s*(yes|present|positive)",
            r"m1[abc]?",
            r"stage\s*iv\b",
        ],
        [
            r"distant\s+metastasis\s*[:\-]?\s*(no|absent|negative|none)",
            r"no\s+distant\s+metastasis",
            r"metastasis\s*[:\-]?\s*(no|absent|negative|none)",
            r"\bm0\b",
        ],
        cleaned,
    )
    if mets is not None:
        values["metastasis"] = mets
        found["metastasis"] = True

    n_pos = _first_float(
        [
            r"(?:number\s+of\s+)?positive\s+nodes?\s*[:\-]?\s*([0-9]+)",
            r"([0-9]+)\s+positive\s+(?:lymph\s+)?nodes?",
        ],
        cleaned,
    )
    if n_pos is not None:
        values["num_positive_nodes"] = int(n_pos)
        found["num_positive_nodes"] = True
        if values["num_positive_nodes"] > 0:
            values["lymph_node_involvement"] = "yes"
            found["lymph_node_involvement"] = True

    return values, found


def parse_clinical_report(path: Path) -> Dict[str, Any]:
    """
    Extract clinical parameters from a report file.

    Returns dict with keys: values, found, raw_excerpt, text_length
    """
    text = extract_text_from_report(path)
    if not text.strip():
        raise ValueError(
            "Could not read text from this report. "
            "Use a text-based PDF/TXT/DOCX, or enter values manually. "
            "Scanned image reports need Tesseract OCR installed."
        )
    values, found = parse_clinical_text(text)
    excerpt = " ".join(text.split())[:500]
    return {
        "values": values,
        "found": found,
        "raw_excerpt": excerpt,
        "text_length": len(text),
        "fields_found": sum(1 for v in found.values() if v),
    }
