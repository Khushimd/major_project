"""
Shared helper utilities for Melanoma CDSS.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import config


def allowed_file(filename: str) -> bool:
    """Return True if the file extension is allowed for upload."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def generate_unique_id(prefix: str = "PID") -> str:
    """Generate a unique patient / session identifier."""
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}-{stamp}-{short}"


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Serialize a dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file; return None if missing."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def timestamp_now() -> str:
    """Return current timestamp as ISO-like string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def confidence_to_percent(confidence: float) -> float:
    """Convert probability [0,1] to percentage rounded to 2 decimals."""
    return round(float(confidence) * 100.0, 2)


def file_sha256(path: str | Path) -> str:
    """Content hash for deterministic prediction reuse."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(original: str) -> str:
    """Create a filesystem-safe unique filename retaining the extension."""
    ext = Path(original).suffix.lower() or ".jpg"
    return f"{uuid.uuid4().hex}{ext}"
