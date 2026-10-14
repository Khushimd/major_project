"""
Build the clean-DICM preprocess cache without loading TensorFlow.

Hair removal + denoise + 224 resize, written under
models/experiments/clean_dicm_melanoma_benign/preprocessed/.
Does not modify dataset CSVs or production files.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from preprocessing.image_preprocess import ImagePreprocessor

DATA_DIR = ROOT / "dataset" / "processed" / "dicm_clean_melanoma_benign"
CACHE_DIR = ROOT / "models" / "experiments" / "clean_dicm_melanoma_benign" / "preprocessed"
STATUS = ROOT / "models" / "experiments" / "clean_dicm_melanoma_benign" / "logs" / "status.json"


MAX_SIDE_BEFORE_HAIR = 1024


def _downscale_for_hair(rgb):
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= MAX_SIDE_BEFORE_HAIR:
        return rgb
    scale = MAX_SIDE_BEFORE_HAIR / float(longest)
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _cache_one(payload: tuple[str, str]) -> str:
    src, dest_s = payload
    dest = Path(dest_s)
    if dest.is_file():
        return dest_s
    raw = cv2.imread(src, cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(src)
    rgb = _downscale_for_hair(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    processed = ImagePreprocessor(
        target_size=(224, 224), apply_clahe=False, apply_hair_removal=True
    ).process_for_model(rgb)
    dest.parent.mkdir(parents=True, exist_ok=True)
    jpeg = np.clip(processed * 255.0, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(
        str(dest),
        cv2.cvtColor(jpeg, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not ok:
        raise RuntimeError(f"Failed to write {dest}")
    return dest_s


def _write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cache_split(name: str) -> None:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in df.itertuples(index=False):
        dest = CACHE_DIR / f"{row.image_id}.jpg"
        jobs.append((str(row.image_path), str(dest)))
    already = sum(Path(dest).is_file() for _, dest in jobs)
    workers = max(1, min(4, (os.cpu_count() or 4) - 1))
    print(f"{name}: {len(jobs)} images, {already} cached, {workers} workers", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_cache_one, job) for job in jobs]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"{name}: {done}/{len(jobs)}", flush=True)
                _write_status({"stage": "preprocess", "split": name, "done": done, "total": len(jobs)})


def main() -> None:
    for name in ("train", "val", "test"):
        cache_split(name)
    print("CACHE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
