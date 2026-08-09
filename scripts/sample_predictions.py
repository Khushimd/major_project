"""
Generate a few sample predictions for documentation / demo history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from utils.pipeline import MelanomaPipeline


def make_lesion(seed: int, darker: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    img = rng.integers(50, 190, size=(384, 384, 3), dtype=np.uint8)
    color = (25, 15, 10) if darker else (160, 120, 100)
    cv2.circle(img, (192, 192), int(rng.integers(40, 90)), color, -1)
    path = config.UPLOAD_DIR / f"sample_{seed}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def main() -> None:
    pipe = MelanomaPipeline()
    cases = [
        ({"name": "Sample A", "age": 34, "sex": "Female"},
         {"breslow_thickness_mm": 0.4, "ulceration": "no", "mitotic_rate": 0,
          "lymph_node_involvement": "no", "metastasis": "no"}, False),
        ({"name": "Sample B", "age": 58, "sex": "Male"},
         {"breslow_thickness_mm": 2.1, "ulceration": "yes", "mitotic_rate": 3,
          "lymph_node_involvement": "no", "metastasis": "no"}, True),
        ({"name": "Sample C", "age": 67, "sex": "Male"},
         {"breslow_thickness_mm": 1.5, "ulceration": "yes", "mitotic_rate": 2,
          "lymph_node_involvement": "yes", "metastasis": "no",
          "num_positive_nodes": 2}, True),
    ]
    for i, (patient, clinical, dark) in enumerate(cases, start=1):
        path = make_lesion(100 + i, darker=dark)
        result = pipe.run(str(path), patient, clinical, doctor_notes="Sample prediction")
        print(
            f"[{i}] {patient['name']}: {result['prediction']['prediction']} "
            f"stage={result['staging']['stage'] if result['staging'] else 'N/A'} "
            f"report={result['report_path']}"
        )


if __name__ == "__main__":
    main()
