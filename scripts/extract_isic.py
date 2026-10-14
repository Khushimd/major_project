"""
Extract SIIM-ISIC 2020 JPEGs from the Kaggle zip into dataset/raw/train/.

Default: every melanoma image plus a seeded benign sample (CPU-friendly).
Use --all to extract the full 33,126 training JPEGs.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIP_DEFAULT = Path.home() / "Downloads" / "ISIC_2020_Training_JPEG.zip"
CSV_DEFAULT = Path.home() / "Downloads" / "ISIC_2020_Training_GroundTruth.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=ZIP_DEFAULT)
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "raw")
    parser.add_argument("--benign-sample", type=int, default=8000)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.zip.exists():
        raise FileNotFoundError(f"ISIC zip not found: {args.zip}")
    if not args.csv.exists():
        raise FileNotFoundError(f"ISIC CSV not found: {args.csv}")

    out = args.out
    train_dir = out / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.csv, out / "train.csv")

    df = pd.read_csv(args.csv)
    if args.all:
        keep = set(df["image_name"].astype(str) + ".jpg")
        print(f"Extracting ALL {len(keep)} SIIM-ISIC JPEGs")
    else:
        mal = df[df["target"] == 1]
        n_ben = min(int(args.benign_sample), int((df["target"] == 0).sum()))
        ben = df[df["target"] == 0].sample(n=n_ben, random_state=42)
        keep = set(pd.concat([mal, ben])["image_name"].astype(str) + ".jpg")
        print(
            f"Extracting {len(keep)} SIIM-ISIC JPEGs "
            f"({len(mal)} malignant + {len(ben)} benign)"
        )

    extracted = 0
    skipped = 0
    with zipfile.ZipFile(args.zip) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name not in keep:
                continue
            dest = train_dir / name
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                extracted += 1
                continue
            with zf.open(info) as src, open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
            extracted += 1
            if extracted % 250 == 0:
                print(f"... {extracted}/{len(keep)}")

    print(f"Done. extracted_or_present={extracted} already_on_disk={skipped}")


if __name__ == "__main__":
    main()
