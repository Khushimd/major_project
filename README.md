# AI-Powered Clinical Decision Support System for Melanoma Detection and AJCC Stage Prediction

Production-ready final-year engineering project: deep learning melanoma classification, Grad-CAM explainability, AJCC staging, PDF reports, and a Flask web application.

> **Medical disclaimer:** This system is intended as a clinical decision support tool only and should not replace professional medical diagnosis.

---

## Features

| Area | Capability |
|------|------------|
| Detection | Benign vs Malignant (EfficientNetB0, MobileNetV2, soft-voting ensemble) |
| Explainability | Grad-CAM only when malignant; otherwise “No malignant regions detected.” |
| Staging | AJCC 8th Edition rule-based stage from clinical parameters |
| Reports | PDF with patient details, confidence, Grad-CAM, stage, recommendations |
| Web app | Home, About, Upload, Prediction, Report, History, Model Comparison |
| Extras | Dark mode, confidence gauge, charts, TTA, image quality / blur checks, U-Net segmentation hook |
| Data | DICM + ISIC 2020 for the frozen 92.28% Exp2-C model |

---

## Project structure

```
├── app.py                 # Flask entry point
├── config.py              # Central configuration
├── requirements.txt
├── README.md
├── dataset/               # DICM metadata, ISIC 2020 labels, frozen splits
├── models/                # Architectures + saved .keras weights
├── training/              # Dataset loader, train, evaluate, callbacks
├── preprocessing/         # Hair removal, CLAHE, augment, quality
├── gradcam/               # Grad-CAM explainability
├── staging/               # AJCC staging engine
├── reports/               # PDF generator + generated PDFs
├── segmentation/          # Optional U-Net lesion segmentation
├── database/              # SQLite layer
├── static/                # CSS, JS, uploads, results
├── templates/             # Jinja2 pages
├── utils/                 # Logger, helpers, inference pipeline
├── scripts/               # Model initialization
└── tests/                 # Smoke / unit tests
```

---

## Installation

### 1. Prerequisites

- Python **3.10 or 3.11** (recommended)
- pip, optional NVIDIA GPU + CUDA for faster training

### 2. Create virtual environment

```bash
cd "major project"
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize models (required before first web run)

```bash
# Saves EfficientNetB0 + MobileNetV2 (ImageNet backbone + classification head)
python scripts/initialize_models.py

# Optional: short synthetic training so demo predictions are not random
python scripts/initialize_models.py --quick-train --epochs 3
```

---

## Dataset setup (92.28% Exp2-C: DICM + ISIC 2020)

The frozen **92.28%** model is Exp2-C (BCE + label smoothing 0.05). It uses **DICM + ISIC 2020**.

| Class | Source | Rule |
|-------|--------|------|
| 0 | ISIC 2020 | Official benign (`benign_malignant=benign`). DICM class=0 was **not** used. |
| 1 | DICM (ISIC-DICM-17K) | Official `class=1`. This is **not** `diagnosis == melanoma` only (BCC, SCC, and others are included). |

JPEG images are local-only (not on GitHub). Metadata and frozen splits **are** in the repo.

### 1. DICM (class 1)

Place official metadata and lesion JPEGs:

```
dataset/isic_dicm_17k/
├── train-set-metadata.csv
└── valid-set-metadata.csv

images/
└── *.jpg                 # DICM / ISIC-DICM-17K images
```

### 2. ISIC 2020 (class 0)

Download **SIIM-ISIC Melanoma Classification 2020** and keep official-benign images:

```
dataset/raw/
├── train.csv             # ISIC 2020 labels
└── train/
    └── ISIC_xxxx.jpg     # official-benign JPEGs used for class 0
```

Optional extract:

```bash
python scripts/extract_isic.py
```

### 3. Frozen splits (already in the repo)

Do **not** retune these. Grouped 70 / 15 / 15, seed 42, total 16,529 images:

| Split | n | Class 1 | Class 0 |
|-------|--:|--------:|--------:|
| Train | 11,611 | 5,966 | 5,645 |
| Val | 2,445 | 1,240 | 1,205 |
| Test | 2,473 | 1,324 | 1,149 |

Files: `dataset/processed/melanoma_benign/{train,val,test,manifest}.csv`

To rebuild the CSVs only (does not train, does not touch production):

```bash
python scripts/prepare_melanoma_benign.py
```

### 4. Frozen 92.28% weights

```
models/experiments/internal_optimization/exp2_loss_comparison/exp2c_bce_smooth/
├── efficientnetb0_bce_smooth.keras
├── mobilenetv2_bce_smooth.keras
└── EXP2C_FROZEN_LOCK.md
```

Ensemble **0.75 / 0.25**, threshold **0.30**. Frozen internal test: accuracy 92.28%, recall 95.24%, specificity 88.86%, precision 90.78%, F1 0.9296.

Do **not** overwrite these files. Do **not** replace production weights in `models/saved/`. This is **not** a clean melanoma-vs-benign result and is **not** clinically validated.

To reproduce Exp2-C (writes only under that experiment folder; needs Exp1 as the BCE control):

```bash
python scripts/exp2_loss_comparison.py
```

---

## Training guide

The trainer:

1. Loads metadata and stratified train/val/test splits  
2. Applies augmentation + class weights / focal loss  
3. Trains **EfficientNetB0** (transfer → fine-tune)  
4. Trains **MobileNetV2** (transfer → fine-tune)  
5. Evaluates **ensemble** soft voting (Exp2-C: `0.75 × Eff + 0.25 × Mob`, threshold 0.30)  
6. Writes metrics under the experiment folder only — production `models/saved/` is not overwritten  

Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, TensorBoard, CSV logger. Mixed precision enables automatically when a GPU is present.

Key knobs live in `config.py`: `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `ENSEMBLE_WEIGHTS`, `USE_FOCAL_LOSS`, `USE_TTA`.

**Frozen Exp2-C internal test (as labeled):** accuracy 92.28%, precision 90.78%, recall 95.24%, F1 0.9296, ROC-AUC 0.9799. Class 1 is DICM official `class=1`, not melanoma-only.

---

## Run the web application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Upload workflow

```
Upload → Preprocess → Segment → EfficientNet + MobileNet → Ensemble
      → Benign / Malignant → Grad-CAM (if malignant) → AJCC → PDF → SQLite
```

### Production (gunicorn)

```bash
gunicorn -w 2 -b 0.0.0.0:5000 "app:create_app()"
```

Set a strong `FLASK_SECRET_KEY` in the environment before deploying.

---

## Testing guide

```bash
# Core smoke tests (no GPU required)
python tests/test_staging.py

# Or with pytest
pytest tests/ -q
```

Coverage includes AJCC rule paths and the OpenCV preprocessing pipeline.

---

## API (optional)

`POST /api/predict` — multipart form with `image` plus optional clinical fields. Returns JSON with prediction, staging, and report path.

`GET /api/stats` — dashboard aggregates.

---

## AJCC clinical inputs (malignant)

| Parameter | Description |
|-----------|-------------|
| Breslow thickness (mm) | Tumor depth |
| Ulceration | Yes / No |
| Mitotic rate | Mitoses per mm² |
| Lymph node involvement | Yes / No (+ count) |
| Metastasis | Distant mets Yes / No |

Stages: **0, IA, IB, IIA–IIC, III, IV** with risk level, recommendation, and follow-up text.

---

## Configuration

See `config.py` for paths, image size (224×224), ensemble weights, Flask settings, and dataset registry.

---

## Deployment checklist

1. Keep Exp2-C frozen under `models/experiments/internal_optimization/exp2_loss_comparison/exp2c_bce_smooth/` (do not overwrite production `models/saved/`)  
2. Keep only best weights in `models/saved/`  
3. Set `FLASK_DEBUG=0` and a secure `FLASK_SECRET_KEY`  
4. Serve behind HTTPS reverse proxy (nginx)  
5. Persist `database/melanoma_cdss.db`, `static/uploads`, `static/results`, `reports/generated`  
6. Display the medical disclaimer on every page (already included)

---

## License & ethics

For academic / educational use. Not a certified medical device. Always involve a qualified clinician for diagnosis and treatment decisions.
