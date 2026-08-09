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
| Data | SIIM-ISIC 2020 primary; HAM10000-ready modular loader |

---

## Project structure

```
├── app.py                 # Flask entry point
├── config.py              # Central configuration
├── requirements.txt
├── README.md
├── dataset/               # SIIM-ISIC / HAM10000 data
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

## Dataset setup (SIIM-ISIC 2020)

1. Download **SIIM-ISIC Melanoma Classification 2020** from Kaggle.
2. Place files under `dataset/raw/`:

```
dataset/raw/
├── train.csv
└── train/          # or jpeg/train/ — *.jpg images
    ├── ISIC_xxxx.jpg
    └── ...
```

3. Train:

```bash
python -m training.train --dataset siim_isic_2020
```

### HAM10000 (later)

Place metadata + images under `dataset/ham10000/`, then:

```bash
python -m training.train --dataset ham10000
```

Or set `ACTIVE_DATASET = "ham10000"` in `config.py`.

### Smoke training without Kaggle data

```bash
python -m training.train --synthetic --epochs 5
```

---

## Training guide

The trainer:

1. Loads metadata and stratified train/val/test splits  
2. Applies augmentation + class weights / focal loss  
3. Trains **EfficientNetB0** (transfer → fine-tune)  
4. Trains **MobileNetV2** (transfer → fine-tune)  
5. Evaluates **ensemble** soft voting (`0.6 × Eff + 0.4 × Mob`)  
6. Writes metrics, ROC, confusion matrices, and auto-selects the best model  

Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, TensorBoard, CSV logger. Mixed precision enables automatically when a GPU is present.

Key knobs live in `config.py`: `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `ENSEMBLE_WEIGHTS`, `USE_FOCAL_LOSS`, `USE_TTA`.

**Target metrics** (with full SIIM-ISIC training + tuning): Accuracy / Precision / Recall / F1 &gt; 95%, ROC-AUC &gt; 0.97.

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

1. Train on SIIM-ISIC and confirm metrics under `models/saved/evaluation_metrics.json`  
2. Keep only best weights in `models/saved/`  
3. Set `FLASK_DEBUG=0` and a secure `FLASK_SECRET_KEY`  
4. Serve behind HTTPS reverse proxy (nginx)  
5. Persist `database/melanoma_cdss.db`, `static/uploads`, `static/results`, `reports/generated`  
6. Display the medical disclaimer on every page (already included)

---

## License & ethics

For academic / educational use. Not a certified medical device. Always involve a qualified clinician for diagnosis and treatment decisions.
