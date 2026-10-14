"""
Read-only design of a clean DICM-only melanoma-vs-benign candidate pool.

Writes only dataset/isic_dicm_17k/CLEAN_DICM_DATASET_DESIGN.txt.
Does not train or modify existing splits, experiments, models, or the app.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config

REPORT = config.DATASET_DIR / "isic_dicm_17k" / "CLEAN_DICM_DATASET_DESIGN.txt"
TRAIN_META = config.ISIC_DICM_LABEL_DIR / "train-set-metadata.csv"
VALID_META = config.ISIC_DICM_LABEL_DIR / "valid-set-metadata.csv"
ISIC2020 = config.RAW_DATA_DIR / "train.csv"
EXISTING = config.PROCESSED_DATA_DIR / "melanoma_benign"

MELANOMA_DX = {"melanoma"}
BENIGN_DX = {
    "nevus",
    "seborrheic keratosis",
    "solar lentigo",
    "acrochordon",
    "lichenoid keratosis",
    "pigmented benign keratosis",
    "lentigo NOS",
    "dermatofibroma",
    "verruca",
    "vascular lesion",
    "angioma",
    "lentigo simplex",
    "neurofibroma",
    "angiokeratoma",
    "sebaceous hyperplasia",
    "angiofibroma or fibrous papule",
    "cafe-au-lait macule",
    "sebaceous adenoma",
    "clear cell acanthoma",
}
EXCLUDE_RULES = {
    "basal cell carcinoma": "Malignant keratinocyte cancer, not melanoma and not benign.",
    "squamous cell carcinoma": "Malignant keratinocyte cancer, not melanoma and not benign.",
    "actinic keratosis": "Premalignant keratinocyte lesion, not confidently benign.",
    "AIMP": "Atypical intraepithelial melanocytic proliferation; not confirmed melanoma or benign.",
    "atypical melanocytic proliferation": "Atypical / uncertain melanocytic lesion; not high-purity melanoma or benign.",
    "atypical spitz tumor": "Ambiguous melanocytic tumor; can be difficult to classify.",
    "melanoma metastasis": "Melanoma biology, but a metastasis rather than a primary cutaneous lesion.",
    "other": "Unspecified diagnosis; cannot assign a clean class.",
    "scar": "May be post-excision or post-treatment; not a confident intact benign lesion.",
    "<MISSING>": "Missing diagnosis. High-purity rule excludes unknown labels.",
}
MALIGNANT_OK = {"malignant", "indeterminate/malignant"}


def normalize_isic_id(value: str) -> str:
    text = str(value or "")
    match = re.search(r"ISIC_(\d+)", text, re.I)
    if match:
        return f"ISIC_{int(match.group(1)):07d}"
    match = re.search(r"ISIC20\d{2}_(\d+)", text, re.I)
    if match:
        return f"ISIC_{int(match.group(1)):07d}"
    match = re.search(r"ISIC(\d{6,})", text, re.I)
    if match:
        return f"ISIC_{int(match.group(1)):07d}"
    return Path(text).stem


def _clean(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _dx_key(value) -> str:
    text = _clean(value)
    return "<MISSING>" if text == "" else text


def _nunique(series: pd.Series) -> int:
    return int(series.map(_clean).replace("", np.nan).nunique())


def _idset(series: pd.Series) -> set[str]:
    return {v for v in series.map(_clean) if v}


def propose_category(dx: str, bm: str) -> tuple[str, str]:
    if dx in MELANOMA_DX:
        if bm in MALIGNANT_OK:
            return "melanoma", "Official diagnosis is melanoma and benign_malignant is malignant or indeterminate/malignant."
        if bm == "":
            return "exclude", "Diagnosis is melanoma but benign_malignant is missing."
        return "exclude", f"Diagnosis is melanoma but benign_malignant={bm}."
    if dx in BENIGN_DX:
        if bm == "benign":
            return "benign", "Clearly benign diagnosis and benign_malignant=benign."
        if bm == "":
            return "exclude", "Benign-appearing diagnosis but benign_malignant is missing."
        return "exclude", f"Benign-appearing diagnosis but benign_malignant={bm}."
    if dx in EXCLUDE_RULES:
        return "exclude", EXCLUDE_RULES[dx]
    return "exclude", "Diagnosis is not in the pre-specified clean melanoma or clean benign lists."


def _group_key(row) -> str:
    if row.patient_id:
        return f"patient:{row.patient_id}"
    if row.lesion_id:
        return f"lesion:{row.lesion_id}"
    return f"image:{row.image_id}"


def simulate_grouped_split(pool: pd.DataFrame, seed: int = 42) -> dict:
    """Keep patient/lesion groups intact. Fill 70/15/15 image targets."""
    rng = np.random.default_rng(seed)
    pool = pool.copy()
    pool["group_id"] = [_group_key(r) for r in pool.itertuples(index=False)]
    groups = []
    for gid, g in pool.groupby("group_id"):
        has_mel = bool((g["proposed"] == "melanoma").any())
        groups.append(
            {
                "group_id": gid,
                "n": int(len(g)),
                "stratum": "melanoma_patient" if has_mel else "benign_only",
            }
        )
    gdf = pd.DataFrame(groups)
    assigned = {}
    for _, sub in gdf.groupby("stratum"):
        order = np.array(sub.index.to_numpy(), copy=True)
        rng.shuffle(order)
        sub = sub.loc[order]
        target = {
            "train": 0.70 * int(sub["n"].sum()),
            "val": 0.15 * int(sub["n"].sum()),
            "test": 0.15 * int(sub["n"].sum()),
        }
        filled = {"train": 0.0, "val": 0.0, "test": 0.0}
        for row in sub.itertuples(index=False):
            remaining = {k: target[k] - filled[k] for k in target}
            split = max(remaining, key=remaining.get)
            assigned[row.group_id] = split
            filled[split] += row.n
    pool["split"] = pool["group_id"].map(assigned)
    out = {}
    for name in ("train", "val", "test"):
        part = pool[pool["split"] == name]
        out[name] = {
            "images": int(len(part)),
            "melanoma": int((part["proposed"] == "melanoma").sum()),
            "benign": int((part["proposed"] == "benign").sum()),
            "patients": _nunique(part["patient_id"]),
            "lesions": _nunique(part["lesion_id"]),
            "melanoma_pct": round(100.0 * (part["proposed"] == "melanoma").mean(), 2) if len(part) else 0.0,
        }
    return out


def main() -> None:
    official = pd.concat(
        [pd.read_csv(TRAIN_META), pd.read_csv(VALID_META)],
        ignore_index=True,
    )
    official["class"] = pd.to_numeric(official["class"], errors="coerce")
    official["image_id"] = official["isic_id"].map(normalize_isic_id)
    official["patient_id"] = official["patient_id"].map(_clean)
    official["lesion_id"] = official["lesion_id"].map(_clean)
    official["diagnosis_key"] = official["diagnosis"].map(_dx_key)
    official["bm"] = official["benign_malignant"].map(_clean).str.lower()
    official["confirm"] = official["diagnosis_confirm_type"].map(_clean)

    cats = []
    reasons = []
    for dx, bm in zip(official["diagnosis_key"], official["bm"]):
        cat, reason = propose_category(dx, bm)
        cats.append(cat)
        reasons.append(reason)
    official["proposed"] = cats
    official["reason"] = reasons

    melanoma = official[official["proposed"] == "melanoma"].copy()
    benign = official[official["proposed"] == "benign"].copy()
    excluded = official[official["proposed"] == "exclude"].copy()
    pool = official[official["proposed"].isin(["melanoma", "benign"])].copy()

    histo_mel = melanoma[melanoma["confirm"].str.contains("histopath", case=False)]
    both_patients = sorted(
        _idset(melanoma["patient_id"]) & _idset(benign["patient_id"])
    )

    # Diagnosis table over unique diagnosis values, using the majority proposed
    # category after the BM rule, plus a note if mixed.
    table_rows = []
    for dx, sub in official.groupby("diagnosis_key", dropna=False):
        vc = sub["proposed"].value_counts()
        if set(vc.index) == {"exclude"}:
            cat = "exclude"
            reason = sub["reason"].iloc[0]
        elif "melanoma" in vc.index and vc.get("exclude", 0) == 0 and "benign" not in vc.index:
            cat = "melanoma"
            reason = sub["reason"].iloc[0]
        elif "benign" in vc.index and vc.get("exclude", 0) == 0 and "melanoma" not in vc.index:
            cat = "benign"
            reason = sub["reason"].iloc[0]
        else:
            cat = "mixed / exclude unless BM rule passes"
            reason = (
                f"Same diagnosis string is not uniform after the benign_malignant filter: "
                f"{vc.to_dict()}. Only rows matching the strict BM rule are kept."
            )
        table_rows.append((dx, int(len(sub)), cat, reason, int(vc.get("melanoma", 0)), int(vc.get("benign", 0)), int(vc.get("exclude", 0))))

    existing = {}
    for name in ("manifest", "train", "val", "test"):
        path = EXISTING / f"{name}.csv"
        df = pd.read_csv(path)
        existing[name] = {
            "images": set(df["image_id"].map(normalize_isic_id)),
            "patients": _idset(df.get("patient_id", pd.Series(dtype=str))),
            "lesions": _idset(df.get("lesion_id", pd.Series(dtype=str))),
            "n": len(df),
        }
    isic = pd.read_csv(ISIC2020)
    isic["image_id"] = isic["image_name"].map(normalize_isic_id)
    isic_benign = isic[isic["benign_malignant"].astype(str).str.strip().str.lower() == "benign"]
    isic_benign = isic_benign[isic_benign["image_id"] != normalize_isic_id("ISIC_4225838")]

    def overlap_block(label: str, images: set[str], patients: set[str], lesions: set[str]) -> list[str]:
        return [
            f"{label}:",
            f"  image IDs: {len(images)}",
            f"  patient IDs: {len(patients)}",
            f"  lesion IDs: {len(lesions)}",
        ]

    clean_img = set(pool["image_id"])
    clean_pat = _idset(pool["patient_id"])
    clean_les = _idset(pool["lesion_id"])
    clean_mel_img = set(melanoma["image_id"])
    clean_ben_img = set(benign["image_id"])

    split = simulate_grouped_split(pool, seed=42)

    def source_counts(df: pd.DataFrame, col: str) -> list[str]:
        if col not in df.columns:
            return ["    <column missing>"]
        vc = df[col].map(_dx_key).value_counts()
        return [f"    {k}: {int(v)}" for k, v in vc.items()]

    archive = pd.read_csv(config.BASE_DIR / "metadata.csv")
    archive["image_id"] = archive["isic_id"].map(normalize_isic_id)
    pool_src = pool.merge(archive[["image_id", "attribution", "image_manipulation"]], on="image_id", how="left")

    ratio = len(benign) / len(melanoma) if len(melanoma) else float("inf")
    go = (
        len(melanoma) >= 1500
        and len(benign) >= 1500
        and _nunique(pool["patient_id"]) >= 400
        and split["test"]["melanoma"] >= 200
        and split["test"]["benign"] >= 200
    )

    lines = [
        "CLEAN DICM-ONLY MELANOMA-VS-BENIGN DATASET DESIGN",
        "Analysis/design only. No training. No existing files were modified except this new report.",
        "",
        "Existing frozen melanoma_benign splits, experiments, production models, and the app were not changed.",
        "",
        "============================================================",
        "1. CLEAN MELANOMA",
        "============================================================",
        "",
        "Recommended rule:",
        "  KEEP if official diagnosis == 'melanoma'",
        "  AND benign_malignant in {malignant, indeterminate/malignant}.",
        "",
        "Do NOT use official class=1.",
        "Do NOT include BCC, SCC, AIMP, atypical melanocytic proliferation,",
        "or melanoma metastasis in the melanoma class.",
        "",
        f"Eligible clean melanoma images: {len(melanoma)}",
        f"Unique non-empty patients: {_nunique(melanoma['patient_id'])}",
        f"Melanoma images with missing patient_id: {int((melanoma['patient_id']=='').sum())}",
        f"Unique lesions: {_nunique(melanoma['lesion_id'])}",
        f"Melanoma images with missing lesion_id: {int((melanoma['lesion_id']=='').sum())}",
        f"Histopathology-confirmed subset: {len(histo_mel)} images, {_nunique(histo_mel['patient_id'])} patients, {_nunique(histo_mel['lesion_id'])} lesions",
        "",
        "benign_malignant among kept melanoma:",
        *[f"  {k}: {int(v)}" for k, v in melanoma["bm"].value_counts().items()],
        "",
        "diagnosis_confirm_type among kept melanoma:",
        *[f"  {k or '<EMPTY>'}: {int(v)}" for k, v in melanoma["confirm"].replace("", "<EMPTY>").value_counts().items()],
        "",
        "============================================================",
        "2. CLEAN BENIGN",
        "============================================================",
        "",
        "Recommended rule:",
        "  KEEP if official diagnosis is in the pre-specified benign list below",
        "  AND benign_malignant == 'benign'",
        "  AND diagnosis is not missing.",
        "",
        "Do NOT use official class=0 as benign.",
        "Exclude malignant, indeterminate, missing diagnosis, premalignant AK,",
        "AIMP/AMP, atypical Spitz, unspecified 'other', scar, BCC, and SCC.",
        "",
        "Allowed benign diagnoses:",
        *[f"  - {d}" for d in sorted(BENIGN_DX)],
        "",
        f"Eligible clean benign images: {len(benign)}",
        f"Unique non-empty patients: {_nunique(benign['patient_id'])}",
        f"Benign images with missing patient_id: {int((benign['patient_id']=='').sum())}",
        f"Unique lesions: {_nunique(benign['lesion_id'])}",
        f"Benign images with missing lesion_id: {int((benign['lesion_id']=='').sum())}",
        "",
        "Kept benign diagnosis counts:",
        *[f"  {k}: {int(v)}" for k, v in benign["diagnosis_key"].value_counts().items()],
        "",
        "============================================================",
        "3. DIAGNOSIS TABLE",
        "============================================================",
        "",
        "diagnosis | count | proposed category | kept as melanoma | kept as benign | excluded | reason",
        "",
    ]
    for dx, n, cat, reason, n_m, n_b, n_e in sorted(table_rows, key=lambda r: (-r[1], r[0])):
        lines.append(f"{dx} | {n} | {cat} | {n_m} | {n_b} | {n_e} | {reason}")

    lines.extend(
        [
            "",
            "============================================================",
            "4. OVERLAP WITH THE EXISTING EXPERIMENT",
            "============================================================",
            "",
            "Audit only. Nothing was removed from the existing 16,529-image dataset.",
            "",
            *overlap_block("Proposed clean pool", clean_img, clean_pat, clean_les),
            "",
            f"Clean melanoma image overlap with existing manifest: {len(clean_mel_img & existing['manifest']['images'])}",
            f"Clean benign image overlap with existing manifest: {len(clean_ben_img & existing['manifest']['images'])}",
            f"Clean pool image overlap with existing manifest: {len(clean_img & existing['manifest']['images'])}",
            f"Clean pool image overlap with existing train.csv: {len(clean_img & existing['train']['images'])}",
            f"Clean pool image overlap with existing val.csv: {len(clean_img & existing['val']['images'])}",
            f"Clean pool image overlap with existing test.csv: {len(clean_img & existing['test']['images'])}",
            f"Clean melanoma image overlap with existing test.csv: {len(clean_mel_img & existing['test']['images'])}",
            f"Clean benign image overlap with existing test.csv: {len(clean_ben_img & existing['test']['images'])}",
            "",
            f"Clean pool patient overlap with existing manifest: {len(clean_pat & existing['manifest']['patients'])}",
            f"Clean pool patient overlap with existing train: {len(clean_pat & existing['train']['patients'])}",
            f"Clean pool patient overlap with existing val: {len(clean_pat & existing['val']['patients'])}",
            f"Clean pool patient overlap with existing test: {len(clean_pat & existing['test']['patients'])}",
            "",
            f"Clean pool lesion overlap with existing manifest: {len(clean_les & existing['manifest']['lesions'])}",
            f"Clean pool lesion overlap with existing train: {len(clean_les & existing['train']['lesions'])}",
            f"Clean pool lesion overlap with existing val: {len(clean_les & existing['val']['lesions'])}",
            f"Clean pool lesion overlap with existing test: {len(clean_les & existing['test']['lesions'])}",
            "",
            f"Clean pool image overlap with ISIC 2020 benign-used pool: {len(clean_img & set(isic_benign['image_id']))}",
            f"Clean melanoma image overlap with ISIC 2020 benign-used pool: {len(clean_mel_img & set(isic_benign['image_id']))}",
            f"Clean benign image overlap with ISIC 2020 benign-used pool: {len(clean_ben_img & set(isic_benign['image_id']))}",
            "",
            "Interpretation:",
            "  Clean melanoma images that already sit in the current experiment are the",
            "  DICM class=1 rows whose diagnosis was truly melanoma. They remain in the",
            "  frozen experiment and must not be deleted. A new DICM-only experiment",
            "  would reuse some of those images, so it must be a NEW split/directory,",
            "  not an edit of the frozen melanoma_benign CSVs.",
            "",
            "============================================================",
            "5. PATIENT/LESION-LEVEL SPLIT FEASIBILITY",
            "============================================================",
            "",
            f"Clean pool images: {len(pool)}  melanoma={len(melanoma)}  benign={len(benign)}",
            f"Unique non-empty patients in pool: {_nunique(pool['patient_id'])}",
            f"Images with missing patient_id: {int((pool['patient_id']=='').sum())} (melanoma {int((melanoma['patient_id']=='').sum())}, benign {int((benign['patient_id']=='').sum())})",
            f"Unique lesions in pool: {_nunique(pool['lesion_id'])}",
            f"Patients with both clean melanoma and clean benign images: {len(both_patients)}",
            "",
            "Feasible: yes, but ONLY if missing patient_id rows fall back to lesion_id,",
            "then image_id. Treating blank patient_id as one group would dump most",
            "melanoma images into a single split. Any real patient with both classes",
            "must stay in one split.",
            "Proposed ratio: 70 / 15 / 15, stratified by whether the patient has melanoma.",
            "Seed for this simulation only: 42. No CSV files were written.",
            "",
            "Proposed simulation:",
        ]
    )
    for name in ("train", "val", "test"):
        s = split[name]
        lines.append(
            f"  {name}: images={s['images']} melanoma={s['melanoma']} benign={s['benign']} "
            f"({s['melanoma_pct']}% melanoma) patients={s['patients']} lesions={s['lesions']}"
        )

    lines.extend(
        [
            "",
            "============================================================",
            "6. BALANCE",
            "============================================================",
            "",
            f"Clean melanoma: {len(melanoma)}",
            f"Clean benign: {len(benign)}",
            f"Melanoma:benign image ratio: {len(melanoma) / len(benign):.2f}" if len(benign) else "Melanoma:benign image ratio: n/a",
            f"Benign:melanoma image ratio: {ratio:.2f}",
            "",
            "The clean pool is large enough for a meaningful experiment. Melanoma is the",
            "larger class (6,787 vs 4,586). Do not randomly discard melanoma images yet.",
            "",
            "Options if training is later approved:",
            "  1. Keep all images and use class-weighted loss.",
            "  2. Keep all images and report prevalence-aware metrics (PR-AUC, sensitivity/specificity).",
            "  3. Controlled patient-level sampling of the majority class, only if weighting is insufficient.",
            "",
            "Recommendation: prefer keeping all clean images and using class weighting /",
            "imbalance-aware metrics. Do not downsample first.",
            "",
            "============================================================",
            "7. SOURCE-CONFOUNDING CHECK",
            "============================================================",
            "",
            "This design removes the main previous confound: melanoma from DICM vs",
            "benign from ISIC 2020. Both classes would come from official DICM metadata",
            "and the same DICM image collection.",
            "",
            "Source confounding is NOT completely eliminated. Remaining within-DICM risks:",
            "",
            "year among clean melanoma:",
            *source_counts(melanoma, "year"),
            "",
            "year among clean benign:",
            *source_counts(benign, "year"),
            "",
            "dermoscopic_type among clean melanoma:",
            *source_counts(melanoma, "dermoscopic_type"),
            "",
            "dermoscopic_type among clean benign:",
            *source_counts(benign, "dermoscopic_type"),
            "",
            "anatom_site_general among clean melanoma:",
            *source_counts(melanoma, "anatom_site_general"),
            "",
            "anatom_site_general among clean benign:",
            *source_counts(benign, "anatom_site_general"),
            "",
            "attribution among clean melanoma:",
            *source_counts(pool_src[pool_src["proposed"] == "melanoma"], "attribution"),
            "",
            "attribution among clean benign:",
            *source_counts(pool_src[pool_src["proposed"] == "benign"], "attribution"),
            "",
            "image_manipulation among clean pool:",
            *source_counts(pool_src, "image_manipulation"),
            "",
            "If year, hospital, or dermoscopic modality still separates classes, a model",
            "can use those shortcuts. A later experiment should report these tables and,",
            "if needed, stratify or audit errors by contributor/year.",
            "",
            "============================================================",
            "8. LABEL QUALITY",
            "============================================================",
            "",
            f"Confirmed melanoma (proposed keep): {len(melanoma)}",
            f"Confirmed benign (proposed keep): {len(benign)}",
            f"Excluded from both classes: {len(excluded)}",
            "",
            "Uncertain / excluded groups include: missing diagnosis, AIMP, atypical",
            "melanocytic proliferation, atypical Spitz, AK, BCC, SCC, metastasis,",
            "scar, other, and any allowed diagnosis whose benign_malignant is missing",
            "or not exactly 'benign' / malignant as required.",
            "",
            "Purity was preferred over size. Official class was not used as the label.",
            "",
            "============================================================",
            "9. FINAL RECOMMENDATION",
            "============================================================",
            "",
            "A. Recommended clean melanoma definition:",
            "   official diagnosis == melanoma AND benign_malignant in",
            "   {malignant, indeterminate/malignant}. Exclude metastasis and all other diagnoses.",
            "",
            "B. Recommended clean benign definition:",
            "   official diagnosis in the listed benign diagnoses AND",
            "   benign_malignant == benign. Exclude missing, indeterminate, malignant,",
            "   premalignant, and ambiguous diagnoses.",
            "",
            f"C. Number of usable melanoma images: {len(melanoma)}",
            f"D. Number of usable benign images: {len(benign)}",
            f"E. Number of unique patients: {_nunique(pool['patient_id'])}",
            f"F. Number of unique lesions: {_nunique(pool['lesion_id'])}",
            "G. Potential overlap with existing experiment:",
            f"   images {len(clean_img & existing['manifest']['images'])} / {len(clean_img)} of the clean pool",
            f"   already appear in the 16,529-image melanoma_benign manifest;",
            f"   {len(clean_img & existing['test']['images'])} overlap the frozen test set.",
            "   Do not edit that experiment. If this design is later approved, create a",
            "   new processed directory and new splits.",
            "H. Proposed split: patient-grouped 70/15/15, seed 42 simulation only.",
            *[
                f"   {name}: n={split[name]['images']} mel={split[name]['melanoma']} ben={split[name]['benign']} "
                f"patients={split[name]['patients']} lesions={split[name]['lesions']}"
                for name in ("train", "val", "test")
            ],
            "I. Remaining risks:",
            "   - residual DICM contributor/year/site/modality imbalance",
            "   - melanoma is the majority class",
            "   - 4,858 clean melanoma images have no patient_id; grouping must fall back",
            "     to lesion_id then image_id or leakage control is incomplete",
            "   - 3 official class=0 rows diagnosed melanoma were excluded if BM missing;",
            "     kept melanoma still depends on the official diagnosis field quality",
            "   - overlap with the previous experiment if results are compared casually",
            f"J. GO / NO-GO recommendation for training: {'GO' if go else 'NO-GO'}",
            "   GO means: the metadata supports a later, separate training experiment.",
            "   It does not authorize training now. Wait for explicit approval.",
            "   No train.csv/val.csv/test.csv were created.",
            "",
            "No model was trained. Production and the live app were not modified.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
