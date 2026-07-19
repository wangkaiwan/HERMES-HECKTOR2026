"""
Build 5-fold cross-validation splits for HECKTOR 2026.

Strategy (chosen 2026-05-09 after profiling the actual release):
    - 782 patients across 8 centers; MDA dominates (444), USZ smallest (11)
    - StratifiedKFold (not GroupKFold) on a composite stratum key
        stratum = "<T-stage>|<N-stage>|<relapse>"
      Patients with NaN T-stage / N-stage / relapse fall into "unk" buckets.
    - This spreads the large MDA cohort across all folds (we want that —
      domain shift between MDA and small centers is real, but holding all
      MDA out of one fold would crater training).
    - Reproducible: random_state=42.

Output (one row per patient):
    data/manifests/splits.csv   columns: patient_id, center, fold (0..4)
    plus per-fold train/val manifests:
        data/manifests/train_fold{i}.csv
        data/manifests/val_fold{i}.csv

The per-fold manifest carries every column the dataset/loaders need
(patient_id, center, has_pet, has_seg, has_staging, has_survival, plus the
raw clinical fields). Downstream code only reads from these manifests.

Usage:
    python data/splits.py \\
        --csv data/raw/hecktor2026_training/HECKTOR_2026_training_data.csv \\
        --raw_dir data/raw/hecktor2026_training \\
        --manifest_dir data/manifests \\
        --n_folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


CSV_COLUMNS = {
    "patient": "PatientID",
    "center": "CenterID",
    "age": "Age",
    "gender": "Gender",
    "tobacco": "Tobacco Consumption",
    "alcohol": "Alcohol Consumption",
    "perf": "Performance Status",
    "treatment": "Treatment",
    "hpv": "HPV Status",
    "relapse": "Relapse",
    "rfs": "RFS",
    "t_stage": "T-stage",
    "n_stage": "N-stage",
}


def _stratum_key(row: pd.Series) -> str:
    """Composite stratification key — T, N, relapse."""
    t = str(row[CSV_COLUMNS["t_stage"]]) if pd.notna(row[CSV_COLUMNS["t_stage"]]) else "unk"
    n = str(row[CSV_COLUMNS["n_stage"]]) if pd.notna(row[CSV_COLUMNS["n_stage"]]) else "unk"
    r = str(int(row[CSV_COLUMNS["relapse"]])) if pd.notna(row[CSV_COLUMNS["relapse"]]) else "unk"
    return f"{t}|{n}|{r}"


def _check_files(raw_dir: Path, patient_id: str) -> dict:
    pdir = raw_dir / patient_id
    ct = pdir / f"{patient_id}__CT.nii.gz"
    pt = pdir / f"{patient_id}__PT.nii.gz"
    lb = pdir / f"{patient_id}.nii.gz"
    return {
        "ct_path": str(ct) if ct.exists() else "",
        "pt_path": str(pt) if pt.exists() else "",
        "label_path": str(lb) if lb.exists() else "",
        "has_ct": ct.exists(),
        "has_pet": pt.exists(),
        "has_seg": lb.exists(),
    }


def build_manifest(csv_path: Path, raw_dir: Path) -> pd.DataFrame:
    """Read the clinical CSV, attach file paths, derive task availability flags."""
    df = pd.read_csv(csv_path)
    df["patient_id"] = df[CSV_COLUMNS["patient"]].astype(str)
    df["center"] = df["patient_id"].str.split("-").str[0]
    df["center_id"] = df[CSV_COLUMNS["center"]].astype("Int64")

    # File presence
    file_info = df["patient_id"].apply(lambda p: _check_files(raw_dir, p)).apply(pd.Series)
    df = pd.concat([df, file_info], axis=1)

    # Task availability flags — has_seg (mask present), has_staging (T+N both),
    # has_survival (relapse + rfs both present). Kept boolean so the trainer
    # can mask losses when labels are absent.
    df["has_staging"] = (df[CSV_COLUMNS["t_stage"]].notna() &
                         df[CSV_COLUMNS["n_stage"]].notna())
    df["has_survival"] = (df[CSV_COLUMNS["relapse"]].notna() &
                          df[CSV_COLUMNS["rfs"]].notna())
    return df


def assign_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Add a 'fold' column (0..n_folds-1) via StratifiedKFold on a composite stratum.

    Rare strata (count < n_folds) are merged into a single "rare" bucket so
    StratifiedKFold doesn't error on classes with too few samples.
    """
    df = df.copy()
    stratum = df.apply(_stratum_key, axis=1)
    counts = stratum.value_counts()
    rare = set(counts[counts < n_folds].index)
    stratum_clean = stratum.where(~stratum.isin(rare), other="rare")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_col = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, stratum_clean)):
        fold_col[val_idx] = fold_idx
    df["fold"] = fold_col
    df["stratum"] = stratum
    return df


def main(csv_path: Path, raw_dir: Path, manifest_dir: Path,
         n_folds: int = 5, seed: int = 42) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    df = build_manifest(csv_path, raw_dir)

    n_with_ct = df["has_ct"].sum()
    n_with_pet = df["has_pet"].sum()
    n_with_seg = df["has_seg"].sum()
    print(f"manifest: {len(df)} patients — CT {n_with_ct}, PET {n_with_pet}, seg {n_with_seg}")
    print(f"  staging available: {df['has_staging'].sum()}, survival: {df['has_survival'].sum()}")

    df = assign_folds(df, n_folds=n_folds, seed=seed)
    full_path = manifest_dir / "splits.csv"
    df.to_csv(full_path, index=False)
    print(f"wrote {full_path}")

    # Per-fold train/val files
    for fold in range(n_folds):
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        train_df.to_csv(manifest_dir / f"train_fold{fold}.csv", index=False)
        val_df.to_csv(manifest_dir / f"val_fold{fold}.csv", index=False)
        # Per-fold center balance — sanity print
        c_t = train_df["center"].value_counts().to_dict()
        c_v = val_df["center"].value_counts().to_dict()
        print(f"fold {fold}: train={len(train_df)} val={len(val_df)}; "
              f"val centers={c_v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--raw_dir", type=Path, required=True)
    p.add_argument("--manifest_dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.csv, a.raw_dir, a.manifest_dir, a.n_folds, a.seed)
