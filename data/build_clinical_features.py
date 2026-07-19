"""
Precompute ClinicalBERT [CLS] embeddings (768-d) per HECKTOR 2026 patient.

CSV schema (verified 2026-05-09 against the actual release):
    PatientID, CenterID, Age, Gender, Tobacco Consumption, Alcohol Consumption,
    Performance Status, Treatment, HPV Status, Relapse, RFS, T-stage, N-stage

All non-string columns are float64 with NaN for missing. Many fields have
substantial missingness (PerformanceStatus 40%, Alcohol 34%, Tobacco 33%,
HPV 24%, Relapse/RFS ~7%).

Outputs:
    data/clinical_features.pt   {PatientID: tensor[768]}  (ClinicalBERT [CLS])
    data/clinical_tabular.pt    {PatientID: tensor[16]}   (encoded for Cox head)

Text template (2026-05-13, task #37: T/N stage REMOVED — they leak the staging
GT through FiLM, which would inflate Phase 3 TN-staging numbers):
    "<age>-year-old <gender> head and neck cancer patient.
     HPV <pos|neg|unknown>.
     <smoker|non-smoker>. <drinks alcohol|no alcohol>.
     ECOG performance status <ps>."

Usage:
    python data/build_clinical_features.py \\
        --csv data/raw/hecktor2026_training/HECKTOR_2026_training_data.csv \\
        --output data/clinical_features.pt \\
        --tabular_output data/clinical_tabular.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# ── Coding helpers ───────────────────────────────────────────────────────────

def _gender_str(g) -> str:
    """1.0 = male, 2.0 = female (HECKTOR convention)."""
    if pd.isna(g):
        return ""
    g = float(g)
    if g == 1.0:
        return "male"
    if g == 2.0:
        return "female"
    return ""


def _binary_yesno(v) -> str | None:
    """0.0 = no, 1.0 = yes, NaN = unknown. Returns 'yes' / 'no' / None."""
    if pd.isna(v):
        return None
    v = float(v)
    if v == 1.0:
        return "yes"
    if v == 0.0:
        return "no"
    return None


def _hpv_str(v) -> str:
    """1.0 = positive, 0.0 = negative, NaN = unknown."""
    if pd.isna(v):
        return "unknown"
    v = float(v)
    if v == 1.0:
        return "positive"
    if v == 0.0:
        return "negative"
    return "unknown"


def _stage_str(v, prefix: str) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip().upper().replace(" ", "")
    if not s:
        return None
    if not s.startswith(prefix):
        s = prefix + s
    return s


# ── Text formatting ──────────────────────────────────────────────────────────

def build_text(row: pd.Series) -> str:
    parts: List[str] = []

    age = row.get("Age")
    gender = _gender_str(row.get("Gender"))
    if not pd.isna(age) and gender:
        parts.append(f"{int(round(float(age)))}-year-old {gender} head and neck cancer patient.")
    elif gender:
        parts.append(f"{gender} head and neck cancer patient.")
    elif not pd.isna(age):
        parts.append(f"{int(round(float(age)))}-year-old head and neck cancer patient.")
    else:
        parts.append("Head and neck cancer patient.")

    # Task #37 (2026-05-13): T/N stage REMOVED from the text template — they
    # leak the staging GT through FiLM into the seg encoder, and Phase 3 has
    # TN-staging as a prediction task. _stage_str() is kept above for any
    # downstream caller that wants formatted strings but is not used here.
    parts.append(f"HPV {_hpv_str(row.get('HPV Status'))}.")

    smoker = _binary_yesno(row.get("Tobacco Consumption"))
    if smoker == "yes":
        parts.append("Smoker.")
    elif smoker == "no":
        parts.append("Non-smoker.")

    drinker = _binary_yesno(row.get("Alcohol Consumption"))
    if drinker == "yes":
        parts.append("Drinks alcohol.")
    elif drinker == "no":
        parts.append("No alcohol.")

    ps = row.get("Performance Status")
    if not pd.isna(ps):
        try:
            parts.append(f"ECOG performance status {int(round(float(ps)))}.")
        except (TypeError, ValueError):
            pass

    return " ".join(parts)


# ── Tabular encoding for Cox-PH head (18-d, matches n_clinical=18 in cfg) ────
#
# Layout (verified 2026-05-10 against the actual release):
#   [0]      age z-score (age - 60) / 12         continuous; never missing in this cohort
#   [1]      gender                              1.0 = male, 0.0 = female; never missing
#   [2:5]    HPV one-hot                         pos / neg / unknown            (24% unknown)
#   [5:8]    Tobacco one-hot                     yes / no / missing             (33% missing)
#   [8:11]   Alcohol one-hot                     yes / no / missing             (34% missing)
#   [11:15]  Performance Status one-hot          PS=0 / PS=1 / PS≥2 / missing   (40% missing)
#   [15:18]  Treatment one-hot                   T=0 / T=1 / missing            (2.4% missing)
#
# CenterID is intentionally NOT included — multi-centric generalisation is the
# point of HECKTOR; if the test set has unseen centers, an all-zero one-hot
# input would silently mislead the model. Center info enters via the splits
# (stratified by patient prefix) and the imaging itself.

def encode_tabular(row: pd.Series) -> torch.Tensor:
    # Age (always present; z-score around HNC-typical mean=60, sd=12)
    age = row.get("Age")
    age_z = (float(age) - 60.0) / 12.0 if pd.notna(age) else 0.0

    # Gender (always present in this cohort; 1.0=male, 2.0=female by HECKTOR convention)
    g = row.get("Gender")
    gender_male = 1.0 if pd.notna(g) and float(g) == 1.0 else 0.0

    # HPV — 3-d one-hot
    hpv = _hpv_str(row.get("HPV Status"))
    hpv_pos = float(hpv == "positive")
    hpv_neg = float(hpv == "negative")
    hpv_unknown = float(hpv == "unknown")

    # Tobacco — 3-d one-hot
    smoker = _binary_yesno(row.get("Tobacco Consumption"))
    smoker_yes = float(smoker == "yes")
    smoker_no = float(smoker == "no")
    smoker_missing = float(smoker is None)

    # Alcohol — 3-d one-hot
    drinker = _binary_yesno(row.get("Alcohol Consumption"))
    drinker_yes = float(drinker == "yes")
    drinker_no = float(drinker == "no")
    drinker_missing = float(drinker is None)

    # Performance Status — 4-d, collapsing rare PS≥2 (n=34 across PS=2,3,4)
    ps = row.get("Performance Status")
    if pd.isna(ps):
        ps_0 = ps_1 = ps_high = 0.0
        ps_missing = 1.0
    else:
        v = float(ps)
        ps_0 = float(v == 0.0)
        ps_1 = float(v == 1.0)
        ps_high = float(v >= 2.0)
        ps_missing = 0.0

    # Treatment — 3-d one-hot (binary 0/1 + missing)
    tx = row.get("Treatment")
    if pd.isna(tx):
        tx_0 = tx_1 = 0.0
        tx_missing = 1.0
    else:
        v = float(tx)
        tx_0 = float(v == 0.0)
        tx_1 = float(v == 1.0)
        tx_missing = 0.0

    return torch.tensor([
        age_z, gender_male,
        hpv_pos, hpv_neg, hpv_unknown,
        smoker_yes, smoker_no, smoker_missing,
        drinker_yes, drinker_no, drinker_missing,
        ps_0, ps_1, ps_high, ps_missing,
        tx_0, tx_1, tx_missing,
    ], dtype=torch.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tabular_output", type=Path, default=None)
    p.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()

    df = pd.read_csv(args.csv)
    print(f"Read {len(df)} patients from {args.csv}")
    if "PatientID" not in df.columns:
        raise ValueError(f"CSV missing PatientID column: {df.columns.tolist()}")

    pairs = [(str(row["PatientID"]).strip(), build_text(row)) for _, row in df.iterrows()]
    print("first 3 sample texts:")
    for pid, txt in pairs[:3]:
        print(f"  {pid}: {txt}")

    features: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i:i + args.batch_size]
            ids = [b[0] for b in batch]
            texts = [b[1] for b in batch]
            enc = tokenizer(texts, padding=True, truncation=True, max_length=128,
                            return_tensors="pt").to(args.device)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu()
            for pid, emb in zip(ids, cls):
                features[pid] = emb
            print(f"  [{i + len(batch)}/{len(pairs)}]", end="\r")

    print(f"\nSaving {len(features)} embeddings → {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, args.output)
    print(f"Embedding dim: {next(iter(features.values())).shape[0]}")

    if args.tabular_output:
        tabular = {str(row["PatientID"]).strip(): encode_tabular(row) for _, row in df.iterrows()}
        args.tabular_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tabular, args.tabular_output)
        print(f"Saved tabular features ({next(iter(tabular.values())).shape[0]}-d) → {args.tabular_output}")


if __name__ == "__main__":
    main()
