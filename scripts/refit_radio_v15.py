"""v15 = v14 + SurvLoss sigmoid RFS expert (SLOT-2 upside, floor = plain v11).

Adds a FIFTH RFS deep expert trained with the sigmoid-concordance loss (SurvLoss)
to the equal-weight z-avg ensemble. Honest OOF: solo 0.7127 ±0.0225 (best & most
stable deep expert vs cox rfs10 0.6933 ±0.0366); 4→5 experts 0.7107→0.7147 (+0.0040
pooled, per-fold var 0.0346→0.0310).

Inherits N (nodalN) + T (gtvpgeom) + RFS clin/deep-znorm + ensemble byte-identical
from v14; only adds rfssig_mu/sd and flips rfs_mode equal4 -> equal5.

Output -> docker/model_v15/task23_radio.joblib
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parents[1]
V14 = ROOT / "docker/model_v14/task23_radio.joblib"
OUT = ROOT / "docker/model_v15/task23_radio.joblib"


def deep_oof_risk(variant):
    rows = []
    for f in range(5):
        p = ROOT / f"predictions/{variant}/fold{f}/predictions.csv"
        if p.exists():
            rows.append(pd.read_csv(p)[["patient_id", "deep_risk"]])
    return pd.concat(rows).set_index("patient_id")["deep_risk"]


def main():
    rfssig = deep_oof_risk("medai_10foldmask_rfs_sigmoid")
    print(f"rfssig OOF: n={len(rfssig)} mean={rfssig.mean():.4f} sd={rfssig.std():.4f}")

    bundle = joblib.load(V14)
    assert bundle["ensemble"]["rfs_mode"] == "equal4", "v14 must be equal4"
    assert "nodal_geom" in bundle["feature_order"] and "gtvp_geom" in bundle["feature_order"], "v14 must carry nodalN+Tgeom"
    bundle["rfs_zn"] = dict(bundle["rfs_zn"])
    bundle["rfs_zn"]["rfssig_mu"] = float(rfssig.mean())
    bundle["rfs_zn"]["rfssig_sd"] = float(rfssig.std())
    bundle["ensemble"] = dict(bundle["ensemble"]); bundle["ensemble"]["rfs_mode"] = "equal5"
    bundle["variant"] = "v15 = v14 + SurvLoss sigmoid 5th RFS expert (equal5)"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT)
    print(f"inherited N/T/RFS/deep-znorm from v14; added rfssig_mu/sd, rfs_mode->equal5")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
