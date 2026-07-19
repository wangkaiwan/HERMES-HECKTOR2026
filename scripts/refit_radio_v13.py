"""v13 (v11-clin + nodalN) full-data refit — SLOT-2 upside bet.

Two robust, clinically-principled changes vs v11, both honest-OOF validated:
  RFS: radio(clin+radiomics LASSO) -> pure-clinical ridge Cox  (v12 swap; OOF 0.7107 ~ tie, +robust)
  N  : clin+radio30 LogReg -> clin + NODAL-GEOMETRY LogReg C=0.03  (OOF 0.6911 -> 0.7200, +0.029, stable)

Nodal geometry (from the predicted GTVn mask, the same `seg` docker N-staging sees):
  n_ccs, largest_ml, largest_frac, largest_maxdim, total_ml, log_largest, log_total.
These encode the N-staging DEFINITION (node count + largest-node size) that radio30 lacked.

Inherits everything else byte-identical from the v12 joblib (clin RFS, T, deep z-norm,
ensemble). Only the `n` model + feature_order gain a nodal_geom block.

Output -> docker/model_v11clinN/task23_radio.joblib
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch
from scipy.ndimage import generate_binary_structure
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from outer5_radiomics_baselines import load_data  # noqa
from n_nodal_geom_probe import nodal_geom, MASK_DIRS  # SAME extraction as the OOF probe
GEOM_ORDER = ["n_ccs", "largest_ml", "largest_frac", "largest_maxdim", "total_ml", "log_largest", "log_total"]
N_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
V12 = ROOT / "docker/model_v11clin/task23_radio.joblib"
OUT = ROOT / "docker/model_v11clinN/task23_radio.joblib"


def geom_frame(pids, geom):
    G = geom.reindex(pids)
    return G[GEOM_ORDER].fillna(0.0)


def main():
    df, clin, radio, tex = load_data()
    # ---- extract nodal geometry from the 10-fold OOF predicted masks ----
    struct = generate_binary_structure(3, 3)
    rows = {}
    for d in MASK_DIRS:
        for mf in sorted(Path(d).glob("fold*_*.mha")):
            rows[mf.stem.split("_", 1)[1]] = nodal_geom(mf, struct)
    geom = pd.DataFrame.from_dict(rows, orient="index")
    geom["log_largest"] = np.log1p(geom["largest_ml"]); geom["log_total"] = np.log1p(geom["total_ml"])
    print(f"nodal-geom for {len(geom)} patients; order={GEOM_ORDER}")

    # ---- N model: clin + nodal-geom, L1-LogReg C=0.03 (robust OOF 0.7200) ----
    npids = [p for p in clin.index if pd.notna(df.loc[p, "N-stage"]) and p in geom.index]
    Xn = pd.concat([clin.loc[npids], geom_frame(npids, geom)], axis=1)
    yn = df.loc[npids, "N-stage"].map(N_MAP).values
    scn = StandardScaler().fit(Xn.values)
    nclf = LogisticRegression(penalty="l1", solver="saga", C=0.03, max_iter=5000,
                              class_weight="balanced").fit(scn.transform(Xn.values), yn)
    nnz = int((np.abs(nclf.coef_).sum(0) > 1e-8).sum())
    print(f"N refit clin+nodalgeom L1 C=0.03: n={len(npids)}, {Xn.shape[1]} feats, {nnz} nonzero")

    # ---- inherit EVERYTHING else from v12 (clin RFS, T, deep z-norm, ensemble) ----
    bundle = joblib.load(V12)
    assert list(bundle["feature_order"]["clin"]) == list(clin.columns), "clin order mismatch vs v12"
    bundle["n"] = {"scaler": scn, "clf": nclf}
    bundle["feature_order"]["nodal_geom"] = list(GEOM_ORDER)   # <-- signals docker to build clin+geom N frame
    bundle["variant"] = "v13 = v11-clin (RFS radio->clin) + nodalN (N radio->nodal-geometry)"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT)
    print(f"inherited clin-RFS/T/deep-znorm/ensemble from v12; swapped N -> clin+nodalgeom")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
