"""v14 = v13 + T-geometry (SLOT-2 upside, floor-protected by plain v11 in slot 1).

Adds PRIMARY-tumor geometry (GTVp largest diameter + volume) to the radio-T model —
the T-staging size criterion that radio30 encoded only as gtvp_vol_ml. KEEP-RADIO
(additive, conservative): T = LogReg on clin + radio30 + gtvp_geom, fused 0.5*deep.
Honest OOF fused-T balacc 0.4439 -> 0.4535 (+0.0096 pooled, per-fold ±0.037->±0.034).

Inherits N (nodalN) + RFS (clin) + deep z-norm + ensemble byte-identical from v13.
Only the `t` model + feature_order["gtvp_geom"] change.

Output -> docker/model_v14/task23_radio.joblib
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.ndimage import generate_binary_structure
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from outer5_radiomics_baselines import load_data  # noqa
from eval_t_geom_oof import gtvp_geom, MASK_DIRS  # SAME extraction as the T probe
GTVP_GEOM_ORDER = ["p_n_ccs", "p_vol_ml", "p_maxdim", "p_logvol"]
T_MAP = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
V13 = ROOT / "docker/model_v11clinN/task23_radio.joblib"
OUT = ROOT / "docker/model_v14/task23_radio.joblib"


def main():
    df, clin, radio, tex = load_data()
    struct = generate_binary_structure(3, 3)
    rows = {}
    for d in MASK_DIRS:
        for mf in sorted(Path(d).glob("fold*_*.mha")):
            rows[mf.stem.split("_", 1)[1]] = gtvp_geom(mf, struct)
    geom = pd.DataFrame.from_dict(rows, orient="index")[GTVP_GEOM_ORDER]
    print(f"gtvp-geom for {len(geom)} patients; order={GTVP_GEOM_ORDER}")

    # ---- T model: clin + radio + gtvp_geom, plain LogReg C=1.0 (same recipe + geom) ----
    tpids = [p for p in clin.index if pd.notna(df.loc[p, "T-stage"]) and p in geom.index]
    Xt = pd.concat([clin.loc[tpids], radio.reindex(tpids).fillna(0), geom.reindex(tpids).fillna(0)], axis=1)
    yt = df.loc[tpids, "T-stage"].map(T_MAP).values
    sct = StandardScaler().fit(Xt.values)
    tclf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", C=1.0)\
        .fit(sct.transform(Xt.values), yt)
    print(f"T refit clin+radio+gtvpgeom: n={len(tpids)}, {Xt.shape[1]} feats, classes={list(tclf.classes_)}")

    # ---- inherit EVERYTHING else from v13 (N nodalN, RFS clin, deep z-norm, ensemble) ----
    bundle = joblib.load(V13)
    assert "nodal_geom" in bundle["feature_order"], "v13 must carry nodalN"
    bundle["t"] = {"scaler": sct, "clf": tclf}
    bundle["feature_order"]["gtvp_geom"] = list(GTVP_GEOM_ORDER)   # signals docker to append gtvp-geom to the T frame
    bundle["variant"] = "v14 = v13 (nodalN + clin-RFS) + T-geometry (radio-T + gtvp geom)"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT)
    print("inherited N/RFS/deep-znorm/ensemble from v13; swapped T -> clin+radio+gtvpgeom")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
