"""Confirm the clin+nodalgeom N-stage win is robust (not 7-feature overfitting):
C-sweep + reduced feature subsets + per-fold breakdown. Reuses the probe's
extraction."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, SimpleITK as sitk
from scipy.ndimage import label as cc_label, generate_binary_structure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from task3_integrator import _encode_clinical_row, CSV_PATH, MANIFEST_DIR  # noqa
from n_nodal_geom_probe import nodal_geom, REDUNDANT, N_MAP, MASK_DIRS  # noqa
def clean(a): return np.nan_to_num(np.asarray(a,float),nan=0,posinf=0,neginf=0)

def main():
    struct = generate_binary_structure(3,3)
    df = pd.read_csv(CSV_PATH); df["PatientID"]=df["PatientID"].astype(str).str.strip(); df=df.set_index("PatientID")
    clin = pd.DataFrame({p:_encode_clinical_row(r) for p,r in df.iterrows()}).T.drop(columns=REDUNDANT)
    rows = {}
    for d in MASK_DIRS:
        for mf in sorted(Path(d).glob("fold*_*.mha")):
            rows[mf.stem.split("_",1)[1]] = nodal_geom(mf,struct)
    geom = pd.DataFrame.from_dict(rows,orient="index")
    geom["log_largest"]=np.log1p(geom["largest_ml"]); geom["log_total"]=np.log1p(geom["total_ml"])
    folds=[pd.read_csv(MANIFEST_DIR/f"val_fold{f}.csv")["patient_id"].astype(str).str.strip().tolist() for f in range(5)]

    def oof(cols, C):
        per=[]; yt=[]; yp=[]
        G = geom[cols]
        for f in range(5):
            vp=[p for p in folds[f] if p in df.index and pd.notna(df.loc[p,"N-stage"]) and p in geom.index]
            tp=[p for p in df.index if p not in folds[f] and pd.notna(df.loc[p,"N-stage"]) and p in geom.index]
            Xtr=pd.concat([clin.loc[tp],G.reindex(tp).fillna(0)],axis=1); Xv=pd.concat([clin.loc[vp],G.reindex(vp).fillna(0)],axis=1)
            sc=StandardScaler().fit(Xtr); a=clean(sc.transform(Xtr)); b=clean(sc.transform(Xv))
            ytr=df.loc[tp,"N-stage"].map(N_MAP).values; yv=df.loc[vp,"N-stage"].map(N_MAP).values
            pred=LogisticRegression(penalty="l1",solver="saga",C=C,max_iter=5000,class_weight="balanced").fit(a,ytr).predict(b)
            per.append(balanced_accuracy_score(yv,pred)); yt.append(yv); yp.append(pred)
        return balanced_accuracy_score(np.concatenate(yt),np.concatenate(yp)),per

    ALL=['n_ccs','largest_ml','largest_frac','largest_maxdim','total_ml','log_largest','log_total']
    CORE=['n_ccs','largest_maxdim','largest_frac']            # 3 staging-aligned only
    MIN=['n_ccs','largest_maxdim']                            # count + size only
    print(f"{'clin + geom subset':<34}{'C':>5}{'pooled':>9}   per-fold")
    for name,cols in [("ALL geom(7)",ALL),("CORE(n_ccs,maxdim,frac)",CORE),("MIN(n_ccs,maxdim)",MIN)]:
        for C in [0.03,0.05,0.1]:
            pc,per=oof(cols,C)
            print(f"{name:<34}{C:>5}{pc:>9.4f}   [{' '.join(f'{x:.3f}' for x in per)}] mu={np.mean(per):.4f}±{np.std(per):.4f}")
    print("\nBASELINE clin+radio C=0.05 = 0.6911 (0.6893±0.0523). Robust if all subsets stay >~0.71.")

if __name__=="__main__": main()
