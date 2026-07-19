"""Probe: does adding CLINICALLY-ALIGNED nodal-geometry features (derived from the
predicted GTVn mask) improve honest 5-fold OOF N-stage balacc over the current
clin+radio L1-LogReg C=0.05 (0.6911)?

N-staging is defined by nodal COUNT and largest-node SIZE — features the current
radio30 set lacks (it has only total gtvn_vol_ml). We add, from the OOF predicted
masks (500 mm^3 GTVn CC filter, matching deploy):
  n_ccs           : number of nodal connected components (N1 vs N2b)
  largest_cc_ml   : largest single node volume  (size criterion)
  largest_frac    : largest / total nodal volume (dominant-node-ness)
  largest_maxdim  : largest node max bbox extent in mm (>6cm -> N3)
  total_ml, log1p variants
Deployable: same computation runs on the docker's predicted mask at inference.

ADOPT only if pooled AND per-fold mean CLEARLY & STABLY beat 0.6911.
"""
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
REDUNDANT = ["hpv_unk","smoker_missing","drinker_missing","ps_missing","treatment_missing"]
N_MAP = {"N0":0,"N1":1,"N2":2,"N3":3}
MASK_DIRS = [ROOT/f"evaluation/results/qa_10fold_oof_fold{f}_masks" for f in range(5)]
GTVN_CC_MM3 = 500.0
def clean(a): return np.nan_to_num(np.asarray(a,float),nan=0,posinf=0,neginf=0)

def nodal_geom(mask_path, struct):
    ps = sitk.ReadImage(str(mask_path)); sx,sy,sz = ps.GetSpacing(); vv = float(sx*sy*sz)
    arr = sitk.GetArrayFromImage(ps).astype(np.uint8)
    m = (arr == 2)
    if not m.any():
        return dict(n_ccs=0,largest_ml=0.0,largest_frac=0.0,largest_maxdim=0.0,total_ml=0.0)
    lab,n = cc_label(m,structure=struct); sizes = np.bincount(lab.ravel(),minlength=n+1)
    vols = sizes[1:]*vv/1000.0                                # ml per CC
    keep = vols >= GTVN_CC_MM3/1000.0
    vols = vols[keep]; keep_ids = np.arange(1,n+1)[keep]
    if vols.size == 0:
        return dict(n_ccs=0,largest_ml=0.0,largest_frac=0.0,largest_maxdim=0.0,total_ml=0.0)
    big = keep_ids[int(np.argmax(vols))]
    coords = np.argwhere(lab == big)
    extent_mm = (coords.max(0)-coords.min(0)+1) * np.array([sz,sy,sx])   # z,y,x spacing
    return dict(n_ccs=int(vols.size), largest_ml=float(vols.max()),
                largest_frac=float(vols.max()/vols.sum()),
                largest_maxdim=float(extent_mm.max()), total_ml=float(vols.sum()))

def main():
    struct = generate_binary_structure(3,3)
    df = pd.read_csv(CSV_PATH); df["PatientID"]=df["PatientID"].astype(str).str.strip(); df=df.set_index("PatientID")
    clin = pd.DataFrame({p:_encode_clinical_row(r) for p,r in df.iterrows()}).T.drop(columns=REDUNDANT)
    fo = torch.load(ROOT/"data/radiomics_predicted_features.pt",weights_only=False)
    radio = pd.DataFrame.from_dict({p:t.numpy() for p,t in fo["features"].items()},orient="index",columns=list(fo["names"]))
    # nodal geometry from OOF masks
    rows = {}
    for d in MASK_DIRS:
        for mf in sorted(Path(d).glob("fold*_*.mha")):
            pid = mf.stem.split("_",1)[1]
            rows[pid] = nodal_geom(mf,struct)
    geom = pd.DataFrame.from_dict(rows,orient="index")
    geom["log_largest"]=np.log1p(geom["largest_ml"]); geom["log_total"]=np.log1p(geom["total_ml"])
    print(f"nodal-geom extracted for {len(geom)} patients; cols={list(geom.columns)}")
    # sanity: mean n_ccs / largest by GT N-stage
    j = geom.join(df["N-stage"])
    print(j.groupby("N-stage")[["n_ccs","largest_ml","largest_maxdim"]].mean().round(2).to_string())

    folds=[pd.read_csv(MANIFEST_DIR/f"val_fold{f}.csv")["patient_id"].astype(str).str.strip().tolist() for f in range(5)]
    def oof(featfn, C=0.05):
        per=[]; yt=[]; yp=[]
        for f in range(5):
            vp=[p for p in folds[f] if p in df.index and pd.notna(df.loc[p,"N-stage"]) and p in geom.index]
            tp=[p for p in df.index if p not in folds[f] and pd.notna(df.loc[p,"N-stage"]) and p in geom.index]
            Xtr=featfn(tp); Xv=featfn(vp)
            sc=StandardScaler().fit(Xtr); a=clean(sc.transform(Xtr)); b=clean(sc.transform(Xv))
            ytr=df.loc[tp,"N-stage"].map(N_MAP).values; yv=df.loc[vp,"N-stage"].map(N_MAP).values
            pred=LogisticRegression(penalty="l1",solver="saga",C=C,max_iter=5000,class_weight="balanced").fit(a,ytr).predict(b)
            per.append(balanced_accuracy_score(yv,pred)); yt.append(yv); yp.append(pred)
        return balanced_accuracy_score(np.concatenate(yt),np.concatenate(yp)),float(np.mean(per)),float(np.std(per))

    base   = lambda ids: pd.concat([clin.loc[ids],radio.reindex(ids).fillna(0)],axis=1)
    withg  = lambda ids: pd.concat([clin.loc[ids],radio.reindex(ids).fillna(0),geom.reindex(ids).fillna(0)],axis=1)
    geomonly = lambda ids: pd.concat([clin.loc[ids],geom.reindex(ids).fillna(0)],axis=1)
    print(f"\n{'N model (honest 5-fold OOF)':<40}{'pooled':>8}{'perfold':>16}")
    for name,fn in [("clin+radio (BASELINE)",base),("clin+radio+nodalgeom",withg),("clin+nodalgeom (no radio30)",geomonly)]:
        for C in ([0.05] if name!="clin+radio+nodalgeom" else [0.05,0.1,0.03]):
            pc,mu,sd=oof(fn,C); print(f"{name+' C='+str(C):<40}{pc:>8.4f}   {mu:.4f}±{sd:.4f}")
    print("\nADOPT nodalgeom only if CLEARLY & STABLY > baseline (0.6911 was clin+radio C=0.05).")

if __name__=="__main__": main()
