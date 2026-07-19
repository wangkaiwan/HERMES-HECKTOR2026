"""Gamble on T-stage (our best task, so a STRICT gate): T-staging is defined by
PRIMARY tumor diameter, which radio30 encodes only as gtvp_vol_ml (not max diameter).
Add GTVp geometry (largest_maxdim, vol, n_ccs) to the radio-T model and evaluate the
DEPLOYED FUSED T (0.5*deep + 0.5*radio, reorder, argmax over T1..T4) on honest OOF.

ADOPT only if fused-T balacc CLEARLY & STABLY beats the current fused baseline —
otherwise leave T (rank #1 on the board) untouched."""
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
T_MAP = {"T0":0,"T1":1,"T2":2,"T3":3,"T4":4}
REORDER = [4,0,1,2,3]                     # deep t_softmax model-order -> [T0..T4]
MASK_DIRS = [ROOT/f"evaluation/results/qa_10fold_oof_fold{f}_masks" for f in range(5)]
GTVP_MIN_ML = 1.0                          # 1000 mm^3 GTVp CC filter (deploy)
def clean(a): return np.nan_to_num(np.asarray(a,float),nan=0,posinf=0,neginf=0)

def gtvp_geom(mf, struct):
    ps=sitk.ReadImage(str(mf)); sx,sy,sz=ps.GetSpacing(); vv=sx*sy*sz
    arr=sitk.GetArrayFromImage(ps).astype(np.uint8); m=(arr==1)
    if not m.any(): return dict(p_n_ccs=0,p_vol_ml=0.0,p_maxdim=0.0,p_logvol=0.0)
    lab,n=cc_label(m,structure=struct); sizes=np.bincount(lab.ravel(),minlength=n+1)
    vols=sizes[1:]*vv/1000.0; keep=vols>=GTVP_MIN_ML; ids=np.arange(1,n+1)[keep]; vols=vols[keep]
    if vols.size==0: return dict(p_n_ccs=0,p_vol_ml=0.0,p_maxdim=0.0,p_logvol=0.0)
    big=ids[int(np.argmax(vols))]; c=np.argwhere(lab==big)
    ext=(c.max(0)-c.min(0)+1)*np.array([sz,sy,sx])
    tot=float(vols.sum())
    return dict(p_n_ccs=float(vols.size),p_vol_ml=tot,p_maxdim=float(ext.max()),p_logvol=float(np.log1p(tot)))

def load_deepT(v):
    fr=[pd.read_csv(ROOT/f"predictions/{v}/fold{f}/predictions.csv") for f in range(5)]
    d=pd.concat(fr).set_index("patient_id")
    return d[[f"t_softmax_{i}" for i in range(5)]]

def main():
    struct=generate_binary_structure(3,3)
    df=pd.read_csv(CSV_PATH); df["PatientID"]=df["PatientID"].astype(str).str.strip(); df=df.set_index("PatientID")
    clin=pd.DataFrame({p:_encode_clinical_row(r) for p,r in df.iterrows()}).T.drop(columns=REDUNDANT)
    rd=torch.load(ROOT/"data/radiomics_predicted_features.pt",weights_only=False)
    radio=pd.DataFrame.from_dict({p:t.numpy() for p,t in rd["features"].items()},orient="index",columns=list(rd["names"]))
    rows={}
    for d in MASK_DIRS:
        for mf in sorted(Path(d).glob("fold*_*.mha")): rows[mf.stem.split("_",1)[1]]=gtvp_geom(mf,struct)
    geom=pd.DataFrame.from_dict(rows,orient="index")
    deepT=load_deepT("medai_10foldmask_triplehead_aug")
    folds=[pd.read_csv(MANIFEST_DIR/f"val_fold{f}.csv")["patient_id"].astype(str).str.strip().tolist() for f in range(5)]

    def fused_eval(featfn):
        per=[]; yt=[]; yp=[]
        for f in range(5):
            vp=[p for p in folds[f] if p in df.index and pd.notna(df.loc[p,"T-stage"]) and p in geom.index and p in deepT.index]
            tp=[p for p in df.index if p not in folds[f] and pd.notna(df.loc[p,"T-stage"]) and p in geom.index]
            Xtr=featfn(tp); Xv=featfn(vp)
            sc=StandardScaler().fit(Xtr); a=clean(sc.transform(Xtr)); b=clean(sc.transform(Xv))
            ytr=df.loc[tp,"T-stage"].map(T_MAP).values; yv=df.loc[vp,"T-stage"].map(T_MAP).values
            clf=LogisticRegression(max_iter=2000,class_weight="balanced",solver="lbfgs",C=1.0).fit(a,ytr)
            t_radio=clf.predict_proba(b)                              # cols = sorted unique ytr
            # map proba cols to canonical [T0..T4]
            pr=np.zeros((len(vp),5));
            for j,cls in enumerate(clf.classes_): pr[:,cls]=t_radio[:,j]
            t_deep=deepT.reindex(vp).values[:,REORDER]                # -> [T0..T4]
            t_ens=0.5*t_deep+0.5*pr
            pred=np.array([1+int(np.argmax(row[1:])) for row in t_ens])   # argmax over T1..T4
            per.append(balanced_accuracy_score(yv,pred)); yt.append(yv); yp.append(pred)
        return balanced_accuracy_score(np.concatenate(yt),np.concatenate(yp)),float(np.mean(per)),float(np.std(per))

    base=lambda ids: pd.concat([clin.loc[ids],radio.reindex(ids).fillna(0)],axis=1)
    withg=lambda ids: pd.concat([clin.loc[ids],radio.reindex(ids).fillna(0),geom.reindex(ids).fillna(0)],axis=1)
    gonly=lambda ids: pd.concat([clin.loc[ids],geom.reindex(ids).fillna(0)],axis=1)
    print("GTVp geom by GT T-stage (maxdim mm should rise with T):")
    print(geom.join(df["T-stage"]).groupby("T-stage")[["p_vol_ml","p_maxdim"]].mean().round(1).to_string())
    print(f"\n{'fused T (0.5 deep + 0.5 radio-variant)':<42}{'pooled':>8}{'perfold':>16}")
    for name,fn in [("clin+radio (BASELINE deploy)",base),("clin+radio+gtvpgeom",withg),("clin+gtvpgeom (drop radio30)",gonly)]:
        pc,mu,sd=fused_eval(fn); print(f"{name:<42}{pc:>8.4f}   {mu:.4f}±{sd:.4f}")
    print("\nADOPT only if a variant CLEARLY & STABLY beats the BASELINE fused-T.")

if __name__=="__main__": main()
