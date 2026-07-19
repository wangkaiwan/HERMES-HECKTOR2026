"""Bootstrap CIs + paired tests for the paper's headline deltas (addresses reviewer
stat-rigor points). Patient-level bootstrap over the pooled OOF predictions.
N-stage: balanced accuracy, clin+radio vs clin+nodalgeom (matched C). RFS: pooled
C-index for cox rfs10, sigmoid, 4-expert, 5-expert + paired deltas."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from task3_integrator import _encode_clinical_row, CSV_PATH, MANIFEST_DIR  # noqa
sys.path.insert(0, str(ROOT)); from training.metrics import harrell_cindex  # noqa
from n_nodal_geom_probe import nodal_geom, MASK_DIRS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from scipy.ndimage import generate_binary_structure
from lifelines import CoxPHFitter
RED=["hpv_unk","smoker_missing","drinker_missing","ps_missing","treatment_missing"]
N_MAP={"N0":0,"N1":1,"N2":2,"N3":3}; GEOM=["n_ccs","largest_ml","largest_frac","largest_maxdim","total_ml","log_largest","log_total"]
RNG=np.random.RandomState(0)                                    # fixed seed (no Date/rand ban issue here)
def zn(x): x=np.asarray(x,float); s=x.std(); return (x-x.mean())/(s if s>0 else 1)
def clean(a): return np.nan_to_num(np.asarray(a,float),nan=0,posinf=0,neginf=0)

def boot_ci(fn, n=2000):
    vals=[fn(None)]; N=None
    stats=[]
    idx0=fn.idx
    for _ in range(n):
        b=RNG.choice(len(idx0),len(idx0),replace=True)
        stats.append(fn(b))
    lo,hi=np.percentile(stats,[2.5,97.5]); return vals[0],lo,hi

def main():
    df=pd.read_csv(CSV_PATH); df["PatientID"]=df["PatientID"].astype(str).str.strip(); df=df.set_index("PatientID")
    clin=pd.DataFrame({p:_encode_clinical_row(r) for p,r in df.iterrows()}).T.drop(columns=RED)
    rd=torch.load(ROOT/"data/radiomics_predicted_features.pt",weights_only=False)
    radio=pd.DataFrame.from_dict({p:t.numpy() for p,t in rd["features"].items()},orient="index",columns=list(rd["names"]))
    struct=generate_binary_structure(3,3); rows={}
    for d in MASK_DIRS:
        for mf in sorted(Path(d).glob("fold*_*.mha")): rows[mf.stem.split("_",1)[1]]=nodal_geom(mf,struct)
    geom=pd.DataFrame.from_dict(rows,orient="index"); geom["log_largest"]=np.log1p(geom["largest_ml"]); geom["log_total"]=np.log1p(geom["total_ml"]); geom=geom[GEOM]
    folds=[pd.read_csv(MANIFEST_DIR/f"val_fold{f}.csv")["patient_id"].astype(str).str.strip().tolist() for f in range(5)]

    # ---------- N-stage OOF predictions (pooled), matched C ----------
    def n_oof(featfn,C):
        yt=[]; yp=[]; pid=[]
        for f in range(5):
            vp=[p for p in folds[f] if p in df.index and pd.notna(df.loc[p,"N-stage"]) and p in geom.index]
            tp=[p for p in df.index if p not in folds[f] and pd.notna(df.loc[p,"N-stage"]) and p in geom.index]
            Xtr=featfn(tp); Xv=featfn(vp); sc=StandardScaler().fit(Xtr)
            clf=LogisticRegression(penalty="l1",solver="saga",C=C,max_iter=5000,class_weight="balanced").fit(clean(sc.transform(Xtr)),df.loc[tp,"N-stage"].map(N_MAP).values)
            yp += clf.predict(clean(sc.transform(Xv))).tolist(); yt += df.loc[vp,"N-stage"].map(N_MAP).tolist(); pid+=vp
        return np.array(yt),np.array(yp)
    base=lambda ids: pd.concat([clin.loc[ids],radio.reindex(ids).fillna(0)],axis=1)
    withg=lambda ids: pd.concat([clin.loc[ids],geom.reindex(ids).fillna(0)],axis=1)
    print("=== N-stage balanced accuracy, patient-level bootstrap (2000) ===")
    for tag,featfn,C in [("clin+radio C=0.05",base,0.05),("clin+geom  C=0.05",withg,0.05),("clin+geom  C=0.03 (deployed)",withg,0.03)]:
        yt,yp=n_oof(featfn,C)
        pt=balanced_accuracy_score(yt,yp)
        bs=[balanced_accuracy_score(yt[b],yp[b]) for b in (RNG.choice(len(yt),len(yt),True) for _ in range(2000))]
        print(f"  {tag:30} balAcc {pt:.3f}  95%CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    # paired delta geom(C=0.05) - radio(C=0.05)
    ytr,ypr=n_oof(base,0.05); ytg,ypg=n_oof(withg,0.05)
    assert len(ytr)==len(ytg)
    deltas=[]
    for _ in range(2000):
        b=RNG.choice(len(ytr),len(ytr),True)
        deltas.append(balanced_accuracy_score(ytg[b],ypg[b])-balanced_accuracy_score(ytr[b],ypr[b]))
    d=np.array(deltas); print(f"  PAIRED Δ(geom-radio)@C=0.05: {d.mean():+.3f}  95%CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  P(Δ>0)={np.mean(d>0):.3f}")

    # ---------- RFS pooled C-index, bootstrap + paired ----------
    EX={"tri10":"medai_10foldmask_triplehead_aug","rfs10":"medai_10foldmask_rfs_only_aug","rfs_sig":"medai_10foldmask_rfs_sigmoid","rfs_aff":"medai_10fold_rfs_affaug"}
    def load(v):
        fr=[pd.read_csv(ROOT/f"predictions/{v}/fold{f}/predictions.csv") for f in range(5)]; return pd.concat(fr).set_index("patient_id")["deep_risk"]
    deep={k:load(v) for k,v in EX.items()}
    def cr(Ct,Cv,t,e):
        mu=Ct.mean(0); sd=Ct.std(0).replace(0.,1.); dfc=((Ct-mu)/sd).copy(); dfc["T"]=t; dfc["E"]=e
        c=CoxPHFitter(penalizer=2.,l1_ratio=0.); c.fit(dfc,duration_col="T",event_col="E",show_progress=False); return np.log(c.predict_partial_hazard((Cv-mu)/sd).values+1e-9)
    pf={k:[] for k in list(deep)+["clin"]}; T=[]; E=[]
    for f in range(5):
        vp=pd.read_csv(MANIFEST_DIR/f"val_fold{f}.csv")["patient_id"].astype(str).str.strip().tolist()
        ok=lambda p: p in df.index and pd.notna(df.loc[p,"RFS"]) and float(df.loc[p,"RFS"])>0 and pd.notna(df.loc[p,"Relapse"])
        vp=[p for p in vp if ok(p)]; tp=[p for p in df.index if p not in vp and ok(p)]
        pf["clin"].append(zn(cr(clin.loc[tp],clin.loc[vp],df.loc[tp,"RFS"].astype(float).values,df.loc[tp,"Relapse"].astype(int).values)))
        for k in deep: pf[k].append(zn(deep[k].reindex(vp).fillna(deep[k].mean()).values))
        T.append(df.loc[vp,"RFS"].astype(float).values); E.append(df.loc[vp,"Relapse"].astype(int).values)
    tm=np.concatenate(T); ev=np.concatenate(E)
    def risk(mem): return np.concatenate([np.mean([pf[m][f] for m in mem],axis=0) for f in range(5)])
    print(f"\n=== RFS pooled C-index (n={len(tm)}, events={int(ev.sum())}), bootstrap (2000) ===")
    configs={"cox rfs10 (solo)":["rfs10"],"sigmoid (solo)":["rfs_sig"],"4-expert (base)":["tri10","rfs10","rfs_aff","clin"],"5-expert (+sig, HERMES+)":["tri10","rfs10","rfs_sig","rfs_aff","clin"]}
    R={}
    for tag,mem in configs.items():
        r=risk(mem); R[tag]=r
        bs=[harrell_cindex(r[b],tm[b],ev[b]) for b in (RNG.choice(len(tm),len(tm),True) for _ in range(2000))]
        print(f"  {tag:26} C {harrell_cindex(r,tm,ev):.3f}  95%CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    def paired(a,b,label):
        ra,rb=R[a],R[b]; ds=[harrell_cindex(ra[i],tm[i],ev[i])-harrell_cindex(rb[i],tm[i],ev[i]) for i in (RNG.choice(len(tm),len(tm),True) for _ in range(2000))]
        ds=np.array(ds); print(f"  PAIRED Δ {label}: {ds.mean():+.3f}  95%CI [{np.percentile(ds,2.5):+.3f},{np.percentile(ds,97.5):+.3f}]  P(Δ>0)={np.mean(ds>0):.3f}")
    paired("sigmoid (solo)","cox rfs10 (solo)","sigmoid - cox (solo)")
    paired("5-expert (+sig, HERMES+)","4-expert (base)","5exp - 4exp (ensemble)")

if __name__=="__main__": main()
