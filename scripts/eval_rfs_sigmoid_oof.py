"""SurvLoss test: does the SIGMOID-concordance-trained RFS expert beat / stabilize the
cox-trained rfs10 in our deployed ensemble? Honest 5-fold OOF, deploy recipes.

Experts: tri10, rfs10(cox), rfs_sig(sigmoid, NEW), rfs_aff.  Non-deep: clin(v14 deploy).
Baselines to beat: v14 ensemble tri+rfs10+rfs_aff+clin = 0.7107 (±0.0346)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from task3_integrator import _encode_clinical_row, CSV_PATH, MANIFEST_DIR  # noqa
sys.path.insert(0, str(ROOT)); from training.metrics import harrell_cindex  # noqa
from lifelines import CoxPHFitter
REDUNDANT = ["hpv_unk","smoker_missing","drinker_missing","ps_missing","treatment_missing"]
EXPERTS = {"tri10":"medai_10foldmask_triplehead_aug","rfs10":"medai_10foldmask_rfs_only_aug",
           "rfs_sig":"medai_10foldmask_rfs_sigmoid","rfs_aff":"medai_10fold_rfs_affaug"}
def load(v):
    fr=[pd.read_csv(ROOT/f"predictions/{v}/fold{f}/predictions.csv") for f in range(5) if (ROOT/f"predictions/{v}/fold{f}/predictions.csv").exists()]
    return pd.concat(fr).set_index("patient_id")["deep_risk"] if len(fr)==5 else None
def zn(x): x=np.asarray(x,float); s=x.std(); return (x-x.mean())/(s if s>0 else 1)
def clin_ridge(Ct,Cv,t,e):
    mu=Ct.mean(0); sd=Ct.std(0).replace(0.0,1.0); dfc=((Ct-mu)/sd).copy(); dfc["T"]=t; dfc["E"]=e
    cph=CoxPHFitter(penalizer=2.0,l1_ratio=0.0); cph.fit(dfc,duration_col="T",event_col="E",show_progress=False)
    return np.log(cph.predict_partial_hazard((Cv-mu)/sd).values+1e-9)

def main():
    df=pd.read_csv(CSV_PATH); df["PatientID"]=df["PatientID"].astype(str).str.strip(); df=df.set_index("PatientID")
    clin=pd.DataFrame({p:_encode_clinical_row(r) for p,r in df.iterrows()}).T.drop(columns=REDUNDANT)
    deep={k:load(v) for k,v in EXPERTS.items()}
    for k,v in deep.items():
        if v is None: print(f"!! expert {k} missing");
    pf={k:[] for k in list(deep)+["clin"]}; truth=[]
    for f in range(5):
        vp=pd.read_csv(MANIFEST_DIR/f"val_fold{f}.csv")["patient_id"].astype(str).str.strip().tolist()
        ok=lambda p: p in df.index and pd.notna(df.loc[p,"RFS"]) and float(df.loc[p,"RFS"])>0 and pd.notna(df.loc[p,"Relapse"])
        vp=[p for p in vp if ok(p)]; tp=[p for p in df.index if p not in vp and ok(p)]
        tmv=df.loc[tp,"RFS"].astype(float).values; evv=df.loc[tp,"Relapse"].astype(int).values
        pf["clin"].append(zn(clin_ridge(clin.loc[tp],clin.loc[vp],tmv,evv)))
        for k in deep: pf[k].append(zn(deep[k].reindex(vp).fillna(deep[k].mean()).values))
        truth.append((df.loc[vp,"RFS"].astype(float).values,df.loc[vp,"Relapse"].astype(int).values))
    tm=np.concatenate([t[0] for t in truth]); ev=np.concatenate([t[1] for t in truth])
    def solo(k):
        cs=[harrell_cindex(pf[k][f],*truth[f]) for f in range(5)]
        pooled=harrell_cindex(np.concatenate([pf[k][f] for f in range(5)]),tm,ev)
        return pooled,float(np.mean(cs)),float(np.std(cs))
    def ens(mem):
        pooled=[np.mean([pf[m][f] for m in mem],axis=0) for f in range(5)]
        cs=[harrell_cindex(np.mean([pf[m][f] for m in mem],axis=0),*truth[f]) for f in range(5)]
        return harrell_cindex(np.concatenate(pooled),tm,ev),float(np.mean(cs)),float(np.std(cs))
    print("=== SOLO deep RFS expert (cox rfs10 vs sigmoid rfs_sig) ===")
    for k in ["rfs10","rfs_sig","rfs_aff","tri10"]:
        pc,mu,sd=solo(k); print(f"  {k:9} pooled {pc:.4f}   perfold {mu:.4f}±{sd:.4f}")
    print("\n=== 4-expert ENSEMBLE (non-deep = clin, deploy) ===")
    combos=[("tri10+rfs10+rfs_aff+clin",["tri10","rfs10","rfs_aff","clin"],"v14 DEPLOY"),
            ("tri10+rfs_sig+rfs_aff+clin",["tri10","rfs_sig","rfs_aff","clin"],"SWAP cox->sig"),
            ("tri10+rfs10+rfs_sig+rfs_aff+clin",["tri10","rfs10","rfs_sig","rfs_aff","clin"],"ADD sig (5-exp)")]
    for name,mem,tag in combos:
        pc,mu,sd=ens(mem); print(f"  {name:36}{pc:.4f}   {mu:.4f}±{sd:.4f}  {tag}")
    print("\nADOPT sigmoid only if SWAP CLEARLY & STABLY >= v14 (0.7107, ±0.0346).")

if __name__=="__main__": main()
