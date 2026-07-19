"""Generate HERMES paper results figures from REAL experimental data (no mock values).
Fig 3: nodal geometry vs GT N-stage (monotone) + N-stage balacc radiomics vs geometry.
Fig 4: per-fold C-index, concordance-loss vs Cox deep expert.
Fig 5: deployment robustness Mean Dice before/after PET->CT co-registration fix.
Output -> docs/figures/."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from task3_integrator import _encode_clinical_row, CSV_PATH, MANIFEST_DIR  # noqa
sys.path.insert(0, str(ROOT)); from training.metrics import harrell_cindex  # noqa
from n_nodal_geom_probe import nodal_geom, MASK_DIRS
from scipy.ndimage import generate_binary_structure
OUT = ROOT / "docs/figures"; OUT.mkdir(parents=True, exist_ok=True)
NAVY="#1f3b5c"; GOLD="#c9a24b"; GREY="#8a8a8a"; RED="#b23b3b"
plt.rcParams.update({"font.size":11,"axes.grid":False,"figure.dpi":200})

def zn(x): x=np.asarray(x,float); s=x.std(); return (x-x.mean())/(s if s>0 else 1)

def load_df():
    df=pd.read_csv(CSV_PATH); df["PatientID"]=df["PatientID"].astype(str).str.strip(); return df.set_index("PatientID")

# ---------------- Fig 3 ----------------
def fig3(df):
    struct=generate_binary_structure(3,3); rows={}
    for d in MASK_DIRS:
        for mf in sorted(Path(d).glob("fold*_*.mha")): rows[mf.stem.split("_",1)[1]]=nodal_geom(mf,struct)
    geom=pd.DataFrame.from_dict(rows,orient="index").join(df["N-stage"]).dropna(subset=["N-stage"])
    order=["N0","N1","N2","N3"]
    g=geom.groupby("N-stage")[["n_ccs","largest_maxdim"]].mean().reindex(order)
    fig,ax=plt.subplots(1,2,figsize=(9,3.6))
    x=np.arange(4)
    a0=ax[0]; a0b=a0.twinx()
    a0.bar(x-0.2,g["largest_maxdim"],0.4,color=NAVY,label="Largest-node extent (mm)")
    a0b.bar(x+0.2,g["n_ccs"],0.4,color=GOLD,label="Node count")
    a0.set_xticks(x); a0.set_xticklabels(order); a0.set_xlabel("Ground-truth N-stage")
    a0.set_ylabel("Largest-node extent (mm)",color=NAVY); a0b.set_ylabel("Nodal component count",color=GOLD)
    a0.set_title("(a) Predicted-mask nodal geometry vs. N-stage")
    h1,l1=a0.get_legend_handles_labels(); h2,l2=a0b.get_legend_handles_labels()
    a0.legend(h1+h2,l1+l2,fontsize=8,loc="upper left")
    # right: N balacc radiomics vs geometry with bootstrap CI (from paper_stats_ci)
    labels=["clinical +\nradiomics","clinical +\ngeometry"]; vals=[0.691,0.720]
    lo=[0.650,0.678]; hi=[0.730,0.760]
    err=[[v-l for v,l in zip(vals,lo)],[h-v for v,h in zip(vals,hi)]]
    a1=ax[1]; a1.bar([0,1],vals,0.55,color=[GREY,NAVY],yerr=err,capsize=5)
    a1.set_xticks([0,1]); a1.set_xticklabels(labels); a1.set_ylim(0.6,0.8)
    a1.set_ylabel("N-stage balanced accuracy (OOF)")
    a1.set_title("(b) N-stage: radiomics vs. geometry")
    for i,v in enumerate(vals): a1.text(i,v+0.012,f"{v:.3f}",ha="center",fontsize=10)
    a1.annotate("+0.030",xy=(0.5,0.745),ha="center",fontsize=10,color=NAVY)
    fig.tight_layout(); fig.savefig(OUT/"fig3_nodal_geometry.png",bbox_inches="tight"); plt.close(fig)
    print("fig3 done:", g.to_dict())

# ---------------- Fig 4 ----------------
def fig4(df):
    EX={"cox (rfs10)":"medai_10foldmask_rfs_only_aug","concordance loss":"medai_10foldmask_rfs_sigmoid"}
    def load(v):
        return [pd.read_csv(ROOT/f"predictions/{v}/fold{f}/predictions.csv").set_index("patient_id")["deep_risk"] for f in range(5)]
    risks={k:load(v) for k,v in EX.items()}
    perfold={k:[] for k in EX}
    for f in range(5):
        vp=pd.read_csv(MANIFEST_DIR/f"val_fold{f}.csv")["patient_id"].astype(str).str.strip().tolist()
        ok=lambda p: p in df.index and pd.notna(df.loc[p,"RFS"]) and float(df.loc[p,"RFS"])>0 and pd.notna(df.loc[p,"Relapse"])
        vp=[p for p in vp if ok(p)]
        tm=df.loc[vp,"RFS"].astype(float).values; ev=df.loc[vp,"Relapse"].astype(int).values
        for k in EX:
            r=risks[k][f].reindex(vp).fillna(risks[k][f].mean()).values
            perfold[k].append(harrell_cindex(zn(r),tm,ev))
    fig,ax=plt.subplots(figsize=(5,3.8))
    x=np.arange(5)
    ax.plot(x,perfold["cox (rfs10)"],"o-",color=GREY,label=f"Cox  (mean {np.mean(perfold['cox (rfs10)']):.3f}±{np.std(perfold['cox (rfs10)']):.3f})")
    ax.plot(x,perfold["concordance loss"],"s-",color=NAVY,label=f"Concordance loss (mean {np.mean(perfold['concordance loss']):.3f}±{np.std(perfold['concordance loss']):.3f})")
    ax.set_xticks(x); ax.set_xticklabels([f"fold {i}" for i in range(5)])
    ax.set_ylabel("RFS C-index (out-of-fold)"); ax.set_title("Deep survival expert: concordance loss vs. Cox")
    ax.legend(fontsize=8,loc="lower right"); ax.grid(axis="y",alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT/"fig4_sigmoid_vs_cox.png",bbox_inches="tight"); plt.close(fig)
    print("fig4 done: cox",[f"{v:.3f}" for v in perfold["cox (rfs10)"]],"| loss",[f"{v:.3f}" for v in perfold["concordance loss"]])

# ---------------- Fig 5 ----------------
def fig5():
    # Sanity-check platform Mean Dice, before vs after the PET->CT co-registration fix
    # (real submitted-container numbers documented in the deployment-robustness study).
    fig,ax=plt.subplots(figsize=(4.2,3.6))
    vals=[0.1944,0.8953]; ax.bar([0,1],vals,0.55,color=[RED,NAVY])
    ax.set_xticks([0,1]); ax.set_xticklabels(["before fix","after fix"]); ax.set_ylim(0,1)
    ax.set_ylabel("Sanity-check Mean Dice (platform)"); ax.set_title("PET→CT co-registration fix")
    for i,v in enumerate(vals): ax.text(i,v+0.02,f"{v:.3f}",ha="center",fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/"fig5_robustness.png",bbox_inches="tight"); plt.close(fig)
    print("fig5 done")

def main():
    df=load_df(); fig3(df); fig4(df); fig5()
    print("figures ->", OUT)

if __name__=="__main__": main()
