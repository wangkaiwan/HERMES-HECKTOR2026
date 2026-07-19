"""Fig 2: predicted GTVp/GTVn overlays across the 8 training centers (real deployed-model
OOF predictions). Per center, pick the patient with the largest GT tumor volume; show the
max-GT axial slice: CT background + predicted GTVp (red) / GTVn (cyan) semi-transparent,
with ground-truth contours (yellow GTVp, green GTVn)."""
from __future__ import annotations
from pathlib import Path
import numpy as np, SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"; OUT.mkdir(parents=True, exist_ok=True)

def collect_preds():
    preds={}
    for f in range(5):
        for mf in (ROOT/f"evaluation/results/qa_10fold_oof_fold{f}_masks").glob("fold*_*.mha"):
            preds[mf.stem.split("_",1)[1]]=mf
    return preds

def load(pid, predpath):
    ct=sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT/f"data/raw/{pid}/{pid}__CT.nii.gz"))).astype(np.float32)
    gt=sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT/f"data/raw/{pid}/{pid}.nii.gz"))).astype(np.uint8)
    pr=sitk.GetArrayFromImage(sitk.ReadImage(str(predpath))).astype(np.uint8)
    return ct,gt,pr

def main():
    preds=collect_preds()
    # per center, pick patient with max GT volume (visible tumor)
    best={}
    for pid,mf in preds.items():
        c=pid.rsplit("-",1)[0]
        gtp=ROOT/f"data/raw/{pid}/{pid}.nii.gz"; ctp=ROOT/f"data/raw/{pid}/{pid}__CT.nii.gz"
        if not (gtp.exists() and ctp.exists()): continue
        try: gt=sitk.GetArrayFromImage(sitk.ReadImage(str(gtp)))
        except Exception: continue
        v=int((gt>0).sum())
        if v> best.get(c,(None,None,-1))[2]: best[c]=(pid,mf,v)
    centers=sorted(best)
    fig,axes=plt.subplots(2,4,figsize=(14,7.4))
    for ax,c in zip(axes.ravel(),centers):
        pid,mf,_=best[c]; ct,gt,pr=load(pid,mf)
        if not (ct.shape==gt.shape==pr.shape):  # geometry mismatch guard
            ax.axis("off"); ax.set_title(f"{c}: shape mismatch",fontsize=9); continue
        z=int(np.argmax((gt>0).sum(axis=(1,2))))
        cts=np.clip(ct[z],-160,240); cts=(cts-cts.min())/(cts.ptp()+1e-6)
        ax.imshow(cts,cmap="gray",interpolation="nearest")
        # predicted overlays (semi-transparent fill)
        for cls,col in [(1,(0.90,0.10,0.10)),(2,(0.10,0.75,0.90))]:
            m=(pr[z]==cls).astype(float)
            rgba=np.zeros((*m.shape,4)); rgba[...,0],rgba[...,1],rgba[...,2]=col; rgba[...,3]=m*0.45
            ax.imshow(rgba,interpolation="nearest")
        # GT contours
        for cls,col in [(1,"yellow"),(2,"lime")]:
            g=(gt[z]==cls).astype(float)
            if g.any(): ax.contour(g,levels=[0.5],colors=col,linewidths=0.9)
        ax.set_title(f"{c}  ({pid})",fontsize=11); ax.axis("off")
        # crop to tumor bbox for visibility
        fg=(gt[z]>0)|(pr[z]>0)
        if fg.any():
            ys,xs=np.where(fg); pad=40
            ax.set_ylim(min(ys.max()+pad,ct.shape[1]),max(ys.min()-pad,0))
            ax.set_xlim(max(xs.min()-pad,0),min(xs.max()+pad,ct.shape[2]))
    # legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles=[Patch(color=(0.90,0.10,0.10),alpha=0.45,label="pred GTVp"),
             Patch(color=(0.10,0.75,0.90),alpha=0.45,label="pred GTVn"),
             Line2D([0],[0],color="yellow",lw=2,label="GT GTVp"),
             Line2D([0],[0],color="lime",lw=2,label="GT GTVn")]
    fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=11,bbox_to_anchor=(0.5,-0.01))
    fig.suptitle("Predicted vs. ground-truth segmentation across the 8 training centers (deployed-model OOF)",fontsize=12)
    fig.tight_layout(rect=[0,0.03,1,1])
    fig.savefig(OUT/"fig2_seg_overlays.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    print("fig2 done:", {c:best[c][0] for c in centers})

if __name__=="__main__": main()
