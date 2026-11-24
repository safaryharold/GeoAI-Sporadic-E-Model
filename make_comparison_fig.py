"""Model-comparison figures: paired per-fold skill + MLP schematic."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "paper", "figures")
OI = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
      "purple":"#CC79A7","gray":"#999999","black":"#222222"}
plt.rcParams.update({"font.size":9,"figure.dpi":300,"savefig.dpi":300,
    "savefig.bbox":"tight","font.family":"DejaVu Sans"})

fm = pd.read_csv(os.path.join(BASE,"paper_outputs","fold_metrics.csv"))
mlp = pd.read_csv(os.path.join(BASE,"paper_outputs","deep_baseline","mlp_metrics.csv"))
tcn = pd.read_csv(os.path.join(BASE,"paper_outputs","tcn_baseline","tcn_metrics.csv"))

# map mlp tags s{seed}f{fold} -> fold_id, seed
mlp["seed"] = mlp.tag.str.extract(r"s(\d+)f")[0].astype(int)
mlp["fold_id"] = mlp.tag.str.extract(r"f(\d+)$")[0].astype(int)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
models = [("xgb","XGBoost",OI["blue"]),("rf","Random forest",OI["green"]),
          ("hgb","Hist. grad. boost.",OI["orange"]),("logreg","Logistic reg.",OI["purple"]),
          ("mlp","MLP (deep)",OI["red"]),("tcn","TCN (seq.)",OI["gray"])]
x = np.arange(len(mlp))
width = 0.13
for i,(key,name,col) in enumerate(models):
    if key=="mlp":
        vals = mlp.sort_values(["seed","fold_id"]).roc.values
    elif key=="tcn":
        t5=tcn.copy(); t5["fold_id"]=t5.tag.str.extract(r"f(\d+)$")[0].astype(int)
        t5=t5.sort_values("fold_id")
        vals=np.concatenate([t5.roc.values, np.full(5,np.nan)])
    else:
        sub = fm[fm.model==key].sort_values(["seed","fold_id"])
        vals = sub.cal_roc_auc.values
    if len(vals)<len(x): vals=np.concatenate([vals,np.full(len(x)-len(vals),np.nan)])
    ax.bar(x+(i-3)*width, vals[:len(x)], width, label=name, color=col, alpha=0.9)
ax.axhline(0.5, ls="--", color="#999999", lw=0.8)
mlp_s=mlp.sort_values(["seed","fold_id"])
ax.set_xticks(x); ax.set_xticklabels([f"P{i+1}" for i in range(len(x))],rotation=0,fontsize=8)
ax.set_xlabel("CV partition (5 folds × 2 seeds)")
ax.set_ylabel("Calibrated ROC-AUC"); ax.set_xlabel("CV partition"); ax.set_ylim(0.4,1.0)
ax.set_title("(a) Per-partition ROC-AUC (10 spatial CV partitions)",fontsize=10)
ax.legend(ncol=3,frameon=False,fontsize=7.5,loc="lower left"); ax.grid(axis="y"); ax.set_axisbelow(True)

ax = axes[1]
# pooled metrics comparison
pooled = {"Logistic reg.":(0.749,0.461,0.130,0.005),"Random forest":(0.899,0.726,0.091,0.007),
          "Hist. grad. boost.":(0.900,0.735,0.090,0.004),"XGBoost":(0.901,0.737,0.090,0.003)}
mlpmean = (float(mlp.roc.mean()),float(mlp.pra.mean()),float(mlp.brier.mean()),float(mlp.ece.mean()))
pooled["MLP (deep)"]=mlpmean
names=list(pooled); roc=[pooled[n][0] for n in names]; ece=[pooled[n][3] for n in names]
cols=[OI["purple"],OI["green"],OI["orange"],OI["blue"],OI["red"]]
bars=ax.bar(names,roc,color=cols,width=0.6)
for b,v in zip(bars,roc): ax.text(b.get_x()+b.get_width()/2,v+0.008,f"{v:.3f}",ha="center",fontsize=8)
ax.set_ylim(0.4,1.0); ax.set_ylabel("Pooled ROC-AUC"); ax.set_xlabel("Model family")
ax.set_title("(b) Pooled out-of-fold ROC-AUC",fontsize=10)
ax.tick_params(axis="x",rotation=20,labelsize=8); ax.grid(axis="y"); ax.set_axisbelow(True)
# annotate ECE under bars
for b,n in zip(bars,names):
    ax.text(b.get_x()+b.get_width()/2,0.42,f"ECE={pooled[n][3]:.3f}",ha="center",fontsize=7,color="#444444")
fig.legend(handles=[plt.Rectangle((0,0),1,1,color=c,alpha=0.9) for _,_,c in models],labels=[n for _,n,_ in models],ncol=6,frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(0.5,-0.02))
fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig22_model_comparison.png"))
fig.savefig(os.path.join(OUT,"fig22_model_comparison.pdf"))

# ---- MLP architecture schematic ----
fig,ax=plt.subplots(figsize=(8.5,3.0)); ax.axis("off")
def box(x,y,w,h,text,fc="#EAF2FB",ec="#336699"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color="#333333",lw=1.4))
box(0.02,0.35,0.16,0.3,"Input\n28 features\n(z-scored)")
box(0.24,0.35,0.16,0.3,"Dense 128\nReLU")
box(0.46,0.35,0.16,0.3,"Dense 64\nReLU")
box(0.68,0.35,0.16,0.3,"Dense 1\nlogit")
box(0.86,0.35,0.12,0.3,"Isotonic\ncalibration")
for a,b in [(0.18,0.24),(0.40,0.46),(0.62,0.68),(0.84,0.86)]:
    arrow(a,0.5,b,0.5)
ax.text(0.02,0.12,"Training: Adam, lr=1e-3, batch=1024, early stopping (8 epochs), class-weighted log-loss",
        fontsize=8,style="italic",color="#333333")
ax.text(0.02,0.04,"Same 10 spatial-CV partitions, 300 km buffer, and inner 80/20 calibration split as XGBoost",
        fontsize=8,style="italic",color="#333333")
ax.set_title("Deep tabular baseline (MLP) architecture and protocol",fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig23_mlp_architecture.png"))
fig.savefig(os.path.join(OUT,"fig23_mlp_architecture.pdf"))
print("fig22, fig23 saved; MLP pooled:",mlpmean)
