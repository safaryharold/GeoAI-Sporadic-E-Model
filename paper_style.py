"""Shared publication style for all manuscript figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed colour semantics across all figures
OBSERVED = "#000000"          # observed / ionosonde = black
PREDICTED = "#0072B2"         # predicted (XGBoost) = XGBoost blue
ES = "#D55E00"                # Es = vermillion
NOES = "#8C8C8C"              # no Es = neutral blue-grey
METRICS = {"ROC-AUC":"#009E73","PR-AUC":"#0072B2","BSS":"#D55E00",
           "POD":"#56B4E9","FAR":"#CC79A7","F1":"#E69F00"}
MODELS = {
    "XGBoost": "#0072B2",
    "Random forest": "#009E73",
    "Hist. grad. boost.": "#E69F00",
    "Logistic reg.": "#CC79A7",
    "MLP (deep)": "#D55E00",
    "TCN (seq.)": "#999999",
}

def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
    })

def panel_label(ax, label="(a)"):
    ax.set_title(label, loc="left", fontweight="bold", fontsize=9, pad=4)

def save(fig, name, outdir):
    import os
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
