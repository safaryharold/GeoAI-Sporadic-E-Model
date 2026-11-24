"""Regenerate manuscript figures whose original ad-hoc script was not saved:
fig16 (spatial verification), fig24 (error diagnostics), fig25 (seasonal
occurrence), fig26 (RO-ionosonde detection agreement), fig27 (per-station
seasonal cycles). All parameters are documented here and match the figure
captions in paper/main.tex."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import roc_auc_score, brier_score_loss, cohen_kappa_score
import paper_style as ps

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "paper", "figures")
VAL = os.path.join(BASE, "paper_outputs", "validation")
OOF = os.path.join(BASE, "paper_outputs", "oof", "xgb_oof.csv")

DOMAIN = [-86, -29, -36, 16]

def basemap(ax, gx=None, gy=None, nolabels=False, nogrid=False):
    ax.set_extent(DOMAIN, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#EAF1F7", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#F2EFE6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="#DCE9F5", edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE, lw=0.6, color="#5A5A5A", zorder=3)
    ax.add_feature(cfeature.BORDERS, lw=0.4, color="#8A8A8A", ls=":", zorder=3)
    if nogrid:
        return
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="#AAAAAA", alpha=0.6,
                      linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}
    if gx is not None: gl.xticks = gx
    if gy is not None: gl.yticks = gy
    gl.top_labels = False; gl.right_labels = False
    if nolabels:
        gl.top_labels = gl.bottom_labels = gl.left_labels = gl.right_labels = False

# ================= FIG 16: spatial verification (2 deg cells, n>=300) =================
def fig16():
    ps.setup()
    df = pd.read_csv(OOF)
    df["clat"] = (df["lat"] / 2).round() * 2
    df["clon"] = (df["lon"] / 2).round() * 2
    rows = []
    for (la, lo), g in df.groupby(["clat", "clon"]):
        if len(g) < 300 or g["y_true"].nunique() < 2:
            rows.append((la, lo, np.nan, np.nan, len(g)))
            continue
        rows.append((la, lo, roc_auc_score(g["y_true"], g["p_cal"]),
                     brier_score_loss(g["y_true"], g["p_cal"]), len(g)))
    sk = pd.DataFrame(rows, columns=["lat", "lon", "auc", "brier", "n"])
    ok = sk.dropna(subset=["auc"])
    print("fig16 cells:", len(ok), "median auc:", round(ok.auc.median(), 3),
          "max brier:", round(ok.brier.max(), 3))

    cmaps = [LinearSegmentedColormap.from_list(
                 "auc", ["#D73027", "#FEE08B", "#F7F7F7", "#91BFDB", "#2166AC"]),
             LinearSegmentedColormap.from_list(
                 "bri", ["#2166AC", "#91BFDB", "#F7F7F7", "#FEE08B", "#D73027"])]
    norms = [plt.Normalize(0.75, 0.95), plt.Normalize(0.02, 0.16)]
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
    for k, (panel, ttl) in enumerate(
            zip(["auc", "brier"],
                ["(a) Cell ROC-AUC (2$^\\circ$ grid, n $\\geq$ 300)",
                 "(b) Cell Brier score"])):
        ax = fig.add_subplot(gs[0, k], projection=ccrs.PlateCarree())
        basemap(ax)
        sc = ax.scatter(sk["lon"], sk["lat"], c=sk[panel], cmap=cmaps[k],
                        norm=norms[k], s=55, marker="s", edgecolors="none",
                        transform=ccrs.PlateCarree(), zorder=4)
        ax.set_title(ttl, fontsize=9, pad=8)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal",
                          fraction=0.05, pad=0.12, aspect=40)
        cb.set_label("ROC-AUC" if panel == "auc" else "Brier score", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(OUT, "fig16_spatial_skill.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig16_spatial_skill.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig16 done")

# ================= FIG 24: error diagnostics =================
def fig24():
    ps.setup()
    d = pd.read_csv(OOF)
    d["err"] = (d["p_cal"] - d["y_true"]).abs()

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))

    # (a) latitude bands of 10 deg
    ax = axes[0, 0]
    d["latband"] = pd.cut(d["lat"], [-36, -25, -15, -5, 5, 16])
    gb = d.groupby("latband", observed=True).err.mean()
    xpos = np.arange(len(gb))
    ax.bar(xpos, gb.values, color="#0072B2", width=0.65)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{i.left:.0f} to {i.right:.0f}" for i in gb.index],
                       rotation=30, ha="right", fontsize=7)
    ax.set_xlabel("Latitude band (°)"); ax.set_ylabel("Mean |error|")
    ax.set_title("(a) Latitude", fontsize=9, loc="left", fontweight="bold")
    ax.grid(axis="y")

    # (b) UTC hour
    ax = axes[0, 1]
    gh = d.groupby("hour").err.mean()
    ax.bar(gh.index, gh.values, color="#0072B2", width=0.8)
    ax.set_xlabel("UTC hour"); ax.set_ylabel("Mean |error|")
    ax.set_title("(b) UTC hour", fontsize=9, loc="left", fontweight="bold")
    ax.set_xlim(-0.5, 23.5); ax.grid(axis="y")

    # (c) month
    ax = axes[0, 2]
    gm = d.groupby("month").err.mean()
    ax.bar(gm.index, gm.values, color="#0072B2", width=0.65)
    ax.set_xlabel("Month"); ax.set_ylabel("Mean |error|")
    ax.set_title("(c) Month", fontsize=9, loc="left", fontweight="bold")
    ax.set_xticks(range(1, 13)); ax.grid(axis="y")

    # (d) year
    ax = axes[1, 0]
    gy = d.groupby("year").err.mean()
    ax.plot(gy.index, gy.values, "-o", color="#0072B2", ms=3, lw=1.2)
    ax.set_xlabel("Year"); ax.set_ylabel("Mean |error|")
    ax.set_title("(d) Year", fontsize=9, loc="left", fontweight="bold")
    ax.grid(axis="y")

    # (e) predicted-probability decile + observed event rate
    ax = axes[1, 1]
    d["dec"] = pd.qcut(d["p_cal"], 10, labels=False, duplicates="drop")
    gd = d.groupby("dec").agg(err=("err", "mean"), obs=("y_true", "mean"),
                              p=("p_cal", "mean"))
    xpos = np.arange(len(gd))
    ax.bar(xpos, gd["err"].values, color="#0072B2", width=0.65,
           label="Mean |error|")
    ax2 = ax.twinx()
    ax2.plot(xpos, gd["obs"].values, "s--", color="#D55E00", ms=4,
             label="Observed event rate")
    ax2.set_ylabel("Observed event rate", color="#D55E00")
    ax2.tick_params(axis="y", colors="#D55E00")
    ax.set_xlabel("Predicted-probability decile")
    ax.set_ylabel("Mean |error|")
    ax.set_title("(e) Probability decile", fontsize=9, loc="left",
                 fontweight="bold")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{i+1}" for i in range(len(gd))], fontsize=7)
    ax.grid(axis="y")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")

    # (f) mean |error| in 5 deg spatial cells
    ax = fig.add_subplot(axes[1, 2].get_subplotspec().subgridspec(1, 1)[0, 0],
                         projection=ccrs.PlateCarree())
    d["clat"] = (d["lat"] / 5).round() * 5
    d["clon"] = (d["lon"] / 5).round() * 5
    ge = d.groupby(["clat", "clon"]).err.mean().reset_index()
    basemap(ax, nogrid=True)
    sc = ax.scatter(ge["clon"], ge["clat"], c=ge["err"], s=400,
                    cmap="YlOrRd", marker="s", edgecolors="none",
                    transform=ccrs.PlateCarree(), zorder=4)
    ax.set_title("(f) Mean error, 5$^\\circ$ cells",
                 fontsize=9, loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.06,
                      pad=0.04, aspect=25)
    mx = float(ge["err"].max())
    cb.set_ticks(np.linspace(0.0, mx, 4))
    cb.set_label("Mean |error|", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.10,
                        hspace=0.42, wspace=0.50)
    fig.savefig(os.path.join(OUT, "fig24_error_diagnostics.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig24_error_diagnostics.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig24 done")

# ================= FIG 25: seasonal occurrence (2.5 deg cells) =================
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8], "SON": [9, 10, 11]}

def fig25():
    ps.setup()
    d = pd.read_csv(OOF, usecols=["lat", "lon", "month", "y_true"])
    d["clat"] = (d["lat"] / 2.5).round() * 2.5
    d["clon"] = (d["lon"] / 2.5).round() * 2.5

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.4),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             gridspec_kw={"wspace": 0.05, "hspace": 0.10})
    cmap = LinearSegmentedColormap.from_list(
        "es", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    for ax, (sname, months) in zip(axes.flat, SEASONS.items()):
        sub = d[d["month"].isin(months)]
        g = sub.groupby(["clat", "clon"]).agg(rate=("y_true", "mean"),
                                             n=("y_true", "size"))
        g = g[g["n"] >= 20]
        basemap(ax)
        sc = ax.scatter(g.index.get_level_values(1),
                        g.index.get_level_values(0), c=g["rate"], s=120,
                        cmap=cmap, vmin=0, vmax=0.4, marker="s",
                        edgecolors="none", transform=ccrs.PlateCarree(),
                        zorder=4)
        ax.set_title(f"({chr(97 + list(SEASONS).index(sname))}) {sname}",
                     fontsize=9, loc="left", fontweight="bold", pad=4)
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.02,
                      pad=0.22, aspect=40)
    cb.set_label("Observed Es occurrence rate (fraction of profiles)", fontsize=8)
    cb.ax.xaxis.labelpad = 8
    cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(OUT, "fig25_seasonal_occurrence.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig25_seasonal_occurrence.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig25 done")

# ================= FIG 26: detection agreement =================
def fig26():
    ps.setup()
    v = pd.read_csv(os.path.join(VAL, "sounding_matches.csv")).dropna(
        subset=["p_mean"])
    v["ro_pos"] = v["ro_any"].astype(int)   # raw RO label: any matched profile flagged
    y = v["es_flag"].values; r = v["ro_pos"].values
    agree = (y == r)
    kappa = cohen_kappa_score(y, r)
    tn = int(((y == 0) & (r == 0)).sum()); fp = int(((y == 0) & (r == 1)).sum())
    fn = int(((y == 1) & (r == 0)).sum()); tp = int(((y == 1) & (r == 1)).sum())
    print("fig26 agreement", round(agree.mean(), 4), "kappa", round(kappa, 3),
          "table", tn, fp, fn, tp)

    v["mon"] = (pd.to_datetime(v["year"].astype(str), format="%Y")
                + pd.to_timedelta(v["doy"] - 1, unit="D")).dt.month
    mag = v.assign(ag=agree.astype(int)).groupby("mon").ag.mean()
    stag = v.assign(ag=agree.astype(int)).groupby("station").ag.mean()

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))

    ax = axes[0]
    ax.plot(mag.index, mag.values, "-o", color="#0072B2", ms=4, lw=1.3)
    ax.set_xlabel("Month"); ax.set_ylabel("Agreement rate")
    ax.set_xticks(range(1, 13))
    ax.set_title("(a) Monthly agreement", fontsize=9, loc="left",
                 fontweight="bold")
    ax.grid(axis="y"); ax.set_ylim(0.5, 0.8)

    ax = axes[1]
    order = stag.sort_values(ascending=False).index
    names = {"BLJ03": "Belem", "SAA0K": "Sao Luis", "FZA0M": "Fortaleza",
             "BVJ03": "Boa Vista", "CGK21": "Campo Grande",
             "CAJ2M": "Cachoeira Paul.", "SMK29": "Santa Maria"}
    xpos = np.arange(len(order))
    ax.bar(xpos, stag[order].values, color="#009E73", width=0.65)
    for xi, v_ in zip(xpos, stag[order].values):
        ax.text(xi, v_ + 0.005, f"{v_:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xpos)
    ax.set_xticklabels([names[s] for s in order], rotation=30, ha="right",
                       fontsize=7)
    ax.set_ylabel("Agreement rate")
    ax.set_title("(b) Per-station agreement", fontsize=9, loc="left",
                 fontweight="bold")
    ax.grid(axis="y"); ax.set_ylim(0.55, 0.8)

    ax = axes[2]
    ax.axis("off")
    M = np.array([[tn, fp], [fn, tp]])
    rowlab = ["No $E_s$ observed", "$E_s$ observed"]
    collab = ["Raw RO label negative", "Raw RO label positive"]
    tab = ax.table(cellText=[[f"{M[i,j]:,}\n({M[i,j]/M[i].sum()*100:.1f}%)"
                              for j in range(2)] for i in range(2)],
                   rowLabels=rowlab, colLabels=collab, cellLoc="center",
                   loc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 2.0)
    for (i, j), cell in tab.get_celld().items():
        if i == 0 or j == -1:
            cell.set_facecolor("#F2EFE6")
    ax.set_title("(c) $2\\times2$ contingency table", fontsize=9,
                 loc="left", fontweight="bold")
    ax.text(0.5, 0.02,
            f"Overall agreement = {agree.mean():.2f},  "
            f"Cohen's $\\kappa$ = {kappa:.2f}",
            transform=ax.transAxes, ha="center", fontsize=8)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.16,
                        wspace=0.35)
    fig.savefig(os.path.join(OUT, "fig26_detection_agreement.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig26_detection_agreement.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig26 done")

# ================= FIG 27: per-station seasonal cycles =================
def fig27():
    ps.setup()
    v = pd.read_csv(os.path.join(VAL, "sounding_matches.csv")).dropna(
        subset=["p_mean"])
    v["mon"] = (pd.to_datetime(v["year"].astype(str), format="%Y")
                + pd.to_timedelta(v["doy"] - 1, unit="D")).dt.month
    names = {"BLJ03": "Belem", "SAA0K": "Sao Luis", "FZA0M": "Fortaleza",
             "BVJ03": "Boa Vista", "CGK21": "Campo Grande",
             "CAJ2M": "Cachoeira Paulista"}

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.6), sharex=True,
                             sharey=True)
    months = np.arange(1, 13)
    for ax, (st, nm) in zip(axes.flat, names.items()):
        g = v[v["station"] == st].groupby("mon").agg(
            obs=("es_flag", "mean"), pred=("p_mean", "mean"),
            n=("es_flag", "size"))
        obs = g["obs"].reindex(months).values
        pred = g["pred"].reindex(months).values
        ax.plot(months, obs, "-o", color="#000000", ms=3.5, lw=1.4,
                label="Ionosonde")
        ax.plot(months, pred, "s--", color="#0072B2", ms=3.5, lw=1.3,
                label="Predicted")
        ax.set_title(nm, fontsize=8, loc="left")
        ax.set_xticks([1, 4, 7, 10, 12])
        ax.grid(alpha=0.4)
        ax.set_ylim(0, 0.75)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5,1.04), ncol=2,
               frameon=False, fontsize=8)
    fig.text(0.5, 0.015, "Month", ha="center", fontsize=8)
    fig.text(0.01, 0.5, "E$_s$ rate / probability", va="center",
             rotation="vertical", fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.10,
                        wspace=0.12, hspace=0.30)
    fig.savefig(os.path.join(OUT, "fig27_station_seasonal.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig27_station_seasonal.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("fig27 done")

if __name__ == "__main__":
    fig16(); fig24(); fig25(); fig26(); fig27()
    print("all done")
