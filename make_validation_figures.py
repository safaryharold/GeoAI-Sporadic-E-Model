"""Publication figures for the ionosonde external-validation section."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc

BASE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(BASE, "paper_outputs", "validation")
OUT = os.path.join(BASE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

OI = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
      "purple":"#CC79A7","gray":"#999999","black":"#222222"}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.7, "lines.linewidth": 1.5,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "text.color": "#222222", "xtick.color": "#444444", "ytick.color": "#444444",
    "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans",
})
sns.set_style("whitegrid")
for s in ["top","right"]:
    plt.rcParams[f"axes.spines.{s}"] = False

def save(fig, name):
    fig.savefig(os.path.join(OUT, name+".png"))
    fig.savefig(os.path.join(OUT, name+".pdf"))
    plt.close(fig)

v = pd.read_csv(os.path.join(VAL, "sounding_matches.csv")).dropna(subset=["p_mean"])
metrics = json.load(open(os.path.join(VAL, "metrics.json")))
perf = pd.read_csv(os.path.join(VAL, "per_station.csv"))
rel = pd.read_csv(os.path.join(VAL, "reliability.csv"))
diur = pd.read_csv(os.path.join(VAL, "diurnal.csv"))
mon = pd.read_csv(os.path.join(VAL, "monthly.csv"))
STATIONS = {
    "BLJ03": ("Belem", -1.4563, -48.5013), "SAA0K": ("Sao Luis", -2.5941, -44.2097),
    "FZA0M": ("Fortaleza", -3.7327, -38.5270), "BVJ03": ("Boa Vista", 2.8701, -60.7109),
    "CGK21": ("Campo Grande", -20.4649, -54.6218), "CAJ2M": ("Cachoeira Paulista", -22.7038, -45.0093),
}
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------- fig13: stations + diurnal ----------
ES_CMAP = LinearSegmentedColormap.from_list(
    "es", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])

def es_occurrence_grid():
    """Observed RO Es frequency in 2-degree bins (cached)."""
    cache = os.path.join(VAL, "es_grid.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["lats"], z["lons"], z["rate"]
    oof = pd.read_csv(os.path.join(BASE, "paper_outputs", "oof", "xgb_oof.csv"),
                      usecols=["lat", "lon", "y_true"])
    latb = (oof.lat / 2).round() * 2
    lonb = (oof.lon / 2).round() * 2
    g = oof.assign(la=latb, lo=lonb).groupby(["la", "lo"]).y_true.mean()
    lats = np.sort(g.index.get_level_values(0).unique())
    lons = np.sort(g.index.get_level_values(1).unique())
    rate = np.full((len(lats), len(lons)), np.nan)
    for (la, lo), val in g.items():
        rate[np.searchsorted(lats, la), np.searchsorted(lons, lo)] = val
    np.savez(cache, lats=lats, lons=lons, rate=rate)
    return lats, lons, rate

def fig_stations_diurnal():
    lats, lons, rate = es_occurrence_grid()
    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.30)
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax.set_extent([-86, -29, -36, 16], crs=ccrs.PlateCarree())
    # Es occurrence shading
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    cf = ax.pcolormesh(LON, LAT, rate, cmap=ES_CMAP, vmin=0, vmax=0.4,
                       transform=ccrs.PlateCarree(), zorder=1, rasterized=True)
    ax.add_feature(cfeature.OCEAN, facecolor="#EAF1F7", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#F2EFE6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="#DCE9F5", edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE, lw=0.6, color="#5A5A5A", zorder=3)
    ax.add_feature(cfeature.BORDERS, lw=0.4, color="#8A8A8A", ls=":", zorder=3)
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="#AAAAAA", alpha=0.6,
                      linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 7}; gl.ylabel_style = {"size": 7}
    # stations sized by matched sounding count, coloured by Es rate
    n_map = v.groupby("station").size().to_dict()
    es_map = v.groupby("station").es_flag.mean().to_dict()
    for st, (nm, la, lo) in STATIONS.items():
        n = n_map.get(st, 0)
        esr = es_map.get(st, np.nan)
        ax.plot(lo, la, "o", ms=4 + 7 * np.sqrt(n / 2500),
                color=OI["red"], mec="white", mew=1.0, zorder=5,
                transform=ccrs.PlateCarree())
        # stagger label offsets so nearby stations do not overlap
        _offs=[(-80,-34),(12,14),(14,-26),(10,8),(-95,-14),(10,-14),(-95,8)]
        _oi=_offs[list(STATIONS).index(st)%len(_offs)]
        ax.annotate(f"{nm}\n(n={n}, Es={esr:.2f})", (lo, la),
                    textcoords="offset points", xytext=_oi, fontsize=8.2,
                    color="#1a1a1a",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#CCCCCC", lw=0.4, alpha=0.85), zorder=6)
    cb = fig.colorbar(cf, ax=ax, orientation="vertical", fraction=0.04,
                      pad=0.10, shrink=0.8)
    cb.set_label("Observed E$_s$ frequency", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("(a) Ionosonde validation network and RO E$_s$ climatology",
                 fontsize=10, pad=8)
    # diurnal
    ax = fig.add_subplot(gs[1])
    ax.plot(diur["lt"], diur.obs, "o-", color=OI["blue"], label="Ionosonde E$_s$ occurrence")
    ax.plot(diur["lt"], diur.pred, "s--", color=OI["red"], label="Mean predicted P(E$_s$)")
    ax.plot(diur["lt"], diur.raw, "^:", color=OI["gray"], label="Raw RO label fraction")
    ax.set_xlabel("Local time (h)"); ax.set_ylabel("E$_s$ rate / probability")
    ax.set_title("(b) Diurnal cycle, 2020 to 2024")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 24); ax.grid(alpha=0.4)
    fig.tight_layout(); save(fig, "fig13_iono_stations_diurnal")

# ---------- fig14: ROC + reliability ----------
def fig_roc_rel():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    y = v.es_flag.values; p = v.p_mean.values
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=OI["red"], lw=1.8,
            label=f"XGBoost (AUC = {auc(fpr, tpr):.3f})")
    fr, tr, _ = roc_curve(y, v.ro_frac.values)
    ax.plot(fr, tr, ":", color=OI["gray"], lw=1.6,
            label=f"Raw RO label fraction (AUC = {auc(fr, tr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=OI["gray"], lw=1)
    ax.set_xlabel("Probability of false detection"); ax.set_ylabel("Probability of detection")
    ax.set_title("(a) Sounding-level ROC")
    ax.text(0.04, 0.92, f"n = {len(y):,} soundings\nEs rate = {y.mean():.2f}",
            transform=ax.transAxes, fontsize=7.5, color="#444444",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.5))
    ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax = axes[1]
    r = rel.dropna()
    ax.plot([0, 1], [0, 1], "--", color=OI["gray"], lw=1, label="Perfect")
    ax.plot(r.p_mean, r.obs, "o-", color=OI["red"], ms=5, label="XGBoost")
    ax.set_xlabel("Mean predicted P(E$_s$)"); ax.set_ylabel("Observed E$_s$ frequency")
    ax.set_title("(b) Reliability (sounding level)")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); save(fig, "fig14_iono_roc_reliability")

# ---------- fig15: monthly + per-station ----------
def fig_monthly_station():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    ax.plot(mon.month, mon.obs, "o-", color=OI["blue"], label="Ionosonde E$_s$")
    ax.plot(mon.month, mon.pred, "s--", color=OI["red"], label="Predicted P(E$_s$)")
    ax.plot(mon.month, mon.raw, "^:", color=OI["gray"], label="Raw RO label fraction")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Month"); ax.set_ylabel("E$_s$ rate / probability")
    ax.set_title("(a) Seasonal cycle, 2020 to 2024")
    ax.grid(alpha=0.4)
    ax = axes[1]
    x = np.arange(len(perf))
    w = 0.38
    ax.bar(x - w/2, perf.POD, w, color=OI["green"], label=f"POD (t = {metrics['thr_016']['threshold']:.3f})")
    ax.bar(x + w/2, perf.FAR, w, color=OI["orange"], label="FAR")
    ax.axhline(metrics["thr_016"]["POD"], ls=":", color=OI["green"], lw=1)
    ax.axhline(metrics["thr_016"]["FAR"], ls=":", color=OI["orange"], lw=1)
    ax.set_xticks(x); ax.set_xticklabels(perf["name"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rate"); ax.set_ylim(0, 1)
    ax.set_title("(b) Per-station scores")
    ax.grid(alpha=0.4)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5,-0.02), ncol=2, columnspacing=2.5, frameon=False, fontsize=9)
    fig.tight_layout(); save(fig, "fig15_iono_monthly_station")

fig_stations_diurnal()
fig_roc_rel()
fig_monthly_station()
print("validation figures saved")
