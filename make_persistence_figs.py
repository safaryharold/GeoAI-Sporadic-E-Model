"""Figures for the forward-in-time persistence nowcast (all values read from
paper_outputs/persistence/*.json)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as ps

BASE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(BASE, "paper_outputs", "persistence")
OUT = os.path.join(BASE, "paper", "figures")
COL = {"loc_time": "#999999", "drivers": "#CC79A7", "refined": "#0072B2", "nowcast": "#0072B2",
       "full_upper": "#009E73", "raw_ro_label": "#555555"}
LAB = {"loc_time": "Location + time", "drivers": "Drivers + geometry",
       "refined": "Persistence nowcast", "full_upper": "Full model (upper bound)"}


def fig_rolling():
    ps.setup()
    r = json.load(open(os.path.join(P, "robustness_results.json")))["rolling"]
    periods = list(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.35})
    ax = axes[0]; x = np.arange(len(periods)); off = {"loc_time": -0.21, "drivers": -0.07, "refined": 0.07, "full_upper": 0.21}
    for k in ["loc_time", "drivers", "refined", "full_upper"]:
        ax.plot(x + off[k], [r[p]["auc"][k] for p in periods], "o", color=COL[k], ms=6, label=LAB[k])
    ax.set_xticks(x); ax.set_xticklabels(periods); ax.set_xlabel("Test period (trained on earlier years only)")
    ax.set_ylabel("ROC-AUC (RO label)"); ax.set_ylim(0.70, 0.94); ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left", numpoints=1); ps.panel_label(ax, "(a)")
    ax = axes[1]
    for j, (k, c, lab) in enumerate([("refined_minus_drivers_ci95", COL["refined"], "Persistence minus drivers"),
                                     ("drivers_minus_loctime_ci95", COL["drivers"], "Drivers minus location and time")]):
        lo = np.array([r[p][k][0] for p in periods]); hi = np.array([r[p][k][1] for p in periods])
        mid = (lo + hi) / 2; xx = x + (j - 0.5) * 0.25
        ax.errorbar(xx, mid, yerr=[mid - lo, hi - mid], fmt="o", color=c, ms=4, capsize=2, label=lab)
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=7); ax.set_ylabel("Δ ROC-AUC (95% CI)")
    ax.grid(axis="y", alpha=0.35); ax.legend(fontsize=6.5, frameon=False, loc="center right", numpoints=1); ps.panel_label(ax, "(b)")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.17)
    ps.save(fig, "fig29_rolling_origin", OUT)


def fig_latency():
    ps.setup()
    lat = json.load(open(os.path.join(P, "robustness_results.json")))["latency"]
    fe = json.load(open(os.path.join(P, "final_eval.json")))
    L = sorted(lat, key=float); xs = [float(l) for l in L]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.33})
    ax = axes[0]
    ax.plot(xs, [lat[l]["auc"] for l in L], "-o", color=COL["nowcast"], ms=5, label="Persistence nowcast")
    ax.axhline(fe["metrics"]["drivers"]["roc_auc"], color=COL["drivers"], ls="--", lw=1.1, label="Drivers + geometry")
    ax.axhline(fe["metrics"]["loc_time"]["roc_auc"], color=COL["loc_time"], ls=":", lw=1.1, label="Location + time")
    ax.set_xlabel("Data latency (h)"); ax.set_ylabel("ROC-AUC (RO label, 2023 and 2024)")
    ax.set_xticks(xs); ax.grid(alpha=0.35); ax.legend(fontsize=6.5, frameon=False); ps.panel_label(ax, "(a)")
    ax = axes[1]
    ax.plot(xs, [lat[l]["iono_auc"] for l in L], "-o", color=COL["nowcast"], ms=5, label="Persistence nowcast")
    io = fe["ionosonde"]["auc"]
    ax.axhline(io["raw_ro_label"], color=COL["raw_ro_label"], ls="-.", lw=1.1, label="Raw RO label (current)")
    ax.axhline(io["full_upper"], color=COL["full_upper"], ls="--", lw=1.1, label="Full model (current S4/alt)")
    ax.axhline(io["drivers"], color=COL["drivers"], ls="--", lw=1.1, label="Drivers + geometry")
    ax.set_xlabel("Data latency (h)"); ax.set_ylabel("ROC-AUC vs ionosonde (2023 and 2024)")
    ax.set_ylim(io["drivers"] - 0.012, None)
    ax.set_xticks(xs); ax.grid(alpha=0.35); ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2); ps.panel_label(ax, "(b)")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.16)
    ps.save(fig, "fig30_latency", OUT)


if __name__ == "__main__":
    fig_rolling(); fig_latency(); print("done")
