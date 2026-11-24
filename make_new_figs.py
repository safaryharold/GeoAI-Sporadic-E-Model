"""Figures for the Es lifetime and scale analysis, the decision-value analysis
and the expanded ionosonde validation. All values are read from
paper_outputs/{es_scales,decision_value,validation_giro}/results.json."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as ps

BASE = os.path.dirname(os.path.abspath(__file__))
PO = os.path.join(BASE, "paper_outputs")
OUT = os.path.join(BASE, "paper", "figures")
C = {"nowcast": "#0072B2", "drivers": "#CC79A7", "loc_time": "#999999", "raw": "#555555",
     "full": "#009E73", "res": "#0072B2", "rawcorr": "#999999", "new": "#D55E00"}


def fig_scales():
    ps.setup()
    r = json.load(open(os.path.join(PO, "es_scales", "results.json")))
    g = r["groups"]; xt = np.array(r["x_time_h"]); xd = np.array(r["x_dist_km"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1.15]})
    a = g["all"]
    ax = axes[0]
    ax.plot(xt, a["time"]["raw"], "s", color=C["rawcorr"], ms=3.5, label="Raw labels")
    ax.plot(xt, a["time"]["res"], "o", color=C["res"], ms=4, label="Minus climatology")
    xx = np.linspace(0, 24, 200)
    ax.plot(xx, a["A_t"] * np.exp(-xx / a["tau_h"]), "-", color=C["res"], lw=1,
            label=rf"Fit, $\tau$ = {a['tau_h']:.1f} h")
    ax.set_xlabel("Time lag (h), pairs within 250 km"); ax.set_ylabel("Correlation of E$_s$ labels")
    ax.set_ylim(0, None); ax.legend(fontsize=6.3, frameon=False); ps.panel_label(ax, "(a)")
    ax = axes[1]
    ax.plot(xd, a["dist"]["raw"], "s", color=C["rawcorr"], ms=3.5, label="Raw labels")
    ax.plot(xd, a["dist"]["res"], "o", color=C["res"], ms=4, label="Minus climatology")
    xx = np.linspace(0, 1000, 200)
    ax.plot(xx, a["A_d"] * np.exp(-xx / a["L_km"]), "-", color=C["res"], lw=1,
            label=rf"Fit, $L$ = {a['L_km']:.0f} km")
    ax.set_xlabel("Distance (km), pairs within 1 h"); ax.set_ylim(0, None)
    ax.legend(fontsize=6.3, frameon=False); ps.panel_label(ax, "(b)")
    ax = axes[2]
    keys = [("band_equatorial", "|MLAT| < 10°"), ("band_low", "10° to 20°"), ("band_mid", "> 20°"),
            ("season_DJF", "DJF"), ("season_MAM", "MAM"), ("season_JJA", "JJA"), ("season_SON", "SON")]
    y = np.arange(len(keys))[::-1]
    ax2 = ax.twiny()
    for yi, (k, lab) in zip(y, keys):
        v = g[k]
        ax.errorbar(v["tau_h"], yi + 0.12, xerr=[[v["tau_h"] - v["tau_ci95"][0]], [v["tau_ci95"][1] - v["tau_h"]]],
                    fmt="o", color=C["res"], ms=3.5, capsize=1.5)
        ax2.errorbar(v["L_km"], yi - 0.12, xerr=[[v["L_km"] - v["L_ci95"][0]], [v["L_ci95"][1] - v["L_km"]]],
                     fmt="s", color=C["new"], ms=3.2, capsize=1.5)
    ax.set_yticks(y); ax.set_yticklabels([k[1] for k in keys], fontsize=6.5)
    ax.axhline(3.5, color="#bbbbbb", lw=0.6)
    ax.set_xlabel(r"Lifetime $\tau$ (h, circles)", color=C["res"])
    ax2.set_xlabel(r"Length $L$ (km, squares)", color=C["new"], fontsize=7)
    ax2.tick_params(axis="x", labelsize=6.5, colors=C["new"]); ax.tick_params(axis="x", colors=C["res"])
    ax2.spines["top"].set_visible(True); ax2.grid(False)
    ax.set_title("(c)", loc="left", fontweight="bold", fontsize=9, pad=18)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.17)
    ps.save(fig, "fig31_es_scales", OUT)


def fig_value():
    ps.setup()
    r = json.load(open(os.path.join(PO, "decision_value", "results.json")))
    a = np.array(r["alpha"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"wspace": 0.3, "width_ratios": [1.4, 1]})
    ax = axes[0]
    for m, lab in [("nowcast", "Persistence nowcast"), ("drivers", "Drivers + geometry"), ("loc_time", "Location + time")]:
        v = np.array(r["V"][m]); lo, hi = map(np.array, r["V_ci95"][m])
        ax.plot(a, v, color=C[m], label=lab); ax.fill_between(a, lo, hi, color=C[m], alpha=0.2, lw=0)
    ax.axvline(r["base_rate"], color="#777777", ls=":", lw=0.9)
    ax.text(r["base_rate"] + 0.01, 0.02, "base rate", fontsize=6.3, color="#555555")
    ax.set_xlabel(r"Cost-loss ratio $\alpha = C/L$"); ax.set_ylabel("Relative economic value $V$")
    ax.set_ylim(0, None); ax.set_xlim(0, 0.8); ax.legend(fontsize=6.5, frameon=False); ps.panel_label(ax, "(a)")
    ax = axes[1]
    lo, hi = map(np.array, r["nowcast_minus_drivers_ci95"])
    d = np.array(r["V"]["nowcast"]) - np.array(r["V"]["drivers"])
    ax.plot(a, d, color=C["nowcast"]); ax.fill_between(a, lo, hi, color=C["nowcast"], alpha=0.25, lw=0)
    ax.axhline(0, color="#777777", ls="--", lw=0.8)
    ax.set_xlabel(r"Cost-loss ratio $\alpha$"); ax.set_ylabel(r"$\Delta V$, nowcast minus drivers")
    ax.set_xlim(0, 0.8); ps.panel_label(ax, "(b)")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.17)
    ps.save(fig, "fig32_decision_value", OUT)


def fig_val8():
    ps.setup()
    r = json.load(open(os.path.join(PO, "validation_giro", "results.json")))
    names = {"BLJ03": "Belém", "SAA0K": "São Luís", "FZA0M": "Fortaleza", "BVJ03": "Boa Vista",
             "CGK21": "Campo Grande", "CAJ2M": "Cachoeira P.", "SMK29": "Santa Maria*", "JI91J": "Jicamarca*"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"wspace": 0.55})
    ax = axes[0]; sp = r["spatial_per_station"]; st = sorted(sp, key=lambda s: sp[s]["auc"])
    y = np.arange(len(st))
    ax.plot([sp[s]["auc_raw"] for s in st], y, "s", color=C["raw"], ms=4, label="Raw RO label")
    ax.plot([sp[s]["auc"] for s in st], y, "o", color=C["full"], ms=4.5, label="Spatial-CV model")
    for yi, s in zip(y, st):
        ax.plot([sp[s]["auc_raw"], sp[s]["auc"]], [yi, yi], color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([names[s] for s in st], fontsize=6.8)
    for t, s in zip(ax.get_yticklabels(), st):
        if s in ("SMK29", "JI91J"):
            t.set_color(C["new"])
    ax.set_xlabel("ROC-AUC vs ionosonde (2020 to 2024)")
    ax.legend(fontsize=6.3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2)
    ps.panel_label(ax, "(a)")
    ax = axes[1]; nw = r["nowcast_per_station"]; st = sorted(nw, key=lambda s: nw[s]["nowcast"])
    y = np.arange(len(st))
    for k, mk, lab, col in [("drivers", "^", "Drivers + geometry", C["drivers"]), ("raw_ro_label", "s", "Raw RO label", C["raw"]),
                            ("nowcast", "o", "Persistence nowcast", C["nowcast"])]:
        ax.plot([nw[s][k] for s in st], y, mk, color=col, ms=4.3, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([names[s] for s in st], fontsize=6.8)
    for t, s in zip(ax.get_yticklabels(), st):
        if s in ("SMK29", "JI91J"):
            t.set_color(C["new"])
    ax.set_xlabel("ROC-AUC vs ionosonde (2023 and 2024)")
    ax.legend(fontsize=6.3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.2, columnspacing=0.8)
    ps.panel_label(ax, "(b)")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.26)
    ps.save(fig, "fig33_iono_eight_stations", OUT)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["scales", "value", "val8"]
    for w in which:
        {"scales": fig_scales, "value": fig_value, "val8": fig_val8}[w]()
    print("done")
