"""Ionosonde check of the forward-in-time nowcasts (test period 2023-2024).

For each model's test predictions, soundings are matched exactly as in
validate_ionosonde.py (500 km, +/-2 h, >=3 profiles). The raw RO label
fraction of the same profiles is the detector-level reference; note that the
nowcast models never see the matched profiles' own labels or S4/alt.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import validate_ionosonde as vi

P = os.path.join(vi.BASE, "paper_outputs", "persistence")
N_BOOT = 1000


def main():
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "test_predictions.csv"
    tp = pd.read_csv(os.path.join(P, src))
    iono = vi.load_iono(); iono = iono[iono.year >= 2023]
    models = [c[2:] for c in tp.columns if c.startswith("p_")]
    base = tp[["lat", "lon", "year", "doy", "hour", "y_true"]].copy()
    base["n_prof"] = 1
    base["t_abs"] = vi.abs_hour(base.year.values, base.doy.values, base.hour.values)
    cols = {}
    for k in models:
        o = base.copy(); o["p_cal"] = tp[f"p_{k}"].values
        v = vi.match(iono, o).dropna(subset=["p_mean"])
        cols[k] = v.p_mean.values
        y = v.es_flag.values; raw = v.ro_frac.values
    out = {"n_soundings": int(len(y)), "es_rate": float(y.mean()),
           "auc": {k: float(roc_auc_score(y, p)) for k, p in cols.items()}}
    out["auc"]["raw_ro_label"] = float(roc_auc_score(y, raw))
    rng = np.random.default_rng(0); B = {k: [] for k in list(cols) + ["raw_ro_label"]}
    allp = dict(cols, raw_ro_label=raw)
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        for k, p in allp.items():
            B[k].append(roc_auc_score(y[i], p[i]))
    out["auc_ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in B.items()}
    json.dump(out, open(os.path.join(P, "ionosonde_check" + ("" if src == "test_predictions.csv" else "_refined") + ".json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
