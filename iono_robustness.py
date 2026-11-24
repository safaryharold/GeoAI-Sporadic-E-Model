"""Ionosonde validation robustness: bootstrap CIs + matching-radius sensitivity.

Uses exactly the same soundings, OOF de-duplication and year-aware matching
as validate_ionosonde.py (imported), so the 500 km entry reproduces the
headline validation.
"""
import os, json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import validate_ionosonde as vi

THR = vi.THRESHOLDS["thr_016"]
N_BOOT = 1000


def pod_far(y, p):
    pred = p >= THR
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return tp / max(tp + fn, 1), fp / max(tp + fp, 1)


def main():
    iono = vi.load_iono(); oof = vi.load_oof()
    rng = np.random.default_rng(0)
    res = {"radius": {}, "ci_500km": {}}
    for R in [200.0, 350.0, 500.0, 750.0, 1000.0]:
        v = vi.match(iono, oof, radius_km=R).dropna(subset=["p_mean"])
        y, p, r = v.es_flag.values, v.p_mean.values, v.ro_frac.values
        pod, far = pod_far(y, p)
        res["radius"][str(int(R))] = dict(n=len(v), es_rate=float(y.mean()),
                                          roc=float(roc_auc_score(y, p)),
                                          roc_raw=float(roc_auc_score(y, r)),
                                          pra=float(average_precision_score(y, p)), POD=pod, FAR=far)
        print(R, res["radius"][str(int(R))], flush=True)
        if R == 500.0:
            B = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(y), len(y))
                a_m = roc_auc_score(y[i], p[i]); a_r = roc_auc_score(y[i], r[i])
                B.append([a_m, a_r, a_m - a_r, *pod_far(y[i], p[i])])
            B = np.array(B)
            q = lambda k: [float(np.percentile(B[:, k], 2.5)), float(np.percentile(B[:, k], 97.5))]
            res["ci_500km"] = dict(n_boot=N_BOOT, roc_auc=q(0), roc_auc_raw=q(1),
                                   roc_auc_model_minus_raw=q(2), POD=q(3), FAR=q(4))
    with open(os.path.join(vi.OUT_DIR, "robustness.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
