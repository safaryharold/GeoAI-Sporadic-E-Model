"""Expanded independent validation with additional GIRO ionosondes.

New stations inside the study domain (GIRO DIDBase, getbest scaled values):
  SMK29 Santa Maria (2020-2023), JI91J Jicamarca (2020-2022).
Soundings are sampled exactly like the EMBRACE set (nearest record to
00, 03, ..., 21 UT within 1 h); Es flag = foEs, fbEs or fminEs present.
Matching (500 km, +/- 2 h, >= 3 RO profiles) is the unchanged
validate_ionosonde.match. Nothing is tuned here.

Steps
  1. Cross-check GIRO Es flags against the EMBRACE .SAO flags where both
     exist (BLJ03 2020 and 2023, SMK29 2023).
  2. Spatial-CV model vs raw RO label, 2020-2024: original 6, new 2, all 8.
  3. Forward-in-time nowcast (final_test_predictions.csv), 2023-2024:
     original 6, Santa Maria 2023, all 7.
Outputs: paper_outputs/validation_giro/*.json|csv
"""
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score

import validate_ionosonde as vi

BASE = os.path.dirname(os.path.abspath(__file__))
GDIR = os.path.join(BASE, "ionosonde_data", "giro")
OUT = os.path.join(BASE, "paper_outputs", "validation_giro")
os.makedirs(OUT, exist_ok=True)
NEW = {"SMK29": ("Santa Maria", -29.73, -53.71), "JI91J": ("Jicamarca", -12.00, -76.80)}
TARGET = np.arange(0, 24, 3)
N_BOOT = 1000


def parse_giro(st):
    rows = []
    for f in sorted(glob.glob(os.path.join(GDIR, st, "*.txt"))):
        for line in open(f):
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 10:
                continue
            es = any(p[i] != "---" for i in (2, 4, 6))
            rows.append((p[0], int(p[1]), int(es), np.nan if p[2] == "---" else float(p[2])))
    d = pd.DataFrame(rows, columns=["time", "cs", "es_flag", "foEs"])
    d["time"] = pd.to_datetime(d.time.str[:19])
    d = d.drop_duplicates("time").sort_values("time")
    d["date"] = d.time.dt.normalize()
    mins = d.time.dt.hour * 60 + d.time.dt.minute
    picks = []
    for h in TARGET:
        dist = (mins - h * 60).abs()
        sel = d.assign(dist=dist)[dist <= 60]
        picks.append(sel.loc[sel.groupby("date").dist.idxmin()])
    s = pd.concat(picks).drop_duplicates("time")
    out = pd.DataFrame(dict(station=st, year=s.time.dt.year, doy=s.time.dt.dayofyear,
                            hour=s.time.dt.hour, es_flag=s.es_flag, foEs=s.foEs, cs=s.cs))
    out["t_abs"] = vi.abs_hour(out.year.values, out.doy.values, out.hour.values)
    return out.reset_index(drop=True)


def embrace_flags(st, years):
    from parse_sao import parse_sao
    rows = []
    for y in years:
        for path in glob.glob(os.path.join(BASE, "ionosonde_data", st, "*", f"{st}_{y}*.SAO")) + \
                glob.glob(os.path.join(BASE, "ionosonde_data", st, "*", f"{st}_{y}*_SAO.XML")):
            n = os.path.basename(path); r = parse_sao(path)
            if r is None:
                continue
            es = any(r.get(k) is not None for k in ("foEs", "fminEs", "fbEs", "ftEs"))
            rows.append(dict(year=y, doy=int(n[10:13]), hour=int(n[13:15]), es_embrace=int(es)))
    return pd.DataFrame(rows).drop_duplicates(["year", "doy", "hour"])


def boot(y, P, pairs, seed=0):
    rng = np.random.default_rng(seed); B = {k: [] for k in P}; D = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a = {k: roc_auc_score(y[i], p[i]) for k, p in P.items()}
        for k in P:
            B[k].append(a[k])
        for x, c in pairs:
            D[f"{x}-{c}"].append(a[x] - a[c])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: q(v) for k, v in B.items()}, {k: q(v) for k, v in D.items()}


def evaluate(v, cols, pairs):
    y = v.es_flag.values; P = {k: v[c].values for k, c in cols.items()}
    ci, dci = boot(y, P, pairs)
    return dict(n=int(len(y)), es_rate=float(y.mean()),
                auc={k: float(roc_auc_score(y, p)) for k, p in P.items()}, auc_ci95=ci, diff_ci95=dci)


def main():
    res = {}
    giro = {st: parse_giro(st) for st in ["SMK29", "JI91J", "BLJ03"]}
    for st, g in giro.items():
        print(st, len(g), g.groupby("year").size().to_dict(), "manual-scaled frac %.3f" % (g.cs == 999).mean())
    res["giro_soundings"] = {st: g.groupby("year").size().to_dict() for st, g in giro.items()}

    # 1. cross-check against EMBRACE .SAO
    xc = {}
    for st, yrs in [("BLJ03", [2020, 2023]), ("SMK29", [2023])]:
        m = giro[st].merge(embrace_flags(st, yrs), on=["year", "doy", "hour"])
        if len(m):
            xc[st] = dict(n=int(len(m)), agreement=float((m.es_flag == m.es_embrace).mean()),
                          kappa=float(cohen_kappa_score(m.es_flag, m.es_embrace)),
                          giro_rate=float(m.es_flag.mean()), embrace_rate=float(m.es_embrace.mean()))
    res["crosscheck_vs_embrace"] = xc
    print("cross-check:", json.dumps(xc, indent=1))

    stations = dict(vi.STATIONS); stations.update(NEW)
    vi.STATIONS = stations                       # match() iterates over this dict
    iono6 = vi.load_iono()
    newio = pd.concat([giro["SMK29"], giro["JI91J"]], ignore_index=True)
    newio = newio[newio.year.between(2020, 2024)]
    iono = pd.concat([iono6, newio], ignore_index=True)

    # 2. spatial-CV model, 2020-2024
    oof = vi.load_oof()
    m = vi.match(iono, oof).dropna(subset=["p_mean"])
    m.to_csv(os.path.join(OUT, "spatial_matches.csv"), index=False)
    cols = {"spatial_model": "p_mean", "raw_ro_label": "ro_frac"}
    pairs = [("spatial_model", "raw_ro_label")]
    res["spatial"] = {name: evaluate(m[sel], cols, pairs) for name, sel in
                      [("original6", ~m.station.isin(NEW)), ("new2", m.station.isin(NEW)), ("all8", m.station == m.station)]}
    res["spatial_per_station"] = {st: dict(n=int(len(g)), es_rate=float(g.es_flag.mean()),
                                           auc=float(roc_auc_score(g.es_flag, g.p_mean)),
                                           auc_raw=float(roc_auc_score(g.es_flag, g.ro_frac)))
                                  for st, g in m.groupby("station")}

    # 3. forward-in-time nowcast, 2023-2024
    fp = pd.read_csv(os.path.join(BASE, "paper_outputs", "persistence", "final_test_predictions.csv"))
    fp["n_prof"] = 1; fp["t_abs"] = vi.abs_hour(fp.year.values, fp.doy.values, fp.hour.values)
    io23 = iono[iono.year >= 2023]; merged = None
    for k in ["nowcast", "drivers", "full_upper", "loc_time"]:
        o = fp[["lat", "lon", "year", "doy", "hour", "y_true", "n_prof", "t_abs"]].copy(); o["p_cal"] = fp[f"p_{k}"]
        v = vi.match(io23, o).dropna(subset=["p_mean"]).rename(columns={"p_mean": k})
        key = ["station", "year", "doy", "hour"]
        merged = v[key + ["es_flag", "ro_frac", k]] if merged is None else merged.merge(v[key + [k]], on=key)
    merged.to_csv(os.path.join(OUT, "nowcast_matches.csv"), index=False)
    cols = {"nowcast": "nowcast", "raw_ro_label": "ro_frac", "full_upper": "full_upper",
            "drivers": "drivers", "loc_time": "loc_time"}
    pairs = [("nowcast", "raw_ro_label"), ("nowcast", "full_upper"), ("nowcast", "drivers")]
    res["nowcast"] = {name: evaluate(merged[sel], cols, pairs) for name, sel in
                      [("original6", ~merged.station.isin(NEW)), ("santa_maria_2023", merged.station == "SMK29"),
                       ("all7", merged.station == merged.station)]}
    res["nowcast_per_station"] = {st: {k: float(roc_auc_score(g.es_flag, g[c])) for k, c in cols.items()}
                                  | dict(n=int(len(g)), es_rate=float(g.es_flag.mean()))
                                  for st, g in merged.groupby("station")}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    for s in ("spatial", "nowcast"):
        for name, r in res[s].items():
            print(s, name, r["n"], {k: round(v, 3) for k, v in r["auc"].items()}, r["diff_ci95"])


if __name__ == "__main__":
    main()
