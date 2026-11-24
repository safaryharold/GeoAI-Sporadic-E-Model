"""External validation: compare XGBoost OOF Es probabilities with
EMBRACE/INPE digisonde (ionosonde) Es detections over South America,
2020-2024 pooled, six stations with complete coverage.

For each sounding we collect RO profiles within RADIUS_KM and +/- TIME_WINDOW_H
of the sounding epoch (same calendar year, absolute time), and score the
mean calibrated OOF probability against the autoscaled Es flag
(foEs, fminEs, fbEs or ftEs present).

Each RO profile appears once per CV seed in the OOF file; the two seed
predictions are averaged so every profile is counted exactly once.
As a detector-level reference we also report the same metrics for the
raw RO label (fraction / any of matched profiles flagged isSporE=1).

Usage: python validate_ionosonde.py [--reparse]
"""
import os, sys, glob, re, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score

BASE = os.path.dirname(os.path.abspath(__file__))
IONO_DIR = os.path.join(BASE, "ionosonde_data")
OOF_PATH = os.path.join(BASE, "paper_outputs", "oof", "xgb_oof.csv")
OUT_DIR = os.path.join(BASE, "paper_outputs", "validation")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE = os.path.join(OUT_DIR, "ionosonde_es.csv")

YEARS = list(range(2020, 2025))
STATIONS = {
    "BLJ03": ("Belem", -1.4563, -48.5013),
    "SAA0K": ("Sao Luis", -2.5941, -44.2097),
    "FZA0M": ("Fortaleza", -3.7327, -38.5270),
    "BVJ03": ("Boa Vista", 2.8701, -60.7109),
    "CGK21": ("Campo Grande", -20.4649, -54.6218),
    "CAJ2M": ("Cachoeira Paulista", -22.7038, -45.0093),
}  # Santa Maria (SMK29) excluded: incomplete 2020-2024 coverage
RADIUS_KM = 500.0
TIME_WINDOW_H = 2
MIN_PROFILES = 3          # distinct RO profiles
THRESHOLDS = {"thr_011": 0.113, "thr_016": 0.163}
EARTH = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(a))


def abs_hour(year, doy, hour):
    """Hours since 2000-01-01 (absolute, crosses year boundaries correctly)."""
    t = pd.to_datetime(pd.Series(year).astype(int).astype(str), format="%Y") \
        + pd.to_timedelta(pd.Series(doy).astype(int) - 1, unit="D")
    return ((t - pd.Timestamp("2000-01-01")).dt.days.values * 24 + np.asarray(hour).astype(int))


def parse_all_sao():
    from parse_sao import parse_sao
    rows = []
    for path in glob.glob(os.path.join(IONO_DIR, "*", "*", "*.SAO")) + \
            glob.glob(os.path.join(IONO_DIR, "*", "*", "*_SAO.XML")):
        parts = path.split(os.sep)
        st = parts[-3]
        m = re.search(r"_(\d{4})(\d{3})", parts[-1])
        if m is None:
            continue
        year, doy = int(m.group(1)), int(m.group(2))
        r = parse_sao(path)
        if r is None:
            continue
        es = any(r.get(k) is not None for k in ("foEs", "fminEs", "fbEs", "ftEs"))
        rows.append(dict(station=st, year=year, doy=doy, hour=int(parts[-1][13:15]),
                         es_flag=int(es), foF2=r["foF2"], foEs=r["foEs"],
                         fminEs=r["fminEs"], fbEs=r["fbEs"], type_es=r["type_es"]))
    return pd.DataFrame(rows)


def load_iono(reparse=False):
    iono = parse_all_sao() if (reparse or not os.path.exists(CACHE)) else pd.read_csv(CACHE)
    iono = iono[iono.year.isin(YEARS) & iono.station.isin(STATIONS)].copy()
    iono.to_csv(CACHE, index=False)
    iono["t_abs"] = abs_hour(iono.year.values, iono.doy.values, iono.hour.values)
    return iono


def load_oof():
    """One row per RO profile: seed predictions averaged."""
    oof = pd.read_csv(OOF_PATH)
    oof = oof[oof.year.isin(YEARS)]
    key = ["lat", "lon", "year", "doy", "hour", "y_true"]
    g = oof.groupby(key, as_index=False).agg(p_cal=("p_cal", "mean"), n=("p_cal", "size"))
    g["n_prof"] = (g["n"] // 2).clip(lower=1)   # identical keys from distinct profiles
    g["t_abs"] = abs_hour(g.year.values, g.doy.values, g.hour.values)
    return g


def match(iono, oof, radius_km=RADIUS_KM, tw=TIME_WINDOW_H, minp=MIN_PROFILES):
    recs = []
    for st, (name, la, lo) in STATIONS.items():
        sub = iono[iono.station == st]
        cand = oof[haversine_km(la, lo, oof.lat.values, oof.lon.values) <= radius_km]
        order = np.argsort(cand.t_abs.values)
        ct = cand.t_abs.values[order]; cp = cand.p_cal.values[order]
        cy = cand.y_true.values[order]; cn = cand.n_prof.values[order]
        lo_i = np.searchsorted(ct, sub.t_abs.values - tw, side="left")
        hi_i = np.searchsorted(ct, sub.t_abs.values + tw, side="right")
        for (_, s), a, b in zip(sub.iterrows(), lo_i, hi_i):
            n = int(cn[a:b].sum())
            ok = n >= minp
            w = cn[a:b]
            recs.append(dict(station=st, station_name=name, year=int(s.year), doy=int(s.doy),
                             hour=int(s.hour), es_flag=int(s.es_flag), foEs=s.get("foEs"),
                             foF2=s.get("foF2"), n_ro=n,
                             p_mean=float(np.average(cp[a:b], weights=w)) if ok else np.nan,
                             p_max=float(cp[a:b].max()) if ok else np.nan,
                             ro_frac=float(np.average(cy[a:b], weights=w)) if ok else np.nan,
                             ro_any=float(cy[a:b].max()) if ok else np.nan))
    return pd.DataFrame(recs)


def scores(y, p):
    out = dict(n=int(len(y)), es_rate=float(np.mean(y)),
               roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)))
    for name, thr in THRESHOLDS.items():
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        out[name] = dict(threshold=thr, POD=tp / max(tp + fn, 1), FAR=fp / max(tp + fp, 1),
                         bias=(tp + fp) / max(tp + fn, 1), CSI=tp / max(tp + fp + fn, 1))
    return out


def main():
    iono = load_iono("--reparse" in sys.argv)
    print("soundings:", len(iono), iono.groupby("year").size().to_dict(), flush=True)
    oof = load_oof()
    print("unique RO profiles 2020-2024:", int(oof.n_prof.sum()), flush=True)
    val = match(iono, oof)
    val.to_csv(os.path.join(OUT_DIR, "sounding_matches.csv"), index=False)
    v = val.dropna(subset=["p_mean"]).copy()
    y = v.es_flag.values

    out = scores(y, v.p_mean.values)
    out["n_soundings_total"] = int(len(val))
    out["matched_frac"] = len(v) / len(val)
    out["n_ro_median"] = float(v.n_ro.median()); out["n_ro_mean"] = float(v.n_ro.mean())
    out["raw_ro_label_frac"] = scores(y, v.ro_frac.values)
    agree = (v.ro_any.values == y)
    out["raw_ro_any_agreement"] = dict(
        agreement=float(agree.mean()), kappa=float(cohen_kappa_score(y, v.ro_any.astype(int))),
        ro_pos_rate=float(v.ro_any.mean()), iono_rate=float(y.mean()),
        tn=int(((y == 0) & (v.ro_any == 0)).sum()), fp=int(((y == 0) & (v.ro_any == 1)).sum()),
        fn=int(((y == 1) & (v.ro_any == 0)).sum()), tp=int(((y == 1) & (v.ro_any == 1)).sum()))
    out["per_year"] = {int(yr): dict(n=len(g), es_rate=float(g.es_flag.mean()),
                                     roc_auc=float(roc_auc_score(g.es_flag, g.p_mean)),
                                     roc_auc_raw=float(roc_auc_score(g.es_flag, g.ro_frac)))
                       for yr, g in v.groupby("year")}

    per = []
    for st, g in v.groupby("station"):
        yy, pp = g.es_flag.values, g.p_mean.values
        pred = (pp >= THRESHOLDS["thr_016"]).astype(int)
        tp = int(((pred == 1) & (yy == 1)).sum()); fp = int(((pred == 1) & (yy == 0)).sum())
        fn = int(((pred == 0) & (yy == 1)).sum())
        per.append(dict(station=st, name=g.station_name.iloc[0], n=len(g), es_rate=float(yy.mean()),
                        roc_auc=float(roc_auc_score(yy, pp)),
                        roc_auc_raw=float(roc_auc_score(yy, g.ro_frac.values)),
                        POD=tp / max(tp + fn, 1), FAR=fp / max(tp + fp, 1),
                        pred_rate=float(pred.mean()),
                        agreement=float((g.ro_any.values == yy).mean())))
    pd.DataFrame(per).to_csv(os.path.join(OUT_DIR, "per_station.csv"), index=False)

    v["bin"] = pd.cut(v.p_mean, bins=np.linspace(0, 1, 11))
    v.groupby("bin", observed=True).agg(p_mean=("p_mean", "mean"), obs=("es_flag", "mean"),
                                        n=("es_flag", "size")).reset_index() \
        .to_csv(os.path.join(OUT_DIR, "reliability.csv"), index=False)

    lon_map = {s: STATIONS[s][2] for s in STATIONS}
    v["local_time"] = (v.hour + v.station.map(lon_map) / 15.0) % 24.0
    v["lt_bin"] = pd.cut(v["local_time"], bins=np.arange(0, 25, 2))
    v.groupby("lt_bin", observed=True).agg(lt=("local_time", "mean"), obs=("es_flag", "mean"),
                                           pred=("p_mean", "mean"), raw=("ro_frac", "mean"),
                                           n=("es_flag", "size")).reset_index() \
        .to_csv(os.path.join(OUT_DIR, "diurnal.csv"), index=False)

    v["month"] = (pd.to_datetime(v.year.astype(str), format="%Y")
                  + pd.to_timedelta(v.doy - 1, unit="D")).dt.month
    v.groupby("month").agg(obs=("es_flag", "mean"), pred=("p_mean", "mean"),
                           raw=("ro_frac", "mean"), n=("es_flag", "size")).reset_index() \
        .to_csv(os.path.join(OUT_DIR, "monthly.csv"), index=False)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(pd.DataFrame(per).to_string(index=False))


if __name__ == "__main__":
    main()
