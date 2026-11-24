"""HWM14 neutral-wind features at sporadic-E heights for every RO profile.

HWM14 (Drob et al., 2015), quiet-time (DWM storm component off, ap = -1),
evaluated at the profile's location and UT at 100, 110, 120 and 130 km.
Features: zonal (u) and meridional (v) wind at 110 km and the vertical shears
du/dz, dv/dz centred at 105, 115 and 125 km (m/s per km). Es forms where the
zonal wind shear is negative (wind-shear theory), so these are the physically
expected predictors. HWM14 depends only on position, day of year and UT, so it
is a physically structured climatology, not an independent observation.

Requires the compiled driver hwm14/hwm14drv (built from hwm14/hwm14.f90 and
hwm14/driver.f90; coefficient files hwm123114.bin, dwm07b104i.dat, gd2qd.dat).
Output: paper_outputs/physics_drivers/hwm_features.parquet (row order of
features_refined.parquet).
"""
import os, subprocess, tempfile
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
HDIR = os.path.join(BASE, "hwm14")
SRC = os.path.join(BASE, "paper_outputs", "persistence", "features_refined.parquet")
OUT = os.path.join(BASE, "paper_outputs", "physics_drivers", "hwm_features.parquet")
ALTS = [100, 110, 120, 130]
HWM = ["hwm_u110", "hwm_v110", "hwm_dudz105", "hwm_dudz115", "hwm_dudz125",
       "hwm_dvdz105", "hwm_dvdz115", "hwm_dvdz125"]


def run_hwm(iyd, sec, alt, lat, lon, stl):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for r in zip(iyd, sec, np.full(len(iyd), alt), lat, lon, stl):
            f.write("%d %.0f %.1f %.3f %.3f %.4f 150 150 -1\n" % r)
        name = f.name
    out = subprocess.run([os.path.join(HDIR, "hwm14drv")], stdin=open(name), cwd=HDIR,
                         capture_output=True, text=True, check=True).stdout
    os.remove(name)
    w = np.loadtxt(out.splitlines())
    return w[:, 1], w[:, 0]          # zonal, meridional


def main():
    d = pd.read_parquet(SRC, columns=["year", "doy", "hour", "minutes", "lat", "lon", "t_h"])
    iyd = (d.year.values % 100) * 1000 + d.doy.values
    sec = d.hour.values * 3600.0 + d.minutes.values * 60.0
    stl = np.mod(sec / 3600.0 + d.lon.values / 15.0, 24.0)
    U, V = {}, {}
    for a in ALTS:
        U[a], V[a] = run_hwm(iyd, sec, a, d.lat.values, d.lon.values, stl)
        print("alt", a, "done", flush=True)
    f = pd.DataFrame({"t_h": d.t_h.values, "lat": d.lat.values, "lon": d.lon.values,
                      "hwm_u110": U[110], "hwm_v110": V[110]})
    for lo, hi in [(100, 110), (110, 120), (120, 130)]:
        c = (lo + hi) // 2
        f[f"hwm_dudz{c}"] = (U[hi] - U[lo]) / (hi - lo)
        f[f"hwm_dvdz{c}"] = (V[hi] - V[lo]) / (hi - lo)
    f.to_parquet(OUT)
    print(f[HWM].describe().T)


if __name__ == "__main__":
    main()
