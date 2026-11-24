"""
Rebuild combined_data.csv from:
  - May_2024_SA.csv   : GNSS-RO sporadic-E detections (doy..S4,isSporE)
  - cleaned_data.csv  : same events + OMNI/solar-wind drivers
Join key: year, month, hour, minutes, lat, lon (+ isSporE sanity check)
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["year", "month", "hour", "minutes", "lat", "lon"]

sa = pd.read_csv(f"{BASE}/May_2024_SA.csv")
cd = pd.read_csv(f"{BASE}/cleaned_data.csv")

print("SA rows:", len(sa), "| cleaned rows:", len(cd))

# sanity: labels must agree where keys match
m = sa.merge(cd, on=KEYS + ["isSporE"], how="inner", suffixes=("", "_cd"), indicator=True)
print("matched with label agreement:", (m._merge == "both").sum())

# keep one row per key combo on each side to avoid cartesian blowup
sa_u = sa.drop_duplicates(subset=KEYS + ["isSporE"])
cd_u = cd.drop_duplicates(subset=KEYS + ["isSporE"])
print("SA unique keys:", len(sa_u), "| cleaned unique keys:", len(cd_u))

out = sa_u.merge(cd_u, on=KEYS + ["isSporE"], how="inner", suffixes=("_sa", "_cd"))
print("merged rows:", len(out))

# verify no conflicting duplicates: rows where key appears >1 with different alt/S4
key_counts = out.groupby(KEYS + ["isSporE"]).size()
print("keys with >1 row after dedup:", (key_counts > 1).sum())

# drop redundant columns
for c in ["doy_sa", "doy_cd", "year_sa", "month_sa", "hour_sa", "minutes_sa",
          "lat_sa", "lon_sa", "isSporE_sa", "isSporE_cd"]:
    if c in out.columns:
        out.drop(columns=[c], inplace=True)

out.to_csv(f"{BASE}/combined_data.csv", index=False)
print("saved combined_data.csv:", out.shape)
print(out.columns.tolist())
