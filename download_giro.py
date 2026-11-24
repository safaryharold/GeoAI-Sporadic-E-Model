"""Download GIRO DIDBase scaled characteristics (foEs, fbEs, fminEs, foF2)
for additional ionosondes inside the study domain (35S-15N, 85W-30W).

Stations with data in 2020-2024 (checked 2026-11-23 via the DIDBase station
list and per-year record counts):
  SMK29 Santa Maria  (2020-2023)
  JI91J Jicamarca    (2020-2022)
Tucuman (TUJ2O, TNJ20) and Cachimbo (CXM9B) return no data for 2020-2024.
BLJ03 (Belem) 2020 and 2023 are also fetched to cross-check the GIRO Es flag
against the EMBRACE .SAO files already used in the paper.

Raw monthly text files are cached under ionosonde_data/giro/<STN>/<YYYY-MM>.txt.
Data: CC-BY-NC-SA 4.0, Lowell GIRO Data Center (acknowledge the station
providers per the DIDBase Rules of the Road).
"""
import os, time, urllib.request, calendar

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "ionosonde_data", "giro")
URL = ("https://lgdc.uml.edu/fastchar/getbest?ursiCode={st}&charName=foEs,fbEs,fminEs,foF2"
       "&DMUF=3000&fromDate={y}%2F{m:02d}%2F01+00%3A00%3A00&toDate={y}%2F{m:02d}%2F{d:02d}+23%3A59%3A59")
JOBS = {"SMK29": [2020, 2021, 2022, 2023], "JI91J": [2020, 2021, 2022], "BLJ03": [2020, 2023]}


def fetch(url, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-client"})
            return urllib.request.urlopen(req, timeout=180).read().decode(errors="replace")
        except Exception as e:
            print("  retry", i, e, flush=True); time.sleep(10 * (i + 1))
    return None


def main():
    for st, years in JOBS.items():
        os.makedirs(os.path.join(OUT, st), exist_ok=True)
        for y in years:
            for m in range(1, 13):
                dest = os.path.join(OUT, st, f"{y}-{m:02d}.txt")
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    continue
                txt = fetch(URL.format(st=st, y=y, m=m, d=calendar.monthrange(y, m)[1]))
                if txt is None:
                    print("FAILED", st, y, m, flush=True); continue
                open(dest, "w").write(txt)
                n = sum(1 for l in txt.splitlines() if l and not l.startswith("#"))
                print(st, y, m, n, flush=True)
                time.sleep(3)


if __name__ == "__main__":
    main()
