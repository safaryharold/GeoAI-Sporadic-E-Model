"""Download EMBRACE/INPE digisonde .SAO files for South American stations,
year 2023, sampling ~every 3 hours (nearest sounding to 00,03,...,21 UT).

Saves files under ionosonde_data/<STN>/<DOY>/ and writes a manifest CSV.
"""
import os, re, sys, time, urllib.request, concurrent.futures
import pandas as pd

BASE = "https://embracedata.inpe.br/ionosonde"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ionosonde_data")
YEAR = int(os.environ.get("IONO_YEAR","2023"))
STATIONS = ["BLJ03", "SAA0K", "FZA0M", "BVJ03", "CGK21", "CAJ2M", "SMK29"]
TARGET_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

UA = {"User-Agent": "Mozilla/5.0 (research; sporadic-E validation)"}


def fetch(url, dest=None, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            data = urllib.request.urlopen(req, timeout=30).read()
            if dest:
                with open(dest, "wb") as f:
                    f.write(data)
            return data
        except Exception:
            time.sleep(1 + i)
    return None


def list_day(st, doy):
    url = f"{BASE}/{st}/{YEAR}/{doy:03d}/"
    data = fetch(url)
    if data is None:
        return []
    html = data.decode(errors="replace")
    names = sorted(set(re.findall(st + r"_" + str(YEAR) + f"{doy:03d}" + r"[0-9]{6}\.SAO", html)))
    return names


def pick_names(names):
    """Choose the .SAO nearest each target hour."""
    chosen = []
    for h in TARGET_HOURS:
        best, bd = None, 1e9
        for n in names:
            hhmmss = int(n[13:19])
            hh = hhmmss // 10000
            d = abs(hh - h)
            if d < bd:
                bd, best = d, n
        if best is not None and bd <= 1:
            chosen.append(best)
    return chosen


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    # step 1: list all days (parallel)
    tasks = [(st, doy) for st in STATIONS for doy in range(1, 366)]
    print(f"listing {len(tasks)} station-days...", flush=True)
    listings = {}
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(list_day, st, doy): (st, doy) for st, doy in tasks}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            st, doy = futs[fut]
            listings[(st, doy)] = fut.result()
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    # step 2: pick files
    jobs = []
    for (st, doy), names in listings.items():
        for n in pick_names(names):
            hhmmss = int(n[13:19])
            jobs.append((st, doy, n, hhmmss // 10000))
    print(f"downloading {len(jobs)} files...", flush=True)
    # step 3: download (parallel)
    def dl(job):
        st, doy, n, hr = job
        d = os.path.join(OUT, st, f"{doy:03d}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, n)
        if os.path.exists(path) and os.path.getsize(path) > 100:
            return (st, doy, n, hr, True)
        url = f"{BASE}/{st}/{YEAR}/{doy:03d}/{n}"
        data = fetch(url, dest=path)
        return (st, doy, n, hr, data is not None)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        for res in ex.map(dl, jobs):
            st, doy, n, hr, ok = res
            if ok:
                manifest.append((st, YEAR, doy, n, hr))
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    m = pd.DataFrame(manifest, columns=["station", "year", "doy", "file", "hour"])
    m.to_csv(os.path.join(OUT, f"manifest_{YEAR}.csv"), index=False)
    print("DONE", len(m), "files")


if __name__ == "__main__":
    main()
