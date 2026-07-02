import glob
import numpy as np
import pandas as pd
from obspy import read
from tqdm import tqdm

DATA_DIR = "/Users/mhemmett/Seismology/axial-splitting-ml/data"
OUT_CSV = f"{DATA_DIR}/axec1_pgv_2015_2021.csv"

batch_files = sorted(glob.glob(f"{DATA_DIR}/axial_mldd_2015_2021_axec1_batch_*.mseed"),
                     key=lambda x: int(x.split("_batch_")[-1].replace(".mseed", "")))

rows = []

for batch_path in tqdm(batch_files, desc="Batches"):
    st = read(batch_path)

    # Group traces by event starttime (rounded to ms to handle float precision)
    events = {}
    for tr in st:
        key = round(tr.stats.starttime.timestamp, 3)
        events.setdefault(key, []).append(tr)

    for t, traces in events.items():
        station = traces[0].stats.station
        # PGV = peak of vector-sum velocity across all available components
        pgv = max(np.max(np.abs(tr.data)) for tr in traces)
        timestamp = traces[0].stats.starttime.isoformat()
        rows.append({"station": station, "timestamp": timestamp, "pgv_counts": pgv})

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} rows to {OUT_CSV}")
print(df.head())
