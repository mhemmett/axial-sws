import glob
import numpy as np
import pandas as pd
from obspy import read, read_inventory
from tqdm import tqdm

DATA_DIR = "/Users/mhemmett/Seismology/axial-splitting-ml/data"
OUT_CSV = f"{DATA_DIR}/axec3_pgv_2015_2021.csv"

inv = read_inventory(f"{DATA_DIR}/AXIAL_stations.xml")

batch_files = sorted(
    glob.glob(f"{DATA_DIR}/axial_mldd_2015_2021_axec3_batch_*.mseed"),
    key=lambda x: int(x.split("_batch_")[-1].replace(".mseed", ""))
)

rows = []

for batch_path in tqdm(batch_files, desc="Batches"):
    st = read(batch_path)

    # Remove instrument response -> output velocity in m/s
    # pre_filt corners: below noise floor and above Nyquist for 200 Hz data
    st.remove_response(inventory=inv, output="VEL", pre_filt=(4.0, 5.0, 40.0, 45.0), water_level=60)

    events = {}
    for tr in st:
        key = round(tr.stats.starttime.timestamp, 3)
        events.setdefault(key, []).append(tr)

    for t, traces in events.items():
        pgv = max(np.max(np.abs(tr.data)) for tr in traces)
        rows.append({
            "station": traces[0].stats.station,
            "timestamp": traces[0].stats.starttime.isoformat(),
            "pgv_ms": pgv,
        })

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} rows to {OUT_CSV}")
print(df.head())
print(f"\nDate range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
