"""
Three-way S-wave incidence angle comparison for AXEC2 batch 1 (Step 1 of the
LQT-rotation/Q_w plan): TauP ray tracing, PyKonal FMM ray tracing through the Baillard 3D
S-velocity model, and the new eigenvalue/polarization-based estimate. Also carries along the
existing P-wave Jurkevics incidence (already cached) for reference.

"Batch 1" = the first 1/100 split of the cached, QC-passing AXEC2 event set
(passing_waveforms_data_mldd_2015_2021_axec2/), matching the np.array_split(keys, 100)
convention used by the production batch notebooks/scripts.

Run with:
    python3 run_incidence_comparison_axec2_batch1.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime

warnings.filterwarnings('ignore')

_SWSPY_LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swspy'))
if _SWSPY_LOCAL_PATH not in sys.path:
    sys.path.insert(0, _SWSPY_LOCAL_PATH)
import swspy  # noqa: F401

from splitting_functions import calculate_incidence_angle, calculate_incidence_angle_eigenvalue_jurkevics_s
from pykonal_raytracer import BaillardRayTracer
from baillard_raytraced_incidence import ll2xy

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'passing_waveforms_data_mldd_2015_2021_axec2')
METADATA_CSV = os.path.join(CACHE_DIR, 'passing_waveforms_metadata.csv')
BATHY_FILE = os.path.join(HERE, '..', 'data', 'AXIAL_MODEL_3P_VELOCITY.S.mod.buf')
OUT_CSV = os.path.join(HERE, 'incidence_angle_comparison_axec2_batch1.csv')

STATION = 'AXEC2'
STA_LAT, STA_LON = 45.93967, -129.9738  # data/stations_axial.llz

N_BATCHES = 100


def get_zne_traces(waveform_path):
    st = obspy.read(waveform_path)
    trace_z = trace_n = trace_e = None
    for tr in st:
        comp = tr.stats.channel[-1].upper()
        if comp == 'Z':
            trace_z = tr
        elif comp in ('N', '1'):
            trace_n = tr
        elif comp in ('E', '2'):
            trace_e = tr
    return trace_z, trace_n, trace_e


def main():
    print(f"Loading metadata: {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)
    print(f"Total cached AXEC2 events: {len(df)}")

    keys = np.arange(len(df))
    batches = np.array_split(keys, N_BATCHES)
    batch1_idx = batches[0]
    batch1 = df.iloc[batch1_idx].reset_index(drop=True)
    print(f"Batch 1: {len(batch1)} events "
          f"({batch1['datetime'].iloc[0]} to {batch1['datetime'].iloc[-1]})")

    print("Precomputing PyKonal FMM travel-time field for AXEC2...")
    tracer = BaillardRayTracer(bathy_file=BATHY_FILE)
    sx, sy = ll2xy(STA_LAT, STA_LON)
    tracer.precompute_station(STATION, float(sx), float(sy))

    rows = []
    t_start = time.time()
    for i, row in batch1.iterrows():
        eq_lat, eq_lon, eq_depth = row['latitude'], row['longitude'], row['depth']

        # 1. TauP ray-traced incidence (velocity-model, ak135, literature-standard)
        try:
            inc_taup = calculate_incidence_angle(eq_lat, eq_lon, eq_depth, STA_LAT, STA_LON)
        except Exception:
            inc_taup = np.nan

        # 2. PyKonal FMM ray-traced incidence (Baillard 3D S-velocity model)
        ex, ey = ll2xy(eq_lat, eq_lon)
        if eq_depth < 0 or eq_depth > 4.0:
            inc_pykonal = np.nan
        else:
            inc_pykonal = tracer.incidence_angle_at_station(STATION, float(ex), float(ey), float(eq_depth))

        # 3. Eigenvalue/polarization-based incidence (new, this session)
        # waveform_file in the metadata CSV is already relative to scripts/ (this file's dir)
        waveform_path = os.path.join(HERE, row['waveform_file'])
        try:
            trace_z, trace_n, trace_e = get_zne_traces(waveform_path)
            if trace_z is None or trace_n is None or trace_e is None:
                inc_eigen = np.nan
            else:
                s_arrival_offset = UTCDateTime(row['datetime']) + float(row['s_arrival_time'])
                inc_eigen = calculate_incidence_angle_eigenvalue_jurkevics_s(
                    trace_z, trace_n, trace_e, s_arrival_offset=s_arrival_offset
                )
        except Exception as e:
            inc_eigen = np.nan

        rows.append({
            'event_id': row['event_id'],
            'datetime': row['datetime'],
            'depth_km': eq_depth,
            'incidence_p_jurkevics': row['incidence_eigenvalue_jurkevics'],
            'incidence_taup_s': inc_taup,
            'incidence_pykonal_s': inc_pykonal,
            'incidence_eigenvalue_s': inc_eigen,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  {i+1}/{len(batch1)}  ({elapsed:.0f}s elapsed)")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(results_df)} rows to {OUT_CSV}")
    print(f"Total time: {time.time()-t_start:.0f}s")

    return results_df


if __name__ == '__main__':
    main()
