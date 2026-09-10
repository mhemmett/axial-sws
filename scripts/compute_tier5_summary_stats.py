#!/usr/bin/env python3
"""
compute_tier5_summary_stats.py

One-off summary-statistics script (not part of the pipeline): reports the average delay time
(dt) with uncertainty, and the average/min/max travel-time-weighted percent anisotropy
(dt/T_travel*100), across ALL 6 stations' tier-5 QC events (this session's tightest filter --
quality>=0.5, dt<T_dom/2, phi_error<10 deg, dt_error<0.05s), pooled together.

Reuses traveltime_anisotropy_7period_6stations_newdata_phi10.py's exact station-file loading
and QC (imported, not duplicated) so the tier-5 population matches that script's PDF output.
T_travel is computed the same way as figure_all_stations_temporal_traveltime_anisotropy_
newdata.py: ray-traced through the Baillard 3D Vs model (pykonal_raytracer.BaillardRayTracer)
with continuous ds/Vs integration (baillard_velocity.vs_at) along the ray.

Uncertainty on the mean dt is the standard error of the mean (SEM = std/sqrt(N)).

Run with:
    python3 compute_tier5_summary_stats.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import traveltime_anisotropy_7period_6stations_newdata_phi10 as t5mod  # noqa: E402
from pykonal_raytracer import BaillardRayTracer  # noqa: E402
from baillard_velocity import vs_at  # noqa: E402

STATIONS = list(t5mod.STATION_FILES.keys())
Z_MAX = 4.0
N_RAY = 200


def main():
    print(f'Loading tier-5 events (quality>={t5mod.QW_MIN}, dt<T_dom/2, '
          f'phi_error<{t5mod.PHI_ERR_MAX}deg, dt_error<{t5mod.DT_ERR_MAX}s) for all 6 stations...')
    dfs = {}
    for sta in STATIONS:
        df = t5mod.load_station(sta)
        dfs[sta] = df
        print(f'  {sta}: {len(df):,}')
    all_df = pd.concat([d.assign(station=s) for s, d in dfs.items()], ignore_index=True)
    n_total = len(all_df)
    print(f'  TOTAL: {n_total:,}')

    # ── Delay time: mean +/- SEM ──────────────────────────────────────────────────────────
    dt_vals = all_df['dt'].values
    dt_mean = float(np.mean(dt_vals))
    dt_std = float(np.std(dt_vals, ddof=1))
    dt_sem = dt_std / np.sqrt(n_total)
    print(f'\nDelay time (dt): {dt_mean:.4f} +/- {dt_sem:.4f} s (SEM, N={n_total:,}; std={dt_std:.4f}s)')

    # ── Travel-time anisotropy: ray-trace each event, then mean/min/max ──────────────────
    print('\nPrecomputing FMM travel-time fields for all 6 stations...')
    tracer = BaillardRayTracer()
    # Station (x,y) needed for precompute_station -- reuse the same station coordinate
    # source as the sibling script (STATION_FILE + ll2xy), not per-event x/y.
    sta_df = pd.read_csv(t5mod.STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                         engine='python').set_index('s')
    for sta in STATIONS:
        sx, sy = t5mod.ll2xy(sta_df.loc[sta, 'lat'], sta_df.loc[sta, 'lon'])
        tracer.precompute_station(sta, float(sx), float(sy))
        print(f'  FMM {sta} done')

    print(f'\nTracing {n_total:,} rays + integrating ds/Vs (Baillard model) for T_travel...')
    t0 = time.time()
    pct_aniso = []
    n_traced = 0
    for sta in STATIONS:
        df = dfs[sta]
        for row in df.itertuples():
            ex, ey, ez = float(row.x), float(row.y), float(row.z)
            if ez < 0 or ez > Z_MAX:
                continue
            try:
                ray = tracer.trace(sta, ex, ey, ez, n_pts=N_RAY)
                mid = 0.5 * (ray[:-1] + ray[1:])
                ds = np.linalg.norm(np.diff(ray, axis=0), axis=1)
                vs_mid = vs_at(mid[:, 0], mid[:, 1], mid[:, 2])
                T_travel = float(np.sum(ds / np.maximum(vs_mid, 1e-9)))
                if T_travel > 0:
                    pct_aniso.append(float(row.dt) * 100.0 / T_travel)
                    n_traced += 1
            except RuntimeError:
                continue
            if n_traced % 10000 == 0 and n_traced > 0:
                print(f'  {n_traced:,}/{n_total:,}  {time.time()-t0:.0f}s', end='\r', flush=True)

    pct_aniso = np.array(pct_aniso)
    print(f'\n  Traced {n_traced:,}/{n_total:,} events successfully ({time.time()-t0:.0f}s)')

    a_mean = float(np.mean(pct_aniso))
    a_std = float(np.std(pct_aniso, ddof=1))
    a_sem = a_std / np.sqrt(len(pct_aniso))
    a_min = float(np.min(pct_aniso))
    a_max = float(np.max(pct_aniso))
    print(f'\nTravel-time-weighted percent anisotropy (dt/T_travel*100):')
    print(f'  mean = {a_mean:.3f}% +/- {a_sem:.3f}% (SEM, N={len(pct_aniso):,}; std={a_std:.3f}%)')
    print(f'  min  = {a_min:.4f}%')
    print(f'  max  = {a_max:.4f}%')

    print('\n=== SUMMARY ===')
    print(f'Tier-5 pooled events (all 6 stations): N={n_total:,}')
    print(f'Delay time:            {dt_mean:.4f} +/- {dt_sem:.4f} s')
    print(f'Percent anisotropy:    mean={a_mean:.3f}%,  min={a_min:.4f}%,  max={a_max:.4f}%')


if __name__ == '__main__':
    main()
