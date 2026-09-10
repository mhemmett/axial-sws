#!/usr/bin/env python3
"""
stats_windowcheck_grade3_dt_and_fractional_anisotropy.py

One-off stats script, per explicit user request: average delay time (dt) and fractional
shear-wave anisotropy (with uncertainty), pooled across all 6 stations' windowcheck grade-3
events (SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg -- same GRADE/
load_station_raw/apply_grade as rose_7period_regions_windowcheck_grade3.py), for three windows:
Before Eruption, During Eruption, and the whole dataset.

dt stats use ALL grade-3 events (event-level column, no ray tracing needed). Fractional
anisotropy requires ray-traced T_travel (same method as figure_ec3_multipanel_temporal_mfast_
windowcheck_grade3.py / spatial_midpoint_aniso_phi_windowcheck_grade3_7period_1row_countmin10.py:
a^2 + 2x*a - 4 = 0, x = 2*T_travel/dt, keep only 0<=a<=1), so it's restricted to events with
valid lat/lon/depth -- AXEC2's 2015-2021 half (the special combined CSV) lacks those columns
entirely and is excluded from the fractional-anisotropy numbers only (not from dt).

Uncertainty reported is the standard error of the mean (std / sqrt(N)).

Run with:
    python3 stats_windowcheck_grade3_dt_and_fractional_anisotropy.py
"""

import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
from rose_7period_regions_windowcheck_grade3 import (
    STATION_ORDER, GRADE, load_station_raw, apply_grade, ERUPTION_START, ERUPTION_END,
)

N_RAY = 200


def load_grade3_with_xyz(sta):
    raw = load_station_raw(sta)
    df = apply_grade(raw, GRADE)
    df = df.dropna(subset=['latitude', 'longitude', 'depth']).copy()
    from sws_forward_model import ll2xy
    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    return df


def mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0, n


def main():
    print('Loading windowcheck grade-3 events, all 6 stations (dt stats: no coordinate '
          'requirement)...')
    all_raw = []
    for sta in STATION_ORDER:
        raw = load_station_raw(sta)
        df = apply_grade(raw, GRADE)
        df['station'] = sta
        all_raw.append(df)
        print(f'  {sta}: {len(df):,} grade-3 events')
    dt_df = pd.concat(all_raw, ignore_index=True)
    n_total_dt = len(dt_df)
    print(f'Total grade-3 events (dt stats): {n_total_dt:,}')

    print('\n=== Delay time (dt, s) ===')
    for label, mask in [
        ('Before Eruption', dt_df['t'] < ERUPTION_START),
        ('During Eruption', (dt_df['t'] >= ERUPTION_START) & (dt_df['t'] < ERUPTION_END)),
        ('Whole dataset', pd.Series(True, index=dt_df.index)),
    ]:
        m, se, n = mean_se(dt_df.loc[mask, 'dt'].values)
        print(f'  {label:20s}: dt = {m:.4f} +/- {se:.4f} s  (N={n:,})')

    print('\nLoading windowcheck grade-3 events with valid coordinates (for ray tracing; '
          'AXEC2 2015-2021 excluded, no lat/lon in that file)...')
    T._setup_environment(need_tracer=True)

    xyz_dfs = []
    for sta in STATION_ORDER:
        df = load_grade3_with_xyz(sta)
        df['station'] = sta
        xyz_dfs.append(df)
        print(f'  {sta}: {len(df):,} events with valid coordinates')
    xdf = pd.concat(xyz_dfs, ignore_index=True)
    n_total_xyz = len(xdf)
    print(f'Total pooled events for ray tracing: {n_total_xyz:,}')

    print(f'\nTracing {n_total_xyz:,} rays (n_pts={N_RAY})...')
    t0 = time.time()
    frac_pct = np.full(n_total_xyz, np.nan)
    n_skipped_depth = 0
    n_trace_fail = 0
    n_bad_frac = 0
    for i, row in enumerate(xdf.itertuples()):
        ex, ey, ez = float(row.x), float(row.y), float(row.z)
        if ez < 0 or ez > T.Z_MAX:
            n_skipped_depth += 1
            continue
        try:
            ray = T.tracer.trace(row.station, ex, ey, ez, n_pts=N_RAY)
        except RuntimeError:
            n_trace_fail += 1
            continue
        vcols, seg_len_km = T.tracer.ray_to_voxels(ray, T.xn, T.yn, T.zn)
        if len(vcols) == 0:
            continue
        T_travel = float(np.sum(seg_len_km / np.maximum(T.vs_flat[vcols], 1e-9)))
        if T_travel <= 0:
            continue
        x_ratio = 2.0 * T_travel / row.dt
        a = -x_ratio + np.sqrt(x_ratio ** 2 + 4.0)
        if not (0.0 <= a <= 1.0):
            n_bad_frac += 1
            continue
        frac_pct[i] = a * 100.0
        if (i + 1) % 10000 == 0:
            print(f'  {i+1:,}/{n_total_xyz:,}  {time.time()-t0:.0f}s', end='\r', flush=True)

    xdf['frac_pct'] = frac_pct
    n_valid = int(np.isfinite(frac_pct).sum())
    print(f'\n{n_valid:,}/{n_total_xyz:,} events with valid fractional anisotropy '
          f'(skipped: depth-out-of-range={n_skipped_depth:,}, trace-fail={n_trace_fail:,}, '
          f'frac-aniso-out-of-[0,1]={n_bad_frac:,}) in {time.time()-t0:.0f}s')

    print('\n=== Fractional shear-wave anisotropy (%) ===')
    for label, mask in [
        ('Before Eruption', xdf['t'] < ERUPTION_START),
        ('During Eruption', (xdf['t'] >= ERUPTION_START) & (xdf['t'] < ERUPTION_END)),
        ('Whole dataset', pd.Series(True, index=xdf.index)),
    ]:
        m, se, n = mean_se(xdf.loc[mask, 'frac_pct'].values)
        print(f'  {label:20s}: a = {m:.3f} +/- {se:.3f} %  (N={n:,})')


if __name__ == '__main__':
    main()
