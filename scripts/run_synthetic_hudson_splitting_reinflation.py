#!/usr/bin/env python3
"""
run_synthetic_hudson_splitting_reinflation.py

Generates synthetic (phi, dt) for the "Re-Inflation" period -- the most recent (7th) of the
standard 7 eruption-cycle periods used throughout this session's rose plots
(rose_7period_6stations_newdata.py), representing post-eruption re-inflation -- via the new
Scenario D physics (synthetic_hudson_reinflation_scenario.py): background + both ring-fault
damage zones + ONE inflating sill at the (retired) two-sills model's SILL2 location (Kidiwela
S2), no dike, no original sill, PLUS a localized isotropic crack-density boost around AXAS2
(hydrothermal pathways, per explicit user request) -- hence this script uses
reinflation.integrate_synthetic_splitting_reinflation(), not the shared
synthetic_hudson_crack_scenarios.integrate_synthetic_splitting().

Same method as run_synthetic_hudson_splitting.py: real station locations + real earthquake
hypocenters (tier-4 QC, same as that script, used ONLY to pick realistic ray start points --
not for their real phi/dt), traced through the actual 3D Baillard S-wave velocity model
(PyKonal FMM), then Hudson-crack-physics (phi, dt) accumulated along each ray. Same per-
station-per-period subsample cap (100 events) as the pre-/syn-eruption periods, so all three
periods are on equal footing.

The Re-Inflation period's time window is taken directly from rose_7period_6stations_newdata.py's
own 7-period split (equal-count post-eruption bins over the combined 6-station "newdata"
catalog) -- period index 6 (the 7th, most recent), NOT a separately-invented window.

Produces:
    synthetic_hudson_reinflation_phidt.csv   (event_id, station, period, scenario, phi_synth, dt_synth)

Run with:
    python3 run_synthetic_hudson_splitting_reinflation.py
"""

import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os

from rose_7period_6stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, _build_time_periods,
)
from pykonal_raytracer import BaillardRayTracer
from hudson_crack_model import fibonacci_sphere

# Import order matters: this installs the Scenario-D-aware total_stress_and_crack_sets onto
# synthetic_hudson_crack_scenarios BEFORE any splitting is integrated.
import synthetic_hudson_reinflation_scenario as reinflation
from synthetic_hudson_crack_scenarios import STATION_XY

HERE = os.path.dirname(os.path.abspath(__file__))
TIER_IDX = 4   # quality>=0.5, dt<T_dom/2, phi_error<20deg, dt_error<0.05s -- same as
               # run_synthetic_hudson_splitting.py, used only to pick real hypocenters

N_EVENTS_PER_STATION_PER_PERIOD = 100
N_RAY_PTS = 30
N_CRACKS = 100
Z_MAX = 4.0

REINFLATION_LABEL = 'Re-Inflation'


def ll2xy_local(lat, lon):
    ini_lon, ini_lat = -130.1, 45.9
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(ini_lat))
    return ((np.asarray(lon) - ini_lon) * km_per_deg_lon,
            (np.asarray(lat) - ini_lat) * km_per_deg_lat)


def get_reinflation_window():
    """Returns (t0, t1) for period index 6 (the 7th/most recent) of
    rose_7period_6stations_newdata.py's own 7-period split, built from the combined
    (unfiltered) 6-station catalog -- i.e. literally "period 7 from our 7 period plots"."""
    raw = {sta: load_station_raw(sta) for sta in STATION_ORDER}
    all_combined = pd.concat(raw.values(), ignore_index=True)
    periods = _build_time_periods(all_combined)
    label, t0, t1 = periods[-1]
    print(f'Re-Inflation window (period 7 of 7): {label!r}  {t0} to {t1}')
    return raw, t0, t1


def load_reinflation_events(raw, t0, t1):
    """Returns dfs[station] -> subsampled tier-4 event dataframe within [t0, t1)."""
    dfs = {}
    for sta in STATION_ORDER:
        df = apply_tier(raw[sta], TIER_IDX)
        df = df.dropna(subset=['latitude', 'longitude', 'depth'])
        m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
        if t1 is not None:
            m = m & (df['t'] < t1)
        sub = df[m]
        n = min(N_EVENTS_PER_STATION_PER_PERIOD, len(sub))
        if n > 0:
            sub = sub.sample(n=n, random_state=42)
        dfs[sta] = sub.reset_index(drop=True)
        print(f'  {sta} / {REINFLATION_LABEL}: {len(dfs[sta]):,} events (subsample)')
    return dfs


def main():
    raw, t0, t1 = get_reinflation_window()
    dfs = load_reinflation_events(raw, t0, t1)

    print('\nPrecomputing FMM travel-time fields for all stations...')
    tracer = BaillardRayTracer()
    for sta, (sx, sy) in STATION_XY.items():
        if sta in STATION_ORDER:
            tracer.precompute_station(sta, sx, sy)

    n_hats = fibonacci_sphere(N_CRACKS)

    results = []
    for sta in STATION_ORDER:
        df = dfs[sta]
        if len(df) == 0:
            print(f'  {sta} (scenario D): 0 events, skipping')
            continue
        x_arr, y_arr = ll2xy_local(df['latitude'].values, df['longitude'].values)
        z_arr = df['depth'].values
        eid_arr = df['event_id'].values if 'event_id' in df.columns else np.arange(len(df))

        t_start = time.time()
        n_traced = 0
        for i in range(len(df)):
            ex, ey, ez = float(x_arr[i]), float(y_arr[i]), float(z_arr[i])
            if ez < 0 or ez > Z_MAX:
                continue
            try:
                ray = tracer.trace(sta, ex, ey, ez, n_pts=N_RAY_PTS)
            except RuntimeError:
                continue
            n_traced += 1
            phi_s, dt_s = reinflation.integrate_synthetic_splitting_reinflation(
                'D', ray, dike_interps=None, n_hats=n_hats)
            results.append(dict(event_id=eid_arr[i], station=sta, period=REINFLATION_LABEL,
                                scenario='D', phi_synth=phi_s, dt_synth=dt_s))
            if (i + 1) % 50 == 0:
                print(f'  {sta} (scenario D): {i+1}/{len(df)}  {time.time()-t_start:.0f}s',
                      end='\r', flush=True)
        print(f'  {sta} (scenario D): {n_traced}/{len(df)} events traced successfully '
              f'({time.time()-t_start:.0f}s)')

    out_csv = os.path.join(HERE, 'synthetic_hudson_reinflation_phidt.csv')
    df_all = pd.DataFrame(results)
    df_all.to_csv(out_csv, index=False)
    print(f'Saved {out_csv} ({len(df_all):,} rows)')


if __name__ == '__main__':
    main()
