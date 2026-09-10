#!/usr/bin/env python3
"""
run_synthetic_hudson_splitting.py

Driver for the synthetic Hudson-crack-model splitting task (see synthetic_hudson_crack_
scenarios.py for the physics/geometry and the plan this was built from). Per explicit user
request, each real time period is now paired with its own matching scenario rather than
running all 3 scenarios over both periods:

  Pre-eruption events -> Scenario B (background + fault-damage-zone cracks + sill)
  Syn-eruption events -> Scenario C (background + fault-damage-zone cracks + dike, no sill)

-- i.e. the sill (inflation source) drives the pre-eruption synthetic splitting, and the
propagating dike drives the syn-eruption synthetic splitting, matching the real geologic
narrative (inflation before the 2015 eruption, dike intrusion during it). Scenario A (bare
background + fault cracks, no sill/dike) is defined in synthetic_hudson_crack_scenarios.py
but not run here as a separate period -- it's the shared baseline embedded in both B and C.

Real station locations and REAL earthquake hypocenters are used (same 5 new-data stations as
the rest of this session: AXAS1, AXCC1, AXEC1, AXEC2, AXEC3 -- AXAS2 excluded), tracing rays
through the actual 3D Baillard S-wave velocity model (PyKonal FMM), then accumulating
synthetic (phi, dt) along each ray via the Hudson crack physics instead of the real measured
(phi, dt). A bounded per-station-per-period subsample is used (not the full event count).

QC: tier 4 from rose_7period_5stations_newdata.py (quality>=0.5, dt<T_dom/2, phi_error<20 deg,
dt_error<0.05s) -- only used here to select the same real event set as the rest of this
session; the resulting phi/dt columns from that QC are NOT used (this script's own synthetic
phi/dt replace them).

Rose-plot colors match this session's established _period_colors scheme (rose_7period_
5stations_newdata.py): pre-eruption = purple (#800080), syn-eruption = red (#CC0000).

Produces:
    synthetic_hudson_pre_syn_phidt.csv   (event_id, station, period, scenario, phi_synth, dt_synth)
    synthetic_hudson_rose_plots.pdf      (1 page, 5 station rows x 2 period columns [pre/syn])

Run with:
    python3 run_synthetic_hudson_splitting.py
"""

import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

from rose_7period_6stations_newdata import (
    STATION_ORDER, apply_tier, _draw_rose, ERUPTION_START, ERUPTION_END, TRANSFER_DIR,
)
from pykonal_raytracer import BaillardRayTracer
from hudson_crack_model import fibonacci_sphere

from synthetic_hudson_crack_scenarios import (
    STATION_XY, build_dike_stress_interpolators, integrate_synthetic_splitting,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TIER_IDX = 4   # quality>=0.5, dt<T_dom/2, phi_error<20deg, dt_error<0.05s

N_EVENTS_PER_STATION_PER_PERIOD = 100   # subsample cap; uses all available if fewer
N_RAY_PTS = 30
N_CRACKS = 100
Z_MAX = 4.0   # km, same cap used elsewhere for ray validity

# Period -> (scenario, color) pairing, per explicit user request.
PERIOD_SCENARIOS = [
    ('Pre-eruption', None, ERUPTION_START, 'B', '#800080'),
    ('Syn-eruption', ERUPTION_START, ERUPTION_END, 'C', '#CC0000'),
]

# Grid covering all stations + the full PyLith dike waypoint trace (x:4-12, y:0-15 -- wider
# than the usual y:0-12 tomography grid since DIKE_WAYPOINTS reaches y=13.928).
DIKE_XN = np.arange(4.0, 12.01, 0.1)
DIKE_YN = np.arange(0.0, 15.01, 0.1)


STATION_FILES_WITH_LOCATION = {
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2015_2021_all_batches.csv',
              'splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv'],   # only period so far
}


def load_station_raw_with_location(sta):
    """Uniform loader for all 5 stations -- now that AXEC2 has its own combined
    splitting_results_AXEC2_2015_2021_all_batches.csv (built once from the 497 individual
    maxdt02 batch files + the metadata CSV's lat/lon/depth, then saved to
    mfast_maxdt_pipeline_transfer/ to match every other station's naming convention), no
    special-casing/glob/merge is needed here anymore."""
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES_WITH_LOCATION[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    df['phi_az'] = df['phi'] % 180.0
    return df


def load_real_events():
    """Returns dfs[station][period_label] -> subsampled event dataframe, pre/syn-eruption
    only (per explicit user request)."""
    print('Loading real event catalog (tier-4 QC, same as this session\'s standard) for '
          'station locations/depths only -- real phi/dt discarded in favor of synthetic...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw_with_location(sta)
    dfs = {sta: {} for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        df = apply_tier(raw[sta], TIER_IDX)
        df = df.dropna(subset=['latitude', 'longitude', 'depth'])
        for label, t0, t1, _sc, _color in PERIOD_SCENARIOS:
            m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
            if t1 is not None:
                m = m & (df['t'] < t1)
            sub = df[m]
            n = min(N_EVENTS_PER_STATION_PER_PERIOD, len(sub))
            if n > 0:
                sub = sub.sample(n=n, random_state=42)
            dfs[sta][label] = sub.reset_index(drop=True)
            print(f'  {sta} / {label}: {len(dfs[sta][label]):,} events (subsample)')
    return dfs


def ll2xy_local(lat, lon):
    ini_lon, ini_lat = -130.1, 45.9
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(ini_lat))
    return ((np.asarray(lon) - ini_lon) * km_per_deg_lon,
            (np.asarray(lat) - ini_lat) * km_per_deg_lat)


def main():
    dfs = load_real_events()

    print('\nPrecomputing FMM travel-time fields for all 5 stations...')
    tracer = BaillardRayTracer()
    for sta, (sx, sy) in STATION_XY.items():
        if sta in STATION_ORDER:
            tracer.precompute_station(sta, sx, sy)

    print('Building dike stress interpolators (Scenario C)...')
    dike_interps = build_dike_stress_interpolators(DIKE_XN, DIKE_YN)

    n_hats = fibonacci_sphere(N_CRACKS)

    results = []

    for sta in STATION_ORDER:
        for label, _t0, _t1, sc, _color in PERIOD_SCENARIOS:
            df = dfs[sta][label]
            if len(df) == 0:
                print(f'  {sta} / {label} (scenario {sc}): 0 events, skipping')
                continue
            x_arr, y_arr = ll2xy_local(df['latitude'].values, df['longitude'].values)
            z_arr = df['depth'].values
            eid_arr = df['event_id'].values if 'event_id' in df.columns else np.arange(len(df))

            t0 = time.time()
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
                phi_s, dt_s = integrate_synthetic_splitting(
                    sc, ray, dike_interps=dike_interps if sc == 'C' else None,
                    n_hats=n_hats)
                results.append(dict(event_id=eid_arr[i], station=sta, period=label,
                                    scenario=sc, phi_synth=phi_s, dt_synth=dt_s))
                if (i + 1) % 50 == 0:
                    print(f'  {sta}/{label} (scenario {sc}): {i+1}/{len(df)}  '
                          f'{time.time()-t0:.0f}s', end='\r', flush=True)
            print(f'  {sta} / {label} (scenario {sc}): {n_traced}/{len(df)} events traced '
                  f'successfully ({time.time()-t0:.0f}s)')

    out_csv = os.path.join(HERE, 'synthetic_hudson_pre_syn_phidt.csv')
    df_all = pd.DataFrame(results)
    df_all.to_csv(out_csv, index=False)
    print(f'Saved {out_csv} ({len(df_all):,} rows)')

    # ── Rose plot: 1 page, 5 station rows x 2 period columns ───────────────────
    out_pdf = os.path.join(HERE, 'synthetic_hudson_rose_plots.pdf')
    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(len(STATION_ORDER), len(PERIOD_SCENARIOS),
                                 figsize=(3.2 * len(PERIOD_SCENARIOS), 3.2 * len(STATION_ORDER)),
                                 subplot_kw=dict(projection='polar'))
        for ri, sta in enumerate(STATION_ORDER):
            for ci, (label, _t0, _t1, sc, color) in enumerate(PERIOD_SCENARIOS):
                ax = axes[ri, ci]
                sub = df_all[(df_all['station'] == sta) & (df_all['period'] == label)]
                phi_az = sub['phi_synth'].values % 180.0
                _draw_rose(ax, phi_az, np.ones(len(sub)), color)
                title = f'{label} (Scenario {sc})\nN={len(sub):,}' if ri == 0 else f'N={len(sub):,}'
                ax.set_title(title, fontsize=8, fontweight='bold')
                if ci == 0:
                    ax.text(-0.25, 0.5, sta, fontsize=10, fontweight='bold',
                            ha='center', va='center', transform=ax.transAxes)
        fig.suptitle('Synthetic Hudson-crack splitting: Pre-eruption = Scenario B (sill), '
                     'Syn-eruption = Scenario C (dike)\n'
                     f'(subsample, up to {N_EVENTS_PER_STATION_PER_PERIOD}/station/period)',
                     fontsize=10, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f'\nSaved {out_pdf}')


if __name__ == '__main__':
    main()
