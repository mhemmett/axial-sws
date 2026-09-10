#!/usr/bin/env python3
"""
dt_and_pct_aniso_vs_depth_5stations_newdata.py

Two figures for the 5 stations with new mfast try_filters + max_t_shift_s=0.2s splitting
results (AXAS1, AXCC1, AXEC1, AXEC2, AXEC3 -- same dataset as rose_7period_5stations_newdata.py;
AXAS2 excluded, no new-data rerun exists for it yet):

  1. Delay time (dt) vs. event depth, one panel per station.
  2. Percent travel-time anisotropy (100 * dt / T_travel) vs. event depth, one panel per
     station -- T_travel is the S-wave travel time from event hypocenter to station,
     computed by ray-tracing through the Baillard 3D Vs model (pykonal_raytracer.
     BaillardRayTracer) + continuous ds/Vs integration (baillard_velocity.vs_at), same method
     as figure_all_stations_temporal_traveltime_anisotropy_newdata.py.

QC (matching rose_7period_5stations_newdata.py's tier 4 -- the full cut before its stricter
phi_err<10 page): success, dt>0, quality>=0.5, dt<T_dom/2 (cycle-skip-risk cut, per-event
dominant period), phi_error<20 deg, dt_error<0.05s.

AXEC2's 2015-2021 half is still the in-progress partial maxdt02 re-run (whatever batches have
completed so far); its event lat/lon/depth come from raw_axec2_all_batches_mfast_filters_data/
raw_axec2_all_batches_mfast_filters_metadata.csv (merged on event_id), since the batch CSVs
themselves don't carry location. All other stations/periods carry lat/lon/depth directly.

Produces:
    dt_vs_depth_5stations_newdata.pdf
    pct_aniso_vs_depth_5stations_newdata.pdf

Run with:
    python3 dt_and_pct_aniso_vs_depth_5stations_newdata.py
"""

import glob
import os
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt

from pykonal_raytracer import BaillardRayTracer
from baillard_velocity import vs_at

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
AXEC2_META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                              'raw_axec2_all_batches_mfast_filters_metadata.csv')
STATION_FILE = os.path.join(HERE, '..', 'data', 'stations_axial.llz')

OUT_DT_PDF = os.path.join(HERE, 'dt_vs_depth_5stations_newdata.pdf')
OUT_PCT_PDF = os.path.join(HERE, 'pct_aniso_vs_depth_5stations_newdata.pdf')

QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05

STATION_ORDER = ['AXAS1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

STATION_FILES = {
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
}

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


def load_station_qc(sta):
    """Load + fully QC one station's new-data events, returning dt/depth (lat/lon too)."""
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = sorted(glob.glob(os.path.join(
            AXEC2_2015_2021_DIR,
            'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_*.csv')))
        batch_dfs = [pd.read_csv(f) for f in batch_files]
        batch_dfs = [d for d in batch_dfs if len(d) > 0]
        meta = pd.read_csv(AXEC2_META_CSV)[['event_id', 'latitude', 'longitude', 'depth']]
        for d in batch_dfs:
            d['dominant_period'] = d['chosen_filter_dom_period_samples'] / 200.0
            d = d.merge(meta, on='event_id', how='left')
            dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality', 'phi_error', 'dt_error', 'dominant_period',
                            'latitude', 'longitude', 'depth'])
    df = df[df['quality'] >= QW_MIN]
    df = df[df['dt'] < df['dominant_period'] / 2.0]
    df = df[df['phi_error'] < PHI_ERR_MAX]
    df = df[df['dt_error'] < DT_ERR_MAX]
    return df[['dt', 'latitude', 'longitude', 'depth']].rename(
        columns={'latitude': 'lat', 'longitude': 'lon'})


def main():
    print('Loading + QC-ing new mfast max_dt=0.2s splitting results '
          '(AXCC1, AXEC1, AXEC3, AXAS1: complete; AXEC2 2015-2021: partial, in-progress re-run)...')
    station_events = {}
    for sta in STATION_ORDER:
        df = load_station_qc(sta)
        station_events[sta] = df
        print(f'  {sta}: {len(df):,} QC-passing events')

    # ── dt vs depth ──────────────────────────────────────────────────────────
    n_sta = len(STATION_ORDER)
    fig1, axes1 = plt.subplots(1, n_sta, figsize=(4.0 * n_sta, 4.0), sharey=True)
    for ax, sta in zip(axes1, STATION_ORDER):
        df = station_events[sta]
        hb = ax.hexbin(df['depth'], df['dt'], gridsize=60, cmap='viridis',
                       mincnt=1, bins='log')
        ax.set_xlabel('Event depth [km]', fontsize=9)
        ax.set_title(f'{sta}\nN={len(df):,}', fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)
        fig1.colorbar(hb, ax=ax, label='log10(count)', fraction=0.046, pad=0.04)
    axes1[0].set_ylabel(r'$\delta t$ [s]', fontsize=10, fontweight='bold')
    fig1.suptitle('Delay time vs. event depth (new mfast max_dt=0.2s data)\n'
                  f'quality>={QW_MIN}, dt<T_dom/2, phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s',
                  fontsize=12, fontweight='bold')
    fig1.tight_layout(rect=[0, 0, 1, 0.90])
    fig1.savefig(OUT_DT_PDF, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'\nSaved {OUT_DT_PDF}')

    # ── percent travel-time anisotropy vs depth ─────────────────────────────
    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                          engine='python').set_index('s')
    _sta_df = _sta_df.loc[[s for s in STATION_ORDER if s in _sta_df.index]]
    _sta_df['x'], _sta_df['y'] = ll2xy(_sta_df['lat'].values, _sta_df['lon'].values)

    print('\nPrecomputing FMM travel-time fields for all 5 stations...')
    tracer = BaillardRayTracer()
    for sta in STATION_ORDER:
        tracer.precompute_station(sta, float(_sta_df.loc[sta, 'x']), float(_sta_df.loc[sta, 'y']))

    Z_MAX = 4.0
    N_RAY = 200

    print('\nTracing rays + integrating ds/Vs (Baillard model) to get S-wave travel time '
          'T_travel per event...')
    for sta in STATION_ORDER:
        df = station_events[sta]
        x, y = ll2xy(df['lat'].values, df['lon'].values)
        T_travel = np.full(len(df), np.nan)
        t0 = time.time()
        for i, (ex, ey, ez) in enumerate(zip(x, y, df['depth'].values)):
            if ez < 0 or ez > Z_MAX:
                continue
            try:
                ray = tracer.trace(sta, float(ex), float(ey), float(ez), n_pts=N_RAY)
                mid = 0.5 * (ray[:-1] + ray[1:])
                ds = np.linalg.norm(np.diff(ray, axis=0), axis=1)
                vs_mid = vs_at(mid[:, 0], mid[:, 1], mid[:, 2])
                tt = float(np.sum(ds / np.maximum(vs_mid, 1e-9)))
                if tt > 0:
                    T_travel[i] = tt
            except RuntimeError:
                continue
            if (i + 1) % 10000 == 0:
                print(f'  {sta}: {i+1:,}/{len(df):,}  {time.time()-t0:.0f}s', end='\r', flush=True)
        df['T_travel'] = T_travel
        n_valid = np.isfinite(T_travel).sum()
        print(f'  {sta}: {n_valid:,}/{len(df):,} events traced successfully ({time.time()-t0:.0f}s)')
        df['pct_aniso'] = df['dt'] / df['T_travel'] * 100.0
        df = df.dropna(subset=['pct_aniso'])
        station_events[sta] = df[np.isfinite(df['pct_aniso'])]

    fig2, axes2 = plt.subplots(1, n_sta, figsize=(4.0 * n_sta, 4.0), sharey=True)
    for ax, sta in zip(axes2, STATION_ORDER):
        df = station_events[sta]
        hb = ax.hexbin(df['depth'], df['pct_aniso'], gridsize=60, cmap='viridis',
                       mincnt=1, bins='log')
        ax.set_xlabel('Event depth [km]', fontsize=9)
        ax.set_title(f'{sta}\nN={len(df):,}', fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)
        fig2.colorbar(hb, ax=ax, label='log10(count)', fraction=0.046, pad=0.04)
    axes2[0].set_ylabel(r'$\delta t / T_{travel}$ [%]', fontsize=10, fontweight='bold')
    fig2.suptitle('Percent travel-time anisotropy vs. event depth (new mfast max_dt=0.2s data)\n'
                  f'quality>={QW_MIN}, dt<T_dom/2, phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s',
                  fontsize=12, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.90])
    fig2.savefig(OUT_PCT_PDF, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'\nSaved {OUT_PCT_PDF}')


if __name__ == '__main__':
    main()
