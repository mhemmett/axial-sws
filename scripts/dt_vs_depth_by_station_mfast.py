#!/usr/bin/env python3
"""
dt_vs_depth_by_station_mfast.py

Same figure as dt_vs_depth_by_station.py (delay time vs. event depth, one panel
per station + "All stations combined"), except AXEC2 uses the new mfast
try_filters results (whatever batches have completed so far) instead of the
old fixed 5-40 Hz AXEC2 results. The other 5 stations (AXAS1, AXAS2, AXCC1,
AXEC1, AXEC3) are unchanged (old/original results) since no mfast pipeline
exists for them.

Same QC gate, depth range/binning, and MIN_N_PER_BIN as the original:
success==True, dt>0, dt_error<0.04 s, phi_error<20 deg; depth in [0, 2] km,
0.125 km bins, >=5 events required per bin to plot a mean/SEM point.

Produces a distinctly-named PDF (does not overwrite the original):
    lqt_pykonal_combined_results/dt_vs_depth_by_station_mfast.pdf

Run with:
    python3 dt_vs_depth_by_station_mfast.py
"""

import glob
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
AXEC2_MFAST_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_lqt_pykonal_results')
AXEC2_MFAST_META = os.path.join(
    HERE, 'raw_axec2_all_batches_mfast_filters_data', 'raw_axec2_all_batches_mfast_filters_metadata.csv')
CATALOG_DIR = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR = DATA_DIR

PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

STATION_FILES = {
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv',
              'splitting_results_AXAS2_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
}

DEPTH_MIN, DEPTH_MAX, DEPTH_BIN_W = 0.0, 2.0, 0.125
DEPTH_BIN_EDGES = np.arange(DEPTH_MIN, DEPTH_MAX + DEPTH_BIN_W * .5, DEPTH_BIN_W)
DEPTH_BIN_MIDS = (DEPTH_BIN_EDGES[:-1] + DEPTH_BIN_EDGES[1:]) / 2.
MIN_N_PER_BIN = 5


def load_old_station(sta, catalog):
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['phi_error', 'dt_error'])
    df = df[(df['dt_error'] < DT_ERR_MAX) & (df['phi_error'] < PHI_ERR_MAX)]
    df['t'] = pd.to_datetime(df['datetime'], utc=True)

    sta_cat = catalog[catalog['station'] == sta].drop_duplicates(subset='event_datetime')
    df = df.merge(sta_cat[['event_datetime', 'event_depth']],
                  left_on='t', right_on='event_datetime', how='left')
    df = df.dropna(subset=['event_depth'])
    df['station'] = sta
    return df[['station', 'dt', 'event_depth']]


def load_axec2_mfast():
    batch_files = glob.glob(os.path.join(
        AXEC2_MFAST_DIR, 'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_*.csv'))
    dfs = [pd.read_csv(f) for f in batch_files]
    dfs = [d for d in dfs if len(d) > 0]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['phi_error', 'dt_error'])
    df = df[(df['dt_error'] < DT_ERR_MAX) & (df['phi_error'] < PHI_ERR_MAX)]

    meta = pd.read_csv(AXEC2_MFAST_META)[['event_id', 'depth']]
    df = df.merge(meta, on='event_id', how='left').dropna(subset=['depth'])
    df = df.rename(columns={'depth': 'event_depth'})
    df['station'] = 'AXEC2'
    return df[['station', 'dt', 'event_depth']]


def bin_stats(dt, depth):
    mids, means, sems, ns = [], [], [], []
    for lo, hi, mid in zip(DEPTH_BIN_EDGES[:-1], DEPTH_BIN_EDGES[1:], DEPTH_BIN_MIDS):
        sel = (depth >= lo) & (depth < hi)
        n = int(sel.sum())
        if n >= MIN_N_PER_BIN:
            vals = dt[sel]
            mids.append(mid)
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(n))
            ns.append(n)
    return np.array(mids), np.array(means), np.array(sems), np.array(ns)


def plot_panel(ax, dt, depth, title):
    ax.plot(depth, dt, '.', c='k', ms=2, alpha=0.25, zorder=2)
    mids, means, sems, ns = bin_stats(dt, depth)
    ax.errorbar(mids, means, yerr=sems, fmt='o', ms=8, color='tab:blue',
                ecolor='tab:blue', elinewidth=1.5, capsize=3, markeredgecolor='k',
                markeredgewidth=0.6, zorder=5, label='Mean dt (+/- SEM)')
    ax.set_xlim(DEPTH_MIN, DEPTH_MAX)
    ax.set_xlabel('Depth (km)', fontsize=8)
    ax.set_ylabel(r'$\delta t$ (s)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f'{title}\nN={len(dt):,}', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)


def main():
    print('Loading MLdd catalogs for event depths (old stations)...')
    cat1 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'))
    cat2 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'))
    cat1['event_datetime'] = pd.to_datetime(cat1['event_datetime'], utc=True, format='mixed')
    cat2['event_datetime'] = pd.to_datetime(cat2['event_datetime'], utc=True, format='mixed')
    catalog = pd.concat([cat1, cat2], ignore_index=True)

    print('Loading per-station QC-passing measurements with depth...')
    dfs = []
    for sta in STATIONS:
        if sta == 'AXEC2':
            df = load_axec2_mfast()
            print(f'  {sta} (mfast try_filters): {len(df):,} events with depth')
        else:
            df = load_old_station(sta, catalog)
            print(f'  {sta} (old): {len(df):,} events with depth')
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[all_df['event_depth'] <= DEPTH_MAX].copy()
    print(f'  TOTAL: {len(all_df):,} events with depth <= {DEPTH_MAX:g} km')

    n_sta = len(STATIONS)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    axes_flat = axes.ravel()

    for i, sta in enumerate(STATIONS):
        sub = all_df[all_df['station'] == sta]
        label = f'{sta} (mfast)' if sta == 'AXEC2' else sta
        plot_panel(axes_flat[i], sub['dt'].values, sub['event_depth'].values, label)

    plot_panel(axes_flat[n_sta], all_df['dt'].values, all_df['event_depth'].values,
               'All stations combined')
    axes_flat[0].legend(loc='upper right', fontsize=7)

    for j in range(n_sta + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(
        r'$\delta t$ vs. event depth, per station (AXEC2: mfast try_filters, others: old)'
        '\ndt_err<0.04 s & phi_err<20 deg, no dt/Q_w cutoff - '
        f'{DEPTH_BIN_W:g} km depth bins, N>={MIN_N_PER_BIN} required per bin',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(OUT_DIR, 'dt_vs_depth_by_station_mfast.pdf')
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
