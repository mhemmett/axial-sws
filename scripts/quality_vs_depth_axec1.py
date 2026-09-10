#!/usr/bin/env python3
"""
quality_vs_depth_axec1.py

2D density histogram of splitting quality (Q_w, Wustefeld 2010) vs. source
earthquake depth, for AXEC1's LQT + PyKonal-FMM production results.

Data: lqt_pykonal_combined_results/splitting_results_AXEC1_2015_2021_all_batches.csv
      + splitting_results_AXEC1_2022_2026_all_batches.csv
      (AXEC1 already has both halves combined there.)

Baseline cleanup only (success==True, dt>0) -- no quality/error thresholding,
so the full Q_w range (including nulls, Q_w<0) is visible, matching
dt_vs_quality_lqt_pykonal.py's / quality_vs_geometry_axec1.py's approach.

Event depths are recovered by joining to the MLdd catalogs
(data/mldd_catalog_2015_2021.csv, data/mldd_catalog_2022_2026.csv) on
(station, event_datetime) -- same join verified exact-match earlier this
session.

Output: lqt_pykonal_combined_results/quality_vs_depth_axec1.pdf

Run with:
    python3 quality_vs_depth_axec1.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
CATALOG_DIR = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR = DATA_DIR

STATION = 'AXEC1'
STATION_FILES = [
    'splitting_results_AXEC1_2015_2021_all_batches.csv',
    'splitting_results_AXEC1_2022_2026_all_batches.csv',
]


def load_axec1():
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True)
    return df


def main():
    print(f'Loading {STATION} LQT + PyKonal-FMM combined results...')
    df = load_axec1()
    print(f'  {len(df):,} measurements (baseline cleanup only)')

    print('Loading MLdd catalogs for event depths...')
    cat1 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'))
    cat2 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'))
    cat1['event_datetime'] = pd.to_datetime(cat1['event_datetime'], utc=True, format='mixed')
    cat2['event_datetime'] = pd.to_datetime(cat2['event_datetime'], utc=True, format='mixed')
    catalog = pd.concat([cat1, cat2], ignore_index=True)
    sta_cat = catalog[catalog['station'] == STATION].drop_duplicates(subset='event_datetime')

    df = df.merge(sta_cat[['event_datetime', 'event_depth']],
                  left_on='t', right_on='event_datetime', how='left')
    n_unmatched = df['event_depth'].isna().sum()
    if n_unmatched:
        print(f'  Dropping {n_unmatched:,} measurements with no catalog depth match')
    df = df.dropna(subset=['event_depth'])
    print(f'  {len(df):,} measurements with recovered event depth')

    fig, ax = plt.subplots(figsize=(8, 6))
    x_edges = np.linspace(df['event_depth'].min(), df['event_depth'].max(), 81)
    y_edges = np.linspace(df['quality'].min(), df['quality'].max(), 81)
    counts, xe, ye = np.histogram2d(df['event_depth'], df['quality'], bins=[x_edges, y_edges])
    im = ax.pcolormesh(xe, ye, counts.T, cmap='viridis',
                        norm=mcolors.LogNorm(vmin=1, vmax=counts.max()))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Count', fontsize=9)

    ax.set_xlabel('Source depth [km]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Quality Q$_w$ (Wustefeld 2010)', fontsize=11, fontweight='bold')
    ax.set_title(f'{STATION} — quality Q$_w$ vs. source depth (LQT + PyKonal-FMM, N={len(df):,})',
                 fontsize=11, fontweight='bold')

    out_path = os.path.join(OUT_DIR, 'quality_vs_depth_axec1.pdf')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
