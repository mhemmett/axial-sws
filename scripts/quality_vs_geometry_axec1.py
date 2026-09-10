#!/usr/bin/env python3
"""
quality_vs_geometry_axec1.py

Polar plot of splitting quality (Q_w, Wustefeld 2010) as a function of
back-azimuth (theta) and incidence angle (radius) for AXEC1's LQT + PyKonal-FMM
production results.

Data: lqt_pykonal_combined_results/splitting_results_AXEC1_2015_2021_all_batches.csv
      + splitting_results_AXEC1_2022_2026_all_batches.csv
      (AXEC1 already has both halves combined there, unlike AXEC2/AXEC3.)

Baseline cleanup only (success==True, dt>0) -- no quality/error thresholding,
so the full Q_w range (including nulls, Q_w<0) is visible, matching
dt_vs_quality_lqt_pykonal.py's approach.

Event locations are recovered by joining to the MLdd catalogs
(data/mldd_catalog_2015_2021.csv, data/mldd_catalog_2022_2026.csv) on
(station, event_datetime) -- same join verified exact-match in
lqt_pykonal_tomography_backprojection.py.

Back-azimuth (station -> event, deg from N) via obspy gps2dist_azimuth.
Incidence angle (deg from vertical) via the TRUE PyKonal FMM ray tracer
(pykonal_raytracer.BaillardRayTracer), same tracer used for the LQT frame
this production run's incidence angle ("35 deg cut") is built from.

Output: lqt_pykonal_combined_results/quality_vs_geometry_axec1_polar.pdf

Run with:
    python3 quality_vs_geometry_axec1.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from obspy.geodetics import gps2dist_azimuth

from pykonal_raytracer import BaillardRayTracer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
CATALOG_DIR = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR = DATA_DIR

STATION = 'AXEC1'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'

STATION_FILES = [
    'splitting_results_AXEC1_2015_2021_all_batches.csv',
    'splitting_results_AXEC1_2022_2026_all_batches.csv',
]

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


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

    print('Loading MLdd catalogs for event locations...')
    cat1 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'))
    cat2 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'))
    cat1['event_datetime'] = pd.to_datetime(cat1['event_datetime'], utc=True, format='mixed')
    cat2['event_datetime'] = pd.to_datetime(cat2['event_datetime'], utc=True, format='mixed')
    catalog = pd.concat([cat1, cat2], ignore_index=True)
    sta_cat = catalog[catalog['station'] == STATION].drop_duplicates(subset='event_datetime')

    df = df.merge(sta_cat[['event_datetime', 'event_lat', 'event_lon', 'event_depth']],
                  left_on='t', right_on='event_datetime', how='left')
    n_unmatched = df['event_lat'].isna().sum()
    if n_unmatched:
        print(f'  Dropping {n_unmatched:,} measurements with no catalog location match')
    df = df.dropna(subset=['event_lat', 'event_lon', 'event_depth'])
    print(f'  {len(df):,} measurements with recovered event location')

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                           engine='python').set_index('s')
    sta_lon, sta_lat = float(_sta_df.loc[STATION, 'lon']), float(_sta_df.loc[STATION, 'lat'])
    sta_x, sta_y = ll2xy(sta_lat, sta_lon)

    print('Computing back-azimuth (station -> event)...')
    back_azi = np.array([
        gps2dist_azimuth(sta_lat, sta_lon, elat, elon)[1]
        for elat, elon in zip(df['event_lat'], df['event_lon'])
    ])
    df['back_azi'] = back_azi

    print('Computing incidence angle via true PyKonal FMM ray tracing...')
    tracer = BaillardRayTracer(stride=5)
    tracer.precompute_station(STATION, float(sta_x), float(sta_y))

    ex, ey = ll2xy(df['event_lat'].values, df['event_lon'].values)
    ez = df['event_depth'].values
    incidence = np.full(len(df), np.nan)
    n_fail = 0
    for i in range(len(df)):
        try:
            incidence[i] = tracer.incidence_angle_at_station(STATION, float(ex[i]), float(ey[i]), float(ez[i]))
        except Exception:
            n_fail += 1
    df['incidence'] = incidence
    if n_fail:
        print(f'  {n_fail:,} incidence-angle trace failures (left as NaN)')
    df = df.dropna(subset=['incidence'])
    print(f'  {len(df):,} measurements with recovered back-azimuth + incidence angle')

    # ── Polar plot: theta = back-azimuth, r = incidence angle, color = quality ──
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    theta = np.radians(df['back_azi'].values)
    r = df['incidence'].values
    c = df['quality'].values

    vmax = max(abs(c.min()), abs(c.max()))
    sc = ax.scatter(theta, r, c=c, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                     s=10, alpha=0.7, edgecolors='none')

    ax.set_rlabel_position(225)
    ax.set_ylim(0, 90)
    ax.set_title(f'{STATION} — quality Q$_w$ vs. back-azimuth and incidence angle\n'
                 f'(LQT + PyKonal-FMM, N={len(df):,})',
                 fontsize=11, fontweight='bold', pad=20)

    cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.75)
    cb.set_label('Quality Q$_w$ (Wustefeld 2010)', fontsize=9)

    out_path = os.path.join(OUT_DIR, 'quality_vs_geometry_axec1_polar.pdf')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
