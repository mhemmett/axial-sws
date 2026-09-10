#!/usr/bin/env python3
"""
enrich_lqt_pykonal_results_geometry.py

Adds geometry columns (event lat/lon/depth, back-azimuth, PyKonal-ray-traced
incidence angle) to the production LQT + PyKonal-FMM combined results for
all 6 stations, 2015-2026 -- to check whether the sampled raypath geometry
(back-azimuth, incidence, depth) drifts or varies significantly over the
10-year record in a way that could confound the observed temporal changes
in splitting (the same check Illsley-Kemp et al. (2018) ran at Afar).

IMPORTANT: this writes brand-new '*_enriched.csv' files. It NEVER modifies,
overwrites, or touches the original 'splitting_results_<STA>_<period>_
all_batches.csv' files in any way -- those are read strictly read-only.

Columns added (every row of the original file is preserved, in the same
order; only these columns are appended):
    event_lat, event_lon, event_depth   -- from the MLdd catalog, joined on
                                            (station, event_datetime); NaN if
                                            no catalog match
    back_azimuth                        -- station -> event, deg from N,
                                            true geodetic azimuth via
                                            obspy.geodetics.gps2dist_azimuth
                                            (station coords from
                                            data/axial_seamount_stations.csv)
    incidence                           -- deg from vertical, via the
                                            corrected PyKonal-FMM ray tracer
                                            (pykonal_raytracer.BaillardRayTracer,
                                            reused from
                                            lqt_pykonal_tomography_raylength_
                                            anisotropy's already-precomputed,
                                            all-6-station tracer -- NOT the
                                            legacy Jurkevics P-wave incidence)
    snr_horizontal, rectilinearity_jurkevics
                                         -- NaN PLACEHOLDER columns only.
                                            These require raw 3-component
                                            waveforms via
                                            calculate_snr_for_organized_
                                            waveforms / calculate_
                                            rectilinearity_jurkevics_for_
                                            organized_waveforms
                                            (splitting_functions.py), which
                                            are only available in data/ for
                                            AXEC1/AXEC3 2015-2021 -- deferred
                                            until the rest of the raw mseed
                                            is located.

Run with:
    python3 enrich_lqt_pykonal_results_geometry.py
"""

import warnings

warnings.filterwarnings('ignore')

import os

import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth

import lqt_pykonal_tomography_raylength_anisotropy as rla

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, 'lqt_pykonal_combined_results')
CATALOG_DIR = os.path.join(REPO_ROOT, 'data')
STATIONS_CSV = os.path.join(CATALOG_DIR, 'axial_seamount_stations.csv')

STATIONS = rla.STATIONS
PERIODS = ['2015_2021', '2022_2026']
CATALOG_FILES = {
    '2015_2021': os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'),
    '2022_2026': os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'),
}


def load_station_coords():
    df = pd.read_csv(STATIONS_CSV).set_index('Station ID')
    return {sta: (float(df.loc[sta, 'Latitude (°N)']), float(df.loc[sta, 'Longitude (°W)']))
            for sta in df.index}


def enrich_one(sta, period, sta_lat, sta_lon, catalog):
    in_path = os.path.join(DATA_DIR, f'splitting_results_{sta}_{period}_all_batches.csv')
    out_path = os.path.join(DATA_DIR, f'splitting_results_{sta}_{period}_all_batches_enriched.csv')

    df = pd.read_csv(in_path)
    n_orig = len(df)
    df['_t'] = pd.to_datetime(df['datetime'], utc=True)

    sta_cat = catalog[catalog['station'] == sta].drop_duplicates(subset='event_datetime')
    merged = df.merge(sta_cat[['event_datetime', 'event_lat', 'event_lon', 'event_depth']],
                      left_on='_t', right_on='event_datetime', how='left')
    assert len(merged) == n_orig, f'{sta} {period}: row count changed on merge ({n_orig} -> {len(merged)})'
    n_matched = merged['event_lat'].notna().sum()

    back_azimuth = np.full(len(merged), np.nan)
    incidence = np.full(len(merged), np.nan)
    elat = merged['event_lat'].values
    elon = merged['event_lon'].values
    edepth = merged['event_depth'].values
    have_loc = merged['event_lat'].notna().values
    n_inc_fail = 0
    for i in np.nonzero(have_loc)[0]:
        try:
            back_azimuth[i] = gps2dist_azimuth(sta_lat, sta_lon, float(elat[i]), float(elon[i]))[1]
        except Exception:
            pass
        try:
            ex, ey = rla.ll2xy(elat[i], elon[i])
            incidence[i] = rla.tracer.incidence_angle_at_station(sta, float(ex), float(ey), float(edepth[i]))
        except Exception:
            n_inc_fail += 1

    merged['back_azimuth'] = back_azimuth
    merged['incidence'] = incidence
    merged['snr_horizontal'] = np.nan
    merged['rectilinearity_jurkevics'] = np.nan

    out_cols = [c for c in df.columns if c != '_t'] + [
        'event_lat', 'event_lon', 'event_depth', 'back_azimuth', 'incidence',
        'snr_horizontal', 'rectilinearity_jurkevics',
    ]
    merged[out_cols].to_csv(out_path, index=False)

    n_inc_ok = int(np.isfinite(incidence).sum())
    print(f'  {sta} {period}: {n_orig:,} rows -> {out_path}')
    print(f'    catalog-matched: {n_matched:,}/{n_orig:,}  '
          f'incidence resolved: {n_inc_ok:,}/{n_matched:,} ({n_inc_fail:,} ray-trace failures)')
    if n_inc_ok:
        print(f'    incidence range: [{np.nanmin(incidence):.1f}, {np.nanmax(incidence):.1f}] deg  '
              f'back_azimuth range: [{np.nanmin(back_azimuth):.1f}, {np.nanmax(back_azimuth):.1f}] deg')
    return n_orig


def main():
    print('Setting up ray tracer (all 6 stations precomputed)...')
    rla._setup_environment()

    print('Loading station coordinates...')
    sta_coords = load_station_coords()

    print('Loading MLdd catalogs...')
    catalogs = {}
    for period, path in CATALOG_FILES.items():
        cat = pd.read_csv(path)
        cat['event_datetime'] = pd.to_datetime(cat['event_datetime'], utc=True, format='mixed')
        catalogs[period] = cat

    total_rows = 0
    for sta in STATIONS:
        sta_lat, sta_lon = sta_coords[sta]
        print(f'\n=== {sta} (lat={sta_lat:.4f}, lon={sta_lon:.4f}) ===')
        for period in PERIODS:
            total_rows += enrich_one(sta, period, sta_lat, sta_lon, catalogs[period])

    print(f'\nTotal rows enriched across all station/period files: {total_rows:,}')
    print('Done. Originals were not modified -- new *_enriched.csv files written alongside them.')


if __name__ == '__main__':
    main()
