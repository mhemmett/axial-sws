#!/usr/bin/env python3
"""
newdata_final_filter.py

Shared loader for the user's settled "final" QC filter (per the 2026-07-27 daily note):
    SNR >= 2.0, phi_err <= 20 deg, dt_err <= 0.05 s, dt <= T_dom/2, Q_w >= 0.75

Same real-data source as traveltime_anisotropy_7period_6stations_newdata_snr_grades.py's
`load_station_broadest`/`page1_q0.75` grade -- the complete mfast_maxdt_pipeline_transfer/
splitting_results_{STA}_{2015_2021,2022_2026}_all_batches.csv files for all 6 stations
(NOT the separate in-progress AXEC2-only maxdt02 partial re-run used elsewhere in this repo),
so all six stations stay on identical, complete processing.

Exposes STATION_ORDER, ERUPTION_START/END, and load_station_final_filtered(sta), which returns
a dataframe with columns ['t', 'phi_az', 'dt', 'dominant_period', ...] already filtered to the
final QC.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')

STATION_ORDER = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

STATION_FILES = {
    sta: [f'splitting_results_{sta}_2015_2021_all_batches.csv',
          f'splitting_results_{sta}_2022_2026_all_batches.csv']
    for sta in STATION_ORDER
}

SNR_MIN = 2.0
QW_MIN = 0.75
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05

FILTER_LABEL = (f'SNR >= {SNR_MIN}, phi_err <= {PHI_ERR_MAX:.0f} deg, dt_err <= {DT_ERR_MAX}s, '
                f'dt <= T_dom/2, Q_w >= {QW_MIN}')

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')


def load_station_raw(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    df['phi_az'] = df['phi'] % 180.0
    return df


def apply_final_filter(df):
    d = df.dropna(subset=['quality', 'snr_horizontal', 'phi_error', 'dt_error', 'dominant_period'])
    d = d[(d['snr_horizontal'] >= SNR_MIN) &
          (d['quality'] >= QW_MIN) &
          (d['phi_error'] <= PHI_ERR_MAX) &
          (d['dt_error'] <= DT_ERR_MAX) &
          (d['dt'] <= d['dominant_period'] / 2.0)]
    return d


def load_station_final_filtered(sta):
    return apply_final_filter(load_station_raw(sta))


if __name__ == '__main__':
    print(f'Final filter: {FILTER_LABEL}\n')
    for sta in STATION_ORDER:
        raw = load_station_raw(sta)
        filt = load_station_final_filtered(sta)
        print(f'  {sta}: {len(raw):,} baseline (success & dt>0) -> {len(filt):,} after final filter')
