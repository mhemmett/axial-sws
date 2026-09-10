#!/usr/bin/env python3
"""
process_bpr_detided_depth_ccal.py

Central Caldera counterpart of process_bpr_detided_depth.py (that script's BPR pair,
RS03ECAL-MJ03E-06-BOTPTA302, is the EASTERN Caldera instrument -- this one,
RS03CCAL-MJ03F-05-BOTPTA301, is the actual Central Caldera BOTPT, per explicit user
clarification when these two new files were added to data/):

    ooi-rs03ccal-mj03f-05-botpta301_27cd_0236_3f9d_mean_depth.csv  -- botsflu_meandepth [m]
    ooi-rs03ccal-mj03f-05-botpta301_14cb_9708_32e5_pred_tide.csv   -- botsflu_predtide  [m]

Same format/convention as the Eastern Caldera pair (1-minute-sampled, identical time grid
2014-07-25 21:05 UTC to present, 6,312,684 rows each, confirmed row-for-row identical
timestamps; qc_agg NOT used as a filter -- see process_bpr_detided_depth.py's docstring for
why). Same processing:

  1. De-tide: detided_depth_m = seafloor_depth_m - predicted_tide_m.
  2. Trim to the SWS catalog's time span, 2015-01-22 to 2026-05-15.
  3. Drop rows where either raw value is NaN.

Produces (NEW file, data/ is gitignored -- not tracked, does not overwrite either input or the
Eastern Caldera output):
    data/bpr_detided_seafloor_depth_ccal_2015-01-22_to_2026-05-15.csv
    columns: time (UTC), seafloor_depth_m, predicted_tide_m, detided_depth_m

Run with:
    python3 process_bpr_detided_depth_ccal.py
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')

TIDE_CSV = os.path.join(DATA_DIR, 'ooi-rs03ccal-mj03f-05-botpta301_14cb_9708_32e5_pred_tide.csv')
DEPTH_CSV = os.path.join(DATA_DIR, 'ooi-rs03ccal-mj03f-05-botpta301_27cd_0236_3f9d_mean_depth.csv')
OUT_CSV = os.path.join(DATA_DIR, 'bpr_detided_seafloor_depth_ccal_2015-01-22_to_2026-05-15.csv')

T_START = pd.Timestamp('2015-01-22', tz='UTC')
T_END = pd.Timestamp('2026-05-16', tz='UTC')   # exclusive upper bound -> covers all of May 15


def main():
    print(f'Loading {os.path.basename(TIDE_CSV)} ...')
    tide = pd.read_csv(TIDE_CSV, skiprows=[1], usecols=['time', 'botsflu_predtide'])
    print(f'Loading {os.path.basename(DEPTH_CSV)} ...')
    depth = pd.read_csv(DEPTH_CSV, skiprows=[1], usecols=['time', 'botsflu_meandepth'])

    assert (tide['time'].values == depth['time'].values).all(), \
        'tide/depth files are not row-for-row time-aligned'
    print(f'  {len(tide):,} rows, confirmed row-for-row time-aligned')

    t = pd.to_datetime(tide['time'], utc=True)
    df = pd.DataFrame({
        'time': t,
        'seafloor_depth_m': depth['botsflu_meandepth'].values,
        'predicted_tide_m': tide['botsflu_predtide'].values,
    })

    print('De-tiding: detided_depth_m = seafloor_depth_m - predicted_tide_m ...')
    df['detided_depth_m'] = df['seafloor_depth_m'] - df['predicted_tide_m']

    mask = (df['time'] >= T_START) & (df['time'] < T_END)
    df = df.loc[mask].reset_index(drop=True)
    print(f'Trimmed to [{T_START.date()}, {T_END.date()}): {len(df):,} rows')

    n_before = len(df)
    df = df.dropna(subset=['detided_depth_m']).reset_index(drop=True)
    print(f'Dropped {n_before - len(df):,} rows with missing seafloor_depth_m/predicted_tide_m '
          f'-> {len(df):,} rows remain')

    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_CSV}')
    print(df['detided_depth_m'].describe())


if __name__ == '__main__':
    main()
