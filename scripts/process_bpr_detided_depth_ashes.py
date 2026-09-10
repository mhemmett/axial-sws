#!/usr/bin/env python3
"""
process_bpr_detided_depth_ashes.py

ASHES vent field (Western Caldera) counterpart of process_bpr_detided_depth.py /
process_bpr_detided_depth_ccal.py, built from the newly added BPR pair:

    ooi-rs03ashs-mj03b-09-botpta304_27cd_0236_3f9d_mean_depth.csv  -- botsflu_meandepth [m]
    ooi-rs03ashs-mj03b-09-botpta304_14cb_9708_32e5_pred_tide.csv   -- botsflu_predtide  [m]

IMPORTANT DIFFERENCE from the Eastern/Central Caldera pairs: this instrument's record starts
2017-08-15 (not 2014), so it has NO coverage of the 2015 eruption itself or the ~2.3 years of
re-inflation immediately after it. Same processing otherwise (1-minute-sampled, identical time
grid, 4,707,241 rows each, confirmed row-for-row identical timestamps; qc_agg not used as a
filter):

  1. De-tide: detided_depth_m = seafloor_depth_m - predicted_tide_m.
  2. Trim to the SWS catalog's time span, 2015-01-22 to 2026-05-15 (in practice this just
     intersects with the available 2017-08-15 to 2026-05-15 stretch of real data).
  3. Drop rows where either raw value is NaN.

Produces (NEW file, data/ is gitignored -- not tracked, does not overwrite any other output):
    data/bpr_detided_seafloor_depth_ashes_2015-01-22_to_2026-05-15.csv
    columns: time (UTC), seafloor_depth_m, predicted_tide_m, detided_depth_m

Run with:
    python3 process_bpr_detided_depth_ashes.py
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')

TIDE_CSV = os.path.join(DATA_DIR, 'ooi-rs03ashs-mj03b-09-botpta304_14cb_9708_32e5_pred_tide.csv')
DEPTH_CSV = os.path.join(DATA_DIR, 'ooi-rs03ashs-mj03b-09-botpta304_27cd_0236_3f9d_mean_depth.csv')
OUT_CSV = os.path.join(DATA_DIR, 'bpr_detided_seafloor_depth_ashes_2015-01-22_to_2026-05-15.csv')

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
    print(f'  Actual data span: {df["time"].min()} to {df["time"].max()}')

    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_CSV}')
    print(df['detided_depth_m'].describe())


if __name__ == '__main__':
    main()
