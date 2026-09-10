#!/usr/bin/env python3
"""
process_bpr_detided_depth.py

De-tides the Central Caldera bottom-pressure-recorder (BOTPT) seafloor-depth record and trims
it to the SWS catalog's time span, per explicit user request. Two new (not-tracked, data/ is
gitignored) OOI data files were added to data/:

    ooi-rs03ecal-mj03e-06-botpta302_27cd_0236_3f9d_mean_depth.csv  -- botsflu_meandepth [m]
    ooi-rs03ecal-mj03e-06-botpta302_14cb_9708_32e5_pred_tide.csv   -- botsflu_predtide  [m]

Both are 1-minute-sampled, identical time grid (2014-09-03 21:00 UTC to present, 6,256,606 rows
each, confirmed row-for-row identical timestamps), each with a 'z' metadata column (always 0.0,
irrelevant) and a qc_agg column. qc_agg is NOT used as a filter here: qc=2 ("not evaluated") is
the overwhelming majority of rows INCLUDING rows with valid (non-NaN) values, so QC status does
not indicate whether a value is present -- only NaN does. The processing here is:

  1. De-tide: detided_depth_m = seafloor_depth_m - predicted_tide_m (removes the tidal
     oscillation, leaving the secular/deformation signal, per explicit user instruction: "we
     want to subtract the predicted tide from the seafloor depth, first").
  2. Trim to the SWS catalog's time span, 2015-01-22 to 2026-05-15 (the min/max event datetimes
     across this session's windowcheck grade-3 dataset -- see rose_7period_regions_windowcheck_
     grade3.py), per explicit user instruction ("filter the timing to match the start and end
     dates of our data").
  3. Drop rows where either raw value is NaN (can't de-tide without both).

Produces (NEW file, data/ is gitignored -- not tracked, does not overwrite either input):
    data/bpr_detided_seafloor_depth_2015-01-22_to_2026-05-15.csv
    columns: time (UTC), seafloor_depth_m, predicted_tide_m, detided_depth_m

Run with:
    python3 process_bpr_detided_depth.py
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')

TIDE_CSV = os.path.join(DATA_DIR, 'ooi-rs03ecal-mj03e-06-botpta302_14cb_9708_32e5_pred_tide.csv')
DEPTH_CSV = os.path.join(DATA_DIR, 'ooi-rs03ecal-mj03e-06-botpta302_27cd_0236_3f9d_mean_depth.csv')
OUT_CSV = os.path.join(DATA_DIR, 'bpr_detided_seafloor_depth_2015-01-22_to_2026-05-15.csv')

# Matches the SWS windowcheck grade-3 catalog's time span (see
# rose_7period_regions_windowcheck_grade3.py: all_combined['t'].min()/.max()).
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
