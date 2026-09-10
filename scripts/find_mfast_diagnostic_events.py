"""
Search for AXEC2 mfast mid-dt (0.15-0.24s) and high-dt (>0.24s) diagnostic events where the
AD HOC RE-RUN (create_splitting_analysis + perform_sws_analysis, same as the
figure_axec2_mfast_event_*_diagnostics.py scripts) itself yields Q_w >= 0.7, phi_error < 20,
dt_error < 0.04 -- not just the production CSV's stored Q_w, which can differ from a
re-run's Q_w due to small waveform-trimming differences (see figure_axec1_event_23919
docstring). Iterates over production QC-passing candidates (sorted by production quality
descending) within each dt range and actually re-runs each one, stopping at the first hit.
"""

import glob
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import obspy
import pandas as pd
from obspy.core.utcdatetime import UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'swspy'))
import swspy  # noqa: E402

sys.path.insert(0, HERE)
from splitting_functions import create_splitting_analysis  # noqa: E402

STATION = 'AXEC2'
MFAST_DATA_DIR = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data')
META_CSV = os.path.join(MFAST_DATA_DIR, 'raw_axec2_all_batches_mfast_filters_metadata.csv')

INCIDENCE_FIELD = 'incidence_pykonal_s'
FIRST_WINDOW_START = 2
LAST_WINDOW_START = 1
FIRST_WINDOW_END = 1.5
LAST_WINDOW_END = 2.5
N_WIN = 7
S_PICK_UNCERTAINTY = 0.0395
COORD_SYSTEM = 'LQT'
SWS_METHOD = 'EV_and_XC'


def try_event(row):
    waveform_path = os.path.join(MFAST_DATA_DIR, row['waveform_file'])
    st = obspy.read(waveform_path)
    event_data = {
        'traces': st,
        'station': STATION,
        'datetime': row['datetime'],
        's_arrival_time': row['s_arrival_time'],
        'p_arrival_time': row['p_arrival_time'],
        'back_azimuth': row['back_azimuth'],
        'incidence_pykonal_s': row['incidence_pykonal_s'],
        'snr_horizontal': row['snr_horizontal'],
        'rectilinearity_jurkevics': row['rectilinearity_jurkevics'],
        'magnitude': 0.0,
    }
    splitting_obj = create_splitting_analysis(
        event_data,
        first_window_start=FIRST_WINDOW_START, last_window_start=LAST_WINDOW_START,
        first_window_end=FIRST_WINDOW_END, last_window_end=LAST_WINDOW_END,
        n_win=N_WIN, s_pick_uncertainty=S_PICK_UNCERTAINTY,
        incidence_field=INCIDENCE_FIELD, coord_system=COORD_SYSTEM,
    )
    splitting_obj.perform_sws_analysis(coord_system=COORD_SYSTEM, sws_method=SWS_METHOD)
    r = splitting_obj.sws_result_df.iloc[0]
    return dict(dt=float(r['dt']), phi=float(r['phi_from_Q']), phi_err=float(r['phi_err']),
               dt_err=float(r['dt_err']), Q_w=float(r['Q_w']))


def main():
    result_files = glob.glob('production_axec2_mfast_filters_lqt_pykonal_results/'
                              'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_*.csv')
    dfs = [pd.read_csv(f) for f in result_files]
    dfs = [d for d in dfs if len(d) > 0]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality', 'phi_error', 'dt_error'])
    qc = df[(df['quality'] > 0.7) & (df['phi_error'] < 20.0) & (df['dt_error'] < 0.04)]

    meta = pd.read_csv(META_CSV)

    ranges = {
        'mid': qc[(qc['dt'] > 0.15) & (qc['dt'] < 0.24)].sort_values('quality', ascending=False),
        'high': qc[qc['dt'] > 0.24].sort_values('quality', ascending=False),
    }

    for label, cand in ranges.items():
        print(f'\n=== Searching {label}-dt candidates (n={len(cand)}) ===')
        found = False
        for _, res_row in cand.head(40).iterrows():
            eid = res_row['event_id']
            meta_row = meta[meta['event_id'] == eid]
            if len(meta_row) == 0:
                continue
            meta_row = meta_row.iloc[0]
            try:
                result = try_event(meta_row)
            except Exception as e:
                print(f'  {eid}: FAILED ({e})')
                continue
            ok = (result['Q_w'] >= 0.7) and (result['phi_err'] < 20.0) and (result['dt_err'] < 0.04)
            marker = ' <-- MATCH' if ok else ''
            print(f'  {eid}: prod_dt={res_row["dt"]:.3f} prod_Q={res_row["quality"]:.3f} | '
                  f'rerun dt={result["dt"]:.3f} phi_err={result["phi_err"]:.1f} '
                  f'dt_err={result["dt_err"]:.3f} Q_w={result["Q_w"]:.3f}{marker}')
            if ok:
                print(f'\n  FOUND {label}-dt event: {eid}')
                print(f'    batch={meta_row["batch"]}, waveform_file={meta_row["waveform_file"]}')
                print(f'    datetime={meta_row["datetime"]}')
                print(f'    s_arrival_time={meta_row["s_arrival_time"]}, p_arrival_time={meta_row["p_arrival_time"]}')
                print(f'    back_azimuth={meta_row["back_azimuth"]}, incidence_pykonal_s={meta_row["incidence_pykonal_s"]}')
                print(f'    snr_horizontal={meta_row["snr_horizontal"]}, rectilinearity={meta_row["rectilinearity_jurkevics"]}')
                print(f'    chosen_filter_band=({meta_row["chosen_filter_min"]}, {meta_row["chosen_filter_max"]})')
                found = True
                break
        if not found:
            print(f'  No {label}-dt match found in top 40 candidates.')


if __name__ == '__main__':
    main()
