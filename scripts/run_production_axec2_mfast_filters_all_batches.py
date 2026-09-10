"""
Production splitting run for AXEC2, all batches, mfast-like per-event bandpass variant.
Mirrors run_production_axec2_all_batches.py exactly (same LQT frame + PyKonal-FMM
incidence 35 deg cut, same window/n_win/s_pick_uncertainty), except it reads from
raw_axec2_all_batches_mfast_filters_data/ (built by
build_raw_axec2_all_batches_mfast_filters.py), where each event's waveform was
already bandpass-filtered exactly once, with the band chosen per-event by
try_filters rather than a fixed 5-40 Hz band. No further filtering happens here or
in create_splitting_analysis/perform_splitting_analysis - the chosen filtering is
reused as-is.

Resumable: skips batches whose results CSV already exists. All output paths are new
(separate from the standard production run's raw_axec2_all_batches_data/ and
production_axec2_lqt_pykonal_results/), so nothing existing is overwritten.

Run with:
    python3 run_production_axec2_mfast_filters_all_batches.py [--start N] [--end M]
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import obspy

warnings.filterwarnings('ignore')

_SWSPY_LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swspy'))
if _SWSPY_LOCAL_PATH not in sys.path:
    sys.path.insert(0, _SWSPY_LOCAL_PATH)
import swspy  # noqa: F401

from splitting_functions import perform_splitting_on_organized_waveforms

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data')
RAW_METADATA_CSV = os.path.join(RAW_DATA_DIR, 'raw_axec2_all_batches_mfast_filters_metadata.csv')
RESULTS_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_lqt_pykonal_results')

STATION = 'AXEC2'
INCIDENCE_FIELD = 'incidence_pykonal_s'
INCIDENCE_CUT_DEG = 35.0

FIRST_WINDOW_START = 2
LAST_WINDOW_START = 1
FIRST_WINDOW_END = 1.5
LAST_WINDOW_END = 2.5
N_WIN = 7
S_PICK_UNCERTAINTY = 0.0395


def build_organized_waveforms(batch_df):
    organized = {}
    for _, row in batch_df.iterrows():
        waveform_path = os.path.join(RAW_DATA_DIR, row['waveform_file'])
        try:
            st = obspy.read(waveform_path)
        except Exception:
            continue
        organized[row['event_id']] = {
            'traces': st,
            'station': row['station'],
            'datetime': row['datetime'],
            's_arrival_time': row['s_arrival_time'],
            'p_arrival_time': row['p_arrival_time'],
            'back_azimuth': row['back_azimuth'],
            'snr_horizontal': row['snr_horizontal'],
            'rectilinearity_jurkevics': row['rectilinearity_jurkevics'],
            'incidence_eigenvalue_jurkevics': row['incidence_p_jurkevics'],
            'incidence_pykonal_s': row['incidence_pykonal_s'],
            'magnitude': row['magnitude'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'depth': row['depth'],
            'chosen_filter_min': row.get('chosen_filter_min'),
            'chosen_filter_max': row.get('chosen_filter_max'),
            'chosen_filter_dom_period_samples': row.get('chosen_filter_dom_period_samples'),
        }
    return organized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=497)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading full mfast-filters raw-rebuilt metadata...")
    meta = pd.read_csv(RAW_METADATA_CSV)
    print(f"Total QC-passed (SNR+rectilinearity) events: {len(meta)}")

    t_start = time.time()
    total_events = 0
    for batch_num in range(args.start, args.end + 1):
        out_csv = os.path.join(RESULTS_DIR, f'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_{batch_num}.csv')
        if os.path.exists(out_csv):
            continue

        batch_df = meta[meta['batch'] == batch_num]
        if len(batch_df) == 0:
            pd.DataFrame(columns=[
                'event_id', 'datetime', 'phi', 'dt', 'phi_error', 'dt_error', 'quality', 'success',
                'magnitude', 'snr_horizontal', 'rectilinearity_jurkevics',
                'incidence_eigenvalue_jurkevics', 'incidence_pykonal_s', 'back_azimuth',
                'chosen_filter_min', 'chosen_filter_max', 'chosen_filter_dom_period_samples',
            ]).to_csv(out_csv, index=False)
            continue

        organized_all = build_organized_waveforms(batch_df)
        filtered = {eid: ed for eid, ed in organized_all.items()
                    if not np.isnan(ed[INCIDENCE_FIELD]) and ed[INCIDENCE_FIELD] <= INCIDENCE_CUT_DEG}

        t0 = time.time()
        results = perform_splitting_on_organized_waveforms(
            filtered, FIRST_WINDOW_START, LAST_WINDOW_START, FIRST_WINDOW_END, LAST_WINDOW_END,
            N_WIN, S_PICK_UNCERTAINTY, mode='swspy',
            coord_system='LQT', sws_method='EV_and_XC', incidence_field=INCIDENCE_FIELD,
            plot_results=False
        )

        rows = []
        for eid, r in results.items():
            res = r['result']
            ed = organized_all[eid]
            rows.append({
                'event_id': eid, 'datetime': ed['datetime'],
                'phi': res.get('phi'), 'dt': res.get('dt'),
                'phi_error': res.get('phi_error'), 'dt_error': res.get('dt_error'),
                'quality': res.get('quality'), 'success': res.get('success'),
                'magnitude': ed.get('magnitude'),
                'snr_horizontal': ed.get('snr_horizontal'),
                'rectilinearity_jurkevics': ed.get('rectilinearity_jurkevics'),
                'incidence_eigenvalue_jurkevics': ed.get('incidence_eigenvalue_jurkevics'),
                'incidence_pykonal_s': ed.get('incidence_pykonal_s'),
                'back_azimuth': ed.get('back_azimuth'),
                'chosen_filter_min': ed.get('chosen_filter_min'),
                'chosen_filter_max': ed.get('chosen_filter_max'),
                'chosen_filter_dom_period_samples': ed.get('chosen_filter_dom_period_samples'),
            })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        total_events += len(rows)

        elapsed = time.time() - t_start
        print(f"  batch {batch_num}: {len(filtered)}/{len(organized_all)} passed incidence cut, "
              f"{len(rows)} results ({time.time()-t0:.1f}s) [{elapsed/60:.1f} min elapsed, {total_events} total so far]")

    print(f"\nDone. Results in {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
