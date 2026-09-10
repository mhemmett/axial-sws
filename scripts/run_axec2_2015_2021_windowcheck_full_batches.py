"""
run_axec2_2015_2021_windowcheck_full_batches.py

Full-scale AXEC2 2015-2021 splitting run (all 497 batches, ~102,715 QC-passed events, mfast
per-event try_filters bandpass), using the WINDOW-CHECK fix
(swspy/swspy/splitting/split_windowcheck.py) instead of production swspy
(swspy/swspy/splitting/split.py). Extends the 10-batch validation test done in
run_axec2_mfast_filters_maxdt02_windowcheck_batches.py (batches 1-10 only, written to
production_axec2_mfast_filters_maxdt02_windowcheck_lqt_pykonal_results/) to the complete
2015-2021 dataset.

split_windowcheck.py is a byte-for-byte copy of split.py's active create_splitting_object
class, modified in __init__ only: if the shortest candidate grid-search window (latest start
window -> earliest end window) would be shorter than max_t_shift_s, the end-window range is
shifted out so the shortest window is exactly max_t_shift_s (keeping the end-window range's
span fixed at one dominant period, Tmid), rather than leaving windows shorter than
max_t_shift_s in the grid search where the cyclic np.roll lag-shift (_phi_dt_grid_search)
would wrap them around on themselves. See swspy/swspy/splitting/split_windowcheck.py's
__init__ docstring/comments for the exact adjustment. Critically, max_t_shift_s must be
passed to create_splitting_object() as a CONSTRUCTOR argument (new kwarg there) so the
window layout is computed with the correct value from the start -- NOT set as a
post-construction attribute override the way production's create_splitting_analysis /
perform_splitting_analysis do it (that happens too late for this fix, after __init__'s
window-layout logic has already run using the class's own 0.30s default).

Since splitting_functions.py's create_splitting_analysis() is hardwired to
swspy.splitting.create_splitting_object (production), this script does NOT edit that file.
Instead it defines create_splitting_analysis_windowcheck() (identical to production's
create_splitting_analysis() except it calls swc.create_splitting_object(...,
max_t_shift_s=MAX_T_SHIFT_S)) and monkey-patches splitting_functions.create_splitting_analysis
to point to it before calling perform_splitting_on_organized_waveforms() -- since
perform_splitting_analysis() (called internally by perform_splitting_on_organized_waveforms())
looks up create_splitting_analysis as a global name in splitting_functions' own module
namespace at call time, this redirect is picked up without touching production code.

Same raw mfast-filtered waveform data, same LQT frame + PyKonal-FMM incidence 35 deg cut, same
window/n_win/s_pick_uncertainty/max_t_shift_s=0.2 as the 10-batch test and as
run_axec2_mfast_filters_maxdt02_batches.py (unpatched production equivalent). ALL OUTPUT PATHS
ARE NEW (production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results/,
distinct from the 10-batch test's _..._windowcheck_lqt_pykonal_results/ directory), so nothing
in any previous run (production, maxdt02, or the 10-batch windowcheck test) is overwritten.
Resumable: skips batches whose results CSV already exists -- safe to run multiple instances in
parallel over disjoint --start/--end ranges, or to re-launch after an interruption.

After all batches finish, combine_axec2_2015_2021_windowcheck_batches.py concatenates every
per-batch CSV in the results directory into a single combined CSV.

Run with:
    python3 run_axec2_2015_2021_windowcheck_full_batches.py [--start N] [--end M]
(defaults: --start 1 --end 497, i.e. the full 2015-2021 dataset)
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import obspy
from obspy.core.utcdatetime import UTCDateTime

warnings.filterwarnings('ignore')

_SWSPY_LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swspy'))
if _SWSPY_LOCAL_PATH not in sys.path:
    sys.path.insert(0, _SWSPY_LOCAL_PATH)
import swspy  # noqa: F401
import swspy.splitting.split_windowcheck as swc

import splitting_functions
from splitting_functions import perform_splitting_on_organized_waveforms

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data')
RAW_METADATA_CSV = os.path.join(RAW_DATA_DIR, 'raw_axec2_all_batches_mfast_filters_metadata.csv')
RESULTS_DIR = os.path.join(HERE, 'production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results')

STATION = 'AXEC2'
INCIDENCE_FIELD = 'incidence_pykonal_s'
INCIDENCE_CUT_DEG = 35.0

FIRST_WINDOW_START = 2
LAST_WINDOW_START = 1
FIRST_WINDOW_END = 1.5
LAST_WINDOW_END = 2.5
N_WIN = 7
S_PICK_UNCERTAINTY = 0.0395
MAX_T_SHIFT_S = 0.2


def create_splitting_analysis_windowcheck(event_data, first_window_start, last_window_start, first_window_end, last_window_end, n_win, s_pick_uncertainty,
                                           incidence_field="incidence_eigenvalue_jurkevics", coord_system="LQT"):
    """Identical to splitting_functions.create_splitting_analysis(), except it builds the
    splitting object via swc.create_splitting_object (split_windowcheck.py) with
    max_t_shift_s passed in at CONSTRUCTION time, instead of swspy.splitting.create_splitting_object
    (split.py) with no max_t_shift_s kwarg (production sets that post-construction instead)."""

    event_traces = event_data.get('traces', [])
    if not event_traces:
        raise ValueError("No traces in event_data")

    if isinstance(event_traces, list):
        stream = obspy.Stream(event_traces)
    else:
        stream = event_traces.copy()

    stream_filtered = stream.copy()

    station_name = event_data.get('station', 'UNKNOWN')
    back_azimuth = event_data.get('back_azimuth')
    incidence_angle = event_data.get(incidence_field)

    if back_azimuth is None or np.isnan(back_azimuth):
        raise ValueError(f"Missing or invalid back_azimuth for station {station_name}")
    if incidence_angle is None or np.isnan(incidence_angle):
        raise ValueError(f"Missing or invalid incidence angle ('{incidence_field}') for station {station_name}")

    if coord_system == "ZNE":
        rotation_inclination = 0.0
    elif coord_system == "LQT":
        rotation_inclination = incidence_angle
    else:
        raise ValueError(f"coord_system must be 'LQT' or 'ZNE', got {coord_system!r}")

    event_time = UTCDateTime(event_data['datetime'])
    s_arrival_time = float(event_data['s_arrival_time'])
    s_arrival_absolute = event_time + s_arrival_time

    splitting_event = swc.create_splitting_object(
        stream_filtered,
        stations_in=[station_name],
        back_azis_all_stations=[back_azimuth],
        receiver_inc_angles_all_stations=[rotation_inclination],
        S_phase_arrival_times=[s_arrival_absolute],
        origin_times=[event_time],
        first_window_start=first_window_start,
        last_window_start=last_window_start,
        first_window_end=first_window_end,
        last_window_end=last_window_end, n_win=n_win,
        s_pick_uncertainty=s_pick_uncertainty,
        max_t_shift_s=MAX_T_SHIFT_S,
    )

    return splitting_event


# Monkey-patch: perform_splitting_analysis (called internally by
# perform_splitting_on_organized_waveforms) looks up create_splitting_analysis as a global
# name in splitting_functions' own module namespace at call time -- this redirect makes it
# use the window-check-aware constructor above, without editing splitting_functions.py.
splitting_functions.create_splitting_analysis = create_splitting_analysis_windowcheck


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
    print(f"max_t_shift_s: {MAX_T_SHIFT_S}s (window-check fix: undersized windows widened, not masked)")
    print(f"Batches {args.start}-{args.end}, output -> {RESULTS_DIR}/")

    t_start = time.time()
    total_events = 0
    for batch_num in range(args.start, args.end + 1):
        out_csv = os.path.join(
            RESULTS_DIR, f'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_batch_{batch_num}.csv')
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
            plot_results=False, max_t_shift_s=MAX_T_SHIFT_S
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
