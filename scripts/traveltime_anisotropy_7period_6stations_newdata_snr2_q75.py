#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_6stations_newdata_snr2_q75.py

Single-grade (regular + Δ-from-pre-eruption) travel-time-weighted percent-anisotropy
tomography for the newdata 6-station dataset, filter:

    SNR >= 2.0, phi_err <= 20 deg, dt_err <= 0.05 s, dt <= T_dom/2, quality (Q_w) >= 0.75

This is GRADES[3] ("Page 1 @ quality>=0.75") from
traveltime_anisotropy_7period_6stations_newdata_snr_grades.py -- REUSES that script's
already-built ray-tracing cache (traveltime_anisotropy_newdata_snr_grades_ray_summary.csv /
_ray_voxel_segments.csv, traced once against the broadest quality>=0.5 filter, of which this
grade is a strict subset) rather than re-tracing. All formula/grid/plotting code is imported
directly from that sibling script, not reimplemented.

Produces two NEW PDFs (does not touch any existing output):
    traveltime_anisotropy_7period_6stations_newdata_snr2_q75.pdf       (regular, 1 page)
    traveltime_anisotropy_7period_6stations_newdata_snr2_q75_diff.pdf  (Δ-from-pre-eruption, 1 page)

Run with:
    python3 traveltime_anisotropy_7period_6stations_newdata_snr2_q75.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF_REGULAR = os.path.join(HERE, 'traveltime_anisotropy_7period_6stations_newdata_snr2_q75.pdf')
OUT_PDF_DIFF = os.path.join(HERE, 'traveltime_anisotropy_7period_6stations_newdata_snr2_q75_diff.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg
assert GRADE['key'] == 'page1_q0.75', GRADE


def main():
    cache_exists = os.path.exists(T.RAY_SUMMARY_CSV) and os.path.exists(T.RAY_VOXEL_CSV)
    if not cache_exists:
        raise SystemExit(
            f'Ray-tracing cache not found ({T.RAY_SUMMARY_CSV} / {T.RAY_VOXEL_CSV}). '
            'Run traveltime_anisotropy_7period_6stations_newdata_snr_grades.py first to build it.')

    T._setup_environment(need_tracer=False)

    print(f'Reusing existing ray-tracing cache:\n  {T.RAY_SUMMARY_CSV}\n  {T.RAY_VOXEL_CSV}')
    summary, vox = T.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    all_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))

    print('\nComputing T_travel per ray and A_ray = dt*100/T_travel from cache...')
    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    mask = ((summary['quality'].values >= GRADE['qw_min']) &
            (summary['snr_horizontal'].values >= GRADE['snr_min']) &
            (summary['phi_error'].values <= GRADE['phi_err_max']))
    ray_period = T.assign_periods(event_ns, periods)

    results = T.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
    n_total = int(mask.sum())
    covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
    print(f'  {GRADE["label"]}: {n_total:,} rays  covered-voxels={covered:,}')

    formula = 'A = δt·100/T_travel (T_travel = Σ seg_len_km/Vs_voxel over crossed voxels)'

    title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — travel-time-weighted mean %% '
             f'anisotropy per voxel (7-period)\n{formula}\n{GRADE["label"]} (n={n_total:,})')
    fig = T.make_page(results, title)
    with PdfPages(OUT_PDF_REGULAR) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF_REGULAR}')

    diff_title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — pre-eruption anisotropy '
                   f'+ difference from pre-eruption (7-period)\n{formula}\n'
                   f'{GRADE["label"]} (n={n_total:,})')
    fig2 = T.make_diff_page(results, diff_title)
    with PdfPages(OUT_PDF_DIFF) as pdf:
        pdf.savefig(fig2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {OUT_PDF_DIFF}')

    print('\nDone.')


if __name__ == '__main__':
    main()
