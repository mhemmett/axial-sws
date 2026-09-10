#!/usr/bin/env python3
"""
sws_tomography_johnson2011_windowcheck_grade3_ccal_periods.py

Windowcheck counterpart of sws_tomography_johnson2011_newdata_grade3_7period.py, per explicit
user request: same Johnson, Savage & Townend (2011) 2-D delay-time SWS tomography machinery
(quad-tree gridding, G/d system assembly, covariance whitening, bounded WLS solve, per-block
circular phi averaging, checkerboard-recovery test -- all reused UNCHANGED via import from
sws_tomography_johnson2011.py as J and the newdata script as S), JOINT across all 6 stations
(AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3), but with two changes:

  1. DATA: this session's windowcheck (mfast max_dt=0.2s + split_windowcheck.py lag-shift-
     wraparound fix) results instead of "newdata" -- same GRADE/load_station_raw/apply_grade
     convention as rose_7period_regions_windowcheck_grade3.py (SNR>=2.0, quality>=0.75,
     dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg). AXEC2's 2015-2021 half (the special combined
     CSV) has no latitude/longitude/depth columns and so CANNOT be ray-traced -- only its
     2022-2026 half contributes here (same limitation already documented in
     spatial_midpoint_aniso_phi_windowcheck_grade3_7period_1row_countmin10.py).
  2. PERIODS: replaces the equal-COUNT 7-period scheme with the 7-period EQUAL-INFLATION-AMOUNT
     scheme from bpr_inflation_periods_ccal.py (Central Caldera BOTPT, RS03CCAL-MJ03F-05-
     BOTPTA301) -- Before Eruption / During Eruption unchanged, 5 post-eruption bins of equal
     physical uplift instead of equal event count.

Produces TWO figures (1 row x 7 periods each), same conventions as the newdata script:
    sws_tomography_johnson2011_windowcheck_grade3_ccal_periods_strength.pdf
    sws_tomography_johnson2011_windowcheck_grade3_ccal_periods_phi_quiver.pdf

Run with:
    python3 sws_tomography_johnson2011_windowcheck_grade3_ccal_periods.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sws_tomography_johnson2011 as J
import sws_tomography_johnson2011_newdata_grade3_7period as S
from pykonal_raytracer import BaillardRayTracer
from sws_forward_model import ll2xy
import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
from rose_7period_regions_windowcheck_grade3 import GRADE, load_station_raw, apply_grade
from bpr_inflation_periods_ccal import build_inflation_based_periods

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'sws_tomography_johnson2011_windowcheck_grade3_ccal_periods_results')
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PDF_STRENGTH = os.path.join(HERE, 'sws_tomography_johnson2011_windowcheck_grade3_ccal_periods_strength.pdf')
OUT_PDF_PHI = os.path.join(HERE, 'sws_tomography_johnson2011_windowcheck_grade3_ccal_periods_phi_quiver.pdf')

STATIONS = J.STATIONS
RAY_ID_OFFSET = S.RAY_ID_OFFSET


def load_station_windowcheck_grade3(sta):
    """Windowcheck grade-3 events with valid coordinates, in the same schema
    load_station_newdata_grade3() produces (event_id, t, dt, dt_error, phi, phi_error,
    quality, x, y, z)."""
    raw = load_station_raw(sta)
    df = apply_grade(raw, GRADE)
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'event_id']).copy()
    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    return df[['event_id', 't', 'dt', 'dt_error', 'phi', 'phi_error', 'quality',
               'x', 'y', 'z']].reset_index(drop=True)


def get_station_cache(sta, tracer):
    s_csv = os.path.join(OUT_DIR, f'{sta}_ray_summary.csv')
    c_csv = os.path.join(OUT_DIR, f'{sta}_ray_finecells.csv')
    if os.path.exists(s_csv) and os.path.exists(c_csv):
        print(f'  {sta}: loading cached rays')
        summ = pd.read_csv(s_csv, parse_dates=['t'])
        cells = pd.read_csv(c_csv, dtype={'ray_id': np.int64, 'ixf': np.int32,
                                          'iyf': np.int32, 'seg_len_km': np.float64})
        return summ, cells
    df = load_station_windowcheck_grade3(sta)
    print(f'  {sta}: tracing {len(df):,} grade-3 rays...')
    return J.build_ray_finecell_cache(sta, df, tracer, out_dir=OUT_DIR)


def build_joint_cache():
    sta_xy = J.station_xy()
    tracer = BaillardRayTracer(stride=5)
    for sta in STATIONS:
        tracer.precompute_station(sta, *sta_xy[sta])

    per_summ, per_cells = [], []
    station_index = {}
    for si, sta in enumerate(STATIONS):
        summ_s, cells_s = get_station_cache(sta, tracer)
        max_rid = int(summ_s['ray_id'].max())
        assert max_rid < RAY_ID_OFFSET, f'{sta}: ray_id overflow vs offset'
        base = si * RAY_ID_OFFSET
        summ_s = summ_s.copy()
        cells_s = cells_s.copy()
        summ_s['ray_id'] = summ_s['ray_id'].astype(np.int64) + base
        cells_s['ray_id'] = cells_s['ray_id'].astype(np.int64) + base
        summ_s['station'] = sta
        per_summ.append(summ_s)
        per_cells.append(cells_s)
        station_index[sta] = si

    joint_summ = pd.concat(per_summ, ignore_index=True)
    joint_cells = pd.concat(per_cells, ignore_index=True)
    assert joint_summ['ray_id'].is_unique, 'ray_id collision across stations'
    return joint_summ, joint_cells, station_index, sta_xy


def main():
    T._setup_environment(need_tracer=False)

    print('Building/loading JOINT six-station grade-3 ray cache (windowcheck)...')
    joint_summ, joint_cells, station_index, sta_xy = build_joint_cache()
    print(f'\nJoint cache: {len(joint_summ):,} rays total across {len(STATIONS)} stations')

    t_parsed = pd.to_datetime(joint_summ['t'], utc=True, format='mixed')
    periods = build_inflation_based_periods()
    period_labels = [lbl for lbl, _t0, _t1 in periods]
    print('\nEqual-inflation-amount 7-period scheme (Central Caldera BOTPT):')
    for label, t0, t1 in periods:
        t0s = t0.date() if t0 is not None else '(start of record)'
        t1s = t1.date() if t1 is not None else '(present)'
        print(f'  {label!r:35s}  {t0s} -> {t1s}')

    print('\n=== Running Johnson et al. (2011) chain independently per period ===')
    results = []
    for lbl, t0, t1 in periods:
        print(f'\n-- {lbl.replace(chr(10), " ")} --')
        try:
            res = S.run_chain_grade3_period(joint_summ, joint_cells, t0, t1)
        except ValueError as exc:
            print(f'  [SKIP] {exc}')
            res = None
        results.append(res)

    print('\nBuilding figures...')
    fig1 = S.make_strength_figure(results, period_labels)
    # make_strength_figure/make_phi_quiver_figure are reused verbatim from the newdata
    # script and hardcode "(newdata, grade-3 filter)" in their suptitle -- fix up here
    # rather than forking the plotting functions just to change a title string.
    if fig1._suptitle:
        fig1._suptitle.set_text(fig1._suptitle.get_text().replace(
            'newdata, grade-3 filter', 'windowcheck, grade-3 filter, equal-inflation periods'))
    with PdfPages(OUT_PDF_STRENGTH) as pdf:
        pdf.savefig(fig1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved {OUT_PDF_STRENGTH}')

    fig2 = S.make_phi_quiver_figure(results, period_labels)
    if fig2._suptitle:
        fig2._suptitle.set_text(fig2._suptitle.get_text().replace(
            'newdata, grade-3 filter', 'windowcheck, grade-3 filter, equal-inflation periods'))
    with PdfPages(OUT_PDF_PHI) as pdf:
        pdf.savefig(fig2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {OUT_PDF_PHI}')


if __name__ == '__main__':
    main()
