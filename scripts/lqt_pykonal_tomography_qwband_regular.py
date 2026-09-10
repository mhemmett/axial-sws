#!/usr/bin/env python3
"""
lqt_pykonal_tomography_qwband_regular.py

"Regular" (absolute-value, NOT time-period-difference) versions of the
fullray, travel-time-weighted, and source-voxel anisotropy tomographies,
restricted to the Q_w mid-band -0.5 < Q_w < 0.5 instead of the sibling
scripts' six standard quality cuts -- dt_err/phi_err/dt-cutoff thresholds
unchanged. Reuses lqt_pykonal_tomography_time_period_difference.py's
_qw_band_mask() (single source of truth for this filter, shared with the
qwband_pm05 DIFF versions already built) so the two families can't drift
apart on the threshold definition.

Same standard rendering as the underlying "regular" scripts (single Blues
sequential colormap per page, one page per variant here since there's only
one filter, not six) -- NOT the two-colormap absolute/difference layout of
the _time_period_difference scripts.

Produces six PDFs (distinct names, does not overwrite any existing output):
    lqt_pykonal_tomography_fullray_qwband_pm05_7period.pdf        (1 page)
    lqt_pykonal_tomography_fullray_qwband_pm05_annual.pdf         (1 page)
    lqt_pykonal_tomography_traveltime_qwband_pm05_7period.pdf     (1 page)
    lqt_pykonal_tomography_traveltime_qwband_pm05_annual.pdf      (1 page)
    lqt_pykonal_tomography_source_voxel_qwband_pm05_7period.pdf   (2 pages: count_min 8, 20)
    lqt_pykonal_tomography_source_voxel_qwband_pm05_annual.pdf    (2 pages)

Run with:
    python3 lqt_pykonal_tomography_qwband_regular.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import lqt_pykonal_tomography_raylength_anisotropy as rla
import lqt_pykonal_tomography_traveltime_anisotropy as tta
import lqt_pykonal_tomography_source_voxel_anisotropy as svm
from lqt_pykonal_tomography_time_period_difference import _qw_band_mask, QW_BAND_LO, QW_BAND_HI

OUT_DIR = rla.OUT_DIR


# ── Fullray ──────────────────────────────────────────────────────────────────

def build_fullray():
    rla._setup_environment()
    print('\n[fullray, Q_w band, regular] Loading baseline events...')
    dfs = {sta: rla.load_station(sta) for sta in rla.STATIONS}
    all_df = pd.concat(dfs.values(), ignore_index=True)
    summary, vox = rla.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values
    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    a_ray_pct = summary['A_ray_pct'].values.astype(np.float64)

    mask, label = _qw_band_mask(summary)
    formula = 'A = dt·Vs·100/r (full ray value assigned to every crossed voxel)'
    cbar_label = 'Mean % shear anisotropy per voxel (full ray value)'

    schemes = [
        ('7period', rla.build_7_periods(all_df),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_fullray_qwband_pm05_7period.pdf')),
        ('annual', rla.build_annual(),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_fullray_qwband_pm05_annual.pdf')),
    ]
    for scheme_name, periods, out_path in schemes:
        ray_period = rla.assign_periods(event_ns, periods)
        results = rla.aggregate_page(vox, vray, mask, ray_period, periods,
                                     assign='fullray', a_ray_pct=a_ray_pct)
        n_total = int(mask.sum())
        print(f'  {scheme_name}: {label}: {n_total:,} rays')
        title = (f'LQT + PyKonal-FMM tomography — fullray % shear anisotropy per voxel '
                 f'({scheme_name})\n{formula}  (n={n_total:,}, filter: {label})')
        rla.write_pdf([(results, title)], out_path, cbar_label=cbar_label)


# ── Travel-time-weighted ─────────────────────────────────────────────────────

def build_traveltime():
    tta._setup_environment(need_tracer=False)
    print('\n[traveltime, Q_w band, regular] Loading baseline events...')
    dfs = {sta: tta.load_station(sta) for sta in tta.STATIONS}
    all_df = pd.concat(dfs.values(), ignore_index=True)
    summary, vox = tta.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values
    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8

    tt_seg, a_ray_tt = tta.compute_ttime_anisotropy(summary, vox, vray)
    mask, label = _qw_band_mask(summary)
    formula = 'A = δt·100/T_travel, travel-time-weighted mean over crossing rays'
    cbar_label = 'Travel-time-weighted mean % shear anisotropy per voxel'

    schemes = [
        ('7period', tta.build_7_periods(all_df),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_traveltime_qwband_pm05_7period.pdf')),
        ('annual', tta.build_annual(),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_traveltime_qwband_pm05_annual.pdf')),
    ]
    for scheme_name, periods, out_path in schemes:
        ray_period = tta.assign_periods(event_ns, periods)
        results = tta.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
        n_total = int(mask.sum())
        print(f'  {scheme_name}: {label}: {n_total:,} rays')
        title = (f'LQT + PyKonal-FMM tomography — travel-time-weighted mean % shear '
                 f'anisotropy per voxel ({scheme_name})\n{formula}  '
                 f'(n={n_total:,}, filter: {label})')
        tta.write_pdf([(results, title)], out_path, cbar_label=cbar_label)


# ── Source voxel ─────────────────────────────────────────────────────────────

def build_source_voxel():
    all_df, summary, a_ray_pct, in_grid, _page_filters, event_ns = svm.prepare_source_voxel_data()
    mask, label = _qw_band_mask(summary)
    formula = 'A = dt·Vs·100/r (full ray value, assigned only to the source voxel)'
    cbar_label = 'Mean % shear anisotropy in source voxel'

    schemes = [
        ('7period', rla.build_7_periods(all_df),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_qwband_pm05_7period.pdf')),
        ('annual', rla.build_annual(),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_qwband_pm05_annual.pdf')),
    ]
    for scheme_name, periods, out_path in schemes:
        ray_period = rla.assign_periods(event_ns, periods)
        pages = []
        for count_min in svm.COUNT_MIN_VARIANTS:
            results = svm.aggregate_source_voxel(
                all_df, a_ray_pct, mask, ray_period, periods, in_grid, count_min)
            n_total = int(mask.sum())
            print(f'  {scheme_name}: {label} (count_min={count_min}): {n_total:,} events')
            title = (f'LQT + PyKonal-FMM tomography — source-voxel % shear anisotropy '
                     f'({scheme_name})\n{formula}\n'
                     f'(n={n_total:,}, filter: {label}, min {count_min} events/voxel)')
            pages.append((results, title))
        rla.write_pdf(pages, out_path, cbar_label=cbar_label)


def main():
    build_fullray()
    build_traveltime()
    build_source_voxel()
    print('\nDone.')


if __name__ == '__main__':
    main()
