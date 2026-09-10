#!/usr/bin/env python3
"""
lqt_pykonal_tomography_source_voxel_per_station_qwband_pm05.py

Per-station breakdown of the Q_w mid-band (-0.5 < Q_w < 0.5, "indeterminate
quality") source-voxel percent-anisotropy tomography
(lqt_pykonal_tomography_qwband_regular.py's build_source_voxel(), which only
produces the 6-station-combined map). Each of the 6 stations gets its own
page, following the exact per-station layout of
lqt_pykonal_tomography_source_voxel_per_station.py (which covers the strict/
loose Q_w filters, not the mid-band) -- this script fills the one gap between
those two: is AXEC1's mid-band population spatially distinct from the other
5 stations' own mid-band populations?

Reuses lqt_pykonal_tomography_source_voxel_anisotropy.prepare_source_voxel_
data() and .aggregate_source_voxel() verbatim (same cache, same alignment
checks, same voxel-binning) and lqt_pykonal_tomography_time_period_
difference._qw_band_mask() (single source of truth for the -0.5/0.5 band,
shared with qwband_regular.py and the qwband_pm05 DIFF scripts, so the
threshold can't drift between scripts).

Only one filter per station (the mid-band itself; strict/loose are already
covered by the sibling per-station script) -- coverage gate: COUNT_MIN=8
events/voxel (same default as the combined qwband script).

Produces two PDFs (distinct names, does not overwrite any existing output):
    lqt_pykonal_tomography_source_voxel_per_station_qwband_pm05_7period.pdf   (6 pages)
    lqt_pykonal_tomography_source_voxel_per_station_qwband_pm05_annual.pdf    (6 pages)

Run with:
    python3 lqt_pykonal_tomography_source_voxel_per_station_qwband_pm05.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import lqt_pykonal_tomography_raylength_anisotropy as rla
from lqt_pykonal_tomography_source_voxel_anisotropy import (
    prepare_source_voxel_data, aggregate_source_voxel,
)
from lqt_pykonal_tomography_time_period_difference import _qw_band_mask

OUT_DIR = rla.OUT_DIR
COUNT_MIN = 8


def main():
    all_df, summary, a_ray_pct, in_grid, _page_filters, event_ns = prepare_source_voxel_data()

    band_mask, band_label = _qw_band_mask(summary)
    station = summary['station'].values

    formula = 'A = dt·Vs·100/r (full ray value, assigned only to the source voxel)'

    schemes = [
        ('7period', rla.build_7_periods(all_df),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_per_station_qwband_pm05_7period.pdf')),
        ('annual', rla.build_annual(),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_per_station_qwband_pm05_annual.pdf')),
    ]

    for scheme_name, periods, out_path in schemes:
        ray_period = rla.assign_periods(event_ns, periods)
        pages = []
        for sta in rla.STATIONS:
            sta_bool = (station == sta)
            mask = band_mask & sta_bool
            print(f'\n=== {sta} ({scheme_name}, Q_w band) ===')
            results = aggregate_source_voxel(
                all_df, a_ray_pct, mask, ray_period, periods, in_grid, COUNT_MIN)
            n_total = int(mask.sum())
            covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
            print(f'  {band_label}: {n_total:,} events  covered-voxels={covered:,}')
            title = (f'LQT + PyKonal-FMM tomography — source-voxel % shear anisotropy — '
                     f'{sta} ({scheme_name})\n{formula}\n'
                     f'(n={n_total:,}, filter: {band_label}, min {COUNT_MIN} events/voxel)')
            pages.append((results, title, sta))
        rla.write_pdf(pages, out_path,
                      cbar_label='Mean % shear anisotropy in source voxel')

    print('\nDone.')


if __name__ == '__main__':
    main()
