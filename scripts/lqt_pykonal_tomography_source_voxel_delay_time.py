#!/usr/bin/env python3
"""
lqt_pykonal_tomography_source_voxel_delay_time.py

Delay-time variant of lqt_pykonal_tomography_source_voxel_anisotropy.py's
7-period PDF: same source-voxel binning (each event's raw measured delay time
dt is assigned to the single voxel containing its own hypocentre, no path
apportionment), same 6 QC filters x 2 count-min thresholds (8, 20), same
grid/period/depth-row/bathymetry rendering -- but the plotted, averaged, and
colour-mapped quantity is the RAW MEASURED dt [s] itself, NOT A_ray_pct
(dt*Vs*100/r). No anisotropy formula is applied here at all.

Reuses lqt_pykonal_tomography_source_voxel_anisotropy.prepare_source_voxel_
data() and .aggregate_source_voxel() verbatim -- aggregate_source_voxel() is
generic over "the per-ray scalar to average per voxel", so passing dt instead
of A_ray_pct requires no new aggregation logic, only new titles/colorbar
wording so it's never confused with the anisotropy version.

Produces one PDF (distinct name, does not overwrite any existing output):
    lqt_pykonal_tomography_source_voxel_delay_time_7period.pdf   (12 pages)

Run with:
    python3 lqt_pykonal_tomography_source_voxel_delay_time.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np

import lqt_pykonal_tomography_raylength_anisotropy as rla
from lqt_pykonal_tomography_source_voxel_anisotropy import (
    prepare_source_voxel_data, aggregate_source_voxel, COUNT_MIN_VARIANTS,
)

OUT_DIR = rla.OUT_DIR


def main():
    all_df, summary, _a_ray_pct, in_grid, page_filters, event_ns = prepare_source_voxel_data()
    dt_vals = summary['dt'].values.astype(np.float64)

    periods = rla.build_7_periods(all_df)
    ray_period = rla.assign_periods(event_ns, periods)

    pages = []
    for count_min in COUNT_MIN_VARIANTS:
        print(f'\n=== Aggregating source-voxel delay time, 7period, count_min={count_min} ===')
        for filt_label, ray_mask in page_filters:
            results = aggregate_source_voxel(
                all_df, dt_vals, ray_mask, ray_period, periods, in_grid, count_min)
            n_total = int(ray_mask.sum())
            covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
            print(f'  {filt_label:60s}: {n_total:,} events  covered-voxels={covered:,}')
            title = (f'LQT + PyKonal-FMM tomography — mean delay time at source voxel (7period)\n'
                     f'Raw measured δt [s], no anisotropy formula applied\n'
                     f'(n={n_total:,}, filter: {filt_label}, min {count_min} events/voxel)')
            pages.append((results, title))

    out_path = os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_delay_time_7period.pdf')
    rla.write_pdf(pages, out_path, cbar_label='Mean delay time δt in source voxel [s]')

    print('\nDone.')


if __name__ == '__main__':
    main()
