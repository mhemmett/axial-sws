#!/usr/bin/env python3
"""
lqt_pykonal_tomography_source_voxel_per_station.py

Per-station breakdown of the source-voxel percent-anisotropy tomography
(lqt_pykonal_tomography_source_voxel_anisotropy.py): each of the 6 stations
gets its own page (depth rows x period columns), rather than all 6 stations
being combined into one voxel grid -- so a real spatial difference between,
say, AXAS1's and AXEC2's own hypocentre-anisotropy patterns is visible rather
than averaged away.

Only 2 filters per station (not the full 6-filter/2-threshold sweep of the
combined script):
  - "strict": dt_err<0.04s & phi_err<20 deg & Q_w>=0.7 & dt<=0.24s
  - "loose":  dt_err<0.04s & phi_err<20 deg & Q_w>=-0.5 & dt<=0.24s
(the same two filter definitions already used by
lqt_pykonal_tomography_source_voxel_anisotropy.py's shared
rla.build_page_filters() -- indices 3 and 5 there -- so thresholds can never
drift between the two scripts.)

Reuses lqt_pykonal_tomography_source_voxel_anisotropy.prepare_source_voxel_
data() and .aggregate_source_voxel() verbatim (same cache, same alignment
checks, same voxel-binning), just adding a per-station boolean mask (from
ray_summary.csv's 'station' column) on top of each of the 2 filters.

Coverage gate: COUNT_MIN=8 events/voxel (the combined script's default;
per-station data is sparser than the combined 6-station set, so this is not
re-swept at 20 the way the combined script's threshold comparison was).

Produces two PDFs (distinct names, does not overwrite any existing output):
    lqt_pykonal_tomography_source_voxel_per_station_7period.pdf   (12 pages: 6 stations x 2 filters)
    lqt_pykonal_tomography_source_voxel_per_station_annual.pdf    (12 pages)

Run with:
    python3 lqt_pykonal_tomography_source_voxel_per_station.py
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

OUT_DIR = rla.OUT_DIR
COUNT_MIN = 8

# Indices into rla.build_page_filters()'s returned list (see that function's
# docstring / lqt_pykonal_tomography_raylength_anisotropy.py for the full set
# of 6): 3 = strict (Q_w>=0.7), 5 = loose (Q_w>=-0.5). Both already include
# the dt_err/phi_err/dt-cutoff thresholds "as before".
STRICT_FILTER_IDX = 3
LOOSE_FILTER_IDX = 5


def main():
    all_df, summary, a_ray_pct, in_grid, page_filters, event_ns = prepare_source_voxel_data()

    strict_label, strict_mask = page_filters[STRICT_FILTER_IDX]
    loose_label, loose_mask = page_filters[LOOSE_FILTER_IDX]
    assert 'Q_w≥0.7' in strict_label or 'Q_w>=0.7' in strict_label, \
        f'unexpected filter at index {STRICT_FILTER_IDX}: {strict_label!r}'
    assert 'Q_w≥-0.5' in loose_label or 'Q_w>=-0.5' in loose_label, \
        f'unexpected filter at index {LOOSE_FILTER_IDX}: {loose_label!r}'

    station = summary['station'].values
    filters = [('strict (Q_w≥0.7)', strict_label, strict_mask),
               ('loose (Q_w≥-0.5)', loose_label, loose_mask)]

    formula = 'A = dt·Vs·100/r (full ray value, assigned only to the source voxel)'

    schemes = [
        ('7period', rla.build_7_periods(all_df),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_per_station_7period.pdf')),
        ('annual', rla.build_annual(),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_per_station_annual.pdf')),
    ]

    for scheme_name, periods, out_path in schemes:
        ray_period = rla.assign_periods(event_ns, periods)
        pages = []
        for sta in rla.STATIONS:
            sta_bool = (station == sta)
            print(f'\n=== {sta} ({scheme_name}) ===')
            for short_label, full_label, base_mask in filters:
                mask = base_mask & sta_bool
                results = aggregate_source_voxel(
                    all_df, a_ray_pct, mask, ray_period, periods, in_grid, COUNT_MIN)
                n_total = int(mask.sum())
                covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
                print(f'  {short_label:22s}: {n_total:,} events  covered-voxels={covered:,}')
                title = (f'LQT + PyKonal-FMM tomography — source-voxel % shear anisotropy — '
                         f'{sta} ({scheme_name})\n{formula}\n'
                         f'(n={n_total:,}, filter: {full_label}, min {COUNT_MIN} events/voxel)')
                pages.append((results, title, sta))
        rla.write_pdf(pages, out_path,
                      cbar_label='Mean % shear anisotropy in source voxel')

    print('\nDone.')


if __name__ == '__main__':
    main()
