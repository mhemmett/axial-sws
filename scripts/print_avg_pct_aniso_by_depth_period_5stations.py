#!/usr/bin/env python3
"""
print_avg_pct_aniso_by_depth_period_5stations.py

Prints the map-averaged travel-time-weighted percent anisotropy for each (depth row, time
period) combination from traveltime_anisotropy_7period_5stations_newdata.py's 5-station
(AXAS1/AXCC1/AXEC1/AXEC2/AXEC3) tomography -- same data, QC, ray tracing, voxel grid, and
aggregation as that script (imported directly, not reimplemented), just printed as a table
instead of (in addition to) the contour-map PDF.

For each depth row, each voxel's value is the travel-time-weighted mean % anisotropy averaged
over that row's 3 z-cells (np.nanmean over the z-slice, same as the sibling script's
make_page); the printed number is the nanmean of THAT 2D slice over all occupied (x,y)
voxels -- i.e. a coverage-gated spatial average, not a per-ray average (voxels with no
COUNT_MIN/MIN_PATH_KM-gated coverage are excluded, not zero-filled).

Run with:
    python3 print_avg_pct_aniso_by_depth_period_5stations.py
"""

import numpy as np
import pandas as pd

from traveltime_anisotropy_7period_5stations_newdata import (
    _setup_environment, load_station, build_7_periods, trace_all, assign_periods,
    aggregate, INCLUDED_STATIONS, DEPTH_ROWS, QW_MIN, PHI_ERR_MAX, DT_ERR_MAX,
)


def main():
    _setup_environment()

    print('Loading new-data QC-filtered events (quality>=0.5, dt<T_dom/2, '
          f'phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s)...')
    dfs = {}
    for sta in INCLUDED_STATIONS:
        dfs[sta] = load_station(sta)
        print(f'  {sta}: {len(dfs[sta]):,}')

    all_df = pd.concat(dfs.values(), ignore_index=True)
    periods = build_7_periods(all_df)

    (event_ns, ray_A, vox_ray_id, vox_flat, vox_seg, vox_tt) = trace_all(dfs)
    ray_period = assign_periods(event_ns, periods)
    results = aggregate(ray_period, ray_A, vox_ray_id, vox_flat, vox_seg, vox_tt, periods)

    period_labels = [r['label'].splitlines()[0] for r in results]
    depth_labels = [zlbl for _iz, zlbl, _z0 in DEPTH_ROWS]

    table = np.full((len(DEPTH_ROWS), len(results)), np.nan)
    for ri, ((iz0, iz1), _zlbl, _z0) in enumerate(DEPTH_ROWS):
        for ci, res in enumerate(results):
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            if np.isfinite(sl).any():
                table[ri, ci] = np.nanmean(sl)

    col_w = max(len(lbl) for lbl in period_labels) + 2
    row_w = max(len(lbl) for lbl in depth_labels) + 2

    header = ' ' * row_w + ''.join(f'{lbl:>{col_w}}' for lbl in period_labels)
    print('\nMap-averaged travel-time-weighted mean % anisotropy, by depth row x time period:')
    print(header)
    for ri, zlbl in enumerate(depth_labels):
        row = f'{zlbl:<{row_w}}' + ''.join(
            f'{table[ri, ci]:>{col_w}.2f}' if np.isfinite(table[ri, ci]) else f'{"n/a":>{col_w}}'
            for ci in range(len(results)))
        print(row)


if __name__ == '__main__':
    main()
