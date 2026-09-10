#!/usr/bin/env python3
"""
print_avg_pct_aniso_by_depth_period_5stations_countmin20.py

Same as print_avg_pct_aniso_by_depth_period_5stations.py (5-station AXAS1/AXCC1/AXEC1/
AXEC2/AXEC3 travel-time-weighted % anisotropy, map-averaged per depth row x time period),
but with the per-voxel coverage gate raised from COUNT_MIN=8 to COUNT_MIN=20 crossing rays
(MIN_PATH_KM=0.5 unchanged) -- a stricter coverage requirement before a voxel's mean is
trusted enough to plot/average. Monkey-patches
traveltime_anisotropy_7period_5stations_newdata's module-level COUNT_MIN before calling its
aggregate(), so the threshold change is applied via the same aggregation code, not
reimplemented.

Run with:
    python3 print_avg_pct_aniso_by_depth_period_5stations_countmin20.py
"""

import numpy as np
import pandas as pd

import traveltime_anisotropy_7period_5stations_newdata as tmod

COUNT_MIN_OVERRIDE = 20
tmod.COUNT_MIN = COUNT_MIN_OVERRIDE


def main():
    tmod._setup_environment()

    print(f'Using COUNT_MIN={tmod.COUNT_MIN} (MIN_PATH_KM={tmod.MIN_PATH_KM} unchanged)')
    print('Loading new-data QC-filtered events (quality>=0.5, dt<T_dom/2, '
          f'phi_err<{tmod.PHI_ERR_MAX:.0f}deg, dt_err<{tmod.DT_ERR_MAX}s)...')
    dfs = {}
    for sta in tmod.INCLUDED_STATIONS:
        dfs[sta] = tmod.load_station(sta)
        print(f'  {sta}: {len(dfs[sta]):,}')

    all_df = pd.concat(dfs.values(), ignore_index=True)
    periods = tmod.build_7_periods(all_df)

    (event_ns, ray_A, vox_ray_id, vox_flat, vox_seg, vox_tt) = tmod.trace_all(dfs)
    ray_period = tmod.assign_periods(event_ns, periods)
    results = tmod.aggregate(ray_period, ray_A, vox_ray_id, vox_flat, vox_seg, vox_tt, periods)

    period_labels = [r['label'].splitlines()[0] for r in results]
    depth_labels = [zlbl for _iz, zlbl, _z0 in tmod.DEPTH_ROWS]

    table = np.full((len(tmod.DEPTH_ROWS), len(results)), np.nan)
    for ri, ((iz0, iz1), _zlbl, _z0) in enumerate(tmod.DEPTH_ROWS):
        for ci, res in enumerate(results):
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            if np.isfinite(sl).any():
                table[ri, ci] = np.nanmean(sl)

    col_w = max(len(lbl) for lbl in period_labels) + 2
    row_w = max(len(lbl) for lbl in depth_labels) + 2

    header = ' ' * row_w + ''.join(f'{lbl:>{col_w}}' for lbl in period_labels)
    print(f'\nMap-averaged travel-time-weighted mean % anisotropy (COUNT_MIN={COUNT_MIN_OVERRIDE}), '
          'by depth row x time period:')
    print(header)
    for ri, zlbl in enumerate(depth_labels):
        row = f'{zlbl:<{row_w}}' + ''.join(
            f'{table[ri, ci]:>{col_w}.2f}' if np.isfinite(table[ri, ci]) else f'{"n/a":>{col_w}}'
            for ci in range(len(results)))
        print(row)


if __name__ == '__main__':
    main()
