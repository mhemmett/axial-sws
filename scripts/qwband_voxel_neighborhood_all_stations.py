#!/usr/bin/env python3
"""
qwband_voxel_neighborhood_all_stations.py

Cross-station control for axec1_qwband_voxel_neighborhood.py: is AXEC1's
broad ~0.28-0.50 mid_frac band across its whole near-station neighborhood
(no clean strict-dominated contrast voxel anywhere nearby) a distinctive
property of AXEC1, or is it just what every station's own near-station
shallow volume looks like?

For each of the 6 stations, in its own map window (padded around its own
voxel, same padding/depth-slice logic as the AXEC1 drill-down) and for the
same 3 depth slices (iz=1,2,3 -> 0.38/0.62/0.88 km), computes mid_frac =
n_mid/(n_mid+n_strict) for every voxel with >=MIN_TOTAL combined mid+strict
rays FROM THAT STATION's OWN rays only (not pooled with other stations).
Reports, per station and per depth: count of well-sampled voxels, and the
median/IQR/min/max of mid_frac among them, plus counts of "strict-dominated"
(mid_frac<0.15) and "mid-dominated" (mid_frac>0.65) large-N (>=50) outliers.

Reuses the same cache/mask/window logic as axec1_qwband_voxel_neighborhood.py
and axec1_qwband_ray_voxel_overlap.py (ray_voxel_segments cache,
_qw_band_mask()).

Run with:
    python3 qwband_voxel_neighborhood_all_stations.py
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import lqt_pykonal_tomography_raylength_anisotropy as rla
from lqt_pykonal_tomography_time_period_difference import _qw_band_mask

MIN_TOTAL = 10        # min mid+strict rays to consider a voxel at all
LARGE_N = 50          # min mid+strict rays to count as a "well-sampled" voxel for the summary stats
IX_PAD, IY_PAD = 8, 8
IZ_RANGE = [1, 2, 3]


def voxel_ray_counts(vray, flat, ray_bool):
    keep = ray_bool[vray]
    if not keep.any():
        return {}
    f = flat[keep]
    uniq, cnt = np.unique(f, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))


def main():
    rla._setup_environment()

    summary, vox = rla.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'

    ok = summary['trace_ok'].values.astype(bool)
    dte = summary['dt_error'].values
    phe = summary['phi_error'].values
    dt = summary['dt'].values
    q = summary['quality'].values
    station = summary['station'].values

    elig = ok & (dte < rla.DT_ERR_MAX) & (phe < rla.PHI_ERR_MAX) & (dt <= rla.DT_CUTOFF)
    mid_mask, mid_label = _qw_band_mask(summary)
    strict_mask = elig & (q >= 0.7)

    vray = vox['ray_id'].values
    ix_all = vox['ix'].values.astype(np.int64)
    iy_all = vox['iy'].values.astype(np.int64)
    iz_all = vox['iz'].values.astype(np.int64)
    flat = ix_all * (rla.NY * rla.NZ) + iy_all * rla.NZ + iz_all

    print(f'Mid-band definition: {mid_label}\n')

    all_rows = []
    for sta in rla.STATIONS:
        sta_x, sta_y = rla.sta_xy[sta]
        sta_ix = int(np.floor((sta_x - rla.X_START) / rla.VXY))
        sta_iy = int(np.floor((sta_y - rla.Y_START) / rla.VXY))

        foc = (station == sta)
        mid_foc = mid_mask & foc
        strict_foc = strict_mask & foc
        cnt_mid = voxel_ray_counts(vray, flat, mid_foc)
        cnt_strict = voxel_ray_counts(vray, flat, strict_foc)

        ix_lo, ix_hi = max(0, sta_ix - IX_PAD), min(rla.NX, sta_ix + IX_PAD + 1)
        iy_lo, iy_hi = max(0, sta_iy - IY_PAD), min(rla.NY, sta_iy + IY_PAD + 1)

        print(f'=== {sta}  (voxel ix={sta_ix}, iy={sta_iy}; x={sta_x:.2f}, y={sta_y:.2f} km) ===')
        for iz in IZ_RANGE:
            depth_km = (iz + 0.5) * rla.VZ
            rows = []
            for ix in range(ix_lo, ix_hi):
                for iy in range(iy_lo, iy_hi):
                    flat_v = ix * (rla.NY * rla.NZ) + iy * rla.NZ + iz
                    n_mid = cnt_mid.get(flat_v, 0)
                    n_strict = cnt_strict.get(flat_v, 0)
                    total = n_mid + n_strict
                    if total < MIN_TOTAL:
                        continue
                    rows.append(dict(ix=ix, iy=iy, n_mid=n_mid, n_strict=n_strict,
                                     total=total, mid_frac=n_mid / total))
            if not rows:
                print(f'  iz={iz} (depth={depth_km:.2f}km): no voxels with >={MIN_TOTAL} rays')
                continue
            df = pd.DataFrame(rows)
            big = df[df['total'] >= LARGE_N]
            n_big = len(big)
            if n_big == 0:
                print(f'  iz={iz} (depth={depth_km:.2f}km): {len(df)} voxels >={MIN_TOTAL} rays, '
                      f'but none reach >={LARGE_N} for summary stats')
                continue
            med = big['mid_frac'].median()
            q25, q75 = big['mid_frac'].quantile([.25, .75])
            mn, mx = big['mid_frac'].min(), big['mid_frac'].max()
            n_strict_dom = int((big['mid_frac'] < 0.15).sum())
            n_mid_dom = int((big['mid_frac'] > 0.65).sum())
            print(f'  iz={iz} (depth={depth_km:.2f}km): n_well_sampled(>={LARGE_N})={n_big:3d}  '
                  f'mid_frac median={med:.2f} IQR=[{q25:.2f},{q75:.2f}] range=[{mn:.2f},{mx:.2f}]  '
                  f'strict_dominated(<0.15)={n_strict_dom}  mid_dominated(>0.65)={n_mid_dom}')
            all_rows.append(dict(station=sta, iz=iz, depth_km=depth_km, n_well_sampled=n_big,
                                 median_mid_frac=med, q25=q25, q75=q75, min_mid_frac=mn, max_mid_frac=mx,
                                 n_strict_dominated=n_strict_dom, n_mid_dominated=n_mid_dom))
        print()

    summary_df = pd.DataFrame(all_rows)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', lambda v: f'{v:.2f}')
    print('=== Cross-station summary ===')
    print(summary_df.to_string(index=False))

    print('\n=== Per-station mid_frac range width (max-min) across depths, pooled ===')
    for sta in rla.STATIONS:
        sub = summary_df[summary_df['station'] == sta]
        if sub.empty:
            print(f'  {sta}: no well-sampled voxels found')
            continue
        width = (sub['max_mid_frac'] - sub['min_mid_frac']).mean()
        med_of_med = sub['median_mid_frac'].mean()
        print(f'  {sta}: mean IQR-window median_mid_frac={med_of_med:.2f}  '
              f'mean(max-min) range width={width:.2f}  '
              f'total strict_dominated outliers={sub["n_strict_dominated"].sum()}  '
              f'total mid_dominated outliers={sub["n_mid_dominated"].sum()}')

    print('\nDone.')


if __name__ == '__main__':
    main()
