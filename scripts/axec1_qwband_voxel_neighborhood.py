#!/usr/bin/env python3
"""
axec1_qwband_voxel_neighborhood.py

Drill-down on axec1_qwband_ray_voxel_overlap.py's top-ranked voxels: three
adjacent, heavily-sampled voxels near (ix,iy)=(20-22,20-22) at iz=2
(depth 0.62 km) show roughly 50/50 coverage between AXEC1's Q_w mid-band
(-0.5<Q_w<0.5) and strict (Q_w>=0.7) rays. Question: is that a localized
patch (bounded by nearby voxels that are strict-dominated instead), or does
the whole near-station shallow neighborhood look like this regardless of
position (in which case it's just "shallow rays near the station", not a
spatially bounded structure)?

Prints a full ix/iy grid (for a window around AXEC1's own voxel) at iz in
{1,2,3} (0.25 km above/below the depth of interest, for depth context), with
n_mid, n_strict, n_other ray-crossing counts and the mid:strict ratio for
every voxel with at least MIN_TOTAL combined mid+strict rays -- so both the
50/50 voxels and any nearby high-strict/low-mid "contrast" voxels are
visible together. Marks AXEC1's own (ix,iy) voxel column.

Reuses the same cache/mask logic as axec1_qwband_ray_voxel_overlap.py
(ray_voxel_segments cache, _qw_band_mask()) -- see that script for the
shared definitions.

Run with:
    python3 axec1_qwband_voxel_neighborhood.py
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import lqt_pykonal_tomography_raylength_anisotropy as rla
from lqt_pykonal_tomography_time_period_difference import _qw_band_mask

FOCUS_STATION = 'AXEC1'
MIN_TOTAL = 10       # min mid+strict rays to print a voxel
IX_PAD, IY_PAD = 8, 8   # window half-width (in voxels) around AXEC1's own voxel
IZ_RANGE = [1, 2, 3]    # depth slices to inspect (0.25 km each; iz=2 is 0.62 km)


def voxel_ray_counts(vray, flat, ray_bool):
    keep = ray_bool[vray]
    if not keep.any():
        return {}
    f = flat[keep]
    uniq, cnt = np.unique(f, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))


def main():
    rla._setup_environment()
    sta_x, sta_y = rla.sta_xy[FOCUS_STATION]
    sta_ix = int(np.floor((sta_x - rla.X_START) / rla.VXY))
    sta_iy = int(np.floor((sta_y - rla.Y_START) / rla.VXY))
    print(f'{FOCUS_STATION} location: x={sta_x:.3f}km y={sta_y:.3f}km  -> voxel (ix,iy)=({sta_ix},{sta_iy})\n')

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

    foc = (station == FOCUS_STATION)
    mid_foc = mid_mask & foc
    strict_foc = strict_mask & foc
    elig_other = elig & ~foc

    vray = vox['ray_id'].values
    ix_all = vox['ix'].values.astype(np.int64)
    iy_all = vox['iy'].values.astype(np.int64)
    iz_all = vox['iz'].values.astype(np.int64)
    flat = ix_all * (rla.NY * rla.NZ) + iy_all * rla.NZ + iz_all

    cnt_mid = voxel_ray_counts(vray, flat, mid_foc)
    cnt_strict = voxel_ray_counts(vray, flat, strict_foc)
    cnt_other = voxel_ray_counts(vray, flat, elig_other)

    ix_lo, ix_hi = sta_ix - IX_PAD, sta_ix + IX_PAD
    iy_lo, iy_hi = sta_iy - IY_PAD, sta_iy + IY_PAD

    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.float_format', lambda v: f'{v:.2f}')

    for iz in IZ_RANGE:
        depth_km = (iz + 0.5) * rla.VZ
        rows = []
        for ix in range(max(0, ix_lo), min(rla.NX, ix_hi + 1)):
            for iy in range(max(0, iy_lo), min(rla.NY, iy_hi + 1)):
                flat_v = ix * (rla.NY * rla.NZ) + iy * rla.NZ + iz
                n_mid = cnt_mid.get(flat_v, 0)
                n_strict = cnt_strict.get(flat_v, 0)
                n_other = cnt_other.get(flat_v, 0)
                total = n_mid + n_strict
                if total < MIN_TOTAL:
                    continue
                rows.append(dict(
                    ix=ix, iy=iy,
                    is_axec1_voxel=(ix == sta_ix and iy == sta_iy),
                    n_mid=n_mid, n_strict=n_strict, n_other=n_other,
                    total_mid_strict=total,
                    mid_frac=n_mid / total,
                ))
        if not rows:
            print(f'--- iz={iz} (depth={depth_km:.2f}km): no voxels with >={MIN_TOTAL} mid+strict rays in window ---\n')
            continue
        df = pd.DataFrame(rows).sort_values('total_mid_strict', ascending=False)
        print(f'--- iz={iz} (depth={depth_km:.2f}km): {len(df)} voxels with >={MIN_TOTAL} mid+strict rays, '
              f'window ix=[{ix_lo},{ix_hi}] iy=[{iy_lo},{iy_hi}] ---')
        print(df.to_string(index=False))
        near_50_50 = df[(df['mid_frac'] > 0.35) & (df['mid_frac'] < 0.65) & (df['total_mid_strict'] >= 50)]
        strict_dominated = df[(df['mid_frac'] < 0.15) & (df['total_mid_strict'] >= 50)]
        print(f'\n  Near-50/50 voxels (mid_frac in [0.35,0.65], total>=50): {len(near_50_50)}')
        if len(near_50_50):
            print('  ' + near_50_50[['ix', 'iy', 'n_mid', 'n_strict', 'mid_frac']].to_string(index=False).replace('\n', '\n  '))
        print(f'  Strict-dominated voxels (mid_frac<0.15, total>=50): {len(strict_dominated)}')
        if len(strict_dominated):
            print('  ' + strict_dominated[['ix', 'iy', 'n_mid', 'n_strict', 'mid_frac']].to_string(index=False).replace('\n', '\n  '))
        print()

    print('Done.')


if __name__ == '__main__':
    main()
