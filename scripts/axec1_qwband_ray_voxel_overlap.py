#!/usr/bin/env python3
"""
axec1_qwband_ray_voxel_overlap.py

Ray-path-proximity companion to axec1_qwband_clustering_stats.py: instead of
comparing event-hypocentre-to-station distance, this checks the actual
VOXELS each ray crosses (the same ray_voxel_segments cache the fullray/
apportioned tomography scripts use). Question: do AXEC1's Q_w mid-band
(-0.5 < Q_w < 0.5, "indeterminate quality") rays preferentially cross a
distinct set of voxels that are poorly covered by (a) AXEC1's own
confidently-split (Q_w >= 0.7) rays and (b) the pooled eligible rays from
the other 5 stations? If so, that would point to a specific structure at
depth (e.g. a magma conduit) that only AXEC1's ray geometry samples and that
looks isotropic because it is genuinely being measured differently there --
rather than the isotropic null being a QC artifact.

Reuses lqt_pykonal_tomography_raylength_anisotropy.load_cache() (the
(summary, vox) cache pair -- vox is one row per ray-voxel crossing, already
deduplicated so each ray contributes at most one row per voxel) and
lqt_pykonal_tomography_time_period_difference._qw_band_mask() for the
mid-band definition (single source of truth, shared with the other qwband
scripts). Does NOT need _setup_environment()/bathymetry/tracer -- the voxel
grid indices (NX, NY, NZ, VXY, VZ, X_START, Y_START) are module-level
constants in the raylength_anisotropy module.

For each voxel crossed by >=MIN_RAYS AXEC1 mid-band rays, reports the same
voxel's crossing count from AXEC1's strict rays and from all other stations'
eligible rays, plus:
  - Jaccard overlap between the three covered-voxel sets
  - the top "mid-dominated" voxels (high mid count, low strict/other count)
    with their (x, y, depth) so a spatial/depth pattern can be read off
  - depth-histogram comparison: mid-dominated voxels vs. voxels covered by
    all three sets

Run with:
    python3 axec1_qwband_ray_voxel_overlap.py
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import lqt_pykonal_tomography_raylength_anisotropy as rla
from lqt_pykonal_tomography_time_period_difference import _qw_band_mask

FOCUS_STATION = 'AXEC1'
MIN_RAYS = 3        # min distinct rays crossing a voxel to call it "covered"
TOP_N = 25           # how many mid-dominated voxels to print


def voxel_ray_counts(vray, flat, ray_bool):
    """Return a dict {flat_voxel_index: n_distinct_rays} for rays where
    ray_bool is True. vox rows are already deduplicated per (ray, voxel), so
    this is exactly the number of distinct rays crossing each voxel."""
    keep = ray_bool[vray]
    if not keep.any():
        return {}
    f = flat[keep]
    uniq, cnt = np.unique(f, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))


def unflatten(flat_idx):
    ix = flat_idx // (rla.NY * rla.NZ)
    rem = flat_idx % (rla.NY * rla.NZ)
    iy = rem // rla.NZ
    iz = rem % rla.NZ
    x = rla.X_START + (ix + 0.5) * rla.VXY
    y = rla.Y_START + (iy + 0.5) * rla.VXY
    z = (iz + 0.5) * rla.VZ
    return ix, iy, iz, x, y, z


def main():
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

    print(f'Focus station: {FOCUS_STATION}')
    print(f'Mid-band definition: {mid_label}')
    print(f'AXEC1 mid-band rays: {int(mid_foc.sum()):,}  '
          f'AXEC1 strict rays: {int(strict_foc.sum()):,}  '
          f'other-station eligible rays: {int(elig_other.sum()):,}\n')

    vray = vox['ray_id'].values
    ix_all = vox['ix'].values.astype(np.int64)
    iy_all = vox['iy'].values.astype(np.int64)
    iz_all = vox['iz'].values.astype(np.int64)
    flat = ix_all * (rla.NY * rla.NZ) + iy_all * rla.NZ + iz_all

    cnt_mid = voxel_ray_counts(vray, flat, mid_foc)
    cnt_strict = voxel_ray_counts(vray, flat, strict_foc)
    cnt_other = voxel_ray_counts(vray, flat, elig_other)

    covered_mid = {v for v, c in cnt_mid.items() if c >= MIN_RAYS}
    covered_strict = {v for v, c in cnt_strict.items() if c >= MIN_RAYS}
    covered_other = {v for v, c in cnt_other.items() if c >= MIN_RAYS}

    def jaccard(a, b):
        if not a and not b:
            return np.nan
        return len(a & b) / len(a | b)

    print(f'Covered voxels (>= {MIN_RAYS} rays): '
          f'mid={len(covered_mid)}  strict={len(covered_strict)}  other-stations={len(covered_other)}')
    print(f'Jaccard(mid, strict)          = {jaccard(covered_mid, covered_strict):.3f}')
    print(f'Jaccard(mid, other-stations)  = {jaccard(covered_mid, covered_other):.3f}')
    print(f'Jaccard(strict, other-stations) = {jaccard(covered_strict, covered_other):.3f}  (reference/baseline overlap)\n')

    mid_only = covered_mid - covered_strict - covered_other
    print(f'"Mid-only" voxels (covered by mid-band rays, NOT by strict AND NOT by other stations): '
          f'{len(mid_only)} / {len(covered_mid)} ({len(mid_only) / max(len(covered_mid), 1):.1%} of covered mid voxels)\n')

    # ── Rank voxels by mid-dominance: high mid count, low strict+other count ──
    rows = []
    for v, c_mid in cnt_mid.items():
        c_strict = cnt_strict.get(v, 0)
        c_other = cnt_other.get(v, 0)
        ix, iy, iz, x, y, z = unflatten(v)
        rows.append(dict(ix=ix, iy=iy, iz=iz, x_km=x, y_km=y, depth_km=z,
                         n_mid=c_mid, n_strict=c_strict, n_other=c_other,
                         mid_dominance=c_mid / (c_strict + c_other + 1)))
    df = pd.DataFrame(rows)
    df = df[df['n_mid'] >= MIN_RAYS].sort_values('mid_dominance', ascending=False)

    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', lambda v: f'{v:.2f}')
    print(f'Top {TOP_N} mid-dominated voxels (high n_mid, low n_strict+n_other):')
    print(df.head(TOP_N).to_string(index=False))

    if len(df) >= 10:
        depth_top = df.head(max(TOP_N, len(mid_only))).sort_values('mid_dominance', ascending=False)
        print(f'\nDepth summary of top mid-dominated voxels: '
              f'median={depth_top["depth_km"].median():.2f}km  '
              f'IQR=[{depth_top["depth_km"].quantile(.25):.2f}, {depth_top["depth_km"].quantile(.75):.2f}]km')

    all_three = covered_mid & covered_strict & covered_other
    if all_three:
        depths_all3 = np.array([unflatten(v)[5] for v in all_three])
        print(f'Depth summary of voxels covered by ALL THREE sets (mid, strict, other): '
              f'median={np.median(depths_all3):.2f}km  n={len(all_three)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
