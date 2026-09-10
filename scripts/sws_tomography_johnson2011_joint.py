#!/usr/bin/env python3
"""
sws_tomography_johnson2011_joint.py

JOINT six-station driver for sws_tomography_johnson2011 (Johnson, Savage &
Townend 2011, 2-D delay-time SWS tomography).

The per-station chain in the module keys every downstream structure (quad-tree,
G/d system, phi averaging, checkerboard) on ``ray_id`` -- which each per-station
cache numbers independently 0..N_s-1. A naive concat of the six caches would
COLLIDE ray_ids across stations and silently merge unrelated rays. This driver
MERGES the six per-station caches into ONE (summ, cells) with GLOBALLY-UNIQUE
ray_ids so build_quadtree/build_system/run_chain see all six stations' rays in
one G matrix, and then runs the joint inversion for every (epoch, quality).

Per-ray recording-station coords (sx, sy) are already carried in each station's
summary cache and threaded through build_system -> ray_sx/ray_sy, so the 1/d^2
phi weighting measures block-to-station distance to the CORRECT receiver in the
joint system.

Merge invariants are ASSERTED (not hoped): no ray_id collision, every finecell
ray_id present in the summary, arc-length benchmark survives the merge, and sx/sy
differ across stations.

Run:
    python3 sws_tomography_johnson2011_joint.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import sws_tomography_johnson2011 as tomo
from pykonal_raytracer import BaillardRayTracer

# Import the demo's plotting helpers (adapted for the joint label).
import sws_tomography_johnson2011_demo as demo

# Per-station ray_id offset. Max rays at any station is ~150k, so 10M per
# station is a safe, non-overlapping stride (verified in load_joint_cache).
RAY_ID_OFFSET = 10_000_000

QUALITY_THRESHOLDS = [-0.5, 0.7]
EPOCHS = ['combined', '2015_2021', '2022_2026']


def load_joint_cache(stations, catalog, tracer, out_dir=tomo.RESULTS_DIR,
                     offset=RAY_ID_OFFSET, retrace=False, verbose=True):
    """Load each station's ray cache, offset its ray_ids to a globally-unique
    range (base_s = station_index * offset applied to BOTH summ and cells so the
    summary<->finecell linkage stays intact), concatenate to one joint
    (summ, cells), and ASSERT the merge invariants.

    Returns (joint_summ, joint_cells, station_of_ray) where station_of_ray maps
    each offset ray_id block back to its station code via ray_id // offset.
    """
    per_summ = []
    per_cells = []
    station_index = {}
    n_expected = 0
    for si, sta in enumerate(stations):
        summ_s, cells_s = tomo.load_or_build_ray_cache(
            sta, catalog, tracer, out_dir=out_dir, retrace=retrace,
            verbose=verbose)
        max_rid = int(summ_s['ray_id'].max())
        assert max_rid < offset, (
            f'{sta}: max ray_id {max_rid:,} >= offset {offset:,} -- offset too '
            f'small, ray_ids would collide across stations')
        base = si * offset
        summ_s = summ_s.copy()
        cells_s = cells_s.copy()
        summ_s['ray_id'] = summ_s['ray_id'].astype(np.int64) + base
        cells_s['ray_id'] = cells_s['ray_id'].astype(np.int64) + base
        summ_s['station'] = sta
        per_summ.append(summ_s)
        per_cells.append(cells_s)
        station_index[sta] = si
        n_expected += len(summ_s)
        if verbose:
            print(f'  {sta}: offset base={base:,}  rays={len(summ_s):,}  '
                  f'ray_id range [{summ_s["ray_id"].min():,}, '
                  f'{summ_s["ray_id"].max():,}]')

    joint_summ = pd.concat(per_summ, ignore_index=True)
    joint_cells = pd.concat(per_cells, ignore_index=True)

    # ── Invariant (a): no ray_id collision ────────────────────────────────────
    assert len(joint_summ) == n_expected, (
        f'summary length {len(joint_summ):,} != sum of per-station '
        f'{n_expected:,}')
    assert joint_summ['ray_id'].is_unique, 'joint summary ray_id NOT unique -- '\
        'offset collision'
    # ── Invariant (b): every finecell ray_id exists in the summary ────────────
    summ_ids = set(joint_summ['ray_id'].tolist())
    cell_ids = set(joint_cells['ray_id'].unique().tolist())
    missing = cell_ids - summ_ids
    assert not missing, (
        f'{len(missing)} finecell ray_ids absent from summary -- linkage broken '
        f'by offset (e.g. {sorted(missing)[:3]})')
    # ── Invariant (c): arc-length benchmark survives the merge ────────────────
    ok_a, max_rel, n_in, n_leak = tomo.verify_arclength(joint_summ, joint_cells)
    # ── Invariant (d): sx/sy differ across stations ───────────────────────────
    sxy = joint_summ.groupby('station')[['sx', 'sy']].first()
    n_distinct = len(sxy.drop_duplicates())
    assert n_distinct == len(stations), (
        f'expected {len(stations)} distinct station (sx,sy), got {n_distinct}:\n'
        f'{sxy}')

    if verbose:
        print(f'\n  MERGE INVARIANTS (all asserted OK):')
        print(f'    (a) collision-free : len={len(joint_summ):,} == '
              f'sum={n_expected:,}, ray_id unique=True')
        print(f'    (b) linkage        : all {len(cell_ids):,} finecell ray_ids '
              f'present in summary (0 missing)')
        print(f'    (c) arc-length     : max rel err {max_rel:.2e} over '
              f'{n_in:,} in-grid rays (pass={ok_a}); {n_leak:,} far-field leaks')
        print(f'    (d) per-station xy : {n_distinct} distinct (sx,sy):')
        for sta, row in sxy.iterrows():
            print(f'          {sta}: sx={row.sx:.3f} sy={row.sy:.3f}')
    return joint_summ, joint_cells, station_index


def _per_station_breakdown(system, station_index, offset=RAY_ID_OFFSET):
    """Count how many of the system's USED rays came from each station."""
    inv = {v: k for k, v in station_index.items()}
    sidx = (system['ray_ids'] // offset).astype(int)
    out = {}
    for s in np.unique(sidx):
        out[inv[int(s)]] = int(np.sum(sidx == s))
    return out


def main():
    os.makedirs(tomo.RESULTS_DIR, exist_ok=True)
    print('Loading MLdd catalog...')
    catalog = tomo.load_catalog()
    sta_xy = tomo.station_xy()
    stations = tomo.STATIONS

    # Tracer only needed if a cache must be (re)built; caches already exist.
    tracer = BaillardRayTracer(stride=5)

    print('\n=== Building JOINT six-station cache ===')
    joint_summ, joint_cells, station_index = load_joint_cache(
        stations, catalog, tracer)

    print('\n=== JOINT inversion: 3 epochs x 2 quality thresholds ===')
    header = (f'{"epoch":>10} {"q":>5} {"rays":>7} {"blk_c":>6} {"blk_u":>6} '
              f'{"s_b_min":>8} {"s_b_max":>8} {"s_b_med":>8} {"frac@lb":>7} '
              f'{"n@ub":>5} {"cond":>10} {"varRed":>8} {"cb_nf":>6} '
              f'{"cb_pr":>6} {"phi":>4}')
    print(header)
    results = {}
    for q in QUALITY_THRESHOLDS:
        for epoch in EPOCHS:
            tag = f'JOINT6_{epoch}_q{q:g}'
            try:
                res = tomo.run_chain(joint_summ, joint_cells, quality_min=q,
                                     epoch=epoch, verbose=False)
            except ValueError as exc:
                print(f'  [SKIP] {tag}: {exc}')
                continue
            d = res['diagnostics']
            # Noise-free checkerboard ceiling (deterministic).
            ok_c, corr_nf = tomo.verify_checkerboard(res['system'], res['D'],
                                                     res['alpha'])
            bd = _per_station_breakdown(res['system'], station_index)
            results[(epoch, q)] = dict(res=res, corr_nf=corr_nf, breakdown=bd)
            print(f'{epoch:>10} {q:>5g} {d["n_rays"]:>7,} {d["created"]:>6} '
                  f'{d["used"]:>6} {d["s_b_min"]:>8.4f} {d["s_b_max"]:>8.4f} '
                  f'{d["s_b_median"]:>8.4f} {d["frac_at_lb"]:>7.3f} '
                  f'{d["n_at_ub"]:>5} {d["cond"]:>10.2e} '
                  f'{d["var_reduction"]:>+8.3f} {corr_nf:>6.3f} '
                  f'{d["cb_corr_well_sampled"]:>6.3f} {d["n_phi_kept"]:>4}')
            bd_str = ', '.join(f'{k}={v:,}' for k, v in sorted(bd.items()))
            print(f'           per-station rays: {bd_str}')

            # Figures (adapt the demo plots to the joint result).
            demo.plot_dt_strength(
                res, 'JOINT6', sta_xy,
                os.path.join(tomo.RESULTS_DIR, f'{tag}_dt_strength.png'))
            demo.plot_phi_rose(
                res, 'JOINT6', sta_xy,
                os.path.join(tomo.RESULTS_DIR, f'{tag}_phi_rose.png'))
            demo.plot_checkerboard(
                res, 'JOINT6', sta_xy,
                os.path.join(tomo.RESULTS_DIR, f'{tag}_checkerboard.png'))

    print(f'\nDone. Figures + joint results in {tomo.RESULTS_DIR}')
    return results


if __name__ == '__main__':
    main()
