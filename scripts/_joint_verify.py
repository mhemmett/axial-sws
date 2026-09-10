#!/usr/bin/env python3
"""Single-vs-joint verification driver (Debugger, task item 6).

Builds a JOINT multi-station tomography system with GLOBALLY-UNIQUE ray_ids by
offsetting each station's cache ray_ids by a per-station base, then merging the
ray-summary and fine-cell tables. Feeds the merged pair to run_chain (which
filters/epochs/quad-trees/solves exactly as for a single station). Reports the
decisive trio [cond(G^T C^-1 G), noisy per-ray well-sampled checkerboard corr,
var_reduction] for the best single station and for the joint inversion.
"""
import numpy as np
import pandas as pd
import sws_tomography_johnson2011 as tomo
from pykonal_raytracer import BaillardRayTracer

BASE = 10_000_000   # per-station ray_id offset (caches are < ~200k rays each)
EPOCH = 'combined'
Q = -0.5


def load_offset(sta, catalog, tracer, k):
    summ, cells = tomo.load_or_build_ray_cache(sta, catalog, tracer)
    off = k * BASE
    summ = summ.copy()
    cells = cells.copy()
    summ['ray_id'] = summ['ray_id'].astype(np.int64) + off
    cells['ray_id'] = cells['ray_id'].astype(np.int64) + off
    summ['station'] = sta
    return summ, cells


def trio(res):
    d = res['diagnostics']
    return (d['cond'], d['cb_corr_well_sampled'], d['var_reduction'],
            d['used'], d['n_rays'], d['frac_at_lb'], d['n_at_ub'],
            d.get('sigma_m2_cond', float('nan')))


def main():
    catalog = tomo.load_catalog()
    tracer = BaillardRayTracer(stride=5)  # only needed if a cache is missing
    stations = ['AXEC1', 'AXEC3', 'AXEC2', 'AXAS1', 'AXAS2', 'AXCC1']

    print('=== SINGLE-STATION (AXEC1, combined, q>=-0.5) ===')
    s1, c1 = load_offset('AXEC1', catalog, tracer, 0)
    res1 = tomo.run_chain(s1, c1, quality_min=Q, epoch=EPOCH, verbose=True)
    t1 = trio(res1)
    print(f'  SINGLE trio: cond={t1[0]:.3e}  cb_corr_ws(noisy)={t1[1]:.3f}  '
          f'var_red={t1[2]:+.4f}  used_blocks={t1[3]}  rays={t1[4]}  '
          f'frac@lb={t1[5]:.2f}  n@ub={t1[6]}  sigma_m2_cond={t1[7]:.2e}')

    for nsta in (3, 6):
        subset = stations[:nsta]
        print(f'\n=== JOINT ({nsta} stations: {subset}, {EPOCH}, q>={Q}) ===')
        summ_list, cells_list = [], []
        for k, sta in enumerate(subset):
            s, c = load_offset(sta, catalog, tracer, k)
            summ_list.append(s)
            cells_list.append(c)
        merged_summ = pd.concat(summ_list, ignore_index=True)
        merged_cells = pd.concat(cells_list, ignore_index=True)
        # collision check: ray_id unique per summary row, and every cells ray_id
        # is present in exactly one summary station.
        n_summ = len(merged_summ)
        n_uniq = merged_summ['ray_id'].nunique()
        assert n_summ == n_uniq, f'ray_id COLLISION: {n_summ} rows, {n_uniq} unique'
        cell_ids = set(merged_cells['ray_id'].unique())
        summ_ids = set(merged_summ['ray_id'].unique())
        assert cell_ids <= summ_ids, 'cells ray_id not a subset of summary ray_id'
        print(f'  merged: {n_summ:,} summary rows, all ray_ids unique; '
              f'{len(cell_ids):,} distinct cell ray_ids (no collisions)')
        resJ = tomo.run_chain(merged_summ, merged_cells, quality_min=Q,
                              epoch=EPOCH, verbose=True)
        tJ = trio(resJ)
        print(f'  JOINT trio: cond={tJ[0]:.3e}  cb_corr_ws(noisy)={tJ[1]:.3f}  '
              f'var_red={tJ[2]:+.4f}  used_blocks={tJ[3]}  rays={tJ[4]}  '
              f'frac@lb={tJ[5]:.2f}  n@ub={tJ[6]}  sigma_m2_cond={tJ[7]:.2e}')


if __name__ == '__main__':
    main()
