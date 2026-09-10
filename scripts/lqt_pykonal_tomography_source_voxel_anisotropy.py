#!/usr/bin/env python3
"""
lqt_pykonal_tomography_source_voxel_anisotropy.py

Spatial percent-anisotropy tomography, SOURCE-VOXEL variant: each ray's full
A_ray = dt * Vs_avg * 100 / r (identical formula/cache as
lqt_pykonal_tomography_raylength_anisotropy.py) is assigned ONLY to the single
voxel containing the earthquake's own hypocentre -- no path-length
apportionment, no assignment to every crossed voxel. A voxel's plotted value
is the mean of A_ray over every event whose hypocentre falls inside it.

This needs NO ray tracing at all: it reuses raylength_anis_ray_summary.csv
(A_ray_pct, dt, phi_error, dt_error, quality, trace_ok -- already computed)
and just re-derives each ray's event (x, y, z) via the same MLdd-catalog join
load_station() already does, then bins that location into the 0.25 km voxel
grid. Row order of a fresh load_station() call is positionally identical to
the cached ray_id order (both iterate STATIONS in the same order and
concatenate station dataframes without reordering), which is checked by
assertion below. This script imports lqt_pykonal_tomography_raylength_
anisotropy.py for all shared grid/plotting/period/QC-filter machinery (that
module's heavy setup lives inside _setup_environment(), not at import time,
so importing it is cheap) -- but does NOT need the ray tracer itself, only
the bathymetry/station/catalog setup, so build_cache()/tracer precompute are
never invoked.

Coverage gate: unlike the path-based variants, there is no "cumulative path
length per voxel" concept for a single source point, so only a minimum-event-
count gate applies. Per the user's request, produces the same 6 filters twice
per PDF -- once at COUNT_MIN=8 pages 1-6, then again at COUNT_MIN=20 pages
7-12 -- for both period schemes.

Produces two PDFs (distinct names, does NOT overwrite any existing output):
    lqt_pykonal_tomography_source_voxel_anisotropy_7period.pdf   (12 pages)
    lqt_pykonal_tomography_source_voxel_anisotropy_annual.pdf    (12 pages)

Run with:
    python3 lqt_pykonal_tomography_source_voxel_anisotropy.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import lqt_pykonal_tomography_raylength_anisotropy as rla

OUT_DIR = rla.OUT_DIR

COUNT_MIN_VARIANTS = [8, 20]


def aggregate_source_voxel(all_df, a_ray_pct, page_mask, ray_period, periods,
                           in_grid, count_min):
    """Same shape/contract as rla.aggregate_page(), but bins by the EVENT'S
    OWN (ix, iy, iz) -- computed once by the caller and passed in via all_df's
    'ix','iy','iz' columns -- rather than by ray-crossed voxels. Coverage gate
    is event COUNT only (no path-length analogue for a single source point)."""
    keep = page_mask & in_grid & (ray_period >= 0)
    results = []
    if not keep.any():
        for pi, (lbl, _t0, _t1) in enumerate(periods):
            n = int((page_mask & (ray_period == pi)).sum())
            results.append(dict(label=lbl, n=n, a=np.full((rla.NX, rla.NY, rla.NZ), np.nan)))
        return results

    ix = all_df['ix'].values[keep].astype(np.int64)
    iy = all_df['iy'].values[keep].astype(np.int64)
    iz = all_df['iz'].values[keep].astype(np.int64)
    flat = ix * (rla.NY * rla.NZ) + iy * rla.NZ + iz
    period = ray_period[keep]
    aval = a_ray_pct[keep]

    key = period * rla.J + flat
    uniq, inv = np.unique(key, return_inverse=True)
    sum_a = np.zeros(len(uniq))
    np.add.at(sum_a, inv, aval)
    cnt = np.bincount(inv, minlength=len(uniq))
    mean_a = sum_a / np.maximum(cnt, 1)
    gate = cnt >= count_min

    up = uniq // rla.J
    uf = uniq % rla.J
    uix = uf // (rla.NY * rla.NZ)
    ur = uf % (rla.NY * rla.NZ)
    uiy = ur // rla.NZ
    uiz = ur % rla.NZ

    for pi, (lbl, _t0, _t1) in enumerate(periods):
        a3d = np.full((rla.NX, rla.NY, rla.NZ), np.nan)
        sel = (up == pi) & gate
        a3d[uix[sel], uiy[sel], uiz[sel]] = mean_a[sel]
        n = int((page_mask & (ray_period == pi)).sum())
        results.append(dict(label=lbl, n=n, a=a3d))
    return results


def prepare_source_voxel_data():
    """Shared setup for every source-voxel script variant: loads baseline
    events (for hypocentre locations), loads the cached ray summary
    (A_ray_pct etc.), verifies positional alignment between the two, and bins
    each event hypocentre into its source voxel. Returns
    (all_df, summary, a_ray_pct, in_grid, page_filters, event_ns). Calls
    rla._setup_environment() itself (bathymetry/station/catalog globals;
    cheap apart from the one-time FMM station precompute, which this whole
    family of scripts doesn't otherwise need but accepts for simplicity/reuse)."""
    rla._setup_environment()

    print('\nLoading baseline events (for event hypocentre locations)...')
    dfs = {}
    for sta in rla.STATIONS:
        df = rla.load_station(sta)
        dfs[sta] = df
        print(f'  {sta}: {len(df):,}')
    all_df = pd.concat(dfs.values(), ignore_index=True)
    print(f'  TOTAL baseline: {len(all_df):,}')

    print('Loading cached ray summary (A_ray_pct etc.)...')
    summary = pd.read_csv(rla.RAY_SUMMARY_CSV).sort_values('ray_id').reset_index(drop=True)
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    assert len(all_df) == N, (
        f'row-count mismatch: fresh baseline load={len(all_df):,} vs cached rays={N:,} -- '
        f'the underlying splitting-results CSVs changed since the cache was built; '
        f're-run lqt_pykonal_tomography_raylength_anisotropy.py first to refresh it.')
    # Positional-alignment check: both this script and the original build_cache()
    # iterate STATIONS in the same order and concatenate without reordering, so
    # row i here must be the same measurement as ray_id i in the cache.
    cached_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601').values
    assert (cached_t == all_df['t'].values).all(), (
        'positional misalignment between fresh baseline load and cached ray_summary -- '
        'refusing to guess event locations for the wrong rays.')

    # Bin each event hypocentre into its source voxel (xn/yn/zn are LEFT EDGES,
    # matching pykonal_raytracer.ray_to_voxels' registration convention).
    all_df = all_df.copy()
    all_df['ix'] = np.floor((all_df['x'].values - rla.X_START) / rla.VXY).astype(np.int64)
    all_df['iy'] = np.floor((all_df['y'].values - rla.Y_START) / rla.VXY).astype(np.int64)
    all_df['iz'] = np.floor((all_df['z'].values - 0.0) / rla.VZ).astype(np.int64)
    in_grid = ((all_df['ix'] >= 0) & (all_df['ix'] < rla.NX) &
               (all_df['iy'] >= 0) & (all_df['iy'] < rla.NY) &
               (all_df['iz'] >= 0) & (all_df['iz'] < rla.NZ)).values
    print(f'  {in_grid.sum():,}/{N:,} event hypocentres fall inside the plotting grid extent')

    a_ray_pct = summary['A_ray_pct'].values.astype(np.float64)
    page_filters = rla.build_page_filters(summary)
    event_ns = pd.DatetimeIndex(all_df['t']).asi8
    return all_df, summary, a_ray_pct, in_grid, page_filters, event_ns


def main():
    all_df, summary, a_ray_pct, in_grid, page_filters, event_ns = prepare_source_voxel_data()

    formula = 'A = dt·Vs·100/r (full ray value, assigned only to the source voxel)'

    schemes = [
        ('7period', rla.build_7_periods(all_df),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_anisotropy_7period.pdf')),
        ('annual', rla.build_annual(),
         os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_anisotropy_annual.pdf')),
    ]

    for scheme_name, periods, out_path in schemes:
        ray_period = rla.assign_periods(event_ns, periods)
        pages = []
        for count_min in COUNT_MIN_VARIANTS:
            print(f'\n=== Aggregating {scheme_name} ({len(periods)} periods, '
                  f'count_min={count_min}) ===')
            for filt_label, ray_mask in page_filters:
                results = aggregate_source_voxel(
                    all_df, a_ray_pct, ray_mask, ray_period, periods, in_grid, count_min)
                n_total = int(ray_mask.sum())
                covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
                print(f'  {filt_label:60s}: {n_total:,} events  covered-voxels={covered:,}')
                title = (f'LQT + PyKonal-FMM tomography — source-voxel % shear anisotropy '
                         f'({scheme_name})\n{formula}\n'
                         f'(n={n_total:,}, filter: {filt_label}, min {count_min} events/voxel)')
                pages.append((results, title))
        rla.write_pdf(pages, out_path,
                      cbar_label='Mean % shear anisotropy in source voxel')

    print('\nDone.')


if __name__ == '__main__':
    main()
