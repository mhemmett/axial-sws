#!/usr/bin/env python3
"""
spatial_midpoint_aniso_phi_windowcheck_grade3_7period_1row_countmin10.py

New pair of figures, per explicit user request: how fractional shear-wave anisotropy and fast
direction vary SPATIALLY across the caldera over the 2015 eruption cycle, using the ray's
LATERAL MIDPOINT between source (event) and receiver (station) as the spatial location assigned
to each measurement -- not station location, not event location. NOTE: this is a DIFFERENT
approach from the sws_tomography_johnson2011_*.py scripts (per explicit user note) -- not a
replacement for those, a separate method.

Data: this session's windowcheck (mfast max_dt=0.2s + split_windowcheck.py lag-shift-wraparound
fix) results for all 6 stations, grade-3 filtered (SNR>=2.0, quality>=0.75, dt_err<=0.05s,
dt<=T_dom/2, phi_err<=20 deg) -- same GRADE/STATION_ORDER/load_station_raw/apply_grade and the
same 7 eruption-relative time periods (single-line labels, period boundaries computed from the
raw success/dt>0 combined set, same as that script) as rose_7period_regions_windowcheck_
grade3.py, reused directly. Per explicit user request, all 6 stations are POOLED onto one shared
map per period (not one map per station) -- a single station's rays only cover a narrow
back-azimuth corridor, so pooling is needed for real spatial (caldera-wide) coverage.

Method (per event, all 6 stations pooled):
  1. Ray-trace event -> station via the Baillard FMM tracer (BaillardRayTracer, same model/grid
     as traveltime_anisotropy_7period_6stations_newdata_snr_grades.py -- that module's
     environment setup (_setup_environment) is reused directly for the Vs model, station
     coordinates, bathymetry image, and a tracer pre-computed for all 6 stations, so this is NOT
     a re-implementation of that machinery).
  2. T_travel = sum(seg_len_km / Vs_voxel) over the ray's voxel crossings (ray_to_voxels), same
     physics as that module's compute_ttime_anisotropy.
  3. Fractional shear-wave anisotropy 'a' solved from a^2 + 2x*a - 4 = 0 with x = 2*T_travel/dt
     (identical definition/derivation to figure_ec3_multipanel_temporal_mfast_windowcheck_
     grade3.py's add_frac_aniso -- events outside 0<=a<=1 are dropped).
  4. The ray's LATERAL MIDPOINT is taken as the (x,y) of the resampled ray path at index
     n_pts//2 (BaillardRayTracer.trace() resamples uniformly by arc length, so this is the
     path's arc-length midpoint) -- z is ignored (a lateral-only, all-depths-collapsed map, per
     explicit user request), binned into the same 0.25x0.25 km voxel grid (X:4-12, Y:0-12 km) as
     the traveltime_anisotropy_*.py tomography scripts.
  5. Per period, per voxel: the MEAN fractional anisotropy (%) and the circular mean fast
     direction (phi_az, via the doubled-angle trick -- rose_7period_regions_windowcheck_
     grade3.py's _circular_mean_and_se_deg convention, reused directly) of all ray midpoints
     landing in that voxel. Voxels with fewer than COUNT_MIN=10 ray midpoints are NOT plotted
     (masked out) -- the sole coverage gate; no smoothing, no diff-from-baseline (per explicit
     user request: "what gets shown is the mean").

Produces two NEW PDFs (each 1 page, 1 row x 7 time-period columns), does not touch any existing
output:
    spatial_midpoint_fractional_anisotropy_windowcheck_grade3_7period_1row_countmin10.pdf
        (mean fractional anisotropy per voxel, Blues colormap)
    spatial_midpoint_phi_quiver_windowcheck_grade3_7period_1row_countmin10.pdf
        (double-headed quiver TICK marks -- no arrowheads, since phi is axial (mod-180, no
        polarity) -- of the circular-mean fast direction per voxel, colored on a cyclic 'hsv'
        colormap normalized to [0,180], same styling convention as phi_quiver_7period_onerow_
        station25pct_countmin10.py)

Both figures share one ray-tracing pass (traced once, aggregated twice) -- ray-tracing ~90k
events across 6 stations takes several minutes.

Run with:
    python3 spatial_midpoint_aniso_phi_windowcheck_grade3_7period_1row_countmin10.py
"""

import os
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sws_forward_model import ll2xy
import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
from rose_7period_regions_windowcheck_grade3 import (
    STATION_ORDER, GRADE, load_station_raw, apply_grade, _build_time_periods_single_line,
    _circular_mean_and_se_deg,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF_ANISO = os.path.join(
    HERE, 'spatial_midpoint_fractional_anisotropy_windowcheck_grade3_7period_1row_countmin10.pdf')
OUT_PDF_QUIVER = os.path.join(
    HERE, 'spatial_midpoint_phi_quiver_windowcheck_grade3_7period_1row_countmin10.pdf')

COUNT_MIN = 10
N_RAY = 200
Z0_FAULT_REF = T.Z_MAX / 2.0   # single reference depth for the depth-corrected fault trace

AZ_CMAP = plt.colormaps['hsv']
AZ_NORM = Normalize(vmin=0.0, vmax=180.0)


def load_grade3_with_xyz(sta):
    raw = load_station_raw(sta)
    df = apply_grade(raw, GRADE)
    df = df.dropna(subset=['latitude', 'longitude', 'depth']).copy()
    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    df['station'] = sta
    return df[['station', 't', 'phi_az', 'dt', 'x', 'y', 'z']]


def assign_period_idx(t_series, periods):
    idx = np.full(len(t_series), -1, dtype=np.int64)
    t_vals = t_series.values
    for pi, (_lbl, t0, t1) in enumerate(periods):
        m = np.ones(len(t_series), dtype=bool)
        if t0 is not None:
            m &= (t_vals >= np.datetime64(t0))
        if t1 is not None:
            m &= (t_vals < np.datetime64(t1))
        idx[m & (idx < 0)] = pi
    return idx


def trace_and_collect(df):
    """Ray-trace every event -> station, keep the ray's lateral arc-length-midpoint voxel
    plus its fractional anisotropy and fast direction. Skips events with z outside [0, Z_MAX],
    a failed trace, non-positive T_travel, or fractional anisotropy outside [0, 1]."""
    n = len(df)
    print(f'Tracing {n:,} grade-3 rays across all 6 stations (n_pts={N_RAY})...')
    t0 = time.time()

    period_idx_out = []
    ix_mid_out = []
    iy_mid_out = []
    frac_pct_out = []
    phi_out = []

    n_skipped_depth = 0
    n_trace_fail = 0
    n_bad_frac = 0

    for i, row in enumerate(df.itertuples()):
        ex, ey, ez = float(row.x), float(row.y), float(row.z)
        if ez < 0 or ez > T.Z_MAX:
            n_skipped_depth += 1
            continue
        try:
            ray = T.tracer.trace(row.station, ex, ey, ez, n_pts=N_RAY)
        except RuntimeError:
            n_trace_fail += 1
            continue

        mid_x, mid_y = ray[N_RAY // 2, 0], ray[N_RAY // 2, 1]
        ix_mid = int(np.floor((mid_x - T.X_START) / T.VXY))
        iy_mid = int(np.floor((mid_y - T.Y_START) / T.VXY))
        if not (0 <= ix_mid < T.NX and 0 <= iy_mid < T.NY):
            continue

        vcols, seg_len_km = T.tracer.ray_to_voxels(ray, T.xn, T.yn, T.zn)
        if len(vcols) == 0:
            continue
        T_travel = float(np.sum(seg_len_km / np.maximum(T.vs_flat[vcols], 1e-9)))
        if T_travel <= 0:
            continue

        x_ratio = 2.0 * T_travel / row.dt
        a = -x_ratio + np.sqrt(x_ratio ** 2 + 4.0)
        if not (0.0 <= a <= 1.0):
            n_bad_frac += 1
            continue

        period_idx_out.append(row.period_idx)
        ix_mid_out.append(ix_mid)
        iy_mid_out.append(iy_mid)
        frac_pct_out.append(a * 100.0)
        phi_out.append(row.phi_az)

        if (i + 1) % 5000 == 0:
            print(f'  {i+1:,}/{n:,}  {time.time()-t0:.0f}s', end='\r', flush=True)

    print(f'\nTraced {len(period_idx_out):,}/{n:,} rays kept '
          f'(skipped: depth-out-of-range={n_skipped_depth:,}, trace-fail={n_trace_fail:,}, '
          f'frac-aniso-out-of-[0,1]={n_bad_frac:,}) in {time.time()-t0:.0f}s')

    return pd.DataFrame({
        'period_idx': period_idx_out, 'ix': ix_mid_out, 'iy': iy_mid_out,
        'frac_pct': frac_pct_out, 'phi_az': phi_out,
    })


def aggregate_voxels(rays, n_periods):
    """Per period, per voxel: mean fractional anisotropy (%), circular mean phi_az (deg,
    doubled-angle trick), and ray-midpoint count -- gated at COUNT_MIN."""
    rays = rays.copy()
    rays['sin2phi'] = np.sin(2.0 * np.radians(rays['phi_az'].values % 180.0))
    rays['cos2phi'] = np.cos(2.0 * np.radians(rays['phi_az'].values % 180.0))

    results = []
    for pi in range(n_periods):
        sub = rays[rays['period_idx'] == pi]
        frac2d = np.full((T.NX, T.NY), np.nan)
        phi2d = np.full((T.NX, T.NY), np.nan)
        cnt2d = np.zeros((T.NX, T.NY), dtype=np.int64)
        n_total = len(sub)
        if n_total == 0:
            results.append(dict(frac=frac2d, phi=phi2d, cnt=cnt2d, n=n_total))
            continue

        g = sub.groupby(['ix', 'iy'])
        agg = g.agg(frac_mean=('frac_pct', 'mean'), sin_mean=('sin2phi', 'mean'),
                    cos_mean=('cos2phi', 'mean'), n=('frac_pct', 'size')).reset_index()
        gate = agg['n'].values >= COUNT_MIN
        ix_g = agg['ix'].values[gate]
        iy_g = agg['iy'].values[gate]
        frac2d[ix_g, iy_g] = agg['frac_mean'].values[gate]
        phi2d[ix_g, iy_g] = (np.degrees(np.arctan2(agg['sin_mean'].values[gate],
                                                    agg['cos_mean'].values[gate])) / 2.0) % 180.0
        cnt2d[agg['ix'].values, agg['iy'].values] = agg['n'].values
        results.append(dict(frac=frac2d, phi=phi2d, cnt=cnt2d, n=n_total))
    return results


def _single_line_label(label):
    return label.replace('\n', ' ')


def make_aniso_page(results, period_labels, title):
    n_per = len(results)
    all_vals = np.concatenate([r['frac'][np.isfinite(r['frac'])].ravel() for r in results])
    vmax = float(np.nanpercentile(all_vals, 99)) if all_vals.size else 30.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 30.0
    cmap = plt.colormaps['Blues']
    levels = np.linspace(0.0, vmax, 15)

    panel_w = 2.2
    fig = plt.figure(figsize=(n_per * panel_w + 0.9, 3.6))
    gs = GridSpec(1, n_per + 1, width_ratios=[1] * n_per + [0.05], wspace=0.06, top=0.82, bottom=0.14)
    last_h = None
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci])
        T._bathy(ax)
        mp = np.ma.masked_invalid(res['frac'])
        if not mp.mask.all():
            ax.contourf(T.Xg, T.Yg, mp, levels=levels, cmap=cmap, vmin=0, vmax=vmax, extend='neither')
            last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
            last_h.set_array([])
        T._sta(ax)
        T._faults(ax, Z0_FAULT_REF)
        ax.set_xlim(T.X_START, T.X_END)
        ax.set_ylim(T.Y_START, T.Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f'{_single_line_label(period_labels[ci])}\nN={res["n"]:,}',
                     fontsize=7.5, fontweight='bold', pad=3)
    if last_h:
        cax = fig.add_subplot(gs[:, n_per])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, vmax, 5),
                          format=FormatStrFormatter('%.1f'))
        cb.set_label('Mean fractional anisotropy per voxel (%)', fontsize=8)
        cb.ax.tick_params(labelsize=6)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    return fig


def make_quiver_page(results, period_labels, title):
    n_per = len(results)
    panel_w = 2.2
    fig = plt.figure(figsize=(n_per * panel_w + 0.9, 3.6))
    gs = GridSpec(1, n_per + 1, width_ratios=[1] * n_per + [0.05], wspace=0.06, top=0.82, bottom=0.14)

    tick_kw = dict(scale=18, width=0.006, headlength=0, headaxislength=0, headwidth=0,
                  pivot='middle', cmap=AZ_CMAP, norm=AZ_NORM, alpha=0.95, zorder=10)

    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci])
        T._bathy(ax)
        gated = np.isfinite(res['phi'])
        if gated.any():
            ix_g, iy_g = np.nonzero(gated)
            xc = T.xn[ix_g] + T.VXY / 2.0
            yc = T.yn[iy_g] + T.VXY / 2.0
            phi_g = res['phi'][ix_g, iy_g]
            u = np.sin(np.radians(phi_g))
            v = np.cos(np.radians(phi_g))
            ax.quiver(xc, yc, u, v, phi_g, **tick_kw)
            ax.quiver(xc, yc, -u, -v, phi_g, **tick_kw)
        T._sta(ax)
        T._faults(ax, Z0_FAULT_REF)
        ax.set_xlim(T.X_START, T.X_END)
        ax.set_ylim(T.Y_START, T.Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f'{_single_line_label(period_labels[ci])}\nN={res["n"]:,}',
                     fontsize=7.5, fontweight='bold', pad=3)

    cax = fig.add_subplot(gs[:, n_per])
    sm = ScalarMappable(cmap=AZ_CMAP, norm=AZ_NORM)
    sm.set_array([])
    cb = fig.colorbar(cax=cax, mappable=sm, ticks=np.linspace(0, 180, 7))
    cb.set_label('Fast direction $\\phi$ (deg, axial 0-180)', fontsize=8)
    cb.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    return fig


def main():
    print('Setting up environment (Vs model, station coords, bathymetry, FMM tracer '
          'for all 6 stations)...')
    T._setup_environment(need_tracer=True)

    print(f'\nLoading windowcheck grade-3 events with valid lat/lon/depth ({GRADE["label"]})...')
    raw = {}
    grade3 = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
        grade3[sta] = load_grade3_with_xyz(sta)
        print(f'  {sta}: {len(grade3[sta]):,} grade-3 events with valid coordinates')

    all_raw_t = pd.concat([raw[sta][['t']] for sta in STATION_ORDER], ignore_index=True)
    periods = _build_time_periods_single_line(all_raw_t)
    period_labels = [lbl for lbl, _t0, _t1 in periods]

    df = pd.concat([grade3[sta] for sta in STATION_ORDER], ignore_index=True)
    df['period_idx'] = assign_period_idx(df['t'], periods)
    df = df[df['period_idx'] >= 0].reset_index(drop=True)
    n_total = len(df)
    print(f'\nTotal pooled grade-3 events (all 6 stations, valid period): {n_total:,}')

    rays = trace_and_collect(df)

    print(f'\nAggregating per-voxel mean fractional anisotropy + circular-mean phi '
          f'(COUNT_MIN={COUNT_MIN})...')
    results = aggregate_voxels(rays, len(periods))
    for pi, res in enumerate(results):
        n_voxels = int(np.isfinite(res['frac']).sum())
        print(f'  {period_labels[pi]!r}: n={res["n"]:,} rays, {n_voxels:,} voxels pass gate')

    aniso_title = ('AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (windowcheck) — mean fractional shear-'
                   'wave anisotropy per voxel, ray-midpoint assignment (7-period)\n'
                   f'{GRADE["label"]} (n={n_total:,}, COUNT_MIN={COUNT_MIN})')
    fig_a = make_aniso_page(results, period_labels, aniso_title)
    with PdfPages(OUT_PDF_ANISO) as pdf:
        pdf.savefig(fig_a, dpi=300, bbox_inches='tight')
    plt.close(fig_a)
    print(f'\nSaved {OUT_PDF_ANISO}')

    quiver_title = ('AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (windowcheck) — fast direction '
                     '(circular mean) per voxel, ray-midpoint assignment (7-period)\n'
                     f'{GRADE["label"]} (n={n_total:,}, COUNT_MIN={COUNT_MIN})')
    fig_q = make_quiver_page(results, period_labels, quiver_title)
    with PdfPages(OUT_PDF_QUIVER) as pdf:
        pdf.savefig(fig_q, dpi=300, bbox_inches='tight')
    plt.close(fig_q)
    print(f'Saved {OUT_PDF_QUIVER}')


if __name__ == '__main__':
    main()
