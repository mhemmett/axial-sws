#!/usr/bin/env python3
"""
lqt_pykonal_tomography_time_period_difference.py

Time-period-DIFFERENCE variant of the fullray and travel-time-weighted
anisotropy tomographies. Same per-voxel per-period grids as:
  - lqt_pykonal_tomography_raylength_anisotropy.py --assign fullray
  - lqt_pykonal_tomography_traveltime_anisotropy.py
just re-rendered so the first (pre-eruption) time-period column is left as
an absolute value (as in the sibling scripts), while every later column
plots that voxel's DIFFERENCE from the pre-eruption column instead of its
own absolute value:

    column 0 (pre-eruption):  a_0                (Blues, sequential)
    column i>0:                a_i - a_0          (RdBu_r, diverging about 0)

This makes the eruption/inflation-driven CHANGE in anisotropy visually
explicit, rather than requiring the reader to mentally subtract two absolute
panels. Reuses both sibling scripts' caches (raylength_anis_ray_summary.csv /
raylength_anis_ray_voxel_segments.csv) read-only -- no re-tracing.

Only the 7-period scheme is produced (not annual) -- "pre-eruption" is a
single clean first column there; the annual scheme's periods diverge across
6 filters x 2 variants would otherwise require 4 PDFs instead of the 2
requested.

Produces two PDFs (distinct names, does not overwrite any existing output):
    lqt_pykonal_tomography_fullray_time_period_difference.pdf      (6 pages)
    lqt_pykonal_tomography_traveltime_time_period_difference.pdf   (6 pages)

Run with:
    python3 lqt_pykonal_tomography_time_period_difference.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import matplotlib.lines as mlines

import lqt_pykonal_tomography_raylength_anisotropy as rla
import lqt_pykonal_tomography_traveltime_anisotropy as tta
import lqt_pykonal_tomography_source_voxel_anisotropy as svm

OUT_DIR = rla.OUT_DIR


# ── Shared diff-page figure builder (works for either sibling module, via
# the `mod` argument -- both define identical-convention DEPTH_ROWS, Xg, Yg,
# X_START/X_END/Y_START/Y_END, COUNT_MIN, and _bathy/_sta/_kid/_faults) ──────

def make_diff_page(mod, results, title,
                   cbar_label_abs='Mean % shear anisotropy per voxel (pre-eruption)',
                   cbar_label_diff='Δ mean % shear anisotropy from pre-eruption (pct. pts.)'):
    """results: list of dict(label, n, a) in period order, column 0 assumed
    pre-eruption. Column 0 is drawn as an absolute value (Blues); every other
    column is drawn as (a_i - a_0) on a diverging scale (RdBu_r) centred at 0."""
    depth_rows = mod.DEPTH_ROWS
    n_per = len(results)
    n_dep = len(depth_rows)

    abs_grid = results[0]['a']
    abs_finite = abs_grid[abs_grid > 0]
    a_vmax = float(np.nanpercentile(abs_finite, 99)) if abs_finite.size else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.

    diff_grids = [results[i]['a'] - abs_grid for i in range(1, n_per)]
    diff_vals = np.concatenate([d[np.isfinite(d)].ravel() for d in diff_grids]) if diff_grids else np.array([])
    d_vmax = float(np.nanpercentile(np.abs(diff_vals), 99)) if diff_vals.size else 10.
    if not np.isfinite(d_vmax) or d_vmax <= 0:
        d_vmax = 10.

    cmap_abs = plt.colormaps['Blues']
    cmap_diff = plt.colormaps['RdBu_r']
    levels_abs = np.linspace(0., a_vmax, 15)
    levels_diff = np.linspace(-d_vmax, d_vmax, 21)

    panel_w = 1.8 if n_per > 7 else 2.2
    fig = plt.figure(figsize=(n_per * panel_w + 1.1, n_dep * 3.0 + 0.5))
    gs = GridSpec(n_dep, n_per + 2, width_ratios=[1] * n_per + [0.04, 0.04],
                 hspace=0.04, wspace=0.08)

    last_abs_h = None
    last_diff_h = None
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(depth_rows):
        abs_slice = np.nanmean(abs_grid[:, :, iz0:iz1], axis=2)
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            mod._bathy(ax)
            if ci == 0:
                sl = abs_slice
                cmap, levels, vmin, vmax = cmap_abs, levels_abs, 0., a_vmax
            else:
                sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2) - abs_slice
                cmap, levels, vmin, vmax = cmap_diff, levels_diff, -d_vmax, d_vmax
            msk = np.isfinite(sl)
            if msk.sum() >= mod.COUNT_MIN:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                mp = np.ma.masked_where(~msk, sm)
                if not mp.mask.all():
                    ax.contourf(mod.Xg, mod.Yg, mp, levels=levels, cmap=cmap,
                                vmin=vmin, vmax=vmax,
                                extend='neither' if ci == 0 else 'both')
                    if ci == 0:
                        last_abs_h = ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
                        last_abs_h.set_array([])
                    else:
                        last_diff_h = ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
                        last_diff_h.set_array([])
            mod._sta(ax)
            mod._kid(ax)
            mod._faults(ax, z0)
            ax.set_xlim(mod.X_START, mod.X_END)
            ax.set_ylim(mod.Y_START, mod.Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=4)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ri == 0:
                suffix = '' if ci == 0 else ' (Δ from pre-eruption)'
                ax.set_title(f'{res["label"]}{suffix}\nN={res["n"]:,}', fontsize=7,
                            fontweight='bold', pad=1)
            if ci == 0:
                ax.set_ylabel(f'{zlbl}', fontsize=7)

    if last_abs_h:
        cax1 = fig.add_subplot(gs[:, n_per])
        cb1 = plt.colorbar(last_abs_h, cax=cax1, ticks=np.linspace(0, a_vmax, 5),
                           format=FormatStrFormatter('%.1f'))
        cb1.set_label(cbar_label_abs, fontsize=7.5)
        cb1.ax.tick_params(labelsize=6)
    if last_diff_h:
        cax2 = fig.add_subplot(gs[:, n_per + 1])
        cb2 = plt.colorbar(last_diff_h, cax=cax2, ticks=np.linspace(-d_vmax, d_vmax, 5),
                           format=FormatStrFormatter('%.1f'))
        cb2.set_label(cbar_label_diff, fontsize=7.5)
        cb2.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='OBS station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
              fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.06)
    return fig


def write_diff_pdf(pages, out_path, cbar_label_abs, cbar_label_diff):
    print(f'\nWriting {os.path.basename(out_path)}...')
    with PdfPages(out_path) as pdf:
        for mod, results, title in pages:
            fig = make_diff_page(mod, results, title,
                                 cbar_label_abs=cbar_label_abs,
                                 cbar_label_diff=cbar_label_diff)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {out_path}')


# ── Fullray variant ──────────────────────────────────────────────────────────

def build_fullray_pages():
    rla._setup_environment()

    print('\n[fullray] Loading baseline events...')
    dfs = {sta: rla.load_station(sta) for sta in rla.STATIONS}
    all_df = pd.concat(dfs.values(), ignore_index=True)
    for sta in rla.STATIONS:
        print(f'  {sta}: {len(dfs[sta]):,}')
    print(f'  TOTAL baseline: {len(all_df):,}')

    summary, vox = rla.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values
    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    a_ray_pct = summary['A_ray_pct'].values.astype(np.float64)

    page_filters = rla.build_page_filters(summary)
    periods = rla.build_7_periods(all_df)
    ray_period = rla.assign_periods(event_ns, periods)

    formula = 'A = dt·Vs·100/r (full ray value assigned to every crossed voxel)'
    pages = []
    print('\n=== Aggregating fullray, 7period ===')
    for filt_label, ray_mask in page_filters:
        results = rla.aggregate_page(vox, vray, ray_mask, ray_period, periods,
                                     assign='fullray', a_ray_pct=a_ray_pct)
        n_total = int(ray_mask.sum())
        print(f'  {filt_label:60s}: {n_total:,} rays')
        title = (f'LQT + PyKonal-FMM tomography — fullray % shear anisotropy, '
                 f'time-period difference from pre-eruption\n{formula}  '
                 f'(n={n_total:,}, filter: {filt_label})')
        pages.append((rla, results, title))
    return pages


# ── Travel-time-weighted variant ─────────────────────────────────────────────

def build_traveltime_pages():
    tta._setup_environment(need_tracer=False)

    print('\n[traveltime] Loading baseline events...')
    dfs = {sta: tta.load_station(sta) for sta in tta.STATIONS}
    all_df = pd.concat(dfs.values(), ignore_index=True)
    for sta in tta.STATIONS:
        print(f'  {sta}: {len(dfs[sta]):,}')
    print(f'  TOTAL baseline: {len(all_df):,}')

    summary, vox = tta.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values
    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8

    tt_seg, a_ray_tt = tta.compute_ttime_anisotropy(summary, vox, vray)
    page_filters = tta.build_page_filters(summary)
    periods = tta.build_7_periods(all_df)
    ray_period = tta.assign_periods(event_ns, periods)

    formula = ('A = δt·100/T_travel, travel-time-weighted mean over crossing rays')
    pages = []
    print('\n=== Aggregating traveltime, 7period ===')
    for filt_label, ray_mask in page_filters:
        results = tta.aggregate_page(vox, vray, ray_mask, ray_period, periods,
                                     tt_seg, a_ray_tt)
        n_total = int(ray_mask.sum())
        print(f'  {filt_label:60s}: {n_total:,} rays')
        title = (f'LQT + PyKonal-FMM tomography — travel-time-weighted % shear '
                 f'anisotropy, time-period difference from pre-eruption\n{formula}  '
                 f'(n={n_total:,}, filter: {filt_label})')
        pages.append((tta, results, title))
    return pages


# ── Source-voxel variant ─────────────────────────────────────────────────────

def build_source_voxel_pages():
    all_df, summary, a_ray_pct, in_grid, page_filters, event_ns = svm.prepare_source_voxel_data()
    periods = rla.build_7_periods(all_df)
    ray_period = rla.assign_periods(event_ns, periods)

    formula = 'A = dt·Vs·100/r (full ray value, assigned only to the source voxel)'
    pages = []
    for count_min in svm.COUNT_MIN_VARIANTS:
        print(f'\n=== Aggregating source-voxel, 7period, count_min={count_min} ===')
        for filt_label, ray_mask in page_filters:
            results = svm.aggregate_source_voxel(
                all_df, a_ray_pct, ray_mask, ray_period, periods, in_grid, count_min)
            n_total = int(ray_mask.sum())
            print(f'  {filt_label:60s}: {n_total:,} events')
            title = (f'LQT + PyKonal-FMM tomography — source-voxel % shear anisotropy, '
                     f'time-period difference from pre-eruption\n{formula}  '
                     f'(n={n_total:,}, filter: {filt_label}, min {count_min} events/voxel)')
            pages.append((rla, results, title))
    return pages


# ── Q_w mid-band variant (-0.5 < Q_w < 0.5): dt_err/phi_err/dt-cutoff
# thresholds unchanged, but quality restricted to this open band instead of
# the sibling scripts' six standard Q_w cuts. Distinct from the existing
# "gray zone" (-0.7 < Q_w < 0.7) and "indeterminate" (-0.2 < Q_w < 0.2) bands
# used elsewhere in the repo -- a different, narrower-than-gray-zone /
# wider-than-indeterminate band requested specifically for this comparison. ──

QW_BAND_LO, QW_BAND_HI = -0.5, 0.5


def _qw_band_mask(summary, lo=QW_BAND_LO, hi=QW_BAND_HI):
    ok = summary['trace_ok'].values.astype(bool)
    dt = summary['dt'].values
    phe = summary['phi_error'].values
    dte = summary['dt_error'].values
    q = summary['quality'].values
    mask = (ok & (dte < rla.DT_ERR_MAX) & (phe < rla.PHI_ERR_MAX) &
           (dt <= rla.DT_CUTOFF) & (q > lo) & (q < hi))
    label = (f'dt_err<{rla.DT_ERR_MAX}s & phi_err<{rla.PHI_ERR_MAX:.0f}° & '
            f'dt≤{rla.DT_CUTOFF:.2f}s & {lo}<Q_w<{hi}')
    return mask, label


def build_fullray_qwband_pages():
    rla._setup_environment()
    print('\n[fullray, Q_w band] Loading baseline events...')
    dfs = {sta: rla.load_station(sta) for sta in rla.STATIONS}
    all_df = pd.concat(dfs.values(), ignore_index=True)
    summary, vox = rla.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values
    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    a_ray_pct = summary['A_ray_pct'].values.astype(np.float64)

    mask, label = _qw_band_mask(summary)
    periods = rla.build_7_periods(all_df)
    ray_period = rla.assign_periods(event_ns, periods)

    results = rla.aggregate_page(vox, vray, mask, ray_period, periods,
                                 assign='fullray', a_ray_pct=a_ray_pct)
    n_total = int(mask.sum())
    print(f'  {label}: {n_total:,} rays')
    formula = 'A = dt·Vs·100/r (full ray value assigned to every crossed voxel)'
    title = (f'LQT + PyKonal-FMM tomography — fullray % shear anisotropy, '
             f'time-period difference from pre-eruption\n{formula}  '
             f'(n={n_total:,}, filter: {label})')
    return [(rla, results, title)]


def build_traveltime_qwband_pages():
    tta._setup_environment(need_tracer=False)
    print('\n[traveltime, Q_w band] Loading baseline events...')
    dfs = {sta: tta.load_station(sta) for sta in tta.STATIONS}
    all_df = pd.concat(dfs.values(), ignore_index=True)
    summary, vox = tta.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values
    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8

    tt_seg, a_ray_tt = tta.compute_ttime_anisotropy(summary, vox, vray)
    mask, label = _qw_band_mask(summary)
    periods = tta.build_7_periods(all_df)
    ray_period = tta.assign_periods(event_ns, periods)

    results = tta.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
    n_total = int(mask.sum())
    print(f'  {label}: {n_total:,} rays')
    formula = 'A = δt·100/T_travel, travel-time-weighted mean over crossing rays'
    title = (f'LQT + PyKonal-FMM tomography — travel-time-weighted % shear '
             f'anisotropy, time-period difference from pre-eruption\n{formula}  '
             f'(n={n_total:,}, filter: {label})')
    return [(tta, results, title)]


def build_source_voxel_qwband_pages():
    all_df, summary, a_ray_pct, in_grid, _page_filters, event_ns = svm.prepare_source_voxel_data()
    mask, label = _qw_band_mask(summary)
    periods = rla.build_7_periods(all_df)
    ray_period = rla.assign_periods(event_ns, periods)

    formula = 'A = dt·Vs·100/r (full ray value, assigned only to the source voxel)'
    pages = []
    for count_min in svm.COUNT_MIN_VARIANTS:
        results = svm.aggregate_source_voxel(
            all_df, a_ray_pct, mask, ray_period, periods, in_grid, count_min)
        n_total = int(mask.sum())
        print(f'  {label} (count_min={count_min}): {n_total:,} events')
        title = (f'LQT + PyKonal-FMM tomography — source-voxel % shear anisotropy, '
                 f'time-period difference from pre-eruption\n{formula}  '
                 f'(n={n_total:,}, filter: {label}, min {count_min} events/voxel)')
        pages.append((rla, results, title))
    return pages


def main():
    fullray_pages = build_fullray_pages()
    write_diff_pdf(
        fullray_pages,
        os.path.join(OUT_DIR, 'lqt_pykonal_tomography_fullray_time_period_difference.pdf'),
        cbar_label_abs='Mean % shear anisotropy per voxel (pre-eruption, full ray value)',
        cbar_label_diff='Δ mean % shear anisotropy from pre-eruption (pct. pts., full ray value)')

    traveltime_pages = build_traveltime_pages()
    write_diff_pdf(
        traveltime_pages,
        os.path.join(OUT_DIR, 'lqt_pykonal_tomography_traveltime_time_period_difference.pdf'),
        cbar_label_abs='Travel-time-weighted mean % shear anisotropy per voxel (pre-eruption)',
        cbar_label_diff='Δ travel-time-weighted mean % shear anisotropy from pre-eruption (pct. pts.)')

    source_voxel_pages = build_source_voxel_pages()
    write_diff_pdf(
        source_voxel_pages,
        os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_time_period_difference.pdf'),
        cbar_label_abs='Mean % shear anisotropy in source voxel (pre-eruption)',
        cbar_label_diff='Δ mean % shear anisotropy in source voxel from pre-eruption (pct. pts.)')

    fullray_qwband_pages = build_fullray_qwband_pages()
    write_diff_pdf(
        fullray_qwband_pages,
        os.path.join(OUT_DIR, 'lqt_pykonal_tomography_fullray_time_period_difference_qwband_pm05.pdf'),
        cbar_label_abs='Mean % shear anisotropy per voxel (pre-eruption, full ray value)',
        cbar_label_diff='Δ mean % shear anisotropy from pre-eruption (pct. pts., full ray value)')

    traveltime_qwband_pages = build_traveltime_qwband_pages()
    write_diff_pdf(
        traveltime_qwband_pages,
        os.path.join(OUT_DIR, 'lqt_pykonal_tomography_traveltime_time_period_difference_qwband_pm05.pdf'),
        cbar_label_abs='Travel-time-weighted mean % shear anisotropy per voxel (pre-eruption)',
        cbar_label_diff='Δ travel-time-weighted mean % shear anisotropy from pre-eruption (pct. pts.)')

    source_voxel_qwband_pages = build_source_voxel_qwband_pages()
    write_diff_pdf(
        source_voxel_qwband_pages,
        os.path.join(OUT_DIR, 'lqt_pykonal_tomography_source_voxel_time_period_difference_qwband_pm05.pdf'),
        cbar_label_abs='Mean % shear anisotropy in source voxel (pre-eruption)',
        cbar_label_diff='Δ mean % shear anisotropy in source voxel from pre-eruption (pct. pts.)')

    print('\nDone.')


if __name__ == '__main__':
    main()
