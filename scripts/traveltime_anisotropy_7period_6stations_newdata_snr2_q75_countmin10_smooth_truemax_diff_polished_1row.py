#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row.py

Single-depth-row variant of
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished.pdf
(same data/filter/cached ray tracing, COUNT_MIN=10, Gaussian smoothing, true-max colorbars,
title, legend, column-header/N= typography, and gap+colorbar layout -- see that script's
docstring, not repeated here), except there is now only ONE row: no depth dependence is
shown at all. Every voxel across the FULL 0.0-Z_MAX km grid depth (every observation that
passes the filter/ray-trace-eligibility test, not the 0.0-2.25 km subset used in the polished
3-row figure) is averaged together for each time period, instead of being split into 3
separate 0.75 km depth rows. The per-row y-axis label is also omitted (single row, label not
needed).

Produces:
    traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row.pdf
(1 page, 1 row) -- a NEW file, does not touch any existing output.

Run with:
    python3 traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(
    HERE,
    'traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75
assert GRADE['key'] == 'page1_q0.75', GRADE

NEW_COUNT_MIN = 10

# Single row: ALL depths that pass the QC/ray-trace-eligibility filter -- i.e. the FULL
# 0.0-Z_MAX km grid depth, not the 0.0-2.25 km subset used in the original 3-row figure.
DEPTH_ROWS_ALL = [
    ([(iz, 1.0) for iz in range(T.NZ)], f'All depths (0.0–{T.Z_MAX:.1f} km)', T.Z_MAX / 2.0),
]

FS_N = 7.5
FS_PERIOD = 8.5   # larger than FS_N


def _weighted_depth_slice(a3d, iz_weights):
    shape2d = a3d.shape[:2]
    num = np.zeros(shape2d)
    den = np.zeros(shape2d)
    for iz, w in iz_weights:
        layer = a3d[:, :, iz]
        valid = np.isfinite(layer)
        num[valid] += layer[valid] * w
        den[valid] += w
    out = np.full(shape2d, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _smoothed_masked_slices(results, depth_rows, count_min_display):
    fields = {}
    for ri, (iz_weights, zlbl, z0) in enumerate(depth_rows):
        for pi, res in enumerate(results):
            sl = _weighted_depth_slice(res['a'], iz_weights)
            msk = np.isfinite(sl)
            if msk.sum() >= count_min_display:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                fields[(pi, ri)] = (sm, msk)
            else:
                fields[(pi, ri)] = (None, msk)
    return fields


def _single_line_label(label):
    return label.replace('\n–', ' – ').replace('\n', ' ')


def make_diff_page(results, period_labels, title, depth_rows, fields):
    n_per = len(results)
    n_dep = len(depth_rows)

    abs_vals = []
    diff_vals = []
    diffs = {}
    pre_masked = {}
    for ri in range(n_dep):
        sm0, msk0 = fields[(0, ri)]
        if sm0 is not None:
            mp0 = np.ma.masked_where(~msk0, sm0)
            pre_masked[ri] = mp0
            abs_vals.append(mp0.compressed()[mp0.compressed() > 0])
        else:
            pre_masked[ri] = None
        for pi in range(1, n_per):
            sm_p, msk_p = fields[(pi, ri)]
            if sm0 is not None and sm_p is not None:
                joint = msk0 & msk_p
                if joint.any():
                    d = np.ma.masked_where(~joint, sm_p - sm0)
                    diffs[(pi, ri)] = d
                    diff_vals.append(d.compressed())
                    continue
            diffs[(pi, ri)] = None

    a_vmax = float(np.nanmax(np.concatenate(abs_vals))) if abs_vals and any(v.size for v in abs_vals) else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.
    all_diff = np.concatenate(diff_vals) if diff_vals else np.array([])
    d_vmax = float(np.nanmax(np.abs(all_diff))) if all_diff.size else 10.
    if not np.isfinite(d_vmax) or d_vmax <= 0:
        d_vmax = 10.

    cmap_abs = plt.colormaps['Blues']
    levels_abs = np.linspace(0., a_vmax, 15)
    cmap_diff = plt.colormaps['RdBu_r']
    levels_diff = np.linspace(-d_vmax, d_vmax, 21)

    panel_w = 2.5
    # Columns: [panel0, cbar_abs, spacer, diff1..diff(n_per-1), cbar_diff]
    n_cols = n_per + 3
    width_ratios = [1, 0.05, 0.18] + [1] * (n_per - 1) + [0.05]
    col_diff0 = 3   # first diff panel's grid column
    col_cbar_abs = 1
    col_cbar_diff = n_cols - 1

    # Extra headroom (in inches) above the axes grid for the period-label/N= header stack and
    # the suptitle, and below the axes grid for the legend -- on a short (1-row) figure this
    # needs to be reserved explicitly, unlike the multi-row version where the grid is tall
    # enough that it falls out naturally. Both reduced from the original draft to tighten the
    # text-to-plot and plot-to-legend gaps per explicit user request.
    header_room_in = 0.85
    footer_room_in = 0.32
    fig_h_in = n_dep * 3.6 + footer_room_in + header_room_in
    fig = plt.figure(figsize=(n_per * panel_w + 1.4, fig_h_in))
    gs_top = 1.0 - header_room_in / fig_h_in
    gs_bottom = footer_room_in / fig_h_in
    gs = GridSpec(n_dep, n_cols, width_ratios=width_ratios, hspace=0.05, wspace=0.04,
                 top=gs_top, bottom=gs_bottom)
    # Period-label offset above the axes top, in absolute inches (converted to figure-fraction
    # below) rather than a fixed figure-fraction -- a fixed fraction shrinks in absolute terms
    # on a short (few-row) figure and can end up SMALLER than the N= title's own points-based
    # pad, reversing the intended stacking order (period label above N=). Reduced from 0.65in
    # to tighten the text-to-plot gap.
    period_label_offset_frac = 0.32 / fig_h_in

    last_h_abs = None
    last_h_diff = None
    ref_ax_pos = None   # actual (aspect-adjusted) bbox of a data panel, used to size colorbars
    for ri, (iz_weights, zlbl, z0) in enumerate(depth_rows):
        for ci, res in enumerate(results):
            col = 0 if ci == 0 else col_diff0 + (ci - 1)
            ax = fig.add_subplot(gs[ri, col])
            T._bathy(ax)
            if ci == 0:
                mp = pre_masked[ri]
                if mp is not None and not mp.mask.all():
                    ax.contourf(T.Xg, T.Yg, mp, levels=levels_abs, cmap=cmap_abs,
                                vmin=0, vmax=a_vmax, extend='neither')
                    last_h_abs = ScalarMappable(cmap=cmap_abs, norm=Normalize(0, a_vmax))
                    last_h_abs.set_array([])
            else:
                d = diffs.get((ci, ri))
                if d is not None and not d.mask.all():
                    ax.contourf(T.Xg, T.Yg, d, levels=levels_diff, cmap=cmap_diff,
                                vmin=-d_vmax, vmax=d_vmax, extend='both')
                    last_h_diff = ScalarMappable(cmap=cmap_diff, norm=Normalize(-d_vmax, d_vmax))
                    last_h_diff.set_array([])
            T._sta(ax)
            T._faults(ax, z0)
            ax.set_xlim(T.X_START, T.X_END)
            ax.set_ylim(T.Y_START, T.Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=4)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f'N={res["n"]:,}', fontsize=FS_N, fontweight='bold', pad=14)
                ax_pos = ax.get_position()
                fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + period_label_offset_frac,
                         _single_line_label(period_labels[ci]),
                         fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
                ref_ax_pos = ax.get_position()   # post-aspect bbox -- same y0/y1 for every
                                                  # column in this row (identical data aspect)
    # Colorbars: match height to the ACTUAL (aspect-adjusted) plotted panels, not the nominal
    # GridSpec cell -- set_aspect('equal','box') shrinks each map panel's effective height to
    # match its 8:12 km data aspect, but a colorbar axis has no such constraint and would
    # otherwise fill the full (taller) nominal cell, making it visibly taller than the maps.
    if last_h_abs:
        cell = gs[:, col_cbar_abs].get_position(fig)
        cax = fig.add_axes([cell.x0, ref_ax_pos.y0, cell.width, ref_ax_pos.height])
        cb = plt.colorbar(last_h_abs, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter('%.1f'))
        cb.set_label('Percent Anisotropy Pre-Eruption', fontsize=7.5)
        cb.ax.tick_params(labelsize=6)
    if last_h_diff:
        cell2 = gs[:, col_cbar_diff].get_position(fig)
        cax2 = fig.add_axes([cell2.x0, ref_ax_pos.y0, cell2.width, ref_ax_pos.height])
        cb2 = plt.colorbar(last_h_diff, cax=cax2, ticks=np.linspace(-d_vmax, d_vmax, 5),
                           format=FormatStrFormatter('%.1f'))
        cb2.set_label('Change in Percent Anisotropy from Pre-Eruption Period', fontsize=7.5)
        cb2.ax.tick_params(labelsize=6)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.05))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    return fig


def main():
    cache_exists = os.path.exists(T.RAY_SUMMARY_CSV) and os.path.exists(T.RAY_VOXEL_CSV)
    if not cache_exists:
        raise SystemExit(
            f'Ray-tracing cache not found ({T.RAY_SUMMARY_CSV} / {T.RAY_VOXEL_CSV}). '
            'Run traveltime_anisotropy_7period_6stations_newdata_snr_grades.py first.')

    T._setup_environment(need_tracer=False)

    print('Reusing existing ray-tracing cache (no re-tracing)...')
    summary, vox = T.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    all_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]

    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    mask = ((summary['quality'].values >= GRADE['qw_min']) &
            (summary['snr_horizontal'].values >= GRADE['snr_min']) &
            (summary['phi_error'].values <= GRADE['phi_err_max']))
    ray_period = T.assign_periods(event_ns, periods)

    print(f'Setting per-voxel coverage gate: COUNT_MIN -> {NEW_COUNT_MIN} '
          f'(MIN_PATH_KM unchanged at {T.MIN_PATH_KM} km)')
    T.COUNT_MIN = NEW_COUNT_MIN

    results = T.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
    n_total = int(mask.sum())
    print(f'  {GRADE["label"]}: {n_total:,} rays')

    print(f'Computing smoothed+masked field (single row, all depths 0-{T.Z_MAX:.1f}km combined)...')
    fields = _smoothed_masked_slices(results, DEPTH_ROWS_ALL, count_min_display=T.COUNT_MIN)

    title = 'Percent Anisotropy Across an Eruption Cycle'
    fig = make_diff_page(results, period_labels, title, DEPTH_ROWS_ALL, fields)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
