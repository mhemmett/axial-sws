#!/usr/bin/env python3
"""
dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff.py

Δ-from-pre-eruption companion to
dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.pdf: same
source-voxel (earthquake-hypocenter-binned, NOT ray-traced) delay-time aggregation, same
filter/grid/depth-row/COUNT_MIN/smoothing/true-max-colorbar conventions (see that script's
docstring, not repeated here) -- but plots column 1 (Pre-eruption) as the absolute smoothed
mean-delay-time field and columns 2-7 as the per-voxel difference from pre-eruption
(diverging RdBu_r), matching the layout/typography of
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_
polished.pdf (gap + colorbar next to the leftmost column, single-line period labels larger
than N=, simplified Station/Faults legend, no Kidiwela).

Produces:
    dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff.pdf
(1 page) -- a NEW file, does not touch any existing output.

Run with:
    python3 dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff.py
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
from dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax import (
    GRADE, COUNT_MIN, DEPTH_ROWS, load_events, aggregate_source_voxel, _weighted_depth_slice,
    _single_line_label,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(
    HERE, 'dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff.pdf')

FS_N = 7.5
FS_PERIOD = 8.5


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

    a_vmax = float(np.nanmax(np.concatenate(abs_vals))) if abs_vals and any(v.size for v in abs_vals) else 0.1
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 0.1
    all_diff = np.concatenate(diff_vals) if diff_vals else np.array([])
    d_vmax = float(np.nanmax(np.abs(all_diff))) if all_diff.size else 0.05
    if not np.isfinite(d_vmax) or d_vmax <= 0:
        d_vmax = 0.05

    cmap_abs = plt.colormaps['Blues']
    levels_abs = np.linspace(0., a_vmax, 15)
    cmap_diff = plt.colormaps['RdBu_r']
    levels_diff = np.linspace(-d_vmax, d_vmax, 21)

    panel_w = 2.5
    n_cols = n_per + 3
    width_ratios = [1, 0.05, 0.18] + [1] * (n_per - 1) + [0.05]
    col_diff0 = 3
    col_cbar_abs = 1
    col_cbar_diff = n_cols - 1

    fig = plt.figure(figsize=(n_per * panel_w + 1.4, n_dep * 3.6 + 0.6))
    gs = GridSpec(n_dep, n_cols, width_ratios=width_ratios, hspace=0.05, wspace=0.04)

    last_h_abs = None
    last_h_diff = None
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
                ax.set_title(f'N={res["n"]:,}', fontsize=FS_N, fontweight='bold', pad=30)
                ax_pos = ax.get_position()
                fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + 0.045,
                         _single_line_label(period_labels[ci]),
                         fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
            if ci == 0:
                ax.set_ylabel(f'{zlbl}', fontsize=7)
    if last_h_abs:
        cax = fig.add_subplot(gs[:, col_cbar_abs])
        cb = plt.colorbar(last_h_abs, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter('%.3f'))
        cb.set_label('Delay Time Pre-Eruption (s)', fontsize=7.5)
        cb.ax.tick_params(labelsize=6)
    if last_h_diff:
        cax2 = fig.add_subplot(gs[:, col_cbar_diff])
        cb2 = plt.colorbar(last_h_diff, cax=cax2, ticks=np.linspace(-d_vmax, d_vmax, 5),
                           format=FormatStrFormatter('%.3f'))
        cb2.set_label('Change in Delay Time from Pre-Eruption Period (s)', fontsize=7.5)
        cb2.ax.tick_params(labelsize=6)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.015))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.045)
    return fig


def main():
    T._setup_environment(need_tracer=False)

    print('Loading source-location events (Page 1 @ quality>=0.75, no ray tracing needed)...')
    df = load_events()
    print(f'  {len(df):,} events pass the filter')

    all_t = df['t']
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]

    results = aggregate_source_voxel(df, periods)

    print('Computing smoothed+masked fields (0-2.25km, 3 equal 0.75km rows) for diff...')
    fields = _smoothed_masked_slices(results, DEPTH_ROWS, count_min_display=COUNT_MIN)

    title = 'Delay Time Across an Eruption Cycle'
    fig = make_diff_page(results, period_labels, title, DEPTH_ROWS, fields)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
