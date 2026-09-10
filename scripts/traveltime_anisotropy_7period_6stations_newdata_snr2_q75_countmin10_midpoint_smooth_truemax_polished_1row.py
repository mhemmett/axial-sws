#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_polished_1row.py

REGULAR (absolute, all 7 periods, no diff-from-pre-eruption) companion to
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_diff_polished_1row.py
-- same ray-midpoint assignment (each ray's scalar percent-anisotropy value credited entirely
to the single voxel containing the arc-length midpoint of its traced path, not distributed
across every voxel it crosses -- see that script's docstring for the full rationale/method),
same cached ray tracing (no re-tracing), same grade-3 filter [T.GRADES[3], "Page 1 @
quality>=0.75": SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg], same
COUNT_MIN=10 coverage gate (voxels with fewer than 10 distinct ray-midpoints are excluded, shown
as blank/unshaded), same single-row-all-depths (0.0-Z_MAX km combined) layout, same Gaussian
smoothing and polished title/legend/typography/layout conventions.

Unlike the diff script, this one shows all 7 periods as independent ABSOLUTE percent-anisotropy
panels on a single shared "true max" colorbar (the actual max value across all 7 periods'
smoothed, count-gated voxels -- not a percentile clip), rather than pre-eruption absolute +
6 diff-from-pre-eruption panels.

Produces:
    traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_polished_1row.pdf
(1 page, 1 row, 7 panels) -- a NEW file, does not touch any existing output.

Run with:
    python3 traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_polished_1row.py
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
from traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_diff_polished_1row import (
    compute_ray_midpoint_voxels, aggregate_page_midpoint, _weighted_depth_slice,
    _smoothed_masked_slices, _single_line_label, GRADE, NEW_COUNT_MIN, DEPTH_ROWS_ALL,
    FS_N, FS_PERIOD,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(
    HERE,
    'traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_polished_1row.pdf')


def make_regular_page(results, period_labels, title, depth_rows, fields):
    n_per = len(results)
    n_dep = len(depth_rows)

    masked = {}
    all_vals = []
    for ri in range(n_dep):
        for pi in range(n_per):
            sm, msk = fields[(pi, ri)]
            if sm is not None:
                mp = np.ma.masked_where(~msk, sm)
                masked[(pi, ri)] = mp
                v = mp.compressed()
                all_vals.append(v[v > 0])
            else:
                masked[(pi, ri)] = None

    a_vmax = float(np.nanmax(np.concatenate(all_vals))) if all_vals and any(v.size for v in all_vals) else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.

    cmap_abs = plt.colormaps['Blues']
    levels_abs = np.linspace(0., a_vmax, 15)

    panel_w = 2.5
    # Columns: [panel0..panel(n_per-1), cbar]
    n_cols = n_per + 1
    width_ratios = [1] * n_per + [0.05]
    col_cbar = n_cols - 1

    header_room_in = 0.85
    footer_room_in = 0.32
    fig_h_in = n_dep * 3.6 + footer_room_in + header_room_in
    fig = plt.figure(figsize=(n_per * panel_w + 1.0, fig_h_in))
    gs_top = 1.0 - header_room_in / fig_h_in
    gs_bottom = footer_room_in / fig_h_in
    gs = GridSpec(n_dep, n_cols, width_ratios=width_ratios, hspace=0.05, wspace=0.04,
                 top=gs_top, bottom=gs_bottom)
    period_label_offset_frac = 0.32 / fig_h_in

    last_h = None
    ref_ax_pos = None
    for ri, (iz_weights, zlbl, z0) in enumerate(depth_rows):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            T._bathy(ax)
            mp = masked[(ci, ri)]
            if mp is not None and not mp.mask.all():
                ax.contourf(T.Xg, T.Yg, mp, levels=levels_abs, cmap=cmap_abs,
                            vmin=0, vmax=a_vmax, extend='neither')
                last_h = ScalarMappable(cmap=cmap_abs, norm=Normalize(0, a_vmax))
                last_h.set_array([])
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
                ref_ax_pos = ax.get_position()
    if last_h:
        cell = gs[:, col_cbar].get_position(fig)
        cax = fig.add_axes([cell.x0, ref_ax_pos.y0, cell.width, ref_ax_pos.height])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter('%.1f'))
        cb.set_label('Percent Anisotropy', fontsize=7.5)
        cb.ax.tick_params(labelsize=6)
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

    _tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    mask = ((summary['quality'].values >= GRADE['qw_min']) &
            (summary['snr_horizontal'].values >= GRADE['snr_min']) &
            (summary['phi_error'].values <= GRADE['phi_err_max']))
    ray_period = T.assign_periods(event_ns, periods)

    print('Computing ray midpoint voxels (arc-length midpoint of each traced ray path)...')
    ray_id_mid, ix_mid, iy_mid, iz_mid = compute_ray_midpoint_voxels(vox)
    print(f'  {len(ray_id_mid):,} rays assigned a midpoint voxel (of {N:,} traced rays)')

    print(f'Setting per-voxel coverage gate: COUNT_MIN -> {NEW_COUNT_MIN} '
          f'(distinct-ray-midpoints per voxel; cells with fewer are excluded/left blank)')

    results = aggregate_page_midpoint(ray_id_mid, ix_mid, iy_mid, iz_mid, mask, ray_period,
                                       periods, a_ray_tt, NEW_COUNT_MIN)
    n_total = int(mask.sum())
    print(f'  {GRADE["label"]}: {n_total:,} rays')

    print(f'Computing smoothed+masked field (single row, all depths 0-{T.Z_MAX:.1f}km combined, '
          f'midpoint assignment, count_min={NEW_COUNT_MIN})...')
    fields = _smoothed_masked_slices(results, DEPTH_ROWS_ALL, count_min_display=NEW_COUNT_MIN)

    title = 'Percent Anisotropy Across an Eruption Cycle (Ray-Midpoint Assignment)'
    fig = make_regular_page(results, period_labels, title, DEPTH_ROWS_ALL, fields)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
