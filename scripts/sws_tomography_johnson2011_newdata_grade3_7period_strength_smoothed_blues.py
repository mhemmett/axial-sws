#!/usr/bin/env python3
"""
sws_tomography_johnson2011_newdata_grade3_7period_strength_smoothed_blues.py

Re-styled version of the anisotropy-strength (s_b [s/km]) panel from
sws_tomography_johnson2011_newdata_grade3_7period.py -- SAME underlying
inversion (grade-3 filter, newdata, 7 periods, reuses the cached rays/inversion
machinery, no re-tracing), but the s_b field is rasterized onto the fixed
0.25 km fine grid (the tomography's own DELTA grid, which coincides in
spacing/phase with the traveltime-anisotropy scripts' VXY/VZ=0.25km grid),
Gaussian-smoothed with a 0.5-cell half-width, and drawn with a Blues
contourf + true-max colorbar -- i.e. the same visual convention as
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_
smooth_truemax_polished_1row.py, per explicit user request ("so it looks more
like our previous traveltime anisotropy plots"), rather than the original
script's raw quad-tree-block PatchCollection (viridis, unsmoothed, hatched
checkerboard-failure overlay).

Does NOT touch sws_tomography_johnson2011_newdata_grade3_7period_strength.pdf
(the original block-map figure) -- this is a new, separate output.

Produces:
    sws_tomography_johnson2011_newdata_grade3_7period_strength_smoothed_blues.pdf
(1 page, 1 row, 7 panels)

Run with:
    python3 sws_tomography_johnson2011_newdata_grade3_7period_strength_smoothed_blues.py
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

import sws_tomography_johnson2011 as J
import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
import sws_tomography_johnson2011_newdata_grade3_7period as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(
    HERE, 'sws_tomography_johnson2011_newdata_grade3_7period_strength_smoothed_blues.pdf')

GAUSSIAN_SIGMA = 0.5   # half-width, in fine-grid cells (0.25 km/cell -> 0.125 km)
COUNT_MIN_DISPLAY = 1  # a fine cell is shown if covered by >=1 used quad-tree block

FS_N = 7.5
FS_PERIOD = 8.5


DISPLAY_COUNT_MIN = 20   # display-only gate, same as the block-map/phi-quiver figures
SIGNIFICANCE_LEVEL = S.SIGNIFICANCE_LEVEL   # same active-set posterior-variance gate


def rasterize_blocks(leaves, values, significant):
    """Broadcast each used quad-tree leaf's scalar value onto every fine cell
    (J.DELTA=0.25km grid) inside its footprint. Cells not covered by any used
    leaf, or whose leaf has fewer than DISPLAY_COUNT_MIN crossing rays, or whose
    leaf fails the resolution/significance gate, are left NaN -- both gates are
    applied BEFORE rasterizing (not after smoothing), so an excluded leaf's
    value never bleeds into a neighbor's Gaussian-smoothed field."""
    field = np.full((J.NXF, J.NYF), np.nan)
    for lf, val, sig in zip(leaves, values, significant):
        if lf['count'] < DISPLAY_COUNT_MIN or not sig:
            continue
        field[lf['ix0']:lf['ix1'], lf['iy0']:lf['iy1']] = val
    return field


def crop_to_display_window(field):
    """J's fine grid (FINE_X0..FINE_X1, FINE_Y0..FINE_Y1) is a superset of the
    display window used by this session's other 7-period figures (T.X_START..
    T.X_END, T.Y_START..T.Y_END); the two grids share the same 0.25km spacing
    and phase (FINE_X0=3.0, T.X_START=4.0 -- both exact multiples of 0.25km
    from a common origin), so cropping is a simple index slice, no
    interpolation needed."""
    ix0 = int(round((T.X_START - J.FINE_X0) / J.DELTA))
    ix1 = ix0 + T.NX
    iy0 = int(round((T.Y_START - J.FINE_Y0) / J.DELTA))
    iy1 = iy0 + T.NY
    return field[ix0:ix1, iy0:iy1]


def make_strength_figure_smoothed(results, period_labels):
    n_per = len(results)

    fields = []
    masks = []
    for res in results:
        if res is None:
            fields.append(None)
            masks.append(None)
            continue
        leaves = res['system']['used_leaves']
        m = res['m']
        significant = S._significant_mask(res)
        raw = rasterize_blocks(leaves, m, significant)
        raw = crop_to_display_window(raw)
        msk = np.isfinite(raw)
        if msk.sum() >= COUNT_MIN_DISPLAY:
            sm = gaussian_filter(np.nan_to_num(raw), sigma=GAUSSIAN_SIGMA)
        else:
            sm = None
        fields.append(sm)
        masks.append(msk)

    all_vals = np.concatenate([f[msk][f[msk] > 0] for f, msk in zip(fields, masks)
                               if f is not None and msk.any()])
    vmax = float(np.nanmax(all_vals)) if all_vals.size else 0.05
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.05
    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., vmax, 15)

    panel_w = 2.5
    n_cols = n_per + 1
    width_ratios = [1] * n_per + [0.05]
    header_room_in, footer_room_in = 0.85, 0.32
    fig_h_in = 3.6 + footer_room_in + header_room_in
    fig = plt.figure(figsize=(n_per * panel_w + 1.0, fig_h_in))
    gs_top = 1.0 - header_room_in / fig_h_in
    gs_bottom = footer_room_in / fig_h_in
    gs = GridSpec(1, n_cols, width_ratios=width_ratios, wspace=0.04, top=gs_top, bottom=gs_bottom)
    period_label_offset_frac = 0.32 / fig_h_in

    ref_ax_pos = None
    last_h = None
    for ci, (res, sm, msk) in enumerate(zip(results, fields, masks)):
        ax = fig.add_subplot(gs[0, ci])
        T._bathy(ax)
        if sm is not None and msk.any():
            mp = np.ma.masked_where(~msk, sm)
            if not mp.mask.all():
                ax.contourf(T.Xg, T.Yg, mp, levels=levels, cmap=cmap,
                            vmin=0, vmax=vmax, extend='neither')
                last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
                last_h.set_array([])
        for sta, row in T._sta_df.iterrows():
            ax.plot(row['x'], row['y'], '^', ms=6, mfc='#FFD700', mec='k', mew=0.8, zorder=13)
        T._faults(ax, 1.5)
        ax.set_xlim(T.X_START, T.X_END)
        ax.set_ylim(T.Y_START, T.Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        n_rays = res['diagnostics']['n_rays'] if res is not None else 0
        ax.set_title(f'N={n_rays:,}', fontsize=FS_N, fontweight='bold', pad=14)
        ax_pos = ax.get_position()
        fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + period_label_offset_frac,
                 period_labels[ci].replace('\n–', ' – ').replace('\n', ' '),
                 fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
        ref_ax_pos = ax_pos

    if last_h:
        cell = gs[:, n_cols - 1].get_position(fig)
        cax = fig.add_axes([cell.x0, ref_ax_pos.y0, cell.width, ref_ax_pos.height])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, vmax, 5),
                          format=FormatStrFormatter('%.3f'))
        cb.set_label(r'$s_b$ anisotropy strength [s/km]', fontsize=7.5)
        cb.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=7,
              framealpha=0.9, bbox_to_anchor=(0.45, 0.05))
    fig.suptitle(
        'Johnson et al. (2011) 2-D SWS tomography: anisotropy strength $s_b$ '
        f'(newdata, grade-3 filter, Gaussian sigma={GAUSSIAN_SIGMA} cells)',
        fontsize=10, fontweight='bold', y=0.99)
    return fig


def main():
    T._setup_environment(need_tracer=False)

    print('Loading JOINT six-station grade-3 ray cache (reusing existing cache, no re-tracing)...')
    joint_summ, joint_cells, station_index, sta_xy = S.build_joint_cache()
    print(f'Joint cache: {len(joint_summ):,} rays total across {len(S.STATIONS)} stations')

    # Cached 't' columns round-trip through CSV as strings; build_7_periods does a
    # direct datetime comparison, so parse explicitly here (restrict_period() inside
    # run_chain_grade3_period already does this itself, per-call).
    t_parsed = pd.to_datetime(joint_summ['t'], utc=True, format='mixed')
    periods = T.build_7_periods(pd.DataFrame({'t': t_parsed}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]

    print('\n=== Running Johnson et al. (2011) chain independently per period ===')
    results = []
    for lbl, t0, t1 in periods:
        print(f'\n-- {lbl.replace(chr(10), " ")} --')
        try:
            res = S.run_chain_grade3_period(joint_summ, joint_cells, t0, t1)
        except ValueError as exc:
            print(f'  [SKIP] {exc}')
            res = None
        results.append(res)

    print('\nBuilding smoothed Blues strength figure...')
    fig = make_strength_figure_smoothed(results, period_labels)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
