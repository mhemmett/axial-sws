#!/usr/bin/env python3
"""
phi_7period_6stations_newdata_snr2_q75_countmin50_midpoint_quadtree_1row.py

Fast-direction (phi) analog of
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_polished_1row.py
(the REGULAR, all-7-periods, ray-MIDPOINT-assignment percent-anisotropy figure), but:
  - the target quantity is phi (fast-polarization azimuth, axial mod 180), not percent
    anisotropy;
  - spatial binning uses the QUAD-TREE regridding from
    spatial_quadtree_fast_direction_tier5_6stations.py (BASE_CELL_KM=1.0km cells,
    recursively subdivided down to a MIN_CELL_KM=0.25km floor wherever a cell holds
    >=SPLIT_THRESHOLD=40 ray midpoints) INSTEAD OF the fixed 0.25km voxel grid used by the
    anisotropy scripts -- i.e. bin size adapts to ray density rather than being uniform;
  - the coverage gate is COUNT_MIN=50 (leaf cells with fewer than 50 ray midpoints are
    excluded/not plotted), not the 10 used by the anisotropy midpoint scripts -- per
    explicit user instruction, since circular-mean phi is noisier at low N than a scalar
    percent-anisotropy mean;
  - single depth row: same as the source spatial_quadtree script, each ray's 3D midpoint
    (traced through the Baillard Vs model, arc-length midpoint of the path -- reusing
    compute_ray_midpoint_voxels() from the percent-anisotropy midpoint script and the SAME
    cached ray-tracing data, no re-tracing) is COLLAPSED BY DEPTH (z dropped) before 2D
    quad-tree binning, so there is inherently only one row/panel per period (not a
    depth-row grid).

Per period, per retained leaf cell: the CIRCULAR MEAN (doubled-angle vector average) of
that cell's contributing rays' phi is computed (via
spatial_quadtree_fast_direction_tier5_6stations.build_all_bins(), which already returns this),
then drawn as a double-headed (undirected) quiver tick mark centered on the cell, whose
AZIMUTH encodes phi and whose COLOR (cyclic 'hsv' colormap normalized to [0, 180]) also
encodes phi -- same convention as phi_quiver_7period_onerow_*.py. The tick's length is scaled
to the cell's OWN quad-tree size (scale_units='xy', so a tick drawn in a 1.0km base cell is
proportionally longer than one drawn in a 0.25km floor cell), and a thin unfilled rectangle
outline is also drawn per cell so the quad-tree partition itself remains visible (matching
spatial_quadtree_fast_direction_tier5_6stations.py's own presentation, but colored by phi
value here rather than by cell size).

Grade-3 filter [T.GRADES[3], "Page 1 @ quality>=0.75": SNR>=2.0, quality>=0.75,
dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg], SAME as the percent-anisotropy midpoint scripts
(not the tier-5 filter the original spatial_quadtree_fast_direction_tier5_6stations.py used --
this script is deliberately consistent with today's other newdata figures' filter choice).
Each period's quad-tree is built independently (a period with fewer events, e.g.
syn-eruption, naturally ends up with fewer/larger surviving cells).

Produces:
    phi_7period_6stations_newdata_snr2_q75_countmin50_midpoint_quadtree_1row.pdf
(1 page, 1 row, 7 panels) -- a NEW file, does not touch any existing output.

Run with:
    python3 phi_7period_6stations_newdata_snr2_q75_countmin50_midpoint_quadtree_1row.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
import spatial_quadtree_fast_direction_tier5_6stations as Q
from traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_diff_polished_1row import (
    compute_ray_midpoint_voxels, FS_N, FS_PERIOD, _single_line_label,
)
from phi_quiver_7period_onerow_station25pct_countmin10 import (
    build_event_phi_lookup,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'phi_7period_6stations_newdata_snr2_q75_countmin50_midpoint_quadtree_1row.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75
assert GRADE['key'] == 'page1_q0.75', GRADE

# Quad-tree parameters: reuse spatial_quadtree_fast_direction_tier5_6stations.py's defaults
# for BASE_CELL_KM/MIN_CELL_KM/SPLIT_THRESHOLD, override only MIN_COUNT per explicit user
# instruction (exclude any cell with < 50 rays).
Q.MIN_COUNT = 50

AZ_CMAP = plt.colormaps['hsv']
AZ_NORM = Normalize(vmin=0.0, vmax=180.0)

FS_N = 7.5
FS_PERIOD = 8.5


def make_regular_page(results, period_labels, title):
    n_per = len(results)
    panel_w = 2.5
    n_cols = n_per + 1
    width_ratios = [1] * n_per + [0.05]
    col_cbar = n_cols - 1

    header_room_in = 0.85
    footer_room_in = 0.32
    fig_h_in = 3.6 + footer_room_in + header_room_in
    fig = plt.figure(figsize=(n_per * panel_w + 1.0, fig_h_in))
    gs_top = 1.0 - header_room_in / fig_h_in
    gs_bottom = footer_room_in / fig_h_in
    gs = GridSpec(1, n_cols, width_ratios=width_ratios, wspace=0.04, top=gs_top, bottom=gs_bottom)
    period_label_offset_frac = 0.32 / fig_h_in

    ref_ax_pos = None
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci])
        T._bathy(ax)
        for b in res['bins']:
            ax.add_patch(Rectangle((b['x0'], b['y0']), b['x1'] - b['x0'], b['y1'] - b['y0'],
                                    fill=False, edgecolor='0.5', lw=0.3, alpha=0.6, zorder=5))
        if res['bins']:
            cx = np.array([b['cx'] for b in res['bins']])
            cy = np.array([b['cy'] for b in res['bins']])
            sz = np.array([b['size'] for b in res['bins']])
            phi_vals = np.array([b['phi'] for b in res['bins']])
            r = np.radians(phi_vals)
            # Tick length in DATA units (km), scaled to each cell's OWN size (scale_units='xy',
            # scale=1 makes u/v exact data-unit lengths) -- so a quiver tick fits its own
            # quad-tree cell regardless of whether that cell is at the 1.0km base size or has
            # been subdivided down to the 0.25km floor.
            length = 0.8 * sz
            u, v = length * np.sin(r), length * np.cos(r)
            tick_kw = dict(scale=1, scale_units='xy', width=0.005, headlength=0,
                           headaxislength=0, headwidth=0, pivot='middle', cmap=AZ_CMAP,
                           norm=AZ_NORM, alpha=0.95, zorder=7)
            ax.quiver(cx, cy, u, v, phi_vals, **tick_kw)
            ax.quiver(cx, cy, -u, -v, phi_vals, **tick_kw)
        T._sta(ax)
        T._faults(ax, Q.FAULT_Z0)
        ax.set_xlim(T.X_START, T.X_END)
        ax.set_ylim(T.Y_START, T.Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f'N={res["n"]:,}', fontsize=FS_N, fontweight='bold', pad=14)
        ax_pos = ax.get_position()
        fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + period_label_offset_frac,
                 _single_line_label(period_labels[ci]),
                 fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
        ref_ax_pos = ax.get_position()

    cell = gs[:, col_cbar].get_position(fig)
    cax = fig.add_axes([cell.x0, ref_ax_pos.y0, cell.width, ref_ax_pos.height])
    sm = ScalarMappable(cmap=AZ_CMAP, norm=AZ_NORM)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, ticks=np.linspace(0, 180, 7))
    cb.set_label('Fast direction $\\phi$ (deg, axial 0-180)', fontsize=7.5)
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

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    all_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]
    ray_period = T.assign_periods(event_ns, periods)

    mask = ((summary['quality'].values >= GRADE['qw_min']) &
            (summary['snr_horizontal'].values >= GRADE['snr_min']) &
            (summary['phi_error'].values <= GRADE['phi_err_max']))

    print('Computing ray midpoint voxels (arc-length midpoint of each traced ray path)...')
    ray_id_mid, ix_mid, iy_mid, iz_mid = compute_ray_midpoint_voxels(vox)
    print(f'  {len(ray_id_mid):,} rays assigned a midpoint voxel (of {N:,} traced rays)')

    # Voxel-center (x, y) of each ray's midpoint -- z is dropped (single depth row, per the
    # source spatial_quadtree script's depth-collapsed convention).
    mx_all = np.full(N, np.nan)
    my_all = np.full(N, np.nan)
    mx_all[ray_id_mid] = T.xn[ix_mid] + T.VXY / 2.0
    my_all[ray_id_mid] = T.yn[iy_mid] + T.VXY / 2.0

    print('Recovering event phi per cached ray...')
    phi_lookup = build_event_phi_lookup()
    summary['t'] = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    phi_ray = np.full(N, np.nan)
    for sta in T.STATIONS:
        m = (summary['station'].values == sta)
        phi_ray[m] = phi_lookup[sta].reindex(summary['t'].values[m]).values
    n_phi_matched = int(np.isfinite(phi_ray).sum())
    print(f'  matched event phi for {n_phi_matched:,}/{N:,} cached rays')

    print(f'\nBuilding per-period quad-trees (base={Q.BASE_CELL_KM*1000:.0f}m, '
          f'floor={Q.MIN_CELL_KM*1000:.0f}m, split>={Q.SPLIT_THRESHOLD}, '
          f'min_count={Q.MIN_COUNT})...')
    results = []
    for pi, lbl in enumerate(period_labels):
        keep = mask & (ray_period == pi) & np.isfinite(mx_all) & np.isfinite(phi_ray)
        n_rays = int(keep.sum())
        bins = Q.build_all_bins(mx_all[keep], my_all[keep], phi_ray[keep]) if n_rays else []
        print(f'  {lbl}: {n_rays:,} rays -> {len(bins):,} bins (N>={Q.MIN_COUNT})')
        results.append(dict(label=lbl, n=n_rays, bins=bins))

    title = (f'Fast direction $\\phi$ by ray-midpoint quad-tree bin (6 stations, ray-traced '
             f'through Baillard Vs model)\n'
             f'{GRADE["label"]}; base cell {Q.BASE_CELL_KM*1000:.0f} m, split at '
             f'N>={Q.SPLIT_THRESHOLD}, floor {Q.MIN_CELL_KM*1000:.0f} m, '
             f'cells excluded if N<{Q.MIN_COUNT}')
    fig = make_regular_page(results, period_labels, title)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
