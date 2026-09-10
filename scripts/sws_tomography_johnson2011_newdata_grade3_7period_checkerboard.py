#!/usr/bin/env python3
"""
sws_tomography_johnson2011_newdata_grade3_7period_checkerboard.py

Checkerboard resolution test (Johnson et al. 2011, eq. 5) for the newdata
grade-3 7-period run in sws_tomography_johnson2011_newdata_grade3_7period.py --
this figure did not exist before (that script only used the checkerboard
result internally, to hatch out unresolvable blocks on the strength map).

For each period: input checkerboard model m_CB (top row) vs. recovered m_rec
(bottom row) after forward-modeling synthetic delta_t with per-ray N(0,
sigma_dt) noise and inverting with the SAME covariance/bounds as the real
data -- same convention as sws_tomography_johnson2011_demo.py's
plot_checkerboard(), RdBu_r colormap, shared vmax across both rows and all
periods. Same DISPLAY_COUNT_MIN=20 display gate as the other newdata grade-3
figures (blocks crossed by <20 rays are not shown, per explicit user
instruction) applied to BOTH rows identically, and the per-period recovery
correlation (corr, corr restricted to well-sampled/count>=20 blocks) is
reported in each column's header.

Reuses the cached rays and re-runs the same per-period inversion (fast, no
re-tracing) -- does not touch any existing Johnson-et-al. output.

Produces:
    sws_tomography_johnson2011_newdata_grade3_7period_checkerboard.pdf
(1 page, 2 rows x 7 periods)

Run with:
    python3 sws_tomography_johnson2011_newdata_grade3_7period_checkerboard.py
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
from matplotlib.patches import Rectangle

import sws_tomography_johnson2011 as J
import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
import sws_tomography_johnson2011_newdata_grade3_7period as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'sws_tomography_johnson2011_newdata_grade3_7period_checkerboard.pdf')

DISPLAY_COUNT_MIN = S.DISPLAY_COUNT_MIN   # same gate as the other newdata grade-3 figures

FS_N = 7.5
FS_PERIOD = 8.5


def make_checkerboard_figure(results, period_labels):
    n_per = len(results)

    all_vals = []
    for res in results:
        if res is None:
            continue
        cb = res['checkerboard']
        leaves = res['system']['used_leaves']
        keep = np.array([lf['count'] >= DISPLAY_COUNT_MIN for lf in leaves])
        if keep.any():
            all_vals.append(cb['m_cb'][keep])
            all_vals.append(cb['m_rec'][keep])
    all_vals = np.concatenate(all_vals) if all_vals else np.array([0.02])
    vmax = float(max(np.percentile(all_vals, 99), 1e-6))
    cmap = plt.colormaps['RdBu_r']

    panel_w = 2.5
    n_cols = n_per + 1
    width_ratios = [1] * n_per + [0.05]
    # Extra header room vs. the 1-row figures: the column title here is TWO lines
    # ("N=...", "corr(...)=...") plus the period label above it via fig.text, i.e.
    # 3 text lines total -- 0.85in (tuned for a 1-line title) crowds/overlaps here.
    header_room_in, footer_room_in = 1.15, 0.32
    fig_h_in = 2 * 3.2 + footer_room_in + header_room_in
    fig = plt.figure(figsize=(n_per * panel_w + 1.0, fig_h_in))
    gs_top = 1.0 - header_room_in / fig_h_in
    gs_bottom = footer_room_in / fig_h_in
    gs = GridSpec(2, n_cols, width_ratios=width_ratios, hspace=0.08, wspace=0.04,
                 top=gs_top, bottom=gs_bottom)
    period_label_offset_frac = 0.55 / fig_h_in

    row_labels = ['Input $m_{CB}$', 'Recovered $m_{rec}$']
    ref_ax_pos = None
    last_h = None
    for ci, res in enumerate(results):
        for ri, key in enumerate(['m_cb', 'm_rec']):
            ax = fig.add_subplot(gs[ri, ci])
            T._bathy(ax)
            if res is not None:
                cb = res['checkerboard']
                leaves = res['system']['used_leaves']
                vals = cb[key]
                for lf, val in zip(leaves, vals):
                    if lf['count'] < DISPLAY_COUNT_MIN:
                        continue
                    xc, yc = lf['center']
                    s = lf['side_km']
                    color = cmap(Normalize(0, vmax)(val))
                    ax.add_patch(Rectangle((xc - s / 2, yc - s / 2), s, s,
                                           facecolor=color, edgecolor='0.4', lw=0.2, zorder=6))
                last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
                last_h.set_array([])
            for sta, row in T._sta_df.iterrows():
                ax.plot(row['x'], row['y'], '^', ms=5, mfc='#FFD700', mec='k', mew=0.6, zorder=13)
            T._faults(ax, 1.5)
            ax.set_xlim(T.X_START, T.X_END)
            ax.set_ylim(T.Y_START, T.Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=4)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ri == 0:
                n_rays = res['diagnostics']['n_rays'] if res is not None else 0
                corr = res['checkerboard']['corr_well_sampled'] if res is not None else float('nan')
                ax.set_title(f'N={n_rays:,}\ncorr(N>={DISPLAY_COUNT_MIN})={corr:.2f}',
                             fontsize=FS_N, fontweight='bold', pad=14)
                ax_pos = ax.get_position()
                fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + period_label_offset_frac,
                         period_labels[ci].replace('\n–', ' – ').replace('\n', ' '),
                         fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
                ref_ax_pos = ax_pos
            if ci == 0:
                ax.set_ylabel(row_labels[ri], fontsize=7)

    if last_h:
        cell = gs[:, n_cols - 1].get_position(fig)
        cax = fig.add_axes([cell.x0, cell.y0, cell.width, cell.height])
        cb_ = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, vmax, 5),
                           format=FormatStrFormatter('%.3f'))
        cb_.set_label('s [s/km]', fontsize=7.5)
        cb_.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=7,
              framealpha=0.9, bbox_to_anchor=(0.45, 0.05))
    fig.suptitle(
        'Johnson et al. (2011) 2-D SWS tomography: checkerboard resolution test '
        f'(newdata, grade-3 filter, blocks shown only if N>={DISPLAY_COUNT_MIN})',
        fontsize=10, fontweight='bold', y=0.99)
    return fig


def main():
    T._setup_environment(need_tracer=False)

    print('Loading JOINT six-station grade-3 ray cache (reusing existing cache, no re-tracing)...')
    joint_summ, joint_cells, station_index, sta_xy = S.build_joint_cache()
    print(f'Joint cache: {len(joint_summ):,} rays total across {len(S.STATIONS)} stations')

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

    print('\nBuilding checkerboard figure...')
    fig = make_checkerboard_figure(results, period_labels)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
