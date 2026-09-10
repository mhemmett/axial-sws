#!/usr/bin/env python3
"""
rose_7period_5stations_newdata_baz.py

Back-azimuth ring version of rose_7period_5stations_newdata.py: same 5-row (station) x
7-col (time period) grid, same data/QC/tiers (imports STATION_ORDER, load_station_raw(),
apply_tier(), TIERS, and every filter threshold directly from that script so the two can
never drift apart) -- but every cell's central rose is now surrounded by 6 back-azimuth
(station -> event, deg from N) sub-roses, one per 60 deg BAZ bin, following this repo's
established ring-plot convention (rose_plots_baz.py / rose_plots_lqt_pykonal_unweighted_baz.py
/ axec1_rose_by_depth_lqt_pykonal.py's page-2 ring): central rose + 6 satellite sub-roses
arranged around it at 60-90-...-330 deg.

Back-azimuth is taken directly from the `back_azimuth` column already present in the new
mfast max_dt=0.2s data (station -> event, deg from N) -- no catalog join needed, unlike the
older LQT-pykonal baz script which had to join back-azimuth in from the MLdd catalogs.

5 pages, same cumulative QC tiers as the sibling (rose_7period_5stations_newdata.py):
    Page 1: raw (success & dt>0)
    Page 2: + quality >= 0.5
    Page 3: + dt < T_dom/2 (cycle-skip-risk cut, per-event dominant period)
    Page 4: + phi_error < 20 deg
    Page 5: + dt_error < 0.05 s
    Page 6: + dt_error < 0.05s, phi_error < 10 deg (stricter)

Produces: rose_7period_5stations_newdata_baz.pdf  (6 pages)

Run with:
    python3 rose_7period_5stations_newdata_baz.py
"""

import numpy as np

from rose_7period_5stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, TIERS, _build_time_periods,
    _period_colors, _draw_rose, QW_MIN, PHI_ERR_MAX, PHI_ERR_MAX_STRICT, DT_ERR_MAX,
)

import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_7period_5stations_newdata_baz.pdf')

# Back-azimuth bins: 6 x 60 deg bins (same convention as rose_plots_baz.py /
# rose_plots_lqt_pykonal_unweighted_baz.py)
BAZ_BINS = np.arange(0, 361, 60)
BAZ_CTRS = (BAZ_BINS[:-1] + BAZ_BINS[1:]) / 2


def _subset(df, t0, t1):
    import pandas as pd
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None:
        m = m & (df['t'] < t1)
    return df[m]


def make_ring_figure(dfs, time_periods, title):
    """5-row (station) x 7-col (time period) grid, each cell a central rose + 6
    back-azimuth sub-roses arranged in a ring around it."""
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_w = 2.6
    panel_h = 2.6
    row_gap = 0.15
    left_margin = 1.0
    top_margin = 0.65

    panel_scale = min(panel_w, panel_h)
    main_r = panel_scale * 0.18
    ring_r_h = panel_scale * 0.35
    ring_r_v = panel_scale * 0.42
    small_r = panel_scale * 0.12

    fig_h = n_rows * (panel_h + row_gap) + top_margin
    fig_w = n_cols * panel_w + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold', y=1.003)

    row_tops = []
    y_cursor = fig_h - top_margin
    for row_idx in range(n_rows):
        row_tops.append(y_cursor)
        y_cursor -= (panel_h + row_gap)

    for row_idx, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        cell_top = row_tops[row_idx]
        cy_in = cell_top - panel_h / 2

        for col_idx, (label, t0, t1) in enumerate(time_periods):
            sub = _subset(df, t0, t1)
            color = colors[col_idx]
            cx_in = left_margin + (col_idx + 0.5) * panel_w

            ax_main = fig.add_axes(
                [(cx_in - main_r) / fig_w, (cy_in - main_r) / fig_h,
                 2 * main_r / fig_w, 2 * main_r / fig_h],
                projection='polar')
            phi_vals = sub['phi_az'].values
            _draw_rose(ax_main, phi_vals, np.ones(len(sub)), color)
            if row_idx == 0:
                ax_main.set_title(f'{label}\nN={len(sub):,}', fontsize=6.5,
                                  fontweight='bold', pad=2)
            else:
                ax_main.set_title(f'N={len(sub):,}', fontsize=5.5,
                                  fontweight='bold', pad=1)

            for baz_lo, baz_hi, baz_c in zip(BAZ_BINS[:-1], BAZ_BINS[1:], BAZ_CTRS):
                baz_rad = np.deg2rad(baz_c)
                sc_cx = cx_in + ring_r_h * np.sin(baz_rad)
                sc_cy = cy_in + ring_r_v * np.cos(baz_rad)

                ax_s = fig.add_axes(
                    [(sc_cx - small_r) / fig_w, (sc_cy - small_r) / fig_h,
                     2 * small_r / fig_w, 2 * small_r / fig_h],
                    projection='polar')

                baz_mask = (sub['back_azimuth'] >= baz_lo) & (sub['back_azimuth'] < baz_hi)
                sub_baz = sub[baz_mask]
                _draw_rose(ax_s, sub_baz['phi_az'].values, np.ones(len(sub_baz)), color)
                ax_s.set_title(f'{int(baz_lo)}–{int(baz_hi)}°\nN={len(sub_baz):,}',
                               fontsize=3.8, pad=1)

        y_center = (cell_top - panel_h / 2) / fig_h
        fig.text(0.0, y_center, sta, fontsize=11, fontweight='bold',
                  ha='left', va='center', transform=fig.transFigure)

    return fig


def main():
    print('Loading new mfast max_dt=0.2s splitting results (AXCC1, AXEC1, AXEC3, AXAS1: '
          'complete; AXEC2 2015-2021: partial, in-progress re-run)...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
        print(f'  {sta}: {len(raw[sta]):,} baseline events (success & dt>0)')

    import pandas as pd
    all_combined = pd.concat(raw.values(), ignore_index=True)
    time_periods = _build_time_periods(all_combined)

    with PdfPages(OUT_PDF) as pdf:
        for tier_idx in range(len(TIERS)):
            dfs = {sta: apply_tier(raw[sta], tier_idx) for sta in STATION_ORDER}
            for sta in STATION_ORDER:
                print(f'  tier {tier_idx} ({TIERS[tier_idx]}) {sta}: {len(dfs[sta]):,}')
            fig = make_ring_figure(
                dfs, time_periods,
                title=(f'AXAS1/AXCC1/AXEC1/AXEC2/AXEC3 fast direction rose w/ back-azimuth '
                       f'(new mfast max_dt=0.2s data) — {TIERS[tier_idx]}'))
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
