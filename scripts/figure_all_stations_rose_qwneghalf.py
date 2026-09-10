"""
All-station, 7-period fast-direction rose plot ("Fast Direction Over Time and Space"),
single page, high resolution. One-off script, not part of the pipeline. Reuses
rose_plots_lqt_pykonal_unweighted.py's exact data loading, colorscale, period
definitions, and 6-row (station) x 7-col (period) layout, with the same
phi_error<20 deg / dt_error<0.04 s filter as the baseline "filter" variant, but
with the quality threshold expanded to Q_w > -0.5 (QW_ABOVE_NEG_HALF) instead of
Q_w > 0.5. No back-azimuth rings, unweighted (count-weighted rose bars).

Run with: python3 figure_all_stations_rose_qwneghalf.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rose_plots_lqt_pykonal_unweighted import (  # noqa: E402
    STATION_ORDER, SECTION_LABELS, QW_ABOVE_NEG_HALF, PHI_ERR_MAX, DT_ERR_MAX,
    load_station_raw, _build_time_periods, _subset, _draw_rose, _period_colors,
)

OUT_PDF = os.path.join(HERE, 'figure_all_stations_rose_qwneghalf.pdf')
OUT_PNG = os.path.join(HERE, 'figure_all_stations_rose_qwneghalf.png')


def _station_label(sta):
    """Drop the "AX" prefix for display (e.g. AXEC2 -> EC2)."""
    return sta[2:] if sta.startswith('AX') else sta


def make_rose_figure_local(dfs, time_periods, title):
    """Local copy of rose_plots_lqt_pykonal_unweighted.make_rose_figure, with
    "AX"-stripped station labels and a larger, closer-spaced title - kept as a
    copy rather than editing the shared production script."""
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_size = 1.8
    row_gap = 0.10
    left_margin = 0.75
    top_margin = 0.72
    section_gap = 0.45

    section_rows = {sta: i for i, sta in enumerate(STATION_ORDER) if sta in SECTION_LABELS}

    fig_h = (n_rows * (panel_size + row_gap) + top_margin
             + len(section_rows) * section_gap)
    fig_w = n_cols * panel_size + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.0)

    row_tops = []
    y_cursor = fig_h - top_margin
    for row_idx, sta in enumerate(STATION_ORDER):
        if sta in SECTION_LABELS:
            y_cursor -= section_gap
        row_tops.append(y_cursor)
        y_cursor -= (panel_size + row_gap)

    col_lefts = [left_margin + col_idx * panel_size for col_idx in range(n_cols)]

    for row_idx, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        cell_top = row_tops[row_idx]

        for col_idx, (label, t_start, t_end) in enumerate(time_periods):
            sub = _subset(df, t_start, t_end)

            x0 = col_lefts[col_idx] / fig_w
            y0 = (cell_top - panel_size) / fig_h
            w = panel_size / fig_w
            h = panel_size / fig_h

            ax = fig.add_axes([x0, y0, w, h], projection='polar')

            phi_vals = sub['phi_az'].values
            wts = np.ones(len(sub))

            _draw_rose(ax, phi_vals, wts, colors[col_idx])

            n_events = len(sub)

            if row_idx == 0:
                ax.set_title(f'{label}\nN={n_events:,}', fontsize=6.5,
                             fontweight='bold', pad=2)
            else:
                ax.set_title(f'N={n_events:,}', fontsize=6.5,
                             fontweight='bold', pad=2)

        y_center = (cell_top - panel_size / 2) / fig_h
        fig.text(0.0, y_center, _station_label(sta), fontsize=11, fontweight='bold',
                  ha='left', va='center', rotation=0,
                  transform=fig.transFigure)

    for sta, row_idx in section_rows.items():
        cell_top = row_tops[row_idx]
        y_label = (cell_top + section_gap * 0.55) / fig_h
        fig.text(left_margin / fig_w, y_label,
                  SECTION_LABELS[sta],
                  fontsize=13, fontweight='bold', color='black',
                  ha='left', va='bottom',
                  transform=fig.transFigure,
                  fontfamily='Arial')

    return fig


def main():
    print(f'Loading LQT + PyKonal-FMM combined results for all stations, '
          f'Q_w>{QW_ABOVE_NEG_HALF}, phi_error<{PHI_ERR_MAX}, dt_error<{DT_ERR_MAX}...')
    dfs = {}
    for sta in STATION_ORDER:
        print(f'  Loading {sta}...')
        raw = load_station_raw(sta)
        dfs[sta] = raw[(raw['quality'] > QW_ABOVE_NEG_HALF) &
                       (raw['phi_error'] < PHI_ERR_MAX) &
                       (raw['dt_error'] < DT_ERR_MAX)]
        print(f'    {sta}: {len(raw):,} baseline -> {len(dfs[sta]):,} filtered')

    all_combined = pd.concat(dfs.values(), ignore_index=True)
    time_periods = _build_time_periods(all_combined)

    fig = make_rose_figure_local(
        dfs, time_periods,
        title='Fast Direction Over Time and Space',
    )
    fig.savefig(OUT_PDF, dpi=600, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
