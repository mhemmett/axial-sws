"""
All-station, 7-period fast-direction rose plot ("Fast Direction Over Time and Space"),
multi-page PDF -- one page per dt filter, all other filters held fixed. One-off script,
not part of the pipeline. Reuses rose_plots_lqt_pykonal_unweighted.py's exact data
loading, colorscale, period definitions, and 6-row (station) x 7-col (period) layout
(same base as figure_all_stations_rose_qwneghalf.py), but with:
  - quality (Q_w) > 0.7 (QW_MIN_STRICT) instead of Q_w > -0.5
  - phi_error<20 deg / dt_error<0.04 s (same as the qwneghalf variant)
  - one page per dt filter, all sharing the same quality/phi_error/dt_error gate and
    the same time-period boundaries (computed once from the Q_w>0.7 baseline, before
    any dt filter, so pages stay comparable to each other):
      1. All dt (no filter)
      2. dt < 0.24 s
      3. dt < 0.15 s
      4. dt < 0.10 s
      5. dt > 0.24 s
      6. dt > 0.15 s
      7. dt > 0.10 s
  ...followed by one page per 0.05 s delay-time bin, covering the full observed
  range (max dt at Q_w>0.7 is 0.29 s, so 6 bins from 0-0.30 s cover everything):
      8.  0.00 <= dt < 0.05 s
      9.  0.05 <= dt < 0.10 s
      10. 0.10 <= dt < 0.15 s
      11. 0.15 <= dt < 0.20 s
      12. 0.20 <= dt < 0.25 s
      13. 0.25 <= dt < 0.30 s
No back-azimuth rings, unweighted (count-weighted rose bars).

Run with: python3 figure_all_stations_rose_qw07_dtpages.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rose_plots_lqt_pykonal_unweighted import (  # noqa: E402
    STATION_ORDER, SECTION_LABELS, QW_MIN_STRICT, PHI_ERR_MAX, DT_ERR_MAX,
    load_station_raw, _build_time_periods, _subset, _draw_rose, _period_colors,
)

OUT_PDF = os.path.join(HERE, 'figure_all_stations_rose_qw07_dtpages.pdf')

# ── dt filters, one page each: (title suffix, mask function over a dt Series) ────
DT_PAGES = [
    ('All dt', lambda dt: pd.Series(True, index=dt.index)),
    ('dt < 0.24 s', lambda dt: dt < 0.24),
    ('dt < 0.15 s', lambda dt: dt < 0.15),
    ('dt < 0.10 s', lambda dt: dt < 0.10),
    ('dt > 0.24 s', lambda dt: dt > 0.24),
    ('dt > 0.15 s', lambda dt: dt > 0.15),
    ('dt > 0.10 s', lambda dt: dt > 0.10),
]

# 0.05 s delay-time bins, one page each, covering the full observed range.
DT_BIN_WIDTH = 0.05
DT_BIN_MAX = 0.30
_bin_edges = np.arange(0.0, DT_BIN_MAX + DT_BIN_WIDTH * .5, DT_BIN_WIDTH)
for _lo, _hi in zip(_bin_edges[:-1], _bin_edges[1:]):
    DT_PAGES.append((
        f'{_lo:.2f} <= dt < {_hi:.2f} s',
        lambda dt, lo=_lo, hi=_hi: (dt >= lo) & (dt < hi),
    ))


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
          f'Q_w>{QW_MIN_STRICT}, phi_error<{PHI_ERR_MAX}, dt_error<{DT_ERR_MAX}...')
    dfs_base = {}
    for sta in STATION_ORDER:
        print(f'  Loading {sta}...')
        raw = load_station_raw(sta)
        dfs_base[sta] = raw[(raw['quality'] > QW_MIN_STRICT) &
                            (raw['phi_error'] < PHI_ERR_MAX) &
                            (raw['dt_error'] < DT_ERR_MAX)]
        print(f'    {sta}: {len(raw):,} baseline -> {len(dfs_base[sta]):,} filtered (Q_w>{QW_MIN_STRICT})')

    # Time-period boundaries fixed once from the Q_w>0.7 baseline (before any dt
    # filter), so all 7 pages share the same period edges and stay comparable.
    all_combined = pd.concat(dfs_base.values(), ignore_index=True)
    time_periods = _build_time_periods(all_combined)

    with PdfPages(OUT_PDF) as pdf:
        for page_label, dt_mask_fn in DT_PAGES:
            print(f'\n--- Page: {page_label} ---')
            dfs_page = {}
            for sta in STATION_ORDER:
                df = dfs_base[sta]
                dfs_page[sta] = df[dt_mask_fn(df['dt'])]
                print(f'  {sta}: {len(dfs_page[sta]):,} events')

            fig = make_rose_figure_local(
                dfs_page, time_periods,
                title=f'Fast Direction Over Time and Space (Q_w>{QW_MIN_STRICT}, {page_label})',
            )
            pdf.savefig(fig, dpi=600, bbox_inches='tight')
            plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
