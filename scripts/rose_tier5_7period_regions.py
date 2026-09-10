#!/usr/bin/env python3
"""
rose_tier5_7period_regions.py

Standalone, large-font remake of the tier-5 (strictest QC) page from
rose_7period_6stations_newdata.py's 6-tier fast-direction rose grid, per explicit user request:

  - All fonts doubled relative to the original tier-5 page (N= count text 6.5->13pt as the
    doubled baseline, station label 11->22pt), with extra row spacing added so the larger
    fonts don't collide.
  - Left-hand-side region group titles -- "Western Caldera" above AXAS1's row, "Central
    Caldera" above AXCC1's row, "Eastern Caldera" above AXEC1's row -- each LARGER than the
    per-station label font (26pt vs 22pt).
  - Overall suptitle "Shear-wave Splitting Fast Direction Across an Eruption Cycle", LARGER
    than the region titles (32pt vs 26pt).
  - Per-period column titles (top row only) forced onto a single line (no wrapped
    "Mon YYYY\\n-- Mon YYYY"), sized 50% larger than the "N=" event-count text below it
    (19.5pt vs 13pt).

Font-size hierarchy (points): N= text (13, = 2x original 6.5) < period title (19.5, = 1.5x
N=) < station label (22, = 2x original 11) < region title (26) < overall suptitle (32).

Uses the SAME tier-5 QC (quality>=0.5, dt<T_dom/2, phi_error<10deg, dt_error<0.05s), 7
eruption-relative periods, and station data loading as rose_7period_6stations_newdata.py
(imported directly, not reimplemented) -- only the figure layout/typography differs.

Produces: rose_tier5_7period_regions.pdf (1 page)

Run with:
    python3 rose_tier5_7period_regions.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_6stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, _draw_rose, _period_colors,
    ERUPTION_START, ERUPTION_END,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_tier5_7period_regions.pdf')

# ── Font-size hierarchy (see module docstring) ───────────────────────────────────────────
FS_N = 19.5   # 1.5x the original 13pt
FS_PERIOD = 19.5
FS_STATION = 22
FS_REGION = 26
FS_SUPTITLE = 32

REGION_TITLES = {
    'AXAS1': 'Western Caldera',
    'AXCC1': 'Central Caldera',
    'AXEC1': 'Eastern Caldera',
}


def _build_time_periods_single_line(all_df):
    """Same 7-period scheme as rose_7period_6stations_newdata.py's _build_time_periods, but
    each post-eruption label is forced onto ONE line (no newline) per explicit user request."""
    post = all_df[all_df['t'] >= ERUPTION_END].copy()
    post_times = post['t'].sort_values().reset_index(drop=True)
    n = len(post_times)

    boundaries = [ERUPTION_END]
    for i in range(1, 5):
        idx = int(round(i * n / 5))
        idx = min(idx, n - 1)
        boundaries.append(post_times.iloc[idx])
    boundaries.append(None)

    def _fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else 'present'

    periods = [
        ('Before Eruption', None, ERUPTION_START),
        ('During Eruption', ERUPTION_START, ERUPTION_END),
    ]
    for i in range(5):
        t0, t1 = boundaries[i], boundaries[i + 1]
        periods.append((f'{_fmt(t0)} - {_fmt(t1)}', t0, t1))

    return periods


def _subset(df, t_start, t_end):
    m = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
    if t_end is not None:
        m = m & (df['t'] < t_end)
    return df[m]


def make_rose_figure(dfs, time_periods, title):
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_size = 3.5   # wide enough for the single-line period titles at FS_PERIOD without overlap
    row_gap = 0.55            # extra vertical gap between rows (doubled fonts need more room)
    region_gap_extra = 0.9    # ADDITIONAL gap reserved above a region's first row, for its title
    left_margin = 1.7
    top_margin = 2.1

    n_region_starts = sum(1 for s in STATION_ORDER if s in REGION_TITLES)
    fig_h = (n_rows * (panel_size + row_gap) + top_margin
             + n_region_starts * region_gap_extra)
    fig_w = n_cols * panel_size + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=FS_SUPTITLE, fontweight='bold', y=0.997)

    row_tops = []
    y_cursor = fig_h - top_margin
    for row_idx, sta in enumerate(STATION_ORDER):
        if sta in REGION_TITLES:
            y_cursor -= region_gap_extra
        row_tops.append(y_cursor)
        y_cursor -= (panel_size + row_gap)

    col_lefts = [left_margin + col_idx * panel_size for col_idx in range(n_cols)]

    for row_idx, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        cell_top = row_tops[row_idx]

        if sta in REGION_TITLES:
            # AXAS1 (Western Caldera) is row 0, which ALSO carries the column headers
            # (period label + N=) directly above its axes -- push the region title higher so
            # it clears that header text instead of overlapping it. AXCC1/AXEC1 have no column
            # headers above them (not row 0), so the simpler offset is fine there.
            region_text_height_frac = (FS_REGION / 72.0) / fig_h   # font size (pt->in) as a fig fraction
            if sta == 'AXAS1':
                region_y = cell_top / fig_h + 0.075 - region_text_height_frac
            else:
                region_y = (cell_top + region_gap_extra * 0.3) / fig_h + region_text_height_frac / 2.0
            fig.text(0.01, region_y, REGION_TITLES[sta], fontsize=FS_REGION, fontweight='bold',
                     ha='left', va='bottom', transform=fig.transFigure)

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
                # Period label (single line, 1.5x the N= fontsize) drawn ABOVE the N= count
                # as its own text element, not sharing one uniform-size title string.
                ax.set_title(f'N={n_events:,}', fontsize=FS_N, fontweight='bold', pad=34)
                ax_pos = ax.get_position()
                fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + 0.026, label,
                         fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
            else:
                ax.set_title(f'N={n_events:,}', fontsize=FS_N, fontweight='bold', pad=8)

        y_center = (cell_top - panel_size / 2) / fig_h
        fig.text(0.01, y_center, sta, fontsize=FS_STATION, fontweight='bold',
                 ha='left', va='center', transform=fig.transFigure)

    return fig


def main():
    print('Loading new mfast max_dt=0.2s splitting results (tier 5) for all 6 stations...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
    dfs = {sta: apply_tier(raw[sta], 5) for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        print(f'  {sta}: {len(dfs[sta]):,}')

    all_combined = pd.concat(raw.values(), ignore_index=True)
    time_periods = _build_time_periods_single_line(all_combined)

    fig = make_rose_figure(
        dfs, time_periods,
        title='Shear-wave Splitting Fast Direction Across an Eruption Cycle',
    )
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
