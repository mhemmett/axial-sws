#!/usr/bin/env python3
"""
rose_tier5_7period_regions_newdata_snr2_q75.py

Same large-font/region-labeled figure layout as rose_tier5_7period_regions.pdf (font-size
hierarchy, region titles, suptitle, single-line period labels -- see that script's docstring
for the full typography rationale, not repeated here), but built from the "newdata" 6-station
dataset (rose_7period_6stations_newdata_snr_grades.py's data loading -- AXAS1, AXAS2, AXCC1,
AXEC1, AXEC2, AXEC3 from mfast_maxdt_pipeline_transfer/, NOT the separate AXEC2-only
max-dt-wraparound-fix re-run) with a DIFFERENT QC filter:

    SNR >= 2.0, phi_err <= 20 deg, dt_err <= 0.05 s, dt <= T_dom/2, quality (Q_w) >= 0.75

This is the "Page 1 (q>=0.75)" grade from rose_7period_6stations_newdata_snr_grades.py /
traveltime_anisotropy_7period_6stations_newdata_snr_grades.py (GRADES[3]) -- reused here
directly, not redefined.

Title, fonts, region groupings, and period-label formatting are all UNCHANGED from
rose_tier5_7period_regions.py; only the underlying data/filter changes.

Produces: rose_tier5_7period_regions_newdata_snr2_q75.pdf (1 page)

Run with:
    python3 rose_tier5_7period_regions_newdata_snr2_q75.py
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

from rose_7period_6stations_newdata_snr_grades import (
    STATION_ORDER, load_station_raw, apply_grade, _draw_rose, _period_colors,
    ERUPTION_START, ERUPTION_END, GRADES,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_tier5_7period_regions_newdata_snr2_q75.pdf')

GRADE = GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg
assert GRADE['snr_min'] == 2.0 and GRADE['qw_min'] == 0.75 and GRADE['phi_err_max'] == 20.0, GRADE

# ── Font-size hierarchy (verbatim from rose_tier5_7period_regions.py) ───────────────────
FS_N = 19.5
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

    panel_size = 3.5
    row_gap = 0.55
    region_gap_extra = 0.9
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
            region_text_height_frac = (FS_REGION / 72.0) / fig_h
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
    print(f'Loading newdata splitting results, filter: {GRADE["label"]} ...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
    dfs = {sta: apply_grade(raw[sta], GRADE) for sta in STATION_ORDER}
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
