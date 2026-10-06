#!/usr/bin/env python3
"""
rose_7period_axec2_windowcheck_grade3_ccal_periods.py

AXEC2 pulled out as a standalone 1-row figure from rose_7period_regions_windowcheck_grade3_ccal_periods.py.
Same data (windowcheck results via load_station_raw), same Grade 3 filter (apply_grade/GRADE),
same equal-inflation-amount 7-period scheme (bpr_inflation_periods_ccal.build_inflation_based_periods),
same rose drawing, period colours and font sizes as the AXEC2 row of that figure. Panel titles show
N and the circular mean +/- SE only (no cosine similarity), and the layout is one row with the
station name in the suptitle instead of a row label.

Produces: rose_7period_axec2_windowcheck_grade3_ccal_periods.pdf (1 page) and a matching .png (300 dpi)

Run with:
    python3 rose_7period_axec2_windowcheck_grade3_ccal_periods.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _subset, _circular_mean_and_se_deg, FS_N, FS_PERIOD, FS_SUPTITLE,
)
from rose_7period_6stations_newdata_snr_grades import _draw_rose, _period_colors
from bpr_inflation_periods_ccal import build_inflation_based_periods

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_7period_axec2_windowcheck_grade3_ccal_periods.pdf')
OUT_PNG = OUT_PDF.replace('.pdf', '.png')

STATION = 'AXEC2'


def make_figure(df, time_periods, title):
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_size = 3.5
    side_margin = 0.3
    top_margin = 1.45  # suptitle (1 line) + period label + N/Avg line
    bottom_margin = 0.3

    fig_w = n_cols * panel_size + 2 * side_margin
    fig_h = panel_size + top_margin + bottom_margin
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=FS_SUPTITLE, fontweight='bold', y=1 - 0.05 / fig_h, va='top')

    for col_idx, (label, t_start, t_end) in enumerate(time_periods):
        sub = _subset(df, t_start, t_end)
        ax = fig.add_axes([(side_margin + col_idx * panel_size) / fig_w, bottom_margin / fig_h,
                           panel_size / fig_w, panel_size / fig_h], projection='polar')

        phi_vals = sub['phi_az'].values
        _draw_rose(ax, phi_vals, np.ones(len(sub)), colors[col_idx])

        n_events = len(sub)
        mean_phi, se_phi = _circular_mean_and_se_deg(phi_vals)

        if n_events == 0 or np.isnan(mean_phi):
            title_txt = f'N={n_events:,}'
        else:
            avg_txt = f'Avg={mean_phi:.0f}°' if np.isnan(se_phi) else f'Avg={mean_phi:.0f}±{se_phi:.0f}°'
            title_txt = f'N={n_events:,}, {avg_txt}'
        ax.set_title(title_txt, fontsize=FS_N, fontweight='bold', pad=12)

        # Period label sits a fixed 0.55 in above the top of the panel.
        fig.text((side_margin + (col_idx + 0.5) * panel_size) / fig_w,
                 (bottom_margin + panel_size + 0.55) / fig_h, label,
                 fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')

    return fig


def main():
    print(f'Loading {STATION} windowcheck splitting results, filter: {GRADE["label"]} ...')
    df = apply_grade(load_station_raw(STATION), GRADE)
    print(f'  {STATION}: {len(df):,}')

    time_periods = build_inflation_based_periods()
    print('\nEqual-inflation-amount 7-period scheme (Central Caldera BOTPT):')
    for label, t0, t1 in time_periods:
        t0s = t0.date() if t0 is not None else '(start of record)'
        t1s = t1.date() if t1 is not None else '(present)'
        n = len(_subset(df, t0, t1))
        print(f'  {label!r:35s}  {t0s} -> {t1s}  N={n:,}')

    fig = make_figure(
        df, time_periods,
        title=f'{STATION} Shear-wave Splitting Fast Direction Across an Eruption Cycle',
    )
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}\nSaved {OUT_PNG}')


if __name__ == '__main__':
    main()
