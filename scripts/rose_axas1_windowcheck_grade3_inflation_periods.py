#!/usr/bin/env python3
"""
rose_axas1_windowcheck_grade3_inflation_periods.py

AXAS1-only rose plot, same data/filter as rose_7period_regions_windowcheck_grade3.py (windowcheck
grade-3: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg) and the same
per-panel N=/Avg=/cos= stats line (circular mean fast direction + standard error, and axial
cosine similarity to the "Before Eruption" panel), but with the 7 TIME PERIODS redefined per
explicit user request: "Before Eruption" and "During Eruption" are unchanged, but the 5
post-eruption periods are no longer equal-COUNT bins -- they are 5 bins of EQUAL INFLATION
AMOUNT, from the ASHES vent field BOTPT (RS03ASHS-MJ03B-09-BOTPTA304, Western Caldera)
de-tided seafloor-depth record (bpr_inflation_periods_ashes.py's build_inflation_based_
periods(), which bins on the 30-day rolling mean of "m of uplift" -- see that module's
docstring for its zero-reference and period-boundary methodology, DIFFERENT from the Eastern/
Central Caldera versions since this record only starts 2017-08-15, and the companion
plot_bpr_detided_depth_and_inflation_ashes.py figure).

This directly compares the AXAS1 fast-direction rose across periods of equal PHYSICAL INFLATION
(rather than equal event count), so the panel-to-panel change reflects equal steps of caldera
uplift instead of equal numbers of measurements.

Produces: rose_axas1_windowcheck_grade3_inflation_periods.pdf (1 page, 1 row x 7 columns) -- a
NEW file, does not touch rose_7period_regions_windowcheck_grade3.pdf or any other existing
output.

Run with:
    python3 rose_axas1_windowcheck_grade3_inflation_periods.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_6stations_newdata_snr_grades import _draw_rose, _period_colors
from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _subset, _circular_mean_and_se_deg,
    _axial_cosine_similarity_deg,
)
from bpr_inflation_periods_ashes import build_inflation_based_periods

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_axas1_windowcheck_grade3_inflation_periods.pdf')

STATION = 'AXAS1'


def _single_line_label(label):
    return label.replace('\n', ' ')


def make_figure(df, periods, title):
    n_cols = len(periods)
    colors = _period_colors(n_cols)
    panel_size = 2.4

    fig, axes = plt.subplots(1, n_cols, figsize=(panel_size * n_cols, panel_size + 1.2),
                             subplot_kw={'projection': 'polar'})

    ref_phi = np.nan
    for col_idx, (label, t0, t1) in enumerate(periods):
        sub = _subset(df, t0, t1)
        ax = axes[col_idx]
        phi_vals = sub['phi_az'].values
        _draw_rose(ax, phi_vals, np.ones(len(sub)), colors[col_idx])

        n_events = len(sub)
        mean_phi, se_phi = _circular_mean_and_se_deg(phi_vals)
        if col_idx == 0:
            ref_phi = mean_phi
        cos_sim = _axial_cosine_similarity_deg(mean_phi, ref_phi)

        if n_events == 0 or np.isnan(mean_phi):
            stats_txt = f'N={n_events:,}'
        else:
            avg_txt = f'Avg={mean_phi:.0f}°' if np.isnan(se_phi) else f'Avg={mean_phi:.0f}±{se_phi:.0f}°'
            cos_txt = f', cos={cos_sim:.4f}' if not np.isnan(cos_sim) else ''
            stats_txt = f'N={n_events:,}, {avg_txt}{cos_txt}'

        ax.set_title(f'{_single_line_label(label)}\n{stats_txt}', fontsize=8, fontweight='bold', pad=6)

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.08)
    fig.tight_layout()
    return fig


def main():
    print(f'Loading {STATION} windowcheck grade-3 events ({GRADE["label"]})...')
    raw = load_station_raw(STATION)
    df = apply_grade(raw, GRADE)
    print(f'  {STATION}: {len(df):,} events pass the filter')

    periods = build_inflation_based_periods()
    print('\nInflation-based 7-period scheme:')
    for label, t0, t1 in periods:
        t0s = t0.date() if t0 is not None else '(start of record)'
        t1s = t1.date() if t1 is not None else '(present)'
        print(f'  {label!r:35s}  {t0s} -> {t1s}')

    title = (f'{STATION} (windowcheck) — fast direction, 5 post-eruption bins of EQUAL '
             f'INFLATION AMOUNT (not equal N)\n{GRADE["label"]}')
    fig = make_figure(df, periods, title)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
