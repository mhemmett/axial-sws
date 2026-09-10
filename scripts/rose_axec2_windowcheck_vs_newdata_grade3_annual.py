#!/usr/bin/env python3
"""
rose_axec2_windowcheck_vs_newdata_grade3_annual.py

Extends rose_axec2_windowcheck_vs_newdata_grade3_matched_batches.py (same matched-batch-number
newdata-vs-windowcheck comparison, same grade-3 filter) to an ANNUAL period breakdown, per
explicit user follow-up request, now that the full AXEC2 2015-2021 windowcheck rerun is
complete (all 497 batches):

    Pre-eruption 2015 | Syn-eruption 2015 | Post-eruption 2015 | 2016 | 2017 | 2018 | 2019 |
    2020 | 2021

(9 panels per dataset, 2 rows x 9 columns = 18 rose panels total: newdata row, windowcheck
row). Same ERUPTION_START/ERUPTION_END boundaries used throughout this project; "Post-eruption
2015" runs from ERUPTION_END to 2016-01-01; each subsequent panel is a plain calendar year
(2021 is the last full year in this dataset -- the windowcheck rerun only covers 2015-2021,
unlike the newdata 6-station catalogs which also have a 2022-2026 half).

Batch numbers are read from disk at run time (all 497, now complete) via the base script's
get_completed_batch_nums(), matched against the same newdata batch numbers.

Produces: rose_axec2_windowcheck_vs_newdata_grade3_annual.pdf (1 page, 2x9 rose grid) -- a NEW
file, does not touch any existing output.

Run with:
    python3 rose_axec2_windowcheck_vs_newdata_grade3_annual.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import rose_axec2_windowcheck_vs_newdata_grade3_matched_batches as R


def circular_mean_phi_deg(phi_az_vals):
    """Circular (vector) mean of axially-symmetric (mod 180) fast directions, via the
    doubled-angle trick -- returns the mean fast direction, mod 180."""
    if len(phi_az_vals) == 0:
        return np.nan
    angles = 2.0 * np.radians(np.asarray(phi_az_vals) % 180.0)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return float(np.degrees(np.arctan2(s, c)) / 2.0) % 180.0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_axec2_windowcheck_vs_newdata_grade3_annual.pdf')

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')


def build_annual_periods():
    periods = [
        ('Pre-eruption\n2015', None, ERUPTION_START),
        ('Syn-eruption\n2015', ERUPTION_START, ERUPTION_END),
        ('Post-eruption\n2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for year in range(2016, 2022):
        t0 = pd.Timestamp(f'{year}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
        periods.append((f'{year}', t0, t1))
    return periods


def restrict_period(df, t0, t1):
    t = df['t']
    m = pd.Series(True, index=df.index)
    if t0 is not None:
        m &= (t >= t0)
    if t1 is not None:
        m &= (t < t1)
    return df[m]


def main():
    batch_nums = R.get_completed_batch_nums()
    print(f'{len(batch_nums)} windowcheck batches complete '
         f'(range {min(batch_nums)}-{max(batch_nums)})')

    new_all = R.load_batches(R.NEWDATA_DIR, R.NEWDATA_PATTERN, batch_nums)
    wc_all = R.load_batches(R.WC_DIR, R.WC_PATTERN, batch_nums)
    new_g3 = R.apply_grade3(new_all).copy()
    wc_g3 = R.apply_grade3(wc_all).copy()
    new_g3['t'] = pd.to_datetime(new_g3['datetime'], utc=True, format='ISO8601')
    wc_g3['t'] = pd.to_datetime(wc_g3['datetime'], utc=True, format='ISO8601')
    print(f'Newdata grade-3: {len(new_g3):,}   Window-check grade-3: {len(wc_g3):,}')

    periods = build_annual_periods()
    n_per = len(periods)

    fig = plt.figure(figsize=(2.1 * n_per, 6.0))
    gs = fig.add_gridspec(2, n_per, hspace=0.85, wspace=0.15)

    rows = [('Newdata', new_g3, '#CC0000'),
           ('Window-check', wc_g3, '#0072B2')]

    for ri, (row_label, df, color) in enumerate(rows):
        for ci, (period_label, t0, t1) in enumerate(periods):
            sub = restrict_period(df, t0, t1)
            ax = fig.add_subplot(gs[ri, ci], projection='polar')
            R._draw_rose(ax, sub['phi_az'].values, color)
            ax.set_title(f'{period_label}\nN={len(sub):,}', fontsize=7.5, fontweight='bold', pad=8)
            if ci == 0:
                ax.text(-0.3, 0.5, row_label, fontsize=9, fontweight='bold', rotation=90,
                       va='center', ha='center', transform=ax.transAxes)

            mean_phi = circular_mean_phi_deg(sub['phi_az'].values)
            mean_dt = float(sub['dt'].mean()) if len(sub) else np.nan
            stat_txt = (f'mean $\\phi$={mean_phi:.0f}°\nmean $\\delta t$={mean_dt:.3f}s'
                       if len(sub) else 'N=0')
            ax.text(0.5, -0.22, stat_txt, fontsize=6.5, ha='center', va='top',
                   transform=ax.transAxes)

            print(f'  {row_label} / {period_label.replace(chr(10), " ")}: N={len(sub):,} '
                 f'mean_phi={mean_phi:.1f} mean_dt={mean_dt:.4f}')

    n_batches = len(batch_nums)
    b_lo, b_hi = min(batch_nums), max(batch_nums)
    fig.suptitle(
        f'AXEC2 batches {b_lo}-{b_hi} ({n_batches} batches, full rerun complete): '
        f'newdata vs. windowcheck, annual breakdown\n{R.GRADE["label"]}',
        fontsize=11, fontweight='bold', y=1.02)

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
