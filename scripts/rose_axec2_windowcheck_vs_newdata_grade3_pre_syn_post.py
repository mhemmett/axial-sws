#!/usr/bin/env python3
"""
rose_axec2_windowcheck_vs_newdata_grade3_pre_syn_post.py

Extends rose_axec2_windowcheck_vs_newdata_grade3_matched_batches.py (same matched-batch-number
newdata-vs-windowcheck comparison, same grade-3 filter) into THREE eruption-relative panels per
dataset -- Pre-eruption, Syn-eruption, Post-eruption -- per explicit user follow-up request.
2 rows (newdata, windowcheck) x 3 columns (pre/syn/post) = 6 rose panels.

Batch numbers (whatever's done so far in the windowcheck rerun) are read from disk at run
time, matched against the same newdata batch numbers, same as the base script.

Produces: rose_axec2_windowcheck_vs_newdata_grade3_pre_syn_post.pdf (1 page, 2x3 rose grid) --
a NEW file, does not touch any existing output.

Run with:
    python3 rose_axec2_windowcheck_vs_newdata_grade3_pre_syn_post.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_axec2_windowcheck_vs_newdata_grade3_pre_syn_post.pdf')

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

PERIODS = [
    ('Pre-eruption', None, ERUPTION_START),
    ('Syn-eruption', ERUPTION_START, ERUPTION_END),
    ('Post-eruption', ERUPTION_END, None),
]


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
    print(f'{len(batch_nums)} windowcheck batches complete so far '
         f'(range {min(batch_nums)}-{max(batch_nums)})')

    new_all = R.load_batches(R.NEWDATA_DIR, R.NEWDATA_PATTERN, batch_nums)
    wc_all = R.load_batches(R.WC_DIR, R.WC_PATTERN, batch_nums)
    new_g3 = R.apply_grade3(new_all)
    wc_g3 = R.apply_grade3(wc_all)
    new_g3 = new_g3.copy()
    wc_g3 = wc_g3.copy()
    new_g3['t'] = pd.to_datetime(new_g3['datetime'], utc=True, format='ISO8601')
    wc_g3['t'] = pd.to_datetime(wc_g3['datetime'], utc=True, format='ISO8601')
    print(f'Newdata grade-3: {len(new_g3):,}   Window-check grade-3: {len(wc_g3):,}')

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.15)

    rows = [('Newdata (production, non-windowcheck)', new_g3, '#CC0000'),
           ('Window-check (in progress)', wc_g3, '#0072B2')]

    for ri, (row_label, df, color) in enumerate(rows):
        for ci, (period_label, t0, t1) in enumerate(PERIODS):
            sub = restrict_period(df, t0, t1)
            ax = fig.add_subplot(gs[ri, ci], projection='polar')
            R._draw_rose(ax, sub['phi_az'].values, color)
            ax.set_title(f'{row_label}\n{period_label}\nN={len(sub):,}',
                        fontsize=8.5, fontweight='bold', pad=10)
            print(f'  {row_label} / {period_label}: N={len(sub):,}')

    n_batches = len(batch_nums)
    b_lo, b_hi = min(batch_nums), max(batch_nums)
    fig.suptitle(
        f'AXEC2 batches {b_lo}-{b_hi} ({n_batches} done so far): newdata vs. windowcheck, '
        f'pre/syn/post-eruption\n{R.GRADE["label"]}',
        fontsize=11, fontweight='bold', y=0.99)

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
