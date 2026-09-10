#!/usr/bin/env python3
"""
plot_bpr_detided_depth_and_inflation_ashes.py

ASHES vent field (Western Caldera BOTPT, RS03ASHS-MJ03B-09-BOTPTA304) counterpart of
plot_bpr_detided_depth_and_inflation_ccal.py, built via process_bpr_detided_depth_ashes.py /
bpr_inflation_periods_ashes.py. Same two panels, EXCEPT this record only starts 2017-08-15 (see
that module's docstring) -- there is no data (and so no line) for 2015-01-01 through 2017-08-15,
which is left visibly blank (x-axis still spans the full 2015-01-01 to 2026-05-15 range) rather
than auto-cropped, so the absence is obvious rather than silently hidden.

  A. De-tided seafloor depth (m).
  B. "m of uplift" (inflation), referenced to zero at 2017-08-20 (the lowest point in the
     available record, not "right after the 2015 eruption" -- there's no 2015 data here), raw
     daily + 30-day rolling mean overlay, with the 5 equal-inflation-amount period boundaries
     (computed over this record's own available span) marked.

Produces (NEW file):
    bpr_detided_depth_and_inflation_ashes.pdf

Run with:
    python3 plot_bpr_detided_depth_and_inflation_ashes.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from bpr_inflation_periods_ashes import (
    load_daily_series, build_inflation_based_periods, ERUPTION_START, ERUPTION_END,
    ROLLING_DAYS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'bpr_detided_depth_and_inflation_ashes.pdf')

XLIM = (pd.Timestamp('2015-01-01', tz='UTC'), pd.Timestamp('2026-05-15', tz='UTC'))


def main():
    daily_depth, inflation, inflation_roll, ref_time, ref_depth = load_daily_series()
    periods = build_inflation_based_periods()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_top.plot(daily_depth.index, daily_depth.values, color='#333333', lw=0.8)
    ax_top.axvspan(ERUPTION_START, ERUPTION_END, color='red', alpha=0.15, label='Eruption')
    ax_top.set_ylabel('De-tided seafloor depth (m)')
    ax_top.set_title('ASHES BOTPT (RS03ASHS-MJ03B-09-BOTPTA304): De-tided Seafloor Depth and '
                     'Inflation\n(record starts 2017-08-15 -- no data before)',
                     fontsize=12, fontweight='bold')
    ax_top.legend(loc='lower left', fontsize=8)
    ax_top.grid(alpha=0.3)
    ax_top.set_xlim(*XLIM)

    ax_bot.plot(inflation.index, inflation.values, color='#0072B2', lw=0.5, alpha=0.4,
               label='Daily mean')
    ax_bot.plot(inflation_roll.index, inflation_roll.values, color='#0072B2', lw=1.8,
               label=f'{ROLLING_DAYS}-day rolling mean')
    ax_bot.axhline(0.0, color='black', lw=0.8, linestyle=':')
    ax_bot.axvspan(ERUPTION_START, ERUPTION_END, color='red', alpha=0.15)
    ax_bot.axvline(ref_time, color='#CC0000', lw=1.0, linestyle='--',
                  label=f'Zero reference ({ref_time.date()})')

    for label, t0, t1 in periods[2:]:
        if t0 is not None:
            ax_bot.axvline(t0, color='gray', lw=0.7, linestyle='--', alpha=0.7)

    ax_bot.set_ylabel('Inflation (m)')
    ax_bot.set_xlabel('Time')
    ax_bot.legend(loc='upper left', fontsize=8)
    ax_bot.grid(alpha=0.3)
    ax_bot.xaxis.set_major_locator(mdates.YearLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_bot.set_xlim(*XLIM)

    fig.tight_layout()
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')

    print('\n7-period scheme used for the boundary markers:')
    for label, t0, t1 in periods:
        t0s = t0.date() if t0 is not None else '(start of record)'
        t1s = t1.date() if t1 is not None else '(present)'
        print(f'  {label!r:45s}  {t0s} -> {t1s}')


if __name__ == '__main__':
    main()
