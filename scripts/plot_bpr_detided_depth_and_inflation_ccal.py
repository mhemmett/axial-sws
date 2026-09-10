#!/usr/bin/env python3
"""
plot_bpr_detided_depth_and_inflation_ccal.py

Central Caldera counterpart of plot_bpr_detided_depth_and_inflation.py (that script actually
uses the EASTERN Caldera BPR pair, RS03ECAL-MJ03E-06-BOTPTA302 -- see process_bpr_detided_
depth_ccal.py's docstring), built from the true Central Caldera BOTPT (RS03CCAL-MJ03F-05-
BOTPTA301) via process_bpr_detided_depth_ccal.py / bpr_inflation_periods_ccal.py. Same two
panels:
  A. De-tided seafloor depth (m) over the SWS catalog's time span (2015-01-22 to 2026-05-15).
  B. "m of uplift" (inflation), referenced to zero at the lowest point observed right after the
     2015 eruption, raw daily + 30-day rolling mean overlay, with the 5 equal-inflation-amount
     post-eruption period boundaries marked.

Note: this station's total re-inflation by 2026 (~2.6 m) is much larger than the Eastern
Caldera site's (~1.1 m) -- consistent with published Axial Seamount deformation studies, where
the caldera center (near the resurgent dome) shows substantially more subsidence/inflation than
flank sites.

Produces (NEW file):
    bpr_detided_depth_and_inflation_ccal.pdf

Run with:
    python3 plot_bpr_detided_depth_and_inflation_ccal.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from bpr_inflation_periods_ccal import (
    load_daily_series, build_inflation_based_periods, ERUPTION_START, ERUPTION_END,
    ROLLING_DAYS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'bpr_detided_depth_and_inflation_ccal.pdf')


def main():
    daily_depth, inflation, inflation_roll, ref_time, ref_depth = load_daily_series()
    periods = build_inflation_based_periods()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax_top.plot(daily_depth.index, daily_depth.values, color='#333333', lw=0.8)
    ax_top.axvspan(ERUPTION_START, ERUPTION_END, color='red', alpha=0.15, label='Eruption')
    ax_top.set_ylabel('De-tided seafloor depth (m)')
    ax_top.set_title('Central Caldera BOTPT (RS03CCAL-MJ03F-05-BOTPTA301): De-tided Seafloor '
                     'Depth and Inflation (2015-01-22 to 2026-05-15)',
                     fontsize=12, fontweight='bold')
    ax_top.legend(loc='lower left', fontsize=8)
    ax_top.grid(alpha=0.3)

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

    fig.tight_layout()
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')

    print('\n7-period scheme used for the post-eruption boundary markers:')
    for label, t0, t1 in periods:
        t0s = t0.date() if t0 is not None else '(start of record)'
        t1s = t1.date() if t1 is not None else '(present)'
        print(f'  {label!r:45s}  {t0s} -> {t1s}')


if __name__ == '__main__':
    main()
