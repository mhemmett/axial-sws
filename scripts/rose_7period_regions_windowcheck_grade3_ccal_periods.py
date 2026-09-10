#!/usr/bin/env python3
"""
rose_7period_regions_windowcheck_grade3_ccal_periods.py

Same as rose_7period_regions_windowcheck_grade3.py (windowcheck grade-3 filter, same figure
layout/fonts/region titles -- see that script's docstring for the full rationale), except the
7 time periods are the EQUAL-INFLATION-AMOUNT scheme (Before/During Eruption + 5 post-eruption
bins of equal Central Caldera BOTPT inflation, via bpr_inflation_periods_ccal.build_inflation_
based_periods()) instead of the equal-EVENT-COUNT scheme used there.

Produces: rose_7period_regions_windowcheck_grade3_ccal_periods.pdf (1 page)

Run with:
    python3 rose_7period_regions_windowcheck_grade3_ccal_periods.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    STATION_ORDER, GRADE, load_station_raw, apply_grade, make_rose_figure,
)
from bpr_inflation_periods_ccal import build_inflation_based_periods

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_7period_regions_windowcheck_grade3_ccal_periods.pdf')


def main():
    print(f'Loading windowcheck splitting results, filter: {GRADE["label"]} ...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
    dfs = {sta: apply_grade(raw[sta], GRADE) for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        print(f'  {sta}: {len(dfs[sta]):,}')

    time_periods = build_inflation_based_periods()
    print('\nEqual-inflation-amount 7-period scheme (Central Caldera BOTPT):')
    for label, t0, t1 in time_periods:
        t0s = t0.date() if t0 is not None else '(start of record)'
        t1s = t1.date() if t1 is not None else '(present)'
        print(f'  {label!r:35s}  {t0s} -> {t1s}')

    fig = make_rose_figure(
        dfs, time_periods,
        title='Shear-wave Splitting Fast Direction Across an Eruption Cycle\n'
              '(equal-inflation-amount periods, Central Caldera BOTPT)',
    )
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
