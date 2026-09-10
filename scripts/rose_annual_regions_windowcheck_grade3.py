#!/usr/bin/env python3
"""
rose_annual_regions_windowcheck_grade3.py

Same figure as rose_7period_regions_windowcheck_grade3.py (same data load, same grade-3 filter,
same font hierarchy/region titles/suptitle, same color scheme via _period_colors) but with an
ANNUAL time-period breakdown instead of the 7 eruption-relative periods, per explicit user
request:

    Pre-eruption 2015 | Syn-eruption 2015 | Post-eruption 2015 | 2016 | 2017 | ... | 2026

(14 columns total; 2026 is partial -- the dataset runs through mid-May 2026.) Boundaries follow
the same ERUPTION_START/ERUPTION_END convention used throughout this project (see
rose_axec2_windowcheck_vs_newdata_grade3_annual.py's build_annual_periods, extended here through
the current data range instead of stopping at 2021).

Reuses rose_7period_regions_windowcheck_grade3.py's data loading (load_station_raw, apply_grade,
GRADE, STATION_ORDER), figure-drawing (make_rose_figure, REGION_TITLES, font sizes), and color
scheme (_period_colors) directly -- only the time-period definition changes.

Produces: rose_annual_regions_windowcheck_grade3.pdf (1 page) -- a NEW file, does not touch
rose_7period_regions_windowcheck_grade3.pdf or any other existing output.

Run with:
    python3 rose_annual_regions_windowcheck_grade3.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_6stations_newdata_snr_grades import ERUPTION_START, ERUPTION_END
from rose_7period_regions_windowcheck_grade3 import (
    STATION_ORDER, GRADE, load_station_raw, apply_grade, make_rose_figure,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_annual_regions_windowcheck_grade3.pdf')


def _build_annual_periods(all_df):
    last_year = int(all_df['t'].max().year)

    periods = [
        ('Before Eruption', None, ERUPTION_START),
        ('During Eruption', ERUPTION_START, ERUPTION_END),
        ('Post-eruption 2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for year in range(2016, last_year + 1):
        t0 = pd.Timestamp(f'{year}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
        periods.append((f'{year}', t0, t1))

    return periods


def main():
    print(f'Loading windowcheck splitting results, filter: {GRADE["label"]} ...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
    dfs = {sta: apply_grade(raw[sta], GRADE) for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        print(f'  {sta}: {len(dfs[sta]):,}')

    all_combined = pd.concat(raw.values(), ignore_index=True)
    time_periods = _build_annual_periods(all_combined)

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
