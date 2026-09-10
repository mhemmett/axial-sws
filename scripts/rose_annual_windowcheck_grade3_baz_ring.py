#!/usr/bin/env python3
"""
rose_annual_windowcheck_grade3_baz_ring.py

Back-azimuth-ring variant of rose_annual_regions_windowcheck_grade3.py: same windowcheck data
source and grade-3 filter, same annual time-period breakdown (Before Eruption | During Eruption
| Post-eruption 2015 | 2016 | ... | 2026), but each cell's central rose is now surrounded by 6
back-azimuth (60 deg bin) sub-roses arranged in a ring -- see
rose_7period_windowcheck_grade3_baz_ring.py's docstring for the ring-layout rationale, not
repeated here; this script differs from that one only in its time-period definition (annual
instead of 7 eruption-relative periods), reusing _build_annual_periods from
rose_annual_regions_windowcheck_grade3.py.

Back-azimuth (station -> event, deg from N) is taken directly from the catalog's own
'back_azimuth' column, not recomputed from station/event geometry.

Produces: rose_annual_windowcheck_grade3_baz_ring.pdf (1 page: 6 station rows x 14 time-period
columns, central rose + 6 baz sub-roses per cell) -- a NEW file, does not touch any existing
output.

Run with:
    python3 rose_annual_windowcheck_grade3_baz_ring.py
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

from rose_7period_regions_windowcheck_grade3 import STATION_ORDER, GRADE
from rose_annual_regions_windowcheck_grade3 import _build_annual_periods
from rose_7period_windowcheck_grade3_baz_ring import load_station_events, make_baz_ring_page

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_annual_windowcheck_grade3_baz_ring.pdf')


def main():
    print(f'Loading windowcheck per-station events ({GRADE["label"]})...')
    station_dfs = {}
    for sta in STATION_ORDER:
        df = load_station_events(sta)
        station_dfs[sta] = df
        print(f'  {sta}: {len(df):,} events pass the filter')

    all_t = pd.concat([station_dfs[sta]['t'] for sta in STATION_ORDER], ignore_index=True)
    periods = _build_annual_periods(pd.DataFrame({'t': all_t}))

    n_total = sum(len(station_dfs[sta]) for sta in STATION_ORDER)
    title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (windowcheck) — fast direction rose by '
             f'station x annual period, with 6 back-azimuth sub-roses (N={n_total:,})\n'
             f'{GRADE["label"]}')
    fig = make_baz_ring_page(station_dfs, periods, title)

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
