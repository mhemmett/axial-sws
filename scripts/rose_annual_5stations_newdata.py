#!/usr/bin/env python3
"""
rose_annual_5stations_newdata.py

Annual-binned version of rose_7period_5stations_newdata.py: same 5-row (station) x N-col
(time period) grid, same data/QC/tiers/drawing code (imports STATION_ORDER, load_station_raw(),
apply_tier(), TIERS, make_rose_figure(), and every filter threshold directly from that script
so the two can never drift apart) -- but columns are ANNUAL bins instead of the 7 data-driven
eruption-relative periods, following this repo's established annual-binning convention
(rose_plots_baz.py's build_annual_periods):

    Pre-eruption 2015, Syn-eruption 2015, Post-eruption 2015 (remainder of 2015),
    then each calendar year 2016-2026 as its own column (14 columns total).

No back-azimuth ring version (plain central roses only, per the user's request).

5 pages, same cumulative QC tiers as the sibling:
    Page 1: raw (success & dt>0)
    Page 2: + quality >= 0.5
    Page 3: + dt < T_dom/2 (cycle-skip-risk cut, per-event dominant period)
    Page 4: + phi_error < 20 deg
    Page 5: + dt_error < 0.05 s
    Page 6: + dt_error < 0.05s, phi_error < 10 deg (stricter)

Produces: rose_annual_5stations_newdata.pdf  (6 pages)

Run with:
    python3 rose_annual_5stations_newdata.py
"""

import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_5stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, TIERS, make_rose_figure,
    ERUPTION_START, ERUPTION_END,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_annual_5stations_newdata.pdf')


def build_annual_periods():
    """Pre/syn/post-eruption 2015, then one column per calendar year 2016-2026
    (verbatim scheme from rose_plots_baz.py's build_annual_periods)."""
    pds = [
        ('Pre-eruption\n2015', None, ERUPTION_START),
        ('Syn-eruption\n2015', ERUPTION_START, ERUPTION_END),
        ('Post-eruption\n2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for yr in range(2016, 2027):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr + 1}-01-01', tz='UTC') if yr < 2026 else None
        pds.append((str(yr), t0, t1))
    return pds


def main():
    print('Loading new mfast max_dt=0.2s splitting results (AXCC1, AXEC1, AXEC3, AXAS1: '
          'complete; AXEC2 2015-2021: partial, in-progress re-run)...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
        print(f'  {sta}: {len(raw[sta]):,} baseline events (success & dt>0)')

    time_periods = build_annual_periods()

    with PdfPages(OUT_PDF) as pdf:
        for tier_idx in range(len(TIERS)):
            dfs = {sta: apply_tier(raw[sta], tier_idx) for sta in STATION_ORDER}
            for sta in STATION_ORDER:
                print(f'  tier {tier_idx} ({TIERS[tier_idx]}) {sta}: {len(dfs[sta]):,}')
            fig = make_rose_figure(
                dfs, time_periods,
                title=(f'AXAS1/AXCC1/AXEC1/AXEC2/AXEC3 fast direction rose, annual '
                       f'(new mfast max_dt=0.2s data) — {TIERS[tier_idx]}'))
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
