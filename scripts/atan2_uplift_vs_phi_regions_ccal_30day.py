#!/usr/bin/env python3
"""
atan2_uplift_vs_phi_regions_ccal_30day.py

Regional companion to atan2_uplift_vs_phi_sixstations_ccal_30day.py: instead of one page per
STATION, this produces one page per CALDERA REGION, each fitted against the SAME Central
Caldera BOTPT uplift series with the same atan2 vector-sum model.

    Page 1  Western Caldera   AXAS1 + AXAS2
    Page 2  Eastern Caldera   AXEC1 + AXEC2 + AXEC3
    Page 3  Central Caldera   AXCC1

Everything downstream of the grouping -- the grade-3 filter, the daily-mean-then-30-day-roll
phi estimator, the CCAL uplift series, the auto-estimated turnover u0, the atan2 vector-sum
fit, and the page layout -- is IMPORTED from the six-station script rather than reimplemented,
so the regional and per-station figures cannot drift apart methodologically.

HOW A REGION'S ANGLE IS AVERAGED
    Member stations' grade-3 measurements are POOLED into a single event table, which is then
    fed to the same rolling_phi_stats_daily_then_roll estimator the per-station figures use.
    That estimator collapses events to per-CALENDAR-DAY circular means first and only then
    applies the 30-day rolling window, so every day with data counts once for the region
    regardless of how many member stations happened to report that day. Within a single day,
    stations contribute in proportion to their measurement count.

    Pooling is reasonable here because the member stations are well balanced under grade 3
    (west: AXAS1 ~8.9k, AXAS2 ~6.5k; east: AXEC1 ~24.7k, AXEC2 ~26.0k, AXEC3 ~24.8k), so no
    single station dominates its region's mean. If that balance changes, revisit this -- the
    alternative is to average the per-station rolling means instead of pooling raw events.

AXEC2's UNLEVELED WINDOW IS EXCLUDED FROM THE EASTERN POOL
    AXEC2's phi drifts ~50 deg between roughly 2021-07 and 2022-09 and then snaps back in a
    single day (2022-10-30), which is instrumental -- progressive sensor tilt followed by a
    releveling -- not tectonic. The per-station figure can afford to merely FLAG those points
    in green because the reader still sees AXEC2's own undrifted data around them. A regional
    mean cannot: the drifted points would silently pull the eastern average. They are dropped
    from the pool here, and the page title says so.

    UNLEVEL_START / UNLEVEL_END come from axec2_uplift_phi_cosine_vs_time so this window stays
    defined in exactly one place.

Produces: atan2_uplift_vs_phi_regions_ccal_30day.pdf (3 pages)

Run with:
    python3 atan2_uplift_vs_phi_regions_ccal_30day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START,
    UNLEVEL_START, UNLEVEL_END,
)
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    fit_station_vector, make_page, ROLL_WINDOW_DAYS, ROLL_MIN_DAYS,
)
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'atan2_uplift_vs_phi_regions_ccal_30day.pdf')

# (label, member stations). Labels feed make_page's title directly.
REGIONS = [
    ('Western Caldera (AXAS1 + AXAS2)', ['AXAS1', 'AXAS2']),
    ('Eastern Caldera (AXEC1 + AXEC2 + AXEC3, AXEC2 unleveled window excluded)',
     ['AXEC1', 'AXEC2', 'AXEC3']),
    ('Central Caldera (AXCC1)', ['AXCC1']),
]

# Stations whose known-instrumental window is dropped rather than flagged (see docstring).
DROP_UNLEVELED = {'AXEC2'}


def load_region_pool(members):
    """Grade-3 filter each member station, drop known-instrumental windows, and pool the
    surviving events into one table for the regional phi estimator."""
    frames = []
    for sta in members:
        raw = load_station_raw(sta)
        df = apply_grade(raw, GRADE)
        n_all = len(df)

        if sta in DROP_UNLEVELED:
            bad = (df['t'] >= UNLEVEL_START) & (df['t'] < UNLEVEL_END)
            df = df[~bad]
            print(f'    {sta}: {n_all:,} pass grade 3; dropped {int(bad.sum()):,} in the '
                  f'unleveled window -> {len(df):,} pooled')
        else:
            print(f'    {sta}: {n_all:,} pass grade 3 -> {len(df):,} pooled')

        df = df.copy()
        df['station'] = sta
        frames.append(df)

    pool = pd.concat(frames, ignore_index=True).sort_values('t').reset_index(drop=True)
    return pool


def main():
    _dd, _infl_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()

    figs = []
    for label, members in REGIONS:
        print(f'{label}')
        pool = load_region_pool(members)
        print(f'    pooled total: {len(pool):,} events from {len(members)} station(s)')

        pre = pool[pool['t'] < ERUPTION_START]
        baseline_phi, baseline_se = _circular_mean_and_se_deg(pre['phi_az'].values)
        print(f'    pre-eruption baseline: N={len(pre):,}, mean phi={baseline_phi:.1f}'
              f'+/-{baseline_se:.1f} deg')

        roll = rolling_phi_stats_daily_then_roll(pool, baseline_phi,
                                                 window_days=ROLL_WINDOW_DAYS,
                                                 min_days=ROLL_MIN_DAYS)
        n_valid = roll['mean_phi'].notna().sum()
        print(f'    {n_valid}/{len(roll)} rolled points have >= {ROLL_MIN_DAYS} days of data')

        fit = fit_station_vector(label, roll, inflation_roll, baseline_phi)
        fit['baseline_phi'] = baseline_phi
        # Unleveled points were removed from the pool, so there is nothing left to flag.
        figs.append(make_page(fit, flag_unleveled=False))
        print()

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
