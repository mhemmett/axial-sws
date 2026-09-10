#!/usr/bin/env python3
"""
atan2_uplift_vs_phi_sixstations_ccal_90day_full_nofit.py

Six-station, no-fit, two-panel counterpart of the earlier single-panel version, per explicit
user request:
  - One page per station, in order: AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3 (all six).
  - ALL six pages use the SAME geodetic series -- the Central Caldera BOTPT
    (bpr_inflation_periods_ccal), not each station's own local BOTPT -- so every page shows
    fast direction against the same inflation reference.
  - Same 90-day two-stage (daily-then-rolled) phi binning as the earlier scripts.
  - FULL time period, split into two side-by-side panels sharing the y-axis (phi), pivoting on
    the eruption at x=0 in the middle:
      * LEFT panel: pre-eruption -> eruption. x-axis runs from positive inflation (left) down
        to zero at the eruption (right) -- i.e. the pre-eruption deflation-toward-the-eruption
        story, x-axis REVERSED so it reads left-to-right as time-forward-into-the-eruption.
      * RIGHT panel: eruption -> post-eruption re-inflation. x-axis runs from zero (left, at
        the eruption) up to positive inflation (right) -- the post-eruption re-inflation story,
        normal (non-reversed) x-axis.
    The two panels are placed adjacent with no gap, so x=0 (the eruption) sits in the middle of
    the page and the two inflation histories visually "meet" there.
  - No fitted curve -- scatter points only.

Produces (6 pages, one per station):
    atan2_uplift_vs_phi_sixstations_ccal_90day_full_nofit.pdf

Run with:
    python3 atan2_uplift_vs_phi_sixstations_ccal_90day_full_nofit.py
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

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
    UNLEVEL_START, UNLEVEL_END,
)
from animate_arctan_stress_vectors import compute_optimal_wrap, compute_full_range_ticks
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'atan2_uplift_vs_phi_sixstations_ccal_90day_full_nofit.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9
UPLIFT_ROLLING_DAYS = 90  # per explicit user request: 90-day rolling window on both axes

STATION_ORDER = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
UNLEVELED_STATIONS = {'AXEC2'}


def make_page(name, roll, inflation_roll, baseline_phi):
    valid = roll.dropna(subset=['mean_phi']).copy()

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m']).reset_index(drop=True)

    x = merged['inflation_m'].values
    y_raw = merged['mean_phi'].values

    wrap = compute_optimal_wrap(y_raw)
    y = (y_raw - wrap) % 180.0
    baseline_plot = (baseline_phi - wrap) % 180.0

    flag_unleveled = name in UNLEVELED_STATIONS
    pre_mask = (merged['t'] < ERUPTION_START).values
    syn_mask = ((merged['t'] >= ERUPTION_START) & (merged['t'] < ERUPTION_END)).values
    post_mask = (merged['t'] >= ERUPTION_END).values
    unlevel_mask = ((merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)).values if flag_unleveled \
        else np.zeros(len(merged), dtype=bool)
    post_mask = post_mask & ~unlevel_mask

    left_mask = pre_mask | syn_mask
    right_mask = post_mask | unlevel_mask

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 6), sharey=True,
                                    gridspec_kw={'wspace': 0.0})

    # LEFT panel: pre-eruption -> eruption, x reversed (positive inflation on the left,
    # zero/eruption on the right, so time reads left-to-right into the eruption).
    axL.scatter(x[pre_mask], y[pre_mask], s=18, color='#999999', alpha=0.7,
               label='Pre-eruption')
    axL.scatter(x[syn_mask], y[syn_mask], s=22, color='#D55E00', alpha=0.85, marker='s',
               label='Eruption transition')
    axL.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
               label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    axL.set_xlabel(f'Central Caldera de-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)\n'
                    f'(pre-eruption -> eruption)')
    axL.set_ylabel(f'Avg fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)\n'
                   f'[axis wraps mod 180, optimal wrap={wrap:.1f}°]')
    axL.grid(alpha=0.3)
    axL.legend(loc='lower left', fontsize=7.5, framealpha=0.9)

    # RIGHT panel: eruption -> post-eruption re-inflation, x normal (zero/eruption on the
    # left, increasing re-inflation to the right).
    axR.scatter(x[post_mask], y[post_mask], s=18, color='#0072B2', alpha=0.7,
               label='Post-eruption')
    if flag_unleveled:
        axR.scatter(x[unlevel_mask], y[unlevel_mask], s=22, color='#2ca02c', alpha=0.9,
                   marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')
    axR.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':')
    axR.set_xlabel(f'Central Caldera de-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)\n'
                    f'(eruption -> re-inflation)')
    axR.grid(alpha=0.3)
    axR.legend(loc='lower right', fontsize=7.5, framealpha=0.9)

    # Shared x-scaling: both panels span the same |inflation| extent, LEFT panel reversed so
    # both panels' zero (the eruption) sits in the page's middle.
    if left_mask.any():
        left_max = float(np.nanmax(x[left_mask]))
    else:
        left_max = 0.0
    if right_mask.any():
        right_max = float(np.nanmax(x[right_mask]))
    else:
        right_max = 0.0
    x_max = max(left_max, right_max, 1e-6) * 1.05
    axL.set_xlim(x_max, 0.0)   # reversed: positive on the left, zero on the right
    axR.set_xlim(0.0, x_max)   # normal: zero on the left, positive on the right

    y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
    axL.set_ylim(y_lo, y_hi)
    axL.set_yticks(tick_positions)
    axL.set_yticklabels([str(v) for v in tick_labels])

    axR.tick_params(labelleft=False)
    axR.spines['left'].set_linewidth(1.2)

    fig.suptitle(f'{name}: De-Tided Central Caldera Uplift vs. Fast Direction\n'
                 f'(full time period, {ROLL_WINDOW_DAYS}-day rolling window, no fit)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def main():
    _dd, infl_raw, _infl_roll_30, _rt, _rd = ccal_infl.load_daily_series()
    inflation_roll = infl_raw.rolling(f'{UPLIFT_ROLLING_DAYS}D', center=True,
                                      min_periods=UPLIFT_ROLLING_DAYS // 2).mean()

    figs = []
    for name in STATION_ORDER:
        print(f'Loading {name} windowcheck grade-3 events ({GRADE["label"]})...')
        raw = load_station_raw(name)
        df = apply_grade(raw, GRADE)
        print(f'  {name}: {len(df):,} events pass the filter')

        pre = df[df['t'] < ERUPTION_START]
        baseline_phi, baseline_se = _circular_mean_and_se_deg(pre['phi_az'].values)
        print(f'  Pre-eruption baseline: N={len(pre):,}, mean phi={baseline_phi:.1f}'
              f'+/-{baseline_se:.1f} deg')

        roll = rolling_phi_stats_daily_then_roll(df, baseline_phi, window_days=ROLL_WINDOW_DAYS,
                                                 min_days=ROLL_MIN_DAYS)
        n_valid = roll['mean_phi'].notna().sum()
        print(f'  {n_valid}/{len(roll)} daily-then-{ROLL_WINDOW_DAYS}-day-rolled points have '
              f'>= {ROLL_MIN_DAYS} days of data')

        figs.append(make_page(name, roll, inflation_roll, baseline_phi))
        print()

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
