#!/usr/bin/env python3
"""
atan2_uplift_vs_phi_axec2_axcc1_axas1_90day_nofit.py

Data-only counterpart of atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.py, per explicit user
request: same data pipeline (AXEC2/AXCC1 post-eruption pages, same wrap/flag conventions, same
90-day two-stage binning), but with NO fitted curve drawn on top -- scatter points only.

Per a later explicit user request, page 3 (originally AXAS1) is replaced with AXAS2 instead --
AXAS2 uses the same bpr_inflation_periods_ashes (ASHES vent field BOTPT) geodetic series as
AXAS1 (see axas2_uplift_phi_cosine_vs_time.py's docstring), so it is a drop-in swap of the
station's SWS catalog only. Because the AXAS1 gap-reconstruction in the original script depends
on inverting the fitted atan2 model, and this is a no-fit page anyway, page 3 here just plots
AXAS2's real data with no gap reconstruction.

Produces (3 pages, one per station: AXEC2, AXCC1, AXAS2):
    atan2_uplift_vs_phi_axec2_axcc1_axas1_90day_nofit.pdf

Run with:
    python3 atan2_uplift_vs_phi_axec2_axcc1_axas1_90day_nofit.py
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
from erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day import STATIONS as _STATIONS_BASE, UPLIFT_ROLLING_DAYS
from animate_arctan_stress_vectors import compute_optimal_wrap, compute_full_range_ticks
import bpr_inflation_periods_ashes as ashes_infl

HERE = os.path.dirname(os.path.abspath(__file__))

# Page 3 uses AXAS2 in place of AXAS1, per explicit user request -- same ASHES BOTPT geodetic
# series (ashes_infl), only the SWS station/catalog changes.
STATIONS = [s for s in _STATIONS_BASE if s['name'] != 'AXAS1'] + [
    dict(name='AXAS2', infl_module=ashes_infl, geodetic_label='ASHES vent field BOTPT',
        flag_unleveled=False, u0=0.0, wrap_axis=True),
]
OUT_PDF = os.path.join(HERE, 'atan2_uplift_vs_phi_axec2_axcc1_axas1_90day_nofit.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9


def make_page(station, roll, inflation_roll, baseline_phi):
    valid = roll.dropna(subset=['mean_phi']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    x = merged['inflation_m'].values
    y_raw = merged['mean_phi'].values

    wrap = compute_optimal_wrap(y_raw)
    y = (y_raw - wrap) % 180.0
    baseline_plot = (baseline_phi - wrap) % 180.0

    name = station['name']

    fig, ax = plt.subplots(figsize=(7, 6))
    if station['flag_unleveled']:
        is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
        ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2',
                  alpha=0.7, label='Post-eruption window')
        ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
                  marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')
    else:
        ax.scatter(x, y, s=18, color='#0072B2', alpha=0.7, label='Post-eruption window')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
    ax.set_ylabel(f'Avg fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)\n'
                  f'[axis wraps mod 180, optimal wrap={wrap:.1f}°]')
    y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(v) for v in tick_labels])
    ax.set_title(f'{station["name"]}: De-Tided Uplift vs. Fast Direction\n'
                 f'({station["geodetic_label"]}, post-eruption, {ROLL_WINDOW_DAYS}-day rolling window, '
                 f'no fit)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    figs = []
    for station in STATIONS:
        name = station['name']
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

        _dd, infl_raw, _infl_roll_30, _rt, _rd = station['infl_module'].load_daily_series()
        inflation_roll = infl_raw.rolling(f'{UPLIFT_ROLLING_DAYS}D', center=True,
                                          min_periods=UPLIFT_ROLLING_DAYS // 2).mean()

        figs.append(make_page(station, roll, inflation_roll, baseline_phi))
        print()

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
