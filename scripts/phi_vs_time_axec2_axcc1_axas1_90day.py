#!/usr/bin/env python3
"""
phi_vs_time_axec2_axcc1_axas1_90day.py

Companion to uplift_vs_phi_axec2_axcc1_axas1_90day.py: SAME y-axis quantity and SAME binning
(rolling_phi_stats_daily_then_roll's two-stage estimator -- per-calendar-day circular means,
then a 90-day centered rolling mean of those daily means, per station, post-eruption only),
but plotted against TIME on the x-axis instead of against de-tided uplift. This isolates what
the fast-direction trend looks like on its own clock, without pairing each point to a
geodetic value (so no merge_asof against the uplift series, and no cube-root fit -- there's no
second variable to fit against).

Produces (NEW file, 3 pages, one per station):
    phi_vs_time_axec2_axcc1_axas1_90day.pdf

Run with:
    python3 phi_vs_time_axec2_axcc1_axas1_90day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END, UNLEVEL_START, UNLEVEL_END,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'phi_vs_time_axec2_axcc1_axas1_90day.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9   # matches uplift_vs_phi_axec2_axcc1_axas1_90day.py's ROLL_MIN_DAYS (~window/10)

STATIONS = [
    dict(name='AXEC2', geodetic_label='Eastern Caldera BOTPT', flag_unleveled=True,
        wrap_axis=False),
    dict(name='AXCC1', geodetic_label='Central Caldera BOTPT', flag_unleveled=False,
        wrap_axis=True),
    dict(name='AXAS1', geodetic_label='ASHES vent field BOTPT', flag_unleveled=False,
        wrap_axis=True),
]


def make_page(station, roll, baseline_phi):
    valid = roll.dropna(subset=['mean_phi']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    t = valid['t'].values
    y_raw = valid['mean_phi'].values

    wrap = 90.0 if station['wrap_axis'] else 0.0
    y = (y_raw - wrap) % 180.0
    baseline_plot = (baseline_phi - wrap) % 180.0

    fig, ax = plt.subplots(figsize=(9, 5))
    if station['flag_unleveled']:
        is_flagged = (valid['t'] >= UNLEVEL_START) & (valid['t'] < UNLEVEL_END)
        ax.scatter(t[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2',
                  alpha=0.7, label='Post-eruption window')
        ax.scatter(t[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
                  marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')
    else:
        ax.scatter(t, y, s=18, color='#0072B2', alpha=0.7, label='Post-eruption window')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel('Time')
    ylabel = f'Avg fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)'
    if station['wrap_axis']:
        ylabel += '\n[axis labels shifted 90° -- axial, mod 180]'
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 180)
    if station['wrap_axis']:
        ticks = np.arange(0, 181, 20)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int((tk + wrap) % 180)}' for tk in ticks])
    ax.set_title(f'{station["name"]}: Fast Direction vs. Time\n'
                 f'({station["geodetic_label"]}, post-eruption, {ROLL_WINDOW_DAYS}-day rolling window)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
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

        figs.append(make_page(station, roll, baseline_phi))
        print()

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
