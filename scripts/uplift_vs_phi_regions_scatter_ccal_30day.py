#!/usr/bin/env python3
"""
uplift_vs_phi_regions_scatter_ccal_30day.py

Data-only, three-panel version of atan2_uplift_vs_phi_regions_ccal_30day.py: Western, Central,
and Eastern Caldera mean fast direction vs. de-tided Central Caldera BOTPT uplift, side by side,
with NO model overlay (no atan2 vector-sum fit, no turnover line, no baseline line) -- just the
scatter.

Everything upstream of the plot is IMPORTED unchanged from the regional script and its
dependencies, so the points are identical to that figure's:
    - region membership + pooling, incl. dropping AXEC2's unleveled window (load_region_pool)
    - grade-3 filter, daily-circular-mean-then-30-day-roll phi estimator
    - post-eruption only, merge_asof onto the 30-day CCAL uplift series (20-day tolerance)
    - per-panel optimal wrap so the cluster never splits across the 0/180 edge, with the
      full 0-180 tick labelling (compute_optimal_wrap / compute_full_range_ticks)

Produces: uplift_vs_phi_regions_scatter_ccal_30day.pdf / .png (1 page, 3 panels)

Run with:
    python3 uplift_vs_phi_regions_scatter_ccal_30day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rose_7period_regions_windowcheck_grade3 import _circular_mean_and_se_deg
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
)
from animate_arctan_stress_vectors import compute_optimal_wrap, compute_full_range_ticks
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    ROLL_WINDOW_DAYS, ROLL_MIN_DAYS, UPLIFT_ROLLING_DAYS, GEODETIC_LABEL,
)
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'uplift_vs_phi_regions_scatter_ccal_30day')

# (panel title, member-station subtitle, members) -- west to east, left to right.
PANELS = [
    ('West', 'AXAS1 + AXAS2', ['AXAS1', 'AXAS2']),
    ('Central', 'AXCC1', ['AXCC1']),
    ('East', 'AXEC1 + AXEC2 + AXEC3\n(AXEC2 unleveled window excluded)',
     ['AXEC1', 'AXEC2', 'AXEC3']),
]

POINT_COLOR = '#0072B2'   # same series color as the atan2 regional/per-station figures

# Individual member stations, drawn UNDER the regional average at 50% opacity. Okabe-Ito hues,
# one fixed colour per station (never reused), validated per panel (average + members, all
# pairs) with the dataviz validate_palette.js: West and East sets both pass CVD/normal-vision.
STATION_COLORS = {
    'AXAS1': '#E69F00',   # orange
    'AXAS2': '#CC79A7',   # reddish purple
    'AXEC1': '#009E73',   # bluish green
    'AXEC2': '#D55E00',   # vermillion
    'AXEC3': '#56B4E9',   # sky blue
}
STATION_ALPHA = 0.5


def rolled_series(members, inflation_roll):
    """Pool -> roll -> post-eruption -> merge with uplift, as fit_station_vector does, minus
    the fit. Returns (x uplift m, raw mean phi deg mod 180, se_phi deg). Works for a single
    station too (members=[sta]); AXEC2's unleveled window is dropped either way."""
    pool = load_region_pool(members)
    pre = pool[pool['t'] < ERUPTION_START]
    baseline_phi, _ = _circular_mean_and_se_deg(pre['phi_az'].values)

    roll = rolling_phi_stats_daily_then_roll(pool, baseline_phi,
                                             window_days=ROLL_WINDOW_DAYS,
                                             min_days=ROLL_MIN_DAYS)
    valid = roll.dropna(subset=['mean_phi'])
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    return merged['inflation_m'].values, merged['mean_phi'].values, merged['se_phi'].values


def region_points(members, inflation_roll):
    """Regional average in the panel's wrapped y-space. Returns (x, y, se, wrap)."""
    x, y_raw, se = rolled_series(members, inflation_roll)
    wrap = compute_optimal_wrap(y_raw)
    return x, (y_raw - wrap) % 180.0, se, wrap


def draw_member_stations(ax, members, inflation_roll, to_y):
    """Scatter each member station's own rolled series at STATION_ALPHA, beneath the average.
    to_y maps raw phi (deg) into the panel's y-space. No-op for single-station panels, where
    the station IS the average."""
    if len(members) < 2:
        return
    for sta in members:
        xs, phis, _ = rolled_series([sta], inflation_roll)
        ax.scatter(xs, to_y(phis), s=12, color=STATION_COLORS[sta], alpha=STATION_ALPHA,
                   linewidths=0, zorder=0.5, label=sta)


def avg_label(title, members):
    return f'{title} average' if len(members) > 1 else f'{members[0]} (only station)'


def main():
    _dd, _infl_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True)
    for ax, (title, subtitle, members) in zip(axes, PANELS):
        print(f'{title}: {", ".join(members)}')
        x, y, se, wrap = region_points(members, inflation_roll)
        print(f'    {len(x)} post-eruption rolled points, wrap={wrap:.1f} deg')

        draw_member_stations(ax, members, inflation_roll, lambda p: (p - wrap) % 180.0)

        # Light error region: each rolled point's +-se_phi (circular SE of the 30-day mean).
        ax.vlines(x, y - se, y + se, color=POINT_COLOR, alpha=0.12, lw=1.5, zorder=1,
                  label='±1 SE (average)')
        ax.scatter(x, y, s=18, color=POINT_COLOR, alpha=0.7, linewidths=0, zorder=2,
                   label=avg_label(title, members))

        y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([str(v) for v in tick_labels])

        ax.legend(loc='best', fontsize=8.5, framealpha=0.9, markerscale=1.4)
        ax.set_title(f'{title}', fontsize=15, fontweight='bold', loc='left')
        ax.text(1.0, 1.02, f'{subtitle}\nN = {len(x)}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, color='0.35')
        ax.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
        ax.grid(alpha=0.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel(f'Mean fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)')

    fig.suptitle(f'Fast Direction vs. De-Tided Central Caldera Uplift, Post-Eruption '
                 f'({GEODETIC_LABEL})', fontsize=16, fontweight='bold', y=1.03)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
