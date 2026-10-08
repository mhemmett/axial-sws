#!/usr/bin/env python3
"""
uplift_vs_dphi_regions_scatter_ccal_30day.py

CHANGE-IN-ANGLE variant of uplift_vs_phi_regions_scatter_ccal_30day.py: the y-axis is the
change in mean fast direction since the start of re-inflation, dphi = phi - phi_ref, wrapped to
[-90, 90] deg (axial data), positive = clockwise (azimuth increasing).

Start of re-inflation = POST_ERUPTION_START (2015-05-02), the post-eruption uplift minimum, which
is also where every post-eruption figure starts and the zero of the uplift axis. phi_ref is each
region's circular mean of its pooled grade-3 EVENTS in
[POST_ERUPTION_START, POST_ERUPTION_START + REF_WINDOW_DAYS), REF_WINDOW_DAYS = the 30-day phi rolling window.

SHARED, CYCLED Y-AXIS: dphi is axial (mod 180), so any 180-deg window [c-90, c+90] is a complete
view. One window is shared by all three panels; its centre c is chosen automatically
(choose_shared_center) to minimise the worst panel's fraction of points within EDGE_MARGIN_DEG of
either edge, so every region's cloud sits away from the wrap. West and Central each span the
full 180 deg (no gap > ~6 deg), so a fully edge-free window does not exist; the chosen one is the
best compromise. Tick values outside +-90 are the equivalent orientation (e.g. -120 == +60). Because dphi is wrapped to +-90, all three
panels share one y-axis (no per-panel wrap needed).

Original docstring of the parent script follows.

Data-only, three-panel version of atan2_uplift_vs_phi_regions_ccal_30day.py: Western, Central,
and Eastern Caldera mean fast direction vs. de-tided Central Caldera BOTPT uplift, side by side,
with NO model overlay (no atan2 vector-sum fit, no turnover line, no baseline line) -- just the
scatter.

Everything upstream of the plot is IMPORTED unchanged from the regional script and its
dependencies, so the points are identical to that figure's:
    - region membership + pooling, incl. dropping AXEC2's unleveled window (load_region_pool)
    - grade-3 filter, daily-circular-mean-then-30-day-roll phi estimator
    - post-eruption only, merge_asof onto the 30-day CCAL uplift series (20-day tolerance)

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
from bpr_inflation_periods_ccal import POST_ERUPTION_START

from rose_7period_regions_windowcheck_grade3 import _circular_mean_and_se_deg
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
)
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    ROLL_WINDOW_DAYS, ROLL_MIN_DAYS, UPLIFT_ROLLING_DAYS, GEODETIC_LABEL,
)
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from uplift_vs_phi_regions_scatter_ccal_30day import draw_member_stations, avg_label, POINT_COLOR
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'uplift_vs_dphi_regions_scatter_ccal_30day')
REF_WINDOW_DAYS = ROLL_WINDOW_DAYS
EDGE_MARGIN_DEG = 15.0

# (panel title, member-station subtitle, members) -- west to east, left to right.
PANELS = [
    ('West', 'AXAS1 + AXAS2', ['AXAS1', 'AXAS2']),
    ('Central', 'AXCC1', ['AXCC1']),
    ('East', 'AXEC1 + AXEC2 + AXEC3\n(AXEC2 unleveled window excluded)',
     ['AXEC1', 'AXEC2', 'AXEC3']),
]



def region_points(members, inflation_roll, ref_start):
    """Pool -> roll -> post-eruption -> merge with uplift, as fit_station_vector does, minus
    the fit. Returns (x uplift m, dphi deg in [-90, 90], phi_ref, phi_ref_se, n_ref)."""
    pool = load_region_pool(members)
    in_ref = (pool['t'] >= ref_start) & (pool['t'] < ref_start + pd.Timedelta(days=REF_WINDOW_DAYS))
    phi_ref, phi_ref_se = _circular_mean_and_se_deg(pool.loc[in_ref, 'phi_az'].values)
    n_ref = int(in_ref.sum())
    pre = pool[pool['t'] < ERUPTION_START]
    baseline_phi, _ = _circular_mean_and_se_deg(pre['phi_az'].values)

    roll = rolling_phi_stats_daily_then_roll(pool, baseline_phi,
                                             window_days=ROLL_WINDOW_DAYS,
                                             min_days=ROLL_MIN_DAYS)
    valid = roll.dropna(subset=['mean_phi'])
    valid = valid[valid['t'] >= POST_ERUPTION_START]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    x = merged['inflation_m'].values
    dphi = (merged['mean_phi'].values - phi_ref + 90.0) % 180.0 - 90.0
    return x, dphi, merged['se_phi'].values, phi_ref, phi_ref_se, n_ref


def choose_shared_center(ys, margin=EDGE_MARGIN_DEG):
    """Centre c (1-deg grid, -90..89) of the shared 180-deg window minimising the worst panel's
    fraction of points within `margin` of the window edges."""
    best_c, best_score = 0.0, np.inf
    for c in np.arange(-90.0, 90.0, 1.0):
        score = max(np.mean(((y - (c - 90.0)) % 180.0 < margin) |
                            ((y - (c - 90.0)) % 180.0 > 180.0 - margin)) for y in ys)
        if score < best_score:
            best_c, best_score = c, score
    return best_c, best_score


def main():
    _dd, _infl_raw, inflation_roll, ref_time, _rd = ccal_infl.load_daily_series()
    print(f'Uplift zero point (ref_time): {ref_time.date()}; dphi reference window starts at '
          f'POST_ERUPTION_START = {POST_ERUPTION_START.date()}')

    data = []
    for title, subtitle, members in PANELS:
        print(f'{title}: {", ".join(members)}')
        x, y, se, phi_ref, phi_ref_se, n_ref = region_points(members, inflation_roll, POST_ERUPTION_START)
        print(f'    {len(x)} post-eruption rolled points; phi_ref={phi_ref:.1f}+/-{phi_ref_se:.1f}'
              f' deg from {n_ref} events')
        data.append((title, subtitle, members, x, y, se, phi_ref, phi_ref_se, n_ref))

    center, edge_frac = choose_shared_center([y for (_t, _s, _m, _x, y, *_rest) in data])
    y_lo = center - 90.0
    print(f'Shared y window: [{y_lo:.0f}, {y_lo + 180:.0f}] deg (centre {center:.0f}); worst panel '
          f'has {edge_frac*100:.1f}% of points within {EDGE_MARGIN_DEG:.0f} deg of an edge')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    for ax, (title, subtitle, members, x, y, se, phi_ref, phi_ref_se, n_ref) in zip(axes, data):
        y = (y - y_lo) % 180.0 + y_lo

        # Member stations relative to the REGION's phi_ref (not their own), so inter-station
        # offsets show as real orientation differences; same shared cycled window.
        draw_member_stations(
            ax, members, inflation_roll,
            lambda p, ref=phi_ref: (((p - ref + 90.0) % 180.0 - 90.0) - y_lo) % 180.0 + y_lo)

        ax.axhline(0, color='0.5', lw=0.8, zorder=0)
        # Light error region: each rolled point's +-se_phi (circular SE of the 30-day mean).
        ax.vlines(x, y - se, y + se, color=POINT_COLOR, alpha=0.12, lw=1.5, zorder=1,
                  label='±1 SE (average)')
        ax.scatter(x, y, s=18, color=POINT_COLOR, alpha=0.7, linewidths=0, zorder=2,
                   label=avg_label(title, members))
        ax.set_ylim(y_lo, y_lo + 180.0)
        ax.set_yticks(np.arange(np.ceil(y_lo / 30.0) * 30.0, y_lo + 180.01, 30.0))

        ax.legend(loc='best', fontsize=8.5, framealpha=0.9, markerscale=1.4)
        ax.set_title(f'{title}', fontsize=15, fontweight='bold', loc='left')
        ax.text(1.0, 1.02, f'{subtitle}\n$\\phi_{{ref}}$ = {phi_ref:.0f}° ± {phi_ref_se:.0f}° (n = {n_ref}), N = {len(x)}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, color='0.35')
        ax.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
        ax.grid(alpha=0.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel(f'$\\Delta\\phi$ since start of re-inflation (deg, clockwise +)\n'
                       f'({ROLL_WINDOW_DAYS}-day rolling window; axis cycles mod 180°)')

    fig.suptitle(f'Change in Fast Direction Since Start of Re-Inflation vs. De-Tided Central Caldera Uplift '
                 f'({GEODETIC_LABEL})', fontsize=16, fontweight='bold', y=1.03)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
