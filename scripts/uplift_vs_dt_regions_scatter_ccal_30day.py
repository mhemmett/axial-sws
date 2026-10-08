#!/usr/bin/env python3
"""
uplift_vs_dt_regions_scatter_ccal_30day.py

Delay-time companion to uplift_vs_phi_regions_scatter_atan2fit_ccal_30day.py, in the same 2 x 3
layout: West / Central / East columns; top row = each member station on its own (station
colour, 50% opacity), bottom row = the regional average (black). Central's only station AXCC1
appears in both rows. Y is the 30-day rolling mean delay time against de-tided Central Caldera
BOTPT uplift (30-day rolling mean), post-eruption, with a light +-1 SE bar on every point.

Same inputs as the phi figures: pooled Grade-3 events per region (load_region_pool, which drops
AXEC2's unleveled 2021-07..2022-09 window), the same daily-then-30-day-roll construction, and the
same 20-day merge tolerance onto the uplift series. The rolled delay time is the 30-day centred
mean of per-calendar-day mean dt (each day with data counts once, matching the phi estimator);
its SE is the std of those daily means / sqrt(days in window).

No model curve: the atan2 vector-sum fit is an angle model and does not apply to dt.
All six panels share one dt axis so stations can be compared directly.

Output: uplift_vs_dt_regions_scatter_ccal_30day.pdf / .png
"""

import os
import io
import contextlib
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from axec2_uplift_phi_cosine_vs_time import ERUPTION_END
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    ROLL_WINDOW_DAYS, ROLL_MIN_DAYS, UPLIFT_ROLLING_DAYS, GEODETIC_LABEL,
)
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from uplift_vs_phi_regions_scatter_ccal_30day import (
    PANELS, POINT_COLOR, STATION_COLORS, STATION_ALPHA,
)
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'uplift_vs_dt_regions_scatter_ccal_30day')


def rolled_dt(members, inflation_roll):
    """Daily mean dt -> 30-day centred rolling mean + SE, post-eruption, merged onto uplift.
    Returns (x uplift m, dt ms, se ms)."""
    with contextlib.redirect_stdout(io.StringIO()):
        pool = load_region_pool(members)
    d = pool[['t', 'dt']].copy()
    d['day'] = d['t'].dt.floor('D')
    daily = d.groupby('day')['dt'].mean() * 1000.0          # ms
    win = f'{ROLL_WINDOW_DAYS}D'
    m = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).mean()
    s = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).std()
    nd = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).count()
    roll = pd.DataFrame({'t': daily.index, 'dt_ms': m.values,
                         'se_ms': (s / np.sqrt(nd.clip(lower=1))).values}).dropna()
    roll = roll[roll['t'] >= ERUPTION_END]

    infl = inflation_roll.dropna().reset_index()
    infl.columns = ['t', 'u']
    merged = pd.merge_asof(roll.sort_values('t'), infl.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D')).dropna(subset=['u'])
    return merged['u'].values, merged['dt_ms'].values, merged['se_ms'].values


def draw(ax, x, y, se, color, label, alpha, se_alpha, s):
    ax.vlines(x, y - se, y + se, color=color, alpha=se_alpha, lw=1.5, zorder=1)
    ax.scatter(x, y, s=s, color=color, alpha=alpha, linewidths=0, zorder=2, label=label)


def style(ax):
    ax.grid(alpha=0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def main():
    _dd, _infl, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4), sharex=True, sharey=True)
    for col, (title, subtitle, members) in enumerate(PANELS):
        ax_sta, ax_avg = axes[0, col], axes[1, col]

        # Bottom row: regional average (black).
        x, y, se = rolled_dt(members, inflation_roll)
        avg_label = f'{title} average' if len(members) > 1 else f'{members[0]} (only station)'
        draw(ax_avg, x, y, se, POINT_COLOR, avg_label, 0.7, 0.12, 18)
        ax_avg.set_title(f'{title} average' if len(members) > 1 else title, fontsize=14,
                         fontweight='bold', loc='left')
        ax_avg.text(1.0, 1.02, f'{subtitle}\nN = {len(x)}', transform=ax_avg.transAxes,
                    ha='right', va='bottom', fontsize=8.5, color='0.35')
        print(f'{title} average: N={len(x)}  dt {np.nanmin(y):.0f}-{np.nanmax(y):.0f} ms')

        # Top row: each member station in its own colour.
        n_txt = []
        for sta in members:
            xs, ys, ses = rolled_dt([sta], inflation_roll)
            draw(ax_sta, xs, ys, ses, STATION_COLORS[sta], sta, STATION_ALPHA, 0.08, 12)
            n_txt.append(f'{sta} N = {len(xs)}')
            r = np.corrcoef(xs, ys)[0, 1]
            print(f'    {sta}: N={len(xs)}  r(dt, uplift)={r:+.2f}')
        ax_sta.set_title(title, fontsize=15, fontweight='bold', loc='left')
        ax_sta.text(1.0, 1.02, '\n'.join(n_txt), transform=ax_sta.transAxes, ha='right',
                    va='bottom', fontsize=8.5, color='0.35')

        for ax in (ax_sta, ax_avg):
            style(ax)
            ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.4)
        ax_avg.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')

    ylab = f'Mean delay time δt (ms, {ROLL_WINDOW_DAYS}-day rolling window)'
    axes[0, 0].set_ylabel('Individual stations\n' + ylab)
    axes[1, 0].set_ylabel('Regional average\n' + ylab)
    fig.suptitle(f'Delay Time vs. De-Tided Central Caldera Uplift, Post-Eruption '
                 f'({GEODETIC_LABEL})', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
