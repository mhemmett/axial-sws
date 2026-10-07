#!/usr/bin/env python3
"""
uplift_vs_phi_regions_scatter_atan2fit_ccal_30day.py

uplift_vs_phi_regions_scatter_ccal_30day.py (West / Central / East side by side, data only) with
each region's best-fit atan2 vector-sum curve drawn on its panel. The curve is shown WITHOUT any
equation or parameter overlay; only the scatter, the curve, and a one-line legend.

The fit is NOT re-implemented: each panel calls fit_station_vector exactly as
atan2_uplift_vs_phi_regions_ccal_30day.py does (same pooled grade-3 events, same AXEC2
unleveled-window drop, same daily-then-30-day-roll phi, same CCAL uplift merge, same
auto-estimated turnover u0 and per-panel optimal wrap), so these curves are the same fits as on
that script's pages. Fitted parameters are printed to stdout for the record.

Central (AXCC1) caveat carried over from that figure: its fit is degenerate (beta collapses onto
the fixed background azimuth alpha, A ~ 42), so its curve has a vertical jump at the turnover and
should not be read as a result.

Produces: uplift_vs_phi_regions_scatter_atan2fit_ccal_30day.pdf / .png (1 page, 3 panels)

Run with:
    python3 uplift_vs_phi_regions_scatter_atan2fit_ccal_30day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rose_7period_regions_windowcheck_grade3 import _circular_mean_and_se_deg
from axec2_uplift_phi_cosine_vs_time import rolling_phi_stats_daily_then_roll, ERUPTION_START
from animate_arctan_stress_vectors import compute_full_range_ticks
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    fit_station_vector, ROLL_WINDOW_DAYS, ROLL_MIN_DAYS, UPLIFT_ROLLING_DAYS, GEODETIC_LABEL,
)
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from uplift_vs_phi_regions_scatter_ccal_30day import PANELS, POINT_COLOR
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'uplift_vs_phi_regions_scatter_atan2fit_ccal_30day')


def region_fit(title, members, inflation_roll):
    pool = load_region_pool(members)
    pre = pool[pool['t'] < ERUPTION_START]
    baseline_phi, _ = _circular_mean_and_se_deg(pre['phi_az'].values)
    roll = rolling_phi_stats_daily_then_roll(pool, baseline_phi,
                                             window_days=ROLL_WINDOW_DAYS,
                                             min_days=ROLL_MIN_DAYS)
    return fit_station_vector(title, roll, inflation_roll, baseline_phi)


def main():
    _dd, _infl_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True)
    for ax, (title, subtitle, members) in zip(axes, PANELS):
        print(f'{title}: {", ".join(members)}')
        fit = region_fit(title, members, inflation_roll)
        x, y, wrap = fit['x'], fit['y'], fit['wrap']
        se = fit['merged']['se_phi'].values

        # Light error region: each rolled point's +-se_phi (circular SE of the 30-day mean).
        ax.vlines(x, y - se, y + se, color=POINT_COLOR, alpha=0.12, lw=1.5, zorder=1,
                  label='±1 SE of rolling mean φ')
        ax.scatter(x, y, s=18, color=POINT_COLOR, alpha=0.7, linewidths=0, zorder=2, label='Data')

        x_line = np.linspace(x.min(), x.max(), 400)
        # Fit lives in the same wrapped y-space as the scatter, so plot it as make_page does.
        y_line = fit['model'](x_line, fit['C1'], fit['C2'], fit['A'], fit['beta'])
        ax.plot(x_line, y_line, color='black', lw=2, linestyle='--', label='atan2 best fit', zorder=3)

        y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([str(v) for v in tick_labels])

        ax.set_title(f'{title}', fontsize=15, fontweight='bold', loc='left')
        ax.text(1.0, 1.02, f'{subtitle}\nN = {len(x)}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, color='0.35')
        ax.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
        ax.grid(alpha=0.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel(f'Mean fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)')
    axes[-1].legend(loc='lower left', fontsize=9, framealpha=0.9)

    fig.suptitle(f'Fast Direction vs. De-Tided Central Caldera Uplift, Post-Eruption, with atan2 Fit '
                 f'({GEODETIC_LABEL})', fontsize=16, fontweight='bold', y=1.03)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
