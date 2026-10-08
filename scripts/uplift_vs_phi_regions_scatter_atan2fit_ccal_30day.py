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
import matplotlib.patheffects as pe

from rose_7period_regions_windowcheck_grade3 import _circular_mean_and_se_deg
from axec2_uplift_phi_cosine_vs_time import rolling_phi_stats_daily_then_roll, ERUPTION_START
from animate_arctan_stress_vectors import (
    compute_full_range_ticks, break_wrapped_line, ALPHA_FIXED_AZ_DEG,
)
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    fit_station_vector, ROLL_WINDOW_DAYS, ROLL_MIN_DAYS, UPLIFT_ROLLING_DAYS, GEODETIC_LABEL,
)
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from uplift_vs_phi_regions_scatter_ccal_30day import (
    PANELS, POINT_COLOR, STATION_COLORS, STATION_ALPHA, avg_label,
)
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


def is_degenerate(fit):
    """The AXCC1 failure mode documented on the site: beta collapses onto the fixed background
    azimuth alpha and A blows up, so the vector sum degenerates and the curve jumps. Collinear
    means parallel OR antiparallel (beta = alpha + 180 is the same failure, e.g. AXEC1), so the
    comparison is mod 180."""
    d = abs((fit['beta_az'] - ALPHA_FIXED_AZ_DEG + 90.0) % 180.0 - 90.0)
    return d < 1.0 or fit['A'] > 10.0


def to_panel(y_own, wrap_own, wrap_panel):
    """Map values from a fit's own wrapped y-space into the panel's wrapped y-space."""
    return (np.asarray(y_own) + wrap_own - wrap_panel) % 180.0


def style_axis(ax, wrap):
    y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(v) for v in tick_labels])
    ax.grid(alpha=0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def draw_series(ax, fit, wrap_panel, color, label, pt_alpha, se_alpha, curve_style, s=18,
                fit_name=None):
    """Scatter + light +-1 SE bars + atan2 curve for one fit, in the panel's y-space."""
    x = fit['x']
    y = to_panel(fit['y'], fit['wrap'], wrap_panel)
    se = fit['merged']['se_phi'].values
    ax.vlines(x, y - se, y + se, color=color, alpha=se_alpha, lw=1.5, zorder=1)
    ax.scatter(x, y, s=s, color=color, alpha=pt_alpha, linewidths=0, zorder=2, label=label)

    x_line = np.linspace(x.min(), x.max(), 400)
    y_own = fit['model'](x_line, fit['C1'], fit['C2'], fit['A'], fit['beta'])
    y_line = break_wrapped_line(to_panel(y_own, fit['wrap'], wrap_panel))
    fit_label = 'atan2 fit' + (' (degenerate)' if is_degenerate(fit) else '')
    ax.plot(x_line, y_line, zorder=3, label=f'{fit_name or label} {fit_label}', **curve_style)


def main():
    _dd, _infl_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4), sharex=True)
    for col, (title, subtitle, members) in enumerate(PANELS):
        ax_sta, ax_avg = axes[0, col], axes[1, col]
        print(f'{title}: {", ".join(members)}')
        fit = region_fit(title, members, inflation_roll)
        wrap = fit['wrap']   # one y-space per column, so the two rows line up
        # Average points are black now, so the black fit gets a white halo to stay readable.
        avg_style = dict(color='black', lw=2, linestyle='--',
                         path_effects=[pe.withStroke(linewidth=4.5, foreground='white')])

        # ---- Bottom row: regional average (unchanged content) -----------------------------
        fit_name = f'{title} average' if len(members) > 1 else members[0]
        draw_series(ax_avg, fit, wrap, POINT_COLOR, avg_label(title, members), 0.7, 0.12, avg_style,
                    fit_name=fit_name)
        ax_avg.set_title(f'{title} average' if len(members) > 1 else f'{title}',
                         fontsize=14, fontweight='bold', loc='left')
        ax_avg.text(1.0, 1.02, f'{subtitle}\nN = {len(fit["x"])}', transform=ax_avg.transAxes,
                    ha='right', va='bottom', fontsize=8.5, color='0.35')

        # ---- Top row: each member station with its own fit; Central replicated as-is -------
        if len(members) == 1:
            draw_series(ax_sta, fit, wrap, POINT_COLOR, avg_label(title, members), 0.7, 0.12,
                        avg_style, fit_name=fit_name)
            ax_sta.text(1.0, 1.02, f'{subtitle}\nN = {len(fit["x"])}', transform=ax_sta.transAxes,
                        ha='right', va='bottom', fontsize=8.5, color='0.35')
        else:
            n_txt = []
            for sta in members:
                fit_s = region_fit(sta, [sta], inflation_roll)
                color = STATION_COLORS[sta]
                draw_series(ax_sta, fit_s, wrap, color, sta, STATION_ALPHA, 0.08,
                            dict(color=color, lw=2.2, linestyle='-',
                                 path_effects=[pe.withStroke(linewidth=4, foreground='white')]),
                            s=12)
                n_txt.append(f'{sta} N = {len(fit_s["x"])}')
            ax_sta.text(1.0, 1.02, '\n'.join(n_txt), transform=ax_sta.transAxes,
                        ha='right', va='bottom', fontsize=8.5, color='0.35')
        ax_sta.set_title(f'{title}', fontsize=15, fontweight='bold', loc='left')

        for ax in (ax_sta, ax_avg):
            style_axis(ax, wrap)
            ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.4)
        ax_avg.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')

    ylab = f'Mean fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)'
    axes[0, 0].set_ylabel('Individual stations\n' + ylab)
    axes[1, 0].set_ylabel('Regional average\n' + ylab)

    fig.suptitle(f'Fast Direction vs. De-Tided Central Caldera Uplift, Post-Eruption, with atan2 Fits '
                 f'({GEODETIC_LABEL})', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
