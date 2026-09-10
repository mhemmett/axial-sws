#!/usr/bin/env python3
"""
arctan_uplift_vs_phi_axec2_axcc1_axas1_90day.py

Arctangent counterpart of erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py, per explicit
user request: same data pipeline (AXEC2/AXCC1/AXAS1 post-eruption pages, same wrap/flag
conventions, same 90-day two-stage binning -- see that script's docstring), but the line of
fit is now

    phi(u_z) = phi_0 + (phi_1 - phi_0)/2 * arctan(alpha*u_z - beta*u_z0)

(a corrected form -- an earlier version of this script tried a bare, unscaled
arctan(A*u_z-B), which badly underfit every station, even landing at NEGATIVE r for AXEC2,
because arctan's raw output is bounded to (-pi/2, pi/2) while phi's real range sits almost
entirely outside that band; every other fit in this family, cube-root/sigmoid/erfc/erf,
includes an explicit amplitude scale to avoid exactly that). u_z0 is the SAME fixed constant
used throughout this script family (0.6 AXEC2, 1.0 AXCC1, 0.0 AXAS1); (phi_0, phi_1, alpha,
beta) are all free, fit the same way as
erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py's fit_erfc_physical (multi-start,
bounded to +-1000). arctan's argument is left in native radians (NOT converted to degrees) --
multiplying a degree-scaled amplitude by radian-valued arctan output (max +-pi/2 =~ +-1.571)
means the model's asymptotes as u_z -> +-infinity are phi_0 +- (phi_1-phi_0)/2*(pi/2), not
exactly phi_0/phi_1 the way the erf models' asymptotes land exactly on phi_0/phi_1 -- a
built-in ~1.571x mismatch from a clean-endpoint interpretation, inherent to using arctan's
non-unit saturation value in place of erf's clean +-1 saturation.

Produces (3 pages, one per station):
    arctan_uplift_vs_phi_axec2_axcc1_axas1_90day.pdf

Run with:
    python3 arctan_uplift_vs_phi_axec2_axcc1_axas1_90day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
    UNLEVEL_START, UNLEVEL_END,
)
from erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day import STATIONS, UPLIFT_ROLLING_DAYS

BOUND = 1000.0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'arctan_uplift_vs_phi_axec2_axcc1_axas1_90day.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9


def fit_arctan(x, y, u0):
    """phi(u) = phi0 + (phi1-phi0)/2*arctan(alpha*u - beta*u0), (phi0,phi1,alpha,beta) free
    and bounded to [-BOUND, BOUND] -- same fit structure as
    erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py's fit_erfc_physical, just with
    arctan (native radians) in place of erfc. Multi-start over alpha/beta sign+scale and
    which of (phi0,phi1) matches the low-x vs. high-x data, keeping the lowest-SSE fit.
    Returns (phi0, phi1, alpha, beta, r, model)."""
    def model(x, phi0, phi1, alpha, beta):
        return phi0 + (phi1 - phi0) / 2.0 * np.arctan(alpha * x - beta * u0)

    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    x_span = float(np.std(x) + 1e-6)

    best = None
    for phi0_0, phi1_0 in ((y_lo, y_hi), (y_hi, y_lo)):
        for alpha0 in (1 / x_span, -1 / x_span, 3 / x_span, -3 / x_span, 10 / x_span, -10 / x_span):
            for beta0 in (1.0, -1.0, 3.0, -3.0):
                p0 = [np.clip(v, -BOUND, BOUND) for v in (phi0_0, phi1_0, alpha0, beta0)]
                bounds = ([-BOUND] * 4, [BOUND] * 4)
                try:
                    popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
                except RuntimeError:
                    continue
                y_pred = model(x, *popt)
                sse = float(np.sum((y_pred - y) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, popt)
    phi0, phi1, alpha, beta = best[1]
    y_pred = model(x, phi0, phi1, alpha, beta)
    r = np.corrcoef(y_pred, y)[0, 1]
    return phi0, phi1, alpha, beta, r, model


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

    wrap = 90.0 if station['wrap_axis'] else 0.0
    y = (y_raw - wrap) % 180.0
    baseline_plot = (baseline_phi - wrap) % 180.0

    u0 = station['u0']
    phi0, phi1, alpha, beta, r, model = fit_arctan(x, y, u0)

    fig, ax = plt.subplots(figsize=(7, 6))
    if station['flag_unleveled']:
        is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
        ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2',
                  alpha=0.7, label='Post-eruption window')
        ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
                  marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')
    else:
        ax.scatter(x, y, s=18, color='#0072B2', alpha=0.7, label='Post-eruption window')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = model(x_line, phi0, phi1, alpha, beta)
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=r'$\phi(u_z)=\phi_0+\frac{\phi_1-\phi_0}{2}\arctan(\alpha u_z-\beta u_{z,0})$'
                 f'\n$\\phi_0$={phi0:.3f}°, $\\phi_1$={phi1:.3f}°, α={alpha:.3f}, β={beta:.3f} '
                 f'($u_{{z,0}}$={u0:.3f})\n(r = {r:.2f}, N = {len(x)})')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
    ylabel = f'Avg fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)'
    if station['wrap_axis']:
        ylabel += '\n[axis labels shifted 90° -- axial, mod 180]'
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 180)
    if station['wrap_axis']:
        ticks = np.arange(0, 181, 20)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int((t + wrap) % 180)}' for t in ticks])
    ax.set_title(f'{station["name"]}: De-Tided Uplift vs. Fast Direction\n'
                 f'({station["geodetic_label"]}, post-eruption, {ROLL_WINDOW_DAYS}-day rolling window)',
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
