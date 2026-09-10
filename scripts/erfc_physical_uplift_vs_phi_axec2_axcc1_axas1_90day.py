#!/usr/bin/env python3
"""
erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py

Physically-parameterized erfc counterpart of erfc_uplift_vs_phi_axec2_axcc1_axas1_90day.py,
per explicit user request. Same data pipeline (AXEC2/AXCC1/AXAS1 post-eruption pages, same
wrap/flag conventions, same 90-day two-stage binning -- daily-then-90-day-rolled phi and a
locally-applied 90-day rolling mean of the raw daily uplift series -- see
uplift_vs_phi_axec2_axcc1_axas1_90day.py's docstring for the two-stage-binning rationale), but
the line of fit is now written in the user's explicit physical form

    phi(u_z) = phi_0 + (phi_1 - phi_0)/2 * erfc(alpha*u_z - beta*u_z0)

where phi_0 is the initial (pre-turnover) fast-direction orientation, phi_1 is the final
(post-turnover) orientation, u_z0 is the FIXED de-tided-uplift inflection point where the
stress-state turnover occurs (0.6 AXEC2, 1.0 AXCC1, 0.0 AXAS1 -- the same values used as the
sigmoid/erfc "C" shift in the earlier fit versions), and alpha, beta are free. (phi_0, phi_1,
alpha, beta) are fit via scipy.optimize.curve_fit with a defensive +-1000 bound (same rationale
as erfc_uplift_vs_phi_axec2_axcc1_axas1_90day.py's BOUND) and a small multi-start grid over
sign/scale of alpha, beta, and which of (phi_0, phi_1) matches the low-x vs. high-x data, since
a single naive initial guess is prone to bad local minima here (same class of issue as that
script's fit_erfc_at_zero). For AXAS1 (u_z0 = 0.0), beta*u_z0 = 0 identically regardless of
beta's value -- beta is structurally unidentifiable for this station (the turnover is pinned
at u_z=0 through alpha alone) -- but the fit is otherwise well-posed since alpha alone can
still shift/scale the transition.

The 3 post-eruption station pages (AXEC2, AXCC1, AXAS1) -- no AXEC2 eruption-transition page.

Page 4 (added per a later explicit user request): AXAS1 2015-2017 gap reconstruction, same
idea as sigmoid_uplift_vs_phi_axec2_axcc1_axas1.py's page 5 (fit_logistic_at_zero +
invert_logistic_at_zero) but using THIS script's physical erfc fit and its analytic inverse
(invert_erfc_physical, via scipy.special.erfcinv) instead of a logistic. AXAS1's real BOTPT
uplift record only starts in 2017-08; before that (back to ERUPTION_END), the windowcheck
SWS catalog still has fast-direction measurements with no matching real u_z. Page 4 inverts
the page-3 phi(u_z) fit (fit on the REAL post-2017 (u_z, phi) pairs only) to back out a
predicted u_z for those real pre-2017 phi observations. NaN where a gap-window phi falls
outside the fitted curve's achievable range (erfcinv's domain is (0,2)).

Produces (NEW file, 4 pages):
    erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.pdf

Run with:
    python3 erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcinv
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
    UNLEVEL_START, UNLEVEL_END,
)
import bpr_inflation_periods as ecal_infl
import bpr_inflation_periods_ccal as ccal_infl
import bpr_inflation_periods_ashes as ashes_infl

# Same defensive bound as erfc_uplift_vs_phi_axec2_axcc1_axas1_90day.py's BOUND.
BOUND = 1000.0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9

UPLIFT_ROLLING_DAYS = 90

STATIONS = [
    dict(name='AXEC2', infl_module=ecal_infl, geodetic_label='Eastern Caldera BOTPT',
        flag_unleveled=True, u0=0.6, wrap_axis=False),
    dict(name='AXCC1', infl_module=ccal_infl, geodetic_label='Central Caldera BOTPT',
        flag_unleveled=False, u0=1.0, wrap_axis=True),
    dict(name='AXAS1', infl_module=ashes_infl, geodetic_label='ASHES vent field BOTPT',
        flag_unleveled=False, u0=0.0, wrap_axis=True),
]


def fit_erfc_physical(x, y, u0):
    """phi(u) = phi_0 + (phi_1-phi_0)/2 * erfc(alpha*u - beta*u0), (phi_0,phi_1,alpha,beta)
    free and bounded to [-BOUND, BOUND]. Multi-start over alpha/beta sign+scale and which of
    (phi_0,phi_1) matches the low-x vs. high-x data, keeping the lowest-SSE fit. Returns
    (phi_0, phi_1, alpha, beta, r, model)."""
    def model(x, phi0, phi1, alpha, beta):
        return phi0 + (phi1 - phi0) / 2.0 * erfc(alpha * x - beta * u0)

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
    phi0, phi1, alpha, beta, r, model = fit_erfc_physical(x, y, u0)

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
           label=r'$\phi(u_z) = \phi_0 + \frac{\phi_1-\phi_0}{2}\mathrm{erfc}(\alpha u_z - \beta u_{z,0})$'
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

    fit_info = dict(phi0=phi0, phi1=phi1, alpha=alpha, beta=beta, r=r, u0=u0, wrap=wrap,
                    x=x, y=y)
    return fig, fit_info


def invert_erfc_physical(y, phi0, phi1, alpha, beta, u0):
    """Inverse of phi(u) = phi0 + (phi1-phi0)/2*erfc(alpha*u - beta*u0), solved for u via
    scipy.special.erfcinv. NaN where y falls outside the fitted curve's achievable range
    (erfcinv's domain is (0,2), i.e. phi strictly between phi0 and phi1)."""
    val = (y - phi0) * 2.0 / (phi1 - phi0)
    u = np.full_like(y, np.nan, dtype=float)
    ok = (val > 0) & (val < 2)
    u[ok] = (erfcinv(val[ok]) + beta * u0) / alpha
    return u


def make_axas1_reconstruction_page(roll, inflation_roll, fit_info):
    """4th page, per explicit user request: use AXAS1's page-3 physical-erfc fit (on real
    post-2017 (u_z, phi) data) to predict u_z for the real pre-2017 (2015-2017) SWS phi
    observations, back when the ASHES BOTPT record doesn't exist yet."""
    phi0, phi1, alpha, beta = fit_info['phi0'], fit_info['phi1'], fit_info['alpha'], fit_info['beta']
    u0, wrap, r_fit = fit_info['u0'], fit_info['wrap'], fit_info['r']
    x_real, y_real = fit_info['x'], fit_info['y']

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    ashes_first_date = infl_df['t'].min()

    gap_mask = (roll['t'] >= ERUPTION_END) & (roll['t'] < ashes_first_date)
    gap = roll[gap_mask].dropna(subset=['mean_phi']).sort_values('t')
    y_gap = (gap['mean_phi'].values - wrap) % 180.0

    x_pred = invert_erfc_physical(y_gap, phi0, phi1, alpha, beta, u0)
    n_dropped = int(np.isnan(x_pred).sum())
    print(f'  AXAS1 gap reconstruction: {len(gap)} real 2015-2017 SWS rolling windows, '
         f'{len(gap) - n_dropped} invertible (phi within the fitted curve\'s range)')

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_real, y_real, s=18, color='#0072B2', alpha=0.7,
              label='Real post-eruption data (2017 onward)')
    ax.scatter(x_pred, y_gap, s=22, color='#FF7F0E', alpha=0.85, marker='D',
              label='Predicted uplift from real 2015-2017 SWS phi\n(no real ASHES data in this window)')

    x_line = np.linspace(min(np.nanmin(x_pred), x_real.min()), x_real.max(), 200)
    y_line = phi0 + (phi1 - phi0) / 2.0 * erfc(alpha * x_line - beta * u0)
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=r'$\phi(u_z) = \phi_0 + \frac{\phi_1-\phi_0}{2}\mathrm{erfc}(\alpha u_z - \beta u_{z,0})$'
                 f'\n$\\phi_0$={phi0:.3f}°, $\\phi_1$={phi1:.3f}°, α={alpha:.3f}, β={beta:.3f} '
                 f'($u_{{z,0}}$={u0:.3f})\n(r = {r_fit:.2f}, N = {len(x_real)})')

    ax.axvline(u0, color='gray', lw=0.6, linestyle=':', label=f'Turnover ($u_{{z,0}}$={u0:.3f})')
    ax.set_xlabel('De-tided uplift $u_z$ (m) -- real (2017 onward) or reconstructed (2015-2017)')
    ax.set_ylabel(r'Avg fast direction $\phi$ (deg) [axis labels shifted 90° -- axial, mod 180]')
    ax.set_ylim(0, 180)
    ticks = np.arange(0, 181, 20)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{int((t + wrap) % 180)}' for t in ticks])
    ax.set_title('AXAS1: Reconstructed Uplift from Fast Direction\n'
                 '(2015-2017 gap, using the post-2017 phi<->uplift erfc fit)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    figs = []
    axas1_roll = None
    axas1_inflation_roll = None
    axas1_fit_info = None
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

        fig, fit_info = make_page(station, roll, inflation_roll, baseline_phi)
        figs.append(fig)
        print()

        if name == 'AXAS1':
            axas1_roll = roll
            axas1_inflation_roll = inflation_roll
            axas1_fit_info = fit_info

    print('Building page 4: AXAS1 2015-2017 gap reconstruction '
          '(physical erfc fit, inverted via erfcinv)...')
    figs.append(make_axas1_reconstruction_page(axas1_roll, axas1_inflation_roll, axas1_fit_info))

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
