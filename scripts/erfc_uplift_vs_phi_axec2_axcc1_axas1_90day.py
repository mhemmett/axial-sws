#!/usr/bin/env python3
"""
erfc_uplift_vs_phi_axec2_axcc1_axas1_90day.py

Complementary-error-function counterpart of uplift_vs_phi_axec2_axcc1_axas1_90day.py, per
explicit user request: identical data pipeline (same AXEC2/AXCC1/AXAS1 post-eruption pages +
AXEC2's 12-hour eruption-transition page, same wrap/flag conventions, same 90-day two-stage
binning -- daily-then-90-day-rolled phi and a locally-applied 90-day rolling mean of the raw
daily uplift series -- see that script's docstring for the two-stage-binning rationale), but
each page's line of fit is now

    f(x) = A * erfc(D*(x - C)) + E

(scipy.special.erfc) instead of that script's sigmoid A/(B + C*exp(-D*x)) + E. C plays the
same "transition location" role as the sigmoid's C (fixed at the old cube-root x-shift value
for AXEC2/AXCC1: 0.6 and 1.0; near-zero at 0.001 for AXAS1, matching that script's choice to
avoid a C=0 exact degeneracy there too -- though erfc's C=0 is NOT degenerate the way the
sigmoid's is, 0.001 is kept for consistency with the sigmoid version's convention). AXEC2's
eruption-transition page keeps C free, for the same reason the sigmoid version does (a fixed
C=-0.6 sat inside a region where the fit behaved poorly there). (A,D,E) are free, fit via
scipy.optimize.curve_fit with the same +-1000 parameter bound used in the sigmoid version
(that script's unbounded curve_fit let sigmoid's A/B blow up to +-1e30 scale on AXCC1's data;
the same bound is applied here defensively, even though erfc's parameterization does not have
that specific numerator/denominator degeneracy).

Produces (NEW file, 4 pages):
    erfc_uplift_vs_phi_axec2_axcc1_axas1_90day.pdf

Run with:
    python3 erfc_uplift_vs_phi_axec2_axcc1_axas1_90day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
    UNLEVEL_START, UNLEVEL_END, rolling_phi_stats_fine, ERUPTION_TRANSITION_START,
)
import bpr_inflation_periods as ecal_infl
import bpr_inflation_periods_ccal as ccal_infl
import bpr_inflation_periods_ashes as ashes_infl

# Same defensive bound as uplift_vs_phi_axec2_axcc1_axas1_90day.py's sigmoid fits -- see that
# script's BOUND comment.
BOUND = 1000.0


def signed_term(v):
    return f'{v:+.3f}'


def fit_erfc(x, y, C):
    """f(x) = A*erfc(D*(x-C)) + E, C fixed, (A,D,E) free but bounded to [-BOUND, BOUND].
    Returns (A,D,E,r,model)."""
    def model(x, A, D, E):
        return A * erfc(D * (x - C)) + E

    E0 = float(np.min(y))
    amp0 = float(np.max(y) - np.min(y)) or 1.0
    A0 = amp0 / 2.0
    D0 = 1.0 / (np.std(x) + 1e-6)
    if np.corrcoef(x, y)[0, 1] < 0:
        D0 = -D0
    p0 = [np.clip(v, -BOUND, BOUND) for v in (A0, D0, E0)]
    bounds = ([-BOUND] * 3, [BOUND] * 3)
    try:
        (A, D, E), _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        A, D, E = p0
    y_pred = model(x, A, D, E)
    r = np.corrcoef(y_pred, y)[0, 1]
    return A, D, E, r, model


def fit_erfc_at_zero(x, y):
    """f(x) = A*erfc(D*x) + E -- C pinned at EXACTLY 0, the point where AXAS1's real record
    begins (its own zero/reference point), mirroring
    sigmoid_uplift_vs_phi_axec2_axcc1_axas1.py's fit_logistic_at_zero for the same station:
    only the right half (x>=0, where the real data lives) is meaningful, so C is not left
    free to wander to some other location. Only A, D, E free (3 params). A single naive
    initial guess is prone to bad local minima here (same issue fit_logistic_at_zero notes),
    so a small multi-start grid is used and the best (lowest SSE) fit is kept. Returns
    (A, D, E, r)."""
    def model(x, A, D, E):
        return A * erfc(D * x) + E

    y_span = float(np.max(y) - np.min(y)) or 1.0
    x_span = float(np.std(x) + 1e-6)
    best = None
    for A0 in (y_span / 2, -y_span / 2, y_span, -y_span, 3 * y_span, -3 * y_span):
        for D0 in (1 / x_span, -1 / x_span, 3 / x_span, -3 / x_span, 10 / x_span, -10 / x_span):
            for E0 in (float(np.min(y)), float(np.max(y)), float(np.mean(y))):
                try:
                    p, _ = curve_fit(model, x, y, p0=[A0, D0, E0], maxfev=10000)
                except RuntimeError:
                    continue
                y_pred = model(x, *p)
                sse = float(np.sum((y_pred - y) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, p)
    A, D, E = best[1]
    y_pred = model(x, A, D, E)
    r = np.corrcoef(y_pred, y)[0, 1]
    return A, D, E, r


def fit_erfc_free(x, y):
    """Same form, f(x) = A*erfc(D*(x-C)) + E, but ALL FOUR parameters free and bounded to
    [-BOUND, BOUND]. Returns (A,C,D,E,r,model4)."""
    def model(x, A, C, D, E):
        return A * erfc(D * (x - C)) + E

    E0 = float(np.min(y))
    amp0 = float(np.max(y) - np.min(y)) or 1.0
    A0 = amp0 / 2.0
    C0 = float(np.mean(x))
    D0 = 1.0 / (np.std(x) + 1e-6)
    if np.corrcoef(x, y)[0, 1] < 0:
        D0 = -D0
    p0 = [np.clip(v, -BOUND, BOUND) for v in (A0, C0, D0, E0)]
    bounds = ([-BOUND] * 4, [BOUND] * 4)
    try:
        (A, C, D, E), _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        A, C, D, E = p0
    y_pred = model(x, A, C, D, E)
    r = np.corrcoef(y_pred, y)[0, 1]

    def model4(x, A=A, C=C, D=D, E=E):
        return A * erfc(D * (x - C)) + E

    return A, C, D, E, r, model4


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'erfc_uplift_vs_phi_axec2_axcc1_axas1_90day.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9   # rolling_phi_stats_daily_then_roll's min_periods, in days (~window/10)

UPLIFT_ROLLING_DAYS = 90   # matches ROLL_WINDOW_DAYS -- overrides the module-default 30-day
                          # geodetic smoothing locally, without touching the shared modules

STATIONS = [
    dict(name='AXEC2', infl_module=ecal_infl, geodetic_label='Eastern Caldera BOTPT',
        flag_unleveled=True, C=0.6, wrap_axis=False, free_C=False),
    dict(name='AXCC1', infl_module=ccal_infl, geodetic_label='Central Caldera BOTPT',
        flag_unleveled=False, C=1.0, wrap_axis=True, free_C=False),
    # AXAS1: transition pinned at exactly x=0 (fit_erfc_at_zero), only the right half
    # (x>=0) shown/fit -- per explicit user request, same convention as
    # sigmoid_uplift_vs_phi_axec2_axcc1_axas1.py's fit_logistic_at_zero for this station.
    dict(name='AXAS1', infl_module=ashes_infl, geodetic_label='ASHES vent field BOTPT',
        flag_unleveled=False, wrap_axis=True, use_at_zero=True),
]


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

    use_at_zero = station.get('use_at_zero', False)
    if use_at_zero:
        A, D, E, r = fit_erfc_at_zero(x, y)
        C = 0.0
        model = lambda xv: A * erfc(D * xv) + E
    elif station['free_C']:
        A, C, D, E, r, model4 = fit_erfc_free(x, y)
        model = lambda xv: model4(xv)
    else:
        C = station['C']
        A, D, E, r, model_fixed = fit_erfc(x, y, C)
        model = lambda xv: model_fixed(xv, A, D, E)

    fig, ax = plt.subplots(figsize=(7, 6))
    if station['flag_unleveled']:
        is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
        ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2',
                  alpha=0.7, label='Post-eruption window')
        ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
                  marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')
    else:
        ax.scatter(x, y, s=18, color='#0072B2', alpha=0.7, label='Post-eruption window')

    x_line = np.linspace(max(x.min(), 0.0), x.max(), 200)
    y_line = model(x_line)
    if use_at_zero:
        c_label = 'C=0.000, transition pinned at x=0, right half only'
    elif station['free_C']:
        c_label = f'C={C:.3f} (free)'
    else:
        c_label = f'C={C:.3f}'
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Erfc fit ({c_label}): $\\phi$ = {A:.3f}$\\cdot$erfc({D:.3f}(x{signed_term(-C)})) + {E:.3f}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel(f'De-tided uplift (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
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


def make_eruption_transition_page(df, baseline_phi):
    """4th page -- AXEC2's pre-through-syn-eruption transition, 12-hour rolling windows,
    C free (same rationale as the sigmoid version's page 4 -- see this script's docstring)."""
    daily_depth, _infl, _infl_roll, _rt, _rd = ecal_infl.load_daily_series()
    ref_window_end = ERUPTION_START - pd.Timedelta(days=5)
    ref_depth = float(daily_depth.loc[ERUPTION_TRANSITION_START:ref_window_end].mean())
    uplift_transition = daily_depth - ref_depth

    roll_fine = rolling_phi_stats_fine(df, baseline_phi, ERUPTION_TRANSITION_START, ERUPTION_END,
                                       window_days=0.5, step_days=0.5)
    valid = roll_fine.dropna(subset=['mean_phi'])

    up_df = uplift_transition.reset_index()
    up_df.columns = ['t', 'uplift_m']
    merged = pd.merge_asof(valid.sort_values('t'), up_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('3D'))
    merged = merged.dropna(subset=['uplift_m'])

    is_syn = merged['t'] >= ERUPTION_START
    x = merged['uplift_m'].values
    y = merged['mean_phi'].values

    A, C, D, E, r, model4 = fit_erfc_free(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_syn.values], y[~is_syn.values], s=20, color='#800080', alpha=0.7,
              label='Pre-eruption')
    ax.scatter(x[is_syn.values], y[is_syn.values], s=20, color='#CC0000', alpha=0.7,
              label='Syn-eruption')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = model4(x_line)
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Erfc fit (C={C:.3f}, free): $\\phi$ = {A:.3f}$\\cdot$erfc({D:.3f}(x{signed_term(-C)})) + {E:.3f}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(baseline_phi, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.axvline(0.0, color='gray', lw=0.6, linestyle=':')
    ax.set_xlabel('De-tided uplift (m, relative to pre-onset baseline -- deflating = negative)')
    ax.set_ylabel(r'Avg fast direction $\phi$ (deg)')
    ax.set_ylim(0, 180)
    ax.set_title('AXEC2: De-Tided Uplift vs. Fast Direction\n'
                 '(pre-eruption through syn-eruption, 12-hour rolling windows)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    figs = []
    axec2_df = None
    axec2_baseline_phi = None
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

        fig = make_page(station, roll, inflation_roll, baseline_phi)
        figs.append(fig)
        print()

        if name == 'AXEC2':
            axec2_df = df
            axec2_baseline_phi = baseline_phi

    print('Building page 4: AXEC2 pre-through-syn-eruption transition '
          '(uplift vs. fast direction, erfc fit, unchanged 12-hour window)...')
    figs.append(make_eruption_transition_page(axec2_df, axec2_baseline_phi))

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
