#!/usr/bin/env python3
"""
uplift_vs_phi_axec2_axcc1_axas1_90day.py

Same as uplift_vs_phi_axec2_axcc1_axas1.py (AXEC2/AXCC1/AXAS1 post-eruption "uplift vs. fast
direction" pages + AXEC2's eruption-transition page, same wrap/x-shift conventions -- see that
script's docstring), EXCEPT BOTH axes now use a matched 90-day window, computed in TWO STAGES
per explicit user request ("calculate daily averages on each axis first, then use those in
90-day rolling windows on both averages"), AND (per a later explicit user request) each page's
line of fit is a SIGMOID f(x) = A/(B + C*exp(-D*x)) + E, fit parameters shown at 3 decimal
places -- same form as sigmoid_uplift_vs_phi_axec2_axcc1_axas1.py, but C is FIXED for every
page here per explicit user request (C=0.6 AXEC2, C=1.0 AXCC1, C=0.0 AXAS1; the AXEC2
eruption-transition page keeps its free-C fit, since a fixed C=-0.6 there is degenerate/
diverges -- see that script's docstring) -- NOT the cube-root fit this script originally used:
  - uplift (x-axis): bpr_inflation_periods*.py's load_daily_series() already returns daily
    means (resample('1D')) before any rolling -- this script takes that raw daily series
    and applies its OWN 90-day centered rolling mean locally (module-default ROLLING_DAYS=30
    is not overridden globally, to avoid touching the shared module).
  - phi (y-axis): axec2_uplift_phi_cosine_vs_time.rolling_phi_stats_daily_then_roll() first
    collapses events into per-CALENDAR-DAY circular means (every day weighted equally
    regardless of event count that day, matching the uplift side), THEN takes a 90-day
    centered rolling mean of those daily means (doubled-angle trick, phi being axial).
    This replaces the earlier single-stage version of this script, which pooled raw events
    directly within each 90-day window (implicitly letting denser days dominate).
Part of a comparison series (this file, the companion _180day.py at 24 weeks/6 months, and
the single-stage _1week.py/_2week.py/_30day.py at 7/14/30 days) to see how much of the
apparent sigmoid shape in "avg fast direction vs. de-tided uplift" is a smoothing artifact of
wide averaging windows versus a real signal that survives finer time resolution. Page 4
(AXEC2's eruption transition) is UNCHANGED -- it already uses a 12-hour window, unrelated to
this comparison.

Produces (NEW file, 4 pages):
    uplift_vs_phi_axec2_axcc1_axas1_90day.pdf

Run with:
    python3 uplift_vs_phi_axec2_axcc1_axas1_90day.py
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
    UNLEVEL_START, UNLEVEL_END, rolling_phi_stats_fine, ERUPTION_TRANSITION_START,
)
import bpr_inflation_periods as ecal_infl
import bpr_inflation_periods_ccal as ccal_infl
import bpr_inflation_periods_ashes as ashes_infl

# BOUND caps |A|,|B|,|D|,|E| during curve_fit. Same sigmoid form as
# sigmoid_uplift_vs_phi_axec2_axcc1_axas1.py's fit_sigmoid/fit_sigmoid_free, but that
# script's UNBOUNDED curve_fit let A and B run away to +-1e30-scale values on this script's
# 90-day two-stage-binned AXCC1 data (a flat numerator/denominator ratio degeneracy that
# still reproduces nearly the same curve shape, just via absurd individual parameter values
# that blew up the page-2 legend). Bounding keeps the fit well-scaled without changing the
# fitted curve itself.
BOUND = 1000.0


def signed_term(v):
    """Same purpose as sigmoid_uplift_vs_phi_axec2_axcc1_axas1.signed_term, but at 3
    decimal places (per explicit user request) instead of that module's 2."""
    return f'{v:+.3f}'


def fit_sigmoid(x, y, C):
    """f(x) = A/(B + C*exp(-D*x)) + E, C fixed, (A,B,D,E) free but bounded to
    [-BOUND, BOUND] so the legend stays readable. Returns (A,B,D,E,r,model)."""
    def model(x, A, B, D, E):
        return A / (B + C * np.exp(-D * x)) + E

    E0 = float(np.min(y))
    amp0 = float(np.max(y) - np.min(y)) or 1.0
    A0 = amp0
    B0 = 1.0
    D0 = 1.0 / (np.std(x) + 1e-6)
    if np.corrcoef(x, y)[0, 1] < 0:
        D0 = -D0
    p0 = [np.clip(v, -BOUND, BOUND) for v in (A0, B0, D0, E0)]
    bounds = ([-BOUND] * 4, [BOUND] * 4)
    try:
        (A, B, D, E), _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        A, B, D, E = p0
    y_pred = model(x, A, B, D, E)
    r = np.corrcoef(y_pred, y)[0, 1]
    return A, B, D, E, r, model


def fit_sigmoid_free(x, y):
    """Same form, f(x) = A/(B + C*exp(-D*x)) + E, but ALL FIVE parameters free and bounded
    to [-BOUND, BOUND]. Returns (A,B,C,D,E,r,model)."""
    def model(x, A, B, C, D, E):
        return A / (B + C * np.exp(-D * x)) + E

    E0 = float(np.min(y))
    amp0 = float(np.max(y) - np.min(y)) or 1.0
    A0 = amp0
    B0 = 1.0
    C0 = 1.0
    D0 = 1.0 / (np.std(x) + 1e-6)
    if np.corrcoef(x, y)[0, 1] < 0:
        D0 = -D0
    p0 = [np.clip(v, -BOUND, BOUND) for v in (A0, B0, C0, D0, E0)]
    bounds = ([-BOUND] * 5, [BOUND] * 5)
    try:
        (A, B, C, D, E), _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        A, B, C, D, E = p0
    y_pred = model(x, A, B, C, D, E)
    r = np.corrcoef(y_pred, y)[0, 1]

    def model4(x, A=A, B=B, C=C, D=D, E=E):
        return A / (B + C * np.exp(-D * x)) + E

    return A, B, C, D, E, r, model4

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'uplift_vs_phi_axec2_axcc1_axas1_90day.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9   # rolling_phi_stats_daily_then_roll's min_periods, in days (~window/10)

UPLIFT_ROLLING_DAYS = 90   # matches ROLL_WINDOW_DAYS -- overrides the module-default 30-day
                          # geodetic smoothing locally, without touching the shared modules

STATIONS = [
    dict(name='AXEC2', infl_module=ecal_infl, geodetic_label='Eastern Caldera BOTPT',
        flag_unleveled=True, C=0.6, wrap_axis=False, free_C=False),
    dict(name='AXCC1', infl_module=ccal_infl, geodetic_label='Central Caldera BOTPT',
        flag_unleveled=False, C=1.0, wrap_axis=True, free_C=False),
    # AXAS1: C fixed near 0.0 per explicit user request. Exactly C=0.0 is mathematically
    # degenerate -- it kills the exp(-Dx) term, collapsing the model to the constant A/B+E
    # with no dependence on D -- so C=0.001 is used instead (same "fixed at 0" intent,
    # without the singularity).
    dict(name='AXAS1', infl_module=ashes_infl, geodetic_label='ASHES vent field BOTPT',
        flag_unleveled=False, C=0.001, wrap_axis=True, free_C=False),
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

    if station['free_C']:
        A, B, C, D, E, r, model4 = fit_sigmoid_free(x, y)
        model = lambda xv: model4(xv)
    else:
        C = station['C']
        A, B, D, E, r, model_fixed = fit_sigmoid(x, y, C)
        model = lambda xv: model_fixed(xv, A, B, D, E)

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
    y_line = model(x_line)
    c_label = f'C={C:.3f}' if not station['free_C'] else f'C={C:.3f} (free)'
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Sigmoid fit ({c_label}): $\\phi$ = {A:.3f}/({B:.3f}{signed_term(C)}$e^{{{signed_term(-D)}x}}$) + {E:.3f}\n'
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
    """4th page -- UNCHANGED from uplift_vs_phi_axec2_axcc1_axas1.py: AXEC2's
    pre-through-syn-eruption transition already uses a 12-hour window, unrelated to the
    1-week/2-week rolling-window comparison this script exists for."""
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

    # C fixed at -0.6 (this page's old cube-root x-shift) is degenerate for the sigmoid --
    # same issue noted in sigmoid_uplift_vs_phi_axec2_axcc1_axas1.py's docstring -- so C is
    # free here.
    A, B, C, D, E, r, model4 = fit_sigmoid_free(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_syn.values], y[~is_syn.values], s=20, color='#800080', alpha=0.7,
              label='Pre-eruption')
    ax.scatter(x[is_syn.values], y[is_syn.values], s=20, color='#CC0000', alpha=0.7,
              label='Syn-eruption')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = model4(x_line)
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Sigmoid fit (C={C:.3f}, free): $\\phi$ = {A:.3f}/({B:.3f}{signed_term(C)}$e^{{{signed_term(-D)}x}}$) + {E:.3f}\n'
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
          '(uplift vs. fast direction, sigmoid fit, unchanged 12-hour window)...')
    figs.append(make_eruption_transition_page(axec2_df, axec2_baseline_phi))

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
