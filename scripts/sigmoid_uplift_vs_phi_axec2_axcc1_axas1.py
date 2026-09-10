#!/usr/bin/env python3
"""
sigmoid_uplift_vs_phi_axec2_axcc1_axas1.py

Sigmoid-fit counterpart of uplift_vs_phi_axec2_axcc1_axas1.py, per explicit user request: same
4 pages (AXEC2, AXCC1, AXAS1 post-eruption "uplift vs. fast direction", + AXEC2's pre-through-
syn-eruption transition page), same data/wrap/flag conventions, but each page's cube-root fit is
replaced with a sigmoid of the form

    f(x) = A / (B + C*exp(-D*x)) + E

where C is FIXED at that page's previous cube-root x-shift value (0.6 for AXEC2 post-eruption,
1.0 for AXCC1, 0.0 for AXAS1, -0.6 for AXEC2's eruption-transition page), and A, B, D, E are
free parameters fit via scipy.optimize.curve_fit.

Produces (NEW file, 4 pages):
    sigmoid_uplift_vs_phi_axec2_axcc1_axas1.pdf

Run with:
    python3 sigmoid_uplift_vs_phi_axec2_axcc1_axas1.py
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
    rolling_phi_stats, ERUPTION_START, ERUPTION_END,
    UNLEVEL_START, UNLEVEL_END, rolling_phi_stats_fine, ERUPTION_TRANSITION_START,
)
import bpr_inflation_periods as ecal_infl
import bpr_inflation_periods_ccal as ccal_infl
import bpr_inflation_periods_ashes as ashes_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'sigmoid_uplift_vs_phi_axec2_axcc1_axas1.pdf')

STATIONS = [
    dict(name='AXEC2', infl_module=ecal_infl, geodetic_label='Eastern Caldera BOTPT',
        flag_unleveled=True, C=0.6, wrap_axis=False, free_C=False),
    dict(name='AXCC1', infl_module=ccal_infl, geodetic_label='Central Caldera BOTPT',
        flag_unleveled=False, C=1.0, wrap_axis=True, free_C=False),
    # AXAS1: per explicit user request, page 3 reuses the SAME transition-at-x=0
    # logistic fit as page 5 (fit_logistic_at_zero, the reconstruction page's fit),
    # rather than its own independent free-C sigmoid fit -- only the right half (x>=0,
    # where the real data lives) is shown here.
    dict(name='AXAS1', infl_module=ashes_infl, geodetic_label='ASHES vent field BOTPT',
        flag_unleveled=False, C=0.0, wrap_axis=True, free_C=True, use_logistic_at_zero=True),
]


def signed_term(v):
    """Format a value with an explicit sign, for building label strings like
    'B+C' or 'e^{-Dx}' without ever concatenating two literal sign characters
    (e.g. '+' followed by a negative number's own '-', or a literal '-' before
    a negative D) into '+-' or '--'."""
    return f'{v:+.2f}'


def fit_sigmoid(x, y, C):
    """f(x) = A/(B + C*exp(-D*x)) + E, C fixed, (A,B,D,E) free. Returns (A,B,D,E,r)."""
    def model(x, A, B, D, E):
        return A / (B + C * np.exp(-D * x)) + E

    E0 = float(np.min(y))
    amp0 = float(np.max(y) - np.min(y)) or 1.0
    A0 = amp0
    B0 = 1.0
    D0 = 1.0 / (np.std(x) + 1e-6)
    if np.corrcoef(x, y)[0, 1] < 0:
        D0 = -D0
    p0 = [A0, B0, D0, E0]
    try:
        (A, B, D, E), _ = curve_fit(model, x, y, p0=p0, maxfev=20000)
    except RuntimeError:
        A, B, D, E = p0
    y_pred = model(x, A, B, D, E)
    r = np.corrcoef(y_pred, y)[0, 1]
    return A, B, D, E, r, model


def fit_sigmoid_free(x, y):
    """Same form, f(x) = A/(B + C*exp(-D*x)) + E, but ALL FIVE parameters free -- per
    explicit user request, for the two pages where fixing C at the old cube-root shift
    value was mathematically degenerate (C=0, AXAS1) or produced a fit that diverged
    through a singularity inside the data range (C=-0.6, AXEC2 eruption-transition page).
    Returns (A,B,C,D,E,r,model)."""
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
    p0 = [A0, B0, C0, D0, E0]
    try:
        (A, B, C, D, E), _ = curve_fit(model, x, y, p0=p0, maxfev=20000)
    except RuntimeError:
        A, B, C, D, E = p0
    y_pred = model(x, A, B, C, D, E)
    r = np.corrcoef(y_pred, y)[0, 1]

    def model4(x, A=A, B=B, C=C, D=D, E=E):
        return A / (B + C * np.exp(-D * x)) + E

    return A, B, C, D, E, r, model4
    y_pred = model(x, A, B, D, E)
    r = np.corrcoef(y_pred, y)[0, 1]
    return A, B, D, E, r, model


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

    use_logistic = station.get('use_logistic_at_zero', False)
    if use_logistic:
        A, D, E, r = fit_logistic_at_zero(x, y)
        model = lambda xv: A / (1.0 + np.exp(-D * xv)) + E
    elif station['free_C']:
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

    x_line = np.linspace(max(x.min(), 0.0), x.max(), 200)
    y_line = model(x_line)
    if use_logistic:
        ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
               label=f'Logistic fit (transition at x=0, same fit as page 5, right half only): '
                     f'$\\phi$ = {A:.2f}/(1+$e^{{{signed_term(-D)}x}}$) + {E:.2f}\n(r = {r:.2f}, N = {len(x)})')
    else:
        c_label = f'C={C:.2f}' if not station['free_C'] else f'C={C:.2f} (free)'
        ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
               label=f'Sigmoid fit ({c_label}): $\\phi$ = {A:.2f}/({B:.2f}{signed_term(C)}$e^{{{signed_term(-D)}x}}$) + {E:.2f}\n'
                     f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel('De-tided uplift (m, 30-day rolling mean)')
    ylabel = r'Avg fast direction $\phi$ (deg)'
    if station['wrap_axis']:
        ylabel += ' [axis labels shifted 90° -- axial, mod 180]'
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 180)
    if station['wrap_axis']:
        ticks = np.arange(0, 181, 20)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int((t + wrap) % 180)}' for t in ticks])
    ax.set_title(f'{station["name"]}: De-Tided Uplift vs. Fast Direction\n'
                 f'({station["geodetic_label"]}, post-eruption rolling windows)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_eruption_transition_page(df, baseline_phi):
    """4th page: AXEC2's pre-through-syn-eruption transition, 12-hour rolling windows.
    C=-0.6 (this page's previous cube-root x-shift) made the fixed-C sigmoid diverge
    through a singularity inside the data range, so per explicit user request C is a
    FREE parameter here instead."""
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

    A, B, C, D, E, r, model4 = fit_sigmoid_free(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_syn.values], y[~is_syn.values], s=20, color='#800080', alpha=0.7,
              label='Pre-eruption')
    ax.scatter(x[is_syn.values], y[is_syn.values], s=20, color='#CC0000', alpha=0.7,
              label='Syn-eruption')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = model4(x_line)
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Sigmoid fit (C={C:.2f}, free): $\\phi$ = {A:.2f}/({B:.2f}{signed_term(C)}$e^{{{signed_term(-D)}x}}$) + {E:.2f}\n'
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


# ── AXAS1 gap reconstruction: use the phi<->uplift relation to back out what uplift ────
# should have been during 2015-2017, when ASHES had no data yet but the splitting
# catalog did, per explicit user request.

def fit_logistic_at_zero(x, y):
    """f(x) = A/(1+exp(-D*x)) + E -- transition pinned exactly at x=0, which is where
    the real ASHES record begins (its own zero/reference point, 2017-08-15 onward) -- no
    shift, no anchor. All A, D, E free (3 params). A naive single initial guess is prone
    to bad local minima here (the data's dynamic range is narrow relative to what the
    curve needs to span), so a small multi-start grid is used and the best (lowest SSE)
    fit is kept. Returns (A, D, E, r)."""
    def model(x, A, D, E):
        return A / (1.0 + np.exp(-D * x)) + E

    y_span = float(np.max(y) - np.min(y)) or 1.0
    x_span = float(np.std(x) + 1e-6)
    best = None
    for A0 in (y_span, -y_span, 3 * y_span, -3 * y_span, 6 * y_span, -6 * y_span):
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


def invert_logistic_at_zero(y, A, D, E):
    """Inverse of f(x') = A/(1+exp(-D*x')) + E, solved for x'. NaN where y is outside
    the fitted curve's achievable range (A/(y-E)-1 <= 0)."""
    val = A / (y - E) - 1.0
    x = np.full_like(y, np.nan, dtype=float)
    ok = val > 0
    x[ok] = -(1.0 / D) * np.log(val[ok])
    return x


def make_axas1_reconstruction_page(df, baseline_phi):
    """New version of page 3, per explicit user request: the sigmoid's transition point
    is pinned exactly where the real data begins -- i.e. at x_real=0, the ASHES record's
    own zero/reference point (2017-08-15 onward), with NO shift and NO forced anchor. Fit
    A, D, E freely to the real (uplift, phi) data alone, transition fixed at x=0. That
    single fitted curve (both halves -- one continuous S-shape) is then used to invert
    ALL of the real 2015-2017 AXAS1 SWS phi observations (the gap before ASHES was
    deployed, when there is no real Uz at all) into predicted x (uplift) values --
    expected to land left of x=0 (negative), i.e. less inflated than when the real
    record starts."""
    wrap = 90.0

    roll_all = rolling_phi_stats(df, baseline_phi)
    _dd, _infl, inflation_roll, _rt, _rd = ashes_infl.load_daily_series()

    valid_real = roll_all.dropna(subset=['mean_phi']).copy()
    valid_real = valid_real[valid_real['t'] >= ERUPTION_END]
    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged_real = pd.merge_asof(valid_real.sort_values('t'), infl_df.sort_values('t'), on='t',
                                direction='nearest', tolerance=pd.Timedelta('20D'))
    merged_real = merged_real.dropna(subset=['inflation_m'])

    x_real = merged_real['inflation_m'].values
    y_real = (merged_real['mean_phi'].values - wrap) % 180.0

    A, D, E, r = fit_logistic_at_zero(x_real, y_real)
    print(f'  AXAS1: fit transition-at-x=0 logistic on real data alone (no shift, no '
         f'anchor); asymptotes {E:.1f} deg (x->+inf) to {A + E:.1f} deg (x->-inf), r={r:.2f}')

    ashes_first_date = infl_df['t'].min()
    gap_mask = (roll_all['t'] >= ERUPTION_END) & (roll_all['t'] < ashes_first_date)
    gap = roll_all[gap_mask].dropna(subset=['mean_phi']).sort_values('t')
    y_gap = (gap['mean_phi'].values - wrap) % 180.0
    x_pred = invert_logistic_at_zero(y_gap, A, D, E)
    n_dropped = int(np.isnan(x_pred).sum())
    print(f'  AXAS1 gap reconstruction: {len(gap)} real 2015-2017 SWS rolling windows, '
         f'{len(gap) - n_dropped} invertible (phi within the fitted curve\'s range)')

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_real, y_real, s=18, color='#0072B2', alpha=0.7,
              label='Real post-eruption data (2017 onward)')
    ax.scatter(x_pred, y_gap, s=22, color='#FF7F0E', alpha=0.85, marker='D',
              label='Predicted uplift from real 2015-2017 SWS phi\n(no real ASHES data in this window)')

    x_line = np.linspace(min(np.nanmin(x_pred), x_real.min()), x_real.max(), 200)
    y_line = A / (1.0 + np.exp(-D * x_line)) + E
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Logistic (transition at x=0, where real data begins): '
                 f'$\\phi$ = {A:.2f}/(1+$e^{{{signed_term(-D)}x}}$) + {E:.2f}\n(r = {r:.2f}, N = {len(x_real)})')

    ax.axvline(0.0, color='gray', lw=0.6, linestyle=':', label='Transition (real data begins, x=0)')
    ax.set_xlabel('De-tided uplift (m) -- real (2017 onward) or reconstructed (2015-2017)')
    ax.set_ylabel(r'Avg fast direction $\phi$ (deg) [axis labels shifted 90° -- axial, mod 180]')
    ax.set_ylim(0, 180)
    ticks = np.arange(0, 181, 20)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{int((t + wrap) % 180)}' for t in ticks])
    ax.set_title('AXAS1: Reconstructed Uplift from Fast Direction\n'
                 '(2015-2017 gap, using the post-2017 phi<->uplift relationship)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    figs = []
    axec2_df = None
    axec2_baseline_phi = None
    axas1_df = None
    axas1_baseline_phi = None
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

        roll = rolling_phi_stats(df, baseline_phi)
        n_valid = roll['mean_phi'].notna().sum()
        print(f'  {n_valid}/{len(roll)} rolling windows have enough events')

        _dd, _infl, inflation_roll, _rt, _rd = station['infl_module'].load_daily_series()

        fig = make_page(station, roll, inflation_roll, baseline_phi)
        figs.append(fig)
        print()

        if name == 'AXEC2':
            axec2_df = df
            axec2_baseline_phi = baseline_phi
        elif name == 'AXAS1':
            axas1_df = df
            axas1_baseline_phi = baseline_phi

    print('Building page 4: AXEC2 pre-through-syn-eruption transition '
          '(uplift vs. fast direction, sigmoid fit)...')
    figs.append(make_eruption_transition_page(axec2_df, axec2_baseline_phi))

    print('Building page 5: AXAS1 2015-2017 gap reconstruction '
          '(logistic transition pinned where real data begins, x=0, no shift)...')
    figs.append(make_axas1_reconstruction_page(axas1_df, axas1_baseline_phi))

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
