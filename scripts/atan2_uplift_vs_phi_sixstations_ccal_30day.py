#!/usr/bin/env python3
"""
atan2_uplift_vs_phi_sixstations_ccal_30day.py

Six-station atan2 vector-sum fit, per explicit user request: return to the geodetic-vs-SWS
comparison, but with two changes from the earlier 3-station version
(atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.py):
  - ALL SIX stations (AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3), not just three, each compared
    against the SAME geodetic series -- the Central Caldera BOTPT (bpr_inflation_periods_ccal)
    -- not each station's own local BOTPT (matching the no-fit 6-station scatter precedent in
    atan2_uplift_vs_phi_sixstations_ccal_90day_full_nofit.py).
  - Matched 30-day rolling windows on BOTH axes (phi and uplift), replacing the earlier
    30-day-uplift-vs-180-day-phi mismatch the user flagged (2026-08-05 note) and the later
    90-day-both convention -- both series now use the SAME window length and the SAME
    daily-mean-then-roll method (uplift already worked this way; phi now does too, via
    rolling_phi_stats_daily_then_roll, matching the geodetic side's own daily-mean-first
    approach rather than pooling raw events per window).

TURNOVER (u0): fit_atan2_vectorsum needs a FIXED per-station turnover u0 (the exact uplift
value where the curve's steepest transition is pinned -- see that function's docstring). The
earlier 3-station script hand-picked u0 per station (0.6/1.0/0.0 m) against each station's OWN
local uplift record -- those values don't transfer here, since every station now shares the
same CCAL x-axis, which has its own different scale/range. Per explicit user request, u0 is
instead AUTO-ESTIMATED per station: fit a free-location 4-parameter logistic sigmoid
(estimate_turnover_u0, multi-start, same multi-start-over-signs-and-scales pattern as
fit_erfc_physical in erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py) to each station's
own (uplift, wrapped-phi) scatter, and use its fitted x0 as u0 -- reproducible, no manual
per-station tuning, and estimated the same way for every station.

AXCC1: checked whether AXCC1's known 2015-03-01 to 2015-04-28 raw-data gap (see
axcc1_uplift_phi_cosine_vs_time.py's GAP_START/GAP_END) needs special flagging here -- it does
NOT, because that gap is entirely PRE-eruption and this fit (like the 3-station version) only
uses POST-eruption data (valid = valid[valid['t'] >= ERUPTION_END]), so the gap never enters
the fitted domain. No code in this repo implements an ongoing post-tip orientation/tilt-flagged
window for AXCC1 distinct from that pre-eruption gap (README's "tipped over... treated
separately" caveat is not backed by a specific flagged date range anywhere in scripts/) --
AXCC1 is therefore included on equal footing with the other 5 stations, with only the raw-data
gap checked (and found irrelevant to this domain). Flag to the user: if there IS a specific
post-tip tilt-affected window in mind, it isn't yet encoded anywhere in this codebase and
should be added explicitly once specified.

Produces (6 pages, one per station, in STATION_ORDER below):
    atan2_uplift_vs_phi_sixstations_ccal_30day.pdf

Run with:
    python3 atan2_uplift_vs_phi_sixstations_ccal_30day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
    UNLEVEL_START, UNLEVEL_END,
)
from animate_arctan_stress_vectors import (
    compute_optimal_wrap, compute_full_range_ticks, fit_atan2_vectorsum,
    ALPHA_FIXED_AZ_DEG, _azimuth,
)
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'atan2_uplift_vs_phi_sixstations_ccal_30day.pdf')

ROLL_WINDOW_DAYS = 30   # phi (was 180 pooled / 90 daily-then-roll in earlier scripts)
ROLL_MIN_DAYS = 3       # matches the max(3, window_days//10) default used elsewhere
UPLIFT_ROLLING_DAYS = 30   # matches bpr_inflation_periods_ccal's own ROLLING_DAYS -- same
                           # window length on both axes, no second re-roll on top

STATION_ORDER = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
GEODETIC_LABEL = 'Central Caldera BOTPT'
UNLEVELED_STATIONS = {'AXEC2'}
BOUND = 1000.0


def estimate_turnover_u0(x, y):
    """Auto-estimate the turnover u0 (see module docstring): fit a free-location logistic
    phi0 + (phi1-phi0)/(1+exp(-k*(x-x0))) via multi-start curve_fit (over which of phi0/phi1
    anchors the low-x vs. high-x data, and k's sign/scale), keep the lowest-SSE fit, return
    its x0. x0 is bounded to the observed uplift range -- the turnover should sit within the
    data, not be extrapolated outside it."""
    def model(x, phi0, phi1, x0, k):
        return phi0 + (phi1 - phi0) / (1.0 + np.exp(-k * (x - x0)))

    x_lo, x_hi = float(np.min(x)), float(np.max(x))
    x_span = float(x_hi - x_lo) or 1.0
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    x0_mid = 0.5 * (x_lo + x_hi)

    best = None
    for phi0_0, phi1_0 in ((y_lo, y_hi), (y_hi, y_lo)):
        for k0 in (4 / x_span, -4 / x_span, 12 / x_span, -12 / x_span):
            p0 = [phi0_0, phi1_0, x0_mid, k0]
            bounds = ([-BOUND, -BOUND, x_lo, -BOUND], [BOUND, BOUND, x_hi, BOUND])
            try:
                popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
            except RuntimeError:
                continue
            y_pred = model(x, *popt)
            sse = float(np.sum((y_pred - y) ** 2))
            if best is None or sse < best[0]:
                best = (sse, popt)
    if best is None:
        return x0_mid
    return float(best[1][2])


def fit_station_vector(name, roll, inflation_roll, baseline_phi):
    """Shared fit step, factored out of make_page so other scripts (e.g. a spatial map of
    inflation-vector directions) can get a station's fitted (C1, C2, A, beta, beta_az, r, u0)
    plus its (merged, x, y, wrap) plot data without duplicating the load/roll/merge/fit
    pipeline. Returns a dict."""
    valid = roll.dropna(subset=['mean_phi']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    x = merged['inflation_m'].values
    y_raw = merged['mean_phi'].values

    wrap = compute_optimal_wrap(y_raw)
    y = (y_raw - wrap) % 180.0

    u0 = estimate_turnover_u0(x, y)
    C1, C2, A, beta, r, model = fit_atan2_vectorsum(x, y, u0)
    beta_az = _azimuth(beta)
    print(f'  {name}: u0={u0:.3f} (auto-estimated), C1={C1:.2f}, C2={C2:.2f}, A={A:.3f}, '
         f'background vector azimuth (alpha, fixed)={ALPHA_FIXED_AZ_DEG:.0f}°, inflation '
         f'vector azimuth (beta)={beta_az:.1f}°, r={r:.2f}, N={len(x)}')

    return dict(name=name, merged=merged, x=x, y=y, wrap=wrap, u0=u0, C1=C1, C2=C2, A=A,
                beta=beta, beta_az=beta_az, r=r, model=model)


def compute_all_station_fits():
    """Runs the full load -> filter -> roll -> merge-with-CCAL-uplift -> fit pipeline for every
    station in STATION_ORDER and returns {name: fit_dict} (see fit_station_vector). Reused by
    main() below and by other scripts that just need the fitted inflation-vector azimuths
    (beta_az) without re-deriving them by hand."""
    _dd, _infl_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()

    fits = {}
    for name in STATION_ORDER:
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

        fit = fit_station_vector(name, roll, inflation_roll, baseline_phi)
        fit['baseline_phi'] = baseline_phi
        fits[name] = fit
        print()
    return fits


def make_page(fit, flag_unleveled):
    name, merged, x, y = fit['name'], fit['merged'], fit['x'], fit['y']
    wrap, u0 = fit['wrap'], fit['u0']
    C1, C2, A, beta, r, model = fit['C1'], fit['C2'], fit['A'], fit['beta'], fit['r'], fit['model']
    beta_az = fit['beta_az']
    baseline_phi = fit['baseline_phi']
    baseline_plot = (baseline_phi - wrap) % 180.0

    fig, ax = plt.subplots(figsize=(7, 6))
    if flag_unleveled:
        is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
        ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2',
                  alpha=0.7, label='Post-eruption window')
        ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
                  marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')
    else:
        ax.scatter(x, y, s=18, color='#0072B2', alpha=0.7, label='Post-eruption window')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = model(x_line, C1, C2, A, beta)
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=r'$\phi(u_z)=C_1+C_2\mathrm{atan2}(Y,X)$'
                 r'$,\ Y=A\sin\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\sin\beta,\ $'
                 r'$X=A\cos\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\cos\beta$'
                 f'\n$C_1$={C1:.2f}, $C_2$={C2:.2f}, A={A:.3f} (all free)\n'
                 f'background vector azimuth α={ALPHA_FIXED_AZ_DEG:.0f}° (fixed), inflation '
                 f'vector azimuth β={beta_az:.1f}°\n(r = {r:.2f}, N = {len(x)})')
    ax.axvline(u0, color='gray', lw=0.6, linestyle=':',
              label=f'Turnover ($u_z$={u0:.3f}, auto-estimated)')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
    ax.set_ylabel(f'Avg fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)\n'
                  f'[axis wraps mod 180, optimal wrap={wrap:.1f}°]')
    y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(v) for v in tick_labels])
    ax.set_title(f'{name}: De-Tided Central Caldera Uplift vs. Fast Direction\n'
                 f'({GEODETIC_LABEL}, post-eruption, {ROLL_WINDOW_DAYS}-day rolling window, '
                 f'atan2 vector-sum fit)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    fits = compute_all_station_fits()

    figs = []
    for name in STATION_ORDER:
        flag_unleveled = name in UNLEVELED_STATIONS
        figs.append(make_page(fits[name], flag_unleveled))

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
