#!/usr/bin/env python3
"""
erfc_physical_phi_vs_time_axec2_axcc1_axas1_90day.py

Time-domain fit of phi(t), derived by substituting an assumed exponential-relaxation uplift
kinematic

    u_z(t) = gamma - exp(-k*t)          (t = years since ERUPTION_END)

into a physical error-function fast-direction model:

    phi(u_z) = phi_0 + (phi_1 - phi_0)/2 * (1 + erf(alpha*(u_z - u_z0)))

which is the SAME shape family as the erfc form (erfc(z) = 1 - erf(z), so this is just erfc's
complement written with erf instead, with the same phi_0/phi_1 asymptotic roles preserved:
phi(-inf)=phi_0, phi(+inf)=phi_1) -- not a different curve, just a different special function
computing it. Substituting u_z(t) and DISTRIBUTING the single alpha into three INDEPENDENT
coefficients (a, b, c), one per additive term inside erf's argument, per a later explicit
user request:

    phi(t) = phi_0 + (phi_1 - phi_0)/2 * (1 + erf(a*gamma - b*exp(-k*t) - c*u_z0))

fit DIRECTLY against the observed phi(t) (same 90-day two-stage-binned series as
phi_vs_time_axec2_axcc1_axas1_90day.py). phi_0, phi_1, AND gamma are all FIXED to real data
values (not fit at all -- an earlier version of this script let curve_fit choose phi_0/phi_1
freely, which reproduced the same amplitude/coefficient degeneracy seen with the sigmoid
fit's A/B: an enormous (phi_1-phi_0) amplitude traded against a tiny alpha, blowing up the
legend, e.g. AXCC1's phi_0=-441 deg / phi_1=822 deg). NOTE: an intermediate attempt fixed
phi_0 at the PRE-eruption baseline instead of the real early-time POST-eruption plateau --
wrong, because phi jumps to a distinct post-eruption plateau immediately after the eruption
(e.g. AXEC2 opens near 160 deg, far from its ~96 deg pre-eruption baseline) before relaxing
toward a late-time value that happens to be close to (but is not assumed equal to) that
baseline. So both endpoints are read directly from the post-eruption record itself:
  - phi_0 = the plain mean of the wrap-converted phi(t) over the FIRST 365 days of the
    post-eruption record (the real early-time/pre-turnover plateau).
  - phi_1 = the plain mean of the wrap-converted phi(t) over the FINAL 365 days of the
    post-eruption record (the real late-time/post-turnover plateau).
  - gamma = the station's own observed maximum post-eruption de-tided uplift (the
    90-day-rolled inflation series' max over t >= ERUPTION_END).

IDENTIFIABILITY NOTE on (a, b, c): with gamma and u_z0 both FIXED constants, a*gamma and
c*u_z0 are each just a single number -- so a and c are NOT individually identifiable from
phi(t) alone; only the combination A = a*gamma - c*u_z0 (and b, k) actually shapes the curve.
This is the same class of degeneracy flagged for alpha/beta/gamma in an earlier version of
this script -- for AXAS1 specifically (u_z0 = 0), c*u_z0 = 0 regardless of c, so c is
COMPLETELY unconstrained there. a, b, c, k are still fit (per explicit user request, one
coefficient per term), and the printed a/c values are real curve_fit outputs, but treat them
as an arbitrary member of an equivalence class rather than uniquely-determined physical
constants -- b, k, and the combination a*gamma - c*u_z0 are the identifiable quantities.

Produces (3 pages, one per station):
    erfc_physical_phi_vs_time_axec2_axcc1_axas1_90day.pdf

Run with:
    python3 erfc_physical_phi_vs_time_axec2_axcc1_axas1_90day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erf
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
YEAR_DAYS = 365.25

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'erfc_physical_phi_vs_time_axec2_axcc1_axas1_90day.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9


def fit_phi_time_abc(tau, y, u0, gamma, phi0, phi1):
    """phi(tau) = phi0 + (phi1-phi0)/2*(1+erf(a*gamma - b*exp(-k*tau) - c*u0)), with
    (phi0, phi1, gamma, u0) all FIXED to real data values (see this script's docstring) and
    (a, b, c, k) free. a and c are NOT individually identifiable here (gamma, u0 both fixed
    constants -- only a*gamma - c*u0 matters; see docstring), so their initial-guess scale is
    seeded from 1/gamma, 1/u0 (a sign-only grid) while b (the only coefficient multiplying the
    time-varying exp(-k*tau) term) gets a real sign+scale grid; k constrained positive.
    Keeps the lowest-SSE multi-start result. Returns (a, b, c, k, r, model)."""
    def model(tau, a, b, c, k):
        return phi0 + (phi1 - phi0) / 2.0 * (1.0 + erf(a * gamma - b * np.exp(-k * tau) - c * u0))

    a0_mag = 1.0 / abs(gamma) if gamma else 1.0
    c0_mag = 1.0 / abs(u0) if u0 else 1.0
    b_scales = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0]
    k_candidates = [1 / 10, 1 / 5, 1 / 3, 1 / 2, 1.0, 2.0, 5.0, 10.0]

    best = None
    for a_sign in (1.0, -1.0):
        a0 = a_sign * a0_mag
        for c_sign in (1.0, -1.0):
            c0 = c_sign * c0_mag
            for b_sign in (1.0, -1.0):
                for b_scale in b_scales:
                    b0 = b_sign * b_scale
                    for k0 in k_candidates:
                        p0 = [np.clip(v, -BOUND, BOUND) for v in (a0, b0, c0, k0)]
                        bounds = ([-BOUND, -BOUND, -BOUND, 1e-6], [BOUND, BOUND, BOUND, 50.0])
                        try:
                            popt, _ = curve_fit(model, tau, y, p0=p0, bounds=bounds, maxfev=20000)
                        except RuntimeError:
                            continue
                        y_pred = model(tau, *popt)
                        sse = float(np.sum((y_pred - y) ** 2))
                        if best is None or sse < best[0]:
                            best = (sse, popt)
    a, b, c, k = best[1]
    y_pred = model(tau, a, b, c, k)
    r = np.corrcoef(y_pred, y)[0, 1]
    return a, b, c, k, r, model


def make_page(station, roll, inflation_roll, baseline_phi):
    valid = roll.dropna(subset=['mean_phi']).copy()
    valid_post = valid[valid['t'] >= ERUPTION_END]

    wrap = 90.0 if station['wrap_axis'] else 0.0
    u0 = station['u0']

    # gamma FIXED at this station's own observed max post-eruption de-tided uplift (the
    # 90-day-rolled inflation series' max over t >= ERUPTION_END) -- a real data value, not
    # a fit parameter.
    infl_post = inflation_roll.dropna()
    infl_post = infl_post[infl_post.index >= ERUPTION_END]
    gamma = float(infl_post.max())

    y_time_raw = valid_post['mean_phi'].values
    y_time = (y_time_raw - wrap) % 180.0
    t_time = valid_post['t']
    tau = (t_time - ERUPTION_END).dt.total_seconds().values / (YEAR_DAYS * 86400.0)

    baseline_plot = (baseline_phi - wrap) % 180.0

    # phi0, phi1 FIXED to real data (see docstring): the plain mean of the wrap-converted
    # phi(t) over the first/final 365 days of the post-eruption record -- the real early-time
    # and late-time plateaus (NOT the pre-eruption baseline -- phi jumps to a distinct
    # post-eruption plateau immediately after the eruption, well away from that baseline).
    head_mask = (t_time <= (t_time.min() + pd.Timedelta(days=365))).values
    tail_mask = (t_time >= (t_time.max() - pd.Timedelta(days=365))).values
    phi0 = float(np.mean(y_time[head_mask]))
    phi1 = float(np.mean(y_time[tail_mask]))

    a, b, c, k, r_t, model = fit_phi_time_abc(tau, y_time, u0, gamma, phi0, phi1)

    fig, ax = plt.subplots(figsize=(9, 5))
    if station['flag_unleveled']:
        is_flagged = (valid_post['t'] >= UNLEVEL_START) & (valid_post['t'] < UNLEVEL_END)
        ax.scatter(t_time.values[~is_flagged.values], y_time[~is_flagged.values], s=18,
                  color='#0072B2', alpha=0.7, label='Post-eruption window')
        ax.scatter(t_time.values[is_flagged.values], y_time[is_flagged.values], s=22,
                  color='#2ca02c', alpha=0.9, marker='^',
                  label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')
    else:
        ax.scatter(t_time.values, y_time, s=18, color='#0072B2', alpha=0.7,
                  label='Post-eruption window')

    tau_line = np.linspace(0.0, tau.max(), 300)
    y_line = model(tau_line, a, b, c, k)
    t_line = ERUPTION_END + pd.to_timedelta(tau_line * YEAR_DAYS, unit='D')
    ax.plot(t_line, y_line, color='black', lw=1.5, linestyle='--',
           label=r'$\phi(t)=\phi_0+\frac{\phi_1-\phi_0}{2}(1+\mathrm{erf}(a\gamma-be^{-kt}-cu_{z,0}))$'
                 f'\na={a:.3f}, b={b:.3f}, c={c:.3f} (a,c not individually identifiable), '
                 f'k={k:.3f}/yr (τ=1/k={1/k:.2f} yr)\n'
                 f'($\\phi_0$={phi0:.3f}°, $\\phi_1$={phi1:.3f}°, γ={gamma:.3f} m, '
                 f'$u_{{z,0}}$={u0:.3f} m, all fixed)\n(r = {r_t:.2f}, N = {len(tau)})')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel('Time')
    ylabel = f'Avg fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)'
    if station['wrap_axis']:
        ylabel += '\n[axis labels shifted 90° -- axial, mod 180]'
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 180)
    if station['wrap_axis']:
        ticks = np.arange(0, 181, 20)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int((tk + wrap) % 180)}' for tk in ticks])
    ax.set_title(f'{station["name"]}: Fast Direction vs. Time\n'
                 f'({station["geodetic_label"]}, post-eruption, {ROLL_WINDOW_DAYS}-day rolling window, '
                 f'erf(exp-relaxation) fit)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
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
