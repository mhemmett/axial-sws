#!/usr/bin/env python3
"""
atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.py

atan2 vector-sum counterpart of arctan_uplift_vs_phi_axec2_axcc1_axas1_90day.py, per explicit
user request: same data pipeline (AXEC2/AXCC1/AXAS1 post-eruption pages, same wrap/flag
conventions, same 90-day two-stage binning -- see erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py's
docstring), but the line of fit is now the two-argument atan2 of a vector sum:

    infl_mag(u) = (u-u0)/A - A*cos(alpha-beta)
    Y(u) = A*sin(alpha) + infl_mag(u)*sin(beta)
    X(u) = A*cos(alpha) + infl_mag(u)*cos(beta)
    phi(u_z) = C1 + C2*atan2(Y(u_z), X(u_z))

(u0 explained below -- an earlier version of this script used a bare u/A inflation term with
no u0 shift; see the TURNOVER note. C1/C2 explained below too -- an earlier version called
these phi_0/C1 respectively and fixed phi_0; per explicit user correction they are renamed
C1/C2 here and BOTH fit freely -- see the PARAMETERS note.)

This is the SAME background(A,alpha) + inflation(u/A,beta) vector-sum idea already used as an
ILLUSTRATIVE cartoon in animate_arctan_stress_vectors.py, and which arctan_uplift_vs_phi_*'s own
docstring documents an EARLIER attempt at fitting directly (a single-argument arctan of the
ratio Y/X) gave markedly worse fits than the plain scaled-arctan form (r as low as -0.59, or
degenerate alpha/theta hitting bounds). The key difference here, per explicit user request: this
uses the proper TWO-ARGUMENT atan2(Y,X), not a single-argument arctan of the ratio Y/X -- atan2
resolves the quadrant and spans the full (-pi,pi] range, whereas a bare arctan(Y/X) is quadrant-
blind and bounded to (-pi/2,pi/2), which is the specific defect the earlier attempt ran into.
Whether atan2's full range actually fixes the fit (rather than just avoiding that one specific
failure mode) is exactly what this script tests.

PARAMETERS, per explicit user request/correction (this went through two iterations -- see git
history of this docstring for the full back-and-forth):
  - alpha = 170 deg (compass azimuth, SAME fixed value for all three stations) --
    ALPHA_FIXED_AZ_DEG below. This is the ONLY truly fixed angle/offset parameter now.
  - C1 (renamed from an earlier "phi_0") and C2 (renamed from an earlier bare "C1", the
    atan2-scale amplitude) are BOTH now fit FREELY. An intermediate version of this script
    fixed C1 (then called phi_0) to the initial (first chronological) fast direction observed
    in the data -- per explicit user correction, this was wrong: it pins the curve through one
    single noisy 90-day-window data point instead of letting it be a genuine fitted plateau
    value (compare erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py's phi_0/phi_1, which
    are both free and land on the data's actual asymptotic plateaus) -- fixing it produced a
    fit that visibly failed to reach a flat plateau at all (AXAS1 was still descending steeply
    at the edge of its plotted domain, unlike the reference erfc fit's genuine S-curve).

TURNOVER, per explicit user request (this went through THREE iterations -- see git history):
the "turnover" must land at an EXACT, pre-specified absolute u_z for each station (u0 -- U_Z0
below: 0.6 AXEC2, 1.0 AXCC1, 0.0 AXAS1 "true zero"), and specifically must be where the curve
plotted on the page ACTUALLY visually turns -- not merely some algebraically-defined point
that happens to carry the name "turnover."
  1st version: fixed A=sqrt(u0) with a bare u/A inflation term (crossover of the two vectors'
    MAGNITUDES forced at u=A^2). Broke down for AXAS1 (u0=0 needs A=0, a division-by-zero
    singularity).
  2nd version (per explicit user correction): shift the inflation term by u0, using
    (u-u0+A^2)/A -- at u=u0 this evaluates to A^2/A=A, exactly matching the background's
    magnitude, for ANY nonzero A, fixing the AXAS1 singularity and freeing A entirely. BUT per
    explicit user correction, this was still wrong: the magnitude-EQUALITY point is not, in
    general, the same point as the curve's actual steepest/visually-apparent transition (they
    only coincide when the inflation direction is exactly antiparallel to the background). For
    AXAS1's fitted parameters the axvline sat at u=0 while the curve's real transition was
    near u=0.4 -- confirmed by directly computing the curve's numerical derivative.
  3rd version (current): infl_mag(u) = (u-u0)/A - A*cos(alpha-beta). Derivation: the (X,Y)
    position is A*[cos a,sin a] + infl_mag*[cos b,sin b], a point moving along a line as
    infl_mag varies; the curve's steepest slope occurs where that point is CLOSEST to the
    origin (standard result for angular velocity of a point on a line as seen from an external
    pivot), which calculus gives as infl_mag*=-A*cos(a-b) -- solving infl_mag(u0)=infl_mag* for
    the shift gives the form above. Verified numerically (multiple A/alpha/beta/u0
    combinations) that this places the curve's steepest point exactly at u0, for ANY A. Free
    parameters: (C1, C2, A, beta); u0 and alpha are the two truly fixed constants (per station
    and globally, respectively).

AXAS1: per explicit user request, the SAME "fit the second half, reconstruct the gap" trick
used by animate_arctan_stress_vectors_axas1_reconstructed.py and
erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day.py's make_axas1_reconstruction_page --
AXAS1's real ASHES BOTPT uplift record only starts in 2017 (well after the 2015 eruption), so
the fit is done on that real (post-2017) data only, then the fitted model is numerically
INVERTED (invert_atan2_vectorsum, analogous to solve_inflation_magnitude's root-finding, but
solving for u_z instead of an inflation magnitude) to reconstruct what u_z would have been
during 2015-2017, using the real SWS phi measurements that already exist for that period (the
windowcheck catalog has no gap, only the uplift record does). Reconstructed points plotted as
green diamonds alongside the real (blue) scatter.

A must stay strictly positive (it is also the divisor for u/A); bounded away from zero rather
than left free, to avoid a division blow-up during optimization.

Produces (3 pages, one per station):
    atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.pdf

Run with:
    python3 atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.py
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

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
    UNLEVEL_START, UNLEVEL_END,
)
from erfc_physical_uplift_vs_phi_axec2_axcc1_axas1_90day import STATIONS, UPLIFT_ROLLING_DAYS
# fit_atan2_vectorsum/invert_atan2_vectorsum/ALPHA_FIXED_RAD/ALPHA_FIXED_AZ_DEG/U_Z0/_azimuth
# now live in animate_arctan_stress_vectors.py (per explicit user request "use these functions
# to remake our gifs" -- moved there so both this PDF script and the animated GIF scripts
# share one copy instead of drifting independently).
from animate_arctan_stress_vectors import (
    compute_optimal_wrap, compute_full_range_ticks, fit_atan2_vectorsum, invert_atan2_vectorsum,
    ALPHA_FIXED_AZ_DEG, U_Z0, _azimuth,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.pdf')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9


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

    # Per-station optimal wrap (animate_arctan_stress_vectors.py's compute_optimal_wrap),
    # NOT the earlier fixed wrap=90-or-0 convention -- confirmed for AXAS1 that fixed wrap=90
    # splits its real data cluster across the plot boundary (largest true gap is at 51.6 deg,
    # not 90), which visually chops what is actually a smooth, continuous curve into two
    # disconnected-looking pieces -- i.e. the "doesn't look like an S-curve" symptom was a
    # display artifact, not a property of the fit itself.
    wrap = compute_optimal_wrap(y_raw)
    y = (y_raw - wrap) % 180.0
    baseline_plot = (baseline_phi - wrap) % 180.0

    name = station['name']
    u0 = U_Z0[name]
    C1, C2, A, beta, r, model = fit_atan2_vectorsum(x, y, u0)
    beta_az = _azimuth(beta)
    print(f'  {name}: C1={C1:.2f} (free), C2={C2:.2f} (free), A={A:.3f} (free), background '
         f'vector azimuth (alpha, fixed)={ALPHA_FIXED_AZ_DEG:.0f}°, inflation vector azimuth '
         f'(beta)={beta_az:.1f}°, turnover u_z={u0:.3f} (exact steepest point), r={r:.2f}, N={len(x)}')

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
    y_line = model(x_line, C1, C2, A, beta)
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=r'$\phi(u_z)=C_1+C_2\mathrm{atan2}(Y,X)$'
                 r'$,\ Y=A\sin\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\sin\beta,\ $'
                 r'$X=A\cos\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\cos\beta$'
                 f'\n$C_1$={C1:.2f}, $C_2$={C2:.2f}, A={A:.3f} (all free)\n'
                 f'background vector azimuth α={ALPHA_FIXED_AZ_DEG:.0f}° (fixed), inflation '
                 f'vector azimuth β={beta_az:.1f}°\n(r = {r:.2f}, N = {len(x)})')
    # Turnover: the curve's actual steepest/visually-apparent transition point -- exactly at
    # u_z=u0 by construction (see fit_atan2_vectorsum's docstring's TURNOVER note for why this
    # is NOT simply where the two vectors' magnitudes are equal, which is a different point in
    # general), pinned to the SAME u_z0 reference already used elsewhere in this script family.
    ax.axvline(u0, color='gray', lw=0.6, linestyle=':',
              label=f'Turnover ($u_z$={u0:.3f}, exact steepest point)')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
    ax.set_ylabel(f'Avg fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)\n'
                  f'[axis wraps mod 180, optimal wrap={wrap:.1f}°]')
    # compute_full_range_ticks (see animate_arctan_stress_vectors.py) builds monotonic,
    # correctly-valued tick labels by choosing round TRUE-degree labels first and mapping
    # each back to its wrapped plot position -- the earlier `(t+wrap)%180` relabeling of
    # evenly-spaced positions had its own unrelated discontinuity (see that script's history).
    y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(v) for v in tick_labels])
    ax.set_title(f'{station["name"]}: De-Tided Uplift vs. Fast Direction\n'
                 f'({station["geodetic_label"]}, post-eruption, {ROLL_WINDOW_DAYS}-day rolling window, '
                 f'atan2 vector-sum fit)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_axas1_reconstruction_page(station, roll, inflation_roll, baseline_phi):
    """AXAS1-only: fit on the real (post-2017) data only, then invert the fitted model to
    reconstruct 2015-2017 from the real SWS phi measurements that already exist for that gap
    (see module docstring)."""
    name = station['name']

    valid = roll.dropna(subset=['mean_phi']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    # Per-station optimal wrap, not the earlier fixed wrap=90 -- see make_page's comment (same
    # fix, same reason: fixed wrap=90 was splitting AXAS1's real data cluster across the plot
    # boundary, making a genuinely smooth curve look chopped/non-sigmoid).
    wrap = compute_optimal_wrap(merged['mean_phi'].values)
    x_real = merged['inflation_m'].values
    y_real = (merged['mean_phi'].values - wrap) % 180.0
    baseline_plot = (baseline_phi - wrap) % 180.0

    u0 = U_Z0[name]
    # A is fit FREELY (see fit_atan2_vectorsum's docstring/module TURNOVER note) -- the
    # (u-u0+A^2)/A inflation term guarantees the crossover lands exactly at u_z=u0=0 (AXAS1's
    # true turnover) for whatever A the fit lands on, avoiding the earlier A=sqrt(u0)=0
    # division-by-zero singularity. C1/C2 (see module docstring's PARAMETERS note) are also
    # both free now -- neither is pinned to a single data point.
    C1, C2, A, beta, r, model = fit_atan2_vectorsum(x_real, y_real, u0)
    beta_az = _azimuth(beta)
    print(f'  {name}: C1={C1:.2f} (free), C2={C2:.2f} (free), A={A:.3f} (free), background '
         f'vector azimuth (alpha, fixed)={ALPHA_FIXED_AZ_DEG:.0f}°, inflation vector azimuth '
         f'(beta)={beta_az:.1f}°, turnover u_z={u0:.3f} (exact steepest point), r={r:.2f}, N={len(x_real)}')

    # -- Gap reconstruction: real 2015-2017 SWS phi, inverted through the fit above. Per
    # explicit user request/assumption: the reconstructed AXAS1 domain is taken to begin at
    # u_z=-1.1 m (matching the earlier, independently-validated arctan-at-zero reconstruction
    # in animate_arctan_stress_vectors_axas1_reconstructed.py, which found a reconstructed
    # range of -1.097 to 0.967 m), with the turnover at u_z=0 (already exact -- see above).
    # u_bound=1.1 keeps the inversion search from wandering past that assumed physical extent
    # (an earlier, more permissive adaptive bound let it reach an implausible u_z=-3.08 m).
    u_bound = 1.1
    ashes_first_date = infl_df['t'].min()
    gap_mask = (valid['t'] >= ERUPTION_END) & (valid['t'] < ashes_first_date)
    gap = valid[gap_mask].sort_values('t')
    y_gap = (gap['mean_phi'].values - wrap) % 180.0
    x_recon = np.array([invert_atan2_vectorsum(yt, C1, C2, A, beta, u0, u_bound=u_bound)
                        for yt in y_gap])
    ok = ~np.isnan(x_recon)
    x_recon, y_recon = x_recon[ok], y_gap[ok]
    print(f'  Gap reconstruction: {len(gap)} real 2015-2017 SWS windows, {ok.sum()} invertible '
         f'(u_bound=±{u_bound:.2f} m)')
    if ok.sum():
        print(f'  Reconstructed u_z range: {x_recon.min():.3f} to {x_recon.max():.3f} m')

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_real, y_real, s=18, color='#0072B2', alpha=0.7, label='Real post-2017 data')
    ax.scatter(x_recon, y_recon, s=26, color='#2ca02c', alpha=0.85, marker='D',
              label='Reconstructed 2015-2017 (from real SWS $\\phi$)')

    u_min = float(min(x_real.min(), x_recon.min())) if len(x_recon) else float(x_real.min())
    u_max = float(x_real.max())
    x_line = np.linspace(u_min, u_max, 200)
    y_line = model(x_line, C1, C2, A, beta)
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=r'$\phi(u_z)=C_1+C_2\mathrm{atan2}(Y,X)$'
                 r'$,\ Y=A\sin\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\sin\beta,\ $'
                 r'$X=A\cos\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\cos\beta$'
                 f'\n$C_1$={C1:.2f}, $C_2$={C2:.2f}, A={A:.3f} (all free)\n'
                 f'background vector azimuth α={ALPHA_FIXED_AZ_DEG:.0f}° (fixed), inflation '
                 f'vector azimuth β={beta_az:.1f}°\n(r = {r:.2f}, N = {len(x_real)}, fit on real '
                 f'data only)')
    ax.axvline(u0, color='gray', lw=0.6, linestyle=':',
              label=f'Turnover ($u_z$={u0:.3f}, exact)')

    ax.axhline(baseline_plot, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel(f'De-tided uplift $u_z$ (m) -- real (2017 onward) or reconstructed (2015-2017)')
    ax.set_ylabel(f'Avg fast direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)\n'
                  f'[axis wraps mod 180, optimal wrap={wrap:.1f}°]')
    y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(v) for v in tick_labels])
    ax.set_title(f'{name}: De-Tided Uplift vs. Fast Direction\n'
                 f'({station["geodetic_label"]}, post-eruption, {ROLL_WINDOW_DAYS}-day rolling '
                 f'window, atan2 vector-sum fit, gap reconstructed)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
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

        page_fn = make_axas1_reconstruction_page if name == 'AXAS1' else make_page
        figs.append(page_fn(station, roll, inflation_roll, baseline_phi))
        print()

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
