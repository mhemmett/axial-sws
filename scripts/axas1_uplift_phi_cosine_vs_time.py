#!/usr/bin/env python3
"""
axas1_uplift_phi_cosine_vs_time.py

AXAS1 / ASHES vent field BOTPT (RS03ASHS-MJ03B-09-BOTPTA304, Western Caldera) counterpart of
axec2_uplift_phi_cosine_vs_time.py -- same structure and methodology, repeated for this
station/BPR pair per explicit user request. See that script's docstring for the full
per-element rationale (rolling-window centering fix, seismicity-bar percentile capping, common
x-range handling, etc.), not repeated in full here. Differences from the AXEC2 version:

  - Data: AXAS1 windowcheck grade-3 events + bpr_inflation_periods_ashes (ASHES vent field
    BOTPT, not the Eastern Caldera one AXEC2's script uses).
  - No "seismometer potentially unleveled" window is assumed here -- that flag was specific to
    an excursion observed in AXEC2's own cosine-similarity curve (2021-07 to 2022-09); whether
    AXAS1 shows a comparable anomaly has NOT been established, so this script does not shade or
    flag any window.
  - The cube-root fit's x-shift (page 2) is FIXED at 0 per explicit user request.
  - ONLY TWO PAGES, not three: the AXCC1/AXEC2 scripts' page 3 (uplift relative to the
    PRE-ERUPTION MAXIMUM uplift) requires BPR data from before the 2015 eruption, which this
    instrument doesn't have at all (record starts 2017-08-15, over 2 years after the eruption
    -- see bpr_inflation_periods_ashes.py's docstring). That reference point cannot be computed
    here, so page 3 is omitted rather than faked from a different quantity.

Three pages:
  1. Shared-axes time series: de-tided uplift, average fast direction, cosine similarity (vs.
     pre-eruption baseline, computed from the SPLITTING CATALOG's own pre-eruption AXAS1
     events -- independent of the BPR record's start date), and monthly seismicity rate
     (bottom bar strip).
  2. Post-eruption rolling-window scatter: uplift vs. cosine similarity, cube-root fit
     (x-shift fixed at 0, per explicit user request).
  3. Same scatter, y-axis = signed angular difference from the pre-eruption baseline fast
     direction (degrees) instead of cosine similarity, cube-root fit (x-shift fixed at 0).

Produces (NEW file):
    axas1_uplift_phi_cosine_vs_time.pdf (3 pages)

Run with:
    python3 axas1_uplift_phi_cosine_vs_time.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg, _axial_cosine_similarity_deg,
)
from bpr_inflation_periods_ashes import (
    load_daily_series, build_inflation_based_periods, ERUPTION_START, ERUPTION_END,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'axas1_uplift_phi_cosine_vs_time.pdf')

STATION = 'AXAS1'
WINDOW_DAYS = 180
STEP_DAYS = 15
MIN_N = 15

# AXAS1 has a genuine ~58-day raw-data gap (no events at all -- confirmed by inspecting the
# raw event timestamps' inter-arrival times: the largest gap in 2015 is 2015-03-01 09:00 to
# 2015-04-28 17:06, 58 days, right up against the eruption onset), per explicit user
# reminder. A 180-day rolling window straddling this gap can still clear MIN_N purely from
# events before/after it (e.g. February 2015 alone has 564 events), and the window's
# MEAN-EVENT-TIME center can land INSIDE the gap itself (confirmed: windows centered
# 2015-03-12 and 2015-04-08, each with hundreds of events, none of which actually occurred in
# March/April) -- misleadingly implying continuous data through a period the station wasn't
# recording. Any window whose center falls inside the gap is blanked (forced to NaN) below.
GAP_START = pd.Timestamp('2015-03-01', tz='UTC')
GAP_END = pd.Timestamp('2015-04-28', tz='UTC')


def rolling_phi_stats(df, baseline_phi):
    t = df['t'].values
    phi = df['phi_az'].values
    t_min, t_max = df['t'].min(), df['t'].max()

    centers = []
    means = []
    ses = []
    coss = []
    ns = []

    window = pd.Timedelta(days=WINDOW_DAYS)
    step = pd.Timedelta(days=STEP_DAYS)
    t0 = t_min
    while t0 < t_max:
        t1 = t0 + window
        mask = (df['t'] >= t0) & (df['t'] < t1)
        n = int(mask.sum())
        if n >= MIN_N:
            t_in_window = df['t'].values[mask.values]
            center = pd.Timestamp(np.array(t_in_window).astype('int64').mean(), tz='UTC')
            if GAP_START <= center < GAP_END:
                centers.append(center)
                means.append(np.nan)
                ses.append(np.nan)
                coss.append(np.nan)
                ns.append(0)
            else:
                mean_phi, se_phi = _circular_mean_and_se_deg(phi[mask.values])
                cos_sim = _axial_cosine_similarity_deg(mean_phi, baseline_phi)
                centers.append(center)
                means.append(mean_phi)
                ses.append(se_phi)
                coss.append(cos_sim)
                ns.append(n)
        else:
            centers.append(t0 + window / 2)
            means.append(np.nan)
            ses.append(np.nan)
            coss.append(np.nan)
            ns.append(n)
        t0 += step

    return pd.DataFrame({'t': centers, 'mean_phi': means, 'se_phi': ses,
                          'cos_sim': coss, 'n': ns})


def _cbrt_model(x, a, b, c):
    return a * np.cbrt(x - c) + b


def _axial_phi_diff_deg(phi_a_deg, phi_b_deg):
    """Smallest SIGNED difference between two axially-symmetric (mod 180) angles, in
    degrees, range [-90, 90) -- unlike cosine similarity, this retains the direction of
    rotation, not just how close the two are."""
    return ((phi_a_deg - phi_b_deg + 90.0) % 180.0) - 90.0


X_SHIFT = 0.0   # fixed per explicit user request (was free-fit before)


def make_uplift_vs_cosine_page(roll, inflation_roll):
    """Post-eruption rolling-window uplift vs. cosine similarity + cube-root fit, x-shift
    fixed at 0 (per explicit user request -- previously a free-fit parameter)."""
    valid = roll.dropna(subset=['mean_phi', 'cos_sim']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    x = merged['inflation_m'].values
    y = merged['cos_sim'].values
    c = X_SHIFT
    x_cbrt = np.cbrt(x - c)
    a, b = np.polyfit(x_cbrt, y, 1)
    r = np.corrcoef(x_cbrt, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=18, color='#0072B2', alpha=0.7, label='Post-eruption window')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = _cbrt_model(x_line, a, b, c)
    c_sign = '-' if c >= 0 else '+'
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Cube-root fit (shift={c:.2f}): cos = {a:.2f}$\\cdot$(uplift {c_sign} {abs(c):.2f})$^{{1/3}}$ + {b:.2f}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(0.0, color='gray', lw=0.6, linestyle=':')
    ax.set_xlabel('De-tided uplift (m, 30-day rolling mean)')
    ax.set_ylabel('Cosine similarity (vs. pre-eruption)')
    ax.set_title(f'{STATION}: De-Tided Uplift vs. Fast-Direction Cosine Similarity\n'
                 '(post-eruption rolling windows)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_uplift_vs_phidiff_page(roll, inflation_roll):
    """Page 3, per explicit user request: same as page 2, except y = the SIGNED angular
    difference from the pre-eruption baseline fast direction (degrees, _axial_phi_diff_deg)
    instead of cosine similarity. Cube-root fit x-shift fixed at 0 (per explicit user
    request, matching page 2)."""
    valid = roll.dropna(subset=['mean_phi', 'cos_sim']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    x = merged['inflation_m'].values
    y = merged['phi_diff'].values
    c = X_SHIFT
    x_cbrt = np.cbrt(x - c)
    a, b = np.polyfit(x_cbrt, y, 1)
    r = np.corrcoef(x_cbrt, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=18, color='#0072B2', alpha=0.7, label='Post-eruption window')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = _cbrt_model(x_line, a, b, c)
    c_sign = '-' if c >= 0 else '+'
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Cube-root fit (shift={c:.2f}): $\\Delta\\phi$ = {a:.2f}$\\cdot$(uplift {c_sign} {abs(c):.2f})$^{{1/3}}$ + {b:.2f}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(0.0, color='gray', lw=0.6, linestyle=':')
    ax.set_xlabel('De-tided uplift (m, 30-day rolling mean)')
    ax.set_ylabel(r'$\Delta\phi$ vs. pre-eruption (deg)')
    ax.set_title(f'{STATION}: De-Tided Uplift vs. Fast-Direction Angular Difference\n'
                 '(post-eruption rolling windows)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    print(f'Loading {STATION} windowcheck grade-3 events ({GRADE["label"]})...')
    raw = load_station_raw(STATION)
    df = apply_grade(raw, GRADE)
    print(f'  {STATION}: {len(df):,} events pass the filter')

    pre = df[df['t'] < ERUPTION_START]
    baseline_phi, baseline_se = _circular_mean_and_se_deg(pre['phi_az'].values)
    print(f'  Pre-eruption baseline: N={len(pre):,}, mean phi={baseline_phi:.1f}+/-{baseline_se:.1f} deg')

    print(f'Computing rolling ({WINDOW_DAYS}-day window, {STEP_DAYS}-day step, MIN_N={MIN_N}) '
          f'phi stats...')
    roll = rolling_phi_stats(df, baseline_phi)
    roll['phi_diff'] = _axial_phi_diff_deg(roll['mean_phi'].values, baseline_phi)
    n_valid = roll['mean_phi'].notna().sum()
    print(f'  {n_valid}/{len(roll)} windows have >= {MIN_N} events')

    daily_depth, inflation, inflation_roll, ref_time, ref_depth = load_daily_series()

    print('Computing monthly seismicity rate (all grade-3 events, for the bottom bar)...')
    monthly_counts = df.set_index('t')['phi_az'].resample('MS').size()

    UPLIFT_COLOR = '#0072B2'
    PHI_COLOR = '#CC0000'
    COS_COLOR = '#800080'
    SEIS_COLOR = '#555555'

    fig, ax_uplift = plt.subplots(figsize=(12, 6))
    ax_phi = ax_uplift.twinx()
    ax_cos = ax_uplift.twinx()
    ax_seis = ax_uplift.twinx()
    ax_cos.spines['right'].set_position(('outward', 60))
    ax_seis.spines['right'].set_position(('outward', 120))

    ax_uplift.plot(inflation.index, inflation.values, color=UPLIFT_COLOR, lw=0.5, alpha=0.3)
    ax_uplift.plot(inflation_roll.index, inflation_roll.values, color=UPLIFT_COLOR, lw=1.8,
                  label='De-tided uplift (30-day rolling mean)')
    ax_uplift.set_ylabel('De-tided uplift (m)', color=UPLIFT_COLOR)
    ax_uplift.tick_params(axis='y', labelcolor=UPLIFT_COLOR)
    ax_uplift.spines['left'].set_color(UPLIFT_COLOR)
    # ASHES re-inflation only reaches ~0.9 m (smaller than AXEC2's ~1.1 m or AXCC1's ~2.6 m --
    # intermediate amplitude for a peripheral vent-field site, see plot_bpr_detided_depth_and_
    # inflation_ashes.py's docstring), so scale the axis to what's actually there.
    y_max = float(inflation_roll.max()) * 1.1
    ax_uplift.set_ylim(-0.25, y_max)

    ax_phi.plot(roll['t'], roll['mean_phi'], color=PHI_COLOR, lw=1.2, marker='o', ms=2.5,
               label=r'Avg fast direction $\phi$')
    lo = roll['mean_phi'] - roll['se_phi']
    hi = roll['mean_phi'] + roll['se_phi']
    ax_phi.fill_between(roll['t'], lo, hi, color=PHI_COLOR, alpha=0.15)
    ax_phi.axhline(baseline_phi, color=PHI_COLOR, lw=0.8, linestyle=':', alpha=0.6,
                  label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax_phi.set_ylabel(r'Avg fast direction $\phi$ (deg)', color=PHI_COLOR)
    ax_phi.tick_params(axis='y', labelcolor=PHI_COLOR)
    ax_phi.spines['right'].set_color(PHI_COLOR)
    ax_phi.set_ylim(0, 180)

    ax_cos.plot(roll['t'], roll['cos_sim'], color=COS_COLOR, lw=1.2, marker='o', ms=2.5,
               label='Cosine similarity (vs. pre-eruption)')
    ax_cos.set_ylabel('Cosine similarity (vs. pre-eruption)', color=COS_COLOR)
    ax_cos.tick_params(axis='y', labelcolor=COS_COLOR)
    ax_cos.spines['right'].set_color(COS_COLOR)
    ax_cos.set_ylim(-1.05, 1.05)

    ax_seis.bar(monthly_counts.index, monthly_counts.values,
               width=25, color=SEIS_COLOR, alpha=0.4, align='edge', zorder=1,
               label='Seismicity rate (events/month)')
    ax_seis.set_ylabel('Events/month', color=SEIS_COLOR, fontsize=8)
    ax_seis.tick_params(axis='y', labelcolor=SEIS_COLOR, labelsize=7)
    ax_seis.spines['right'].set_color(SEIS_COLOR)
    post_counts = monthly_counts.loc[monthly_counts.index >= ERUPTION_END]
    seis_cap = post_counts.quantile(0.95) * 5
    ax_seis.set_ylim(0, seis_cap)
    n_clipped = int((monthly_counts > seis_cap).sum())
    if n_clipped:
        print(f'  {n_clipped} month(s) clipped off the top of the seismicity-rate bar '
              f'(values > {seis_cap:.0f}/month)')

    ax_uplift.set_xlabel('Time')
    ax_uplift.set_title(f'{STATION}: De-Tided Uplift, Fast Direction, and Cosine Similarity vs. Time',
                        fontsize=12, fontweight='bold')

    periods = build_inflation_based_periods()
    period_starts = [t0 for _label, t0, _t1 in periods if t0 is not None]

    ax_uplift.axvspan(ERUPTION_START, ERUPTION_END, color='red', alpha=0.15)
    for t0 in period_starts:
        ax_uplift.axvline(t0, color='gray', lw=0.7, linestyle='--', alpha=0.7)
    ax_uplift.grid(alpha=0.3)

    h1, l1 = ax_uplift.get_legend_handles_labels()
    h2, l2 = ax_phi.get_legend_handles_labels()
    h3, l3 = ax_cos.get_legend_handles_labels()
    h4, l4 = ax_seis.get_legend_handles_labels()
    fig.legend(h1 + h2 + h3 + h4, l1 + l2 + l3 + l4, loc='lower center', ncol=3,
              fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    ax_uplift.xaxis.set_major_locator(mdates.YearLocator())
    ax_uplift.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_uplift.set_xlim(pd.Timestamp('2015-01-01', tz='UTC'), pd.Timestamp('2026-05-15', tz='UTC'))

    fig.tight_layout()

    print('\nBuilding page 2: uplift vs. cosine-similarity scatter + cube-root fit '
          '(shift fixed at 0, post-eruption windows only)...')
    fig2 = make_uplift_vs_cosine_page(roll, inflation_roll)

    print('\nBuilding page 3: uplift vs. fast-direction angular difference scatter + '
          'cube-root fit (shift fixed at 0)...')
    fig3 = make_uplift_vs_phidiff_page(roll, inflation_roll)

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
        pdf.savefig(fig2, dpi=200, bbox_inches='tight')
        pdf.savefig(fig3, dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)
    print(f'\nSaved {OUT_PDF} (3 pages)')


if __name__ == '__main__':
    main()
