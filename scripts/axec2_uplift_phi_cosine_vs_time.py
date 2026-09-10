#!/usr/bin/env python3
"""
axec2_uplift_phi_cosine_vs_time.py

Three-panel time-series figure for AXEC2 only, per explicit user request: de-tided uplift,
average fast direction, and cosine similarity (to the pre-eruption baseline), all vs time, on
the same figure (stacked, shared x-axis).

Data: AXEC2 windowcheck grade-3 events (same GRADE/load_station_raw/apply_grade as
rose_7period_regions_windowcheck_grade3.py / rose_axec2_windowcheck_grade3_inflation_periods.py).

  A. De-tided seafloor uplift (m) -- reuses bpr_inflation_periods.load_daily_series() directly
     (same zero reference at the 2015-05-17 post-eruption low point, same 30-day rolling mean).
  B. Average fast direction (deg, circular mean +/- standard error) in ROLLING TIME WINDOWS
     (180-day window, 15-day step, skipped where fewer than MIN_N=15 events fall in the
     window -- AXEC2 has a real data gap in 2016-2017, see rose_axec2_windowcheck_grade3_
     inflation_periods.py's per-year event counts, so those windows are left as gaps rather
     than computed from too few events).
  C. Axial cosine similarity (cos(2*(mean_phi(t) - mean_phi_pre-eruption)), same convention as
     rose_7period_regions_windowcheck_grade3.py's _axial_cosine_similarity_deg) of each rolling
     window's mean fast direction relative to the FIXED pre-eruption ("Before Eruption") baseline
     mean, in the SAME rolling windows as panel B.
  D. Monthly seismicity rate (all grade-3 AXEC2 events/month), plotted as a short bar strip
     along the bottom of the same axes (own y-scale capped high so bars only occupy roughly the
     bottom 20% of the panel) -- lets you see whether/when seismicity picks up during the
     post-eruption re-inflation, per explicit user request.

All four are drawn on ONE shared axes (not stacked subplots), using multiple y-axes (left:
uplift; right: fast direction; outer right: cosine similarity; outermost right: seismicity
rate), color-coded to match each series.

Each rolling window (panel B/C) is centered on the MEAN TIMESTAMP of the events actually inside
it, not the nominal window midpoint -- near the end of the record a window is only partially
populated (events run out before the window closes), so a fixed geometric center would plot
points past where real data exists. The x-axis is then clipped to the overlap where BOTH the
uplift and phi/cosine series have real (non-NaN) values -- the BPR record has a genuine data
gap after 2026-02-25 (last valid tide+depth reading; see process_bpr_detided_depth.py), which
otherwise made the uplift line appear to end well before the phi/cosine lines.

The 2015 eruption window is marked on the shared axes.

Produces (NEW file):
    axec2_uplift_phi_cosine_vs_time.pdf

Run with:
    python3 axec2_uplift_phi_cosine_vs_time.py
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
from scipy.optimize import curve_fit

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg, _axial_cosine_similarity_deg,
)
from bpr_inflation_periods import (
    load_daily_series, build_inflation_based_periods, ERUPTION_START, ERUPTION_END,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'axec2_uplift_phi_cosine_vs_time.pdf')

STATION = 'AXEC2'
WINDOW_DAYS = 180
STEP_DAYS = 15
MIN_N = 15


def rolling_phi_stats(df, baseline_phi, window_days=WINDOW_DAYS, step_days=STEP_DAYS, min_n=MIN_N):
    t = df['t'].values
    phi = df['phi_az'].values
    dt_vals = df['dt'].values
    t_min, t_max = df['t'].min(), df['t'].max()

    centers = []
    means = []
    ses = []
    coss = []
    ns = []
    dt_means = []
    dt_ses = []

    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    t0 = t_min
    while t0 < t_max:
        t1 = t0 + window
        mask = (df['t'] >= t0) & (df['t'] < t1)
        n = int(mask.sum())
        # Center on the MEAN timestamp of events actually inside the window, not the nominal
        # t0+window/2 -- near the start/end of the record a window is only partly populated
        # (or, past the last event, populated only up to t_max), so a fixed geometric center
        # would plot the point well past where real data actually is. Using the data's own
        # mean time keeps every plotted point anchored inside the true data span.
        if n >= min_n:
            t_in_window = df['t'].values[mask.values]
            center = pd.Timestamp(np.array(t_in_window).astype('int64').mean(), tz='UTC')
            mean_phi, se_phi = _circular_mean_and_se_deg(phi[mask.values])
            cos_sim = _axial_cosine_similarity_deg(mean_phi, baseline_phi)
            dt_in_window = dt_vals[mask.values]
            dt_mean = float(np.mean(dt_in_window))
            dt_se = float(np.std(dt_in_window, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            centers.append(center)
            means.append(mean_phi)
            ses.append(se_phi)
            coss.append(cos_sim)
            ns.append(n)
            dt_means.append(dt_mean)
            dt_ses.append(dt_se)
        else:
            centers.append(t0 + window / 2)
            means.append(np.nan)
            ses.append(np.nan)
            coss.append(np.nan)
            ns.append(n)
            dt_means.append(np.nan)
            dt_ses.append(np.nan)
        t0 += step

    return pd.DataFrame({'t': centers, 'mean_phi': means, 'se_phi': ses,
                          'cos_sim': coss, 'n': ns, 'mean_dt': dt_means, 'se_dt': dt_ses})


def rolling_phi_stats_daily_then_roll(df, baseline_phi, window_days=90, min_days=None):
    """Two-stage averaging, per explicit user request: first collapse events into per-
    CALENDAR-DAY circular means (every day gets equal weight regardless of how many events
    occurred that day -- matching the geodetic uplift series, which is already a daily mean
    BEFORE any rolling smoothing is applied), then take a centered window_days-day rolling
    mean of those daily means (via the doubled-angle unit-vector trick, since phi is axial).
    This is a different estimator than rolling_phi_stats, which pools raw events directly
    within each window -- there, a day with more events implicitly outweighs a sparser day;
    here every day (with >=1 event) counts the same before smoothing. min_days (rolling
    min_periods, in days) defaults to max(3, window_days // 10). Returns a DataFrame with
    columns t, mean_phi, se_phi, cos_sim, n (summed raw event count in the window), n_days
    (number of days with data in the window), mean_dt."""
    d = df[['t', 'phi_az', 'dt']].copy()
    d['day'] = d['t'].dt.floor('D')
    ang = 2.0 * np.radians(d['phi_az'].values % 180.0)
    d['sin2'] = np.sin(ang)
    d['cos2'] = np.cos(ang)

    daily = d.groupby('day').agg(sin2=('sin2', 'mean'), cos2=('cos2', 'mean'),
                                 n=('phi_az', 'size'), dt_mean=('dt', 'mean'))
    daily.index.name = 't'

    if min_days is None:
        min_days = max(3, window_days // 10)

    win = f'{window_days}D'
    roll_sin = daily['sin2'].rolling(win, center=True, min_periods=min_days).mean()
    roll_cos = daily['cos2'].rolling(win, center=True, min_periods=min_days).mean()
    roll_n = daily['n'].rolling(win, center=True, min_periods=min_days).sum()
    roll_ndays = daily['n'].rolling(win, center=True, min_periods=min_days).count()
    roll_dt = daily['dt_mean'].rolling(win, center=True, min_periods=min_days).mean()

    r_roll = np.hypot(roll_sin.values, roll_cos.values)
    mean_phi = (np.degrees(np.arctan2(roll_sin.values, roll_cos.values)) / 2.0) % 180.0
    r_safe = np.clip(r_roll, 1e-12, 1.0)
    circ_std_deg = np.degrees(np.sqrt(-2.0 * np.log(r_safe))) / 2.0
    se_phi = circ_std_deg / np.sqrt(np.clip(roll_ndays.values, 1, None))
    cos_sim = np.cos(2.0 * np.radians(mean_phi - baseline_phi))

    return pd.DataFrame({'t': daily.index, 'mean_phi': mean_phi, 'se_phi': se_phi,
                          'cos_sim': cos_sim, 'n': roll_n.values, 'n_days': roll_ndays.values,
                          'mean_dt': roll_dt.values})


UNLEVEL_START = pd.Timestamp('2021-07-01', tz='UTC')
UNLEVEL_END = pd.Timestamp('2022-09-01', tz='UTC')


def make_uplift_vs_cosine_page(roll, inflation_roll):
    """Scatter of de-tided uplift vs. cosine similarity, post-eruption rolling windows only
    (same set the r/rho values quoted in conversation were computed from), + a cube-root line
    of fit (per explicit user request, replacing an earlier straight-line fit).
    Points inside the flagged 'seismometer potentially unleveled' window (2021-07 to 2022-09)
    are marked distinctly, since they may be an instrument artifact rather than a real
    de-correlation (per the excursion visible in axec2_uplift_phi_cosine_vs_time.pdf's page 1)."""
    valid = roll.dropna(subset=['mean_phi', 'cos_sim']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
    x = merged['inflation_m'].values
    y = merged['cos_sim'].values

    # Cube-root fit, shifted per explicit user request so the cube root is centered on
    # x=0.55 m uplift (not x=0): y = a*cbrt(x - 0.55) + b. np.cbrt (not x**(1/3)) is used
    # since it's defined for negative inputs too, needed here as x ranges below 0.55.
    X_SHIFT = 0.55
    x_cbrt = np.cbrt(x - X_SHIFT)
    slope, intercept = np.polyfit(x_cbrt, y, 1)
    r = np.corrcoef(x_cbrt, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2', alpha=0.7,
              label='Post-eruption window')
    ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
              marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * np.cbrt(x_line - X_SHIFT) + intercept
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Cube-root fit: cos = {slope:.2f}$\\cdot$(uplift$-${X_SHIFT})$^{{1/3}}$ + {intercept:.2f}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(0.0, color='gray', lw=0.6, linestyle=':')
    ax.set_xlabel('De-tided uplift (m, 30-day rolling mean)')
    ax.set_ylabel('Cosine similarity (vs. pre-eruption)')
    ax.set_title('AXEC2: De-Tided Uplift vs. Fast-Direction Cosine Similarity\n'
                 '(post-eruption rolling windows)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _cbrt_model(x, a, b, c):
    return a * np.cbrt(x - c) + b


def _axial_phi_diff_deg(phi_a_deg, phi_b_deg):
    """Smallest SIGNED difference between two axially-symmetric (mod 180) angles, in
    degrees, range [-90, 90) -- unlike cosine similarity, this retains the direction of
    rotation, not just how close the two are."""
    return ((phi_a_deg - phi_b_deg + 90.0) % 180.0) - 90.0


def make_uplift_diff_vs_cosine_page(roll, inflation_roll):
    """Page 3, per explicit user request: same as page 2 (post-eruption rolling-window
    uplift vs. cosine similarity, same flagged 'seismometer potentially unleveled' window
    marked distinctly), except the x-axis is the uplift DIFFERENCE FROM THE PRE-ERUPTION
    MAXIMUM (uplift(t) - max(uplift before ERUPTION_START), both on the 30-day rolling mean)
    instead of raw uplift -- so x=0 means "back to as inflated as the caldera ever got before
    the 2015 eruption", x<0 means still less inflated than that peak. The cube-root fit's
    x-shift (previously fixed at 0.55 m) is now a FREE parameter, fit via nonlinear least
    squares (scipy.optimize.curve_fit) on y = a*cbrt(x - c) + b."""
    pre_max = float(inflation_roll.loc[inflation_roll.index < ERUPTION_START].max())
    print(f'  Pre-eruption maximum uplift (30-day rolling mean): {pre_max:.4f} m')

    valid = roll.dropna(subset=['mean_phi', 'cos_sim']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
    x = merged['inflation_m'].values - pre_max
    y = merged['cos_sim'].values

    # Initial guess for c: in the original (unshifted-by-pre_max) x space the fixed shift used
    # in page 2 was 0.55 m; translated into this page's x = uplift - pre_max coordinate that's
    # 0.55 - pre_max.
    c0 = 0.55 - pre_max
    (a, b, c), _ = curve_fit(_cbrt_model, x, y, p0=[1.0, 0.0, c0], maxfev=5000)
    y_pred = _cbrt_model(x, a, b, c)
    r = np.corrcoef(y_pred, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2', alpha=0.7,
              label='Post-eruption window')
    ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
              marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = _cbrt_model(x_line, a, b, c)
    c_sign = '-' if c >= 0 else '+'
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Cube-root fit (free shift): cos = {a:.2f}$\\cdot$(diff {c_sign} {abs(c):.2f})$^{{1/3}}$ + {b:.2f}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(0.0, color='gray', lw=0.6, linestyle=':')
    ax.axvline(0.0, color='gray', lw=0.6, linestyle=':')
    ax.set_xlabel('Uplift $-$ pre-eruption maximum uplift (m, 30-day rolling mean)')
    ax.set_ylabel('Cosine similarity (vs. pre-eruption)')
    ax.set_title('AXEC2: Uplift Relative to Pre-Eruption Maximum vs. Cosine Similarity\n'
                 '(post-eruption rolling windows)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


PHIDIFF_X_SHIFT = 0.6   # fixed per explicit user request (was free-fit before)


def make_uplift_vs_phidiff_page(roll, inflation_roll):
    """Page 4, per explicit user request: same as page 2 (post-eruption rolling-window
    uplift vs. y, same flagged 'seismometer potentially unleveled' window marked distinctly),
    except y = the SIGNED angular difference from the pre-eruption baseline fast direction
    (degrees, _axial_phi_diff_deg) instead of cosine similarity -- retains the direction of
    rotation that cosine similarity discards. Cube-root fit x-shift fixed at 0.6 m (per
    explicit user request -- previously a free-fit parameter)."""
    valid = roll.dropna(subset=['mean_phi', 'cos_sim']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
    x = merged['inflation_m'].values
    y = merged['phi_diff'].values

    c = PHIDIFF_X_SHIFT
    x_cbrt = np.cbrt(x - c)
    a, b = np.polyfit(x_cbrt, y, 1)
    r = np.corrcoef(x_cbrt, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2', alpha=0.7,
              label='Post-eruption window')
    ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
              marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')

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
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


PHI_X_SHIFT = 0.6   # fixed per explicit user request (was free-fit before)


def make_uplift_vs_phi_page(roll, inflation_roll, baseline_phi):
    """Page 6, per explicit user request: replicates page 4, except y = the raw mean fast
    direction phi (degrees, axial 0-180) instead of the difference from the pre-eruption
    baseline -- cube-root fit, x-shift fixed at 0.6 (per explicit user request)."""
    valid = roll.dropna(subset=['mean_phi', 'cos_sim']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
    x = merged['inflation_m'].values
    y = merged['mean_phi'].values

    c = PHI_X_SHIFT
    x_cbrt = np.cbrt(x - c)
    a, b = np.polyfit(x_cbrt, y, 1)
    r = np.corrcoef(x_cbrt, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2', alpha=0.7,
              label='Post-eruption window')
    ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
              marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = _cbrt_model(x_line, a, b, c)
    c_sign = '-' if c >= 0 else '+'
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Cube-root fit (shift={c:.2f}): $\\phi$ = {a:.2f}$\\cdot$(uplift {c_sign} {abs(c):.2f})$^{{1/3}}$ + {b:.2f}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(baseline_phi, color='gray', lw=0.8, linestyle=':',
              label=f'Pre-eruption baseline ({baseline_phi:.0f}°)')
    ax.set_xlabel('De-tided uplift (m, 30-day rolling mean)')
    ax.set_ylabel(r'Avg fast direction $\phi$ (deg)')
    ax.set_ylim(0, 180)
    ax.set_title(f'{STATION}: De-Tided Uplift vs. Fast Direction\n'
                 '(post-eruption rolling windows)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_uplift_vs_dt_page(roll, inflation_roll):
    """Page 5, per explicit user request: de-tided uplift (x-axis) vs. delay time dt
    (y-axis), post-eruption rolling windows, same flagged 'seismometer potentially
    unleveled' window marked distinctly, + an inverse-cube fit: dt = a/uplift^3 + b (fit
    as OLS on the linearized x' = uplift^-3, same convention as the other pages' fits,
    using all points including flagged, per explicit user follow-up request)."""
    valid = roll.dropna(subset=['mean_dt']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']
    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])
    merged = merged[merged['inflation_m'] > 0]   # x^-3 undefined/blows up at x<=0

    is_flagged = (merged['t'] >= UNLEVEL_START) & (merged['t'] < UNLEVEL_END)
    x = merged['inflation_m'].values
    y = merged['mean_dt'].values

    x_inv3 = 1.0 / x ** 3
    a, b = np.polyfit(x_inv3, y, 1)
    r = np.corrcoef(x_inv3, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_flagged.values], y[~is_flagged.values], s=18, color='#0072B2', alpha=0.7,
              label='Post-eruption window')
    ax.scatter(x[is_flagged.values], y[is_flagged.values], s=22, color='#2ca02c', alpha=0.9,
              marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = a / x_line ** 3 + b
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Inverse-cube fit: dt = {a:.2e}$/$uplift$^3$ + {b:.4f}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.set_xlabel('De-tided uplift (m, 30-day rolling mean)')
    ax.set_ylabel('Delay time dt (s, rolling-window mean)')
    ax.set_title(f'{STATION}: De-Tided Uplift vs. Delay Time\n'
                 '(post-eruption rolling windows)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── Eruption-transition page: pre- through syn-eruption, fine time resolution ─────────────

ERUPTION_WINDOW_DAYS = 7   # much finer than the 180-day post-eruption window -- the
ERUPTION_STEP_DAYS = 1     # pre-to-syn-eruption transition itself only spans ~1 month
ERUPTION_MIN_N = 10
ERUPTION_TRANSITION_START = pd.Timestamp('2015-01-22', tz='UTC')   # catalog start


def rolling_phi_stats_fine(df, baseline_phi, t_start, t_end,
                          window_days=ERUPTION_WINDOW_DAYS, step_days=ERUPTION_STEP_DAYS,
                          min_n=ERUPTION_MIN_N):
    """Same idea as rolling_phi_stats() but a much narrower window/step, for resolving the
    ~1-month pre-to-syn-eruption transition rather than the multi-year post-eruption trend."""
    sub = df[(df['t'] >= t_start) & (df['t'] < t_end)]
    phi = sub['phi_az'].values

    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    centers, means, ses, phidiffs, ns = [], [], [], [], []
    t0 = t_start
    while t0 < t_end:
        t1 = t0 + window
        mask = (sub['t'] >= t0) & (sub['t'] < t1)
        n = int(mask.sum())
        if n >= min_n:
            t_in_window = sub['t'].values[mask.values]
            center = pd.Timestamp(np.array(t_in_window).astype('int64').mean(), tz='UTC')
            mean_phi, se_phi = _circular_mean_and_se_deg(phi[mask.values])
            phidiff = _axial_phi_diff_deg(mean_phi, baseline_phi)
            centers.append(center)
            means.append(mean_phi)
            ses.append(se_phi)
            phidiffs.append(phidiff)
            ns.append(n)
        else:
            centers.append(t0 + window / 2)
            means.append(np.nan)
            ses.append(np.nan)
            phidiffs.append(np.nan)
            ns.append(n)
        t0 += step

    return pd.DataFrame({'t': centers, 'mean_phi': means, 'se_phi': ses,
                          'phi_diff': phidiffs, 'n': ns})


def make_subsidence_vs_phidiff_page(df, baseline_phi):
    """Page 7, per explicit user request: fast-direction change (from the pre-eruption
    baseline) vs. de-tided SUBSIDENCE, spanning pre-eruption through syn-eruption (not the
    post-eruption re-inflation the other pages cover). Subsidence(t) = detided_depth_ref -
    detided_depth(t), referenced to the mean depth over a stable pre-onset window (2015-01-22
    to 2015-04-19, i.e. ending 5 days before ERUPTION_START) so subsidence ~0 before the
    eruption starts and grows positive as the seafloor drops during it. Cubic (degree-3
    polynomial) fit, per explicit user request -- NOT the cube-root fit used elsewhere in
    this script."""
    daily_depth, _inflation, _inflation_roll, _rt, _rd = load_daily_series()
    ref_window_end = ERUPTION_START - pd.Timedelta(days=5)
    ref_depth = float(daily_depth.loc[ERUPTION_TRANSITION_START:ref_window_end].mean())
    subsidence = ref_depth - daily_depth

    roll_fine = rolling_phi_stats_fine(df, baseline_phi, ERUPTION_TRANSITION_START, ERUPTION_END)
    valid = roll_fine.dropna(subset=['phi_diff'])

    subs_df = subsidence.reset_index()
    subs_df.columns = ['t', 'subsidence_m']
    merged = pd.merge_asof(valid.sort_values('t'), subs_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('3D'))
    merged = merged.dropna(subset=['subsidence_m'])

    is_syn = merged['t'] >= ERUPTION_START
    x = merged['subsidence_m'].values
    y = merged['phi_diff'].values

    coeffs = np.polyfit(x, y, 3)
    y_pred = np.polyval(coeffs, x)
    r = np.corrcoef(y_pred, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[~is_syn.values], y[~is_syn.values], s=20, color='#800080', alpha=0.7,
              label='Pre-eruption')
    ax.scatter(x[is_syn.values], y[is_syn.values], s=20, color='#CC0000', alpha=0.7,
              label='Syn-eruption')

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coeffs, x_line)
    a3, a2, a1, a0 = coeffs
    ax.plot(x_line, y_line, color='black', lw=1.5, linestyle='--',
           label=f'Cubic fit: $\\Delta\\phi$ = {a3:.2g}$x^3$ + {a2:.2g}$x^2$ + {a1:.2g}$x$ + {a0:.2g}\n'
                 f'(r = {r:.2f}, N = {len(x)})')

    ax.axhline(0.0, color='gray', lw=0.6, linestyle=':')
    ax.axvline(0.0, color='gray', lw=0.6, linestyle=':')
    ax.set_xlabel('De-tided subsidence (m, relative to pre-onset baseline)')
    ax.set_ylabel(r'$\Delta\phi$ vs. pre-eruption (deg)')
    ax.set_title(f'{STATION}: De-Tided Subsidence vs. Fast-Direction Change\n'
                 '(pre-eruption through syn-eruption, 7-day rolling windows)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
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
    ax_uplift.set_ylim(-0.25, 1.25)

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
    # Bars kept short (bottom ~20% of the panel) so they read as a rate strip along the
    # bottom, not a fourth full-height time series competing with the three lines. The ylim
    # is capped from the 95th percentile of POST-ERUPTION months only (per explicit user
    # request) -- the pre-eruptive swarm in Feb-Apr 2015 spikes to ~1800-5100 events/month, an
    # order of magnitude above every other month (typical post-eruption months run tens to a
    # few hundred); including it in the scaling calc would flatten the very re-inflation-era
    # pattern this panel exists to show, so those pre-eruption bars are intentionally clipped
    # at the top instead (the scale is chosen to resolve post-eruption only).
    post_counts = monthly_counts.loc[monthly_counts.index >= ERUPTION_END]
    seis_cap = post_counts.quantile(0.95) * 5
    ax_seis.set_ylim(0, seis_cap)
    n_clipped = int((monthly_counts > seis_cap).sum())
    if n_clipped:
        print(f'  {n_clipped} month(s) clipped off the top of the seismicity-rate bar '
              f'(pre-eruptive swarm, values > {seis_cap:.0f}/month)')

    ax_uplift.set_xlabel('Time')
    ax_uplift.set_title(f'{STATION}: De-Tided Uplift, Fast Direction, and Cosine Similarity vs. Time',
                        fontsize=12, fontweight='bold')

    periods = build_inflation_based_periods()
    period_starts = [t0 for _label, t0, _t1 in periods if t0 is not None]

    ax_uplift.axvspan(ERUPTION_START, ERUPTION_END, color='red', alpha=0.15)
    ax_uplift.axvspan(UNLEVEL_START, UNLEVEL_END,
                      color='lightgreen', alpha=0.35, label='Seismometer potentially unleveled')
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

    # x-range: 2015-01-01 (a buffer before the catalog starts 2015-01-22) through 2026-05-15
    # (the SWS catalog's end), per explicit user request -- phi/cosine and seismicity are
    # plotted through the full range regardless of the BPR record's shorter span; the uplift
    # line simply stops at its last real (non-NaN) value (2026-02-25 -- a genuine gap in the
    # BPR record past that date, see process_bpr_detided_depth.py) rather than being clipped
    # or extrapolated.
    ax_uplift.set_xlim(pd.Timestamp('2015-01-01', tz='UTC'), pd.Timestamp('2026-05-15', tz='UTC'))

    fig.tight_layout()

    print('\nBuilding page 2: uplift vs. cosine-similarity scatter + line of fit '
          '(post-eruption windows only, same set used for the correlation)...')
    fig2 = make_uplift_vs_cosine_page(roll, inflation_roll)

    print('\nBuilding page 3: uplift-relative-to-pre-eruption-max vs. cosine-similarity '
          'scatter + free-shift cube-root fit...')
    fig3 = make_uplift_diff_vs_cosine_page(roll, inflation_roll)

    print('\nBuilding page 4: uplift vs. fast-direction angular difference scatter + '
          'cube-root fit (free shift)...')
    fig4 = make_uplift_vs_phidiff_page(roll, inflation_roll)

    print('\nBuilding page 5: uplift vs. delay time (dt) scatter...')
    fig5 = make_uplift_vs_dt_page(roll, inflation_roll)

    print('\nBuilding page 6: uplift vs. fast direction scatter + cube-root fit (shift fixed at 0.6)...')
    fig6 = make_uplift_vs_phi_page(roll, inflation_roll, baseline_phi)

    print('\nBuilding page 7: de-tided subsidence vs. fast-direction change '
          '(pre- through syn-eruption) + cubic fit...')
    fig7 = make_subsidence_vs_phidiff_page(df, baseline_phi)

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
        pdf.savefig(fig2, dpi=200, bbox_inches='tight')
        pdf.savefig(fig3, dpi=200, bbox_inches='tight')
        pdf.savefig(fig4, dpi=200, bbox_inches='tight')
        pdf.savefig(fig5, dpi=200, bbox_inches='tight')
        pdf.savefig(fig6, dpi=200, bbox_inches='tight')
        pdf.savefig(fig7, dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
    plt.close(fig7)
    print(f'\nSaved {OUT_PDF} (7 pages)')


if __name__ == '__main__':
    main()
