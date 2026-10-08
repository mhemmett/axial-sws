#!/usr/bin/env python3
"""
phi_uplift_residual_periodogram_ccal.py

What periods are in the oscillations riding on the broad atan2 phi-vs-uplift fits?

For each regional average and each station (same pooled grade-3 events, AXEC2 unleveled-window
drop, and atan2 fits as uplift_vs_phi_regions_scatter_atan2fit_ccal_30day.py), this:

  1. takes DAILY circular-mean phi (post-eruption) -- NOT the 30-day rolled series, whose boxcar
     smoothing suppresses periods below ~30-60 d and can manufacture spectral structure;
  2. subtracts that series' atan2 fit evaluated at the day's 30-day CCAL uplift, so the broad
     trend is removed and only the oscillation is left (residual r, wrapped to [-90, 90]);
  3. computes a Lomb-Scargle periodogram (uneven sampling, gaps) of s = sin(2r). phi is axial,
     so sin(2r) is the wrap-safe variable; for small residuals s ~ 2r (rad);
  4. tests peaks against a RED-NOISE null, not white noise: N_SURR AR(1) surrogates with the
     residual's own lag-1 autocorrelation (estimated from consecutive-day pairs) and variance,
     simulated on the full daily grid and sampled on the observed days. A white-noise
     false-alarm level would be badly over-confident because adjacent days are correlated.
     Two thresholds: per-frequency 95% (pointwise) and the 95th percentile of each surrogate's
     MAXIMUM power over the band (global / family-wise -- the one to trust).

Peaks above the GLOBAL 95% level are reported with period and a sinusoid amplitude converted
back to degrees of phi. Light reference lines mark the annual (365.25 d) and semi-annual
(182.6 d) periods and the lunar fortnightly (Mf, 13.66 d) and monthly (Mm, 27.55 d) tides.

Caveat to keep in mind when reading an annual peak: the OOI cabled array is serviced on an
annual summer cruise (instrument swaps / relevelling -- cf. AXEC2's 2022-10-30 relevel), and
seismicity rate also varies, so an annual line is not by itself evidence of a geophysical
annual cycle.

Outputs:
    phi_uplift_residual_periodogram_ccal.pdf / .png
    phi_uplift_residual_periodogram_ccal_peaks.csv
"""

import os
import io
import contextlib
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bpr_inflation_periods_ccal import POST_ERUPTION_START
from scipy.signal import lombscargle, find_peaks

from axec2_uplift_phi_cosine_vs_time import ERUPTION_END
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from uplift_vs_phi_regions_scatter_ccal_30day import POINT_COLOR, STATION_COLORS
from uplift_vs_phi_regions_scatter_atan2fit_ccal_30day import region_fit
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'phi_uplift_residual_periodogram_ccal')

# (label, members, colour); panel order = 2 rows x 4
SERIES = [
    ('West average', ['AXAS1', 'AXAS2'], POINT_COLOR),
    ('AXAS1', ['AXAS1'], STATION_COLORS['AXAS1']),
    ('AXAS2', ['AXAS2'], STATION_COLORS['AXAS2']),
    ('Central (AXCC1)', ['AXCC1'], STATION_COLORS['AXCC1']),
    ('East average', ['AXEC1', 'AXEC2', 'AXEC3'], POINT_COLOR),
    ('AXEC1', ['AXEC1'], STATION_COLORS['AXEC1']),
    ('AXEC2', ['AXEC2'], STATION_COLORS['AXEC2']),
    ('AXEC3', ['AXEC3'], STATION_COLORS['AXEC3']),
]

P_MIN_D, P_MAX_D, N_FREQ = 5.0, 2000.0, 4000
N_SURR = 300
SEED = 20261008
REF_PERIODS = {365.25: 'annual', 182.62: 'semi-annual', 27.55: 'Mm', 13.66: 'Mf'}

FREQS = np.geomspace(1.0 / P_MAX_D, 1.0 / P_MIN_D, N_FREQ)   # cycles / day
OMEGA = 2.0 * np.pi * FREQS


def wrap90(a):
    return (np.asarray(a) + 90.0) % 180.0 - 90.0


def daily_residuals(members, fit, inflation_roll):
    """Daily circular-mean phi minus the atan2 fit at that day's uplift. Returns DataFrame
    (t, day_index, n, resid_deg)."""
    with contextlib.redirect_stdout(io.StringIO()):
        pool = load_region_pool(members)
    pool = pool[pool['t'] >= POST_ERUPTION_START]
    d = pool[['t', 'phi_az']].copy()
    d['day'] = d['t'].dt.floor('D')
    ang = 2.0 * np.radians(d['phi_az'].values % 180.0)
    d['s2'], d['c2'] = np.sin(ang), np.cos(ang)
    daily = d.groupby('day').agg(s2=('s2', 'mean'), c2=('c2', 'mean'), n=('phi_az', 'size'))
    daily['phi'] = (np.degrees(np.arctan2(daily['s2'], daily['c2'])) / 2.0) % 180.0
    daily = daily.reset_index().rename(columns={'day': 't'})

    infl = inflation_roll.dropna().reset_index()
    infl.columns = ['t', 'u']
    daily = pd.merge_asof(daily.sort_values('t'), infl.sort_values('t'), on='t',
                          direction='nearest', tolerance=pd.Timedelta('20D')).dropna(subset=['u'])

    pred_own = fit['model'](daily['u'].values, fit['C1'], fit['C2'], fit['A'], fit['beta'])
    pred_raw = pred_own + fit['wrap']          # fit lives in its own wrapped y-space
    daily['resid_deg'] = wrap90(daily['phi'].values - pred_raw)
    t0 = daily['t'].min()
    daily['day_index'] = ((daily['t'] - t0) / pd.Timedelta('1D')).round().astype(int)
    return daily


def ls_power(t, y):
    y = y - y.mean()
    return lombscargle(t.astype(float), y, OMEGA, normalize=True)


def ar1_surrogates(day_index, y, rng):
    """REDFIT-style null: AR(1) on the full daily grid with lag-1 autocorrelation from
    consecutive-day pairs, scaled to y's variance, sampled on the observed days."""
    s = pd.Series(y, index=day_index)
    nxt = s.reindex(s.index + 1)
    ok = ~np.isnan(nxt.values)
    rho = float(np.corrcoef(s.values[ok], nxt.values[ok])[0, 1]) if ok.sum() > 10 else 0.0
    rho = float(np.clip(rho, 0.0, 0.99))
    n_grid = int(day_index.max()) + 1
    powers = np.empty((N_SURR, N_FREQ))
    for k in range(N_SURR):
        e = rng.standard_normal(n_grid)
        x = np.empty(n_grid)
        x[0] = e[0]
        for i in range(1, n_grid):
            x[i] = rho * x[i - 1] + np.sqrt(1.0 - rho ** 2) * e[i]
        powers[k] = ls_power(day_index.astype(float), x[day_index])
    return rho, powers


def sinusoid_amp_deg(t, s, f):
    """Least-squares amplitude of a sinusoid at frequency f in s = sin(2r), as degrees of phi."""
    X = np.column_stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(X, s, rcond=None)
    amp_s = float(np.hypot(coef[0], coef[1]))
    return float(np.degrees(np.arcsin(min(amp_s, 1.0))) / 2.0)


def main():
    _dd, _ir_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    rows = []
    periods = 1.0 / FREQS
    for ax, (label, members, color) in zip(axes.ravel(), SERIES):
        with contextlib.redirect_stdout(io.StringIO()):
            fit = region_fit(label, members, inflation_roll)
        daily = daily_residuals(members, fit, inflation_roll)
        t = daily['day_index'].values
        s = np.sin(2.0 * np.radians(daily['resid_deg'].values))

        p = ls_power(t.astype(float), s)
        rho, surr = ar1_surrogates(t, s - s.mean(), rng)
        pw95 = np.percentile(surr, 95, axis=0)
        glob95 = float(np.percentile(surr.max(axis=1), 95))
        glob99 = float(np.percentile(surr.max(axis=1), 99))

        pk, _ = find_peaks(p)
        sig = [i for i in pk if p[i] > glob95]
        sig = sorted(sig, key=lambda i: -p[i])
        print(f'{label:16s} days={len(t):5d}  rho1={rho:.2f}  global95={glob95:.4f}  '
              f'significant peaks: ' +
              (', '.join(f'{periods[i]:.0f} d' for i in sig[:6]) or 'none'))
        for i in sig:
            rows.append(dict(series=label, period_days=periods[i], power=p[i],
                             global95=glob95, global99=glob99,
                             above_global99=bool(p[i] > glob99),
                             amp_deg=sinusoid_amp_deg(t.astype(float), s, FREQS[i]),
                             n_days=len(t), rho1=rho))

        for P in REF_PERIODS:
            ax.axvline(P, color='0.8', lw=0.8, zorder=0)
        ax.plot(periods, p, color=color, lw=1.2, zorder=2)
        ax.plot(periods, pw95, color='0.5', lw=0.8, ls='--', zorder=1)
        ax.axhline(glob95, color='0.15', lw=0.9, ls=':', zorder=1)
        for i in sig[:4]:
            ax.annotate(f'{periods[i]:.0f} d', (periods[i], p[i]), xytext=(0, 4),
                        textcoords='offset points', ha='center', fontsize=8, color='0.15')
        ax.set_xscale('log')
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left')
        ax.text(1.0, 1.01, f'{len(t)} days, AR(1) ρ = {rho:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='0.35')
        ax.grid(alpha=0.25, which='both')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel('Period (days)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Normalised Lomb–Scargle power')
    for ax in axes[0]:
        ymax = ax.get_ylim()[1]
        for P, name in REF_PERIODS.items():
            ax.text(P, ymax, name, rotation=90, ha='right', va='top', fontsize=7, color='0.55')

    handles = [plt.Line2D([], [], color='0.5', lw=0.8, ls='--'),
               plt.Line2D([], [], color='0.15', lw=0.9, ls=':'),
               plt.Line2D([], [], color='0.8', lw=0.8)]
    fig.legend(handles, ['Red-noise 95% (per frequency)', 'Red-noise 95% (global, across band)',
                         'Annual / semi-annual / Mm / Mf reference'],
               loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Periods in the φ residual about each atan2 fit (daily circular means, '
                 'post-eruption, sin 2r)', fontsize=15, fontweight='bold', y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    out = pd.DataFrame(rows)
    out.to_csv(f'{OUT_BASE}_peaks.csv', index=False)
    print(f'Saved {OUT_BASE}.pdf / .png and _peaks.csv ({len(out)} significant peaks)')


if __name__ == '__main__':
    main()
