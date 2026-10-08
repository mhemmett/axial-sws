#!/usr/bin/env python3
"""
station_correlation_matrices.py

Correlation matrices for phi and dt, every station against every other station and against
the geodetic (BOTPT seafloor uplift) records.

WHAT IS CORRELATED WITH WHAT, AND HOW
    phi is AXIAL and circular (mod 180), dt and uplift are linear, so three different
    coefficients are used and they are NOT interchangeable:

      phi <-> phi      Jammalamadaka-Sarma circular-circular correlation on DOUBLED angles
                       (theta = 2*phi, which is what makes an axial quantity behave like a
                       circular one). Signed, in [-1, 1].
      dt  <-> dt       ordinary Pearson r. Signed, in [-1, 1].
      dt  <-> uplift   ordinary Pearson r. Signed, in [-1, 1].
      phi <-> dt       circular-linear correlation R. UNSIGNED, in [0, 1] -- there is no
      phi <-> uplift   meaningful sign for "how strongly does a circular quantity track a
                       linear one", so these panels use a sequential scale, not a diverging
                       one, and must not be read as if a high value meant "positive".

    Using Pearson on raw phi would be wrong: it would treat 179 deg and 1 deg as maximally
    different when for an axial quantity they are 2 deg apart.

THE AUTOCORRELATION PROBLEM (same as the model comparison)
    These are DAILY samples of a 30-DAY ROLLING MEAN, so consecutive points share 29/30 of
    their input. Correlations computed on them are not wrong, but their SIGNIFICANCE is
    massively overstated -- an effective sample size of order N/30 is closer to the truth.
    Every page therefore reports its own effective N, and a THINNED page (one point per 30
    days, ~decorrelated) is produced alongside the raw one. Where raw and thinned disagree,
    believe the thinned page.

    Pairs are computed on pairwise-complete observations, so N differs cell to cell; the
    per-page caption reports the median and minimum.

DATA HANDLING
    * grade-3 filter throughout (imported, not redefined).
    * AXEC2's 2021-07 to 2022-09 unlevelled window is DROPPED, as in the regional figures --
      its ~50 deg instrumental drift would otherwise show up as real covariation with
      anything else that happens to trend over that window.
    * AXCC1 has no real data 2015-03-01 to 2015-04-28; those days are simply absent and
      pairwise-complete handles them.

Produces:
    station_correlation_matrices.pdf   (3 pages: post-eruption raw / post-eruption thinned /
                                        full record thinned)
    station_correlation_matrices.csv   (every cell, long format)

Run with:
    python3 station_correlation_matrices.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bpr_inflation_periods_ccal import POST_ERUPTION_START
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, STATION_ORDER, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
    UNLEVEL_START, UNLEVEL_END,
)
from atan2_uplift_vs_phi_sixstations_ccal_30day import ROLL_WINDOW_DAYS, ROLL_MIN_DAYS
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'station_correlation_matrices.pdf')
OUT_CSV = os.path.join(HERE, 'station_correlation_matrices.csv')

THIN_DAYS = ROLL_WINDOW_DAYS
DROP_UNLEVELED = {'AXEC2'}

# Diverging: blue <-> red with a NEUTRAL GRAY midpoint (never a hue at the midpoint, never a
# rainbow). Anchored symmetrically at 0 so the gray always means "no correlation".
DIVERGING = LinearSegmentedColormap.from_list(
    'corr_div', ['#1a4f9c', '#5b8fd4', '#a8c4e8', '#f0efec', '#eeb0a0', '#d4735c', '#a8321c'])
# Sequential single hue for the unsigned circular-linear panels.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'corr_seq', ['#f0efec', '#c3d4ea', '#7fa5d4', '#3f6fb5', '#12376e'])

INK = '#0b0b0b'
INK2 = '#52514e'


# ── correlation coefficients ────────────────────────────────────────────────────────────────

def circ_circ_corr(a_deg, b_deg):
    """Jammalamadaka-Sarma circular correlation on doubled (axial) angles."""
    ta, tb = 2 * np.radians(a_deg), 2 * np.radians(b_deg)
    ma = np.arctan2(np.mean(np.sin(ta)), np.mean(np.cos(ta)))
    mb = np.arctan2(np.mean(np.sin(tb)), np.mean(np.cos(tb)))
    sa, sb = np.sin(ta - ma), np.sin(tb - mb)
    den = np.sqrt(np.sum(sa**2) * np.sum(sb**2))
    return float(np.sum(sa * sb) / den) if den > 0 else np.nan


def circ_lin_corr(theta_deg, x):
    """Circular-linear correlation R in [0,1]. UNSIGNED by construction."""
    t = 2 * np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    def pear(u, v):
        if np.std(u) == 0 or np.std(v) == 0:
            return 0.0
        return float(np.corrcoef(u, v)[0, 1])
    rxc, rxs, rcs = pear(x, c), pear(x, s), pear(c, s)
    den = 1 - rcs**2
    if den <= 0:
        return np.nan
    r2 = (rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / den
    return float(np.sqrt(max(r2, 0.0)))


def pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def eff_n(a, b):
    """Effective sample size from lag-1 autocorrelation of both series (Bartlett-style)."""
    def rho1(v):
        v0 = v - np.mean(v)
        den = np.sum(v0 * v0)
        return float(np.sum(v0[:-1] * v0[1:]) / den) if den > 0 else 0.0
    ra, rb = rho1(np.asarray(a, float)), rho1(np.asarray(b, float))
    ra, rb = min(max(ra, -0.99), 0.99), min(max(rb, -0.99), 0.99)
    n = len(a)
    return max(n * (1 - ra * rb) / (1 + ra * rb), 2.0)


# ── data assembly ───────────────────────────────────────────────────────────────────────────

def build_series():
    """Daily-indexed rolling phi and dt per station, plus the geodetic uplift columns."""
    cols = {}
    for sta in STATION_ORDER:
        df = apply_grade(load_station_raw(sta), GRADE)
        if sta in DROP_UNLEVELED:
            n0 = len(df)
            df = df[~((df['t'] >= UNLEVEL_START) & (df['t'] < UNLEVEL_END))]
            print(f'  {sta}: {n0:,} -> {len(df):,} after dropping the unlevelled window')
        else:
            print(f'  {sta}: {len(df):,} grade-3 measurements')

        pre = df[df['t'] < ERUPTION_START]
        baseline, _ = _circular_mean_and_se_deg(pre['phi_az'].values)
        roll = rolling_phi_stats_daily_then_roll(df, baseline, window_days=ROLL_WINDOW_DAYS,
                                                 min_days=ROLL_MIN_DAYS)
        roll = roll.set_index('t') if 't' in roll.columns else roll
        cols[f'{sta} phi'] = roll['mean_phi']
        cols[f'{sta} dt'] = roll['mean_dt']

    frame = pd.DataFrame(cols)
    frame.index = pd.to_datetime(frame.index, utc=True)

    _dd, _raw, infl_roll, _rt, _rd = ccal_infl.load_daily_series()
    g = infl_roll.dropna()
    g.index = pd.to_datetime(g.index, utc=True)
    frame['CCAL uplift'] = g.reindex(frame.index, method='nearest',
                                     tolerance=pd.Timedelta('5D'))
    return frame


def corr_block(frame, rows, cols, kind):
    """Pairwise-complete correlation block. kind: 'cc' | 'cl' | 'pp'."""
    R = np.full((len(rows), len(cols)), np.nan)
    N = np.full((len(rows), len(cols)), np.nan)
    NE = np.full((len(rows), len(cols)), np.nan)
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            if a == b:
                # Self-correlation. Selecting frame[[a, a]] yields duplicate column names,
                # so sub[a] comes back as a DataFrame and the coefficient silently goes NaN.
                # It is 1 by definition for the like-vs-like blocks; fill it directly.
                n_self = int(frame[a].notna().sum())
                if n_self >= 10:
                    R[i, j] = 1.0
                    N[i, j] = n_self
                    s = frame[a].dropna().values
                    NE[i, j] = eff_n(s, s)
                continue
            sub = frame[[a, b]].dropna()
            if len(sub) < 10:
                continue
            u, v = sub[a].values, sub[b].values
            if kind == 'cc':
                R[i, j] = circ_circ_corr(u, v)
            elif kind == 'cl':
                R[i, j] = circ_lin_corr(u, v)
            else:
                R[i, j] = pearson(u, v)
            N[i, j] = len(sub)
            NE[i, j] = eff_n(u, v)
    return R, N, NE


def draw(ax, R, rows, cols, title, diverging):
    cmap = DIVERGING if diverging else SEQUENTIAL
    vmin, vmax = (-1, 1) if diverging else (0, 1)
    im = ax.imshow(R, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels([c.replace(' phi', ' φ').replace(' dt', ' δt') for c in cols],
                       rotation=45, ha='right', fontsize=7.5, color=INK2)
    ax.set_yticklabels([r.replace(' phi', ' φ').replace(' dt', ' δt') for r in rows],
                       fontsize=7.5, color=INK2)
    ax.set_title(title, fontsize=9.5, fontweight='bold', color=INK, pad=8)
    # value in every cell: identity is never carried by colour alone
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(R[i, j]):
                continue
            shade = abs(R[i, j]) if diverging else R[i, j]
            ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=6.6,
                    color='white' if shade > 0.55 else INK)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which='minor', color='#fcfcfb', linewidth=1.4)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def make_page(frame, label, note):
    phis = [f'{s} phi' for s in STATION_ORDER]
    dts = [f'{s} dt' for s in STATION_ORDER]
    geo = ['CCAL uplift']

    fig = plt.figure(figsize=(15.5, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    Rpp, Npp, NEpp = corr_block(frame, phis, phis, 'cc')
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = draw(ax1, Rpp, phis, phis, 'φ vs φ — circular-circular (signed)', True)
    fig.colorbar(im1, ax=ax1, fraction=0.045).ax.tick_params(labelsize=7)

    Rdd, Ndd, NEdd = corr_block(frame, dts, dts, 'pp')
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = draw(ax2, Rdd, dts, dts, 'δt vs δt — Pearson (signed)', True)
    fig.colorbar(im2, ax=ax2, fraction=0.045).ax.tick_params(labelsize=7)

    Rpd, Npd, NEpd = corr_block(frame, phis, dts, 'cl')
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = draw(ax3, Rpd, phis, dts, 'φ vs δt — circular-linear R (UNSIGNED, 0–1)', False)
    fig.colorbar(im3, ax=ax3, fraction=0.045).ax.tick_params(labelsize=7)

    Rpg, Npg, NEpg = corr_block(frame, phis, geo, 'cl')
    Rdg, Ndg, NEdg = corr_block(frame, dts, geo, 'pp')
    ax4 = fig.add_subplot(gs[1, 1])
    combo = np.vstack([Rpg, np.abs(Rdg)])
    labels = [f'{s} φ' for s in STATION_ORDER] + [f'{s} |δt|' for s in STATION_ORDER]
    im4 = draw(ax4, combo, labels, geo,
               'vs geodetic uplift — φ: circ-lin R;  δt: |Pearson|  (both unsigned here)', False)
    fig.colorbar(im4, ax=ax4, fraction=0.045).ax.tick_params(labelsize=7)

    allN = np.concatenate([v[~np.isnan(v)].ravel() for v in (Npp, Ndd, Npd, Npg, Ndg)])
    allNE = np.concatenate([v[~np.isnan(v)].ravel() for v in (NEpp, NEdd, NEpd, NEpg, NEdg)])
    fig.suptitle(f'Station correlation matrices — {label}', fontsize=13, fontweight='bold',
                 color=INK, y=0.975)
    fig.text(0.5, 0.945,
             f'{note}   |   pairwise-complete N: median {np.median(allN):.0f}, '
             f'min {np.min(allN):.0f}   |   median effective N ≈ {np.median(allNE):.0f} '
             f'(lag-1 corrected)',
             ha='center', fontsize=8.5, color=INK2)

    rows = []
    for name, R, N, NE, rl, cl in (
            ('phi-phi', Rpp, Npp, NEpp, phis, phis),
            ('dt-dt', Rdd, Ndd, NEdd, dts, dts),
            ('phi-dt', Rpd, Npd, NEpd, phis, dts),
            ('phi-geo', Rpg, Npg, NEpg, phis, geo),
            ('dt-geo', Rdg, Ndg, NEdg, dts, geo)):
        for i, a in enumerate(rl):
            for j, b in enumerate(cl):
                rows.append(dict(page=label, block=name, a=a, b=b,
                                 r=R[i, j], n=N[i, j], n_eff=NE[i, j]))
    return fig, pd.DataFrame(rows)


def main():
    print('Building rolling phi / dt series per station...')
    frame = build_series()
    print(f'  frame: {frame.shape[0]} daily rows x {frame.shape[1]} columns\n')

    post = frame[frame.index >= POST_ERUPTION_START]
    post_thin = post.iloc[::THIN_DAYS]
    full_thin = frame.iloc[::THIN_DAYS]

    pages = [
        (post, 'post-eruption, daily',
         f'{ROLL_WINDOW_DAYS}-day rolling, sampled daily — AUTOCORRELATED, significance overstated'),
        (post_thin, f'post-eruption, thinned 1 per {THIN_DAYS}d',
         'decimated to ~independent samples — believe this page where it disagrees with the daily one'),
        (full_thin, f'full record incl. eruption, thinned 1 per {THIN_DAYS}d',
         'spans the 2015 eruption, so shared eruption-step response inflates station-station values'),
    ]

    figs, tables = [], []
    for fr, label, note in pages:
        print(f'{label}: {len(fr)} rows')
        fig, tab = make_page(fr, label, note)
        figs.append(fig)
        tables.append(tab)

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)

    pd.concat(tables, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_PDF} ({len(figs)} pages)')
    print(f'Saved {OUT_CSV}')


if __name__ == '__main__':
    main()
