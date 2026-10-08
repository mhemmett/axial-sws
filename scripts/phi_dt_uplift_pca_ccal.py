#!/usr/bin/env python3
"""
phi_dt_uplift_pca_ccal.py

Principal component analysis of fast direction (phi), delay time (dt) and Central Caldera BOTPT
uplift (de-tided AND non-de-tided), post-eruption, Grade 3:

  (A) WITHIN each station: PCA of [phi, dt, uplift (de-tided), uplift (raw)]
  (B) ACROSS the network: one PCA of all six stations' phi and dt plus both uplift series
      (14 variables), so shared modes between stations and with the geodesy show up as common
      loadings on the same component.

Sampling: non-overlapping 30-day bins from ERUPTION_END. Daily values are dominated by
single-event noise and overlapping rolled windows duplicate information, so neither is used.
Per station and bin (>= MIN_EVENTS Grade-3 events, else missing):
    phi  -> circular mean, kept as a real azimuth in degrees. phi is axial (mod 180), so each
            station's bin means are placed on one contiguous 180-deg branch with the cut at that
            station's largest data gap (compute_optimal_wrap, as in the geodetic figures) --
            otherwise a near-N-S station would split across 0/180. Standardisation removes each
            variable's mean, so this is equivalent to using deviations from the station mean.
    dt   -> mean delay time (s)
Uplift: de-tided (bpr_inflation_periods_ccal) and raw seafloor depth, both referenced to the
same post-eruption zero (ref_time) and averaged per bin. AXEC2's unleveled window is dropped.

All variables are standardised (correlation-matrix PCA) because units differ. (B) uses the
pairwise-complete correlation matrix (stations have different gaps); small negative eigenvalues
from pairwise deletion are clipped to 0 and reported. Uncertainty: moving-block bootstrap over
bins (BLOCK_BINS-bin blocks, so autocorrelation is respected), loadings sign-aligned to the
full-data solution, 95% intervals.

Reading caveat: uplift rises almost monotonically after 2015, so ANY variable with a secular
trend loads with it on the leading component. Shared loading means shared trend, not a causal or
mechanistic link. A linearly DETRENDED version of (B) is printed for comparison.

Outputs:
    phi_dt_uplift_pca_ccal.pdf / .png
    phi_dt_uplift_pca_ccal_loadings.csv
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rose_7period_regions_windowcheck_grade3 import GRADE, load_station_raw, apply_grade
from rose_7period_6stations_newdata_snr_grades import STATION_ORDER
from axec2_uplift_phi_cosine_vs_time import ERUPTION_END, UNLEVEL_START, UNLEVEL_END
from animate_arctan_stress_vectors import compute_optimal_wrap
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'phi_dt_uplift_pca_ccal')

BIN_DAYS = 30
MIN_EVENTS = 5
BLOCK_BINS = 6
N_BOOT = 1000
SEED = 20261008

# Colours: IBM colour-blind-safe blue / orange for phi / dt, black / gray for the two uplifts.
C_PHI, C_DT, C_UDET, C_URAW = '#648FFF', '#FE6100', '#000000', '#8C8C8C'


def wrap90(a):
    return (np.asarray(a) + 90.0) % 180.0 - 90.0


def circ_mean_axial(phi_deg):
    a = 2.0 * np.radians(np.asarray(phi_deg) % 180.0)
    return (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0) % 180.0


def build_table():
    daily_det, _infl, _roll, ref_time, _ref_depth = ccal_infl.load_daily_series()
    raw = pd.read_csv(ccal_infl.DETIDED_CSV, usecols=['time', 'seafloor_depth_m'])
    raw['time'] = pd.to_datetime(raw['time'], utc=True)
    raw_daily = raw.set_index('time')['seafloor_depth_m'].resample('1D').mean()
    u_det = daily_det - float(daily_det.loc[ref_time])
    u_raw = raw_daily - float(raw_daily.loc[ref_time])

    t_end = u_det.dropna().index.max()
    edges = pd.date_range(ERUPTION_END, t_end, freq=f'{BIN_DAYS}D')
    mids = edges[:-1] + pd.Timedelta(days=BIN_DAYS / 2)

    def bin_series(s):
        s = s.loc[ERUPTION_END:t_end].dropna()
        idx = edges.searchsorted(s.index, side='right') - 1
        ok = (idx >= 0) & (idx < len(edges) - 1)
        return pd.Series(s.values[ok]).groupby(idx[ok]).mean().reindex(range(len(mids)))

    table = pd.DataFrame(index=range(len(mids)))
    table['t'] = mids
    table['u_detided'] = bin_series(u_det).values
    table['u_raw'] = bin_series(u_raw).values

    for sta in STATION_ORDER:
        df = apply_grade(load_station_raw(sta), GRADE)
        df = df[df['t'] >= ERUPTION_END]
        if sta == 'AXEC2':
            df = df[~((df['t'] >= UNLEVEL_START) & (df['t'] < UNLEVEL_END))]
        idx = edges.searchsorted(pd.DatetimeIndex(df['t']), side='right') - 1
        df = df.assign(bin=idx)
        df = df[(df['bin'] >= 0) & (df['bin'] < len(mids))]
        g = df.groupby('bin')
        n = g.size()
        phi = g['phi_az'].apply(circ_mean_axial)
        dt = g['dt'].mean()
        good = n[n >= MIN_EVENTS].index
        phi_g = phi.reindex(good).values % 180.0
        w = compute_optimal_wrap(phi_g)
        phi_az = pd.Series((phi_g - w) % 180.0 + w, index=good)   # contiguous azimuth branch
        table[f'{sta} φ'] = phi_az.reindex(range(len(mids))).values
        table[f'{sta} δt'] = dt.reindex(good).reindex(range(len(mids))).values
    return table


def pca_corr(X, pairwise=False):
    """Correlation-matrix PCA. Returns (explained fraction, loadings [var x pc], n_clipped)."""
    df = pd.DataFrame(X)
    C = df.corr(min_periods=8).values if pairwise else np.corrcoef(df.dropna().values.T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    n_clip = int(np.sum(w < 0))
    w = np.clip(w, 0.0, None)
    loadings = V * np.sqrt(w)          # variable-PC correlations
    return w / w.sum(), loadings, n_clip


def align(L, ref):
    """Flip PC signs to match a reference loading matrix."""
    L = L.copy()
    for j in range(min(L.shape[1], ref.shape[1])):
        if np.dot(L[:, j], ref[:, j]) < 0:
            L[:, j] *= -1
    return L


def block_bootstrap(X, pairwise, ref_L, rng, n_pc=3):
    n = len(X)
    nb = int(np.ceil(n / BLOCK_BINS))
    evs, Ls = [], []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK_BINS + 1, size=nb)
        rows = np.concatenate([np.arange(s, s + BLOCK_BINS) for s in starts])[:n]
        ev, L, _ = pca_corr(X[rows], pairwise=pairwise)
        evs.append(ev[:n_pc])
        Ls.append(align(L, ref_L)[:, :n_pc])
    return np.array(evs), np.array(Ls)


def detrend(X, t):
    tt = (t - t.min()) / np.ptp(t)
    out = X.copy()
    for j in range(X.shape[1]):
        ok = ~np.isnan(X[:, j])
        if ok.sum() > 3:
            p = np.polyfit(tt[ok], X[ok, j], 1)
            out[ok, j] = X[ok, j] - np.polyval(p, tt[ok])
    return out


def var_color(name):
    return C_PHI if 'φ' in name else C_DT if 'δt' in name else \
        C_UDET if name == 'u_detided' else C_URAW


def main():
    rng = np.random.default_rng(SEED)
    table = build_table()
    print(f'{len(table)} bins of {BIN_DAYS} d from {ERUPTION_END.date()}')

    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.25], hspace=0.55, wspace=0.42,
                          top=0.86, bottom=0.07, left=0.06, right=0.98)
    rows = []

    # ---- (A) within each station -------------------------------------------------------------
    for k, sta in enumerate(STATION_ORDER):
        cols = [f'{sta} φ', f'{sta} δt', 'u_detided', 'u_raw']
        sub = table[cols].dropna()
        X = sub.values
        ev, L, _ = pca_corr(X)
        ev_b, L_b = block_bootstrap(X, False, L, rng, n_pc=2)
        ax = fig.add_subplot(gs[0, k])
        circle = plt.Circle((0, 0), 1, fill=False, color='0.6', lw=0.8)
        ax.add_patch(circle)
        ax.axhline(0, color='0.85', lw=0.6)
        ax.axvline(0, color='0.85', lw=0.6)
        names = ['φ', 'δt', 'uplift (de-tided)', 'uplift (raw)']
        for j, (nm, c) in enumerate(zip(names, [C_PHI, C_DT, C_UDET, C_URAW])):
            ax.annotate('', xy=(L[j, 0], L[j, 1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=c, lw=2 if j < 2 else 1.4,
                                        ls='-' if j != 3 else '--'))
            rows.append(dict(analysis='within-station', station=sta, variable=nm,
                             PC1=L[j, 0], PC1_lo=np.percentile(L_b[:, j, 0], 2.5),
                             PC1_hi=np.percentile(L_b[:, j, 0], 97.5), PC2=L[j, 1],
                             EV1=ev[0], EV2=ev[1], n_bins=len(sub)))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(sta, fontsize=12, fontweight='bold', loc='left')
        ax.text(1.0, 1.01, f'{len(sub)} bins', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='0.35')
        ax.set_xlabel(f'PC1 ({ev[0]*100:.0f}%, 95% {np.percentile(ev_b[:,0],2.5)*100:.0f}–'
                      f'{np.percentile(ev_b[:,0],97.5)*100:.0f})', fontsize=8.5)
        ax.set_ylabel(f'PC2 ({ev[1]*100:.0f}%)', fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        r_pd = np.corrcoef(sub[cols[0]], sub[cols[2]])[0, 1]
        r_td = np.corrcoef(sub[cols[1]], sub[cols[2]])[0, 1]
        print(f'  {sta}: bins={len(sub)} EV1={ev[0]*100:.0f}% EV2={ev[1]*100:.0f}%  '
              f'r(phi,u)={r_pd:+.2f} r(dt,u)={r_td:+.2f} '
              f'r(u_det,u_raw)={np.corrcoef(sub["u_detided"], sub["u_raw"])[0,1]:.4f}')
    handles = [plt.Line2D([], [], color=c, lw=2, ls=ls) for c, ls in
               ((C_PHI, '-'), (C_DT, '-'), (C_UDET, '-'), (C_URAW, '--'))]
    fig.legend(handles, ['φ (fast-direction azimuth)', 'δt', 'Uplift, de-tided',
                         'Uplift, raw (non-de-tided)'],
               loc='upper center', ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.935))

    # ---- (B) across the network ---------------------------------------------------------------
    net_cols = [f'{s} φ' for s in STATION_ORDER] + [f'{s} δt' for s in STATION_ORDER] + \
        ['u_detided', 'u_raw']
    X = table[net_cols].values
    ev, L, n_clip = pca_corr(X, pairwise=True)
    ev_b, L_b = block_bootstrap(X, True, L, rng, n_pc=3)
    ev_dt, L_dt, _ = pca_corr(detrend(X, table['t'].astype('int64').values / 8.64e13),
                              pairwise=True)
    n_complete = int(table[net_cols].dropna().shape[0])
    print(f'Network PCA: {len(net_cols)} variables, pairwise-complete '
          f'({n_complete} fully complete bins), {n_clip} negative eigenvalue(s) clipped')
    print('  EV (%):', ' '.join(f'PC{i+1}={e*100:.1f}' for i, e in enumerate(ev[:4])))
    print('  detrended EV (%):', ' '.join(f'PC{i+1}={e*100:.1f}' for i, e in enumerate(ev_dt[:4])))

    ax_s = fig.add_subplot(gs[1, 0:2])
    n_show = 8
    ax_s.bar(np.arange(1, n_show + 1), ev[:n_show] * 100, color='0.55', edgecolor='white', lw=2,
             label='All variables')
    lo = np.percentile(ev_b, 2.5, axis=0) * 100
    hi = np.percentile(ev_b, 97.5, axis=0) * 100
    # Bootstrap percentile intervals need not bracket the full-data value, so draw them as
    # lo-hi segments rather than +- error bars.
    ax_s.vlines(np.arange(1, 4), lo, hi, color='black', lw=1.4, label='95% block bootstrap')
    ax_s.plot(np.arange(1, n_show + 1), ev_dt[:n_show] * 100, 'o--', color='0.15', ms=5,
              lw=1, label='Linearly detrended')
    ax_s.axhline(100.0 / len(net_cols), color='0.4', lw=0.9, ls=':',
                 label=f'Equal share (1/{len(net_cols)})')
    ax_s.set_xticks(np.arange(1, n_show + 1))
    ax_s.set_xlabel('Principal component')
    ax_s.set_ylabel('Variance explained (%)')
    ax_s.set_title('Network PCA: variance explained', fontsize=12, fontweight='bold', loc='left')
    ax_s.legend(fontsize=8.5, frameon=False)
    for side in ('top', 'right'):
        ax_s.spines[side].set_visible(False)

    for p, gs_slice in ((0, gs[1, 2:4]), (1, gs[1, 4:6])):
        ax = fig.add_subplot(gs_slice)
        y = np.arange(len(net_cols))
        ax.barh(y, L[:, p], color=[var_color(c) for c in net_cols], height=0.7,
                edgecolor='white', lw=1.5)
        lo = np.percentile(L_b[:, :, p], 2.5, axis=0)
        hi = np.percentile(L_b[:, :, p], 97.5, axis=0)
        ax.hlines(y, lo, hi, color='black', lw=1.0)   # 95% block-bootstrap interval
        ax.axvline(0, color='0.3', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([c.replace('u_detided', 'Uplift, de-tided')
                            .replace('u_raw', 'Uplift, raw') for c in net_cols], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel('Loading (correlation with the component)')
        ax.set_title(f'Network PC{p + 1} loadings ({ev[p]*100:.0f}% of variance)', fontsize=12,
                     fontweight='bold', loc='left')
        ax.grid(axis='x', alpha=0.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for j, c in enumerate(net_cols):
            rows.append(dict(analysis='network', station='', variable=c, PC=p + 1,
                             loading=L[j, p], lo95=lo[j], hi95=hi[j], EV=ev[p],
                             EV_detrended=ev_dt[p]))
        print(f'  PC{p+1} top loadings: ' + ', '.join(
            f'{net_cols[j]} {L[j, p]:+.2f}' for j in np.argsort(-np.abs(L[:, p]))[:6]))

    fig.text(0.01, 0.935, 'A  Within each station', fontsize=14, fontweight='bold', va='top')
    fig.text(0.01, ax_s.get_position().y1 + 0.055, 'B  Across all stations + geodesy',
             fontsize=14, fontweight='bold', va='bottom')
    fig.suptitle(f'PCA of φ, δt and Central Caldera uplift (Grade 3, post-eruption, '
                 f'{BIN_DAYS}-day bins)', fontsize=16, fontweight='bold', y=0.99)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f'{OUT_BASE}_loadings.csv', index=False)
    print(f'Saved {OUT_BASE}.pdf / .png / _loadings.csv')


if __name__ == '__main__':
    main()
