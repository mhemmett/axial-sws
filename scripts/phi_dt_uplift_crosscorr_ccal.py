#!/usr/bin/env python3
"""
phi_dt_uplift_crosscorr_ccal.py

Cross-correlation of the same 14 variables as phi_dt_uplift_pca_ccal.py (each station's phi and
dt, de-tided and raw Central Caldera uplift; 30-day post-eruption bins, Grade 3; built by that
script's build_table, so the two figures cannot drift apart).

  (a) zero-lag correlation matrix of the LEVELS
  (b) zero-lag correlation matrix of the 30-DAY CHANGES (first differences between consecutive
      bins, both present). Uplift rises almost monotonically, so level correlations are
      dominated by shared secular trend; differencing asks whether month-to-month CHANGES move
      together, which is the testable version of the question.
  (c, d) lagged cross-correlation of each station's phi change (c) and dt change (d) with the
      de-tided uplift change, lags +-12 bins (~ +-1 yr). Positive lag = the seismic variable
      LAGS uplift.

Significance (cells dotted, bands shaded) uses a Bretherton et al. (1999) effective sample size
for each pair, N_eff = N (1 - r1a r1b) / (1 + r1a r1b), r1 = lag-1 autocorrelations, and a
two-sided 5% critical |r| = 1.96 / sqrt(N_eff - 2) (Fisher-z approximation). For the trending
LEVELS N_eff is tiny, which is why panel (a) has few significant cells despite large |r|.

phi caveat (as in the PCA): phi enters as a linear azimuth on each station's own contiguous
180-deg branch. AXAS1 and AXCC1 span nearly the full 180 deg, so their phi correlations depend on
where that branch is cut.

Output: phi_dt_uplift_crosscorr_ccal.pdf / .png
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phi_dt_uplift_pca_ccal import build_table, BIN_DAYS
from rose_7period_6stations_newdata_snr_grades import STATION_ORDER
from uplift_vs_phi_regions_scatter_ccal_30day import STATION_COLORS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'phi_dt_uplift_crosscorr_ccal')
MAX_LAG = 12
MIN_PAIRS = 15


def lag1(x):
    x = pd.Series(x)
    return float(x.autocorr(1)) if x.notna().sum() > 5 else 0.0


def r_crit(n, r1a, r1b):
    prod = np.clip(r1a * r1b, -0.99, 0.99)
    n_eff = n * (1.0 - prod) / (1.0 + prod)
    return 1.96 / np.sqrt(max(n_eff - 2.0, 1.0)), n_eff


def corr_matrix(df):
    cols = df.columns
    k = len(cols)
    R = np.full((k, k), np.nan)
    sig = np.zeros((k, k), dtype=bool)
    r1 = {c: lag1(df[c]) for c in cols}
    for i in range(k):
        for j in range(k):
            ok = df[cols[i]].notna() & df[cols[j]].notna()
            if ok.sum() < MIN_PAIRS:
                continue
            R[i, j] = np.corrcoef(df.loc[ok, cols[i]], df.loc[ok, cols[j]])[0, 1]
            if i != j:
                rc, _ = r_crit(int(ok.sum()), r1[cols[i]], r1[cols[j]])
                sig[i, j] = abs(R[i, j]) > rc
    return R, sig


def first_diff(table, cols):
    """Change between consecutive bins; NaN unless both bins present."""
    return table[cols].diff()


def lagged_ccf(x, y, max_lag):
    """r(x[t+lag], y[t]) for lag in [-max_lag, max_lag]; positive lag: x lags y."""
    out = []
    for lag in range(-max_lag, max_lag + 1):
        xs = x.shift(-lag)
        ok = xs.notna() & y.notna()
        out.append(np.corrcoef(xs[ok], y[ok])[0, 1] if ok.sum() >= MIN_PAIRS else np.nan)
    return np.array(out)


def pretty(c):
    return c.replace('u_detided', 'Uplift, de-tided').replace('u_raw', 'Uplift, raw')


def draw_matrix(ax, R, sig, cols, title):
    im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1)
    ii, jj = np.nonzero(sig)
    ax.scatter(jj, ii, s=10, color='black', marker='o', lw=0)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([pretty(c) for c in cols], rotation=90, fontsize=8)
    ax.set_yticklabels([pretty(c) for c in cols], fontsize=8)
    for b in (5.5, 11.5):
        ax.axhline(b, color='white', lw=2)
        ax.axvline(b, color='white', lw=2)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    return im


def main():
    table = build_table()
    cols = [f'{s} φ' for s in STATION_ORDER] + [f'{s} δt' for s in STATION_ORDER] + \
        ['u_detided', 'u_raw']

    R_lev, S_lev = corr_matrix(table[cols])
    D = first_diff(table, cols)
    R_dif, S_dif = corr_matrix(D)

    fig = plt.figure(figsize=(19, 15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.38, wspace=0.28,
                          top=0.93, bottom=0.06, left=0.08, right=0.92)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    im = draw_matrix(ax_a, R_lev, S_lev, cols, '(a) Levels: zero-lag correlation')
    draw_matrix(ax_b, R_dif, S_dif, cols, f'(b) {BIN_DAYS}-day changes: zero-lag correlation')
    cax = fig.add_axes([0.935, 0.56, 0.01, 0.3])
    fig.colorbar(im, cax=cax, label='Pearson r')

    du = D['u_detided']
    r1u = lag1(du)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    for p, (kind, label) in enumerate((('φ', '(c) φ change vs uplift change'),
                                       ('δt', '(d) δt change vs uplift change'))):
        ax = fig.add_subplot(gs[1, p])
        crits = []
        for sta in STATION_ORDER:
            x = D[f'{sta} {kind}']
            ccf = lagged_ccf(x, du, MAX_LAG)
            ok = x.notna() & du.notna()
            rc, _ = r_crit(int(ok.sum()), lag1(x), r1u)
            crits.append(rc)
            ax.plot(lags, ccf, color=STATION_COLORS[sta], lw=1.8, marker='o', ms=3.5, label=sta)
            k = int(np.nanargmax(np.abs(ccf)))
            print(f'  {kind} {sta}: r0={ccf[MAX_LAG]:+.2f}  peak r={ccf[k]:+.2f} at lag '
                  f'{lags[k]:+d} bins  (|r| crit ~{rc:.2f})')
        rc = float(np.median(crits))
        ax.axhspan(-rc, rc, color='0.88', zorder=0, label=f'Not significant (|r| < {rc:.2f})')
        ax.axhline(0, color='0.4', lw=0.8)
        ax.axvline(0, color='0.4', lw=0.8, ls=':')
        ax.set_xlim(-MAX_LAG - 0.5, MAX_LAG + 0.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xlabel(f'Lag ({BIN_DAYS}-day bins; positive = {kind} lags uplift)')
        ax.set_ylabel('Cross-correlation r')
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, ncol=4, frameon=False, loc='lower center')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for name, R, S in (('levels', R_lev, S_lev), ('changes', R_dif, S_dif)):
        iu = np.triu_indices(len(cols), 1)
        print(f'{name}: {int(S[iu].sum())} / {len(iu[0])} pairs significant (N_eff-corrected)')
        top = np.argsort(-np.nan_to_num(np.abs(R[iu])))[:8]
        print('   strongest: ' + '; '.join(f'{cols[iu[0][t]]}~{cols[iu[1][t]]} {R[iu][t]:+.2f}'
                                          f'{"*" if S[iu][t] else ""}' for t in top))

    fig.suptitle('Cross-correlation of φ, δt and Central Caldera uplift (Grade 3, post-eruption, '
                 f'{BIN_DAYS}-day bins)', fontsize=16, fontweight='bold', y=0.985)
    fig.text(0.5, 0.012, 'Dots in (a, b): significant at 5% after correcting for autocorrelation '
             '(Bretherton effective N). Diagonal blocks: φ (top-left), δt (centre), uplift '
             '(bottom-right).', ha='center', fontsize=9.5, color='0.3')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
