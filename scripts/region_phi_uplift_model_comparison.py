#!/usr/bin/env python3
"""
region_phi_uplift_model_comparison.py

Which functional form actually describes regional fast direction vs. caldera uplift best?

Fits a battery of candidate families to the Western and Eastern caldera regional means (the
same pooled, grade-3, daily-mean-then-30-day-rolled phi and the same Central Caldera BOTPT
uplift used by atan2_uplift_vs_phi_regions_ccal_30day.py) and ranks them.

WHY THE NAIVE RANKING CANNOT BE TRUSTED ON ITS OWN
    The points are DAILY samples of a 30-DAY ROLLING MEAN. Consecutive points share 29/30 of
    their input, so they are massively autocorrelated and N (~2200 west, ~3100 east) hugely
    overstates how much independent information is present. Every likelihood-based criterion
    (AIC, BIC) scales with N, so on the raw series they will happily "prove" that a
    higher-parameter model is decisively better when the extra flexibility is really just
    tracking correlated noise.

    Two guards are therefore reported alongside the raw numbers:

    1. THINNED refit -- the series is decimated to one point per 30 days (the rolling-window
       length), which is roughly the decorrelation scale, and every model is refitted and
       re-ranked on that. If a model only wins on the raw series and not on the thinned one,
       its win is an autocorrelation artifact.
    2. Effective sample size -- n_eff = n(1-rho)/(1+rho) from the lag-1 autocorrelation of
       that model's own residuals, reported per model so the degree of inflation is visible.

    A ranking is only worth acting on where raw and thinned AGREE.

WHAT IS AND IS NOT BEING COMPARED
    All families are fitted to the SAME wrapped y (phi mapped through compute_optimal_wrap,
    exactly as the production atan2 fit does) so the numbers are commensurable. The incumbent
    atan2 vector-sum model is included in the field.

    The atan2 model is charged k=5 (C1, C2, A, beta, plus u0). u0 is not free in the final
    curve_fit, but it IS estimated from the same data by a 4-parameter logistic pre-fit, so
    charging it nothing would flatter the incumbent. Even k=5 is generous -- the honest count
    is somewhere between 5 and 8.

    Not addressed here: phi is axial/circular and these are all ordinary least-squares fits in
    the unwrapped space. That is the same assumption the production fit already makes, so the
    comparison is internally fair, but none of these are proper circular regressions.

Produces:
    region_phi_uplift_model_comparison.pdf   (one page per region: data + top fits)
    region_phi_uplift_model_comparison.csv   (full ranking table, both raw and thinned)

Run with:
    python3 region_phi_uplift_model_comparison.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bpr_inflation_periods_ccal import POST_ERUPTION_START
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import _circular_mean_and_se_deg
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
)
from animate_arctan_stress_vectors import compute_optimal_wrap
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    estimate_turnover_u0, ROLL_WINDOW_DAYS, ROLL_MIN_DAYS,
)
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool, REGIONS
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'region_phi_uplift_model_comparison.pdf')
OUT_CSV = os.path.join(HERE, 'region_phi_uplift_model_comparison.csv')

THIN_DAYS = ROLL_WINDOW_DAYS   # decimate to ~1 independent point per rolling window
BIG = 1e4


# ── candidate families ──────────────────────────────────────────────────────────────────────
# Each entry: (name, function, n_params, p0_builder(x, y) -> list of starting guesses)

def _span(x, y):
    return (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))


def f_linear(x, a, b):            return a + b * x
def f_quad(x, a, b, c):           return a + b * x + c * x**2
def f_cubic(x, a, b, c, d):       return a + b * x + c * x**2 + d * x**3
def f_log(x, a, b):               return a + b * np.log(np.clip(x, 1e-6, None))
def f_sqrt(x, a, b):              return a + b * np.sqrt(np.clip(x, 0, None))
def f_cbrt(x, a, b):              return a + b * np.cbrt(np.clip(x, 0, None))
def f_power(x, a, b, c):          return a + b * np.power(np.clip(x, 1e-6, None), c)
def f_expdecay(x, a, b, c):       return a + b * np.exp(-c * x)
def f_expsat(x, a, b, c):         return a + b * (1.0 - np.exp(-c * x))
def f_hyperbolic(x, a, b, c):     return a + b * x / (c + x)
def f_logistic(x, a, d, x0, k):   return d + (a - d) / (1.0 + np.exp(-k * (x - x0)))
def f_tanh(x, a, b, x0, k):       return a + b * np.tanh(k * (x - x0))
def f_atan(x, a, b, x0, k):       return a + b * np.arctan(k * (x - x0))
def f_erfc(x, a, b, x0, k):       return a + b * erfc(k * (x - x0))


def _starts(x, y):
    xlo, xhi, ylo, yhi = _span(x, y)
    xs = (xhi - xlo) or 1.0
    xm = 0.5 * (xlo + xhi)
    dy = (yhi - ylo) or 1.0
    return xlo, xhi, ylo, yhi, xs, xm, dy


FAMILIES = []


def _reg(name, func, k, p0s, bounds=None):
    FAMILIES.append(dict(name=name, func=func, k=k, p0s=p0s, bounds=bounds))


def build_families(x, y):
    """Rebuild the family table with data-dependent starting guesses."""
    FAMILIES.clear()
    xlo, xhi, ylo, yhi, xs, xm, dy = _starts(x, y)
    ym = float(np.mean(y))

    _reg('linear',            f_linear,     2, [[ym, 0.0]])
    _reg('quadratic',         f_quad,       3, [[ym, 0.0, 0.0]])
    _reg('cubic',             f_cubic,      4, [[ym, 0.0, 0.0, 0.0]])
    _reg('logarithmic',       f_log,        2, [[ym, 0.0]])
    _reg('square root',       f_sqrt,       2, [[ym, 0.0]])
    _reg('cube root',         f_cbrt,       2, [[ym, 0.0]])
    _reg('power law',         f_power,      3, [[ym, s * dy, p]
                                                for s in (1, -1) for p in (0.33, 0.5, 2.0)])
    _reg('exponential decay', f_expdecay,   3, [[ym, s * dy, r]
                                                for s in (1, -1) for r in (1/xs, 4/xs)])
    _reg('exponential sat.',  f_expsat,     3, [[ylo, s * dy, r]
                                                for s in (1, -1) for r in (1/xs, 4/xs)])
    _reg('hyperbolic (MM)',   f_hyperbolic, 3, [[ym, s * dy, c]
                                                for s in (1, -1) for c in (0.25*xs, xs)])
    _reg('logistic',          f_logistic,   4, [[a0, d0, xm, k]
                                                for a0, d0 in ((ylo, yhi), (yhi, ylo))
                                                for k in (4/xs, -4/xs, 12/xs)])
    _reg('tanh',              f_tanh,       4, [[ym, s * dy/2, xm, k]
                                                for s in (1, -1) for k in (4/xs, 12/xs)])
    _reg('arctangent',        f_atan,       4, [[ym, s * dy/2, xm, k]
                                                for s in (1, -1) for k in (4/xs, 12/xs)])
    _reg('erfc',              f_erfc,       4, [[ym, s * dy/2, xm, k]
                                                for s in (1, -1) for k in (2/xs, 6/xs)])
    return FAMILIES


def fit_family(fam, x, y):
    """Multi-start least squares. Returns (popt, rss) or (None, inf)."""
    best = (None, np.inf)
    for p0 in fam['p0s']:
        try:
            popt, _ = curve_fit(fam['func'], x, y, p0=p0, maxfev=40000)
        except Exception:
            continue
        pred = fam['func'](x, *popt)
        if not np.all(np.isfinite(pred)):
            continue
        rss = float(np.sum((y - pred) ** 2))
        if rss < best[1]:
            best = (popt, rss)
    return best


def fit_atan2_incumbent(x, y):
    """The production atan2 vector-sum model, scored on the same footing."""
    from animate_arctan_stress_vectors import fit_atan2_vectorsum
    u0 = estimate_turnover_u0(x, y)
    C1, C2, A, beta, r, model = fit_atan2_vectorsum(x, y, u0)
    pred = model(x, C1, C2, A, beta)
    rss = float(np.sum((y - pred) ** 2))
    return (np.array([C1, C2, A, beta]), rss, u0, model)


def scores(rss, n, k, resid):
    """R^2 is filled in by the caller (needs TSS). Returns AICc/BIC/RMSE/n_eff."""
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    denom = n - k - 1
    aicc = aic + (2 * k * (k + 1) / denom if denom > 0 else np.inf)
    bic = n * np.log(rss / n) + k * np.log(n)
    rmse = float(np.sqrt(rss / n))
    # lag-1 autocorrelation of residuals -> effective sample size
    if len(resid) > 2:
        r0 = resid - resid.mean()
        denom_ac = float(np.sum(r0 * r0))
        rho = float(np.sum(r0[:-1] * r0[1:]) / denom_ac) if denom_ac > 0 else 0.0
        rho = min(max(rho, -0.999), 0.999)
        n_eff = n * (1 - rho) / (1 + rho)
    else:
        rho, n_eff = np.nan, np.nan
    return aicc, bic, rmse, rho, max(n_eff, 1.0)


def rank_on(x, y, label):
    """Fit every family (plus the incumbent) to (x, y); return a ranked DataFrame."""
    n = len(x)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    rows = []

    for fam in build_families(x, y):
        popt, rss = fit_family(fam, x, y)
        if popt is None:
            continue
        resid = y - fam['func'](x, *popt)
        aicc, bic, rmse, rho, n_eff = scores(rss, n, fam['k'], resid)
        rows.append(dict(model=fam['name'], k=fam['k'], rss=rss,
                         r2=1 - rss / tss, rmse=rmse, aicc=aicc, bic=bic,
                         rho1=rho, n_eff=n_eff, n=n,
                         _func=fam['func'], _popt=popt))

    popt, rss, u0, model = fit_atan2_incumbent(x, y)
    resid = y - model(x, *popt)
    aicc, bic, rmse, rho, n_eff = scores(rss, n, 5, resid)
    rows.append(dict(model='atan2 vector-sum', k=5, rss=rss,
                     r2=1 - rss / tss, rmse=rmse, aicc=aicc, bic=bic,
                     rho1=rho, n_eff=n_eff, n=n,
                     _func=(lambda xx, *p: model(xx, *p)), _popt=popt))

    df = pd.DataFrame(rows).sort_values('aicc').reset_index(drop=True)
    df['d_aicc'] = df['aicc'] - df['aicc'].min()
    df['rank'] = np.arange(1, len(df) + 1)
    df['series'] = label
    return df


def print_table(df, title):
    print(f'\n  {title}')
    print(f'  {"rank":<5}{"model":<20}{"k":>3}{"R2":>9}{"RMSE":>9}{"dAICc":>11}{"n_eff":>9}')
    print('  ' + '-' * 66)
    for _, r in df.iterrows():
        print(f'  {int(r["rank"]):<5}{r["model"]:<20}{int(r["k"]):>3}{r["r2"]:>9.4f}'
              f'{r["rmse"]:>9.3f}{r["d_aicc"]:>11.1f}{r["n_eff"]:>9.0f}')


def main():
    _dd, _infl_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']

    wanted = {'Western Caldera (AXAS1 + AXAS2)',
              'Eastern Caldera (AXEC1 + AXEC2 + AXEC3, AXEC2 unleveled window excluded)'}

    all_tables = []
    figs = []

    for label, members in REGIONS:
        if label not in wanted:
            continue
        short = label.split(' (')[0]
        print(f'\n{"="*74}\n{short}\n{"="*74}')

        pool = load_region_pool(members)
        pre = pool[pool['t'] < ERUPTION_START]
        baseline_phi, _ = _circular_mean_and_se_deg(pre['phi_az'].values)
        roll = rolling_phi_stats_daily_then_roll(pool, baseline_phi,
                                                 window_days=ROLL_WINDOW_DAYS,
                                                 min_days=ROLL_MIN_DAYS)

        valid = roll.dropna(subset=['mean_phi']).copy()
        valid = valid[valid['t'] >= POST_ERUPTION_START]
        merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                               direction='nearest', tolerance=pd.Timedelta('20D'))
        merged = merged.dropna(subset=['inflation_m']).reset_index(drop=True)

        x = merged['inflation_m'].values
        wrap = compute_optimal_wrap(merged['mean_phi'].values)
        y = (merged['mean_phi'].values - wrap) % 180.0

        raw = rank_on(x, y, 'raw (daily)')
        print_table(raw, f'RAW series, N={len(x)} daily points '
                         f'(autocorrelated -- see n_eff)')

        # thinned: one point per rolling-window length, i.e. ~decorrelated
        keep = merged['t'] >= merged['t'].min()
        idx = np.arange(0, len(merged), 1)
        t0 = merged['t'].iloc[0]
        bucket = ((merged['t'] - t0).dt.days // THIN_DAYS).values
        first_in_bucket = np.concatenate(([True], bucket[1:] != bucket[:-1]))
        xt, yt = x[first_in_bucket], y[first_in_bucket]
        thin = rank_on(xt, yt, f'thinned (1 per {THIN_DAYS}d)')
        print_table(thin, f'THINNED series, N={len(xt)} points '
                          f'(1 per {THIN_DAYS} days, ~independent)')

        # agreement
        raw_top = raw.iloc[0]['model']
        thin_top = thin.iloc[0]['model']
        agree = raw_top == thin_top
        print(f'\n  Raw best:     {raw_top}')
        print(f'  Thinned best: {thin_top}')
        print(f'  -> {"AGREE" if agree else "DISAGREE -- treat the raw ranking as unreliable"}')
        inc_raw = raw[raw['model'] == 'atan2 vector-sum'].iloc[0]
        inc_thin = thin[thin['model'] == 'atan2 vector-sum'].iloc[0]
        print(f'  Incumbent atan2 vector-sum: rank {int(inc_raw["rank"])}/{len(raw)} raw '
              f'(dAICc={inc_raw["d_aicc"]:.1f}), rank {int(inc_thin["rank"])}/{len(thin)} '
              f'thinned (dAICc={inc_thin["d_aicc"]:.1f})')

        for df, tag in ((raw, 'raw (daily)'), (thin, f'thinned (1 per {THIN_DAYS}d)')):
            out = df.drop(columns=['_func', '_popt']).copy()
            out.insert(0, 'region', short)
            out['series'] = tag
            all_tables.append(out)

        figs.append(make_figure(short, x, y, raw, thin, wrap))

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)

    pd.concat(all_tables, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_PDF} ({len(figs)} pages)')
    print(f'Saved {OUT_CSV}')


def make_figure(short, x, y, raw, thin, wrap):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2),
                             gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    ax.scatter(x, y, s=12, color='#0072B2', alpha=0.35, label='30-day rolling mean', zorder=1)

    xl = np.linspace(x.min(), x.max(), 400)
    colors = ['#d55e00', '#009e73', '#cc79a7', '#e69f00', '#56b4e9']
    for i, (_, r) in enumerate(raw.head(5).iterrows()):
        ax.plot(xl, r['_func'](xl, *r['_popt']), lw=1.9, color=colors[i % len(colors)],
                label=f'{int(r["rank"])}. {r["model"]} (k={int(r["k"])}, '
                      f'R²={r["r2"]:.3f}, ΔAICc={r["d_aicc"]:.0f})', zorder=3)
    inc = raw[raw['model'] == 'atan2 vector-sum'].iloc[0]
    if inc['rank'] > 5:
        ax.plot(xl, inc['_func'](xl, *inc['_popt']), lw=1.6, color='black', linestyle='--',
                label=f'{int(inc["rank"])}. atan2 vector-sum (incumbent, '
                      f'ΔAICc={inc["d_aicc"]:.0f})', zorder=2)

    ax.set_xlabel('De-tided uplift $u_z$ (m, 30-day rolling mean)')
    ax.set_ylabel(f'Regional mean $\\phi$ (deg)\n[axis wraps mod 180, optimal wrap={wrap:.1f}°]')
    ax.set_title(f'{short}: top-5 candidate fits', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.axis('off')
    lines = [f'{short} — model ranking', '']
    lines.append(f'{"model":<19}{"k":>2}{"R²":>8}{"ΔAICc raw":>11}{"ΔAICc thin":>12}')
    lines.append('-' * 52)
    thin_rank = {r['model']: r for _, r in thin.iterrows()}
    for _, r in raw.iterrows():
        t = thin_rank.get(r['model'])
        lines.append(f'{r["model"]:<19}{int(r["k"]):>2}{r["r2"]:>8.3f}'
                     f'{r["d_aicc"]:>11.1f}'
                     f'{(t["d_aicc"] if t is not None else float("nan")):>12.1f}')
    lines += ['', f'raw N = {int(raw.iloc[0]["n"])} daily points',
              f'thinned N = {int(thin.iloc[0]["n"])} (1 per {THIN_DAYS} d)',
              f'median residual lag-1 ρ (raw) = {raw["rho1"].median():.3f}',
              f'→ median effective N ≈ {raw["n_eff"].median():.0f}', '',
              'ΔAICc on the RAW series is inflated by',
              'autocorrelation; trust a ranking only where',
              'raw and thinned agree.']
    ax2.text(0.0, 1.0, '\n'.join(lines), family='monospace', fontsize=8.5,
             va='top', ha='left', transform=ax2.transAxes)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    main()
