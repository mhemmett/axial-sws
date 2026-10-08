#!/usr/bin/env python3
"""
atan2_beta_profile_likelihood_ccal.py

Is the atan2 vector-sum model's inflation azimuth beta actually constrained by the data?

The linearised covariance of the production fits gives corr(A, beta) = +-1.000 for 7 of 8
series (6 stations + West/East averages), which says A and beta trade off along a ridge -- but a
linearised covariance is only a local approximation. A profile likelihood is the decisive test:
fix beta on a grid over the full circle, refit everything else (C1, C2, A; alpha and the turnover
u0 stay fixed exactly as in production), and look at how the best achievable misfit changes.

    Delta chi^2(beta) = N_eff * ln( SSE(beta) / SSE_min )      (Gaussian-errors LR statistic)

95% interval: Delta chi^2 < 3.84 (chi^2, 1 dof). N_eff corrects for the strong autocorrelation
of the 30-day rolled series the fit is made on: N_eff = N (1 - rho1) / (1 + rho1), rho1 = lag-1
autocorrelation of the residual at the optimum. The naive N-based interval is also reported --
it is much too narrow and is shown only to make that point.

Series, pooling, the AXEC2 unleveled-window drop and the production fit are identical to
uplift_vs_phi_regions_scatter_atan2fit_ccal_30day.py (region_fit).

beta is scanned over the full 360 deg: beta and beta+180 are NOT equivalent in this model (the
uplift term of infl_mag changes sign). Reference lines mark alpha (170 deg) and alpha+180.

Outputs:
    atan2_beta_profile_likelihood_ccal.pdf / .png
    atan2_beta_profile_likelihood_ccal.csv   (per series: beta_hat, N, N_eff, 95% interval width)
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
from scipy.optimize import curve_fit

from animate_arctan_stress_vectors import ALPHA_FIXED_RAD, ALPHA_FIXED_AZ_DEG, BOUND, A_MIN, _azimuth
from phi_uplift_residual_periodogram_ccal import SERIES
from uplift_vs_phi_regions_scatter_atan2fit_ccal_30day import region_fit
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'atan2_beta_profile_likelihood_ccal')

BETA_STEP_DEG = 2.0
CHI2_95_1DOF = 3.841


def fixed_beta_model(u0, beta):
    def model(u, C1, C2, A):
        infl_mag = (u - u0) / A - A * np.cos(ALPHA_FIXED_RAD - beta)
        Y = A * np.sin(ALPHA_FIXED_RAD) + infl_mag * np.sin(beta)
        X = A * np.cos(ALPHA_FIXED_RAD) + infl_mag * np.cos(beta)
        return C1 + C2 * np.arctan2(Y, X)
    return model


def best_fixed_beta(x, y, u0, beta, warm):
    model = fixed_beta_model(u0, beta)
    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    span = float(y_hi - y_lo) or 1.0
    starts = [] if warm is None else [warm]
    starts += [[c1, s * span / np.pi, a] for c1 in (y_lo, y_hi, float(np.mean(y)))
               for s in (1.0, -1.0) for a in (0.3, 3.0)]
    bounds = ([-BOUND, -BOUND, A_MIN], [BOUND, BOUND, BOUND])
    best = None
    for p0 in starts:
        p0 = [float(np.clip(p0[0], -BOUND, BOUND)), float(np.clip(p0[1], -BOUND, BOUND)),
              float(np.clip(p0[2], A_MIN, BOUND))]
        try:
            popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((model(x, *popt) - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt)
    return best


def interval_width(betas_deg, dchi2):
    """Total angular width (deg) of the beta set inside the 95% region (may be disjoint)."""
    return float(np.sum(dchi2 < CHI2_95_1DOF) * BETA_STEP_DEG)


def main():
    _dd, _ir_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    betas_deg = np.arange(0.0, 360.0, BETA_STEP_DEG)

    fig, axes = plt.subplots(2, 4, figsize=(18, 7.6), sharex=True)
    rows = []
    for ax, (label, members, color) in zip(axes.ravel(), SERIES):
        with contextlib.redirect_stdout(io.StringIO()):
            fit = region_fit(label, members, inflation_roll)
        x, y, u0 = fit['x'], fit['y'], fit['u0']

        prod_model = fit['model']
        prod_pred = prod_model(x, fit['C1'], fit['C2'], fit['A'], fit['beta'])
        sse_prod = float(np.sum((prod_pred - y) ** 2))
        res = y - prod_pred
        rho1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        n = len(x)
        n_eff = max(2.0, n * (1.0 - rho1) / (1.0 + rho1))

        # beta in the model is a math angle (radians, CCW from E); scan in AZIMUTH for reading.
        sse = np.full(len(betas_deg), np.nan)
        warm = [fit['C1'], fit['C2'], fit['A']]
        for k, b_az in enumerate(betas_deg):
            beta = np.radians(90.0 - b_az)
            best = best_fixed_beta(x, y, u0, beta, warm)
            if best is not None:
                sse[k] = best[0]
                warm = list(best[1])
        sse_min = float(min(np.nanmin(sse), sse_prod))
        dchi2_eff = n_eff * np.log(sse / sse_min)
        dchi2_naive = n * np.log(sse / sse_min)
        w_eff = interval_width(betas_deg, dchi2_eff)
        w_naive = interval_width(betas_deg, dchi2_naive)
        beta_hat = float(fit['beta_az'])
        # dynamic range of the profile: the worst beta's penalty, in N_eff units
        dmax = float(np.nanmax(dchi2_eff))
        print(f'{label:16s} beta_hat={beta_hat:6.1f}  N={n}  rho1={rho1:.3f}  N_eff={n_eff:.0f}  '
              f'95% width: N_eff {w_eff:.0f} deg | naive {w_naive:.0f} deg  '
              f'max dchi2_eff={dmax:.1f}')
        rows.append(dict(series=label, beta_hat_az=beta_hat, N=n, rho1=rho1, N_eff=n_eff,
                         width95_Neff_deg=w_eff, width95_naive_deg=w_naive,
                         max_dchi2_Neff=dmax, sse_prod=sse_prod, sse_profile_min=sse_min))

        ax.axhline(CHI2_95_1DOF, color='0.15', lw=0.9, ls=':')
        for a_ref in (ALPHA_FIXED_AZ_DEG % 360, (ALPHA_FIXED_AZ_DEG + 180) % 360):
            ax.axvline(a_ref, color='0.8', lw=0.9, zorder=0)
        ax.plot(betas_deg, dchi2_eff, color=color, lw=1.6, zorder=2)
        ax.axvline(beta_hat, color=color, lw=0.9, ls='--', zorder=1)
        ax.set_yscale('symlog', linthresh=1.0)
        ax.set_ylim(-0.1, max(10.0, dmax * 1.3))
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 90))
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left')
        ax.text(1.0, 1.01, f'N_eff = {n_eff:.0f} (N = {n}); 95% set = {w_eff:.0f}° of 360°',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.35')
        ax.grid(alpha=0.25)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        print(f'    (profile min SSE {sse_min:.1f} vs production SSE {sse_prod:.1f})')

    for ax in axes[1]:
        ax.set_xlabel('Fixed inflation azimuth β (deg)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Δχ² (effective N)')
    handles = [plt.Line2D([], [], color='0.15', lw=0.9, ls=':'),
               plt.Line2D([], [], color='0.4', lw=0.9, ls='--'),
               plt.Line2D([], [], color='0.8', lw=0.9)]
    fig.legend(handles, ['95% (Δχ² = 3.84, 1 dof)', 'Production β̂', 'α = 170° and α + 180°'],
               loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Profile likelihood of the atan2 inflation azimuth β (C1, C2, A refit at each β)',
                 fontsize=15, fontweight='bold', y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f'{OUT_BASE}.csv', index=False)
    print(f'Saved {OUT_BASE}.pdf / .png / .csv')


if __name__ == '__main__':
    main()
