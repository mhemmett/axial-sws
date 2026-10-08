#!/usr/bin/env python3
"""
phi_uplift_residual_wavelet_ccal.py

Transient-oscillation companion to phi_uplift_residual_periodogram_ccal.py. The periodogram tests
for oscillations that persist through the whole record and finds none; a wavelet scalogram can
catch an oscillation that exists only for a few cycles somewhere in time.

Signal: the same daily residual about each series' atan2 fit, as s = sin(2r) (wrap-safe for
axial phi), placed on the full daily grid with missing days set to the mean (0 after demeaning).

Transform: Morlet (omega0 = 6) continuous wavelet transform after Torrence & Compo (1998), via FFT,
scales from 2 d with dj = 1/8, power normalised by the series variance.

Guards against false positives:
  * cone of influence (edge effects) and a COVERAGE mask: cells where fewer than MIN_COVER of the
    days within +-1/2 period carry data are excluded (zero-filled gaps would otherwise look like
    signal-free stretches or create edge artefacts);
  * pointwise 95% significance against the Torrence & Compo AR(1) red-noise background
    (rho1 from consecutive observed days, chi^2_2 / 2);
  * AREA-WISE test: a pointwise test marks ~5% of the plane significant under pure noise, so a
    contiguous significant patch only counts if its duration in CYCLES exceeds the 95th
    percentile of the largest patch found in N_SURR AR(1) surrogates sampled on the same days
    and run through the identical pipeline.

Outputs:
    phi_uplift_residual_wavelet_ccal.pdf / .png
    phi_uplift_residual_wavelet_ccal_patches.csv   (patches passing the area-wise test)
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
import matplotlib.dates as mdates
from scipy import ndimage

from phi_uplift_residual_periodogram_ccal import SERIES, daily_residuals
from uplift_vs_phi_regions_scatter_atan2fit_ccal_30day import region_fit
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'phi_uplift_residual_wavelet_ccal')

OMEGA0 = 6.0
FOURIER_FACTOR = 4.0 * np.pi / (OMEGA0 + np.sqrt(2.0 + OMEGA0 ** 2))   # ~1.033
S0_DAYS, DJ = 2.0, 0.125
P_MAX_DAYS = 1500.0
MIN_COVER = 0.5
CHI2_2_95 = 5.991
N_SURR = 60
SEED = 20261008


def cwt_morlet(x, dt=1.0):
    n0 = len(x)
    pad = int(2 ** np.ceil(np.log2(n0)) * 2)
    xp = np.zeros(pad)
    xp[:n0] = x
    f = np.fft.fft(xp)
    k = 2.0 * np.pi * np.fft.fftfreq(pad, d=dt)
    j_max = int(np.floor(np.log2(P_MAX_DAYS / FOURIER_FACTOR / S0_DAYS) / DJ))
    scales = S0_DAYS * 2.0 ** (DJ * np.arange(j_max + 1))
    W = np.empty((len(scales), n0), dtype=complex)
    for i, s in enumerate(scales):
        daughter = (np.pi ** -0.25) * np.sqrt(2.0 * np.pi * s / dt) * \
            np.exp(-0.5 * (s * k - OMEGA0) ** 2) * (k > 0)
        W[i] = np.fft.ifft(f * daughter)[:n0]
    return W, scales * FOURIER_FACTOR


def coi_mask(n, periods):
    idx = np.arange(n)
    edge = np.minimum(idx, n - 1 - idx)
    coi_period = FOURIER_FACTOR / np.sqrt(2.0) * edge   # dt = 1 d
    return periods[:, None] <= coi_period[None, :]


def coverage(present, periods):
    cov = np.empty((len(periods), len(present)))
    for i, P in enumerate(periods):
        w = max(1, int(round(P)))
        cov[i] = ndimage.uniform_filter1d(present.astype(float), size=w, mode='constant')
    return cov


def red_noise_signif(rho, periods, n):
    freq = 1.0 / periods
    pk = (1.0 - rho ** 2) / (1.0 + rho ** 2 - 2.0 * rho * np.cos(2.0 * np.pi * freq))
    return pk * CHI2_2_95 / 2.0


def lag1(day_index, y):
    s = pd.Series(y, index=day_index)
    nxt = s.reindex(s.index + 1)
    ok = ~np.isnan(nxt.values)
    if ok.sum() < 10:
        return 0.0
    return float(np.clip(np.corrcoef(s.values[ok], nxt.values[ok])[0, 1], 0.0, 0.99))


def analyse(grid, present, rho, periods_ref=None):
    """Return (power, sig_mask, valid_mask, periods)."""
    var = np.var(grid[present])
    W, periods = cwt_morlet(grid)
    power = np.abs(W) ** 2 / var
    valid = coi_mask(len(grid), periods) & (coverage(present, periods) >= MIN_COVER)
    sig = (power / red_noise_signif(rho, periods, len(grid))[:, None] > 1.0) & valid
    return power, sig, valid, periods


def patches(sig, periods):
    """Contiguous significant patches; duration in cycles = time extent / median period."""
    lab, n = ndimage.label(sig)
    out = []
    for p in range(1, n + 1):
        rr, cc = np.nonzero(lab == p)
        med_period = float(np.median(periods[rr]))
        dur = cc.max() - cc.min() + 1
        out.append(dict(cycles=dur / med_period, p_lo=float(periods[rr].min()),
                        p_hi=float(periods[rr].max()), i0=int(cc.min()), i1=int(cc.max()),
                        mask=(lab == p)))
    return out


def main():
    _dd, _ir_raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharey=True)
    rows = []
    for ax, (label, members, color) in zip(axes.ravel(), SERIES):
        with contextlib.redirect_stdout(io.StringIO()):
            fit = region_fit(label, members, inflation_roll)
        d = daily_residuals(members, fit, inflation_roll)
        t0 = d['t'].min()
        n = int(d['day_index'].max()) + 1
        s = np.sin(2.0 * np.radians(d['resid_deg'].values))
        s = s - s.mean()
        grid = np.zeros(n)
        present = np.zeros(n, dtype=bool)
        grid[d['day_index'].values] = s
        present[d['day_index'].values] = True
        rho = lag1(d['day_index'].values, s)

        power, sig, valid, periods = analyse(grid, present, rho)
        frac_sig = float(sig.sum() / max(valid.sum(), 1))

        # area-wise null: largest patch (in cycles) in AR(1) surrogates on the same days
        null_max = []
        for _ in range(N_SURR):
            e = rng.standard_normal(n)
            x = np.empty(n)
            x[0] = e[0]
            for i in range(1, n):
                x[i] = rho * x[i - 1] + np.sqrt(1.0 - rho ** 2) * e[i]
            g = np.where(present, x, 0.0)
            g[present] -= g[present].mean()
            _, sg, _, _ = analyse(g, present, rho)
            ps = patches(sg, periods)
            null_max.append(max((q['cycles'] for q in ps), default=0.0))
        cyc95 = float(np.percentile(null_max, 95))

        real = [q for q in patches(sig, periods) if q['cycles'] > cyc95]
        dates = t0 + pd.to_timedelta(np.arange(n), unit='D')
        print(f'{label:16s} days={present.sum():5d} rho1={rho:.2f}  pointwise-significant '
              f'fraction={frac_sig*100:.1f}% (noise expectation ~5%)  area-wise threshold='
              f'{cyc95:.1f} cycles  patches passing: {len(real)}')
        for q in sorted(real, key=lambda q: -q['cycles']):
            print(f'    {q["p_lo"]:.0f}-{q["p_hi"]:.0f} d, {dates[q["i0"]].date()} to '
                  f'{dates[q["i1"]].date()}, {q["cycles"]:.1f} cycles')
            rows.append(dict(series=label, period_lo_d=q['p_lo'], period_hi_d=q['p_hi'],
                             start=dates[q['i0']].date(), end=dates[q['i1']].date(),
                             cycles=q['cycles'], areawise_threshold_cycles=cyc95,
                             pointwise_sig_fraction=frac_sig))

        tnum = mdates.date2num(dates.to_pydatetime())
        lp = np.log2(power)
        lp_plot = np.where(valid, lp, np.nan)
        im = ax.pcolormesh(tnum, periods, lp_plot, cmap='Blues', vmin=-2, vmax=4,
                           shading='auto', rasterized=True)
        ax.pcolormesh(tnum, periods, np.where(valid, np.nan, 1.0), cmap='Greys', vmin=0,
                      vmax=4, shading='auto', rasterized=True, alpha=0.6)
        ax.contour(tnum, periods, sig.astype(float), levels=[0.5], colors='0.35',
                   linewidths=0.4)
        for q in real:
            ax.contour(tnum, periods, q['mask'].astype(float), levels=[0.5], colors='#DC267F',
                       linewidths=1.6)
        ax.set_yscale('log')
        ax.set_ylim(periods[0], P_MAX_DAYS)
        ax.invert_yaxis()
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left')
        ax.text(1.0, 1.01, f'{frac_sig*100:.1f}% pointwise-sig.; {len(real)} area-wise',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.35')

    for ax in axes[:, 0]:
        ax.set_ylabel('Period (days)')
    cax = fig.add_axes([0.92, 0.25, 0.01, 0.5])
    fig.colorbar(im, cax=cax, label='log₂ normalised wavelet power')
    handles = [plt.Line2D([], [], color='0.35', lw=0.6),
               plt.Line2D([], [], color='#DC267F', lw=1.6),
               plt.Rectangle((0, 0), 1, 1, color='0.75')]
    fig.legend(handles, ['Pointwise 95% vs red noise',
                         'Patch passing area-wise test (longer than 95% of noise patches)',
                         'Cone of influence / < 50% data coverage'],
               loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.47, -0.02))
    fig.suptitle('Wavelet scalograms of the φ residual about each atan2 fit (Morlet, daily, sin 2r)',
                 fontsize=15, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.12, hspace=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(f'{OUT_BASE}_patches.csv', index=False)
    print(f'Saved {OUT_BASE}.pdf / .png and _patches.csv ({len(rows)} patches)')


if __name__ == '__main__':
    main()
