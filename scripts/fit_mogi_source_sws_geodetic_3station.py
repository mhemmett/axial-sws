#!/usr/bin/env python3
"""
fit_mogi_source_sws_geodetic_3station.py

Fits a Mogi (1958) point-source inflation model jointly to co-located shear-wave-splitting
(SWS) and geodetic (BOTPT) data at three sites, per explicit user request:
    AXAS1  SWS + ASHES vent field BOTPT     (Western Caldera)
    AXCC1  SWS + Central Caldera BOTPT      (bpr_inflation_periods_ccal.py)
    AXEC2  SWS + Eastern Caldera BOTPT      (bpr_inflation_periods.py)
each pair assumed EXACTLY co-located at the seismometer site (per explicit user instruction --
in reality the BOTPT and seismometer at a given site are physically close but not identical
points; this is a simplifying assumption, not a measured fact).

Two epochs are fit independently: "Before Eruption" (t < 2015-04-24 06:00 UTC) and "During
Eruption" (2015-04-24 06:00 to 2015-05-19 00:00 UTC) -- windowcheck grade-3 SWS data (same
GRADE/load_station_raw/apply_grade convention as rose_7period_regions_windowcheck_grade3.py)
gives real values for both epochs at all three stations.

CRITICAL CAVEAT (per explicit user resolution of this exact issue): the ASHES BOTPT record does
not start until 2017-08-15, over two years after the eruption -- there is NO real "Before/
During Eruption" Uz value for AXAS1's geodetic pair. Per explicit user instruction, this is
handled by substituting, for AXAS1 ONLY, Uz_before = 0 (ASHES's own zero-inflation reference,
2017-08-20) and Uz_during = the latest available 30-day-rolling-mean uplift (effectively "how
much ASHES has re-inflated since its own reference low", NOT the 2015 eruption's coseismic
signal). AXCC1 and AXEC2 use their REAL Before/During Eruption Uz. This means the "Before/
During" contrast being fit is NOT physically the same event at all three sites -- two sites
(AXCC1, AXEC2) represent the actual 2015 eruption's deflation, one site (AXAS1) represents an
unrelated multi-year post-2017 re-inflation increment. This is reported prominently, not
glossed over: any source parameters "explaining" the AXAS1 data point should be read with that
caveat in mind.

AXEC2 also excludes its flagged "seismometer potentially unleveled" window (2021-07 to 2022-09,
see axec2_uplift_phi_cosine_vs_time.py) from any statistic computed over it, per explicit user
instruction -- implemented as a blanket filter even though neither the 2015 Before/During
Eruption windows nor AXEC2's own latest-value lookup actually falls inside that range (checked
below), so it has no numerical effect here but is applied defensively per instruction.

MODEL
-----
Mogi point source in an elastic half-space, source at (x0, y0, d) [km, d = depth below surface,
positive down], strength C (proportional to volume change dV; C = 3*dV/(4*pi)).  At a surface
station a horizontal distance r from the source epicenter:

    Uz(r)  = C * d / (d^2 + r^2)^1.5                              (vertical displacement)
    dsigma(r) = sigma_rr - sigma_theta = -6*mu*C*r^2/(d^2+r^2)^2.5  (radial minus tangential
                                                                     horizontal stress; derived
                                                                     below, always <= 0 for
                                                                     inflation (C>0) -- the
                                                                     tangential (hoop) stress is
                                                                     always the more compressive
                                                                     of the two, everywhere,
                                                                     for a pure Mogi point
                                                                     source)
    phi_pred(r) = tangential bearing at the station = (azimuth(source->station) + 90) mod 180
                  (fast direction aligns with the more compressive horizontal stress, per the
                  standard stress-aligned-crack/EDA convention -- see docstring derivation)

dt is related to |dsigma| by a single shared linear scaling K (folding together the crack
compliance/EDA proportionality constant and the shear modulus mu, since neither is
independently well constrained here): dt_pred = K * 6*C*r^2/(d^2+r^2)^2.5.

Free parameters per epoch: source location (x0, y0), depth d, strength C -- 4 unknowns.  K is
shared across both epochs (a material-response constant, not source-geometry-dependent) and fit
jointly. Total: 2*4 + 1 = 9 unknowns, from 2 epochs x 3 stations x 3 observables (Uz, phi, dt) =
18 data points -- nominally overdetermined, but with only 3 spatial points the source location/
depth are very weakly resolved (expect large, likely non-unique, parameter uncertainty; this is
an exploratory fit, not a resolved inversion -- treat results accordingly).

Run with:
    python3 fit_mogi_source_sws_geodetic_3station.py
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import sws_tomography_johnson2011 as J
from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg, ERUPTION_START, ERUPTION_END,
)
import bpr_inflation_periods as ecal_infl
import bpr_inflation_periods_ccal as ccal_infl
import bpr_inflation_periods_ashes as ashes_infl

STATIONS = ['AXAS1', 'AXCC1', 'AXEC2']
STA_XY = J.station_xy(STATIONS)

UNLEVEL_START = pd.Timestamp('2021-07-01', tz='UTC')
UNLEVEL_END = pd.Timestamp('2022-09-01', tz='UTC')


def mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    return float(np.mean(x)), (float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0), n


def sws_before_during(sta):
    """Real windowcheck grade-3 dt/phi means +/- SE for Before/During Eruption at one
    station. AXEC2's flagged unleveled window is excluded defensively (doesn't overlap
    either epoch, verified below, so has no numerical effect)."""
    raw = load_station_raw(sta)
    df = apply_grade(raw, GRADE)
    if sta == 'AXEC2':
        # The unleveled window (2021-07 to 2022-09) doesn't overlap either 2015 epoch
        # (Before Eruption ends 2015-04-24, During Eruption ends 2015-05-19) -- this
        # filter is applied anyway per explicit user instruction, but has no numerical
        # effect on THESE two epochs specifically (verified: neither epoch's date range
        # intersects the unleveled window).
        assert ERUPTION_END < UNLEVEL_START, 'unleveled window unexpectedly overlaps a 2015 epoch'
        excl = (df['t'] >= UNLEVEL_START) & (df['t'] < UNLEVEL_END)
        df = df[~excl]

    out = {}
    for label, mask in [('before', df['t'] < ERUPTION_START),
                        ('during', (df['t'] >= ERUPTION_START) & (df['t'] < ERUPTION_END))]:
        sub = df[mask]
        dt_m, dt_se, n = mean_se(sub['dt'].values)
        phi_m, phi_se = _circular_mean_and_se_deg(sub['phi_az'].values)
        out[label] = dict(dt=dt_m, dt_se=dt_se, phi=phi_m, phi_se=phi_se, n=n)
    return out


def uz_before_during(sta):
    """Uz (m) Before/During Eruption. AXCC1/AXEC2 use their REAL 2015 eruption values;
    AXAS1/ASHES substitutes zero-reference (before) vs latest available value (during) --
    see module docstring's caveat."""
    if sta == 'AXCC1':
        _dd, _infl, roll, _rt, _rd = ccal_infl.load_daily_series()
        before = roll.loc[roll.index < ERUPTION_START]
        during = roll.loc[(roll.index >= ERUPTION_START) & (roll.index < ERUPTION_END)]
        return dict(before=float(before.mean()), during=float(during.mean()), approx=False)
    if sta == 'AXEC2':
        _dd, _infl, roll, _rt, _rd = ecal_infl.load_daily_series()
        before = roll.loc[roll.index < ERUPTION_START]
        during = roll.loc[(roll.index >= ERUPTION_START) & (roll.index < ERUPTION_END)]
        return dict(before=float(before.mean()), during=float(during.mean()), approx=False)
    if sta == 'AXAS1':
        _dd, _infl, roll, _rt, _rd = ashes_infl.load_daily_series()
        roll_valid = roll.dropna()
        return dict(before=0.0, during=float(roll_valid.iloc[-1]), approx=True)
    raise ValueError(sta)


def build_dataset():
    rows = []
    for sta in STATIONS:
        sws = sws_before_during(sta)
        uz = uz_before_during(sta)
        sx, sy = STA_XY[sta]
        for epoch in ('before', 'during'):
            rows.append(dict(
                station=sta, epoch=epoch, x=sx, y=sy,
                dt=sws[epoch]['dt'], dt_se=sws[epoch]['dt_se'], n_sws=sws[epoch]['n'],
                phi=sws[epoch]['phi'], phi_se=sws[epoch]['phi_se'],
                uz=uz[epoch], uz_approx=uz['approx'],
            ))
    return pd.DataFrame(rows)


# ── Mogi forward model ──────────────────────────────────────────────────────────────────

def mogi_uz(x0, y0, d, C, sx, sy):
    r2 = (sx - x0) ** 2 + (sy - y0) ** 2
    return C * d / (d ** 2 + r2) ** 1.5


def mogi_dsigma(x0, y0, d, C, sx, sy):
    """sigma_rr - sigma_theta (proportional; the shared K below absorbs 2*mu)."""
    r2 = (sx - x0) ** 2 + (sy - y0) ** 2
    return -6.0 * C * r2 / (d ** 2 + r2) ** 2.5


def mogi_phi_pred(x0, y0, sx, sy):
    """Tangential bearing (deg, axial 0-180) at (sx,sy) from source (x0,y0)."""
    dx, dy = sx - x0, sy - y0
    radial_az = np.degrees(np.arctan2(dx, dy))
    return (radial_az + 90.0) % 180.0


def fit_epoch(df_epoch, K, x0_guess):
    """Fit (x0,y0,d,C) for one epoch given a FIXED shared K (called inside the outer
    joint optimizer, which also varies K)."""
    sx = df_epoch['x'].values
    sy = df_epoch['y'].values
    uz_obs = df_epoch['uz'].values
    dt_obs = df_epoch['dt'].values
    phi_obs = df_epoch['phi'].values

    uz_scale = max(np.nanstd(uz_obs), 1e-3)
    dt_scale = max(np.nanstd(dt_obs), 1e-3)

    def resid(p):
        x0, y0, d, C = p
        d = abs(d)
        uz_p = mogi_uz(x0, y0, d, C, sx, sy)
        dsig_p = mogi_dsigma(x0, y0, d, C, sx, sy)
        dt_p = K * (-dsig_p)
        phi_p = mogi_phi_pred(x0, y0, sx, sy)
        r_uz = (uz_p - uz_obs) / uz_scale
        r_dt = (dt_p - dt_obs) / dt_scale
        r_phi = 1.0 - np.cos(2.0 * np.radians(phi_obs - phi_p))
        return np.concatenate([r_uz, r_dt, r_phi])

    p0 = [x0_guess[0], x0_guess[1], 3.33, 0.01]
    sol = least_squares(resid, p0, method='lm', max_nfev=20000)
    return sol


def joint_objective_and_fit(df):
    """Outer loop over shared K: for each K, fit both epochs' (x0,y0,d,C) and return the
    total residual norm; use a 1-D scan + local refine since K is a single scalar."""
    x0_guess = (np.mean([xy[0] for xy in STA_XY.values()]),
               np.mean([xy[1] for xy in STA_XY.values()]))

    def total_cost(K):
        sols = {}
        cost = 0.0
        for epoch in ('before', 'during'):
            sol = fit_epoch(df[df['epoch'] == epoch], K, x0_guess)
            sols[epoch] = sol
            cost += float(np.sum(sol.fun ** 2))
        return cost, sols

    K_grid = np.logspace(-3, 3, 25)
    costs = []
    for K in K_grid:
        c, _ = total_cost(K)
        costs.append(c)
    best_idx = int(np.argmin(costs))
    K_best = K_grid[best_idx]

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda logK: total_cost(10 ** logK)[0],
                          bounds=(np.log10(K_best) - 1, np.log10(K_best) + 1),
                          method='bounded')
    K_final = 10 ** res.x
    _, sols_final = total_cost(K_final)
    return K_final, sols_final


def main():
    print(f'Stations: {STATIONS}  (assumed exactly co-located SWS+geodetic per explicit '
          f'user instruction)')
    print('Station (x,y) km:', {s: tuple(round(v, 3) for v in STA_XY[s]) for s in STATIONS})

    df = build_dataset()
    pd.set_option('display.width', 140)
    print('\n=== Input dataset ===')
    print(df.to_string(index=False))

    approx_rows = df[df['uz_approx']]
    if len(approx_rows):
        print('\n*** CAVEAT: Uz for the following rows is NOT the real 2015 eruption value '
              '(ASHES has no data until 2017-08-15) -- see module docstring ***')
        print(approx_rows[['station', 'epoch', 'uz']].to_string(index=False))

    print('\nFitting joint Mogi source (shared K, per-epoch x0,y0,d,C)...')
    K_final, sols = joint_objective_and_fit(df)
    print(f'\nShared stress-to-dt scaling K = {K_final:.4g}')

    for epoch in ('before', 'during'):
        sol = sols[epoch]
        x0, y0, d, C = sol.x
        d = abs(d)
        print(f'\n--- {epoch.upper()} ERUPTION epoch ---')
        print(f'  Fitted source: x0={x0:.3f} km, y0={y0:.3f} km, d={d:.3f} km, C={C:.5g}')
        print(f'  Residual norm: {np.sum(sol.fun**2):.4f}  (cost function value, dimensionless)')

        sub = df[df['epoch'] == epoch]
        for _, row in sub.iterrows():
            sx, sy = row['x'], row['y']
            uz_p = mogi_uz(x0, y0, d, C, sx, sy)
            dsig_p = mogi_dsigma(x0, y0, d, C, sx, sy)
            dt_p = K_final * (-dsig_p)
            phi_p = mogi_phi_pred(x0, y0, sx, sy)
            print(f'  {row["station"]:6s}: Uz obs={row["uz"]:+.4f} m, pred={uz_p:+.4f} m | '
                  f'dt obs={row["dt"]:.4f} s, pred={dt_p:.4f} s | '
                  f'phi obs={row["phi"]:.1f}, pred={phi_p:.1f} deg')

    print('\n=== Observed vs. predicted temporal rotation (delta-theta = during - before) ===')
    for sta in STATIONS:
        b = df[(df['station'] == sta) & (df['epoch'] == 'before')].iloc[0]
        du = df[(df['station'] == sta) & (df['epoch'] == 'during')].iloc[0]
        dtheta_obs = ((du['phi'] - b['phi'] + 90.0) % 180.0) - 90.0

        x0b, y0b, db, Cb = sols['before'].x
        x0d, y0d, dd, Cd = sols['during'].x
        phi_pred_before = mogi_phi_pred(x0b, y0b, b['x'], b['y'])
        phi_pred_during = mogi_phi_pred(x0d, y0d, du['x'], du['y'])
        dtheta_pred = ((phi_pred_during - phi_pred_before + 90.0) % 180.0) - 90.0

        print(f'  {sta:6s}: observed dtheta = {dtheta_obs:+.1f} deg, '
              f'model-predicted dtheta = {dtheta_pred:+.1f} deg '
              f'(from source shift ({x0b:.2f},{y0b:.2f}) -> ({x0d:.2f},{y0d:.2f}))')


if __name__ == '__main__':
    main()
