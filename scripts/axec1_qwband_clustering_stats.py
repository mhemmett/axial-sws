#!/usr/bin/env python3
"""
axec1_qwband_clustering_stats.py

Quantitative companion to lqt_pykonal_tomography_source_voxel_per_station_
qwband_pm05.py: does AXEC1 have an outsized share of "indeterminate quality"
(-0.5 < Q_w < 0.5) measurements relative to the other 5 stations, and do
AXEC1's mid-band events sit systematically closer to the station (in map
distance and/or depth) than its confidently-split (Q_w >= 0.7) events?

This is the quantitative test behind the 2026-07-08/09 hypothesis that
AXEC1's isotropic-looking null population is a real null from a magma
conduit with strong short-scale anisotropy variation, not a QC/false-null
artifact (Wustefeld et al. 2010).

Reuses lqt_pykonal_tomography_source_voxel_anisotropy.prepare_source_voxel_
data() for the aligned (all_df, summary) pair (event x/y/z + cached Q_w/dt/
errors, positionally verified against each other) and
lqt_pykonal_tomography_time_period_difference._qw_band_mask() for the
mid-band definition, so none of the thresholds here can drift from the
tomography scripts.

Prints, per station:
  - n passing the standard dt_err/phi_err/dt-cutoff QC ("eligible")
  - count and fraction in the Q_w mid-band, in the strict split band
    (Q_w>=0.7), and in a symmetric confident-null band (Q_w<=-0.7)
  - a chi-square test of independence (station x {mid-band, not}) across all
    6 stations, plus a 2x2 proportion test of AXEC1 vs. the pooled other 5
  - for AXEC1 (and, for context, all other stations): Mann-Whitney U
    comparison of event-to-station map distance and depth, mid-band vs.
    strict-band events -- tests whether the mid-band population sits closer
    to the station than the confidently-split population.

Run with:
    python3 axec1_qwband_clustering_stats.py
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

import lqt_pykonal_tomography_raylength_anisotropy as rla
from lqt_pykonal_tomography_source_voxel_anisotropy import prepare_source_voxel_data
from lqt_pykonal_tomography_time_period_difference import _qw_band_mask

FOCUS_STATION = 'AXEC1'
NULL_HI = -0.7   # confident-null band, symmetric with the existing strict-split Q_w>=0.7


def eligible_mask(summary):
    ok = summary['trace_ok'].values.astype(bool)
    dte = summary['dt_error'].values
    phe = summary['phi_error'].values
    dt = summary['dt'].values
    return ok & (dte < rla.DT_ERR_MAX) & (phe < rla.PHI_ERR_MAX) & (dt <= rla.DT_CUTOFF)


def main():
    all_df, summary, a_ray_pct, in_grid, _page_filters, event_ns = prepare_source_voxel_data()

    station = summary['station'].values
    q = summary['quality'].values
    elig = eligible_mask(summary)
    mid_mask, mid_label = _qw_band_mask(summary)
    strict_mask = elig & (q >= 0.7)
    null_mask = elig & (q <= NULL_HI)

    print(f'Mid-band definition: {mid_label}')
    print(f'Confident-null band: Q_w <= {NULL_HI} (symmetric with the strict Q_w>=0.7 split band)\n')

    # ── Per-station counts/fractions ────────────────────────────────────────
    rows = []
    for sta in rla.STATIONS:
        sbool = (station == sta)
        n_elig = int((elig & sbool).sum())
        n_mid = int((mid_mask & sbool).sum())
        n_strict = int((strict_mask & sbool).sum())
        n_null = int((null_mask & sbool).sum())
        rows.append(dict(
            station=sta, n_eligible=n_elig,
            n_mid=n_mid, frac_mid=n_mid / n_elig if n_elig else np.nan,
            n_strict=n_strict, frac_strict=n_strict / n_elig if n_elig else np.nan,
            n_null=n_null, frac_null=n_null / n_elig if n_elig else np.nan,
        ))
    tbl = pd.DataFrame(rows).set_index('station')
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print('Per-station Q_w-band breakdown (of QC-eligible measurements):')
    print(tbl.to_string())
    print()

    # ── Chi-square: station x {mid-band, not-mid-band}, over eligible rays ──
    contingency = np.array([
        [int((mid_mask & (station == sta)).sum()),
         int((elig & ~mid_mask & (station == sta)).sum())]
        for sta in rla.STATIONS
    ])
    chi2, p_chi2, dof, _exp = stats.chi2_contingency(contingency)
    print(f'Chi-square test, station x {{mid-band, not}} (all 6 stations): '
          f'chi2={chi2:.2f}, dof={dof}, p={p_chi2:.3g}')

    # ── AXEC1 vs. pooled other 5: 2x2 proportion test ───────────────────────
    foc = (station == FOCUS_STATION)
    n_mid_foc = int((mid_mask & foc).sum())
    n_elig_foc = int((elig & foc).sum())
    n_mid_rest = int((mid_mask & elig & ~foc).sum())
    n_elig_rest = int((elig & ~foc).sum())
    table_2x2 = [[n_mid_foc, n_elig_foc - n_mid_foc],
                 [n_mid_rest, n_elig_rest - n_mid_rest]]
    chi2_2, p_2, _dof2, _exp2 = stats.chi2_contingency(table_2x2, correction=True)
    print(f'\n{FOCUS_STATION} vs. pooled other 5 -- mid-band fraction: '
          f'{n_mid_foc}/{n_elig_foc} ({n_mid_foc / n_elig_foc:.1%}) vs. '
          f'{n_mid_rest}/{n_elig_rest} ({n_mid_rest / n_elig_rest:.1%}); '
          f'chi2={chi2_2:.2f}, p={p_2:.3g}')

    # ── Spatial concentration: event-to-own-station distance & depth,      ──
    # ── mid-band vs. strict-band, per station                              ──
    print('\nEvent-to-own-station map distance (km) and depth (km): '
          'mid-band vs. strict-band (Mann-Whitney U, two-sided)')
    x = all_df['x'].values
    y = all_df['y'].values
    z = all_df['z'].values
    for sta in rla.STATIONS:
        if sta not in rla.sta_xy:
            print(f'  {sta}: no station (x,y) available, skipping')
            continue
        sx, sy = rla.sta_xy[sta]
        dist = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        sbool = (station == sta)
        mid_d = dist[mid_mask & sbool]
        strict_d = dist[strict_mask & sbool]
        mid_z = z[mid_mask & sbool]
        strict_z = z[strict_mask & sbool]
        if len(mid_d) < 5 or len(strict_d) < 5:
            print(f'  {sta}: too few events (mid={len(mid_d)}, strict={len(strict_d)}), skipping test')
            continue
        u_d, p_d = stats.mannwhitneyu(mid_d, strict_d, alternative='two-sided')
        u_z, p_z = stats.mannwhitneyu(mid_z, strict_z, alternative='two-sided')
        flag = '  <== focus station' if sta == FOCUS_STATION else ''
        print(f'  {sta}: n_mid={len(mid_d):5d} n_strict={len(strict_d):5d}  '
              f'median dist mid={np.median(mid_d):.2f}km strict={np.median(strict_d):.2f}km (p={p_d:.3g})  '
              f'median depth mid={np.median(mid_z):.2f}km strict={np.median(strict_z):.2f}km (p={p_z:.3g}){flag}')

    print('\nDone.')


if __name__ == '__main__':
    main()
