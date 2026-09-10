#!/usr/bin/env python3
"""
print_mean_phi_tables_5stations.py

Prints, for both the 7-period and annual 5-station rose plots (rose_7period_5stations_
newdata.py / rose_annual_5stations_newdata.py -- same data/QC/STATION_ORDER, imported
directly so this can never drift from those scripts), two tables per time-binning scheme:

  1. Circular mean fast direction (phi, deg, axial 0-180) per station x time period.
  2. Signed difference from that station's own PRE-ERUPTION circular mean phi, wrapped to
     (-90, +90] deg (the shortest angular rotation, since phi is axial/mod-180 -- e.g. a
     period whose mean sits at 179 deg when pre-eruption was at 1 deg is a 2 deg rotation,
     not 178).

Circular mean uses the doubled-angle trick (same convention as this repo's
circular_median_phi() in spatial_map_fast_direction_*.py, despite that function's name --
it's the circular MEAN: arctan2(mean(sin(2*phi)), mean(cos(2*phi)))/2).

QC tier used: tier 4 from rose_7period_5stations_newdata.TIERS -- the standard full filter
(quality>=0.5, dt<T_dom/2, phi_error<20 deg, dt_error<0.05s), NOT the stricter phi_error<10
deg tier 5 -- chosen as the more representative/higher-N production tier; rerun with
TIER_IDX=5 below for the stricter version if wanted.

Run with:
    python3 print_mean_phi_tables_5stations.py
"""

import numpy as np

from rose_7period_5stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, _build_time_periods,
)
from rose_annual_5stations_newdata import build_annual_periods

TIER_IDX = 4   # + dt_err<0.05s (standard QC, not the stricter phi_err<10 tier)


def circular_mean_phi(phi_deg):
    """Circular mean of axially-symmetric (mod 180) fast directions (doubled-angle trick)."""
    if len(phi_deg) == 0:
        return np.nan
    angles = 2.0 * np.radians(phi_deg)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return float(np.degrees(np.arctan2(s, c)) / 2.0) % 180.0


def circular_diff_deg(mean_phi, pre_phi):
    """Signed shortest rotation from pre_phi to mean_phi, both axial (mod 180),
    wrapped to (-90, +90] deg."""
    if not (np.isfinite(mean_phi) and np.isfinite(pre_phi)):
        return np.nan
    return ((mean_phi - pre_phi + 90.0) % 180.0) - 90.0


def _subset(df, t0, t1):
    import pandas as pd
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None:
        m = m & (df['t'] < t1)
    return df[m]


def build_tables(periods, dfs):
    period_labels = [lbl.splitlines()[0] for lbl, _t0, _t1 in periods]
    mean_table = np.full((len(STATION_ORDER), len(periods)), np.nan)
    diff_table = np.full((len(STATION_ORDER), len(periods)), np.nan)
    n_table = np.zeros((len(STATION_ORDER), len(periods)), dtype=int)
    pre_means = np.full(len(STATION_ORDER), np.nan)

    for si, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        pre_lbl, pre_t0, pre_t1 = periods[0]
        pre_sub = _subset(df, pre_t0, pre_t1)
        pre_mean = circular_mean_phi(pre_sub['phi_az'].values)
        pre_means[si] = pre_mean

        for pi, (_lbl, t0, t1) in enumerate(periods):
            sub = _subset(df, t0, t1)
            m = circular_mean_phi(sub['phi_az'].values)
            mean_table[si, pi] = m
            diff_table[si, pi] = circular_diff_deg(m, pre_mean)
            n_table[si, pi] = len(sub)

    return period_labels, mean_table, diff_table, n_table, pre_means


def print_table(title, period_labels, table, fmt='{:.1f}'):
    col_w = max(len(lbl) for lbl in period_labels) + 2
    row_w = max(len(s) for s in STATION_ORDER) + 2
    print(f'\n{title}')
    print(' ' * row_w + ''.join(f'{lbl:>{col_w}}' for lbl in period_labels))
    for si, sta in enumerate(STATION_ORDER):
        row = f'{sta:<{row_w}}' + ''.join(
            (f'{table[si, pi]:>{col_w}.1f}' if np.isfinite(table[si, pi])
             else f'{"n/a":>{col_w}}')
            for pi in range(len(period_labels)))
        print(row)


def print_diff_table_with_pre_angle(title, period_labels, diff_table, pre_means):
    """Same as print_table but with an extra leading 'Pre-eruption angle' column
    showing each station's actual pre-eruption circular mean phi (deg), ahead of
    the per-period differences."""
    pre_col_label = 'Pre-eruption angle'
    col_w = max(max(len(lbl) for lbl in period_labels), len(pre_col_label)) + 2
    row_w = max(len(s) for s in STATION_ORDER) + 2
    print(f'\n{title}')
    print(' ' * row_w + f'{pre_col_label:>{col_w}}'
          + ''.join(f'{lbl:>{col_w}}' for lbl in period_labels))
    for si, sta in enumerate(STATION_ORDER):
        pre_str = f'{pre_means[si]:>{col_w}.1f}' if np.isfinite(pre_means[si]) else f'{"n/a":>{col_w}}'
        row = f'{sta:<{row_w}}' + pre_str + ''.join(
            (f'{diff_table[si, pi]:>{col_w}.1f}' if np.isfinite(diff_table[si, pi])
             else f'{"n/a":>{col_w}}')
            for pi in range(len(period_labels)))
        print(row)


def main():
    print('Loading new mfast max_dt=0.2s splitting results (AXCC1, AXEC1, AXEC3, AXAS1: '
          'complete; AXEC2 2015-2021: partial, in-progress re-run)...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
    dfs = {sta: apply_tier(raw[sta], TIER_IDX) for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        print(f'  {sta}: {len(dfs[sta]):,} QC-passing events (tier {TIER_IDX})')

    import pandas as pd
    all_combined = pd.concat(dfs.values(), ignore_index=True)

    # ── 7-period ─────────────────────────────────────────────────────────────
    periods_7 = _build_time_periods(all_combined)
    labels_7, mean_7, diff_7, n_7, pre_7 = build_tables(periods_7, dfs)
    print('\n' + '=' * 70)
    print('7-PERIOD (eruption-relative)')
    print_table('Circular mean fast direction phi (deg, axial 0-180):', labels_7, mean_7)
    print_diff_table_with_pre_angle(
        'Difference from pre-eruption mean phi (deg, signed, wrapped to +/-90), '
        'with the pre-eruption angle itself:', labels_7, diff_7, pre_7)

    # ── Annual ───────────────────────────────────────────────────────────────
    periods_a = build_annual_periods()
    labels_a, mean_a, diff_a, n_a, pre_a = build_tables(periods_a, dfs)
    print('\n' + '=' * 70)
    print('ANNUAL')
    print_table('Circular mean fast direction phi (deg, axial 0-180):', labels_a, mean_a)
    print_diff_table_with_pre_angle(
        'Difference from pre-eruption mean phi (deg, signed, wrapped to +/-90), '
        'with the pre-eruption angle itself:', labels_a, diff_a, pre_a)


if __name__ == '__main__':
    main()
