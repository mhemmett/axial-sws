#!/usr/bin/env python3
"""
print_pre_syn_binned_phi_dt_tables_5stations.py

For the 5 new-data stations (AXAS1, AXCC1, AXEC1, AXEC2, AXEC3 -- AXAS2 excluded, no new-data
rerun yet), same tier-4 QC as this session's standard (quality>=0.5, dt<T_dom/2, phi_error<20
deg, dt_error<0.05s), imported directly from rose_7period_5stations_newdata.py:

  1. PRE-ERUPTION: split each station's own events (sorted by time) into 10 equal-COUNT bins.
     Print per-bin mean dt and circular mean phi, per station -- is there a significant change
     across these bins, and if so, in which bin does it happen?
  2. SYN-ERUPTION (2015-04-24 06:00 to 2015-05-19 00:00 UTC): same 10 equal-count bins, same
     two tables.
  3. FIRST 19 HOURS of the eruption (2015-04-24 06:00 to 2015-05-25 01:00 UTC): fixed HOURLY
     bins (not equal-count -- eruption onset is a fixed clock-time window), same two tables
     (dt, phi), one row per hour.

Each table cell shows "value (N)" -- the per-bin/per-hour statistic and its event count --
since some bins (especially AXCC1's small syn-eruption sample, and the first-19h hourly bins)
have low N and should be read with that in mind.

Run with:
    python3 print_pre_syn_binned_phi_dt_tables_5stations.py
"""

import numpy as np
import pandas as pd

from rose_7period_5stations_newdata import STATION_ORDER, load_station_raw, apply_tier

TIER_IDX = 4   # quality>=0.5, dt<T_dom/2, phi_error<20deg, dt_error<0.05s

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')
N_HOURS_ZOOM = 19

N_BINS = 10


def circular_mean_phi(phi_deg):
    if len(phi_deg) == 0:
        return np.nan
    angles = 2.0 * np.radians(phi_deg)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return float(np.degrees(np.arctan2(s, c)) / 2.0) % 180.0


def circular_diff_deg(a, b):
    """Signed shortest rotation from b to a, both axial (mod 180), wrapped to (-90, +90]."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return ((a - b + 90.0) % 180.0) - 90.0


def equal_count_bins(df, n_bins):
    """Split df (must have 't') into n_bins equal-COUNT bins ordered by time.
    Returns list of dicts: label, t0, t1, n, mean_dt, mean_phi."""
    d = df.sort_values('t').reset_index(drop=True)
    n = len(d)
    bins = []
    if n == 0:
        return [dict(label=f'Bin {i+1}', t0=None, t1=None, n=0, mean_dt=np.nan, mean_phi=np.nan)
                for i in range(n_bins)]
    edges = [int(round(i * n / n_bins)) for i in range(n_bins + 1)]
    edges[-1] = n
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sub = d.iloc[lo:hi]
        if len(sub) == 0:
            bins.append(dict(label=f'Bin {i+1}', t0=None, t1=None, n=0,
                              mean_dt=np.nan, mean_phi=np.nan))
            continue
        t0, t1 = sub['t'].iloc[0], sub['t'].iloc[-1]
        bins.append(dict(
            label=f'Bin {i+1}\n{t0.strftime("%Y-%m-%d")}\n{t1.strftime("%Y-%m-%d")}',
            t0=t0, t1=t1, n=len(sub),
            mean_dt=float(sub['dt'].mean()),
            mean_phi=circular_mean_phi(sub['phi_az'].values)))
    return bins


def hourly_bins(df, t_start, n_hours):
    """Fixed hourly bins starting at t_start, n_hours of them."""
    bins = []
    for h in range(n_hours):
        h0 = t_start + pd.Timedelta(hours=h)
        h1 = t_start + pd.Timedelta(hours=h + 1)
        sub = df[(df['t'] >= h0) & (df['t'] < h1)]
        bins.append(dict(
            label=f'Hour {h}\n{h0.strftime("%m-%d %H:%M")}',
            t0=h0, t1=h1, n=len(sub),
            mean_dt=float(sub['dt'].mean()) if len(sub) else np.nan,
            mean_phi=circular_mean_phi(sub['phi_az'].values) if len(sub) else np.nan))
    return bins


def print_metric_table(title, bins_per_station, metric_key, fmt='{:.3f}'):
    labels = [b['label'].splitlines()[0] for b in bins_per_station[STATION_ORDER[0]]]
    col_w = 16
    row_w = max(len(s) for s in STATION_ORDER) + 2
    print(f'\n{title}')
    print(' ' * row_w + ''.join(f'{lbl:>{col_w}}' for lbl in labels))
    for sta in STATION_ORDER:
        row_cells = []
        for b in bins_per_station[sta]:
            val = b[metric_key]
            n = b['n']
            if np.isfinite(val):
                cell = f'{fmt.format(val)} ({n})'
            else:
                cell = f'n/a ({n})'
            row_cells.append(f'{cell:>{col_w}}')
        print(f'{sta:<{row_w}}' + ''.join(row_cells))


def report_biggest_jump(bins_per_station, metric_key, label, is_phi=False):
    print(f'\nBiggest bin-to-bin jump per station, {label}:')
    for sta in STATION_ORDER:
        bins = bins_per_station[sta]
        vals = [b[metric_key] for b in bins]
        best_i, best_jump = None, -1
        for i in range(1, len(vals)):
            if not (np.isfinite(vals[i]) and np.isfinite(vals[i - 1])):
                continue
            if is_phi:
                jump = abs(circular_diff_deg(vals[i], vals[i - 1]))
            else:
                jump = abs(vals[i] - vals[i - 1])
            if jump > best_jump:
                best_jump, best_i = jump, i
        if best_i is None:
            print(f'  {sta}: n/a (insufficient data)')
        else:
            b_prev, b_cur = bins[best_i - 1], bins[best_i]
            print(f'  {sta}: largest change between "{b_prev["label"].splitlines()[0]}" and '
                  f'"{b_cur["label"].splitlines()[0]}" (jump={best_jump:.2f})')


def main():
    print('Loading new mfast max_dt=0.2s splitting results (AXCC1, AXEC1, AXEC3, AXAS1: '
          'complete; AXEC2 2015-2021: partial, in-progress re-run)...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
    dfs = {sta: apply_tier(raw[sta], TIER_IDX) for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        print(f'  {sta}: {len(dfs[sta]):,} QC-passing events (tier {TIER_IDX})')

    # ── Pre-eruption, 10 equal-count bins ───────────────────────────────────
    pre_bins = {}
    for sta in STATION_ORDER:
        pre_df = dfs[sta][dfs[sta]['t'] < ERUPTION_START]
        pre_bins[sta] = equal_count_bins(pre_df, N_BINS)
        print(f'  {sta} pre-eruption N={len(pre_df):,}')

    print('\n' + '=' * 70)
    print(f'PRE-ERUPTION: {N_BINS} equal-count bins')
    print_metric_table('Mean delay time dt (s), "value (N)":', pre_bins, 'mean_dt')
    print_metric_table('Circular mean fast direction phi (deg, axial 0-180), "value (N)":',
                        pre_bins, 'mean_phi', fmt='{:.1f}')
    report_biggest_jump(pre_bins, 'mean_dt', 'pre-eruption dt')
    report_biggest_jump(pre_bins, 'mean_phi', 'pre-eruption phi', is_phi=True)

    # ── Syn-eruption, 10 equal-count bins ────────────────────────────────────
    syn_bins = {}
    for sta in STATION_ORDER:
        syn_df = dfs[sta][(dfs[sta]['t'] >= ERUPTION_START) & (dfs[sta]['t'] < ERUPTION_END)]
        syn_bins[sta] = equal_count_bins(syn_df, N_BINS)
        print(f'  {sta} syn-eruption N={len(syn_df):,}')

    print('\n' + '=' * 70)
    print(f'SYN-ERUPTION: {N_BINS} equal-count bins')
    print_metric_table('Mean delay time dt (s), "value (N)":', syn_bins, 'mean_dt')
    print_metric_table('Circular mean fast direction phi (deg, axial 0-180), "value (N)":',
                        syn_bins, 'mean_phi', fmt='{:.1f}')
    report_biggest_jump(syn_bins, 'mean_dt', 'syn-eruption dt')
    report_biggest_jump(syn_bins, 'mean_phi', 'syn-eruption phi', is_phi=True)

    # ── First 19 hours of the eruption, hourly bins ──────────────────────────
    hour_bins = {}
    for sta in STATION_ORDER:
        hour_bins[sta] = hourly_bins(dfs[sta], ERUPTION_START, N_HOURS_ZOOM)

    print('\n' + '=' * 70)
    print(f'FIRST {N_HOURS_ZOOM} HOURS OF THE ERUPTION (hourly bins from {ERUPTION_START})')
    print_metric_table('Mean delay time dt (s), "value (N)":', hour_bins, 'mean_dt')
    print_metric_table('Circular mean fast direction phi (deg, axial 0-180), "value (N)":',
                        hour_bins, 'mean_phi', fmt='{:.1f}')
    report_biggest_jump(hour_bins, 'mean_dt', 'first-19h dt')
    report_biggest_jump(hour_bins, 'mean_phi', 'first-19h phi', is_phi=True)


if __name__ == '__main__':
    main()
