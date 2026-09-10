#!/usr/bin/env python3
"""
bpr_inflation_periods_ashes.py

ASHES vent field (Western Caldera BOTPT, RS03ASHS-MJ03B-09-BOTPTA304) counterpart of
bpr_inflation_periods.py / bpr_inflation_periods_ccal.py -- turns process_bpr_detided_depth_
ashes.py's output into a daily "m of uplift" series and a 7-period eruption-relative scheme.

KEY DIFFERENCE from the Eastern/Central Caldera versions: this instrument's record only starts
2017-08-15 (see process_bpr_detided_depth_ashes.py's docstring) -- there is NO coverage of the
2015 eruption itself or the first ~2.3 years of re-inflation. This breaks the usual reference-
point method (search a window bracketing the eruption for the post-eruption low), since that
window has no data at all here. Adaptations:

  - Zero-inflation reference: the minimum of a 5-day-centered rolling mean of detided_depth_m
    over the ENTIRE available record (not a window bracketing the eruption -- there's nothing to
    bracket). This lands at 2018-08-30, i.e. the seafloor kept subsiding for about a year after
    this instrument was deployed before turning around to inflate -- a real, physically
    plausible continued post-eruptive relaxation, not an instrument artifact (checked against
    the surrounding daily series, no single-day spike).
  - The 5 equal-inflation-amount "post-eruption" bins are computed over the inflation curve's
    OWN available range (from its first valid rolling-mean date, not from ERUPTION_END, since
    there is no data between ERUPTION_END 2015-05-19 and deployment 2017-08-15) through the
    last available date. "Before Eruption" and "During Eruption" period boundaries are kept
    for consistency with the other stations' rose plots (splitting-catalog events DO span
    those periods, even though this BPR record doesn't) -- they will simply have no uplift
    line in the companion time-series plot for that span.

inflation_m(t) = detided_depth_m(t) - reference_depth_m (same sign convention as the other two
modules: positive = more inflated/elevated than the reference low).

Run this module directly to print the resulting periods table; import
build_inflation_based_periods() from other scripts to reuse it.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DETIDED_CSV = os.path.join(HERE, '..', 'data',
                            'bpr_detided_seafloor_depth_ashes_2015-01-22_to_2026-05-15.csv')

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

ROLLING_DAYS = 30
REF_SMOOTH_DAYS = 5   # smoothing used only to LOCATE the reference date, avoiding 1-day spikes


def load_daily_series():
    """Daily-mean detided depth, referenced daily-mean inflation, and its rolling mean."""
    df = pd.read_csv(DETIDED_CSV, usecols=['time', 'detided_depth_m'])
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    daily_depth = df['detided_depth_m'].resample('1D').mean()

    # No eruption-adjacent window to search (no data there) -- use the whole record.
    smoothed = daily_depth.rolling(f'{REF_SMOOTH_DAYS}D', center=True, min_periods=3).mean()
    ref_time = smoothed.idxmin()
    ref_depth = float(daily_depth.loc[ref_time])

    inflation = daily_depth - ref_depth
    inflation_roll = inflation.rolling(f'{ROLLING_DAYS}D', center=True, min_periods=ROLLING_DAYS // 2).mean()

    return daily_depth, inflation, inflation_roll, ref_time, ref_depth


def build_inflation_based_periods():
    """7 periods: Before/During Eruption (kept for rose-plot consistency, no BPR data there) +
    5 equal-inflation-amount bins, boundary VALUES from this record's own available inflation
    range, but the FIRST bin starts at ERUPTION_END (2015-05-19), not at this record's first
    available date (2017-08-15) -- otherwise every splitting-catalog event between the eruption
    and ASHES deployment (2015-05-19 to 2017-08-15) would fall in the gap between "During
    Eruption" and the first inflation bin and be silently dropped from every rose-plot panel
    (confirmed: 168 AXAS1 events land in exactly that gap). The label still describes the true
    time range (starting from ERUPTION_END), it's only the inflation-amount partitioning that's
    computed over the shorter 2017-08-15-onward span where uplift is actually measured."""
    _daily_depth, _inflation, inflation_roll, _ref_time, _ref_depth = load_daily_series()

    post = inflation_roll.dropna()
    v_start = float(post.iloc[0])
    v_end = float(post.iloc[-1])
    last_date = post.index[-1]

    edges = np.linspace(v_start, v_end, 6)   # 5 bins -> 6 edges (band values, not dates)
    boundaries = [ERUPTION_END]
    increasing = v_end >= v_start
    for edge_val in edges[1:5]:
        if increasing:
            cross = post[post >= edge_val]
        else:
            cross = post[post <= edge_val]
        t_cross = cross.index[0] if len(cross) else last_date
        boundaries.append(t_cross)
    boundaries.append(None)

    def _fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else 'present'

    periods = [
        ('Before Eruption', None, ERUPTION_START),
        ('During Eruption', ERUPTION_START, ERUPTION_END),
    ]
    for i in range(5):
        t0, t1 = boundaries[i], boundaries[i + 1]
        periods.append((f'{_fmt(t0)} - {_fmt(t1)}', t0, t1))
    return periods


if __name__ == '__main__':
    daily_depth, inflation, inflation_roll, ref_time, ref_depth = load_daily_series()
    print(f'Zero-inflation reference: {ref_time.date()}  (detided depth = {ref_depth:.4f} m)')
    print(f'  Data span: {daily_depth.index.min().date()} to {daily_depth.index.max().date()}')
    print(f'\n{ROLLING_DAYS}-day rolling-mean inflation:')
    post = inflation_roll.dropna()
    print(f'  at first available date ({post.index[0].date()}): {post.iloc[0]:.4f} m')
    print(f'  at last available date ({post.index[-1].date()}): {post.iloc[-1]:.4f} m')

    periods = build_inflation_based_periods()
    print('\n7-period scheme (equal-inflation-amount bins over the available record, via rolling mean):')
    for label, t0, t1 in periods:
        t0s = t0.date() if t0 is not None else '(start of record)'
        t1s = t1.date() if t1 is not None else '(present)'
        print(f'  {label!r:45s}  {t0s} -> {t1s}')
