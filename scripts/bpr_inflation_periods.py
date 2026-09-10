#!/usr/bin/env python3
"""
bpr_inflation_periods.py

Shared module: turns the de-tided BOTPT seafloor-depth record (process_bpr_detided_depth.py's
output) into (1) a daily-mean "m of uplift" (inflation) series referenced to zero at the
lowest point observed right after the 2015 eruption, and (2) a 7-period eruption-relative
scheme for rose plots, per explicit user request, where the 5 post-eruption periods are of
EQUAL INFLATION AMOUNT (not equal event count) -- defined on a ROLLING MEAN of the inflation
curve (per explicit user follow-up: "plot a rolling mean line and use that to define the time
periods"), since the raw daily inflation isn't perfectly monotonic (short-period noise/dips
would otherwise make a raw first-crossing boundary jump around).

Reference (zero-inflation) point: the minimum of a 5-day-centered rolling mean of
detided_depth_m (i.e. maximum subsidence -- most negative depth) within a window bracketing the
eruption (ERUPTION_START - 5 days to ERUPTION_END + 30 days). The 5-day smoothing is needed
because the RAW daily series has one-day instrument-glitch spikes (e.g. 2015-06-15 and
2015-07-15 both show single-day dips of ~0.05-0.25m below the surrounding plateau, confirmed by
inspection -- not part of the actual post-eruption trough) that would otherwise get picked as
the reference minimum instead of the genuine plateau. With smoothing, the reference lands at
2015-05-18, matching the user's expectation of "May 19, 2015 or in April 2015".

inflation_m(t) = detided_depth_m(t) - reference_depth_m
  (detided_depth_m is more NEGATIVE when the seafloor is deeper/more subsided, and less
  negative when uplifted, so this convention makes pre-eruption inflation POSITIVE --
  elevated relative to the post-eruption low, which is 0 by definition -- dropping to ~0
  during the eruption, then rising again during re-inflation.)

5 equal-inflation post-eruption bins: computed on the ROLLING_DAYS-day centered rolling mean of
daily inflation (not the raw daily series), by dividing the range [rolling mean at
ERUPTION_END, rolling mean at the last available date] into 5 equal-height bands and taking the
first date the rolling-mean curve crosses each internal band edge.

Run this module directly to print the resulting periods table; import
build_inflation_based_periods() from other scripts to reuse it.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DETIDED_CSV = os.path.join(HERE, '..', 'data',
                            'bpr_detided_seafloor_depth_2015-01-22_to_2026-05-15.csv')

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

# Window for locating the post-eruption zero-inflation reference point (tight around the
# eruption -- see docstring for why NOT searching the whole year).
REF_WINDOW_START = ERUPTION_START - pd.Timedelta(days=5)
REF_WINDOW_END = ERUPTION_END + pd.Timedelta(days=30)

ROLLING_DAYS = 30
REF_SMOOTH_DAYS = 5   # smoothing used only to LOCATE the reference date, avoiding 1-day spikes


def load_daily_series():
    """Daily-mean detided depth, referenced daily-mean inflation, and its rolling mean."""
    df = pd.read_csv(DETIDED_CSV, usecols=['time', 'detided_depth_m'])
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    daily_depth = df['detided_depth_m'].resample('1D').mean()

    smoothed = daily_depth.rolling(f'{REF_SMOOTH_DAYS}D', center=True, min_periods=3).mean()
    ref_window_smoothed = smoothed.loc[REF_WINDOW_START:REF_WINDOW_END]
    ref_time = ref_window_smoothed.idxmin()
    ref_depth = float(daily_depth.loc[ref_time])   # report the actual (unsmoothed) value at that date

    inflation = daily_depth - ref_depth
    inflation_roll = inflation.rolling(f'{ROLLING_DAYS}D', center=True, min_periods=ROLLING_DAYS // 2).mean()

    return daily_depth, inflation, inflation_roll, ref_time, ref_depth


def build_inflation_based_periods():
    """7 periods: Before/During Eruption (unchanged) + 5 equal-inflation-amount post-eruption
    bins, boundaries found via first-crossing on the rolling-mean inflation curve."""
    _daily_depth, _inflation, inflation_roll, _ref_time, _ref_depth = load_daily_series()

    post = inflation_roll.loc[ERUPTION_END:].dropna()
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
    print(f'\n{ROLLING_DAYS}-day rolling-mean inflation, post-eruption:')
    post = inflation_roll.loc[ERUPTION_END:].dropna()
    print(f'  at Post-eruption start ({post.index[0].date()}): {post.iloc[0]:.4f} m')
    print(f'  at last available date ({post.index[-1].date()}): {post.iloc[-1]:.4f} m')

    periods = build_inflation_based_periods()
    print('\n7-period scheme (equal-inflation-amount post-eruption bins, via rolling mean):')
    for label, t0, t1 in periods:
        t0s = t0.date() if t0 is not None else '(start of record)'
        t1s = t1.date() if t1 is not None else '(present)'
        print(f'  {label!r:45s}  {t0s} -> {t1s}')
