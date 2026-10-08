#!/usr/bin/env python3
"""
uplift_vs_phi_aniso_pre_syn_ccal.py

Pre-eruption and syn-eruption versions of the fast-direction and percent-anisotropy vs. uplift
figures (uplift_vs_phi_regions_scatter_atan2fit_ccal_30day.py, uplift_vs_aniso_regions_scatter_
ccal_30day.py). Two figures, each 4 rows x 3 columns (West / Central / East):

    row 1  pre-eruption, individual stations (station colours, 50% opacity)
    row 2  pre-eruption, regional averages (black)
    row 3  syn-eruption, individual stations
    row 4  syn-eruption, regional averages

Periods:
    pre  = start of record (2015-01-22) -> the pre-eruption uplift PEAK
    syn  = the peak -> the post-eruption uplift MINIMUM (load_daily_series' ref_time, 2015-05-02,
           inclusive) -- the zero reference of every post-eruption figure.
The peak is taken at eruption onset (ERUPTION_START, 2015-04-24 06:00), by user decision: uplift
was still rising into it (Apr 23 daily mean 2.458 m) and the earlier plateau high (Apr 6,
2.466 m) is within day-to-day noise of it. NOTE: the post-eruption figures still start at
ERUPTION_END (2015-05-19), so 2015-05-03 .. 05-18 (earliest re-inflation) is in neither set.

Uplift reference: PRE-eruption uplift is relative to the post-eruption minimum, as everywhere
else (positive, ~2.1-2.45 m). SYN-eruption uplift is relative to the PEAK level -- the mean
de-tided level over the PEAK_REF_HOURS before onset (24 h = two semidiurnal cycles, because the
hourly de-tided record carries a ~+-1 m tidal residual around onset) -- so it runs from 0 down to
about -2.4 m: the deflation.

Smoothing: pre-eruption uses ROLL_DAYS-day centred rolling windows; syn-eruption (~9 days, with
most of the deflation inside the first day) uses DAILY values, no rolling, with +-1 SE from the
spread of that day's events. Everything is computed STRICTLY within each period, seismic series
and uplift alike, so nothing mixes across the boundaries. Uplift is de-tided
CCAL BOTPT, referenced to the post-eruption minimum as everywhere else, averaged per day from
the minute-level record restricted to the period (the onset day is split at 06:00).

Metrics, from the same pooled Grade-3 events and AXEC2 unleveled-window drop:
    phi  per-day circular mean -> rolling circular mean (+-1 SE = circ. std / sqrt(days))
    A    dt / T_S * 100, T_S from the PyKonal travel-time cache (see the anisotropy script)
Each series gets a least-squares LINEAR trendline (slope in legend). The atan2 fit is not used:
9 days cannot constrain it, and its beta is not identifiable anyway.

Missing data are stated on the panels; AXCC1 has no data 2015-03-01 to 2015-04-28 (known gap).

Outputs:
    uplift_vs_phi_pre_syn_ccal.pdf / .png
    uplift_vs_aniso_pre_syn_ccal.pdf / .png
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
import matplotlib.patheffects as pe

from axec2_uplift_phi_cosine_vs_time import ERUPTION_START, ERUPTION_END
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from animate_arctan_stress_vectors import compute_optimal_wrap, compute_full_range_ticks
from uplift_vs_phi_regions_scatter_ccal_30day import (
    PANELS, POINT_COLOR, STATION_COLORS, STATION_ALPHA,
)
from uplift_vs_aniso_regions_scatter_ccal_30day import travel_times
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
ROLL_DAYS = 5
ROLL_MIN_DAYS = 2
PEAK_REF_HOURS = 24
RECORD_START = pd.Timestamp('2015-01-22', tz='UTC')
PEAK_TIME = ERUPTION_START                       # pre-eruption uplift peak (see docstring)
_MIN_TIME = ccal_infl.load_daily_series()[3]     # post-eruption minimum = post zero reference
SYN_END = _MIN_TIME + pd.Timedelta(days=1)       # minimum day inclusive
PERIODS = [
    ('Pre-eruption', RECORD_START, PEAK_TIME),
    ('Syn-eruption', PEAK_TIME, SYN_END),
]
ROLL_BY_PERIOD = {'Pre-eruption': ROLL_DAYS, 'Syn-eruption': 1}   # 1 = daily, no rolling
HALO = [pe.withStroke(linewidth=4, foreground='white')]


def uplift_daily_by_period():
    """{period label: daily de-tided uplift (m), averaged from minute data inside the period,
    then ROLL_DAYS-day centred rolling mean within the period}."""
    _dd, _infl, _roll, _rt, ref_depth = ccal_infl.load_daily_series()
    raw = pd.read_csv(ccal_infl.DETIDED_CSV, usecols=['time', 'detided_depth_m'])
    raw['time'] = pd.to_datetime(raw['time'], utc=True)
    raw = raw.set_index('time')['detided_depth_m'] - ref_depth
    pre_peak = raw[(raw.index >= PEAK_TIME - pd.Timedelta(hours=PEAK_REF_HOURS)) &
                   (raw.index < PEAK_TIME)]
    u_peak = float(pre_peak.mean())
    print(f'Uplift at the pre-eruption peak (mean of {PEAK_REF_HOURS} h before onset): '
          f'{u_peak:.3f} m above the post-eruption minimum')
    out = {}
    for label, t0, t1 in PERIODS:
        s = raw[(raw.index >= t0) & (raw.index < t1)]
        if label == 'Syn-eruption':
            s = s - u_peak           # relative to the peak: 0 -> ~-2.4 m (deflation)
        daily = s.groupby(s.index.floor('D')).mean()
        n = ROLL_BY_PERIOD[label]
        out[label] = daily if n == 1 else \
            daily.rolling(f'{n}D', center=True, min_periods=ROLL_MIN_DAYS).mean()
    return out


def rolled(pool, metric, t0, t1, u_daily, roll_days=ROLL_DAYS):
    """Daily -> within-period rolling series of `metric` ('phi' or 'A'), joined to uplift by day.
    Returns DataFrame(t, u, v, se) with v in raw degrees (phi) or % (A)."""
    p = pool[(pool['t'] >= t0) & (pool['t'] < t1)].copy()
    if len(p) == 0:
        return pd.DataFrame(columns=['t', 'u', 'v', 'se'])
    p['day'] = p['t'].dt.floor('D')
    if roll_days == 1:                       # daily values; SE from that day's events
        if metric == 'phi':
            a = 2.0 * np.radians(p['phi_az'] % 180.0)
            g = pd.DataFrame({'s2': np.sin(a), 'c2': np.cos(a), 'day': p['day']}).groupby('day')
            ms, mc, n = g['s2'].mean(), g['c2'].mean(), g['s2'].count()
            v = (np.degrees(np.arctan2(ms, mc)) / 2.0) % 180.0
            R = np.clip(np.hypot(ms, mc), 1e-12, 1.0)
            se = np.degrees(np.sqrt(-2.0 * np.log(R))) / 2.0 / np.sqrt(n.clip(lower=1))
        else:
            g = p.groupby('day')['A_pct']
            v, se = g.mean(), g.std() / np.sqrt(g.count().clip(lower=1))
        df = pd.DataFrame({'v': v, 'se': se.fillna(0.0)}).dropna(subset=['v'])
        df['u'] = u_daily.reindex(df.index).values
        return df.dropna(subset=['u']).reset_index().rename(columns={'day': 't'})
    win = f'{roll_days}D'
    if metric == 'phi':
        a = 2.0 * np.radians(p['phi_az'] % 180.0)
        p['s2'], p['c2'] = np.sin(a), np.cos(a)
        daily = p.groupby('day').agg(s2=('s2', 'mean'), c2=('c2', 'mean'))
        rs = daily['s2'].rolling(win, center=True, min_periods=ROLL_MIN_DAYS).mean()
        rc = daily['c2'].rolling(win, center=True, min_periods=ROLL_MIN_DAYS).mean()
        nd = daily['s2'].rolling(win, center=True, min_periods=ROLL_MIN_DAYS).count()
        v = (np.degrees(np.arctan2(rs, rc)) / 2.0) % 180.0
        R = np.clip(np.hypot(rs, rc), 1e-12, 1.0)
        se = np.degrees(np.sqrt(-2.0 * np.log(R))) / 2.0 / np.sqrt(nd.clip(lower=1))
    else:
        daily = p.groupby('day')['A_pct'].mean()
        v = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).mean()
        sd = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).std()
        nd = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).count()
        se = sd / np.sqrt(nd.clip(lower=1))
    df = pd.DataFrame({'v': v, 'se': se}).dropna()
    df['u'] = u_daily.reindex(df.index).values
    df = df.dropna(subset=['u']).reset_index().rename(columns={'day': 't'})
    return df


def coverage_note(pool, t0, t1):
    p = pool[(pool['t'] >= t0) & (pool['t'] < t1)]
    if len(p) == 0:
        return 'no data'
    days = p['t'].dt.floor('D')
    return f'{days.min():%b %d}–{days.max():%b %d}, {days.nunique()} d'


def load_pools(tt):
    pools = {}
    ttm = tt.assign(event_id=tt['event_id'].astype(str))
    for _title, _sub, members in PANELS:
        for key in [tuple(members)] + [(m,) for m in members]:
            if key in pools:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                pool = load_region_pool(list(key))
            pool = pool.assign(event_id=pool['event_id'].astype(str))
            pool = pool.merge(ttm, on=['station', 'event_id'], how='left')
            pool['A_pct'] = pool['dt'] / pool['T_s'] * 100.0
            pools[key] = pool
    return pools


def trend(ax, x, y, label, line_kw, unit):
    if len(x) < 3 or np.ptp(x) == 0:
        return
    slope, icpt = np.polyfit(x, y, 1)
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, slope * xl + icpt, zorder=3, label=f'{label} trend: {slope:+.2f} {unit}/m',
            **line_kw)


def make_figure(metric, pools, u_by_period, out_base):
    is_phi = metric == 'phi'
    unit = '°' if is_phi else '%'
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 19), sharey='col' if is_phi else True)

    # phi: one wrap per column from the regional average across BOTH periods (rows line up).
    wraps = {}
    if is_phi:
        for col, (title, _s, members) in enumerate(PANELS):
            vals = np.concatenate([rolled(pools[tuple(members)], 'phi', t0, t1,
                                          u_by_period[lbl], ROLL_BY_PERIOD[lbl])['v'].values
                                   for lbl, t0, t1 in PERIODS])
            wraps[col] = compute_optimal_wrap(vals) if len(vals) > 1 else 0.0

    def to_y(v, col):
        return (np.asarray(v) - wraps[col]) % 180.0 if is_phi else np.asarray(v)

    for p_idx, (plabel, t0, t1) in enumerate(PERIODS):
        u_daily = u_by_period[plabel]
        for col, (title, subtitle, members) in enumerate(PANELS):
            ax_sta, ax_avg = axes[2 * p_idx, col], axes[2 * p_idx + 1, col]

            # averages (black)
            pool = pools[tuple(members)]
            d = rolled(pool, metric, t0, t1, u_daily, ROLL_BY_PERIOD[plabel])
            avg_lab = f'{title} average' if len(members) > 1 else members[0]
            if len(d):
                y = to_y(d['v'], col)
                ax_avg.vlines(d['u'], y - d['se'], y + d['se'], color=POINT_COLOR, alpha=0.15,
                              lw=1.5, zorder=1)
                ax_avg.scatter(d['u'], y, s=22, color=POINT_COLOR, alpha=0.75, linewidths=0,
                               zorder=2, label=avg_lab)
                trend(ax_avg, d['u'].values, y, avg_lab,
                      dict(color='black', lw=2, ls='--', path_effects=HALO), unit)
            ax_avg.set_title(f'{plabel} — {title} average' if len(members) > 1
                             else f'{plabel} — {title}', fontsize=12, fontweight='bold',
                             loc='left')
            ax_avg.text(1.0, 1.02, f'{subtitle.splitlines()[0]}\n'
                        f'{coverage_note(pool, t0, t1)}; {len(d)} pts',
                        transform=ax_avg.transAxes, ha='right', va='bottom', fontsize=8,
                        color='0.35')

            # stations
            notes = []
            for sta in members:
                ps = pools[(sta,)]
                ds = rolled(ps, metric, t0, t1, u_daily, ROLL_BY_PERIOD[plabel])
                c = STATION_COLORS[sta]
                if len(ds):
                    y = to_y(ds['v'], col)
                    ax_sta.vlines(ds['u'], y - ds['se'], y + ds['se'], color=c, alpha=0.12,
                                  lw=1.5, zorder=1)
                    ax_sta.scatter(ds['u'], y, s=16, color=c, alpha=STATION_ALPHA, linewidths=0,
                                   zorder=2, label=sta)
                    trend(ax_sta, ds['u'].values, y, sta,
                          dict(color=c, lw=2.2, path_effects=HALO), unit)
                notes.append(f'{sta}: {coverage_note(ps, t0, t1)}')
            ax_sta.set_title(f'{plabel} — {title}', fontsize=13, fontweight='bold', loc='left')
            ax_sta.text(1.0, 1.02, '\n'.join(notes), transform=ax_sta.transAxes, ha='right',
                        va='bottom', fontsize=7.5, color='0.35')
            if title == 'Central':
                gap = ('no data 2015-03-01 → 04-28 (known AXCC1 gap)' if p_idx == 0
                       else 'no data before 2015-04-28 (known AXCC1 gap)')
                for ax in (ax_sta, ax_avg):
                    ax.text(0.5, 0.04, gap, transform=ax.transAxes, ha='center', va='bottom',
                            fontsize=8.5, color='#B00020', fontweight='bold',
                            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))

            for ax in (ax_sta, ax_avg):
                ax.grid(alpha=0.3)
                for side in ('top', 'right'):
                    ax.spines[side].set_visible(False)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(loc='upper right', fontsize=7.2, framealpha=0.9, markerscale=1.3)
                ax.set_xlabel(f'De-tided uplift $u_z$ (m, {ROLL_DAYS}-day rolling mean)'
                              if p_idx == 0 else
                              'De-tided uplift relative to the pre-eruption peak (m, daily; '
                              'negative = deflation)', fontsize=9)
                if is_phi:
                    lo, hi, tp, tl = compute_full_range_ticks(wraps[col])
                    ax.set_ylim(lo, hi)
                    ax.set_yticks(tp)
                    ax.set_yticklabels([str(v) for v in tl])

    if not is_phi:
        for ax in axes.ravel():
            ax.set_ylim(0, 8.5)
    ylab = 'Mean fast direction φ (deg)' if is_phi else 'Percent anisotropy δt / T_S × 100 (%)'
    for r, rl in enumerate([f'Pre — stations ({ROLL_DAYS}-day rolling)',
                            f'Pre — average ({ROLL_DAYS}-day rolling)', 'Syn — stations (daily)',
                            'Syn — average (daily)']):
        axes[r, 0].set_ylabel(f'{rl}\n{ylab}', fontsize=9)
    what = 'Fast Direction' if is_phi else 'Percent Anisotropy'
    fig.suptitle(f'{what} vs. De-Tided Central Caldera Uplift — Pre-eruption (to the peak at onset, '
                 f'2015-04-24 06:00) and Syn-eruption (peak to minimum, '
                 f'{_MIN_TIME:%Y-%m-%d})', fontsize=14.5, fontweight='bold', y=1.005)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_base}.pdf / .png')


def main():
    tt = travel_times()
    u_by_period = uplift_daily_by_period()
    for lbl, s in u_by_period.items():
        s = s.dropna()
        print(f'{lbl}: uplift {s.min():.2f} → {s.max():.2f} m over {len(s)} days')
    pools = load_pools(tt)
    make_figure('phi', pools, u_by_period, os.path.join(HERE, 'uplift_vs_phi_pre_syn_ccal'))
    make_figure('A', pools, u_by_period, os.path.join(HERE, 'uplift_vs_aniso_pre_syn_ccal'))


if __name__ == '__main__':
    main()
