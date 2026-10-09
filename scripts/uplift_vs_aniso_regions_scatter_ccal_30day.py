#!/usr/bin/env python3
"""
uplift_vs_aniso_regions_scatter_ccal_30day.py

Percent-anisotropy version of uplift_vs_dt_regions_scatter_ccal_30day.py, same 2 x 3 layout
(stations top row in station colours at 50% opacity, regional averages bottom row in black,
AXCC1 in both), with a least-squares linear trendline on every series.

Metric, per event:   A = dt / T_S * 100   (percent anisotropy along the ray)

T_S is the S travel time from hypocentre to station, read DIRECTLY from the PyKonal eikonal
travel-time field through the Baillard 3D S-velocity model (pykonal_raytracer.BaillardRayTracer,
the production incidence-angle tracer; field solved from each station at the seafloor, event at
`depth` km below seafloor, 0-4 km, exactly as the production build_raw_* scripts do). This is
the exact first-arrival time, rather than sws_percent_anisotropy.py's r / mean(Vs) proxy.
Travel times are cached to results/pykonal_s_travel_times_grade3_v2.csv (recomputed only if
missing), keyed on (station, event_id, origin time): event_id restarts between the 2015-2021 and
2022-2026 files, so (station, event_id) alone is not unique. The earlier cache
(pykonal_s_travel_times_grade3.csv) was keyed on event_id only and gave every colliding 2022-2026
event the T_S of a 2015 event; it is left in place but no longer read.

Normalising by T_S removes the part of dt that is just "longer path, more delay". To show
whether the dt trends ARE path-length driven, each panel also reports r with uplift for dt, A and
T_S itself: if T_S has no trend while A keeps dt's trend, the dt changes are not a ray-path effect.

Pooling, the AXEC2 unleveled-window drop, the daily-then-30-day roll and the uplift merge are as
in the dt and phi figures. Trendline: OLS of rolled A on uplift; the slope is quoted in % per m.
The rolled series are strongly autocorrelated, so r and slopes are descriptive, not tests.

Output: uplift_vs_aniso_regions_scatter_ccal_30day.pdf / .png
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
from bpr_inflation_periods_ccal import POST_ERUPTION_START
import matplotlib.patheffects as pe

from axec2_uplift_phi_cosine_vs_time import ERUPTION_END
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    ROLL_WINDOW_DAYS, ROLL_MIN_DAYS, UPLIFT_ROLLING_DAYS, GEODETIC_LABEL,
)
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from rose_7period_regions_windowcheck_grade3 import GRADE, load_station_raw, apply_grade
from rose_7period_6stations_newdata_snr_grades import STATION_ORDER
from uplift_vs_phi_regions_scatter_ccal_30day import (
    PANELS, POINT_COLOR, STATION_COLORS, STATION_ALPHA,
)
from animate_arctan_stress_vectors import load_station_xy
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'uplift_vs_aniso_regions_scatter_ccal_30day')
TT_CACHE = os.path.join(HERE, '..', 'results', 'pykonal_s_travel_times_grade3_v2.csv')
# AXEC2's 2015-2021 windowcheck CSV carries no hypocentres; they come from the raw-batch metadata
# the run was built from (same event_id scheme; origin times verified identical for all 91,710).
AXEC2_META = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                          'raw_axec2_all_batches_mfast_filters_metadata.csv')

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))
Z_MAX_KM = 4.0
Y_MAX_PCT = 8.5   # shared axis cap; a few isolated spikes above it are counted and noted


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


def tt_key(df):
    """Join key for the travel-time cache: event_id is only unique together with origin time."""
    return df['event_id'].astype(str) + '|' + df['t'].astype(str)


def travel_times():
    """Table of (station, key, T_S seconds) for every Grade-3 event, cached; key = tt_key()."""
    if os.path.exists(TT_CACHE):
        tt = pd.read_csv(TT_CACHE)
        print(f'Loaded {len(tt):,} cached travel times from {TT_CACHE}')
        return tt
    from pykonal_raytracer import BaillardRayTracer
    tracer = BaillardRayTracer()
    rows = []
    for sta in STATION_ORDER:
        sx, sy = load_station_xy(sta)
        tracer.precompute_station(sta, float(sx), float(sy))
        field = tracer._tt[sta]
        df = apply_grade(load_station_raw(sta), GRADE).drop_duplicates(['event_id', 't'])
        if sta == 'AXEC2':
            meta = pd.read_csv(AXEC2_META, usecols=['event_id', 'latitude', 'longitude', 'depth'])
            meta = meta.set_index('event_id')
            miss = df['depth'].isna()
            for c in ('latitude', 'longitude', 'depth'):
                df.loc[miss, c] = df.loc[miss, 'event_id'].map(meta[c]).values
            print(f'  AXEC2: filled {int(miss.sum()):,} missing hypocentres from raw-batch metadata')
        ex, ey = ll2xy(df['latitude'].values, df['longitude'].values)
        ez = df['depth'].values
        n_bad = 0
        for eid, x, y, z in zip(tt_key(df).values, ex, ey, ez):
            if not (0.0 <= z <= Z_MAX_KM):
                n_bad += 1
                continue
            try:
                t = float(field.value(np.array([x, y, z], dtype=float)))
            except Exception:
                t = np.nan
            if not np.isfinite(t) or t <= 0:
                n_bad += 1
                continue
            rows.append((sta, eid, t))
        print(f'  {sta}: {len(df) - n_bad:,} travel times ({n_bad} out of model / depth range)')
    tt = pd.DataFrame(rows, columns=['station', 'key', 'T_s'])
    os.makedirs(os.path.dirname(TT_CACHE), exist_ok=True)
    tt.to_csv(TT_CACHE, index=False)
    print(f'Cached {len(tt):,} travel times to {TT_CACHE}')
    return tt


def rolled(pool, col, inflation_roll):
    """Daily mean of `col` -> 30-day centred rolling mean + SE, post-eruption, on uplift."""
    d = pool.loc[pool['t'] >= POST_ERUPTION_START, ['t', col]].dropna().copy()   # post only
    d['day'] = d['t'].dt.floor('D')
    daily = d.groupby('day')[col].mean()
    win = f'{ROLL_WINDOW_DAYS}D'
    m = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).mean()
    s = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).std()
    nd = daily.rolling(win, center=True, min_periods=ROLL_MIN_DAYS).count()
    roll = pd.DataFrame({'t': daily.index, 'v': m.values,
                         'se': (s / np.sqrt(nd.clip(lower=1))).values}).dropna()
    roll = roll[roll['t'] >= POST_ERUPTION_START]
    infl = inflation_roll.dropna().reset_index()
    infl.columns = ['t', 'u']
    merged = pd.merge_asof(roll.sort_values('t'), infl.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D')).dropna(subset=['u'])
    return merged['u'].values, merged['v'].values, merged['se'].values


def series(members, tt, inflation_roll):
    with contextlib.redirect_stdout(io.StringIO()):
        pool = load_region_pool(members)
    pool = pool.assign(key=tt_key(pool)).merge(tt, on=['station', 'key'], how='inner')
    pool['A_pct'] = pool['dt'] / pool['T_s'] * 100.0
    xA, yA, seA = rolled(pool, 'A_pct', inflation_roll)
    xd, yd, _ = rolled(pool, 'dt', inflation_roll)
    xT, yT, _ = rolled(pool, 'T_s', inflation_roll)
    r = {k: float(np.corrcoef(a, b)[0, 1]) for k, (a, b) in
         {'dt': (xd, yd), 'A': (xA, yA), 'T': (xT, yT)}.items()}
    return xA, yA, seA, r, float(pool['T_s'].median())


def draw(ax, x, y, se, color, label, alpha, se_alpha, s, line_kw):
    n_off = int(np.sum(y > Y_MAX_PCT))
    if n_off:
        print(f'      {label}: {n_off} rolled point(s) above the {Y_MAX_PCT}% axis cap')
    ax.vlines(x, y - se, y + se, color=color, alpha=se_alpha, lw=1.5, zorder=1)
    ax.scatter(x, y, s=s, color=color, alpha=alpha, linewidths=0, zorder=2, label=label)
    slope, icpt = np.polyfit(x, y, 1)
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, slope * xl + icpt, zorder=3,
            label=f'{label} trend: {slope:+.2f} %/m', **line_kw)
    return slope


def style(ax):
    ax.grid(alpha=0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def main():
    tt = travel_times()
    _dd, _infl, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    halo = [pe.withStroke(linewidth=4, foreground='white')]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.8), sharex=True, sharey=True)
    for col, (title, subtitle, members) in enumerate(PANELS):
        ax_sta, ax_avg = axes[0, col], axes[1, col]

        x, y, se, r, tmed = series(members, tt, inflation_roll)
        lab = f'{title} average' if len(members) > 1 else members[0]
        draw(ax_avg, x, y, se, POINT_COLOR, lab, 0.7, 0.12, 18,
             dict(color='black', lw=2, ls='--', path_effects=halo))
        ax_avg.set_title(f'{title} average' if len(members) > 1 else title, fontsize=14,
                         fontweight='bold', loc='left')
        ax_avg.text(1.0, 1.02, f'{subtitle}\nN = {len(x)}', transform=ax_avg.transAxes,
                    ha='right', va='bottom', fontsize=8.5, color='0.35')
        ax_avg.text(0.02, 0.97, f'r with uplift — δt {r["dt"]:+.2f}, A {r["A"]:+.2f}, '
                    f'T_S {r["T"]:+.2f}', transform=ax_avg.transAxes, ha='left', va='top',
                    fontsize=8, color='0.2',
                    bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
        print(f'{lab:16s} N={len(x)}  median T_S={tmed:.3f}s  r(dt,u)={r["dt"]:+.2f}  '
              f'r(A,u)={r["A"]:+.2f}  r(T,u)={r["T"]:+.2f}')

        rtxt = []
        for sta in members:
            xs, ys, ses, rs, tm = series([sta], tt, inflation_roll)
            c = STATION_COLORS[sta]
            draw(ax_sta, xs, ys, ses, c, sta, STATION_ALPHA, 0.08, 12,
                 dict(color=c, lw=2.2, path_effects=halo))
            rtxt.append(f'{sta}: δt {rs["dt"]:+.2f}, A {rs["A"]:+.2f}, T_S {rs["T"]:+.2f}')
            print(f'    {sta:12s} N={len(xs)}  median T_S={tm:.3f}s  r(dt,u)={rs["dt"]:+.2f}  '
                  f'r(A,u)={rs["A"]:+.2f}  r(T,u)={rs["T"]:+.2f}')
        ax_sta.set_title(title, fontsize=15, fontweight='bold', loc='left')
        ax_sta.text(0.02, 0.97, 'r with uplift\n' + '\n'.join(rtxt), transform=ax_sta.transAxes,
                    ha='left', va='top', fontsize=7.5, color='0.2',
                    bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))

        for ax in (ax_sta, ax_avg):
            style(ax)
            ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9, markerscale=1.4)
            ax.set_ylim(0, Y_MAX_PCT)
        ax_avg.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')

    ylab = f'Percent anisotropy δt / T_S × 100 (%, {ROLL_WINDOW_DAYS}-day rolling)'
    axes[0, 0].set_ylabel('Individual stations\n' + ylab)
    axes[1, 0].set_ylabel('Regional average\n' + ylab)
    fig.suptitle(f'Percent Anisotropy vs. De-Tided Central Caldera Uplift, Post-Eruption '
                 f'({GEODETIC_LABEL}; T_S from PyKonal, Baillard 3-D Vs)', fontsize=15,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
