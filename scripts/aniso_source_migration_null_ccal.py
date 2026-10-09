#!/usr/bin/env python3
"""
aniso_source_migration_null_ccal.py

Test 1 of the source-migration synthetic test: can the post-eruption trend of percent anisotropy
(A = dt / T_S * 100) with de-tided CCAL uplift be produced by the earthquakes moving, with the
anisotropy field held fixed?

Post-eruption hypocentres deepen with uplift at every station (~0.1-0.2 km West, ~0.35-0.45 km
East/Central), yet A rises with uplift in the West and is flat-to-falling in the East. H0 here is
"A depends only on where the event is (relative to its station), not on when it happened".

Two estimates under H0, per station and per region (PANELS, as in
uplift_vs_aniso_regions_scatter_ccal_30day.py):

1. Location-matched prediction. Each event is put in a hypocentre cell (CELL_KM cube, per
   station). Its predicted A is the mean A of the OTHER calendar years' events in that cell
   (leave-one-year-out, so the prediction never sees the event's own time / uplift level).
   The predicted series goes through the same daily -> 30-day centred roll -> OLS-on-uplift as
   the observed one. slope_pred / slope_obs = the fraction of the trend that migration explains;
   the observed - predicted residual is what it leaves over. Events whose cell has no other-year
   events are dropped from both series (coverage reported).

2. Within-cell permutation. A is shuffled among events in the same (station, cell), keeping every
   event's time and location, N_PERM times; the slope distribution is what migration alone
   produces. p = two-sided fraction of |slope_perm - median| >= |slope_obs - median|.
   Shuffling event-level A also removes any non-stress slow wander in A, so p is optimistic if
   such wander exists; the leave-one-year-out prediction does not have that weakness.

Run at two cell sizes (CELL_SIZES) to show the answer is not a binning choice.

The fit functions (run_null) take any event table with columns
station, t, x, y, z, A_pct, so aniso_synthetic_static_models_ccal.py reuses them on synthetic A.

Outputs (new files):
  aniso_source_migration_null_ccal.pdf / .png
  ../results/aniso_source_migration_null_ccal_summary.csv
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

from bpr_inflation_periods_ccal import POST_ERUPTION_START
import bpr_inflation_periods_ccal as ccal_infl
from atan2_uplift_vs_phi_sixstations_ccal_30day import (
    ROLL_WINDOW_DAYS, ROLL_MIN_DAYS, GEODETIC_LABEL,
)
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from uplift_vs_phi_regions_scatter_ccal_30day import PANELS, STATION_COLORS
from uplift_vs_aniso_regions_scatter_ccal_30day import travel_times, tt_key, ll2xy, rolled
from animate_arctan_stress_vectors import load_station_xy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'aniso_source_migration_null_ccal')
SUMMARY_CSV = os.path.join(HERE, '..', 'results', 'aniso_source_migration_null_ccal_summary.csv')

CELL_SIZES = (0.25, 0.5)   # km; 0.25 matches the voxel back-projection grid
PLOT_CELL = 0.25
N_PERM = 1000
SEED = 20261009


def load_events(tt=None):
    """Post-eruption Grade-3 events of all six stations with station-relative geometry and A."""
    if tt is None:
        tt = travel_times()
    members = [m for _, _, mem in PANELS for m in mem]
    with contextlib.redirect_stdout(io.StringIO()):
        pool = load_region_pool(members)
    # keyed on event_id + origin time: event_id repeats across the 2015-21 / 2022-26 files
    pool = pool.assign(key=tt_key(pool)).merge(tt, on=['station', 'key'], how='inner')
    pool['x'], pool['y'] = ll2xy(pool['latitude'].values, pool['longitude'].values)
    pool['z'] = pool['depth'].values
    for sta in members:
        sx, sy = load_station_xy(sta)
        m = pool['station'] == sta
        pool.loc[m, 'sx'], pool.loc[m, 'sy'] = float(sx), float(sy)
    pool = pool[pool['t'] >= POST_ERUPTION_START].copy()
    pool['A_pct'] = pool['dt'] / pool['T_s'] * 100.0
    return pool.dropna(subset=['x', 'y', 'z', 'A_pct']).reset_index(drop=True)


# ── fast equivalent of uplift_vs_aniso_regions_scatter_ccal_30day.rolled() ────────────────────
class Roller:
    """Precomputes the day grid and uplift match for one event set so the daily -> 30-day roll
    -> OLS slope can be repeated cheaply for many A vectors (permutations, synthetics)."""

    def __init__(self, t, inflation_roll):
        day = t.dt.floor('D')
        self.days, self.code = np.unique(day.values, return_inverse=True)
        self.n_per_day = np.bincount(self.code)
        self.idx = pd.DatetimeIndex(self.days)
        infl = inflation_roll.dropna().reset_index()
        infl.columns = ['t', 'u']
        self.infl = infl.sort_values('t')
        self.infl['t'] = pd.to_datetime(self.infl['t'], utc=True)

    def series(self, v):
        daily = pd.Series(np.bincount(self.code, weights=v) / self.n_per_day, index=self.idx)
        r = daily.rolling(f'{ROLL_WINDOW_DAYS}D', center=True, min_periods=ROLL_MIN_DAYS)
        roll = pd.DataFrame({'t': self.idx, 'v': r.mean().values}).dropna()
        roll['t'] = pd.to_datetime(roll['t'], utc=True)
        m = pd.merge_asof(roll.sort_values('t'), self.infl, on='t', direction='nearest',
                          tolerance=pd.Timedelta('20D')).dropna(subset=['u'])
        return m['u'].values, m['v'].values

    def slope(self, v):
        u, y = self.series(v)
        return float(np.polyfit(u, y, 1)[0])


def cell_keys(df, cell_km):
    ix = np.floor(df['x'].values / cell_km).astype(int)
    iy = np.floor(df['y'].values / cell_km).astype(int)
    iz = np.floor(df['z'].values / cell_km).astype(int)
    return pd.Series([f'{s}|{a}|{b}|{c}' for s, a, b, c in
                      zip(df['station'].values, ix, iy, iz)], index=df.index)


def loyo_prediction(df, cell, col='A_pct'):
    """Leave-one-calendar-year-out cell mean of `col`; NaN where the cell has no other-year events."""
    yr = df['t'].dt.year.values
    g = pd.DataFrame({'cell': cell.values, 'yr': yr, 'v': df[col].values})
    tot = g.groupby('cell')['v'].agg(['sum', 'count'])
    cy = g.groupby(['cell', 'yr'])['v'].agg(['sum', 'count'])
    s_all = tot.loc[g['cell'], 'sum'].values
    n_all = tot.loc[g['cell'], 'count'].values
    key = pd.MultiIndex.from_arrays([g['cell'], g['yr']])
    s_yr = cy.loc[key, 'sum'].values
    n_yr = cy.loc[key, 'count'].values
    n = n_all - n_yr
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, (s_all - s_yr) / n, np.nan)


def within_cell_perm(values, cell, rng):
    out = values.copy()
    for idx in pd.Series(np.arange(len(cell))).groupby(cell.values).indices.values():
        if len(idx) > 1:
            out[idx] = values[rng.permutation(idx)]
    return out


def run_null(df, inflation_roll, cell_km, groups, col='A_pct', n_perm=N_PERM, seed=SEED,
             keep_series=False):
    """Test 1 on one event table. groups: [(label, [stations])]. Returns summary rows
    (and per-group series for plotting when keep_series)."""
    rng = np.random.default_rng(seed)
    cell = cell_keys(df, cell_km)
    pred = loyo_prediction(df, cell, col)
    rows, series = [], {}
    for label, members in groups:
        m = df['station'].isin(members).values
        ok = m & np.isfinite(pred)
        sub = df.loc[ok].reset_index(drop=True)
        roller = Roller(sub['t'], inflation_roll)
        obs, prd = sub[col].values, pred[ok]
        s_obs, s_pred = roller.slope(obs), roller.slope(prd)
        s_res = roller.slope(obs - prd)
        # permutation on the full station set (no LOYO coverage requirement)
        sub_all = df.loc[m].reset_index(drop=True)
        r_all = Roller(sub_all['t'], inflation_roll)
        c_all = cell.loc[m].reset_index(drop=True)
        v_all = sub_all[col].values
        s_obs_all = r_all.slope(v_all)
        perm = np.array([r_all.slope(within_cell_perm(v_all, c_all, rng)) for _ in range(n_perm)])
        med = np.median(perm)
        p = (np.sum(np.abs(perm - med) >= abs(s_obs_all - med)) + 1) / (n_perm + 1)
        rows.append(dict(group=label, cell_km=cell_km, n_events=int(m.sum()),
                         loyo_coverage=float(ok.sum() / max(m.sum(), 1)),
                         slope_obs=s_obs, slope_pred_loyo=s_pred, slope_resid_loyo=s_res,
                         frac_explained_loyo=s_pred / s_obs if s_obs != 0 else np.nan,
                         slope_obs_all=s_obs_all, perm_median=med,
                         perm_lo95=np.percentile(perm, 2.5), perm_hi95=np.percentile(perm, 97.5),
                         p_perm=p))
        if keep_series:
            series[label] = dict(obs=roller.series(obs), pred=roller.series(prd),
                                 perm=perm, s_obs_all=s_obs_all)
    return pd.DataFrame(rows), series


def groups_all():
    out = []
    for title, _, members in PANELS:
        out += [(m, [m]) for m in members]
        if len(members) > 1:
            out.append((f'{title} average', members))
    return out


def plot(series, summ):
    halo = [pe.withStroke(linewidth=3.5, foreground='white')]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14.5))
    for col, (title, _, members) in enumerate(PANELS):
        top = f'{title} average' if len(members) > 1 else members[0]
        for row, (ax, key) in enumerate(zip(axes[:2, col], [top, None])):
            labels = [top] if row == 0 else members
            for lab in labels:
                if row == 1 and len(members) == 1:
                    break
                c = 'black' if row == 0 else STATION_COLORS[lab]
                for kind, ls, mk, a in (('obs', '-', 'o', 0.55), ('pred', '--', '^', 0.35)):
                    u, y = series[lab][kind]
                    ax.scatter(u, y, s=8, color=c, alpha=a, marker=mk, linewidths=0)
                    sl, ic = np.polyfit(u, y, 1)
                    xl = np.array([u.min(), u.max()])
                    name = 'observed' if kind == 'obs' else 'location-matched (fixed field)'
                    ax.plot(xl, sl * xl + ic, color=c, ls=ls, lw=2, path_effects=halo,
                            label=f'{lab} {name}: {sl:+.2f} %/m')
            if row == 1 and len(members) == 1:
                ax.axis('off')
                continue
            ax.set_title(title if row == 0 else f'{title}: stations', loc='left',
                         fontsize=13, fontweight='bold')
            ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
            ax.grid(alpha=0.3)
            ax.set_xlabel('De-tided uplift (m, 30-day rolling)')
            ax.set_ylabel('A = δt/T_S × 100 (%, 30-day rolling)')
        ax = axes[2, col]
        labs = members + ([top] if len(members) > 1 else [])
        for i, lab in enumerate(labs):
            s = series[lab]
            c = 'black' if lab == top else STATION_COLORS[lab]
            vp = ax.violinplot(s['perm'], positions=[i], widths=0.7, showextrema=False)
            for b in vp['bodies']:
                b.set_facecolor(c)
                b.set_alpha(0.3)
            ax.plot(i, s['s_obs_all'], 'D', color=c, ms=9, mec='k')
            p = summ.loc[(summ['group'] == lab) & (summ['cell_km'] == PLOT_CELL), 'p_perm'].iloc[0]
            ax.annotate(f'p={p:.3f}', (i, s['s_obs_all']), xytext=(12, 0),
                        textcoords='offset points', fontsize=8, va='center', ha='left')
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_xticks(range(len(labs)), labs, fontsize=8)
        ax.set_xlim(-0.5, len(labs) - 0.2)
        ax.set_ylabel('Slope of A on uplift (%/m)')
        ax.set_title(f'{title}: observed slope (diamond) vs\nwithin-cell permutations (fixed field)',
                     loc='left', fontsize=11)
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle(f'Test 1 — Is the A–uplift trend source migration? Location-matched null, '
                 f'{PLOT_CELL} km hypocentre cells, leave-one-year-out\n'
                 f'(post-eruption, Grade 3, {GEODETIC_LABEL})', fontsize=14, fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


def main():
    df = load_events()
    _dd, _infl, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    print(f'{len(df):,} post-eruption Grade-3 events with hypocentres and T_S')

    # sanity: the fast roller reproduces the production rolled() slope
    for lab, mem in [('West average', ['AXAS1', 'AXAS2']), ('AXEC2', ['AXEC2'])]:
        d = df[df['station'].isin(mem)]
        x, y, _ = rolled(d, 'A_pct', inflation_roll)
        print(f'  check {lab}: rolled() slope {np.polyfit(x, y, 1)[0]:+.4f}  '
              f'Roller {Roller(d["t"].reset_index(drop=True), inflation_roll).slope(d["A_pct"].values):+.4f}')

    frames, keep = [], None
    for ck in CELL_SIZES:
        s, ser = run_null(df, inflation_roll, ck, groups_all(), keep_series=(ck == PLOT_CELL))
        frames.append(s)
        if ck == PLOT_CELL:
            keep = ser
    summ = pd.concat(frames, ignore_index=True)
    pd.set_option('display.width', 200)
    print(summ.round(3).to_string(index=False))
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    summ.to_csv(SUMMARY_CSV, index=False)
    print(f'Saved {SUMMARY_CSV}')
    plot(keep, summ)


if __name__ == '__main__':
    main()
