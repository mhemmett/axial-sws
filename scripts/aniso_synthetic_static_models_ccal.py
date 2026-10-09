#!/usr/bin/env python3
"""
aniso_synthetic_static_models_ccal.py

Test 2 of the source-migration synthetic test (Test 1 is aniso_source_migration_null_ccal.py):
forward-model delay times for the REAL post-eruption event-station rays through anisotropy fields
that do NOT change in time, push them through the same measurement grid / selection and the same
30-day roll on uplift, and see which (if any) reproduces the observed A-uplift trends
(West +, East flat-to-negative).

Rays: each Grade-3 event-station pair is traced through the Baillard 3-D Vs model with the
production PyKonal tracer (BaillardRayTracer, station at seafloor, 50 points source->receiver).
Each ray is stored as segment midpoints and segment S times dtau = ds / Vs(midpoint), cached once
to ../results/aniso_synthetic_ray_segments_grade3_v2.npz, keyed on
station | event_id | origin time (event_id alone repeats across the 2015-21 / 2022-26 files).

Forward model: k(x) = fractional S anisotropy dVs/Vs, so dt_syn = sum_seg k(mid) * dtau.

Static fields (normalised so the network-mean noise-free A equals the observed mean A, except D):
  A  uniform k                          -> A constant per ray; any trend is selection/noise only
  B  shallow-concentrated k ~ exp(-z/0.5 km)   (cracks closing with confining pressure)
  C  deep-increasing k ~ (z + 0.1 km)
  D  3-D field back-projected from the observed A (0.25 km voxels, each ray's A/100 spread over
     its voxels weighted by S time in the voxel). Built leave-one-calendar-year-out: rays of year
     Y are forward-modelled through the field built from the other years only, so no ray sees its
     own time-varying signal. Unhit voxels fall back to that year-fold's depth-layer mean.
  D+ control: D, with West-station rays multiplied by (1 + gamma * (u - mean u)), gamma set so the
     injected A slope is INJECT_PCT_PER_M. Checks that a genuine stress-driven change survives the
     grid/selection and is recovered by Test 1, while D alone is not flagged.

Measurement emulation per realisation: dt_meas = round(dt_syn + N(0, event dt_error), 10 ms);
keep 0 < dt_meas <= T_dom/2 (the Grade-3 dt cap, event's own T_dom). A = dt_meas / T_S * 100 with
the cached PyKonal T_S, as for the observed data. N_REAL noise realisations.

Test 1 (run_null) is then re-run on the synthetic A of D and D+ (one realisation each).

Outputs (new files):
  aniso_synthetic_static_models_ccal.pdf / .png
  ../results/aniso_synthetic_static_models_ccal_slopes.csv
  ../results/aniso_synthetic_static_models_ccal_test1_on_synthetics.csv
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import bpr_inflation_periods_ccal as ccal_infl
from atan2_uplift_vs_phi_sixstations_ccal_30day import GEODETIC_LABEL
from uplift_vs_phi_regions_scatter_ccal_30day import PANELS, STATION_COLORS
from aniso_source_migration_null_ccal import load_events, Roller, run_null, groups_all

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'aniso_synthetic_static_models_ccal')
RAY_CACHE = os.path.join(HERE, '..', 'results', 'aniso_synthetic_ray_segments_grade3_v2.npz')
SLOPES_CSV = os.path.join(HERE, '..', 'results', 'aniso_synthetic_static_models_ccal_slopes.csv')
T1_CSV = os.path.join(HERE, '..', 'results',
                      'aniso_synthetic_static_models_ccal_test1_on_synthetics.csv')

N_PTS = 50
VOX_KM = 0.25
DT_GRID = 0.01          # s, the measured dt grid
N_REAL = 20
INJECT_PCT_PER_M = 0.5  # ~ the West residual Test 1 leaves over
WEST = ['AXAS1', 'AXAS2']
SEED = 20261009
MODELS = ['A uniform', 'B shallow exp(-z/0.5)', 'C deep (z+0.1)', 'D back-projected 3-D (LOYO)',
          'D+ control: D + West injected']


def trace_segments(df):
    """(mid (N, N_PTS-1, 3), dtau (N, N_PTS-1)) for each row of df; cached."""
    if os.path.exists(RAY_CACHE):
        z = np.load(RAY_CACHE, allow_pickle=True)
        key = df['station'].astype(str) + '|' + df['event_id'].astype(str) + '|' + df['t'].astype(str)
        pos = pd.Series(np.arange(len(z['key'])), index=z['key'])
        if key.isin(pos.index).all():
            i = pos.loc[key].values
            print(f'Loaded {len(i):,} cached rays from {RAY_CACHE}')
            return z['mid'][i], z['dtau'][i]
        print('Ray cache does not cover the event set; re-tracing')
    from pykonal_raytracer import BaillardRayTracer
    tr = BaillardRayTracer()
    nseg = N_PTS - 1
    mid = np.full((len(df), nseg, 3), np.nan)
    dtau = np.full((len(df), nseg), np.nan)
    for sta, g in df.groupby('station'):
        tr.precompute_station(sta, g['sx'].iloc[0], g['sy'].iloc[0])
        nfail = 0
        for i, x, y, zz in zip(g.index, g['x'].values, g['y'].values, g['z'].values):
            try:
                ray = tr.trace(sta, x, y, zz, n_pts=N_PTS)
            except RuntimeError:
                nfail += 1
                continue
            m = 0.5 * (ray[1:] + ray[:-1])
            ds = np.linalg.norm(np.diff(ray, axis=0), axis=1)
            ix = np.clip(np.rint((m[:, 0] - tr._ox) / tr._dx).astype(int), 0, tr._nx - 1)
            iy = np.clip(np.rint((m[:, 1] - tr._oy) / tr._dy).astype(int), 0, tr._ny - 1)
            iz = np.clip(np.rint((m[:, 2] - tr._oz) / tr._dz).astype(int), 0, tr._nz - 1)
            mid[i] = m
            dtau[i] = ds / tr._vs[ix, iy, iz]
        print(f'  {sta}: traced {len(g) - nfail:,} rays ({nfail} failed)')
    key = (df['station'].astype(str) + '|' + df['event_id'].astype(str) + '|' + df['t'].astype(str)).values
    np.savez_compressed(RAY_CACHE, key=key, mid=mid, dtau=dtau)
    print(f'Cached rays to {RAY_CACHE}')
    return mid, dtau


def vox_index(mid, lo, shape):
    ijk = np.floor((mid - lo) / VOX_KM).astype(int)
    ijk = np.clip(ijk, 0, np.array(shape) - 1)
    return np.ravel_multi_index((ijk[..., 0], ijk[..., 1], ijk[..., 2]), shape)


def backprojected_loyo(df, mid, dtau):
    """Noise-free dt through the leave-one-year-out back-projected field (model D)."""
    lo = np.nanmin(mid.reshape(-1, 3), axis=0) - 1e-6
    hi = np.nanmax(mid.reshape(-1, 3), axis=0)
    shape = tuple(np.ceil((hi - lo) / VOX_KM).astype(int) + 1)
    nv = int(np.prod(shape))
    vid = vox_index(np.nan_to_num(mid, nan=0.0), lo, shape)
    w = np.nan_to_num(dtau)
    a = (df['A_pct'].values / 100.0)[:, None] * np.ones_like(w)
    zlayer = np.floor((np.nan_to_num(mid[..., 2]) - lo[2]) / VOX_KM).astype(int)
    yr = df['t'].dt.year.values
    dt_out = np.zeros(len(df))
    for y in np.unique(yr):
        tr_ = yr != y
        num = np.bincount(vid[tr_].ravel(), weights=(a[tr_] * w[tr_]).ravel(), minlength=nv)
        den = np.bincount(vid[tr_].ravel(), weights=w[tr_].ravel(), minlength=nv)
        lnum = np.bincount(zlayer[tr_].ravel(), weights=(a[tr_] * w[tr_]).ravel(),
                           minlength=shape[2] + 1)
        lden = np.bincount(zlayer[tr_].ravel(), weights=w[tr_].ravel(), minlength=shape[2] + 1)
        lmean = np.where(lden > 0, lnum / np.maximum(lden, 1e-12), np.nan)
        lmean = np.where(np.isfinite(lmean), lmean, np.nansum(lnum) / np.nansum(lden))
        k = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        te = yr == y
        kk = k[vid[te]]
        kk = np.where(np.isfinite(kk), kk, lmean[zlayer[te]])
        dt_out[te] = np.sum(kk * w[te], axis=1)
    return dt_out


def noise_free_models(df, mid, dtau, u_evt):
    w = np.nan_to_num(dtau)
    z = np.clip(np.nan_to_num(mid[..., 2]), 0.0, None)
    target = df['A_pct'].mean() / 100.0
    out = {}
    for name, shape_fn in ((MODELS[0], lambda z: np.ones_like(z)),
                           (MODELS[1], lambda z: np.exp(-z / 0.5)),
                           (MODELS[2], lambda z: z + 0.1)):
        raw = np.sum(shape_fn(z) * w, axis=1)
        out[name] = raw * target / np.mean(raw / df['T_s'].values)
    dD = backprojected_loyo(df, mid, dtau)
    out[MODELS[3]] = dD
    west = df['station'].isin(WEST).values
    gamma = INJECT_PCT_PER_M / df.loc[west, 'A_pct'].mean()
    uc = u_evt - np.nanmean(u_evt[west])
    out[MODELS[4]] = np.where(west, dD * (1 + gamma * uc), dD)
    return out


def measure(dt_true, df, rng):
    """Emulate the measurement: Gaussian dt_error noise, 10 ms grid, 0 < dt <= T_dom/2."""
    d = dt_true + rng.normal(0.0, df['dt_error'].values)
    d = np.round(d / DT_GRID) * DT_GRID
    keep = (d > 0) & (d <= df['dominant_period'].values / 2.0 + 1e-9)
    return np.where(keep, d / df['T_s'].values * 100.0, np.nan)


def slope_masked(df, A, members, inflation_roll):
    m = df['station'].isin(members).values & np.isfinite(A)
    sub = df.loc[m, ['t']].reset_index(drop=True)
    return Roller(sub['t'], inflation_roll).slope(A[m]), Roller(sub['t'], inflation_roll).series(A[m])


def main():
    rng = np.random.default_rng(SEED)
    df = load_events()
    _dd, _infl, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    ir = inflation_roll.dropna()
    tsec = lambda s: pd.to_datetime(s, utc=True).values.astype('datetime64[s]').astype(float)
    u_evt = np.interp(tsec(df['t']), tsec(pd.Series(ir.index)), ir.values)

    mid, dtau = trace_segments(df)
    ok = np.isfinite(dtau).all(axis=1)
    if (~ok).any():
        print(f'Dropping {int((~ok).sum())} events whose ray failed')
    df, mid, dtau, u_evt = df[ok].reset_index(drop=True), mid[ok], dtau[ok], u_evt[ok]
    t_ray = dtau.sum(axis=1)
    print(f'Ray-summed S time vs eikonal T_S: median ratio {np.median(t_ray / df["T_s"]):.3f}, '
          f'IQR {np.percentile(t_ray / df["T_s"], 25):.3f}-{np.percentile(t_ray / df["T_s"], 75):.3f}')

    nf = noise_free_models(df, mid, dtau, u_evt)
    groups = groups_all()
    rows, keep_series = [], {}
    for lab, mem in groups:
        s, ser = slope_masked(df, df['A_pct'].values, mem, inflation_roll)
        rows.append(dict(group=lab, model='Observed', real=0, slope=s))
        keep_series[(lab, 'Observed')] = ser
    synth_one = {}
    for name, dtt in nf.items():
        for r in range(N_REAL):
            A = measure(dtt, df, rng)
            if r == 0:
                synth_one[name] = A
            for lab, mem in groups:
                s, ser = slope_masked(df, A, mem, inflation_roll)
                rows.append(dict(group=lab, model=name, real=r, slope=s))
                if r == 0:
                    keep_series[(lab, name)] = ser
        print(f'  done {name}')
    sl = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SLOPES_CSV), exist_ok=True)
    sl.to_csv(SLOPES_CSV, index=False)
    tab = sl.groupby(['group', 'model'])['slope'].agg(['mean', 'std']).unstack('model')
    pd.set_option('display.width', 250)
    print('\nSlope of A on uplift (%/m), mean over realisations:')
    print(tab['mean'].round(3).to_string())
    print('\nstd over realisations:')
    print(tab['std'].round(3).to_string())

    # Test 1 on the synthetics: should explain D fully (resid ~0) and leave the injection in D+
    t1 = []
    for name in (MODELS[3], MODELS[4]):
        d = df.assign(A_pct=synth_one[name])
        d = d[np.isfinite(d['A_pct'])].reset_index(drop=True)
        s, _ = run_null(d, inflation_roll, 0.25, groups, n_perm=300)
        t1.append(s.assign(model=name))
    t1 = pd.concat(t1, ignore_index=True)
    t1.to_csv(T1_CSV, index=False)
    print('\nTest 1 re-run on synthetic A (one realisation):')
    print(t1[['model', 'group', 'slope_obs', 'slope_pred_loyo', 'slope_resid_loyo',
              'perm_median', 'p_perm']].round(3).to_string(index=False))
    print(f'Saved {SLOPES_CSV}\nSaved {T1_CSV}')

    plot(sl, keep_series)


def plot(sl, ser):
    halo = [pe.withStroke(linewidth=3.5, foreground='white')]
    cols = {'Observed': 'black', MODELS[0]: '#999999', MODELS[1]: '#648fff',
            MODELS[2]: '#785ef0', MODELS[3]: '#fe6100', MODELS[4]: '#dc267f'}
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for c, (title, _, members) in enumerate(PANELS):
        labs = members + ([f'{title} average'] if len(members) > 1 else [])
        ax = axes[0, c]
        order = ['Observed'] + MODELS
        for j, m in enumerate(order):
            for i, lab in enumerate(labs):
                v = sl[(sl['group'] == lab) & (sl['model'] == m)]['slope']
                y = i + (j - (len(order) - 1) / 2) * 0.12
                if m == 'Observed':
                    ax.plot(v.iloc[0], y, 'D', color='black', ms=8, label=m if i == 0 else None)
                else:
                    ax.errorbar(v.mean(), y, xerr=2 * v.std(), fmt='o', color=cols[m], ms=6,
                                capsize=2, label=m if i == 0 else None)
        ax.axvline(0, color='0.5', lw=0.8)
        ax.set_yticks(range(len(labs)), labs)
        ax.invert_yaxis()
        ax.set_xlabel('Slope of A on uplift (%/m); synthetics mean ± 2σ over noise draws')
        ax.set_title(f'{title}: slope by model', loc='left', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        if c == 0:
            ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

        ax = axes[1, c]
        lab = labs[-1]
        for m in ['Observed', MODELS[1], MODELS[2], MODELS[3]]:
            u, y = ser[(lab, m)]
            ax.scatter(u, y, s=7, color=cols[m], alpha=0.35, linewidths=0)
            s, i0 = np.polyfit(u, y, 1)
            xl = np.array([u.min(), u.max()])
            ax.plot(xl, s * xl + i0, color=cols[m], lw=2, path_effects=halo,
                    label=f'{m}: {s:+.2f} %/m')
        ax.set_title(f'{lab}: observed vs static-field synthetics', loc='left', fontsize=12)
        ax.set_xlabel('De-tided uplift (m, 30-day rolling)')
        ax.set_ylabel('A = δt/T_S × 100 (%, 30-day rolling)')
        ax.legend(fontsize=7.5, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)
    fig.suptitle('Test 2 — Static-anisotropy synthetics through the real (migrating) rays, '
                 'with 10 ms grid, dt_error noise and dt ≤ T_dom/2 selection\n'
                 f'(post-eruption, Grade 3, {GEODETIC_LABEL}; rays: PyKonal, Baillard 3-D Vs)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
