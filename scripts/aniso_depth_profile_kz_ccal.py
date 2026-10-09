#!/usr/bin/env python3
"""
aniso_depth_profile_kz_ccal.py

Follow-up to aniso_synthetic_static_models_ccal.py (Test 2): replace the assumed depth shapes
(B exp(-z/0.5), C z+0.1) with a depth profile of fractional S anisotropy k(z) FITTED to the
delay times, per region, then push the real (migrating) rays through it with the same 10 ms grid,
dt_error noise and dt <= T_dom/2 selection, and compare the synthetic A-uplift slope to the
observed one.

Forward model per ray i: dt_i = sum_l k_l * tau_il, tau_il = S time of the ray in depth layer l
(from the cached PyKonal ray segments, ../results/aniso_synthetic_ray_segments_grade3_v2.npz).

Two fits per region (West = AXAS1+AXAS2, Central = AXCC1, East = AXEC1-3), k_l >= 0:
  static : dt_i = sum_l k_l tau_il
  joint  : dt_i = sum_l k_l tau_il + g * (u_i - mean u) * T_i
           g is a uniform fractional change of anisotropy per m of uplift, fitted together with
           k(z). Because the events deepen as uplift grows, the static fit can absorb a genuine
           uplift change into k(z) (deeper = later = higher uplift); the joint fit controls for
           that. 100 * g (%/m) is a direct, migration-corrected estimate of the A-uplift slope.
Uncertainty: block bootstrap over calendar months (N_BOOT), respecting temporal clustering.

Synthetics: the static k(z) is fitted leave-one-calendar-year-out (rays of year Y forward-modelled
with the profile fitted to the other years), and the joint k(z) (static part only) likewise;
N_REAL measurement realisations each. Both are fixed fields, so any synthetic slope is migration +
selection. Caveat: k is fitted to the measured (gridded, capped) dt, then re-gridded/capped.

Outputs (new files):
  aniso_depth_profile_kz_ccal.pdf / .png
  ../results/aniso_depth_profile_kz_ccal_profiles.csv
  ../results/aniso_depth_profile_kz_ccal_slopes.csv
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import bpr_inflation_periods_ccal as ccal_infl
from atan2_uplift_vs_phi_sixstations_ccal_30day import GEODETIC_LABEL
from uplift_vs_phi_regions_scatter_ccal_30day import PANELS, STATION_COLORS
from aniso_source_migration_null_ccal import load_events, groups_all
from aniso_synthetic_static_models_ccal import trace_segments, measure, slope_masked

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'aniso_depth_profile_kz_ccal')
PROF_CSV = os.path.join(HERE, '..', 'results', 'aniso_depth_profile_kz_ccal_profiles.csv')
SLOPES_CSV = os.path.join(HERE, '..', 'results', 'aniso_depth_profile_kz_ccal_slopes.csv')

Z_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0])   # km below seafloor
N_BOOT = 200
N_REAL = 20
SEED = 20261009
REGION_COLORS = {'West': '#648fff', 'Central': '#785ef0', 'East': '#dc267f'}


def layer_times(mid, dtau):
    z = np.clip(np.nan_to_num(mid[..., 2]), 0.0, Z_EDGES[-1] - 1e-9)
    lay = np.clip(np.searchsorted(Z_EDGES, z, side='right') - 1, 0, len(Z_EDGES) - 2)
    nl = len(Z_EDGES) - 1
    tau = np.zeros((len(dtau), nl))
    w = np.nan_to_num(dtau)
    for l in range(nl):
        tau[:, l] = np.sum(np.where(lay == l, w, 0.0), axis=1)
    return tau


def fit(tau, dt, extra=None):
    """Bounded LSQ: k_l >= 0, optional extra column(s) unbounded. Layers with no ray time are 0."""
    used = tau.sum(axis=0) > 0
    X = tau[:, used] if extra is None else np.column_stack([tau[:, used], extra])
    nk = int(used.sum())
    lb = np.r_[np.zeros(nk), np.full(X.shape[1] - nk, -np.inf)]
    res = lsq_linear(X, dt, bounds=(lb, np.full(X.shape[1], np.inf)), method='bvls')
    k = np.full(tau.shape[1], np.nan)
    k[used] = res.x[:nk]
    return k, res.x[nk:]


def main():
    rng = np.random.default_rng(SEED)
    df = load_events()
    _dd, _infl, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    ir = inflation_roll.dropna()
    tsec = lambda s: pd.to_datetime(s, utc=True).values.astype('datetime64[s]').astype(float)
    df['u'] = np.interp(tsec(df['t']), tsec(pd.Series(ir.index)), ir.values)
    mid, dtau = trace_segments(df)
    ok = np.isfinite(dtau).all(axis=1)
    df, mid, dtau = df[ok].reset_index(drop=True), mid[ok], dtau[ok]
    tau = layer_times(mid, dtau)
    dt = df['dt'].values
    yr = df['t'].dt.year.values
    month = df['t'].dt.to_period('M').astype(str).values

    prof_rows, dt_static, dt_joint = [], np.full(len(df), np.nan), np.full(len(df), np.nan)
    gam = {}
    for title, _, members in PANELS:
        m = df['station'].isin(members).values
        tr, d, T = tau[m], dt[m], df['T_s'].values[m]
        uc = df['u'].values[m] - df['u'].values[m].mean()
        k_s, _ = fit(tr, d)
        k_j, g = fit(tr, d, (uc * T)[:, None])
        # block bootstrap over calendar months
        mo = month[m]
        umo = np.unique(mo)
        idx_by = pd.Series(np.arange(len(mo))).groupby(mo).indices
        bs_s, bs_j, bs_g = [], [], []
        for _ in range(N_BOOT):
            pick = np.concatenate([idx_by[x] for x in rng.choice(umo, len(umo))])
            a, _ = fit(tr[pick], d[pick])
            b, gg = fit(tr[pick], d[pick], (uc[pick] * T[pick])[:, None])
            bs_s.append(a); bs_j.append(b); bs_g.append(gg[0])
        bs_s, bs_j, bs_g = np.array(bs_s), np.array(bs_j), np.array(bs_g)
        gam[title] = (100 * g[0], 100 * np.percentile(bs_g, 2.5), 100 * np.percentile(bs_g, 97.5))
        hits = (tr > 0).sum(axis=0)
        for l in range(len(Z_EDGES) - 1):
            prof_rows.append(dict(region=title, z_top=Z_EDGES[l], z_bot=Z_EDGES[l + 1],
                                  n_rays=int(hits[l]), ray_time_s=float(tr[:, l].sum()),
                                  k_static_pct=100 * k_s[l],
                                  k_static_lo=100 * np.nanpercentile(bs_s[:, l], 2.5),
                                  k_static_hi=100 * np.nanpercentile(bs_s[:, l], 97.5),
                                  k_joint_pct=100 * k_j[l],
                                  k_joint_lo=100 * np.nanpercentile(bs_j[:, l], 2.5),
                                  k_joint_hi=100 * np.nanpercentile(bs_j[:, l], 97.5)))
        # leave-one-year-out static profiles -> noise-free synthetic dt
        mi = np.where(m)[0]
        for y in np.unique(yr[m]):
            te, trn = mi[yr[mi] == y], mi[yr[mi] != y]
            uct = df['u'].values[trn] - df['u'].values[trn].mean()
            a, _ = fit(tau[trn], dt[trn])
            b, _ = fit(tau[trn], dt[trn], (uct * df['T_s'].values[trn])[:, None])
            dt_static[te] = tau[te] @ np.nan_to_num(a)
            dt_joint[te] = tau[te] @ np.nan_to_num(b)
        print(f'{title}: 100*g = {gam[title][0]:+.3f} %/m  (95% block-bootstrap '
              f'{gam[title][1]:+.3f} .. {gam[title][2]:+.3f})')

    prof = pd.DataFrame(prof_rows)
    os.makedirs(os.path.dirname(PROF_CSV), exist_ok=True)
    prof.to_csv(PROF_CSV, index=False)
    pd.set_option('display.width', 220)
    print(prof.round(3).to_string(index=False))

    rows = []
    groups = groups_all()
    for lab, mem in groups:
        rows.append(dict(group=lab, model='Observed', real=0,
                         slope=slope_masked(df, df['A_pct'].values, mem, inflation_roll)[0]))
    for name, dtt in (('k(z) static fit (LOYO)', dt_static),
                      ('k(z) from joint fit (LOYO)', dt_joint)):
        for r in range(N_REAL):
            A = measure(dtt, df, rng)
            for lab, mem in groups:
                rows.append(dict(group=lab, model=name, real=r,
                                 slope=slope_masked(df, A, mem, inflation_roll)[0]))
    sl = pd.DataFrame(rows)
    for title in gam:
        lab = f'{title} average' if title != 'Central' else 'AXCC1'
        rows.append(dict(group=lab, model='joint-fit 100*g', real=0, slope=gam[title][0]))
    sl = pd.DataFrame(rows)
    sl.to_csv(SLOPES_CSV, index=False)
    tab = sl.groupby(['group', 'model'])['slope'].agg(['mean', 'std']).unstack('model')
    print('\nSlope of A on uplift (%/m): mean')
    print(tab['mean'].round(3).to_string())
    print('std over realisations')
    print(tab['std'].round(3).to_string())
    print(f'Saved {PROF_CSV}\nSaved {SLOPES_CSV}')
    plot(prof, sl, gam)


def plot(prof, sl, gam):
    fig, axes = plt.subplots(1, 4, figsize=(20, 7.5),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]))
    for ax, (title, _, members) in zip(axes[:3], PANELS):
        p = prof[prof['region'] == title]
        zc = 0.5 * (p['z_top'] + p['z_bot'])
        c = REGION_COLORS[title]
        for kind, ls, off in (('static', '-', -0.02), ('joint', '--', 0.02)):
            v = p[f'k_{kind}_pct']
            ax.errorbar(v, zc + off, xerr=[v - p[f'k_{kind}_lo'], p[f'k_{kind}_hi'] - v],
                        fmt='o' + ls, color=c, mfc='white' if kind == 'joint' else c, ms=6,
                        capsize=2, label=f'{kind} fit' + (' (with uplift term)' if kind == 'joint' else ''))
        for _, r in p.iterrows():
            ax.annotate(f'{r["n_rays"]:,}', (0, 0.5 * (r['z_top'] + r['z_bot'])),
                        xytext=(-4, 0), textcoords='offset points', ha='right', va='center',
                        fontsize=7, color='0.4')
        ax.set_ylim(2.1, -0.05)
        ax.set_xlim(left=-0.3 * max(p['k_static_hi'].max(), p['k_joint_hi'].max()))
        ax.axvline(0, color='0.6', lw=0.8)
        ax.set_xlabel('k = δVs/Vs (%) per depth layer, 95% block-bootstrap')
        ax.set_ylabel('Depth below seafloor (km)')
        g = gam[title]
        ax.set_title(f'{title} ({", ".join(members)})\njoint-fit uplift term 100g = {g[0]:+.2f} '
                     f'[{g[1]:+.2f}, {g[2]:+.2f}] %/m', loc='left', fontsize=11, fontweight='bold')
        ax.text(0.02, 0.02, 'grey numbers: rays sampling the layer; 2-4 km layer not shown',
                transform=ax.transAxes, fontsize=7, color='0.4')
        ax.legend(fontsize=8, loc='center right')
        ax.grid(alpha=0.3)

    ax = axes[3]
    labs = [g for g, _ in groups_all()]
    cols = {'Observed': 'black', 'k(z) static fit (LOYO)': '#fe6100',
            'k(z) from joint fit (LOYO)': '#ffb000', 'joint-fit 100*g': '#009e73'}
    order = list(cols)
    for j, mname in enumerate(order):
        for i, lab in enumerate(labs):
            v = sl[(sl['group'] == lab) & (sl['model'] == mname)]['slope']
            if v.empty:
                continue
            y = i + (j - 1.5) * 0.17
            if mname == 'Observed':
                ax.plot(v.iloc[0], y, 'D', color='black', ms=7, label=mname if i == 0 else None)
            elif mname == 'joint-fit 100*g':
                reg = 'Central' if lab == 'AXCC1' else lab.split()[0]
                g = gam[reg]
                ax.errorbar(g[0], y, xerr=[[g[0] - g[1]], [g[2] - g[0]]], fmt='s',
                            color=cols[mname], ms=6, capsize=2,
                            label='joint-fit uplift term 100g (migration-corrected), 95% CI'
                            if lab == 'West average' else None)
            else:
                ax.errorbar(v.mean(), y, xerr=2 * v.std(), fmt='o', color=cols[mname], ms=5,
                            capsize=2, label=f'{mname} synthetic, ±2σ' if i == 0 else None)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.set_yticks(range(len(labs)), labs)
    ax.invert_yaxis()
    ax.set_xlabel('Slope of A on uplift (%/m)')
    ax.set_title('Observed vs fitted-k(z) static-field synthetics', loc='left', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3, axis='x')
    fig.suptitle('Fitted depth profile of anisotropy k(z) per region, and the A–uplift slope it '
                 'predicts from source migration alone\n(post-eruption, Grade 3, rays: PyKonal / '
                 f'Baillard 3-D Vs; {GEODETIC_LABEL})', fontsize=13, fontweight='bold')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
