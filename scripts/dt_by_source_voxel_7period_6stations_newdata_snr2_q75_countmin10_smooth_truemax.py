#!/usr/bin/env python3
"""
dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.py

SOURCE-VOXEL variant of the "3 depth range" newdata figure family, per explicit user
request: instead of travel-time-weighted PERCENT ANISOTROPY aggregated by RAY-TRACING
COVERAGE (traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_
truemax.py and its diff/depth2p5 siblings), this plots mean DELAY TIME (dt, seconds)
aggregated by EARTHQUAKE SOURCE (hypocenter) VOXEL -- each event is binned into the voxel
containing its own location (x, y, z from the catalog), not distributed along a traced ray
path. No ray tracing at all is needed for this method (T._setup_environment is still called,
but only with need_tracer=False, to reuse the bathymetry/station-plotting helpers).

Same data/filter (Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s,
dt<=T_dom/2, phi_err<=20 deg), same 0.25 km voxel grid, same 7 eruption-relative periods, same
0.0-2.25 km / 3 equal 0.75 km depth rows, same COUNT_MIN=10 coverage gate (here: event count
per voxel, since there is no ray path length to also gate on), same Gaussian smoothing
(sigma=1.0), and same true-max colorbar convention (reflecting the actual smoothed/masked
plotted field, not a percentile and not the raw pre-smoothing data) as the ray-traced sibling.

Produces:
    dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.pdf
(regular, 1 page, single shared true-max colorbar) -- a NEW file, does not touch any existing
output.

Run with:
    python3 dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(
    HERE, 'dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75
assert GRADE['key'] == 'page1_q0.75', GRADE

COUNT_MIN = 10

DEPTH_ROWS = [
    ([(0, 1.0), (1, 1.0), (2, 1.0)], '0.0–0.75 km', 0.375),
    ([(3, 1.0), (4, 1.0), (5, 1.0)], '0.75–1.5 km', 1.125),
    ([(6, 1.0), (7, 1.0), (8, 1.0)], '1.5–2.25 km', 1.875),
]

FS_N = 7.5
FS_PERIOD = 8.5


def load_events():
    dfs = [T.load_station_broadest(sta) for sta in T.STATIONS]
    df = pd.concat(dfs, ignore_index=True)
    mask = ((df['quality'].values >= GRADE['qw_min']) &
            (df['snr_horizontal'].values >= GRADE['snr_min']) &
            (df['phi_error'].values <= GRADE['phi_err_max']))
    return df[mask].reset_index(drop=True)


def voxel_indices(df):
    ix = np.floor((df['x'].values - T.X_START) / T.VXY).astype(np.int64)
    iy = np.floor((df['y'].values - T.Y_START) / T.VXY).astype(np.int64)
    iz = np.floor(df['z'].values / T.VZ).astype(np.int64)
    valid = ((df['x'].values >= T.X_START) & (df['x'].values < T.X_END) &
             (df['y'].values >= T.Y_START) & (df['y'].values < T.Y_END) &
             (df['z'].values >= 0) & (df['z'].values < T.Z_MAX) &
             (ix >= 0) & (ix < T.NX) & (iy >= 0) & (iy < T.NY) & (iz >= 0) & (iz < T.NZ))
    return ix, iy, iz, valid


def aggregate_source_voxel(df, periods):
    event_ns = pd.DatetimeIndex(df['t']).asi8
    event_period = T.assign_periods(event_ns, periods)
    ix, iy, iz, valid = voxel_indices(df)
    flat = ix * (T.NY * T.NZ) + iy * T.NZ + iz
    dt = df['dt'].values.astype(np.float64)

    J = T.NX * T.NY * T.NZ
    keep = valid & (event_period >= 0)
    key = event_period[keep].astype(np.int64) * J + flat[keep]
    d = dt[keep]

    uniq, inv = np.unique(key, return_inverse=True)
    sum_dt = np.zeros(len(uniq)); np.add.at(sum_dt, inv, d)
    cnt = np.bincount(inv, minlength=len(uniq))
    mean_dt = sum_dt / np.maximum(cnt, 1)
    gate = cnt >= COUNT_MIN

    up = uniq // J
    uf = uniq % J
    uix = uf // (T.NY * T.NZ)
    ur = uf % (T.NY * T.NZ)
    uiy = ur // T.NZ
    uiz = ur % T.NZ

    results = []
    for pi, (lbl, _t0, _t1) in enumerate(periods):
        a3d = np.full((T.NX, T.NY, T.NZ), np.nan)
        sel = (up == pi) & gate
        a3d[uix[sel], uiy[sel], uiz[sel]] = mean_dt[sel]
        n = int((event_period == pi).sum())
        results.append(dict(label=lbl, n=n, a=a3d))
    return results


def _weighted_depth_slice(a3d, iz_weights):
    shape2d = a3d.shape[:2]
    num = np.zeros(shape2d)
    den = np.zeros(shape2d)
    for iz, w in iz_weights:
        layer = a3d[:, :, iz]
        valid = np.isfinite(layer)
        num[valid] += layer[valid] * w
        den[valid] += w
    out = np.full(shape2d, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _single_line_label(label):
    return label.replace('\n–', ' – ').replace('\n', ' ')


def make_page(results, period_labels, title):
    fields = {}
    all_vals = []
    for ri, (iz_weights, zlbl, z0) in enumerate(DEPTH_ROWS):
        for pi, res in enumerate(results):
            sl = _weighted_depth_slice(res['a'], iz_weights)
            msk = np.isfinite(sl)
            if msk.sum() >= COUNT_MIN:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                mp = np.ma.masked_where(~msk, sm)
                if not mp.mask.all():
                    fields[(pi, ri)] = mp
                    all_vals.append(mp.compressed())
                    continue
            fields[(pi, ri)] = None
    pooled = np.concatenate(all_vals) if all_vals else np.array([0.1])
    pooled = pooled[pooled > 0] if (pooled > 0).any() else pooled
    a_vmax = float(np.nanmax(pooled)) if pooled.size else 0.1
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 0.1

    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., a_vmax, 15)
    n_per = len(results)
    n_dep = len(DEPTH_ROWS)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.6))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.06, wspace=0.04,
                 top=0.87, bottom=0.11)
    last_h = None
    for ri, (iz_weights, zlbl, z0) in enumerate(DEPTH_ROWS):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            T._bathy(ax)
            mp = fields.get((ci, ri))
            if mp is not None:
                ax.contourf(T.Xg, T.Yg, mp, levels=levels, cmap=cmap,
                            vmin=0, vmax=a_vmax, extend='neither')
                last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, a_vmax))
                last_h.set_array([])
            T._sta(ax)
            T._faults(ax, z0)
            ax.set_xlim(T.X_START, T.X_END)
            ax.set_ylim(T.Y_START, T.Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=4)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f'N={res["n"]:,}', fontsize=FS_N, fontweight='bold', pad=30)
                ax_pos = ax.get_position()
                fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + 0.03,
                         _single_line_label(period_labels[ci]),
                         fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
            if ci == 0:
                ax.set_ylabel(f'{zlbl}', fontsize=7)
    if last_h:
        cax = fig.add_subplot(gs[:, n_per])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter('%.3f'))
        cb.set_label('Mean delay time per source voxel (s)\n(true max of plotted field)', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.015))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    return fig


def main():
    T._setup_environment(need_tracer=False)

    print('Loading source-location events (Page 1 @ quality>=0.75, no ray tracing needed)...')
    df = load_events()
    print(f'  {len(df):,} events pass the filter')

    all_t = df['t']
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]

    results = aggregate_source_voxel(df, periods)
    for r in results:
        covered = int(np.isfinite(r['a']).sum())
        print(f'  {r["label"].splitlines()[0]:20s}: N={r["n"]:,}  covered-voxels={covered:,}')

    title = 'Delay Time Across an Eruption Cycle'
    fig = make_page(results, period_labels, title)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
