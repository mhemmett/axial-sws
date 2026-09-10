#!/usr/bin/env python3
"""
dt_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10_truemax.py

Per-station variant of dt_by_source_voxel_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.py:
same new 6-station "newdata" dataset, same single filter (Page 1 @ quality>=0.75: SNR>=2.0,
quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg), same 0.25 km voxel grid, same 7
eruption-relative periods, same 0.0-2.25 km / 3 equal 0.75 km depth rows, same COUNT_MIN=10
event-count-per-voxel gate, same true-max colorbar convention -- but:
  - NO Gaussian smoothing (plots the raw voxel means directly).
  - NO pre-eruption-difference page.
  - ONE page PER STATION (6 pages total), each page binning ONLY that station's own events
    into the source-voxel grid (not all 6 stations pooled together). All 6 station triangles
    are still drawn on every page for spatial context; only the page's own station is
    highlighted gold ('#FFD700'), the rest are plain white/black-outline triangles.

Produces:
    dt_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10_truemax.pdf
(6 pages, one station per page, each with its own true-max color scale) -- a NEW file, does
not touch any existing output.

Run with:
    python3 dt_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10_truemax.py
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

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(
    HERE, 'dt_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10_truemax.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg
assert GRADE['key'] == 'page1_q0.75', GRADE

COUNT_MIN = 10

DEPTH_ROWS = [
    ([(0, 1.0), (1, 1.0), (2, 1.0)], '0.0–0.75 km', 0.375),
    ([(3, 1.0), (4, 1.0), (5, 1.0)], '0.75–1.5 km', 1.125),
    ([(6, 1.0), (7, 1.0), (8, 1.0)], '1.5–2.25 km', 1.875),
]

FS_N = 7.5
FS_PERIOD = 8.5


def load_station_events(sta):
    df = T.load_station_broadest(sta)
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


def aggregate_station(df, periods):
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


def _sta_markers(ax, active_sta):
    for sta, row in T._sta_df.iterrows():
        if sta == active_sta:
            ax.plot(row['x'], row['y'], '^', ms=7, mfc='#FFD700', mec='k', mew=0.8, zorder=13)
        else:
            ax.plot(row['x'], row['y'], '^', ms=6, mfc='white', mec='k', mew=0.6, zorder=12)
        dx, dy = T.LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, T.STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


def make_page(sta, results, period_labels, title):
    fields = {}
    all_vals = []
    for ri, (iz_weights, zlbl, z0) in enumerate(DEPTH_ROWS):
        for pi, res in enumerate(results):
            sl = _weighted_depth_slice(res['a'], iz_weights)
            msk = np.isfinite(sl)
            if msk.any():
                mp = np.ma.masked_where(~msk, sl)
                fields[(pi, ri)] = mp
                all_vals.append(mp.compressed())
            else:
                fields[(pi, ri)] = None
    pooled = np.concatenate(all_vals) if all_vals else np.array([0.1])
    pooled = pooled[pooled > 0] if (pooled > 0).any() else pooled
    vmax = float(np.nanmax(pooled)) if pooled.size else 0.1
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.1

    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., vmax, 15)
    n_per = len(results)
    n_dep = len(DEPTH_ROWS)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.6))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.06, wspace=0.04,
                 top=0.85, bottom=0.12)
    last_h = None
    for ri, (iz_weights, zlbl, z0) in enumerate(DEPTH_ROWS):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            T._bathy(ax)
            mp = fields.get((ci, ri))
            if mp is not None:
                ax.contourf(T.Xg, T.Yg, mp, levels=levels, cmap=cmap,
                            vmin=0, vmax=vmax, extend='neither')
                last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
                last_h.set_array([])
            _sta_markers(ax, sta)
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
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, vmax, 5),
                          format=FormatStrFormatter('%.3f'))
        cb.set_label('Mean delay time per source voxel (s)\n(true max of plotted field)', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=7,
                      label=f'{sta} (this page)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='white', mec='k', ms=6, label='Other stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.02))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    return fig


def main():
    T._setup_environment(need_tracer=False)

    print(f'Loading per-station events ({GRADE["label"]})...')
    station_dfs = {}
    for sta in T.STATIONS:
        df = load_station_events(sta)
        station_dfs[sta] = df
        print(f'  {sta}: {len(df):,} events pass the filter')

    all_t = pd.concat([station_dfs[sta]['t'] for sta in T.STATIONS], ignore_index=True)
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]

    with PdfPages(OUT_PDF) as pdf:
        for sta in T.STATIONS:
            df = station_dfs[sta]
            results = aggregate_station(df, periods)
            for r in results:
                covered = int(np.isfinite(r['a']).sum())
                print(f'  {sta} [{r["label"].splitlines()[0]:20s}]: N={r["n"]:,}  '
                      f'covered-voxels={covered:,}')
            title = (f'Delay Time by Source Voxel — {sta}\n'
                     f'{GRADE["label"]}, >={COUNT_MIN} events/voxel required')
            fig = make_page(sta, results, period_labels, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
