#!/usr/bin/env python3
"""
phi_quiver_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10.py

Fast-direction (phi) analog of
dt_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10_truemax.py:
same new 6-station "newdata" dataset, same single filter (Page 1 @ quality>=0.75: SNR>=2.0,
quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg), same 0.25 km voxel grid, same 7
eruption-relative periods, same 0.0-2.25 km / 3 equal 0.75 km depth rows, same COUNT_MIN=10
event-count-per-voxel gate, same one-page-per-station layout with the standard yellow
(active station) / white (other stations) marker convention -- but plots the CIRCULAR MEDIAN
fast direction (phi, axial mod 180) at each occupied voxel as a double-headed quiver tick
mark (undirected axial symmetry: two opposing arrowheads of length 0, i.e. a plain line
segment), colored by phi on a cyclic ('hsv') colormap normalized to [0, 180] so phi=0 and
phi=180 map to the same color -- same tick-mark convention as
spatial_map_fast_direction_5stations_newdata_7period.py and
deformation_geometry_stress_hemmett.py -- instead of a filled delay-time color field.

Each event is binned into the single voxel containing its own hypocenter (x, y, z); no ray
tracing.

Produces:
    phi_quiver_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10.pdf
(6 pages, one station per page) -- a NEW file, does not touch any existing output.

Run with:
    python3 phi_quiver_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(
    HERE, 'phi_quiver_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg
assert GRADE['key'] == 'page1_q0.75', GRADE
DT_ERR_MAX = T.DT_ERR_MAX

COUNT_MIN = 10

DEPTH_ROWS = [
    ((0, 3), '0.0–0.75 km', 0.375),
    ((3, 6), '0.75–1.5 km', 1.125),
    ((6, 9), '1.5–2.25 km', 1.875),
]

FS_N = 7.5
FS_PERIOD = 8.5

AZ_CMAP = plt.colormaps['hsv']
AZ_NORM = Normalize(vmin=0.0, vmax=180.0)


def load_station_events(sta):
    dfs = [pd.read_csv(os.path.join(T.TRANSFER_DIR, f)) for f in T.STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'quality', 'dominant_period',
                            'snr_horizontal', 'phi_error', 'dt_error', 'phi'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')

    mask = ((df['quality'] >= GRADE['qw_min']) &
            (df['snr_horizontal'] >= GRADE['snr_min']) &
            (df['dt_error'] <= DT_ERR_MAX) &
            (df['dt'] <= df['dominant_period'] / 2.0) &
            (df['phi_error'] <= GRADE['phi_err_max']))
    df = df[mask].copy()

    df['x'], df['y'] = T.ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    df['phi_az'] = df['phi'] % 180.0
    return df[['t', 'phi_az', 'x', 'y', 'z']].reset_index(drop=True)


def circular_median_phi(phi_deg):
    """Circular median of axially-symmetric (mod 180) fast directions (doubled-angle trick)."""
    if len(phi_deg) == 0:
        return np.nan
    angles = 2.0 * np.radians(phi_deg)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return float(np.degrees(np.arctan2(s, c)) / 2.0) % 180.0


def bin_by_epicenter(df, z_lo, z_hi):
    """Bin events into the 2D voxel containing their OWN epicenter (x,y), restricted to
    events whose own hypocentral depth falls in [z_lo, z_hi). Returns (NX,NY) circular-median
    phi array (NaN where count < COUNT_MIN)."""
    sub = df[(df['z'].values >= z_lo) & (df['z'].values < z_hi)]
    ix = np.floor((sub['x'].values - T.X_START) / T.VXY).astype(int)
    iy = np.floor((sub['y'].values - T.Y_START) / T.VXY).astype(int)
    valid = (ix >= 0) & (ix < T.NX) & (iy >= 0) & (iy < T.NY)
    ix, iy, phi = ix[valid], iy[valid], sub['phi_az'].values[valid]

    PHI = np.full((T.NX, T.NY), np.nan)
    if len(ix) == 0:
        return PHI

    flat = ix * T.NY + iy
    order = np.argsort(flat)
    flat_sorted, phi_sorted = flat[order], phi[order]
    uniq, start_idx, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
    for u, s0, c in zip(uniq, start_idx, counts):
        if c >= COUNT_MIN:
            vi, vj = u // T.NY, u % T.NY
            PHI[vi, vj] = circular_median_phi(phi_sorted[s0:s0 + c])
    return PHI


def aggregate_station(df, periods):
    results = []
    for lbl, t0, t1 in periods:
        m = np.ones(len(df), dtype=bool)
        if t0 is not None:
            m &= (df['t'].values >= np.datetime64(t0))
        if t1 is not None:
            m &= (df['t'].values < np.datetime64(t1))
        sub = df[m]
        phi_rows = [bin_by_epicenter(sub, iz0 * T.VZ, iz1 * T.VZ) for (iz0, iz1), _z, _z0 in DEPTH_ROWS]
        results.append(dict(label=lbl, n=len(sub), PHI_rows=phi_rows))
    return results


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
    n_per = len(results)
    n_dep = len(DEPTH_ROWS)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.6))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.06, wspace=0.04,
                 top=0.85, bottom=0.12)

    tick_kw = dict(scale=18, width=0.006, headlength=0, headaxislength=0, headwidth=0,
                  pivot='middle', cmap=AZ_CMAP, norm=AZ_NORM, alpha=0.95, zorder=7)

    for ri, (_iz_range, zlbl, z0) in enumerate(DEPTH_ROWS):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            T._bathy(ax)

            PHI = res['PHI_rows'][ri]
            iix, iiy = np.where(np.isfinite(PHI))
            if len(iix):
                xo = T.X_START + (iix + 0.5) * T.VXY
                yo = T.Y_START + (iiy + 0.5) * T.VXY
                phi_vals = PHI[iix, iiy]
                ro = np.radians(phi_vals)
                uo, vo = np.sin(ro), np.cos(ro)
                ax.quiver(xo, yo, uo, vo, phi_vals, **tick_kw)
                ax.quiver(xo, yo, -uo, -vo, phi_vals, **tick_kw)

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

    cax = fig.add_subplot(gs[:, n_per])
    sm = ScalarMappable(cmap=AZ_CMAP, norm=AZ_NORM)
    sm.set_array([])
    cb = fig.colorbar(cax=cax, mappable=sm, ticks=np.linspace(0, 180, 7))
    cb.set_label('Fast direction $\\phi$ (deg, axial 0-180)', fontsize=8)
    cb.ax.tick_params(labelsize=6)

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
                covered = sum(int(np.isfinite(p).sum()) for p in r['PHI_rows'])
                print(f'  {sta} [{r["label"].splitlines()[0]:20s}]: N={r["n"]:,}  '
                      f'covered-voxels={covered:,}')
            title = (f'Fast Direction by Source Voxel — {sta}\n'
                     f'{GRADE["label"]}, >={COUNT_MIN} events/voxel required')
            fig = make_page(sta, results, period_labels, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
