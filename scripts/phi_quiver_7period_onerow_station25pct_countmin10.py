#!/usr/bin/env python3
"""
phi_quiver_7period_onerow_station25pct_countmin10.py

Fast-direction (phi) analog of traveltime_anisotropy_7period_onerow_station90pct_countmin10.py:
same single-row (0.0-2.25 km), 7-period, single grade-3 filter (Page 1 @ quality>=0.75:
SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg), same existing
ray-tracing cache (no re-tracing), same "credit only the station-nearest 25% of each ray's
path" geometric cut (voxel crossings projected onto the straight event->station line;
crossings with t < 0.75, i.e. the source-side 75% of the path, are discarded) and same
per-(x,y)-cell, per-ray dedup + COUNT_MIN=10 coverage gate -- but instead of crediting each
surviving cell with the ray's scalar percent-anisotropy value, each ray's FAST-DIRECTION
AZIMUTH (phi, axial mod 180) is credited, and each cell's value is the CIRCULAR MEAN
(not median) of the phi values of all its distinct contributing rays. Plotted as
double-headed quiver tick marks (undirected axial symmetry) colored on a cyclic ('hsv')
colormap normalized to [0, 180], same convention as phi_quiver_by_source_voxel_per_station_*.

Produces ONE PDF with 7 pages:
    - 6 per-station pages (rays restricted to that station only, its own triangle
      highlighted gold, the other 5 white)
    - 1 combined page (all 6 stations' rays pooled, all 6 triangles gold)

Output:
    phi_quiver_7period_onerow_station25pct_countmin10.pdf

Run with:
    python3 phi_quiver_7period_onerow_station25pct_countmin10.py
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
OUT_PDF = os.path.join(HERE, 'phi_quiver_7period_onerow_station25pct_countmin10.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg
assert GRADE['key'] == 'page1_q0.75', GRADE

COUNT_MIN = 10
KEEP_FRAC_NEAR_STATION = 0.25   # keep ONLY the station-nearest 25% of each ray's path
STATION_FRAC_MIN = 1.0 - KEEP_FRAC_NEAR_STATION   # t-threshold: keep crossings with t >= this

# Single depth row spanning 0.0-2.25 km (iz 0..8, VZ=0.25km cells -> 9 cells)
IZ0, IZ1 = 0, 9
ROW_LABEL = '0.0–2.25 km'
ROW_Z0 = 1.125   # mid-depth of the row, for the fault-trace depth correction

AZ_CMAP = plt.colormaps['hsv']
AZ_NORM = Normalize(vmin=0.0, vmax=180.0)


def build_event_location_lookup():
    print('Rebuilding broadest per-station loads to recover event (x, y, z) per ray...')
    lookup = {}
    for sta in T.STATIONS:
        df = T.load_station_broadest(sta)
        lookup[sta] = pd.DataFrame({'x': df['x'].values, 'y': df['y'].values,
                                     'z': df['z'].values}, index=df['t'].values)
        print(f'  {sta}: {len(df):,} broadest-filter events loaded')
    return lookup


def build_event_phi_lookup():
    print('Rebuilding broadest per-station loads to recover event phi per ray...')
    lookup = {}
    for sta in T.STATIONS:
        dfs = [pd.read_csv(os.path.join(T.TRANSFER_DIR, f)) for f in T.STATION_FILES[sta]]
        df = pd.concat(dfs, ignore_index=True)
        df = df[df['success'] == True].copy()
        df = df[df['dt'] > 0]
        df = df.dropna(subset=['latitude', 'longitude', 'depth', 'quality', 'dominant_period',
                                'snr_horizontal', 'phi_error', 'dt_error', 'phi'])
        df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
        df = df[(df['quality'] >= T.QW_MIN_BROAD) &
                (df['snr_horizontal'] >= T.SNR_MIN_BROAD) &
                (df['dt_error'] <= T.DT_ERR_MAX) &
                (df['dt'] <= df['dominant_period'] / 2.0) &
                (df['phi_error'] <= T.PHI_ERR_MAX_BROAD)]
        df['phi_az'] = df['phi'] % 180.0
        lookup[sta] = pd.Series(df['phi_az'].values, index=df['t'].values)
        print(f'  {sta}: {len(df):,} broadest-filter events loaded')
    return lookup


def _sta_markers(ax, active_sta):
    for sta, row in T._sta_df.iterrows():
        if active_sta is None or sta == active_sta:
            ax.plot(row['x'], row['y'], '^', ms=7, mfc='#FFD700', mec='k', mew=0.8, zorder=13)
        else:
            ax.plot(row['x'], row['y'], '^', ms=6, mfc='white', mec='k', mew=0.6, zorder=12)
        dx, dy = T.LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, T.STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


def make_page(active_sta, results, title):
    n_per = len(results)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, 1 * 3.0 + 0.7))
    gs = GridSpec(1, n_per + 1, width_ratios=[1] * n_per + [0.05], hspace=0.04, wspace=0.04,
                 top=0.78, bottom=0.14)

    tick_kw = dict(scale=18, width=0.006, headlength=0, headaxislength=0, headwidth=0,
                  pivot='middle', cmap=AZ_CMAP, norm=AZ_NORM, alpha=0.95, zorder=7)

    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci])
        T._bathy(ax)

        PHI = res['field']
        iix, iiy = np.where(np.isfinite(PHI))
        if len(iix):
            xo = T.X_START + (iix + 0.5) * T.VXY
            yo = T.Y_START + (iiy + 0.5) * T.VXY
            phi_vals = PHI[iix, iiy]
            ro = np.radians(phi_vals)
            uo, vo = np.sin(ro), np.cos(ro)
            ax.quiver(xo, yo, uo, vo, phi_vals, **tick_kw)
            ax.quiver(xo, yo, -uo, -vo, phi_vals, **tick_kw)

        _sta_markers(ax, active_sta)
        T._kid(ax)
        T._faults(ax, ROW_Z0)
        ax.set_xlim(T.X_START, T.X_END)
        ax.set_ylim(T.Y_START, T.Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f'{res["label"]}\nN={res["n"]:,}', fontsize=7.5, fontweight='bold', pad=2)
        if ci == 0:
            ax.set_ylabel(ROW_LABEL, fontsize=7)

    cax = fig.add_subplot(gs[:, n_per])
    sm = ScalarMappable(cmap=AZ_CMAP, norm=AZ_NORM)
    sm.set_array([])
    cb = fig.colorbar(cax=cax, mappable=sm, ticks=np.linspace(0, 180, 7))
    cb.set_label('Fast direction $\\phi$ (deg, axial 0-180)', fontsize=8)
    cb.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
    ]
    if active_sta is None:
        legend_handles.append(mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6,
                                            label='Station (new data)'))
    else:
        legend_handles.append(mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6,
                                            label=f'{active_sta} (this page)'))
        legend_handles.append(mlines.Line2D([], [], marker='^', color='w', mfc='white', mec='k', ms=6,
                                            label='Other stations'))
    legend_handles.append(mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.98)
    return fig


def main():
    T._setup_environment(need_tracer=False)

    cache_exists = os.path.exists(T.RAY_SUMMARY_CSV) and os.path.exists(T.RAY_VOXEL_CSV)
    if not cache_exists:
        raise RuntimeError(
            f'Ray-tracing cache not found ({T.RAY_SUMMARY_CSV}); run '
            f'traveltime_anisotropy_7period_6stations_newdata_snr_grades.py once first to build it.')

    summary, vox = T.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    all_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]
    ray_period = T.assign_periods(event_ns, periods)

    q = summary['quality'].values
    snr = summary['snr_horizontal'].values
    phe = summary['phi_error'].values
    grade_mask = (q >= GRADE['qw_min']) & (snr >= GRADE['snr_min']) & (phe <= GRADE['phi_err_max'])

    location_lookup = build_event_location_lookup()
    phi_lookup = build_event_phi_lookup()
    summary['t'] = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')

    ex = np.full(N, np.nan); ey = np.full(N, np.nan); ez = np.full(N, np.nan)
    sx = np.full(N, np.nan); sy = np.full(N, np.nan)
    phi_ray = np.full(N, np.nan)
    for sta in T.STATIONS:
        m = (summary['station'].values == sta)
        loc = location_lookup[sta].reindex(summary['t'].values[m])
        ex[m] = loc['x'].values
        ey[m] = loc['y'].values
        ez[m] = loc['z'].values
        phi_ray[m] = phi_lookup[sta].reindex(summary['t'].values[m]).values
        stx, sty = T.sta_xy[sta]
        sx[m] = stx
        sy[m] = sty
    n_loc_matched = int(np.isfinite(ex).sum())
    n_phi_matched = int(np.isfinite(phi_ray).sum())
    print(f'  matched event location for {n_loc_matched:,}/{N:,} cached rays')
    print(f'  matched event phi for {n_phi_matched:,}/{N:,} cached rays')

    print('\nProjecting each voxel crossing onto its ray\'s event->station line, '
          f'keeping only the station-nearest {100*KEEP_FRAC_NEAR_STATION:.0f}% of the path...')
    ix = vox['ix'].values.astype(np.int64)
    iy = vox['iy'].values.astype(np.int64)
    iz = vox['iz'].values.astype(np.int64)

    cx = T.xn[ix] + T.VXY / 2.0
    cy = T.yn[iy] + T.VXY / 2.0
    cz = T.zn[iz] + T.VZ / 2.0

    vex, vey, vez = ex[vray], ey[vray], ez[vray]
    vsx, vsy = sx[vray], sy[vray]

    dvx, dvy, dvz = vsx - vex, vsy - vey, (0.0 - vez)
    wx, wy, wz = cx - vex, cy - vey, cz - vez
    denom = dvx * dvx + dvy * dvy + dvz * dvz
    tproj = (wx * dvx + wy * dvy + wz * dvz) / np.maximum(denom, 1e-9)

    in_row = (iz >= IZ0) & (iz < IZ1)
    station_side = tproj >= STATION_FRAC_MIN
    keep = in_row & station_side & np.isfinite(vex) & np.isfinite(phi_ray[vray])
    n_before = int(in_row.sum())
    n_after = int(keep.sum())
    print(f'  {n_before:,} single-row voxel crossings -> {n_after:,} kept after the station-nearest '
          f'{100*KEEP_FRAC_NEAR_STATION:.0f}%% cut ({100*n_after/max(n_before,1):.1f}%%)')

    flat2d = ix[keep].astype(np.int64) * T.NY + iy[keep].astype(np.int64)
    ray_id_kept = vray[keep]

    # Dedupe: a ray contributes at most once per (x,y) cell.
    combo = ray_id_kept.astype(np.int64) * (T.NX * T.NY) + flat2d
    _, uniq_idx = np.unique(combo, return_index=True)
    ray_id_u = ray_id_kept[uniq_idx]
    flat2d_u = flat2d[uniq_idx]

    phi_u = phi_ray[ray_id_u]
    period_u = ray_period[ray_id_u]
    grade_u = grade_mask[ray_id_u]
    station_col_u = summary['station'].values[ray_id_u]

    def aggregate(page_mask):
        keep2 = grade_u & page_mask & (period_u >= 0) & np.isfinite(phi_u)
        results = []
        for pi, lbl in enumerate(period_labels):
            sel = keep2 & (period_u == pi)
            n_rays = int(np.unique(ray_id_u[sel]).size) if sel.any() else 0
            field = np.full((T.NX, T.NY), np.nan)
            if sel.any():
                key = flat2d_u[sel]
                phi_deg = phi_u[sel]
                ang2 = 2.0 * np.radians(phi_deg)
                uniq, inv = np.unique(key, return_inverse=True)
                sum_sin = np.zeros(len(uniq)); np.add.at(sum_sin, inv, np.sin(ang2))
                sum_cos = np.zeros(len(uniq)); np.add.at(sum_cos, inv, np.cos(ang2))
                cnt = np.bincount(inv, minlength=len(uniq))
                mean_phi = (np.degrees(np.arctan2(sum_sin, sum_cos)) / 2.0) % 180.0
                gate = cnt >= COUNT_MIN
                fi, fj = uniq // T.NY, uniq % T.NY
                field[fi[gate], fj[gate]] = mean_phi[gate]
            results.append(dict(label=lbl, n=n_rays, field=field))
        return results

    formula = (f'phi credited only to (x,y) cells within the station-nearest '
               f'{100*KEEP_FRAC_NEAR_STATION:.0f}% of each ray\'s event->station path, '
               'circular MEAN per cell')

    with PdfPages(OUT_PDF) as pdf:
        for sta in T.STATIONS:
            page_mask = (station_col_u == sta)
            n_total = int(np.unique(ray_id_u[grade_u & page_mask]).size)
            print(f'\n=== {sta}: {n_total:,} rays pass the filter ===')
            results = aggregate(page_mask)
            title = (f'{sta} (newdata) — fast direction by cell (circular mean), '
                     f'{ROW_LABEL} (7-period, single station, station-nearest '
                     f'{100*KEEP_FRAC_NEAR_STATION:.0f}%% of path only)\n{formula}\n'
                     f'{GRADE["label"]} (n={n_total:,})')
            fig = make_page(sta, results, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

        page_mask = np.ones(len(ray_id_u), dtype=bool)
        n_total = int(np.unique(ray_id_u[grade_u & page_mask]).size)
        print(f'\n=== Combined (all 6 stations): {n_total:,} rays pass the filter ===')
        results = aggregate(page_mask)
        title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — fast direction by cell '
                 f'(circular mean), {ROW_LABEL} (7-period, combined, station-nearest '
                 f'{100*KEEP_FRAC_NEAR_STATION:.0f}%% of path only)\n{formula}\n'
                 f'{GRADE["label"]} (n={n_total:,})')
        fig = make_page(None, results, title)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
