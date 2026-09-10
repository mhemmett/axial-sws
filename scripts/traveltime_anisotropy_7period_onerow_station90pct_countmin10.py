#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_onerow_station90pct_countmin10.py

New version of the single-row (0.0-2.25 km), 7-period travel-time anisotropy map -- NO
diff-from-pre-eruption this time (absolute mean %% anisotropy per cell only) -- with a
different assumption about WHERE along each ray its anisotropy should be plotted:

Per explicit request, instead of distributing a ray's anisotropy across every voxel it
crosses (weighted by local path length/velocity, as the sibling ray-traced scripts do),
each ray's WHOLE per-ray anisotropy value (A_ray = dt*100/T_travel, same formula, same
existing ray-tracing cache) is plotted only in the 2D (x,y) cells that fall within the
station-NEAREST 25% of that ray's path -- i.e. each voxel crossing is projected onto the
straight event->station line, giving a fractional position t in [0,1] (0=event/source,
1=station); crossings with t < 0.75 (the source-side 75% of the path) are DISCARDED, keeping
only the last 25% of the path closest to the receiver, and every surviving crossing's cell
gets credited with the ray's full A_ray value (not a path-length-weighted fraction of it).
This reflects an assumption that the anisotropic signal is localized in the crust
immediately beneath the receiver, not distributed along the path.

A ray is counted at most once per (x,y) cell (multiple retained z-layers within the single
0.0-2.25 km row, for the same ray, are deduplicated) before computing each cell's coverage
count and mean A. Cells require >=10 distinct crossing rays (COUNT_MIN=10) to be plotted.

Same 0.25 km voxel grid, same 7 eruption-relative periods, same single grade-3 filter (Page
1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg) used
throughout this session, reusing the EXISTING ray-tracing cache (no re-tracing).

Produces ONE PDF with 7 pages:
    - 6 per-station pages (rays restricted to that station only, its own triangle
      highlighted gold, the other 5 white)
    - 1 combined page (all 6 stations' rays pooled, all 6 triangles gold)

Output:
    traveltime_anisotropy_7period_onerow_station90pct_countmin10.pdf

Run with:
    python3 traveltime_anisotropy_7period_onerow_station90pct_countmin10.py
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
OUT_PDF = os.path.join(HERE, 'traveltime_anisotropy_7period_onerow_station90pct_countmin10.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg
assert GRADE['key'] == 'page1_q0.75', GRADE

COUNT_MIN = 10
KEEP_FRAC_NEAR_STATION = 0.25   # keep ONLY the station-nearest 25% of each ray's path
STATION_FRAC_MIN = 1.0 - KEEP_FRAC_NEAR_STATION   # t-threshold: keep crossings with t >= this

# Single depth row spanning 0.0-2.25 km (iz 0..8, VZ=0.25km cells -> 9 cells)
IZ0, IZ1 = 0, 9
ROW_LABEL = '0.0–2.25 km'
ROW_Z0 = 1.125   # mid-depth of the row, for the fault-trace depth correction


def build_event_location_lookup():
    print('Rebuilding broadest per-station loads to recover event (x, y, z) per ray...')
    lookup = {}
    for sta in T.STATIONS:
        df = T.load_station_broadest(sta)
        lookup[sta] = pd.DataFrame({'x': df['x'].values, 'y': df['y'].values,
                                     'z': df['z'].values}, index=df['t'].values)
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
    all_vals = [r['field'][np.isfinite(r['field']) & (r['field'] > 0)] for r in results]
    all_vals = np.concatenate(all_vals) if any(v.size for v in all_vals) else np.array([])
    a_vmax = float(np.nanpercentile(all_vals, 99)) if all_vals.size else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.

    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., a_vmax, 15)
    n_per = len(results)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, 1 * 3.0 + 0.7))
    gs = GridSpec(1, n_per + 1, width_ratios=[1] * n_per + [0.05], hspace=0.04, wspace=0.04,
                 top=0.78, bottom=0.14)
    last_h = None
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci])
        T._bathy(ax)
        sl = res['field']
        msk = np.isfinite(sl)
        if msk.any():
            mp = np.ma.masked_where(~msk, np.nan_to_num(sl))
            ax.contourf(T.Xg, T.Yg, mp, levels=levels, cmap=cmap, vmin=0, vmax=a_vmax, extend='neither')
            last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, a_vmax))
            last_h.set_array([])
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

    if last_h:
        cax = fig.add_subplot(gs[:, n_per])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter('%.1f'))
        cb.set_label('Travel-time-weighted mean % shear anisotropy per cell', fontsize=8)
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

    print('\nComputing T_travel per ray and A_ray = dt*100/T_travel from cache...')
    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    q = summary['quality'].values
    snr = summary['snr_horizontal'].values
    phe = summary['phi_error'].values
    grade_mask = (q >= GRADE['qw_min']) & (snr >= GRADE['snr_min']) & (phe <= GRADE['phi_err_max'])

    location_lookup = build_event_location_lookup()
    summary['t'] = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')

    ex = np.full(N, np.nan); ey = np.full(N, np.nan); ez = np.full(N, np.nan)
    sx = np.full(N, np.nan); sy = np.full(N, np.nan)
    for sta in T.STATIONS:
        m = (summary['station'].values == sta)
        loc = location_lookup[sta].reindex(summary['t'].values[m])
        ex[m] = loc['x'].values
        ey[m] = loc['y'].values
        ez[m] = loc['z'].values
        stx, sty = T.sta_xy[sta]
        sx[m] = stx
        sy[m] = sty
    n_loc_matched = int(np.isfinite(ex).sum())
    print(f'  matched event location for {n_loc_matched:,}/{N:,} cached rays')

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
    keep = in_row & station_side & np.isfinite(vex)
    n_before = int(in_row.sum())
    n_after = int(keep.sum())
    print(f'  {n_before:,} single-row voxel crossings -> {n_after:,} kept after the station-nearest '
          f'{100*KEEP_FRAC_NEAR_STATION:.0f}%% cut ({100*n_after/max(n_before,1):.1f}%%)')

    flat2d = ix[keep].astype(np.int64) * T.NY + iy[keep].astype(np.int64)
    ray_id_kept = vray[keep]

    # Dedupe: a ray contributes at most once per (x,y) cell, even if it crosses multiple
    # retained z-layers within the single row at that same horizontal cell.
    combo = ray_id_kept.astype(np.int64) * (T.NX * T.NY) + flat2d
    _, uniq_idx = np.unique(combo, return_index=True)
    ray_id_u = ray_id_kept[uniq_idx]
    flat2d_u = flat2d[uniq_idx]

    a_ray_u = a_ray_tt[ray_id_u]
    period_u = ray_period[ray_id_u]
    grade_u = grade_mask[ray_id_u]
    station_col_u = summary['station'].values[ray_id_u]

    def aggregate(page_mask):
        keep2 = grade_u & page_mask & (period_u >= 0) & np.isfinite(a_ray_u)
        results = []
        for pi, lbl in enumerate(period_labels):
            sel = keep2 & (period_u == pi)
            n_rays = int(np.unique(ray_id_u[sel]).size) if sel.any() else 0
            field = np.full((T.NX, T.NY), np.nan)
            if sel.any():
                key = flat2d_u[sel]
                a = a_ray_u[sel]
                uniq, inv = np.unique(key, return_inverse=True)
                sum_a = np.zeros(len(uniq)); np.add.at(sum_a, inv, a)
                cnt = np.bincount(inv, minlength=len(uniq))
                mean_a = sum_a / np.maximum(cnt, 1)
                gate = cnt >= COUNT_MIN
                fi, fj = uniq // T.NY, uniq % T.NY
                field[fi[gate], fj[gate]] = mean_a[gate]
            results.append(dict(label=lbl, n=n_rays, field=field))
        return results

    formula = ('A = δt·100/T_travel per ray, credited only to (x,y) cells within the '
               f'station-nearest {100*KEEP_FRAC_NEAR_STATION:.0f}% of that ray\'s event->station path')

    with PdfPages(OUT_PDF) as pdf:
        for sta in T.STATIONS:
            page_mask = (station_col_u == sta)
            n_total = int(np.unique(ray_id_u[grade_u & page_mask]).size)
            print(f'\n=== {sta}: {n_total:,} rays pass the filter ===')
            results = aggregate(page_mask)
            title = (f'{sta} (newdata) — travel-time-weighted mean %% anisotropy per cell, '
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
        title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — travel-time-weighted mean %% '
                 f'anisotropy per cell, {ROW_LABEL} (7-period, combined, station-nearest '
                 f'{100*KEEP_FRAC_NEAR_STATION:.0f}%% of path only)\n'
                 f'{formula}\n{GRADE["label"]} (n={n_total:,})')
        fig = make_page(None, results, title)
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
