#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_per_station_newdata_snr2_q75_diff_onerow.py

Per-station version of traveltime_anisotropy_7period_6stations_newdata_snr_grades.py's
pre-eruption-difference page (make_diff_page): same new 6-station "newdata" dataset, same
0.25 km voxel grid, same 7 eruption-relative periods, same travel-time-weighted percent-
anisotropy formula (A = dt*100/T_travel, reusing the EXISTING ray-tracing cache
traveltime_anisotropy_newdata_snr_grades_ray_summary.csv / _ray_voxel_segments.csv, already
built on disk -- no re-tracing), same single grade-3 filter (Page 1 @ quality>=0.75:
SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg) used throughout this
session's other newdata scripts -- but:
  - ONE ROW instead of 4 depth rows: rays are collapsed into a single 0.0-2.25 km depth band
    (the same span used by this session's other "three depth bin" scripts, dropping the
    sparse 2.25-3.0 km band), not split by depth.
  - ONE PAGE PER STATION (6 pages total), each restricted to ONLY that station's own rays
    (ray_survive mask ANDs the grade-3 filter with station==sta), not all 6 stations' rays
    pooled together.
  - Diff-from-pre-eruption layout only (pre-eruption absolute map + Delta-from-pre-eruption
    for the other 6 periods), matching make_diff_page's two-colorbar (Blues abs / RdBu_r
    diff) convention.

Produces:
    traveltime_anisotropy_7period_per_station_newdata_snr2_q75_diff_onerow.pdf
(6 pages, one station per page) -- a NEW file, does not touch any existing output.

Run with:
    python3 traveltime_anisotropy_7period_per_station_newdata_snr2_q75_diff_onerow.py
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
    HERE, 'traveltime_anisotropy_7period_per_station_newdata_snr2_q75_diff_onerow.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg
assert GRADE['key'] == 'page1_q0.75', GRADE

# Single depth row spanning 0.0-2.25 km (iz 0..8, VZ=0.25km cells -> 9 cells)
IZ0, IZ1 = 0, 9
ROW_LABEL = '0.0–2.25 km'
ROW_Z0 = 1.125   # mid-depth of the row, for the fault-trace depth correction


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

    print('\nComputing T_travel per ray and A_ray = dt*100/T_travel from cache...')
    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    q = summary['quality'].values
    snr = summary['snr_horizontal'].values
    phe = summary['phi_error'].values
    grade_mask = (q >= GRADE['qw_min']) & (snr >= GRADE['snr_min']) & (phe <= GRADE['phi_err_max'])

    ray_period = T.assign_periods(event_ns, periods)
    station_col = summary['station'].values

    formula = 'A = δt·100/T_travel (T_travel = Σ seg_len_km/Vs_voxel over crossed voxels)'

    with PdfPages(OUT_PDF) as pdf:
        for sta in T.STATIONS:
            mask = grade_mask & (station_col == sta)
            n_total = int(mask.sum())
            print(f'\n=== {sta}: {n_total:,} rays pass the filter ===')

            results = T.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)

            title = (f'{sta} (newdata) — travel-time-weighted mean %% anisotropy per voxel, '
                     f'{ROW_LABEL} (7-period, single station)\n{formula}\n'
                     f'{GRADE["label"]} (n={n_total:,})')
            fig = make_diff_page_onerow(sta, results, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


def _sta_markers(ax, active_sta):
    for sta, row in T._sta_df.iterrows():
        if sta == active_sta:
            ax.plot(row['x'], row['y'], '^', ms=7, mfc='#FFD700', mec='k', mew=0.8, zorder=13)
        else:
            ax.plot(row['x'], row['y'], '^', ms=6, mfc='white', mec='k', mew=0.6, zorder=12)
        dx, dy = T.LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, T.STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


def make_diff_page_onerow(active_sta, results, title):
    pre_a2d = np.nanmean(results[0]['a'][:, :, IZ0:IZ1], axis=2)

    diff_2d = []
    for r in results[1:]:
        sl = np.nanmean(r['a'][:, :, IZ0:IZ1], axis=2)
        diff_2d.append(sl - pre_a2d)

    all_abs_pre = pre_a2d[np.isfinite(pre_a2d) & (pre_a2d > 0)]
    a_vmax = float(np.nanpercentile(all_abs_pre, 99)) if all_abs_pre.size else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.

    all_diff_vals = np.concatenate([d[np.isfinite(d)].ravel() for d in diff_2d]) if diff_2d else np.array([])
    d_vmax = float(np.nanpercentile(np.abs(all_diff_vals), 99)) if all_diff_vals.size else 10.
    if not np.isfinite(d_vmax) or d_vmax <= 0:
        d_vmax = 10.

    cmap_abs = plt.colormaps['Blues']
    levels_abs = np.linspace(0., a_vmax, 15)
    cmap_diff = plt.colormaps['RdBu_r']
    levels_diff = np.linspace(-d_vmax, d_vmax, 21)

    n_per = len(results)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 1.0, 1 * 3.0 + 0.9))
    gs = GridSpec(1, n_per + 2, width_ratios=[0.05] + [1] * n_per + [0.05], hspace=0.04, wspace=0.04,
                 top=0.72, bottom=0.14)
    last_h_abs = None
    last_h_diff = None
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci + 1])
        T._bathy(ax)
        if ci == 0:
            sl = pre_a2d
        else:
            sl = diff_2d[ci - 1]
        msk = np.isfinite(sl)
        if msk.any():
            mp = np.ma.masked_where(~msk, np.nan_to_num(sl))
            if ci == 0:
                ax.contourf(T.Xg, T.Yg, mp, levels=levels_abs, cmap=cmap_abs,
                            vmin=0, vmax=a_vmax, extend='neither')
                last_h_abs = ScalarMappable(cmap=cmap_abs, norm=Normalize(0, a_vmax))
                last_h_abs.set_array([])
            else:
                ax.contourf(T.Xg, T.Yg, mp, levels=levels_diff, cmap=cmap_diff,
                            vmin=-d_vmax, vmax=d_vmax, extend='both')
                last_h_diff = ScalarMappable(cmap=cmap_diff, norm=Normalize(-d_vmax, d_vmax))
                last_h_diff.set_array([])
        _sta_markers(ax, active_sta)
        T._kid(ax)
        T._faults(ax, ROW_Z0)
        ax.set_xlim(T.X_START, T.X_END)
        ax.set_ylim(T.Y_START, T.Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        title_txt = f'{res["label"]}\nN={res["n"]:,}' + ('' if ci == 0 else ' (Δ vs pre)')
        ax.set_title(title_txt, fontsize=7.5, fontweight='bold', pad=1)
        if ci == 0:
            ax.set_ylabel(ROW_LABEL, fontsize=7)

    if last_h_abs:
        cax = fig.add_subplot(gs[:, 0])
        cb = plt.colorbar(last_h_abs, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter('%.1f'))
        cb.set_label('Pre-eruption: mean % anisotropy per voxel', fontsize=8)
        cb.ax.tick_params(labelsize=6)
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')
    if last_h_diff:
        cax2 = fig.add_subplot(gs[:, n_per + 1])
        cb2 = plt.colorbar(last_h_diff, cax=cax2, ticks=np.linspace(-d_vmax, d_vmax, 5),
                           format=FormatStrFormatter('%.1f'))
        cb2.set_label('Difference from pre-eruption (percentage points)', fontsize=8)
        cb2.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6,
                      label=f'{active_sta} (this page)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='white', mec='k', ms=6, label='Other stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    return fig


if __name__ == '__main__':
    main()
