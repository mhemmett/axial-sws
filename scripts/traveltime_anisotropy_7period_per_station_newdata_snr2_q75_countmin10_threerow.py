#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_per_station_newdata_snr2_q75_countmin10_threerow.py

Per-station, 3-depth-row travel-time-weighted percent-anisotropy map, accumulated along the
WHOLE ray (the standard path-length/velocity-weighted formula A_ray = dt*100/T_travel,
distributed across every voxel a ray crosses weighted by seg_len_km/Vs_voxel -- NOT the
"credit only the station-nearest X% of the path" restriction used by
traveltime_anisotropy_7period_onerow_station90pct_countmin10.py / phi_quiver_7period_onerow_
station25pct_countmin10.py this session).

Same new 6-station "newdata" dataset, same 0.25 km voxel grid, same 7 eruption-relative
periods, same single grade-3 filter (Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75,
dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg) used throughout this session, reusing the
EXISTING ray-tracing cache (no re-tracing). Depth is resolved into the 3 standard depth rows
used by this session's other "three depth bin" scripts (0.0-0.75 / 0.75-1.5 / 1.5-2.25 km,
dropping the sparse 2.25-3.0 km band), gated at >=10 crossing rays per voxel (COUNT_MIN=10,
raised from the sibling script's default COUNT_MIN=8).

ONE PAGE PER STATION (6 pages total), each restricted to ONLY that station's own rays (its
own triangle highlighted gold, the other 5 white) -- no diff-from-pre-eruption, no combined
all-station page.

Produces:
    traveltime_anisotropy_7period_per_station_newdata_snr2_q75_countmin10_threerow.pdf
(6 pages, one station per page) -- a NEW file, does not touch any existing output.

Run with:
    python3 traveltime_anisotropy_7period_per_station_newdata_snr2_q75_countmin10_threerow.py
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
    HERE, 'traveltime_anisotropy_7period_per_station_newdata_snr2_q75_countmin10_threerow.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg
assert GRADE['key'] == 'page1_q0.75', GRADE

T.COUNT_MIN = 10   # override module default (8) -- monkey-patch, same pattern as
                   # print_avg_pct_aniso_by_depth_period_5stations_countmin20.py

DEPTH_ROWS = [
    ((0, 3), '0.0–0.75 km', 0.375),
    ((3, 6), '0.75–1.5 km', 1.125),
    ((6, 9), '1.5–2.25 km', 1.875),
]


def _sta_markers(ax, active_sta):
    for sta, row in T._sta_df.iterrows():
        if sta == active_sta:
            ax.plot(row['x'], row['y'], '^', ms=7, mfc='#FFD700', mec='k', mew=0.8, zorder=13)
        else:
            ax.plot(row['x'], row['y'], '^', ms=6, mfc='white', mec='k', mew=0.6, zorder=12)
        dx, dy = T.LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, T.STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


def make_page(active_sta, results, title):
    all_a = [r['a'][r['a'] > 0].ravel() for r in results if np.isfinite(r['a']).any()]
    a_vmax = float(np.nanpercentile(np.concatenate(all_a), 99)) if all_a else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.
    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., a_vmax, 15)
    n_per = len(results)
    n_dep = len(DEPTH_ROWS)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.5))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.04, wspace=0.04)
    last_h = None
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(DEPTH_ROWS):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            T._bathy(ax)
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            msk = np.isfinite(sl)
            if msk.any():
                mp = np.ma.masked_where(~msk, np.nan_to_num(sl))
                ax.contourf(T.Xg, T.Yg, mp, levels=levels, cmap=cmap,
                            vmin=0, vmax=a_vmax, extend='neither')
                last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, a_vmax))
                last_h.set_array([])
            _sta_markers(ax, active_sta)
            T._kid(ax)
            T._faults(ax, z0)
            ax.set_xlim(T.X_START, T.X_END)
            ax.set_ylim(T.Y_START, T.Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=4)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f'{res["label"]}\nN={res["n"]:,}', fontsize=7.5, fontweight='bold', pad=1)
            if ci == 0:
                ax.set_ylabel(f'{zlbl}', fontsize=7)
    if last_h:
        cax = fig.add_subplot(gs[:, n_per])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter('%.1f'))
        cb.set_label('Travel-time-weighted mean % shear anisotropy per voxel', fontsize=9)
        cb.ax.tick_params(labelsize=7)
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
    fig.subplots_adjust(bottom=0.08)
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

    print('\nComputing T_travel per ray and A_ray = dt*100/T_travel from cache...')
    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    q = summary['quality'].values
    snr = summary['snr_horizontal'].values
    phe = summary['phi_error'].values
    grade_mask = (q >= GRADE['qw_min']) & (snr >= GRADE['snr_min']) & (phe <= GRADE['phi_err_max'])
    ray_period = T.assign_periods(event_ns, periods)
    station_col = summary['station'].values

    formula = 'A = δt·100/T_travel (T_travel = Σ seg_len_km/Vs_voxel over crossed voxels, whole ray)'

    with PdfPages(OUT_PDF) as pdf:
        for sta in T.STATIONS:
            mask = grade_mask & (station_col == sta)
            n_total = int(mask.sum())
            print(f'\n=== {sta}: {n_total:,} rays pass the filter ===')

            results = T.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
            for r in results:
                covered = sum(int(np.isfinite(r['a'][:, :, iz0:iz1]).any(axis=2).sum())
                              for (iz0, iz1), _z, _z0 in DEPTH_ROWS)
                print(f'  {r["label"].splitlines()[0]:20s}: N={r["n"]:,}  covered-cells(3 rows)={covered:,}')

            title = (f'{sta} (newdata) — travel-time-weighted mean %% anisotropy per voxel, '
                     f'whole ray, 3 depth rows (7-period, single station)\n{formula}\n'
                     f'{GRADE["label"]}, >={T.COUNT_MIN} crossing rays/voxel required (n={n_total:,})')
            fig = make_page(sta, results, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
