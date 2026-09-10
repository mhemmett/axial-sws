#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_truemax.py

Same data/filter/grid/aggregation/cached ray tracing as
traveltime_anisotropy_7period_6stations_newdata_snr2_q75.py (and its _nosmooth sibling), but
the colorbar/contour scale (a_vmax) is set to the TRUE MAXIMUM over all covered voxel-period
cells instead of the 99th percentile. The 99th-percentile version silently clips ~1% of
cells (up to 9.98%, vs. a 6.13% colorbar top) to the same top color with no visual indicator
(contourf's extend='neither') -- this version fixes the scale itself so every cell's true
value is represented, at the cost of compressing color contrast among the other 99% of cells.

Produces two NEW PDFs (does not touch any existing output):
    traveltime_anisotropy_7period_6stations_newdata_snr2_q75_truemax.pdf           (smoothed, 1 page)
    traveltime_anisotropy_7period_6stations_newdata_snr2_q75_truemax_nosmooth.pdf  (unsmoothed, 1 page)

Run with:
    python3 traveltime_anisotropy_7period_6stations_newdata_snr2_q75_truemax.py
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
OUT_PDF_SMOOTH = os.path.join(HERE, 'traveltime_anisotropy_7period_6stations_newdata_snr2_q75_truemax.pdf')
OUT_PDF_NOSMOOTH = os.path.join(HERE, 'traveltime_anisotropy_7period_6stations_newdata_snr2_q75_truemax_nosmooth.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75
assert GRADE['key'] == 'page1_q0.75', GRADE


def _true_vmax(results):
    all_a = [r['a'][r['a'] > 0].ravel() for r in results if np.isfinite(r['a']).any()]
    if not all_a:
        return 30.
    pooled = np.concatenate(all_a)
    vmax = float(np.nanmax(pooled))
    return vmax if (np.isfinite(vmax) and vmax > 0) else 30.


def _make_page(results, title, a_vmax, smooth):
    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., a_vmax, 15)
    n_per = len(results)
    n_dep = len(T.DEPTH_ROWS)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.5))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.04, wspace=0.04)
    last_h = None
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(T.DEPTH_ROWS):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            T._bathy(ax)
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            msk = np.isfinite(sl)
            if msk.sum() >= T.COUNT_MIN:
                if smooth:
                    field = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                else:
                    field = sl
                mp = np.ma.masked_where(~msk, field)
                if not mp.mask.all():
                    ax.contourf(T.Xg, T.Yg, mp, levels=levels, cmap=cmap,
                                vmin=0, vmax=a_vmax, extend='neither')
                    last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, a_vmax))
                    last_h.set_array([])
            T._sta(ax)
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
        cb.set_label('Travel-time-weighted mean % shear anisotropy per voxel (true max scale)',
                     fontsize=9)
        cb.ax.tick_params(labelsize=7)
    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6,
                      label='Station (new data)'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.08)
    return fig


def main():
    cache_exists = os.path.exists(T.RAY_SUMMARY_CSV) and os.path.exists(T.RAY_VOXEL_CSV)
    if not cache_exists:
        raise SystemExit(
            f'Ray-tracing cache not found ({T.RAY_SUMMARY_CSV} / {T.RAY_VOXEL_CSV}). '
            'Run traveltime_anisotropy_7period_6stations_newdata_snr_grades.py first.')

    T._setup_environment(need_tracer=False)

    print('Reusing existing ray-tracing cache (no re-tracing)...')
    summary, vox = T.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    all_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))

    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    mask = ((summary['quality'].values >= GRADE['qw_min']) &
            (summary['snr_horizontal'].values >= GRADE['snr_min']) &
            (summary['phi_error'].values <= GRADE['phi_err_max']))
    ray_period = T.assign_periods(event_ns, periods)

    results = T.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
    n_total = int(mask.sum())
    covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
    print(f'  {GRADE["label"]}: {n_total:,} rays  covered-voxels={covered:,}')

    a_vmax = _true_vmax(results)
    print(f'  True-max colorbar scale: 0 to {a_vmax:.3f}% (vs. 99th-percentile scale used '
          f'in the sibling scripts)')

    formula = 'A = δt·100/T_travel (T_travel = Σ seg_len_km/Vs_voxel over crossed voxels)'

    title_smooth = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — travel-time-weighted '
                     f'mean %% anisotropy per voxel (7-period, TRUE-MAX color scale, smoothed)\n'
                     f'{formula}\n{GRADE["label"]} (n={n_total:,})')
    fig1 = _make_page(results, title_smooth, a_vmax, smooth=True)
    with PdfPages(OUT_PDF_SMOOTH) as pdf:
        pdf.savefig(fig1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved {OUT_PDF_SMOOTH}')

    title_nosmooth = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — travel-time-weighted '
                       f'mean %% anisotropy per voxel (7-period, TRUE-MAX color scale, UNSMOOTHED)\n'
                       f'{formula}\n{GRADE["label"]} (n={n_total:,})')
    fig2 = _make_page(results, title_nosmooth, a_vmax, smooth=False)
    with PdfPages(OUT_PDF_NOSMOOTH) as pdf:
        pdf.savefig(fig2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {OUT_PDF_NOSMOOTH}')

    print('\nDone.')


if __name__ == '__main__':
    main()
