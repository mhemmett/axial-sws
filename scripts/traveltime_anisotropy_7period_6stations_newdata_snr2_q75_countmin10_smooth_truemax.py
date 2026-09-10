#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.py

Same data/filter/grid/cached ray tracing as the sibling scripts, with:
  - Per-voxel coverage gate COUNT_MIN=10 (MIN_PATH_KM unchanged at 0.5 km) -- between the
    original COUNT_MIN=8 and the COUNT_MIN=20 check, applied via a module-level override of
    traveltime_anisotropy_7period_6stations_newdata_snr_grades.COUNT_MIN before
    aggregate_page() runs (no re-tracing -- same cached ray-voxel crossings).
  - Gaussian smoothing (sigma=1.0 pixel = 0.25 km, same as the original smoothed scripts)
    IS applied this time, per explicit user request.
  - Colorbar/contour scale (a_vmax) is the TRUE MAX, but crucially computed from the ACTUAL
    SMOOTHED, MASKED field that gets drawn on each panel -- not the raw pre-smoothing
    per-voxel data. This matters because smoothing changes cell values (blends neighbors,
    including zero-filled uncovered voxels), so the honest "true max" for a colorbar that
    must match what's on the page is the max of the post-smoothing, post-masking array, not
    the raw aggregated array used in the earlier _truemax script.

Two-pass approach: first compute every panel's smoothed+masked field and take the true max
across all of them, then re-plot all panels on that single shared scale (so panels stay
mutually comparable, and the colorbar top is guaranteed to equal the single highest value
actually rendered anywhere on the page).

Produces: traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.pdf
(smoothed, 1 page) -- does not touch any existing output.

Run with:
    python3 traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.py
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
    HERE, 'traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75
assert GRADE['key'] == 'page1_q0.75', GRADE

NEW_COUNT_MIN = 10


def _smoothed_masked_fields(results, count_min_display):
    """Pass 1: compute the smoothed+masked field for every (period, depth-row) panel --
    exactly what make_page_smooth_truemax will later draw -- and return them alongside the
    true max over all panels' actually-displayed values."""
    fields = {}   # (period_idx, depth_row_idx) -> masked array or None
    all_vals = []
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(T.DEPTH_ROWS):
        for pi, res in enumerate(results):
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            msk = np.isfinite(sl)
            if msk.sum() >= count_min_display:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                mp = np.ma.masked_where(~msk, sm)
                if not mp.mask.all():
                    fields[(pi, ri)] = mp
                    all_vals.append(mp.compressed())
                    continue
            fields[(pi, ri)] = None
    pooled = np.concatenate(all_vals) if all_vals else np.array([30.])
    pooled = pooled[pooled > 0] if (pooled > 0).any() else pooled
    vmax = float(np.nanmax(pooled)) if pooled.size else 30.
    return fields, (vmax if (np.isfinite(vmax) and vmax > 0) else 30.)


def make_page_smooth_truemax(results, title, fields, a_vmax):
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
            mp = fields.get((ci, ri))
            if mp is not None:
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
        cb.set_label('Travel-time-weighted mean % shear anisotropy per voxel\n'
                     '(true max of the SMOOTHED, plotted field, COUNT_MIN=10)', fontsize=8)
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

    print(f'Setting per-voxel coverage gate: COUNT_MIN -> {NEW_COUNT_MIN} '
          f'(MIN_PATH_KM unchanged at {T.MIN_PATH_KM} km)')
    T.COUNT_MIN = NEW_COUNT_MIN

    results = T.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
    n_total = int(mask.sum())
    covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
    print(f'  {GRADE["label"]}: {n_total:,} rays  covered-voxels(count_min={NEW_COUNT_MIN})={covered:,}')

    print('Pass 1: computing smoothed+masked field for every panel to find the true '
          'plotted max...')
    fields, a_vmax = _smoothed_masked_fields(results, count_min_display=T.COUNT_MIN)
    print(f'  True max of the SMOOTHED, plotted field (colorbar top) = {a_vmax:.3f}%')

    formula = 'A = δt·100/T_travel (T_travel = Σ seg_len_km/Vs_voxel over crossed voxels)'
    title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — travel-time-weighted mean %% '
             f'anisotropy per voxel (7-period, smoothed, COUNT_MIN=10, true-max colorbar)\n'
             f'{formula}\n{GRADE["label"]} (n={n_total:,})')
    fig = make_page_smooth_truemax(results, title, fields, a_vmax)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
