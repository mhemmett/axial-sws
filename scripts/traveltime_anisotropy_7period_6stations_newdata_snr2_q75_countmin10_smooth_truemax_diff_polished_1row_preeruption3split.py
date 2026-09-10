#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row_preeruption3split.py

Pre-eruption-split variant of
traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row.pdf
(same data/filter/cached ray tracing, GRADE=page1_q0.75, COUNT_MIN=10, Gaussian smoothing,
true-max colorbars, single row spanning the FULL 0.0-Z_MAX km depth, gap+colorbar layout -- see
that script's docstring, not repeated here). ALL of the original 7 eruption-cycle panels are
kept (per explicit user request -- an earlier draft of this script wrongly dropped syn-eruption
and the 5 post-eruption panels), but the original single "Pre-eruption" panel is now SPLIT into
3 EQUAL-EVENT-COUNT sub-periods, to check for significant percent-anisotropy changes in the
run-up to the 2015 eruption. The pre-eruption window (2015-01-22 to 2015-04-24) is short enough
that month/year-level period labels aren't distinguishing -- the 3 pre-eruption panel titles
show exact dates (day precision); the unchanged syn-eruption/post-eruption panels keep the
original month/year labels from build_7_periods.

So the figure now has 9 panels total: 3 pre-eruption thirds + Syn-eruption + 5 equal-count
post-eruption bins (the last 6 identical in definition/boundaries to the original 7-period
figure this is derived from).

Layout (same convention as the "_diff_polished_1row" figure this is derived from): panel 1
(earliest pre-eruption third) plotted in absolute percent anisotropy (Blues, true-max colorbar,
own colorbar immediately to its right); ALL remaining 8 panels (the other two pre-eruption
thirds, Syn-eruption, and the 5 post-eruption bins) plotted as DIFFERENCES from panel 1 (RdBu_r,
true-max symmetric colorbar), positioned to the right of panel 1's colorbar, per explicit user
request.

Produces:
    traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row_preeruption3split.pdf
(1 page, 1 row, 9 panels) -- a NEW file, does not touch any existing output.

Run with:
    python3 traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row_preeruption3split.py
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
    HERE,
    'traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_smooth_truemax_diff_polished_1row_preeruption3split.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75
assert GRADE['key'] == 'page1_q0.75', GRADE

NEW_COUNT_MIN = 10

# Single row: ALL depths that pass the QC/ray-trace-eligibility filter -- same convention as
# the "_diff_polished_1row" figure this is derived from.
DEPTH_ROWS_ALL = [
    ([(iz, 1.0) for iz in range(T.NZ)], f'All depths (0.0–{T.Z_MAX:.1f} km)', T.Z_MAX / 2.0),
]

FS_N = 7.5
FS_PERIOD = 8.5


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


def _smoothed_masked_slices(results, depth_rows, count_min_display):
    fields = {}
    for ri, (iz_weights, zlbl, z0) in enumerate(depth_rows):
        for pi, res in enumerate(results):
            sl = _weighted_depth_slice(res['a'], iz_weights)
            msk = np.isfinite(sl)
            if msk.sum() >= count_min_display:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                fields[(pi, ri)] = (sm, msk)
            else:
                fields[(pi, ri)] = (None, msk)
    return fields


def _single_line_label(label):
    return label.replace('\n–', ' – ').replace('\n', ' ')


def make_diff_page(results, period_labels, title, depth_rows, fields):
    n_per = len(results)
    n_dep = len(depth_rows)

    abs_vals = []
    diff_vals = []
    diffs = {}
    pre_masked = {}
    for ri in range(n_dep):
        sm0, msk0 = fields[(0, ri)]
        if sm0 is not None:
            mp0 = np.ma.masked_where(~msk0, sm0)
            pre_masked[ri] = mp0
            abs_vals.append(mp0.compressed()[mp0.compressed() > 0])
        else:
            pre_masked[ri] = None
        for pi in range(1, n_per):
            sm_p, msk_p = fields[(pi, ri)]
            if sm0 is not None and sm_p is not None:
                joint = msk0 & msk_p
                if joint.any():
                    d = np.ma.masked_where(~joint, sm_p - sm0)
                    diffs[(pi, ri)] = d
                    diff_vals.append(d.compressed())
                    continue
            diffs[(pi, ri)] = None

    a_vmax = float(np.nanmax(np.concatenate(abs_vals))) if abs_vals and any(v.size for v in abs_vals) else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.
    all_diff = np.concatenate(diff_vals) if diff_vals else np.array([])
    d_vmax = float(np.nanmax(np.abs(all_diff))) if all_diff.size else 10.
    if not np.isfinite(d_vmax) or d_vmax <= 0:
        d_vmax = 10.

    cmap_abs = plt.colormaps['Blues']
    levels_abs = np.linspace(0., a_vmax, 15)
    cmap_diff = plt.colormaps['RdBu_r']
    levels_diff = np.linspace(-d_vmax, d_vmax, 21)

    panel_w = 2.5
    # Columns: [panel0, cbar_abs, spacer, diff1..diff(n_per-1), cbar_diff]
    n_cols = n_per + 3
    width_ratios = [1, 0.05, 0.18] + [1] * (n_per - 1) + [0.05]
    col_diff0 = 3   # first diff panel's grid column
    col_cbar_abs = 1
    col_cbar_diff = n_cols - 1

    header_room_in = 0.85
    footer_room_in = 0.32
    fig_h_in = n_dep * 3.6 + footer_room_in + header_room_in
    fig = plt.figure(figsize=(n_per * panel_w + 1.4, fig_h_in))
    gs_top = 1.0 - header_room_in / fig_h_in
    gs_bottom = footer_room_in / fig_h_in
    gs = GridSpec(n_dep, n_cols, width_ratios=width_ratios, hspace=0.05, wspace=0.04,
                 top=gs_top, bottom=gs_bottom)
    period_label_offset_frac = 0.32 / fig_h_in

    last_h_abs = None
    last_h_diff = None
    ref_ax_pos = None
    for ri, (iz_weights, zlbl, z0) in enumerate(depth_rows):
        for ci, res in enumerate(results):
            col = 0 if ci == 0 else col_diff0 + (ci - 1)
            ax = fig.add_subplot(gs[ri, col])
            T._bathy(ax)
            if ci == 0:
                mp = pre_masked[ri]
                if mp is not None and not mp.mask.all():
                    ax.contourf(T.Xg, T.Yg, mp, levels=levels_abs, cmap=cmap_abs,
                                vmin=0, vmax=a_vmax, extend='neither')
                    last_h_abs = ScalarMappable(cmap=cmap_abs, norm=Normalize(0, a_vmax))
                    last_h_abs.set_array([])
            else:
                d = diffs.get((ci, ri))
                if d is not None and not d.mask.all():
                    ax.contourf(T.Xg, T.Yg, d, levels=levels_diff, cmap=cmap_diff,
                                vmin=-d_vmax, vmax=d_vmax, extend='both')
                    last_h_diff = ScalarMappable(cmap=cmap_diff, norm=Normalize(-d_vmax, d_vmax))
                    last_h_diff.set_array([])
            T._sta(ax)
            T._faults(ax, z0)
            ax.set_xlim(T.X_START, T.X_END)
            ax.set_ylim(T.Y_START, T.Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=4)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f'N={res["n"]:,}', fontsize=FS_N, fontweight='bold', pad=14)
                ax_pos = ax.get_position()
                fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + period_label_offset_frac,
                         _single_line_label(period_labels[ci]),
                         fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
                ref_ax_pos = ax.get_position()
    if last_h_abs:
        cell = gs[:, col_cbar_abs].get_position(fig)
        cax = fig.add_axes([cell.x0, ref_ax_pos.y0, cell.width, ref_ax_pos.height])
        cb = plt.colorbar(last_h_abs, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter('%.1f'))
        cb.set_label('Percent Anisotropy', fontsize=7.5)
        cb.ax.tick_params(labelsize=6)
    if last_h_diff:
        cell2 = gs[:, col_cbar_diff].get_position(fig)
        cax2 = fig.add_axes([cell2.x0, ref_ax_pos.y0, cell2.width, ref_ax_pos.height])
        cb2 = plt.colorbar(last_h_diff, cax=cax2, ticks=np.linspace(-d_vmax, d_vmax, 5),
                           format=FormatStrFormatter('%.1f'))
        cb2.set_label('Change in Percent Anisotropy from Earliest Pre-Eruption Third', fontsize=7.5)
        cb2.ax.tick_params(labelsize=6)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.05))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    return fig


def _build_preeruption_3split_periods(summary, mask):
    """Splits the pre-eruption (t < ERUPTION_START), GRADE-eligible events into 3 equal-count
    sub-periods by time, mirroring T.build_7_periods' equal-count-bin technique, then appends
    the UNCHANGED Syn-eruption + 5 post-eruption periods from T.build_7_periods (built over the
    full, un-filtered event population, exactly as in the original 7-period figure). Returns
    (periods, period_labels) covering all 9 panels: periods is a (label, t0, t1) list for
    assign_periods/aggregate_page (pre-eruption bin boundaries are the NEXT bin's first event
    time, so those 3 bins are contiguous with no gaps); period_labels are separate,
    human-readable strings for display -- day-precision dates for the 3 pre-eruption panels,
    the original month/year labels for the other 6."""
    event_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    pre_mask = mask & (event_t < T.ERUPTION_START).values
    pre_times = event_t[pre_mask].sort_values().reset_index(drop=True)
    n = len(pre_times)
    if n < 3:
        raise SystemExit(f'Only {n} pre-eruption GRADE-eligible events -- cannot split into 3.')

    idxs = [0]
    for i in range(1, 3):
        idxs.append(min(int(round(i * n / 3)), n - 1))
    idxs.append(n)   # idxs = [0, i1, i2, n] -- bin k covers pre_times[idxs[k]:idxs[k+1]]

    bounds = [pre_times.iloc[idxs[1]], pre_times.iloc[idxs[2]]]   # b1, b2

    pre_periods = [
        ('Pre-eruption 1', None, bounds[0]),
        ('Pre-eruption 2', bounds[0], bounds[1]),
        ('Pre-eruption 3', bounds[1], T.ERUPTION_START),
    ]

    def _fmt(ts, with_year):
        return ts.strftime('%b %d, %Y') if with_year else ts.strftime('%b %d')

    pre_labels = []
    for k in range(3):
        start_ts = pre_times.iloc[idxs[k]]
        end_ts = pre_times.iloc[idxs[k + 1] - 1]
        pre_labels.append(f'{_fmt(start_ts, False)} – {_fmt(end_ts, True)}')

    all_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    periods_full = T.build_7_periods(pd.DataFrame({'t': all_t}))
    other_periods = periods_full[1:]   # Syn-eruption + 5 post-eruption bins, unchanged
    other_labels = [lbl for lbl, _t0, _t1 in other_periods]

    periods = pre_periods + list(other_periods)
    period_labels = pre_labels + other_labels
    return periods, period_labels


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

    mask = ((summary['quality'].values >= GRADE['qw_min']) &
            (summary['snr_horizontal'].values >= GRADE['snr_min']) &
            (summary['phi_error'].values <= GRADE['phi_err_max']))

    periods, period_labels = _build_preeruption_3split_periods(summary, mask)
    for lbl, (_, t0, t1) in zip(period_labels, periods):
        print(f'  {_single_line_label(lbl)}  (bin bounds: {t0} to {t1})')

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    ray_period = T.assign_periods(event_ns, periods)

    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    print(f'Setting per-voxel coverage gate: COUNT_MIN -> {NEW_COUNT_MIN} '
          f'(MIN_PATH_KM unchanged at {T.MIN_PATH_KM} km)')
    T.COUNT_MIN = NEW_COUNT_MIN

    results = T.aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
    n_total = int(mask.sum())
    print(f'  {GRADE["label"]}: {n_total:,} rays total; '
          f'{sum(r["n"] for r in results):,} across all 9 panels')

    print(f'Computing smoothed+masked field (single row, all depths 0-{T.Z_MAX:.1f}km combined)...')
    fields = _smoothed_masked_slices(results, DEPTH_ROWS_ALL, count_min_display=T.COUNT_MIN)

    title = ('Percent Anisotropy Across an Eruption Cycle '
              '(Pre-Eruption split into 3 Equal-Count Thirds)')
    fig = make_diff_page(results, period_labels, title, DEPTH_ROWS_ALL, fields)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
