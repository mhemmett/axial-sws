#!/usr/bin/env python3
"""
axec1_windowcheck_vs_previous_dt_and_midpoint_traveltime_anisotropy.py

Single-station (AXEC1) BEFORE/AFTER comparison: the previous (pre-windowcheck-fix) production
AXEC1 2015-2021 dataset vs. the new partial windowcheck-fix rerun
(mfast_maxdt_windowcheck_pipeline_transfer/splitting_results_AXEC1_2015_2021_combined_partial.csv,
produced on another computer per user instruction), both restricted to the SAME date range --
up to the new partial data's latest event (2015-04-24T18:14:47.446Z, i.e. right at/just before
the 2015 eruption onset) -- so the comparison isn't confounded by the new run simply covering
less time.

Two figures, each with 2 pages (one per filter set) and 2 side-by-side panels per page
(Previous | Window-check), single row (all depths 0.0-Z_MAX km combined, per the session's
current convention for these single-station comparison figures):

  1. dt_by_source_voxel_axec1_windowcheck_vs_previous_1row.pdf -- mean delay time (dt, seconds)
     binned directly by EARTHQUAKE SOURCE location (event hypocenter x,y, depth collapsed),
     NO ray tracing. COUNT_MIN=10 events/cell (raised from an initial 5 per explicit user
     instruction, now matching the midpoint-anisotropy figure's gate below).

  2. traveltime_anisotropy_axec1_windowcheck_vs_previous_midpoint_1row.pdf -- percent shear-wave
     anisotropy (100*dt/T_travel, T_travel = summed segment travel time along the ray through
     the Baillard Vs model) assigned ENTIRELY to the voxel containing the arc-length MIDPOINT of
     each ray's traced path (same midpoint-assignment convention as
     traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_*.py),
     COUNT_MIN=10 distinct ray-midpoints/cell. Both the previous and window-check AXEC1 events
     are traced FRESH here (n_pts=200, stride=5, matching
     traveltime_anisotropy_7period_6stations_newdata_snr_grades.py's tracer settings) -- the
     previous dataset IS part of that script's existing ray cache, but that cache doesn't store
     dominant_period (needed for the grade-3 dt<=T_dom/2 cut) or a matching event-datetime key
     cheap enough to look up per filter here, so both sides are re-traced identically for a
     clean, symmetric comparison rather than mixing a cached-lookup path for one side with a
     fresh-trace path for the other.

Two filter sets (both also require success=True, dt>0, and the same base QC as this session's
grade-3 filter -- SNR>=2.0, quality>=0.75, dt_err<=0.05s, phi_err<=20deg):
  - "Grade 3": + dt <= T_dom/2 (per-event dominant-period cycle-skip cut)
  - "Grade 3 (dt<=0.8*max_dt)": + dt <= 0.16s (0.8 * this session's max_t_shift_s=0.2s), in
    place of the T_dom/2 cut

Run with:
    python3 axec1_windowcheck_vs_previous_dt_and_midpoint_traveltime_anisotropy.py
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
OLD_CSV = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer', 'splitting_results_AXEC1_2015_2021_all_batches.csv')
NEW_CSV = os.path.join(HERE, '..', 'mfast_maxdt_windowcheck_pipeline_transfer', 'splitting_results_AXEC1_2015_2021_combined_partial.csv')

OUT_PDF_VOXEL = os.path.join(HERE, 'dt_by_source_voxel_axec1_windowcheck_vs_previous_1row.pdf')
OUT_PDF_ANISO = os.path.join(HERE, 'traveltime_anisotropy_axec1_windowcheck_vs_previous_midpoint_1row.pdf')

STATION = 'AXEC1'
MAX_DT = 0.2                 # this session's production max_t_shift_s override
DT_ALT_MAX = 0.8 * MAX_DT     # 0.16s

COUNT_MIN_VOXEL = 10
COUNT_MIN_ANISO = 10

FILTERS = [
    dict(key='grade3',
         label='Grade 3: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg'),
    dict(key='grade3_dt016',
         label=f'Grade 3 (dt<={DT_ALT_MAX:.2f}s = 0.8*max_dt instead of T_dom/2): '
               f'SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<={DT_ALT_MAX:.2f}s, phi_err<=20deg'),
    dict(key='grade3_phierr10',
         label='Grade 3 (tightened phi_err<10deg): SNR>=2.0, quality>=0.75, dt_err<=0.05s, '
               'dt<=T_dom/2, phi_err<10deg'),
]


def ll2xy(lat, lon):
    return ((np.asarray(lon) - T.INI_LON) * T.KM_PER_DEG_LON,
            (np.asarray(lat) - T.INI_LAT) * T.KM_PER_DEG_LAT)


def load_df(csv_path, cutoff):
    df = pd.read_csv(csv_path)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality', 'snr_horizontal', 'dt_error', 'dominant_period',
                            'phi_error', 'latitude', 'longitude', 'depth'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    df = df[df['t'] <= cutoff].copy()
    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    return df.reset_index(drop=True)


def apply_filter(df, key):
    base = ((df['snr_horizontal'] >= 2.0) & (df['quality'] >= 0.75) &
            (df['dt_error'] <= 0.05) & (df['phi_error'] <= 20.0))
    if key == 'grade3':
        return df[base & (df['dt'] <= df['dominant_period'] / 2.0)].copy()
    elif key == 'grade3_dt016':
        return df[base & (df['dt'] <= DT_ALT_MAX)].copy()
    elif key == 'grade3_phierr10':
        base10 = ((df['snr_horizontal'] >= 2.0) & (df['quality'] >= 0.75) &
                  (df['dt_error'] <= 0.05) & (df['phi_error'] < 10.0))
        return df[base10 & (df['dt'] <= df['dominant_period'] / 2.0)].copy()
    raise ValueError(key)


def _sta_markers(ax, active_sta):
    for sta, row in T._sta_df.iterrows():
        if sta == active_sta:
            ax.plot(row['x'], row['y'], '^', ms=7, mfc='#FFD700', mec='k', mew=0.8, zorder=13)
        else:
            ax.plot(row['x'], row['y'], '^', ms=6, mfc='white', mec='k', mew=0.6, zorder=12)
        dx, dy = T.LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, T.STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


# ── Figure 1: dt by source (hypocenter) voxel, no ray tracing ──────────────────────────────

def voxelize_2d(sub, value_col, count_min):
    ix = np.floor((sub['x'].values - T.X_START) / T.VXY).astype(int)
    iy = np.floor((sub['y'].values - T.Y_START) / T.VXY).astype(int)
    valid = (ix >= 0) & (ix < T.NX) & (iy >= 0) & (iy < T.NY)
    ix, iy, v = ix[valid], iy[valid], sub[value_col].values[valid]
    flat = ix.astype(np.int64) * T.NY + iy.astype(np.int64)
    uniq, inv = np.unique(flat, return_inverse=True)
    sum_v = np.zeros(len(uniq)); np.add.at(sum_v, inv, v)
    cnt = np.bincount(inv, minlength=len(uniq))
    mean_v = sum_v / np.maximum(cnt, 1)
    gate = cnt >= count_min
    a2d = np.full((T.NX, T.NY), np.nan)
    fi, fj = uniq // T.NY, uniq % T.NY
    a2d[fi[gate], fj[gate]] = mean_v[gate]
    return a2d, int(valid.sum())


def make_side_by_side_page(field_old, n_old, field_new, n_new, cbar_label, title, fmt='%.3f'):
    fig = plt.figure(figsize=(3.2 * 2 + 1.0, 4.0))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.06, top=0.80, bottom=0.14)

    all_v = np.concatenate([
        field_old[np.isfinite(field_old)] if np.isfinite(field_old).any() else np.array([]),
        field_new[np.isfinite(field_new)] if np.isfinite(field_new).any() else np.array([]),
    ])
    vmax = float(np.nanmax(all_v)) if all_v.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., vmax, 15)

    last_h = None
    for ci, (field, n, lbl) in enumerate([(field_old, n_old, f'Previous\nN={n_old:,}'),
                                          (field_new, n_new, f'Window-check\nN={n_new:,}')]):
        ax = fig.add_subplot(gs[0, ci])
        T._bathy(ax)
        msk = np.isfinite(field)
        if msk.sum() > 0:
            sm = gaussian_filter(np.nan_to_num(field), sigma=1.0)
            mp = np.ma.masked_where(~msk, sm)
            if not mp.mask.all():
                ax.contourf(T.Xg, T.Yg, mp, levels=levels, cmap=cmap, vmin=0, vmax=vmax, extend='neither')
                last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
                last_h.set_array([])
        _sta_markers(ax, STATION)
        T._faults(ax, T.Z_MAX / 2.0)
        ax.set_xlim(T.X_START, T.X_END)
        ax.set_ylim(T.Y_START, T.Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(lbl, fontsize=9, fontweight='bold', pad=6)

    if last_h:
        cax = fig.add_subplot(gs[0, 2])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, vmax, 5), format=FormatStrFormatter(fmt))
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=7, label=STATION),
        mlines.Line2D([], [], marker='^', color='w', mfc='white', mec='k', ms=6, label='Other stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.42, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    return fig


def build_dt_source_voxel_pdf(old_all, new_all, cutoff):
    print('\n=== dt by source voxel (no ray tracing) ===')
    with PdfPages(OUT_PDF_VOXEL) as pdf:
        for filt in FILTERS:
            old_f = apply_filter(old_all, filt['key'])
            new_f = apply_filter(new_all, filt['key'])
            print(f"  {filt['key']}: previous N={len(old_f):,}, windowcheck N={len(new_f):,}")

            field_old, n_old = voxelize_2d(old_f, 'dt', COUNT_MIN_VOXEL)
            field_new, n_new = voxelize_2d(new_f, 'dt', COUNT_MIN_VOXEL)

            title = (f'AXEC1: delay time by earthquake source voxel (all depths combined)\n'
                     f'{filt["label"]}\nboth datasets restricted to events <= {cutoff} '
                     f'(count_min={COUNT_MIN_VOXEL})')
            fig = make_side_by_side_page(field_old, n_old, field_new, n_new,
                                          'Mean delay time per source voxel (s)', title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {OUT_PDF_VOXEL}')


# ── Figure 2: midpoint traveltime anisotropy (fresh ray tracing, both datasets) ─────────────

def trace_and_compute(df):
    """For each event in df, trace a ray from the source to STATION through the Baillard Vs
    model, compute (a) percent anisotropy 100*dt/T_travel and (b) the (ix, iy) voxel containing
    the ray's arc-length midpoint (cumulative seg_len_km reaches half the total path length),
    same convention as compute_ray_midpoint_voxels() in the 6-station midpoint scripts. Depth
    (iz) is discarded at the aggregation step, not here -- kept per-ray in case it's useful."""
    n = len(df)
    mid_ix = np.full(n, -1, dtype=np.int64)
    mid_iy = np.full(n, -1, dtype=np.int64)
    a_pct = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    n_failed = 0
    for i, row in enumerate(df.itertuples()):
        ex, ey, ez = float(row.x), float(row.y), float(row.z)
        if ez < 0 or ez > T.Z_MAX:
            n_failed += 1
            continue
        try:
            ray = T.tracer.trace(STATION, ex, ey, ez, n_pts=T.N_RAY)
            vcols, seg_len_km = T.tracer.ray_to_voxels(ray, T.xn, T.yn, T.zn)
        except RuntimeError:
            n_failed += 1
            continue
        if len(vcols) == 0:
            n_failed += 1
            continue
        ix = vcols // (T.NY * T.NZ)
        rem = vcols % (T.NY * T.NZ)
        iy = rem // T.NZ
        vs_seg = T.vs_flat[vcols]
        tt_seg = seg_len_km / np.maximum(vs_seg, 1e-9)
        t_travel = float(tt_seg.sum())
        if t_travel <= 0:
            n_failed += 1
            continue
        a_pct[i] = 100.0 * float(row.dt) / t_travel

        cum = np.cumsum(seg_len_km)
        target = cum[-1] / 2.0
        j = int(np.searchsorted(cum, target))
        j = min(j, len(cum) - 1)
        mid_ix[i] = ix[j]
        mid_iy[i] = iy[j]
        ok[i] = True
        if (i + 1) % 2000 == 0:
            print(f'    traced {i+1:,}/{n:,}', end='\r', flush=True)
    print(f'    traced {n:,}/{n:,} ({n_failed:,} failed/out-of-bounds)          ')
    return mid_ix, mid_iy, a_pct, ok


def aggregate_midpoint_2d(mid_ix, mid_iy, values, ok_mask, count_min):
    ix = mid_ix[ok_mask]
    iy = mid_iy[ok_mask]
    v = values[ok_mask]
    flat = ix.astype(np.int64) * T.NY + iy.astype(np.int64)
    uniq, inv = np.unique(flat, return_inverse=True)
    sum_v = np.zeros(len(uniq)); np.add.at(sum_v, inv, v)
    cnt = np.bincount(inv, minlength=len(uniq))
    mean_v = sum_v / np.maximum(cnt, 1)
    gate = cnt >= count_min
    a2d = np.full((T.NX, T.NY), np.nan)
    fi, fj = uniq // T.NY, uniq % T.NY
    a2d[fi[gate], fj[gate]] = mean_v[gate]
    return a2d


def build_midpoint_anisotropy_pdf(old_all, new_all, cutoff):
    print('\n=== Midpoint traveltime anisotropy (fresh ray tracing) ===')
    union_old = apply_filter(old_all, FILTERS[0]['key']).index
    union_new = apply_filter(new_all, FILTERS[0]['key']).index
    for filt in FILTERS[1:]:
        union_old = union_old.union(apply_filter(old_all, filt['key']).index)
        union_new = union_new.union(apply_filter(new_all, filt['key']).index)
    old_u = old_all.loc[union_old].reset_index(drop=True)
    new_u = new_all.loc[union_new].reset_index(drop=True)
    print(f'  Tracing union-of-filters events: previous={len(old_u):,}, windowcheck={len(new_u):,}')

    print('  Tracing previous dataset...')
    old_mid_ix, old_mid_iy, old_a, old_ok = trace_and_compute(old_u)
    print('  Tracing window-check dataset...')
    new_mid_ix, new_mid_iy, new_a, new_ok = trace_and_compute(new_u)

    with PdfPages(OUT_PDF_ANISO) as pdf:
        for filt in FILTERS:
            old_f_idx = apply_filter(old_u, filt['key']).index.values
            new_f_idx = apply_filter(new_u, filt['key']).index.values
            old_mask = old_ok.copy(); old_mask[:] = False; old_mask[old_f_idx] = old_ok[old_f_idx]
            new_mask = new_ok.copy(); new_mask[:] = False; new_mask[new_f_idx] = new_ok[new_f_idx]

            n_old = int(old_mask.sum())
            n_new = int(new_mask.sum())
            print(f"  {filt['key']}: previous traced-and-filtered N={n_old:,}, "
                  f"windowcheck traced-and-filtered N={n_new:,}")

            field_old = aggregate_midpoint_2d(old_mid_ix, old_mid_iy, old_a, old_mask, COUNT_MIN_ANISO)
            field_new = aggregate_midpoint_2d(new_mid_ix, new_mid_iy, new_a, new_mask, COUNT_MIN_ANISO)

            title = (f'AXEC1: percent anisotropy by ray-midpoint (all depths combined)\n'
                     f'{filt["label"]}\nboth datasets restricted to events <= {cutoff} '
                     f'(count_min={COUNT_MIN_ANISO})')
            fig = make_side_by_side_page(field_old, n_old, field_new, n_new,
                                          'Percent anisotropy (ray-midpoint mean)', title,
                                          fmt='%.1f')
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {OUT_PDF_ANISO}')


def main():
    print('Setting up ray-tracer/Vs model/bathymetry environment...')
    T._setup_environment(need_tracer=True)

    new_all = load_df(NEW_CSV, cutoff=pd.Timestamp('2100-01-01', tz='UTC'))
    cutoff = new_all['t'].max()
    print(f'Window-check partial AXEC1 data spans up to {cutoff} -- using this as the common cutoff.')

    old_all = load_df(OLD_CSV, cutoff=cutoff)
    new_all = new_all[new_all['t'] <= cutoff].reset_index(drop=True)
    old_all = old_all.reset_index(drop=True)
    print(f'Previous AXEC1 (<= cutoff): {len(old_all):,} rows; '
          f'Window-check AXEC1 (<= cutoff): {len(new_all):,} rows')

    build_dt_source_voxel_pdf(old_all, new_all, cutoff)
    build_midpoint_anisotropy_pdf(old_all, new_all, cutoff)


if __name__ == '__main__':
    main()
