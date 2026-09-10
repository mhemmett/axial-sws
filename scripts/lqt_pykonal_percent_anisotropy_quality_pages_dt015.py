#!/usr/bin/env python3
"""
lqt_pykonal_percent_anisotropy_quality_pages_dt015.py

Tighter-dt-threshold variant of lqt_pykonal_percent_anisotropy_quality_pages.py: identical
method, pages, and depth rows/quality thresholds, but with the dt cutoff tightened from
dt < 0.24 s to dt < 0.15 s, excluding all measurements above that delay time.
  - SAME travel-time-weighted % anisotropy method (A_ray = dt*100/T_travel,
    voxel value = travel-time-weighted mean over crossing rays).
  - SAME grayscale bathymetry, station triangles, fault outlines.
  - Kidiwela sources REMOVED.
  - Fixed QC gate on every page: dt_err < 0.04 s, phi_err < 20 deg, dt < 0.15 s.
  - One depth row at a time (see DEPTH_ROWS): 1.25-1.75 km and 1.25-2.0 km.
  - Colorbar kept on the side but shortened to match the single-row height.

Produces 18 pages (2 depth rows x 3 quality thresholds x 3 page sets) -- same structure as
the original, tighter-dt version only (the dt-unfiltered companion pdf from the original
script doesn't depend on this cutoff, so it isn't reproduced here).

Reuses the existing ray-tracing cache CSVs (read-only, never re-traces):
    lqt_pykonal_combined_results/raylength_anis_ray_summary.csv
    lqt_pykonal_combined_results/raylength_anis_ray_voxel_segments.csv

Run with:
    python3 lqt_pykonal_percent_anisotropy_quality_pages_dt015.py
"""

import glob
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
from PIL import Image as PILImage
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import matplotlib.lines as mlines
import math

from baillard_velocity import vs_at
from sws_forward_model import STATION_FILE, ll2xy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_lqt_pykonal_results')
CATALOG_DIR = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR = DATA_DIR

RAY_SUMMARY_CSV = os.path.join(DATA_DIR, 'raylength_anis_ray_summary.csv')
RAY_VOXEL_CSV = os.path.join(DATA_DIR, 'raylength_anis_ray_voxel_segments.csv')

# ── QC constants (per this figure's request) ─────────────────────────────────
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04
DT_CUTOFF = 0.15   # tightened from the original script's 0.24 s

QUALITY_THRESHOLDS = [0.7, 0.5, -0.5]

# Coverage gate (same as the sibling scripts).
COUNT_MIN = 8
MIN_PATH_KM = 0.5

# Higher-threshold variant of the percent-anisotropy ("travel time") pages:
# same method, same depth rows/quality thresholds, but >=20 contributing rays
# required per displayed voxel instead of >=8.
COUNT_MIN_HIGH = 20

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

# ── Voxel grid: isotropic 0.25 km voxels (identical to the sibling scripts) ──
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VXY = 0.25
VZ = 0.25
xn = np.arange(X_START, X_END + VXY * .5, VXY)
yn = np.arange(Y_START, Y_END + VXY * .5, VXY)
zn = np.arange(0., Z_MAX + VZ * .5, VZ)
NX, NY, NZ = len(xn), len(yn), len(zn)
J = NX * NY * NZ

# ── Depth rows, each rendered as its own set of quality pages ───────────────
DEPTH_ROWS = [
    ((5, 7), '1.25–1.75 km', 1.5),      # 2 voxels of the 0.25 km grid
    ((5, 8), '1.25–2.0 km', 1.625),     # 3 voxels of the 0.25 km grid
]

Xg, Yg = np.meshgrid(xn + VXY / 2., yn + VXY / 2., indexing='ij')

vs_voxels = None
vs_flat = None
CATALOG = None
sta_xy = None
_sta_df = None
_gray = None
_ext = None


STATION_FILES = {
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv',
              'splitting_results_AXAS2_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
}


def load_station(sta):
    """Load a station's baseline (success==True, dt>0) measurements and join
    each to its MLdd event location -- IDENTICAL row set/order to the sibling
    scripts' load_station(), so that concatenating in STATIONS order reproduces
    the exact ray_id positional alignment used when the cache was built."""
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = glob.glob(os.path.join(
            AXEC2_2015_2021_DIR, 'splitting_results_mldd_2015_2021_axec2_batch_*.csv'))
        dfs.extend(pd.read_csv(f) for f in batch_files)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df['t'] = pd.to_datetime(df['datetime'], utc=True)

    sta_cat = CATALOG[CATALOG['station'] == sta].drop_duplicates(subset='event_datetime')
    df = df.merge(sta_cat[['event_datetime', 'event_lat', 'event_lon', 'event_depth']],
                  left_on='t', right_on='event_datetime', how='left')
    df = df.dropna(subset=['event_lat', 'event_lon', 'event_depth'])

    df['x'], df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
    df['z'] = df['event_depth'].values
    return df[['t', 'x', 'y', 'z']]


def build_7_periods(all_df):
    post = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post)
    bounds = [ERUPTION_END]
    for i in range(1, 5):
        bounds.append(post['t'].iloc[min(int(round(i * n / 5)), n - 1)])
    bounds.append(None)

    def fmt(ts):
        return ts.strftime('%b %Y') if ts else 'present'

    pds = [('Pre-eruption', None, ERUPTION_START), ('Syn-eruption', ERUPTION_START, ERUPTION_END)]
    for i in range(5):
        pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i + 1])}', bounds[i], bounds[i + 1]))
    return pds


def _setup_environment():
    global vs_voxels, vs_flat, CATALOG, sta_xy, _sta_df, _gray, _ext

    print('Querying Baillard Vs at each voxel centre...')
    X3, Y3, Z3 = np.meshgrid(xn + VXY / 2., yn + VXY / 2., zn + VZ / 2., indexing='ij')
    vs_voxels = vs_at(X3.ravel(), Y3.ravel(), Z3.ravel()).reshape(NX, NY, NZ)
    vs_voxels = np.clip(vs_voxels, 0.3, 5.0)
    vs_flat = vs_voxels.ravel()
    print(f'  Vs range: {vs_voxels.min():.3f}–{vs_voxels.max():.3f} km/s')

    print('Loading MLdd catalogs for event locations...')
    _cat1 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'))
    _cat2 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'))
    _cat1['event_datetime'] = pd.to_datetime(_cat1['event_datetime'], utc=True, format='mixed')
    _cat2['event_datetime'] = pd.to_datetime(_cat2['event_datetime'], utc=True, format='mixed')
    CATALOG = pd.concat([_cat1, _cat2], ignore_index=True)

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                          engine='python').set_index('s')
    _sta_df = _sta_df.loc[[s for s in STATIONS if s in _sta_df.index]]
    _sta_df['x'], _sta_df['y'] = ((np.asarray(_sta_df['lon']) - INI_LON) * KM_PER_DEG_LON,
                                  (np.asarray(_sta_df['lat']) - INI_LAT) * KM_PER_DEG_LAT)
    sta_xy = {s: (float(_sta_df.loc[s, 'x']), float(_sta_df.loc[s, 'y']))
              for s in STATIONS if s in _sta_df.index}

    bathy = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
             'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
    PILImage.MAX_IMAGE_PIXELS = None
    _p = PILImage.open(bathy)
    _tag = _p.tag_v2
    _olon, _olat = _tag[33922][3], _tag[33922][4]
    _pl, _pb = _tag[33550][0], _tag[33550][1]
    _nc, _nr = _p.size
    _p.close()
    _c0 = max(0, int(((INI_LON + X_START / KM_PER_DEG_LON) - _olon) / _pl) - 2)
    _c1 = min(_nc, int(((INI_LON + X_END / KM_PER_DEG_LON) - _olon) / _pl) + 2)
    _r0 = max(0, int((_olat - (INI_LAT + Y_END / KM_PER_DEG_LAT)) / _pb) - 2)
    _r1 = min(_nr, int((_olat - (INI_LAT + Y_START / KM_PER_DEG_LAT)) / _pb) + 2)
    _rgb = tifffile.imread(bathy)[_r0:_r1, _c0:_c1]
    _ds = max(1, max(_rgb.shape[:2]) // 1024)
    _rgb = _rgb[::_ds, ::_ds]
    _gray = np.dot(_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
    _ext = [(_olon + _c0 * _pl - INI_LON) * KM_PER_DEG_LON,
            (_olon + _c1 * _pl - INI_LON) * KM_PER_DEG_LON,
            (_olat - _r1 * _pb - INI_LAT) * KM_PER_DEG_LAT,
            (_olat - _r0 * _pb - INI_LAT) * KM_PER_DEG_LAT]


def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)


def _sta(ax):
    for sta, row in _sta_df.iterrows():
        ax.plot(row['x'], row['y'], '^', ms=6, mfc='#FFD700', mec='k', mew=0.7, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


COT66 = math.cos(math.radians(66)) / math.sin(math.radians(66))
DEPTH_TOP = 0.1
strike_rad = math.radians(330.)
L = 8.
s_hat = np.array([math.sin(strike_rad), math.cos(strike_rad)])


def _faults(ax, z0):
    for cx, cy, dip_az in [(8.5, 4.0, 60.), (7.0, 4.0, 240.)]:
        horiz = (z0 - DEPTH_TOP) * COT66
        da = math.radians(dip_az)
        cx_z = cx + horiz * math.sin(da)
        cy_z = cy + horiz * math.cos(da)
        x1, y1 = cx_z - L / 2 * s_hat[0], cy_z - L / 2 * s_hat[1]
        x2, y2 = cx_z + L / 2 * s_hat[0], cy_z + L / 2 * s_hat[1]
        ax.plot([x1, x2], [y1, y2], '-', color='#cc0000', lw=0.9, alpha=0.75, zorder=11)


def load_cache():
    if not (os.path.exists(RAY_SUMMARY_CSV) and os.path.exists(RAY_VOXEL_CSV)):
        raise FileNotFoundError(
            f'Ray-coverage cache CSVs not found in {DATA_DIR}. This script reuses '
            f'the existing cache and never re-traces -- run one of the sibling '
            f'lqt_pykonal_tomography_*_anisotropy.py scripts first if it is missing.')
    print('Loading cached rays...')
    summary = pd.read_csv(RAY_SUMMARY_CSV)
    summary = summary.sort_values('ray_id').reset_index(drop=True)
    vox = pd.read_csv(RAY_VOXEL_CSV, dtype={
        'ray_id': np.int32, 'ix': np.int16, 'iy': np.int16, 'iz': np.int16,
        'seg_len_km': np.float32, 'A_voxel_pct': np.float32})
    print(f'  ray_summary: {len(summary):,} rows  '
          f'ray_voxel_segments: {len(vox):,} rows')
    return summary, vox


def compute_ttime_anisotropy(summary, vox, vray):
    N = len(summary)
    ix = vox['ix'].values.astype(np.int64)
    iy = vox['iy'].values.astype(np.int64)
    iz = vox['iz'].values.astype(np.int64)
    flat = ix * (NY * NZ) + iy * NZ + iz
    seg = vox['seg_len_km'].values.astype(np.float64)
    vs_seg = vs_flat[flat]
    tt_seg = seg / np.maximum(vs_seg, 1e-9)

    T_travel = np.zeros(N, dtype=np.float64)
    np.add.at(T_travel, vray, tt_seg)

    dt = summary['dt'].values.astype(np.float64)
    a_ray_tt = dt * 100.0 / np.maximum(T_travel, 1e-9)
    a_ray_tt[T_travel <= 0] = np.nan
    return tt_seg, a_ray_tt


def _to_ns(ts):
    return None if ts is None else pd.Timestamp(ts).value


def assign_periods(event_ns, periods):
    idx = np.full(len(event_ns), -1, dtype=np.int64)
    for pi, (_lbl, t0, t1) in enumerate(periods):
        lo, hi = _to_ns(t0), _to_ns(t1)
        m = np.ones(len(event_ns), dtype=bool)
        if lo is not None:
            m &= event_ns >= lo
        if hi is not None:
            m &= event_ns < hi
        idx[m & (idx < 0)] = pi
    return idx


def aggregate_page(vox, vray, ray_survive, ray_period, periods, tt_seg, a_ray_tt, iz0, iz1,
                    count_min=COUNT_MIN):
    """Same coverage-gated travel-time-weighted aggregation as the sibling
    scripts, but restricted to the single depth slab [iz0, iz1)."""
    depth_ok = (vox['iz'].values >= iz0) & (vox['iz'].values < iz1)
    keep = ray_survive[vray] & (ray_period[vray] >= 0) & depth_ok
    results = []
    if not keep.any():
        for pi, (lbl, _t0, _t1) in enumerate(periods):
            n = int((ray_survive & (ray_period == pi)).sum())
            results.append(dict(label=lbl, n=n, a=np.full((NX, NY), np.nan)))
        return results

    ix = vox['ix'].values[keep].astype(np.int64)
    iy = vox['iy'].values[keep].astype(np.int64)
    flat = ix * NY + iy
    period = ray_period[vray][keep]

    w = tt_seg[keep]
    a = a_ray_tt[vray][keep].astype(np.float64)
    seg_vals = vox['seg_len_km'].values[keep].astype(np.float64)

    JXY = NX * NY
    key = period * JXY + flat
    uniq, inv = np.unique(key, return_inverse=True)
    sum_wa = np.zeros(len(uniq))
    np.add.at(sum_wa, inv, a * w)
    sum_w = np.zeros(len(uniq))
    np.add.at(sum_w, inv, w)
    sum_p = np.zeros(len(uniq))
    np.add.at(sum_p, inv, seg_vals)
    cnt = np.bincount(inv, minlength=len(uniq))
    wmean_a = sum_wa / np.maximum(sum_w, 1e-9)
    gate = (cnt >= count_min) & (sum_p >= MIN_PATH_KM)

    up = uniq // JXY
    uf = uniq % JXY
    uix = uf // NY
    uiy = uf % NY

    for pi, (lbl, _t0, _t1) in enumerate(periods):
        a2d = np.full((NX, NY), np.nan)
        sel = (up == pi) & gate
        a2d[uix[sel], uiy[sel]] = wmean_a[sel]
        n = int((ray_survive & (ray_period == pi)).sum())
        results.append(dict(label=lbl, n=n, a=a2d))
    return results


def aggregate_source_dt_page(ray_ix, ray_iy, ray_iz, dt_vals, ray_survive, ray_period,
                              periods, iz0, iz1, count_min=COUNT_MIN):
    """Per-period (NX,NY) grid of the MEAN delay time dt, plotted ONLY at each
    event's own source voxel (ix,iy) -- not apportioned/averaged across every
    voxel the ray crosses, unlike aggregate_page(). Same coverage-count gate
    (>=COUNT_MIN contributing events per displayed voxel); no path-length gate
    since there is no ray path here."""
    depth_ok = (ray_iz >= iz0) & (ray_iz < iz1) & (ray_ix >= 0) & (ray_iy >= 0)
    keep = ray_survive & (ray_period >= 0) & depth_ok
    results = []
    if not keep.any():
        for pi, (lbl, _t0, _t1) in enumerate(periods):
            n = int((ray_survive & (ray_period == pi)).sum())
            results.append(dict(label=lbl, n=n, a=np.full((NX, NY), np.nan)))
        return results

    ix = ray_ix[keep]
    iy = ray_iy[keep]
    period = ray_period[keep]
    dtv = dt_vals[keep]

    JXY = NX * NY
    flat = ix * NY + iy
    key = period * JXY + flat
    uniq, inv = np.unique(key, return_inverse=True)
    sum_dt = np.zeros(len(uniq))
    np.add.at(sum_dt, inv, dtv)
    cnt = np.bincount(inv, minlength=len(uniq))
    mean_dt = sum_dt / np.maximum(cnt, 1)
    gate = cnt >= count_min

    up = uniq // JXY
    uf = uniq % JXY
    uix = uf // NY
    uiy = uf % NY

    for pi, (lbl, _t0, _t1) in enumerate(periods):
        a2d = np.full((NX, NY), np.nan)
        sel = (up == pi) & gate
        a2d[uix[sel], uiy[sel]] = mean_dt[sel]
        n = int((ray_survive & (ray_period == pi)).sum())
        results.append(dict(label=lbl, n=n, a=a2d))
    return results


def make_page(results, title, depth_label, z0, count_min=COUNT_MIN,
              cbar_label='Travel-time-weighted mean % shear anisotropy per voxel',
              tick_fmt='%.1f'):
    """One row x len(results) columns; colorbar height matches the single row."""
    all_a = [r['a'][r['a'] > 0].ravel() for r in results if np.isfinite(r['a']).any()]
    a_vmax = float(np.nanpercentile(np.concatenate(all_a), 99)) if all_a else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.
    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., a_vmax, 15)
    n_per = len(results)
    panel_w = 1.8 if n_per > 7 else 2.2
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, 4.0))
    gs = GridSpec(1, n_per + 1, width_ratios=[1] * n_per + [0.04], wspace=0.04)
    last_h = None
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci])
        _bathy(ax)
        sl = res['a']
        msk = np.isfinite(sl)
        if msk.sum() >= count_min:
            sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
            mp = np.ma.masked_where(~msk, sm)
            if not mp.mask.all():
                ax.contourf(Xg, Yg, mp, levels=levels, cmap=cmap,
                            vmin=0, vmax=a_vmax, extend='neither')
                last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, a_vmax))
                last_h.set_array([])
        _sta(ax)
        _faults(ax, z0)
        ax.set_xlim(X_START, X_END)
        ax.set_ylim(Y_START, Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f'{res["label"]}\nN={res["n"]:,}', fontsize=7.5, fontweight='bold', pad=1)
        if ci == 0:
            ax.set_ylabel(depth_label, fontsize=7)
    if last_h:
        cax = fig.add_subplot(gs[0, n_per])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, a_vmax, 5),
                          format=FormatStrFormatter(tick_fmt))
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=7)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='OBS station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    fig.subplots_adjust(top=0.78, bottom=0.13)
    return fig


def write_all_pages(out_path, base_mask, ray_ix, ray_iy, ray_iz, dt, q, ray_period, periods,
                     vox, vray, tt_seg, a_ray_tt, cbar_label, dt_cbar_label, title_suffix=''):
    """Write the full 18-page set (percent anisotropy COUNT_MIN=8, delay-time-
    at-source, percent anisotropy COUNT_MIN_HIGH) for the given base_mask (the
    QC gate common to every page, before the per-page quality threshold)."""
    print(f'\nWriting {os.path.basename(out_path)}...')
    with PdfPages(out_path) as pdf:
        print('\n--- Percent shear anisotropy (travel-time-weighted, COUNT_MIN=8) ---')
        for (iz0, iz1), depth_label, z0 in DEPTH_ROWS:
            print(f'\n=== Depth row {depth_label} ===')
            for q_thresh in QUALITY_THRESHOLDS:
                ray_mask = base_mask & (q >= q_thresh)
                results = aggregate_page(vox, vray, ray_mask, ray_period, periods,
                                         tt_seg, a_ray_tt, iz0, iz1)
                n_total = int(ray_mask.sum())
                covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
                print(f'  Quality >= {q_thresh:+.1f}: {n_total:,} rays  covered-voxels={covered:,}')
                title = f'Percent Shear-wave Anisotropy, Quality >= {q_thresh:g}{title_suffix}'
                fig = make_page(results, title, depth_label, z0, cbar_label=cbar_label)
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)

        print('\n--- Delay times at earthquake source locations ---')
        for (iz0, iz1), depth_label, z0 in DEPTH_ROWS:
            print(f'\n=== Depth row {depth_label} ===')
            for q_thresh in QUALITY_THRESHOLDS:
                ray_mask = base_mask & (q >= q_thresh)
                results = aggregate_source_dt_page(ray_ix, ray_iy, ray_iz, dt, ray_mask,
                                                    ray_period, periods, iz0, iz1)
                n_total = int(ray_mask.sum())
                covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
                print(f'  Quality >= {q_thresh:+.1f}: {n_total:,} events  covered-voxels={covered:,}')
                title = (f'Delay Times at Earthquake Source Locations, '
                         f'Quality >= {q_thresh:g}{title_suffix}')
                fig = make_page(results, title, depth_label, z0, cbar_label=dt_cbar_label,
                                tick_fmt='%.2f')
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)

        print(f'\n--- Percent shear anisotropy (travel-time-weighted, COUNT_MIN={COUNT_MIN_HIGH}) ---')
        for (iz0, iz1), depth_label, z0 in DEPTH_ROWS:
            print(f'\n=== Depth row {depth_label} ===')
            for q_thresh in QUALITY_THRESHOLDS:
                ray_mask = base_mask & (q >= q_thresh)
                results = aggregate_page(vox, vray, ray_mask, ray_period, periods,
                                         tt_seg, a_ray_tt, iz0, iz1, count_min=COUNT_MIN_HIGH)
                n_total = int(ray_mask.sum())
                covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
                print(f'  Quality >= {q_thresh:+.1f}: {n_total:,} rays  covered-voxels={covered:,}')
                title = f'Percent Shear-wave Anisotropy, Quality >= {q_thresh:g}{title_suffix}'
                fig = make_page(results, title, depth_label, z0, cbar_label=cbar_label)
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
    print(f'Saved {out_path}')


def main():
    _setup_environment()

    print('\nLoading baseline events (success==True, dt>0) with source locations '
          '(for period bin edges and the delay-time-at-source pages)...')
    dfs = [load_station(sta) for sta in STATIONS]
    all_df = pd.concat(dfs, ignore_index=True)
    print(f'  TOTAL baseline: {len(all_df):,}')

    summary, vox = load_cache()

    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    assert len(all_df) == N, (
        f'baseline event count ({len(all_df):,}) does not match the ray cache '
        f'({N:,}) -- load_station() must reproduce the exact row set/order used '
        f'to build the cache for ray_id positional alignment to hold')
    vray = vox['ray_id'].values

    # Per-ray source voxel indices (event location), positionally aligned to
    # ray_id via the STATIONS-ordered concatenation above (same order used to
    # assign ray_id when the cache was built).
    ray_ix = np.floor((all_df['x'].values - X_START) / VXY).astype(np.int64)
    ray_iy = np.floor((all_df['y'].values - Y_START) / VXY).astype(np.int64)
    ray_iz = np.floor((all_df['z'].values - 0.) / VZ).astype(np.int64)
    _oob = ((ray_ix < 0) | (ray_ix >= NX) | (ray_iy < 0) | (ray_iy >= NY) |
            (ray_iz < 0) | (ray_iz >= NZ))
    ray_ix[_oob] = -1
    ray_iy[_oob] = -1
    ray_iz[_oob] = -1

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8

    print('\nComputing T_travel per ray and A_ray = dt·100/T_travel from cache...')
    tt_seg, a_ray_tt = compute_ttime_anisotropy(summary, vox, vray)

    periods = build_7_periods(all_df)
    ray_period = assign_periods(event_ns, periods)

    ok = summary['trace_ok'].values.astype(bool)
    dt = summary['dt'].values
    phe = summary['phi_error'].values
    dte = summary['dt_error'].values
    q = summary['quality'].values
    base_mask_filtered = ok & (dte < DT_ERR_MAX) & (phe < PHI_ERR_MAX) & (dt < DT_CUTOFF)

    cbar_label = 'Travel-time-weighted mean % shear anisotropy per voxel'
    dt_cbar_label = 'Mean delay time at earthquake source voxel (s)'

    out_path = os.path.join(OUT_DIR, 'lqt_pykonal_percent_anisotropy_quality_pages_dt015.pdf')
    write_all_pages(out_path, base_mask_filtered, ray_ix, ray_iy, ray_iz, dt, q,
                    ray_period, periods, vox, vray, tt_seg, a_ray_tt,
                    cbar_label, dt_cbar_label, title_suffix=', dt < 0.15 s')

    print('\nDone.')


if __name__ == '__main__':
    main()
