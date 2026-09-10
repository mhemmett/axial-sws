#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_5stations_newdata.py

Extends traveltime_anisotropy_7period_axcc1_axec1_axec2_newdata.py (kept as-is, not
overwritten) with the two stations whose new-mfast max_dt=0.2s reruns completed 2026-07-23:
AXAS1 and AXEC3. 5-station 7-period travel-time-weighted percent-anisotropy tomography:

  AXCC1: mfast_maxdt_pipeline_transfer/splitting_results_AXCC1_{2015_2021,2022_2026}_all_batches.csv
  AXEC1: mfast_maxdt_pipeline_transfer/splitting_results_AXEC1_{2015_2021,2022_2026}_all_batches.csv
  AXEC2: mfast_maxdt_pipeline_transfer/splitting_results_AXEC2_2022_2026_all_batches.csv
         + our own COMPLETED maxdt02 re-run's 2015-2021 batches (all 497 batches done --
           production_axec2_mfast_filters_maxdt02_lqt_pykonal_results/*.csv), joined against
           raw_axec2_all_batches_mfast_filters_data/raw_axec2_all_batches_mfast_filters_metadata.csv
           for event lat/lon/depth (that pipeline stage doesn't carry location directly).
  AXAS1: mfast_maxdt_pipeline_transfer/splitting_results_AXAS1_{2015_2021,2022_2026}_all_batches.csv
  AXEC3: mfast_maxdt_pipeline_transfer/splitting_results_AXEC3_{2015_2021,2022_2026}_all_batches.csv

AXAS2 is EXCLUDED from the anisotropy computation entirely (no new-data rerun exists for it
yet) but its station marker is still drawn for spatial reference, in WHITE, vs. YELLOW for the
5 included stations -- same marker shape/size/edge as
lqt_pykonal_tomography_traveltime_anisotropy.py's station markers, just split by inclusion.

Method: identical travel-time-weighted formula to the sibling script --
    T_travel = sum_k seg_len_km[k] / Vs_voxel[k]   (per ray, k = crossed voxels)
    A_ray    = dt * 100 / T_travel                 (percent)
    V_voxel  = (sum_rays A_ray * tt_k) / (sum_rays tt_k),  tt_k = seg_len_km/Vs_voxel
Same 0.25 km voxel grid (X:4-12, Y:0-12 km, Z:0-4 km), same 4 depth rows, same bathymetry/
fault/Kidiwela overlays, same coverage gate (>=COUNT_MIN rays, >=MIN_PATH_KM cumulative
path per voxel).

QC (single tier, not the sibling's 6-page scheme -- one filtered view, matching the
convention already established for this new data in rose_7period_axcc1_axec1_axec2_newdata.py
/ histograms_all_columns_newdata_combined.py): success, dt>0, quality>=0.5, dt<T_dom/2
(cycle-skip-risk cut, per-event dominant period), phi_error<20 deg, dt_error<0.05s. Filtering
is applied BEFORE ray tracing (not after, unlike the sibling's cache-everything-then-filter
design) since this is a one-off single-tier run, not a multi-tier exploration -- traces only
the ~62k survivors instead of ~397k baseline events.

7 eruption-relative periods (verbatim scheme from the sibling / rose_plots_temporal.py).

Produces: traveltime_anisotropy_7period_5stations_newdata.pdf (1 page)

Run with:
    python3 traveltime_anisotropy_7period_5stations_newdata.py
"""

import glob
import math
import os
import time
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

from pykonal_raytracer import BaillardRayTracer
from baillard_velocity import vs_at
from sws_forward_model import STATION_FILE, ll2xy

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
AXEC2_META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                              'raw_axec2_all_batches_mfast_filters_metadata.csv')
OUT_PDF = os.path.join(HERE, 'traveltime_anisotropy_7period_5stations_newdata.pdf')

# ── QC (single tier) ──────────────────────────────────────────────────────────
QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05

COUNT_MIN = 8
MIN_PATH_KM = 0.5
N_RAY = 200

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

ALL_STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
INCLUDED_STATIONS = ['AXAS1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']   # new-data stations -> traced + yellow markers
EXCLUDED_STATIONS = ['AXAS2']   # reference-only -> white marker, not traced (no new-data rerun yet)

STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}
KIDIWELA = [dict(x=7.57, y=4.55, label='S1'), dict(x=7.53, y=6.60, label='S2')]

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

# ── Voxel grid (identical to the sibling) ────────────────────────────────────
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VXY = 0.25
VZ = 0.25
xn = np.arange(X_START, X_END + VXY * .5, VXY)
yn = np.arange(Y_START, Y_END + VXY * .5, VXY)
zn = np.arange(0., Z_MAX + VZ * .5, VZ)
NX, NY, NZ = len(xn), len(yn), len(zn)

DEPTH_ROWS = [
    ((0, 3), '0.0–0.75 km', 0.375),
    ((3, 6), '0.75–1.5 km', 1.125),
    ((6, 9), '1.5–2.25 km', 1.875),
    ((9, 12), '2.25–3.0 km', 2.625),
]

Xg, Yg = np.meshgrid(xn + VXY / 2., yn + VXY / 2., indexing='ij')

STATION_FILES = {
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
}


def load_station(sta):
    """Load new-data baseline (success, dt>0), with lat/lon/depth + dominant_period,
    then apply the single QC tier (quality/dt<T_dom/2/phi_error/dt_error)."""
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = sorted(glob.glob(os.path.join(
            AXEC2_2015_2021_DIR,
            'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_*.csv')))
        batch_dfs = [pd.read_csv(f) for f in batch_files]
        batch_dfs = [d for d in batch_dfs if len(d) > 0]
        meta = pd.read_csv(AXEC2_META_CSV)[['event_id', 'latitude', 'longitude', 'depth']]
        for d in batch_dfs:
            d['dominant_period'] = d['chosen_filter_dom_period_samples'] / 200.0
            d = d.merge(meta, on='event_id', how='left')
            dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'quality', 'dominant_period',
                            'phi_error', 'dt_error'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')

    df = df[(df['quality'] >= QW_MIN) &
            (df['dt'] < df['dominant_period'] / 2.0) &
            (df['phi_error'] < PHI_ERR_MAX) &
            (df['dt_error'] < DT_ERR_MAX)]

    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    return df[['t', 'dt', 'x', 'y', 'z']]


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
    global vs_voxels, vs_flat, sta_xy, _sta_df, _gray, _ext, tracer

    print('Querying Baillard Vs at each voxel centre...')
    X3, Y3, Z3 = np.meshgrid(xn + VXY / 2., yn + VXY / 2., zn + VZ / 2., indexing='ij')
    vs_voxels = vs_at(X3.ravel(), Y3.ravel(), Z3.ravel()).reshape(NX, NY, NZ)
    vs_voxels = np.clip(vs_voxels, 0.3, 5.0)
    vs_flat = vs_voxels.ravel()
    print(f'  Vs range: {vs_voxels.min():.3f}–{vs_voxels.max():.3f} km/s')

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                          engine='python').set_index('s')
    _sta_df = _sta_df.loc[[s for s in ALL_STATIONS if s in _sta_df.index]]
    _sta_df['x'], _sta_df['y'] = ((np.asarray(_sta_df['lon']) - INI_LON) * KM_PER_DEG_LON,
                                  (np.asarray(_sta_df['lat']) - INI_LAT) * KM_PER_DEG_LAT)
    sta_xy = {s: (float(_sta_df.loc[s, 'x']), float(_sta_df.loc[s, 'y']))
              for s in ALL_STATIONS if s in _sta_df.index}

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

    tracer = BaillardRayTracer(stride=5)
    for sta in INCLUDED_STATIONS:
        tracer.precompute_station(sta, *sta_xy[sta])


def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)


def _sta(ax):
    """Station markers: INCLUDED (new-data) stations in yellow, EXCLUDED in white."""
    for sta, row in _sta_df.iterrows():
        color = '#FFD700' if sta in INCLUDED_STATIONS else '#FFFFFF'
        ax.plot(row['x'], row['y'], '^', ms=6, mfc=color, mec='k', mew=0.7, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


def _kid(ax):
    for s in KIDIWELA:
        ax.plot(s['x'], s['y'], 'o', ms=4, mfc='red', mec='k', mew=0.5, zorder=14)
        ax.text(s['x'] + 0.12, s['y'] + 0.12, s['label'], fontsize=5, color='red',
                fontweight='bold', zorder=15)


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


def trace_all(dfs):
    """Trace every QC-surviving ray for the 3 included stations. Returns per-ray
    arrays: station, event_ns, dt, T_travel, A_ray (dt*100/T_travel), plus per-
    crossing arrays: ray_id, flat voxel index, seg_len_km, tt_k."""
    total = sum(len(dfs[s]) for s in INCLUDED_STATIONS)
    print(f'\nTracing {total:,} QC-surviving rays (n_pts={N_RAY})...')
    t0 = time.time()

    ray_station, ray_event_ns, ray_dt, ray_T, ray_A = [], [], [], [], []
    vox_ray_id, vox_flat, vox_seg, vox_tt = [], [], [], []

    ray_id = 0
    done = 0
    for sta in INCLUDED_STATIONS:
        df = dfs[sta]
        for row in df.itertuples():
            ex, ey, ez = float(row.x), float(row.y), float(row.z)
            if not (ez < 0 or ez > Z_MAX):
                try:
                    ray = tracer.trace(sta, ex, ey, ez, n_pts=N_RAY)
                    vcols, seg_len_km = tracer.ray_to_voxels(ray, xn, yn, zn)
                    if len(vcols) > 0:
                        vs_seg = vs_flat[vcols]
                        tt_seg = seg_len_km / np.maximum(vs_seg, 1e-9)
                        T_travel = float(tt_seg.sum())
                        if T_travel > 0:
                            a_ray = float(row.dt) * 100.0 / T_travel
                            ray_station.append(sta)
                            ray_event_ns.append(row.t.value)
                            ray_dt.append(float(row.dt))
                            ray_T.append(T_travel)
                            ray_A.append(a_ray)
                            vox_ray_id.extend([ray_id] * len(vcols))
                            vox_flat.extend(vcols.tolist())
                            vox_seg.extend(seg_len_km.tolist())
                            vox_tt.extend(tt_seg.tolist())
                            ray_id += 1
                except RuntimeError:
                    pass
            done += 1
            if done % 5000 == 0:
                print(f'  {done:,}/{total:,}  {time.time()-t0:.0f}s', end='\r', flush=True)

    print(f'\nTraced {ray_id:,}/{total:,} rays successfully in {time.time()-t0:.0f}s')
    return (np.array(ray_event_ns, dtype=np.int64), np.array(ray_A, dtype=np.float64),
            np.array(vox_ray_id, dtype=np.int64), np.array(vox_flat, dtype=np.int64),
            np.array(vox_seg, dtype=np.float64), np.array(vox_tt, dtype=np.float64))


def assign_periods(event_ns, periods):
    idx = np.full(len(event_ns), -1, dtype=np.int64)
    for pi, (_lbl, t0, t1) in enumerate(periods):
        lo = None if t0 is None else pd.Timestamp(t0).value
        hi = None if t1 is None else pd.Timestamp(t1).value
        m = np.ones(len(event_ns), dtype=bool)
        if lo is not None:
            m &= event_ns >= lo
        if hi is not None:
            m &= event_ns < hi
        idx[m & (idx < 0)] = pi
    return idx


def aggregate(ray_period, ray_A, vox_ray_id, vox_flat, vox_seg, vox_tt, periods):
    J = NX * NY * NZ
    period_per_crossing = ray_period[vox_ray_id]
    keep = period_per_crossing >= 0
    key = period_per_crossing[keep] * J + vox_flat[keep]
    a = ray_A[vox_ray_id[keep]]
    w = vox_tt[keep]
    seg = vox_seg[keep]

    uniq, inv = np.unique(key, return_inverse=True)
    sum_wa = np.zeros(len(uniq)); np.add.at(sum_wa, inv, a * w)
    sum_w = np.zeros(len(uniq)); np.add.at(sum_w, inv, w)
    sum_p = np.zeros(len(uniq)); np.add.at(sum_p, inv, seg)
    cnt = np.bincount(inv, minlength=len(uniq))
    wmean = sum_wa / np.maximum(sum_w, 1e-9)
    gate = (cnt >= COUNT_MIN) & (sum_p >= MIN_PATH_KM)

    up = uniq // J
    uf = uniq % J
    uix = uf // (NY * NZ)
    ur = uf % (NY * NZ)
    uiy = ur // NZ
    uiz = ur % NZ

    results = []
    for pi, (lbl, _t0, _t1) in enumerate(periods):
        a3d = np.full((NX, NY, NZ), np.nan)
        sel = (up == pi) & gate
        a3d[uix[sel], uiy[sel], uiz[sel]] = wmean[sel]
        n = int((ray_period == pi).sum())
        results.append(dict(label=lbl, n=n, a=a3d))
    return results


def make_page(results, title):
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
            _bathy(ax)
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            msk = np.isfinite(sl)
            if msk.sum() >= COUNT_MIN:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                mp = np.ma.masked_where(~msk, sm)
                if not mp.mask.all():
                    ax.contourf(Xg, Yg, mp, levels=levels, cmap=cmap,
                                vmin=0, vmax=a_vmax, extend='neither')
                    last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, a_vmax))
                    last_h.set_array([])
            _sta(ax)
            _kid(ax)
            _faults(ax, z0)
            ax.set_xlim(X_START, X_END)
            ax.set_ylim(Y_START, Y_END)
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
                      label='Included station (new data)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFFFFF', mec='k', ms=6,
                      label='Excluded station (no new data)'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.08)
    return fig


def main():
    _setup_environment()

    print('\nLoading new-data QC-filtered events (quality>=0.5, dt<T_dom/2, '
          f'phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s)...')
    dfs = {}
    for sta in INCLUDED_STATIONS:
        dfs[sta] = load_station(sta)
        print(f'  {sta}: {len(dfs[sta]):,}')
    all_df = pd.concat(dfs.values(), ignore_index=True)
    print(f'  TOTAL: {len(all_df):,}')

    periods = build_7_periods(all_df)

    (event_ns, ray_A, vox_ray_id, vox_flat, vox_seg, vox_tt) = trace_all(dfs)
    ray_period = assign_periods(event_ns, periods)

    results = aggregate(ray_period, ray_A, vox_ray_id, vox_flat, vox_seg, vox_tt, periods)
    for r in results:
        covered = int(np.isfinite(r['a']).sum())
        print(f'  {r["label"].splitlines()[0]:20s}: N={r["n"]:,}  covered-voxels={covered:,}')

    title = (f'AXAS1/AXCC1/AXEC1/AXEC2/AXEC3 (new mfast max_dt=0.2s data) — travel-time-weighted mean % '
             f'shear anisotropy per voxel (7-period)\n'
             f'A = δt·100/T_travel; quality>={QW_MIN}, dt<T_dom/2, phi_err<{PHI_ERR_MAX:.0f}deg, '
             f'dt_err<{DT_ERR_MAX}s (n={len(ray_A):,})')
    fig = make_page(results, title)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
