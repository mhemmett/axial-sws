#!/usr/bin/env python3
"""
traveltime_anisotropy_7period_6stations_newdata_snr_grades.py

Travel-time-weighted percent-anisotropy tomography (regular AND pre-eruption-difference
versions) for the six SNR/quality filter grades used in
rose_7period_6stations_newdata_snr_grades.py, built from the same "newdata" 6-station
dataset (AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3 from
mfast_maxdt_pipeline_transfer/splitting_results_{STA}_{2015_2021,2022_2026}_all_batches.csv --
NOT the separate in-progress AXEC2-only max-dt-wraparound-fix re-run, so all six stations stay
on identical processing).

The six grades (identical definitions/labels to the rose-plot script):
    Page 1        : SNR >= 2.0, quality >= 0.5,  dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg
    Grade B        : SNR >= 3.0, quality >= 0.5,  dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg
    Grade A        : SNR >= 4.0, quality >= 0.5,  dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 10 deg
    Page 1 (q0.75) : SNR >= 2.0, quality >= 0.75, dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg
    Grade B (q0.75): SNR >= 3.0, quality >= 0.75, dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg
    Grade A (q0.75): SNR >= 4.0, quality >= 0.75, dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 10 deg

All six grades are subsets of the loosest one (Page 1, quality>=0.5): every grade requires
quality>=0.5 and SNR>=2.0 and phi_err<=20 at minimum, tightening one or more of those three
knobs, while dt_err<=0.05s and dt<=T_dom/2 are common to all six. So RAY TRACING IS DONE ONCE,
against the Page-1/quality>=0.5 survivor set only (not the full ~730k-event baseline), and
cached to two CSVs:
    traveltime_anisotropy_newdata_snr_grades_ray_summary.csv        (per-ray metadata)
    traveltime_anisotropy_newdata_snr_grades_ray_voxel_segments.csv (per-ray voxel crossings:
                                                                      ray_id, ix, iy, iz, seg_len_km)
No matching ray-tracing cache existed anywhere else in the repo for this exact ("newdata",
6-station, no-AXEC2-maxdt02-fix) dataset, so this is a fresh cache -- checked first, only
(re)built if missing or if --retrace is passed. All 5 stricter grades are then produced by
re-filtering the cached summary's stored quality/snr_horizontal/phi_error columns and
recomputing T_travel = sum(seg_len_km / Vs_voxel) + A_ray = dt*100/T_travel from the cached
per-ray voxel crossings (cheap, vectorized -- no re-tracing).

Same 0.25 km voxel grid (X:4-12, Y:0-12 km, Z:0-4 km), 4 depth rows, bathymetry/fault/Kidiwela
overlays, coverage gate (>=COUNT_MIN rays, >=MIN_PATH_KM cumulative path per voxel), and 7
eruption-relative periods as the sibling newdata scripts.

Produces two NEW PDFs (does not touch any existing output):
    traveltime_anisotropy_7period_6stations_newdata_snr_grades.pdf       (regular, 6 pages)
    traveltime_anisotropy_7period_6stations_newdata_snr_grades_diff.pdf  (Δ-from-pre-eruption, 6 pages)

Run with:
    python3 traveltime_anisotropy_7period_6stations_newdata_snr_grades.py [--retrace]
"""

import argparse
import csv
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

RAY_SUMMARY_CSV = os.path.join(HERE, 'traveltime_anisotropy_newdata_snr_grades_ray_summary.csv')
RAY_VOXEL_CSV = os.path.join(HERE, 'traveltime_anisotropy_newdata_snr_grades_ray_voxel_segments.csv')

OUT_PDF_REGULAR = os.path.join(HERE, 'traveltime_anisotropy_7period_6stations_newdata_snr_grades.pdf')
OUT_PDF_DIFF = os.path.join(HERE, 'traveltime_anisotropy_7period_6stations_newdata_snr_grades_diff.pdf')

# ── Broadest filter (defines the ray-trace universe -- all 6 grades are subsets) ───
QW_MIN_BROAD = 0.5
SNR_MIN_BROAD = 2.0
PHI_ERR_MAX_BROAD = 20.0
DT_ERR_MAX = 0.05

COUNT_MIN = 8
MIN_PATH_KM = 0.5
N_RAY = 200

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}
KIDIWELA = [dict(x=7.57, y=4.55, label='S1'), dict(x=7.53, y=6.60, label='S2')]

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

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

DEPTH_ROWS = [
    ((0, 3), '0.0–0.75 km', 0.375),
    ((3, 6), '0.75–1.5 km', 1.125),
    ((6, 9), '1.5–2.25 km', 1.875),
    ((9, 12), '2.25–3.0 km', 2.625),
]

Xg, Yg = np.meshgrid(xn + VXY / 2., yn + VXY / 2., indexing='ij')

STATION_FILES = {
    sta: [f'splitting_results_{sta}_2015_2021_all_batches.csv',
          f'splitting_results_{sta}_2022_2026_all_batches.csv']
    for sta in STATIONS
}

vs_voxels = None
vs_flat = None
sta_xy = None
_sta_df = None
_gray = None
_ext = None
tracer = None


def _make_grades(qw_min):
    return [
        dict(key=f'page1_q{qw_min}',
             label=f'SNR >= 2.0, quality >= {qw_min}, dt_err <= {DT_ERR_MAX}s, dt <= T_dom/2, phi_err <= 20 deg',
             snr_min=2.0, phi_err_max=20.0, qw_min=qw_min),
        dict(key=f'gradeB_q{qw_min}',
             label=f'"Grade B": SNR >= 3.0, quality >= {qw_min}, dt_err <= {DT_ERR_MAX}s, dt <= T_dom/2, phi_err <= 20 deg',
             snr_min=3.0, phi_err_max=20.0, qw_min=qw_min),
        dict(key=f'gradeA_q{qw_min}',
             label=f'"Grade A": SNR >= 4.0, quality >= {qw_min}, dt_err <= {DT_ERR_MAX}s, dt <= T_dom/2, phi_err <= 10 deg',
             snr_min=4.0, phi_err_max=10.0, qw_min=qw_min),
    ]


GRADES = _make_grades(0.5) + _make_grades(0.75)


# ── Data loading (broadest filter applied here, BEFORE ray tracing) ─────────────

def load_station_broadest(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'quality', 'dominant_period',
                            'snr_horizontal', 'phi_error', 'dt_error'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')

    df = df[(df['quality'] >= QW_MIN_BROAD) &
            (df['snr_horizontal'] >= SNR_MIN_BROAD) &
            (df['dt_error'] <= DT_ERR_MAX) &
            (df['dt'] <= df['dominant_period'] / 2.0) &
            (df['phi_error'] <= PHI_ERR_MAX_BROAD)]

    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    df['station'] = sta
    return df[['station', 't', 'dt', 'quality', 'snr_horizontal', 'phi_error', 'dt_error', 'x', 'y', 'z']]


def build_7_periods(all_df):
    post = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post)
    bounds = [ERUPTION_END]
    for i in range(1, 5):
        bounds.append(post['t'].iloc[min(int(round(i * n / 5)), n - 1)])
    bounds.append(None)

    def fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else 'present'

    pds = [('Pre-eruption', None, ERUPTION_START), ('Syn-eruption', ERUPTION_START, ERUPTION_END)]
    for i in range(5):
        pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i + 1])}', bounds[i], bounds[i + 1]))
    return pds


# ── Environment (Vs model, station coords, bathymetry, ray tracer) ─────────────

def _setup_environment(need_tracer):
    global vs_voxels, vs_flat, sta_xy, _sta_df, _gray, _ext, tracer

    print('Querying Baillard Vs at each voxel centre...')
    X3, Y3, Z3 = np.meshgrid(xn + VXY / 2., yn + VXY / 2., zn + VZ / 2., indexing='ij')
    vs_voxels = vs_at(X3.ravel(), Y3.ravel(), Z3.ravel()).reshape(NX, NY, NZ)
    vs_voxels = np.clip(vs_voxels, 0.3, 5.0)
    vs_flat = vs_voxels.ravel()
    print(f'  Vs range: {vs_voxels.min():.3f}–{vs_voxels.max():.3f} km/s')

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

    if need_tracer:
        tracer = BaillardRayTracer(stride=5)
        for sta in STATIONS:
            if sta in sta_xy:
                tracer.precompute_station(sta, *sta_xy[sta])


def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)


def _sta(ax):
    for sta, row in _sta_df.iterrows():
        ax.plot(row['x'], row['y'], '^', ms=6, mfc='#FFD700', mec='k', mew=0.7, zorder=12)
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


# ── Ray-tracing cache: build once against the broadest (Page-1, q>=0.5) filter ──

def build_cache(dfs):
    total = sum(len(dfs[s]) for s in STATIONS)
    print(f'\nTracing {total:,} broadest-filter (Page 1, quality>=0.5) rays (n_pts={N_RAY})...')
    t0 = time.time()

    summary_rows = []
    vox_buffer = []
    FLUSH_EVERY = 500_000

    ray_id = 0
    done = 0
    trace_fail = 0

    with open(RAY_VOXEL_CSV, 'w', newline='') as fh:
        wtr = csv.writer(fh)
        wtr.writerow(['ray_id', 'ix', 'iy', 'iz', 'seg_len_km'])

        for sta in STATIONS:
            df = dfs[sta]
            for row in df.itertuples():
                ex, ey, ez = float(row.x), float(row.y), float(row.z)
                if not (ez < 0 or ez > Z_MAX):
                    try:
                        ray = tracer.trace(sta, ex, ey, ez, n_pts=N_RAY)
                        vcols, seg_len_km = tracer.ray_to_voxels(ray, xn, yn, zn)
                        if len(vcols) > 0:
                            ix = vcols // (NY * NZ)
                            rem = vcols % (NY * NZ)
                            iy = rem // NZ
                            iz = rem % NZ
                            for k in range(len(vcols)):
                                vox_buffer.append((ray_id, int(ix[k]), int(iy[k]),
                                                   int(iz[k]), float(seg_len_km[k])))
                            summary_rows.append((ray_id, sta, row.t, float(row.dt),
                                                 float(row.quality), float(row.snr_horizontal),
                                                 float(row.phi_error), float(row.dt_error)))
                            ray_id += 1
                    except RuntimeError:
                        trace_fail += 1
                done += 1
                if done % 5000 == 0:
                    print(f'  {done:,}/{total:,}  {time.time()-t0:.0f}s', end='\r', flush=True)
                if len(vox_buffer) >= FLUSH_EVERY:
                    wtr.writerows(vox_buffer)
                    vox_buffer = []

        if vox_buffer:
            wtr.writerows(vox_buffer)

    print(f'\nTraced {ray_id:,}/{total:,} rays successfully in {time.time()-t0:.0f}s '
          f'(trace-fail={trace_fail:,})')

    summary = pd.DataFrame(summary_rows, columns=[
        'ray_id', 'station', 'event_datetime', 'dt', 'quality', 'snr_horizontal',
        'phi_error', 'dt_error'])
    summary.to_csv(RAY_SUMMARY_CSV, index=False)
    print(f'Wrote {RAY_SUMMARY_CSV} ({len(summary):,} rows)')
    print(f'Wrote {RAY_VOXEL_CSV}')


def load_cache():
    print('Loading cached rays...')
    summary = pd.read_csv(RAY_SUMMARY_CSV)
    summary = summary.sort_values('ray_id').reset_index(drop=True)
    vox = pd.read_csv(RAY_VOXEL_CSV, dtype={
        'ray_id': np.int32, 'ix': np.int16, 'iy': np.int16, 'iz': np.int16,
        'seg_len_km': np.float32})
    print(f'  ray_summary: {len(summary):,} rows  ray_voxel_segments: {len(vox):,} rows')
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


def build_grade_masks(summary):
    q = summary['quality'].values
    snr = summary['snr_horizontal'].values
    phe = summary['phi_error'].values
    masks = {}
    for grade in GRADES:
        masks[grade['key']] = ((q >= grade['qw_min']) & (snr >= grade['snr_min']) &
                                (phe <= grade['phi_err_max']))
    return masks


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


def aggregate_page(vox, vray, ray_survive, ray_period, periods, tt_seg, a_ray_tt):
    keep = ray_survive[vray] & (ray_period[vray] >= 0)
    results = []
    if not keep.any():
        for pi, (lbl, _t0, _t1) in enumerate(periods):
            n = int((ray_survive & (ray_period == pi)).sum())
            results.append(dict(label=lbl, n=n, a=np.full((NX, NY, NZ), np.nan)))
        return results

    ix = vox['ix'].values[keep].astype(np.int64)
    iy = vox['iy'].values[keep].astype(np.int64)
    iz = vox['iz'].values[keep].astype(np.int64)
    flat = ix * (NY * NZ) + iy * NZ + iz
    period = ray_period[vray][keep]

    w = tt_seg[keep]
    a = a_ray_tt[vray][keep].astype(np.float64)
    seg_vals = vox['seg_len_km'].values[keep].astype(np.float64)

    key = period * J + flat
    uniq, inv = np.unique(key, return_inverse=True)
    sum_wa = np.zeros(len(uniq)); np.add.at(sum_wa, inv, a * w)
    sum_w = np.zeros(len(uniq)); np.add.at(sum_w, inv, w)
    sum_p = np.zeros(len(uniq)); np.add.at(sum_p, inv, seg_vals)
    cnt = np.bincount(inv, minlength=len(uniq))
    wmean_a = sum_wa / np.maximum(sum_w, 1e-9)
    gate = (cnt >= COUNT_MIN) & (sum_p >= MIN_PATH_KM)

    up = uniq // J
    uf = uniq % J
    uix = uf // (NY * NZ)
    ur = uf % (NY * NZ)
    uiy = ur // NZ
    uiz = ur % NZ

    for pi, (lbl, _t0, _t1) in enumerate(periods):
        a3d = np.full((NX, NY, NZ), np.nan)
        sel = (up == pi) & gate
        a3d[uix[sel], uiy[sel], uiz[sel]] = wmean_a[sel]
        n = int((ray_survive & (ray_period == pi)).sum())
        results.append(dict(label=lbl, n=n, a=a3d))
    return results


# ── Plotting: regular page ──────────────────────────────────────────────────────

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
                      label='Station (new data)'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.08)
    return fig


# ── Plotting: diff-from-pre-eruption page ───────────────────────────────────────

def make_diff_page(results, title):
    pre_a = results[0]['a']

    diffs = []
    for r in results[1:]:
        d = r['a'] - pre_a
        diffs.append(d)

    all_abs_pre = pre_a[pre_a > 0]
    a_vmax = float(np.nanpercentile(all_abs_pre, 99)) if np.isfinite(all_abs_pre).any() else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.

    all_diff_vals = np.concatenate([d[np.isfinite(d)].ravel() for d in diffs]) if diffs else np.array([])
    d_vmax = float(np.nanpercentile(np.abs(all_diff_vals), 99)) if all_diff_vals.size else 10.
    if not np.isfinite(d_vmax) or d_vmax <= 0:
        d_vmax = 10.

    cmap_abs = plt.colormaps['Blues']
    levels_abs = np.linspace(0., a_vmax, 15)
    cmap_diff = plt.colormaps['RdBu_r']
    levels_diff = np.linspace(-d_vmax, d_vmax, 21)

    n_per = len(results)
    n_dep = len(DEPTH_ROWS)
    panel_w = 1.8
    fig = plt.figure(figsize=(n_per * panel_w + 1.0, n_dep * 3.0 + 0.5))
    gs = GridSpec(n_dep, n_per + 2, width_ratios=[0.04] + [1] * n_per + [0.04],
                 hspace=0.04, wspace=0.04)
    last_h_abs = None
    last_h_diff = None
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(DEPTH_ROWS):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci + 1])
            _bathy(ax)
            if ci == 0:
                sl = np.nanmean(pre_a[:, :, iz0:iz1], axis=2)
            else:
                sl = np.nanmean(diffs[ci - 1][:, :, iz0:iz1], axis=2)
            msk = np.isfinite(sl)
            if msk.sum() >= COUNT_MIN:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                mp = np.ma.masked_where(~msk, sm)
                if not mp.mask.all():
                    if ci == 0:
                        ax.contourf(Xg, Yg, mp, levels=levels_abs, cmap=cmap_abs,
                                    vmin=0, vmax=a_vmax, extend='neither')
                        last_h_abs = ScalarMappable(cmap=cmap_abs, norm=Normalize(0, a_vmax))
                        last_h_abs.set_array([])
                    else:
                        ax.contourf(Xg, Yg, mp, levels=levels_diff, cmap=cmap_diff,
                                    vmin=-d_vmax, vmax=d_vmax, extend='both')
                        last_h_diff = ScalarMappable(cmap=cmap_diff, norm=Normalize(-d_vmax, d_vmax))
                        last_h_diff.set_array([])
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
                title_txt = f'{res["label"]}\nN={res["n"]:,}' + ('' if ci == 0 else ' (Δ vs pre)')
                ax.set_title(title_txt, fontsize=7.5, fontweight='bold', pad=1)
            if ci == 0:
                ax.set_ylabel(f'{zlbl}', fontsize=7)
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
                      label='Station (new data)'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.08)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--retrace', action='store_true')
    args = ap.parse_args()

    cache_exists = os.path.exists(RAY_SUMMARY_CSV) and os.path.exists(RAY_VOXEL_CSV)
    need_tracer = args.retrace or not cache_exists

    _setup_environment(need_tracer=need_tracer)

    if cache_exists and not args.retrace:
        print(f'\nFound existing ray-tracing cache -- skipping ray tracing.\n'
              f'  {RAY_SUMMARY_CSV}\n  {RAY_VOXEL_CSV}')
    else:
        print('\nNo existing ray-tracing cache found for this dataset -- tracing the broadest '
              '(Page 1, quality>=0.5) filter once.')
        print('Loading newdata (broadest filter: SNR>=2.0, quality>=0.5, dt_err<=0.05s, '
              'dt<=T_dom/2, phi_err<=20deg)...')
        dfs = {}
        for sta in STATIONS:
            dfs[sta] = load_station_broadest(sta)
            print(f'  {sta}: {len(dfs[sta]):,}')
        build_cache(dfs)

    summary, vox = load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    all_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    periods = build_7_periods(pd.DataFrame({'t': all_t}))

    print('\nComputing T_travel per ray and A_ray = dt*100/T_travel from cache...')
    tt_seg, a_ray_tt = compute_ttime_anisotropy(summary, vox, vray)

    grade_masks = build_grade_masks(summary)
    ray_period = assign_periods(event_ns, periods)

    formula = 'A = δt·100/T_travel (T_travel = Σ seg_len_km/Vs_voxel over crossed voxels)'
    cbar_label = 'Travel-time-weighted mean % shear anisotropy per voxel'

    regular_pages = []
    diff_pages = []
    for grade in GRADES:
        mask = grade_masks[grade['key']]
        results = aggregate_page(vox, vray, mask, ray_period, periods, tt_seg, a_ray_tt)
        n_total = int(mask.sum())
        covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
        print(f'  {grade["label"]:80s}: {n_total:,} rays  covered-voxels={covered:,}')

        title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — travel-time-weighted mean %% '
                 f'anisotropy per voxel (7-period)\n{formula}\n{grade["label"]} (n={n_total:,})')
        regular_pages.append((results, title))

        diff_title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — pre-eruption anisotropy '
                       f'+ difference from pre-eruption (7-period)\n{formula}\n'
                       f'{grade["label"]} (n={n_total:,})')
        diff_pages.append((results, diff_title))

    print(f'\nWriting {os.path.basename(OUT_PDF_REGULAR)}...')
    with PdfPages(OUT_PDF_REGULAR) as pdf:
        for results, title in regular_pages:
            fig = make_page(results, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {OUT_PDF_REGULAR}')

    print(f'\nWriting {os.path.basename(OUT_PDF_DIFF)}...')
    with PdfPages(OUT_PDF_DIFF) as pdf:
        for results, title in diff_pages:
            fig = make_diff_page(results, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {OUT_PDF_DIFF}')

    print('\nDone.')


if __name__ == '__main__':
    main()
