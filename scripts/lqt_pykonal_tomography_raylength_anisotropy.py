#!/usr/bin/env python3
"""
lqt_pykonal_tomography_raylength_anisotropy.py

Spatial percent-anisotropy tomography for the LQT + PyKonal-FMM production run,
using an explicit, SINGLE-RAY-FIRST definition of percent anisotropy (replaces
the back-projection / travel-time-kernel method of
lqt_pykonal_tomography_backprojection.py).

NEW METHOD (single ray first, then apportion to voxels by path length)
----------------------------------------------------------------------
For each measurement, trace the TRUE (bent) eikonal ray from event to station
through the Baillard 3D Vs model (pykonal_raytracer.BaillardRayTracer), then:

    r        = ray's true bent arc-path length [km]
    Vs_avg   = ray's length-weighted average shear velocity [km/s]
             = sum_k seg_len_km[k] * Vs(voxel_k) / r
    A_ray    = dt * Vs_avg * 100 / r        [percent anisotropy for THIS ray]

A_ray is then apportioned ADDITIVELY across every voxel the ray crosses, in
proportion to the ray-segment length in that voxel:

    A_voxel_k = A_ray * (seg_len_km[k] / r)   for each crossed voxel k

so the per-voxel shares sum back to A_ray (for the portion of the ray inside the
grid extent). Each voxel's reported value is the MEAN of every ray's share that
lands in that voxel. This is a mean of dimensionless percent-anisotropy shares,
NOT the back-projection ΔVs/Vs travel-time kernel of the old script.

Caching
-------
Ray tracing (~580k rays) is the only expensive step and is done ONCE for the
baseline measurement set (a superset of all 6 page filters), then cached to two
CSVs in lqt_pykonal_combined_results/:
    raylength_anis_ray_summary.csv          (one row per ray)
    raylength_anis_ray_voxel_segments.csv   (one row per crossed voxel per ray)
If both caches exist at start they are loaded directly (no re-tracing). Pass
--retrace to force a full re-trace. The 2 period schemes x 6 filter pages are
then computed by filtering/aggregating the cached rays, never re-tracing.

Data source: lqt_pykonal_combined_results/ (all 6 stations) plus AXEC2's
2015-2021 half from production_axec2_lqt_pykonal_results/ (same join against the
MLdd catalogs for event locations as the old script).

Produces two PDFs (distinct names, do NOT overwrite the old outputs):
    lqt_pykonal_tomography_raylength_anisotropy_7period.pdf   (6 pages)
    lqt_pykonal_tomography_raylength_anisotropy_annual.pdf    (6 pages)

Run with:
    python3 lqt_pykonal_tomography_raylength_anisotropy.py [--retrace]
"""

import argparse
import csv
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
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_lqt_pykonal_results')
CATALOG_DIR = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR = DATA_DIR

RAY_SUMMARY_CSV = os.path.join(DATA_DIR, 'raylength_anis_ray_summary.csv')
RAY_VOXEL_CSV = os.path.join(DATA_DIR, 'raylength_anis_ray_voxel_segments.csv')

# ── QC constants (identical values to the old script) ───────────────────────
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04
MAX_DT = 0.30
DT_CUTOFF = MAX_DT * 0.8   # 0.24 s

# Coverage gate (same constant values as the old script's ray-coverage QC).
COUNT_MIN = 8       # minimum contributing rays per displayed voxel
MIN_PATH_KM = 0.5   # minimum cumulative ray-segment length [km] per voxel

# Ray-tracing resample density (200, not the old script's 50).
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

# ── Voxel grid: isotropic 0.25 km voxels (same extent as old script; spacing
# changes from VXY=0.30/VZ=0.25 to VXY=VZ=0.25) ──────────────────────────────
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VXY = 0.25
VZ = 0.25
xn = np.arange(X_START, X_END + VXY * .5, VXY)   # LEFT EDGES (see ray_to_voxels note)
yn = np.arange(Y_START, Y_END + VXY * .5, VXY)
zn = np.arange(0., Z_MAX + VZ * .5, VZ)
NX, NY, NZ = len(xn), len(yn), len(zn)
J = NX * NY * NZ


def voxel_idx(ix, iy, iz):
    # Matches pykonal_raytracer.ray_to_voxels flat-index convention exactly:
    #   flat = ix*NY*NZ + iy*NZ + iz  (== C-order ravel of an (NX,NY,NZ) array)
    return ix * NY * NZ + iy * NZ + iz


# ── Depth rows: 4 clean rows of 0.75 km (3 voxels each) spanning 0-3 km ──────
# (idx_range, label, representative_depth_for_fault_traces) triples, in the
# make_page(depth_rows=...) format. Each idx_range = (iz0, iz1) is used as
# res['a'][:, :, iz0:iz1]; z_for_faults is the row midpoint.
DEPTH_ROWS = [
    ((0, 3), '0.0–0.75 km', 0.375),
    ((3, 6), '0.75–1.5 km', 1.125),
    ((6, 9), '1.5–2.25 km', 1.875),
    ((9, 12), '2.25–3.0 km', 2.625),
]

# Grid centres for plotting (xn/yn are LEFT EDGES, so cell centres are +VXY/2).
Xg, Yg = np.meshgrid(xn + VXY / 2., yn + VXY / 2., indexing='ij')

# ── Module globals populated by _setup_environment() (kept out of import time
# so a plain `import` of this module stays cheap) ────────────────────────────
vs_flat = None        # (J,) Vs per voxel at TRUE centres, flat-index order
CATALOG = None        # MLdd catalogs, concatenated
sta_xy = None         # {sta: (x_km, y_km)}
_sta_df = None        # station table (for _sta/_bathy plotting)
_gray = None          # bathymetry grayscale raster
_ext = None           # bathymetry extent [x0, x1, y0, y1]
tracer = None         # BaillardRayTracer with precomputed station fields


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
    """Load and location-join the LQT + PyKonal results for one station with
    BASELINE cleanup only (success==True, dt>0). No phi_error/dt_error/quality/
    dt-cutoff filtering here -- this baseline set is a superset of all 6 page
    variants, so it is traced ONCE and the per-page filters are applied later
    over the cached ray_summary. quality/phi_error/dt_error columns are kept
    (they may be NaN) for that downstream page filtering."""
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
    n_unmatched = df['event_lat'].isna().sum()
    if n_unmatched:
        print(f'  {sta}: dropping {n_unmatched:,} measurements with no catalog location match')
    df = df.dropna(subset=['event_lat', 'event_lon', 'event_depth'])

    df['x'], df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
    df['z'] = df['event_depth'].values
    # Keep only the columns the pipeline needs downstream.
    for col in ('quality', 'phi_error', 'dt_error', 'phi'):
        if col not in df.columns:
            df[col] = np.nan
    return df[['t', 'dt', 'quality', 'phi_error', 'dt_error', 'phi', 'x', 'y', 'z']]


# ── Period definitions (verbatim from the old script) ───────────────────────

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


def build_annual():
    pds = [('Pre-eruption\n2015', None, ERUPTION_START),
           ('Syn-eruption\n2015', ERUPTION_START, ERUPTION_END),
           ('Post-eruption\n2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC'))]
    for yr in range(2016, 2027):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr + 1}-01-01', tz='UTC') if yr < 2026 else None
        pds.append((str(yr), t0, t1))
    return pds


# ── Bathymetry + plotting helpers (ported verbatim from the old script) ─────

def _setup_environment():
    """Load the heavy shared state (Vs per voxel, MLdd catalogs, station table,
    bathymetry, ray tracer) once, into module globals. Kept out of import time."""
    global vs_flat, CATALOG, sta_xy, _sta_df, _gray, _ext, tracer

    # Vs per voxel from the Baillard model, sampled at TRUE voxel centres
    # (xn+VXY/2, yn+VXY/2, zn+VZ/2) -- xn/yn/zn are LEFT EDGES, so sampling at
    # them directly would offset every lookup by half a voxel. Clip [0.3, 5.0]
    # as the old script does. vs_flat ravel order matches ray_to_voxels' vcols.
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

    # Station table (for plotting markers + tracer.precompute_station).
    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                          engine='python').set_index('s')
    _sta_df = _sta_df.loc[[s for s in STATIONS if s in _sta_df.index]]
    _sta_df['x'], _sta_df['y'] = ((np.asarray(_sta_df['lon']) - INI_LON) * KM_PER_DEG_LON,
                                  (np.asarray(_sta_df['lat']) - INI_LAT) * KM_PER_DEG_LAT)
    sta_xy = {s: (float(_sta_df.loc[s, 'x']), float(_sta_df.loc[s, 'y']))
              for s in STATIONS if s in _sta_df.index}

    # Bathymetry raster (grayscale background).
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

    # Ray tracer with a precomputed FMM travel-time field per station.
    tracer = BaillardRayTracer(stride=5)
    for sta in STATIONS:
        if sta in sta_xy:
            tracer.precompute_station(sta, *sta_xy[sta])


def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)


def _sta(ax, highlight=None):
    """Draw station markers. highlight=None (default, all other callers):
    every station drawn gold, as before. highlight=<station code>: that
    station stays gold, every other station is drawn white -- so a
    per-station page makes obvious which station its data actually came
    from."""
    for sta, row in _sta_df.iterrows():
        color = '#FFD700' if (highlight is None or sta == highlight) else 'white'
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


# ── Ray tracing + cache build ───────────────────────────────────────────────

def build_cache(dfs):
    """Trace every baseline ray ONCE across all 6 stations and write the two
    cache CSVs. Writes the (large) voxel-segment file incrementally in chunks so
    peak memory stays bounded (never a 30M-row concat). Returns nothing; the
    caller re-loads the caches from disk for a single aggregation code path."""
    total = sum(len(dfs[s]) for s in STATIONS if s in sta_xy)
    print(f'\nTracing {total:,} baseline rays (n_pts={N_RAY})...')
    t0_rt = time.time()

    summary_rows = []          # one tuple per ray (small; ~580k rows)
    vox_buffer = []            # accumulates voxel rows, flushed in chunks
    FLUSH_EVERY = 500_000      # flush the voxel buffer at this many rows

    ray_id = 0
    done = 0
    trace_fail = 0
    n_oob = 0
    n_empty = 0

    with open(RAY_VOXEL_CSV, 'w', newline='') as fh:
        wtr = csv.writer(fh)
        wtr.writerow(['ray_id', 'ix', 'iy', 'iz', 'seg_len_km', 'A_voxel_pct'])

        for sta in STATIONS:
            if sta not in sta_xy:
                continue
            for _, row in dfs[sta].iterrows():
                dt = float(row['dt'])
                ev_dt = row['t']
                phi_err = row['phi_error']
                dt_err = row['dt_error']
                qual = row['quality']
                ex, ey, ez = float(row['x']), float(row['y']), float(row['z'])

                trace_ok = 0
                r_km = np.nan
                vs_avg = np.nan
                a_ray = np.nan

                if not (ez < 0 or ez > Z_MAX):
                    try:
                        ray = tracer.trace(sta, ex, ey, ez, n_pts=N_RAY)
                        seglens = np.linalg.norm(np.diff(ray, axis=0), axis=1)
                        r = float(seglens.sum())        # true bent arc length [km]
                        if r > 0:
                            vcols, seg_len_km = tracer.ray_to_voxels(ray, xn, yn, zn)
                            if len(vcols) > 0:
                                vs_avg = float((seg_len_km * vs_flat[vcols]).sum() / r)
                                a_ray = dt * vs_avg * 100.0 / r
                                r_km = r
                                trace_ok = 1
                                # Decode flat vcols -> (ix, iy, iz).
                                ix = vcols // (NY * NZ)
                                rem = vcols % (NY * NZ)
                                iy = rem // NZ
                                iz = rem % NZ
                                # Additive apportionment: shares sum to A_ray.
                                a_vox = a_ray * (seg_len_km / r)
                                for k in range(len(vcols)):
                                    vox_buffer.append((ray_id, int(ix[k]), int(iy[k]),
                                                       int(iz[k]), float(seg_len_km[k]),
                                                       float(a_vox[k])))
                            else:
                                n_empty += 1
                        else:
                            n_empty += 1
                    except RuntimeError:
                        trace_fail += 1
                else:
                    n_oob += 1

                summary_rows.append((ray_id, sta, ev_dt, dt, phi_err, dt_err, qual,
                                     r_km, vs_avg, a_ray, trace_ok))
                ray_id += 1
                done += 1

                if done % 5000 == 0:
                    el = time.time() - t0_rt
                    print(f'  {done:,}/{total:,}  {el:.0f}s', end='\r', flush=True)
                if len(vox_buffer) >= FLUSH_EVERY:
                    wtr.writerows(vox_buffer)
                    vox_buffer = []

        if vox_buffer:
            wtr.writerows(vox_buffer)

    print(f'\nTraced {ray_id:,} rays in {time.time() - t0_rt:.0f}s  '
          f'(ok={sum(r[10] for r in summary_rows):,}, '
          f'out-of-range={n_oob:,}, trace-fail={trace_fail:,}, empty={n_empty:,})')

    summary = pd.DataFrame(summary_rows, columns=[
        'ray_id', 'station', 'event_datetime', 'dt', 'phi_error', 'dt_error',
        'quality', 'r_km', 'vs_avg_kms', 'A_ray_pct', 'trace_ok'])
    summary.to_csv(RAY_SUMMARY_CSV, index=False)
    print(f'Wrote {RAY_SUMMARY_CSV} ({len(summary):,} rows)')
    print(f'Wrote {RAY_VOXEL_CSV}')


def load_cache():
    """Load both cache CSVs from disk with memory-lean dtypes."""
    print('Loading cached rays...')
    summary = pd.read_csv(RAY_SUMMARY_CSV)
    summary = summary.sort_values('ray_id').reset_index(drop=True)
    vox = pd.read_csv(RAY_VOXEL_CSV, dtype={
        'ray_id': np.int32, 'ix': np.int16, 'iy': np.int16, 'iz': np.int16,
        'seg_len_km': np.float32, 'A_voxel_pct': np.float32})
    print(f'  ray_summary: {len(summary):,} rows  '
          f'ray_voxel_segments: {len(vox):,} rows')
    return summary, vox


# ── Page filters + period aggregation ───────────────────────────────────────

def build_page_filters(summary):
    """Return a list of (filter_label, ray_mask) pairs. ray_mask is a boolean
    array over rays (indexed by ray_id, contiguous 0..N-1). Rays with NaN in a
    compared column fail that comparison (excluded), which is the intended
    behaviour for pages 2-6. Page 1 keeps all successfully-traced rays."""
    ok = summary['trace_ok'].values.astype(bool)
    dt = summary['dt'].values
    phe = summary['phi_error'].values
    dte = summary['dt_error'].values
    q = summary['quality'].values

    m1 = ok
    m2 = ok & (dte < DT_ERR_MAX) & (phe < PHI_ERR_MAX)
    m3 = m2 & (q >= 0.7)
    m4 = m3 & (dt <= DT_CUTOFF)
    m5 = m2 & (q >= -0.5)
    m6 = m5 & (dt <= DT_CUTOFF)
    return [
        ('baseline only (success & dt>0)', m1),
        (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}°', m2),
        (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}° & Q_w≥0.7', m3),
        (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}° & Q_w≥0.7 & dt≤{DT_CUTOFF:.2f}s', m4),
        (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}° & Q_w≥-0.5', m5),
        (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}° & Q_w≥-0.5 & dt≤{DT_CUTOFF:.2f}s', m6),
    ]


def _to_ns(ts):
    return None if ts is None else pd.Timestamp(ts).value


def assign_periods(event_ns, periods):
    """Return an int array (length N rays) giving each ray's period index, or -1
    if the ray falls in no period. Periods are disjoint and ordered."""
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


def aggregate_page(vox, vray, ray_survive, ray_period, periods,
                   assign='apportioned', a_ray_pct=None, vs_avg_ray=None):
    """Aggregate cached voxel values into one (NX,NY,NZ) percent-anisotropy grid
    per period, applying the coverage gate (>=COUNT_MIN contributing rays AND
    >=MIN_PATH_KM cumulative segment length per displayed voxel). The coverage
    gate always uses seg_len_km from ray_voxel_segments; only the AVERAGED
    quantity changes with `assign`:
      'apportioned' (default): voxel value = MEAN of the surviving rays'
          apportioned A_voxel_pct shares (A_ray * seg_len_km / r).
      'fullray': voxel value = MEAN over distinct crossing rays of the FULL
          ray anisotropy A_ray_pct (a_ray_pct positional array, indexed by
          ray_id). No de-dup needed: ray_to_voxels merges duplicate crossings,
          so each ray contributes at most one row per voxel.
      'invvs_weighted': voxel value = 1/Vs_ray-WEIGHTED MEAN of the SAME
          apportioned A_voxel_pct shares as 'apportioned', where each row's
          weight w = 1/vs_avg_ray[ray] uses THAT RAY's own length-weighted
          average Vs (vs_avg_ray positional array, indexed by ray_id), not the
          voxel's local Vs:
              voxel_value = Σ_r (A_k(r)/Vs_avg(r)) / Σ_r (1/Vs_avg(r)).
          Down-weights fast-Vs rays to reduce sensitivity to background velocity
          structure. Coverage gate (cnt, sum_p) is unchanged -- it still uses raw
          counts and seg_len_km, never the weights."""
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
    if assign == 'fullray':
        # Full ray anisotropy assigned to every crossed voxel (joined by ray_id).
        a_vals = a_ray_pct[vray][keep].astype(np.float64)
    else:
        # Both 'apportioned' and 'invvs_weighted' use the cached path-length
        # share A_voxel_pct = A_ray * seg_len_km / r (never recomputed).
        a_vals = vox['A_voxel_pct'].values[keep].astype(np.float64)
    seg_vals = vox['seg_len_km'].values[keep].astype(np.float64)

    key = period * J + flat
    uniq, inv = np.unique(key, return_inverse=True)
    sum_a = np.zeros(len(uniq))
    np.add.at(sum_a, inv, a_vals)
    sum_p = np.zeros(len(uniq))
    np.add.at(sum_p, inv, seg_vals)
    cnt = np.bincount(inv, minlength=len(uniq))
    if assign == 'invvs_weighted':
        # 1/Vs_ray-weighted mean of the apportioned shares (weight = 1/ray-own
        # average Vs). Only the averaged quantity changes; the coverage gate
        # below still uses raw cnt and sum_p (seg_len_km), never the weights.
        vs_row = vs_avg_ray[vray][keep].astype(np.float64)
        # Guard: traced rows always have finite, positive vs_avg, but be safe --
        # drop any non-finite / non-positive Vs from the weighting (contributes
        # nothing to either weighted sum).
        good = np.isfinite(vs_row) & (vs_row > 0)
        w = np.where(good, 1.0 / np.where(good, vs_row, 1.0), 0.0)
        sum_wa = np.zeros(len(uniq))
        np.add.at(sum_wa, inv, w * a_vals)
        sum_w = np.zeros(len(uniq))
        np.add.at(sum_w, inv, w)
        mean_a = sum_wa / np.where(sum_w > 0, sum_w, 1.0)
    else:
        mean_a = sum_a / np.maximum(cnt, 1)
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
        a3d[uix[sel], uiy[sel], uiz[sel]] = mean_a[sel]
        n = int((ray_survive & (ray_period == pi)).sum())
        results.append(dict(label=lbl, n=n, a=a3d))
    return results


# ── Figure builder (same look as the old script; wording states the new
# formula, values are already percent so no *100) ────────────────────────────

def make_page(results, title, count_min=COUNT_MIN, depth_rows=None,
              cbar_label='Mean % shear anisotropy per voxel', highlight_station=None):
    """results: list of dict(label, n, a) where a is an (NX,NY,NZ) grid already
    in PERCENT (the new A_voxel_pct mean). depth_rows: (idx_range, label,
    z_for_faults) triples; defaults to DEPTH_ROWS. cbar_label: colorbar wording
    (branches by assignment mode). highlight_station: None (default) draws
    every station gold, as before; a station code draws only that station
    gold and every other station white (per-station pages)."""
    if depth_rows is None:
        depth_rows = DEPTH_ROWS
    all_a = [r['a'][r['a'] > 0].ravel() for r in results if np.isfinite(r['a']).any()]
    a_vmax = float(np.nanpercentile(np.concatenate(all_a), 99)) if all_a else 30.
    if not np.isfinite(a_vmax) or a_vmax <= 0:
        a_vmax = 30.
    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., a_vmax, 15)
    n_per = len(results)
    n_dep = len(depth_rows)
    panel_w = 1.8 if n_per > 7 else 2.2
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.5))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.04, wspace=0.04)
    last_h = None
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(depth_rows):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            _bathy(ax)
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            msk = np.isfinite(sl)
            if msk.sum() >= count_min:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                mp = np.ma.masked_where(~msk, sm)
                if not mp.mask.all():
                    ax.contourf(Xg, Yg, mp, levels=levels, cmap=cmap,
                                vmin=0, vmax=a_vmax, extend='neither')
                    last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, a_vmax))
                    last_h.set_array([])
            _sta(ax, highlight=highlight_station)
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
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=7)
    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6,
                      label='OBS station' if highlight_station is None else f'{highlight_station} (this page\'s data)'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    if highlight_station is not None:
        legend_handles.insert(2, mlines.Line2D(
            [], [], marker='^', color='w', mfc='white', mec='k', ms=6,
            label='other stations (not included)'))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.06)
    return fig


def write_pdf(pages, out_path, cbar_label='Mean % shear anisotropy per voxel'):
    """pages: list of (results, title) pairs, one PDF page each -- or
    (results, title, highlight_station) triples for per-station pages (see
    make_page's highlight_station); 2-tuples default to highlight_station=None
    (unchanged behaviour for every other caller)."""
    print(f'\nWriting {os.path.basename(out_path)}...')
    with PdfPages(out_path) as pdf:
        for page in pages:
            results, title = page[0], page[1]
            highlight_station = page[2] if len(page) > 2 else None
            fig = make_page(results, title, cbar_label=cbar_label,
                            highlight_station=highlight_station)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {out_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--retrace', action='store_true',
                    help='force a full re-trace even if both cache CSVs exist')
    ap.add_argument('--assign', choices=['apportioned', 'fullray', 'invvs_weighted'],
                    default='apportioned',
                    help="voxel-value definition: 'apportioned' (default) averages "
                         "each ray's path-length share A_voxel_pct=A_ray*seg/r; "
                         "'fullray' averages the FULL ray anisotropy A_ray_pct "
                         "assigned to every crossed voxel; 'invvs_weighted' takes "
                         "the same apportioned shares A_voxel_pct but aggregates "
                         "them as a 1/Vs_ray-weighted mean (weight = 1/ray-average "
                         "Vs). Coverage gate (seg_len_km) and caches are identical "
                         "for all three.")
    args = ap.parse_args()

    _setup_environment()

    # Baseline data (needed for the period bin edges, and for tracing if the
    # caches are absent / --retrace).
    print('\nLoading baseline events (success==True, dt>0; no quality filter)...')
    dfs = {}
    for sta in STATIONS:
        df = load_station(sta)
        dfs[sta] = df
        print(f'  {sta}: {len(df):,}')
    all_df = pd.concat(dfs.values(), ignore_index=True)
    print(f'  TOTAL baseline: {len(all_df):,}')

    cache_exists = os.path.exists(RAY_SUMMARY_CSV) and os.path.exists(RAY_VOXEL_CSV)
    if cache_exists and not args.retrace:
        print('\nBoth cache CSVs found -- skipping ray tracing.')
    else:
        if args.retrace and cache_exists:
            print('\n--retrace: rebuilding caches from scratch.')
        build_cache(dfs)

    summary, vox = load_cache()

    # ray_id must be contiguous 0..N-1 so positional arrays index by ray_id.
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values

    event_ns = pd.DatetimeIndex(pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8

    # Full-ray anisotropy per ray, positional (ray_id is contiguous 0..N-1, so
    # this indexes by ray_id). Only used in --assign fullray.
    a_ray_pct = summary['A_ray_pct'].values.astype(np.float64)
    # Ray-own length-weighted average Vs per ray, positional (indexes by
    # ray_id). Only used in --assign invvs_weighted for the 1/Vs weighting.
    vs_avg_ray = summary['vs_avg_kms'].values.astype(np.float64)

    page_filters = build_page_filters(summary)

    if args.assign == 'fullray':
        pdf_infix = 'fullray_'
        formula = ('A = dt·Vs·100/r (full ray anisotropy assigned to each '
                   'crossed voxel)')
        cbar_label = 'Mean % shear anisotropy per voxel (full ray value)'
    elif args.assign == 'invvs_weighted':
        pdf_infix = 'invvsweighted_'
        formula = ('A = dt·Vs·100/r, apportioned by path length; voxels '
                   'aggregated as a 1/Vs_ray-weighted mean (weight = 1/ray-'
                   'average Vs) to reduce sensitivity to background velocity '
                   'structure')
        cbar_label = ('Mean % shear anisotropy per voxel, 1/Vs-weighted '
                      '(ray-own Vs)')
    else:
        pdf_infix = ''
        formula = ('A = dt·Vs·100/r, apportioned to voxels by path length')
        cbar_label = 'Mean % shear anisotropy per voxel'

    schemes = [
        ('7period', build_7_periods(all_df),
         os.path.join(OUT_DIR,
                      f'lqt_pykonal_tomography_raylength_anisotropy_{pdf_infix}7period.pdf')),
        ('annual', build_annual(),
         os.path.join(OUT_DIR,
                      f'lqt_pykonal_tomography_raylength_anisotropy_{pdf_infix}annual.pdf')),
    ]

    for scheme_name, periods, out_path in schemes:
        print(f'\n=== Aggregating {scheme_name} ({len(periods)} periods, '
              f'assign={args.assign}) ===')
        ray_period = assign_periods(event_ns, periods)
        pages = []
        for filt_label, ray_mask in page_filters:
            results = aggregate_page(vox, vray, ray_mask, ray_period, periods,
                                     assign=args.assign, a_ray_pct=a_ray_pct,
                                     vs_avg_ray=vs_avg_ray)
            n_total = int(ray_mask.sum())
            covered = sum(int(np.isfinite(r['a']).sum()) for r in results)
            print(f'  {filt_label:60s}: {n_total:,} rays  covered-voxels={covered:,}')
            title = (f'LQT + PyKonal-FMM tomography — mean % shear anisotropy per voxel '
                     f'({scheme_name})\n{formula}  (n={n_total:,}, filter: {filt_label})')
            pages.append((results, title))
        write_pdf(pages, out_path, cbar_label=cbar_label)

    print('\nDone.')


if __name__ == '__main__':
    main()
