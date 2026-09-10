#!/usr/bin/env python3
"""
spatial_quadtree_fast_direction_tier5_6stations.py

2D quad-tree spatial map of fast-polarization direction (phi), using tier-5 QC splitting
results (quality>=0.5, dt<T_dom/2, phi_error<10deg, dt_error<0.05s) from all 6 stations
(AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3), per explicit user request.

Ray midpoint: each event-station ray is TRACED in 3D through the Baillard S-wave velocity
model with pykonal_raytracer.BaillardRayTracer (same FMM eikonal ray tracer + Vs model used
by traveltime_anisotropy_7period_6stations_newdata.py and the lqt_pykonal_tomography_*
scripts -- NOT a straight-line source-receiver approximation), and the ray's own arc-length
midpoint (x, y, z) is taken directly from the traced path. Per explicit user instruction,
the 3D midpoint is then COLLAPSED BY DEPTH (z is dropped) so binning is purely by (x, y)
map-view location.

Quad-tree binning: starting from a base grid of BASE_CELL_KM=1.0 x 1.0 km cells, each cell
is recursively subdivided into 4 quadrants (halving the cell size each time, down to a
floor of MIN_CELL_KM=0.25 km -- the repo's "standard" bin size, and the SMALLEST a cell is
allowed to get) wherever the number of ray midpoints landing in it is >= SPLIT_THRESHOLD --
i.e. bin size only ever DECREASES in higher ray-density regions, it is never increased/
merged in low-density regions. Any leaf cell (at whatever level it stops subdividing) with
< MIN_COUNT crossing rays is excluded (not plotted) entirely. BASE_CELL_KM, MIN_CELL_KM,
SPLIT_THRESHOLD and MIN_COUNT are all top-of-file constants, easily tuned larger/smaller
per explicit user instruction ("we can vary the bin size bigger or smaller").

Per retained leaf cell, the CIRCULAR MEAN of its rays' phi (axial, mod 180 -- doubled-angle
vector-average, same convention as circular_median_phi() in
spatial_map_fast_direction_axcc1_axec1_axec2_newdata_7period.py, but MEAN not median here
per explicit user request for "the average fast direction") is plotted as a double-headed
(undirected) quiver tick mark whose AZIMUTH encodes phi, and whose COLOR (via colorbar)
encodes the leaf cell's size in km -- i.e. color shows quad-tree resolution/density, not
direction (unlike the sibling spatial_map_fast_direction_* scripts, which color by phi
itself). A light rectangle outline is also drawn per leaf cell so the quad-tree partition
itself is visible.

Produces: spatial_quadtree_fast_direction_tier5_6stations.pdf (1 page)

Run with:
    python3 spatial_quadtree_fast_direction_tier5_6stations.py
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

from sws_forward_model import STATION_FILE, ll2xy, INI_LON, INI_LAT, KM_PER_DEG_LON, KM_PER_DEG_LAT
from pykonal_raytracer import BaillardRayTracer

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
AXEC2_META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                              'raw_axec2_all_batches_mfast_filters_metadata.csv')
OUT_PDF = os.path.join(HERE, 'spatial_quadtree_fast_direction_tier5_6stations.pdf')

# ── Tier-5 QC (same definition as rose_7period_6stations_newdata.py's tier 5) ───────────
QW_MIN = 0.5
PHI_ERR_MAX = 10.0
DT_ERR_MAX = 0.05

STATION_ORDER = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}
KIDIWELA = [dict(x=7.57, y=4.55, label='S1'), dict(x=7.53, y=6.60, label='S2')]

STATION_FILES = {
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv',
              'splitting_results_AXAS2_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
}

# ── Map extent (same box as the other spatial-map scripts) ──────────────────────────────
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0   # ray tracer sanity bound on event depth (km), same as traveltime_anisotropy_*

# ── Quad-tree parameters (tune freely) ───────────────────────────────────────────────────
BASE_CELL_KM = 1.0      # starting bin size
MIN_CELL_KM = 0.25      # floor -- BASE_CELL_KM / 4 (2 subdivision levels); our "standard" bin size
SPLIT_THRESHOLD = 40    # subdivide a cell into 4 quadrants if it holds >= this many rays
MIN_COUNT = 10          # exclude (don't plot) any leaf cell with fewer rays than this

# ── Ray tracer parameters (same as traveltime_anisotropy_7period_6stations_newdata.py) ──
N_RAY = 200
TRACER_STRIDE = 5


def load_station(sta):
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
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'phi', 'quality', 'dominant_period',
                            'phi_error', 'dt_error'])

    # Tier 5: quality>=0.5, dt<T_dom/2, phi_error<10deg, dt_error<0.05s
    df = df[(df['quality'] >= QW_MIN) &
            (df['dt'] < df['dominant_period'] / 2.0) &
            (df['phi_error'] < PHI_ERR_MAX) &
            (df['dt_error'] < DT_ERR_MAX)]

    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    df['phi_az'] = df['phi'] % 180.0
    return df[['phi_az', 'x', 'y', 'z']].reset_index(drop=True)


def circular_mean_phi(phi_deg):
    """Circular (vector) mean of axially-symmetric (mod 180) fast directions, via the
    doubled-angle trick -- returns the AVERAGE fast direction, mod 180."""
    if len(phi_deg) == 0:
        return np.nan
    angles = 2.0 * np.radians(phi_deg)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return float(np.degrees(np.arctan2(s, c)) / 2.0) % 180.0


def trace_ray_midpoints(dfs, sta_xy):
    """Trace every tier-5 ray in 3D through the Baillard Vs model and return arrays of
    (mid_x, mid_y, phi_az) -- one entry per successfully-traced ray. mid_z is discarded
    per explicit user instruction (trace in 3D, then collapse by depth for 2D binning)."""
    tracer = BaillardRayTracer(stride=TRACER_STRIDE)
    for sta in STATION_ORDER:
        print(f'  Precomputing FMM travel-time field for {sta}...')
        tracer.precompute_station(sta, *sta_xy[sta])

    total = sum(len(dfs[s]) for s in STATION_ORDER)
    print(f'\nTracing {total:,} tier-5 rays (n_pts={N_RAY})...')
    t0 = time.time()

    mid_x, mid_y, phi_az_out = [], [], []
    done = 0
    n_failed = 0
    for sta in STATION_ORDER:
        df = dfs[sta]
        for row in df.itertuples():
            ex, ey, ez = float(row.x), float(row.y), float(row.z)
            if not (ez < 0 or ez > Z_MAX):
                try:
                    ray = tracer.trace(sta, ex, ey, ez, n_pts=N_RAY)
                    d = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(ray, axis=0), axis=1))])
                    mx, my = [np.interp(d[-1] / 2.0, d, ray[:, k]) for k in range(2)]
                    mid_x.append(mx)
                    mid_y.append(my)
                    phi_az_out.append(row.phi_az)
                except RuntimeError:
                    n_failed += 1
            else:
                n_failed += 1
            done += 1
            if done % 5000 == 0:
                print(f'  {done:,}/{total:,}  {time.time()-t0:.0f}s', end='\r', flush=True)
    print(f'\nDone tracing in {time.time()-t0:.0f}s. '
          f'{len(mid_x):,} rays traced successfully, {n_failed:,} failed/out-of-bounds.')
    return np.array(mid_x), np.array(mid_y), np.array(phi_az_out)


def build_quadtree(mx, my, phi, x0, x1, y0, y1, cell_km):
    """Recursively quad-tree-bin ray midpoints (mx, my) with fast direction phi over the
    box [x0,x1) x [y0,y1). Returns a list of dicts (one per retained leaf cell):
    {'cx','cy','size','n','phi'}. Cells with >=SPLIT_THRESHOLD rays are subdivided into 4
    quadrants (as long as cell_km > MIN_CELL_KM); leaf cells with <MIN_COUNT rays are
    dropped entirely."""
    in_cell = (mx >= x0) & (mx < x1) & (my >= y0) & (my < y1)
    n = int(np.sum(in_cell))
    if n == 0:
        return []

    if n >= SPLIT_THRESHOLD and cell_km > MIN_CELL_KM + 1e-9:
        xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = cell_km / 2.0
        out = []
        for xa, xb, ya, yb in [(x0, xm, y0, ym), (xm, x1, y0, ym),
                                (x0, xm, ym, y1), (xm, x1, ym, y1)]:
            out.extend(build_quadtree(mx, my, phi, xa, xb, ya, yb, half))
        return out

    if n < MIN_COUNT:
        return []

    phi_vals = phi[in_cell]
    return [dict(cx=(x0 + x1) / 2.0, cy=(y0 + y1) / 2.0, size=cell_km, n=n,
                 phi=circular_mean_phi(phi_vals), x0=x0, x1=x1, y0=y0, y1=y1)]


def build_all_bins(mx, my, phi):
    base_x = np.arange(X_START, X_END, BASE_CELL_KM)
    base_y = np.arange(Y_START, Y_END, BASE_CELL_KM)
    bins = []
    for x0 in base_x:
        for y0 in base_y:
            bins.extend(build_quadtree(mx, my, phi, x0, x0 + BASE_CELL_KM,
                                        y0, y0 + BASE_CELL_KM, BASE_CELL_KM))
    return bins


def _setup_environment():
    global _sta_df, _gray, _ext

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                          engine='python').set_index('s')
    _sta_df = _sta_df.loc[[s for s in STATION_ORDER if s in _sta_df.index]]
    _sta_df['x'], _sta_df['y'] = ll2xy(_sta_df['lat'].values, _sta_df['lon'].values)

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
        ax.plot(row['x'], row['y'], '^', ms=7, mfc='#FFD700', mec='k', mew=0.8, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=7, zorder=13)


def _kid(ax):
    for s in KIDIWELA:
        ax.plot(s['x'], s['y'], 'o', ms=5, mfc='red', mec='k', mew=0.6, zorder=14)
        ax.text(s['x'] + 0.15, s['y'] + 0.15, s['label'], fontsize=6.5, color='red',
                fontweight='bold', zorder=15)


COT66 = math.cos(math.radians(66)) / math.sin(math.radians(66))
DEPTH_TOP = 0.1
strike_rad = math.radians(330.)
L = 8.
s_hat = np.array([math.sin(strike_rad), math.cos(strike_rad)])
FAULT_Z0 = 1.5   # representative depth (km) for the fault trace projection -- this map is
                 # depth-collapsed (2D), so no single depth is strictly correct; 1.5 km is a
                 # mid-column reference matching the other spatial-map scripts' middle rows.


def _faults(ax):
    for cx, cy, dip_az in [(8.5, 4.0, 60.), (7.0, 4.0, 240.)]:
        horiz = (FAULT_Z0 - DEPTH_TOP) * COT66
        da = math.radians(dip_az)
        cx_z = cx + horiz * math.sin(da)
        cy_z = cy + horiz * math.cos(da)
        x1, y1 = cx_z - L / 2 * s_hat[0], cy_z - L / 2 * s_hat[1]
        x2, y2 = cx_z + L / 2 * s_hat[0], cy_z + L / 2 * s_hat[1]
        ax.plot([x1, x2], [y1, y2], '-', color='#cc0000', lw=1.1, alpha=0.75, zorder=11)


SIZE_CMAP = plt.colormaps['viridis_r']   # smaller (denser) cells -> brighter
SIZE_LEVELS = sorted({BASE_CELL_KM / (2 ** k) for k in range(
    int(round(math.log2(BASE_CELL_KM / MIN_CELL_KM))) + 1)})
SIZE_NORM = Normalize(vmin=min(SIZE_LEVELS), vmax=max(SIZE_LEVELS))


def make_figure(bins, n_total):
    fig, ax = plt.subplots(figsize=(9, 9))
    _bathy(ax)

    for b in bins:
        ax.add_patch(Rectangle((b['x0'], b['y0']), b['x1'] - b['x0'], b['y1'] - b['y0'],
                                fill=False, edgecolor='0.4', lw=0.4, alpha=0.6, zorder=6))

    if bins:
        cx = np.array([b['cx'] for b in bins])
        cy = np.array([b['cy'] for b in bins])
        sz = np.array([b['size'] for b in bins])
        phi_vals = np.array([b['phi'] for b in bins])
        r = np.radians(phi_vals)
        u, v = np.sin(r), np.cos(r)
        colors = SIZE_CMAP(SIZE_NORM(sz))
        tick_kw = dict(scale=16, width=0.002, headlength=0, headaxislength=0, headwidth=0,
                       pivot='middle', color=colors, alpha=0.95, zorder=7)
        ax.quiver(cx, cy, u, v, **tick_kw)
        ax.quiver(cx, cy, -u, -v, **tick_kw)

    _sta(ax)
    _kid(ax)
    _faults(ax)
    ax.set_xlim(X_START, X_END)
    ax.set_ylim(Y_START, Y_END)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X (km east of 130.1°W)', fontsize=9)
    ax.set_ylabel('Y (km north of 45.9°N)', fontsize=9)

    sm = ScalarMappable(cmap=SIZE_CMAP, norm=SIZE_NORM)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=SIZE_LEVELS)
    cb.ax.set_yticklabels([f'{s*1000:.0f} m' for s in SIZE_LEVELS])
    cb.set_label('Quad-tree cell size', fontsize=9)

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=7, label='Station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label=f'Ring faults (z={FAULT_Z0} km ref.)'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        'Fast direction $\\phi$ by ray-midpoint quad-tree bin (tier 5, 6 stations, ray-traced '
        'through Baillard Vs model)\n'
        f'quality>={QW_MIN}, dt<T_dom/2, phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s '
        f'(N={n_total:,} rays traced, {len(bins):,} bins with N>={MIN_COUNT}); '
        f'base cell {BASE_CELL_KM*1000:.0f} m, split at N>={SPLIT_THRESHOLD}, '
        f'floor {MIN_CELL_KM*1000:.0f} m',
        fontsize=10, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main():
    _setup_environment()

    print(f'Loading tier-5 splitting results (quality>={QW_MIN}, dt<T_dom/2, '
          f'phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s) for all 6 stations...')
    dfs = {}
    for sta in STATION_ORDER:
        dfs[sta] = load_station(sta)
        print(f'  {sta}: {len(dfs[sta]):,} tier-5 events')
    print(f'  TOTAL: {sum(len(d) for d in dfs.values()):,}')

    sta_xy = {sta: (float(_sta_df.loc[sta, 'x']), float(_sta_df.loc[sta, 'y']))
              for sta in STATION_ORDER}

    mid_x, mid_y, phi_az = trace_ray_midpoints(dfs, sta_xy)

    print(f'\nBuilding quad-tree (base={BASE_CELL_KM*1000:.0f}m, floor={MIN_CELL_KM*1000:.0f}m, '
          f'split>={SPLIT_THRESHOLD}, min_count={MIN_COUNT})...')
    bins = build_all_bins(mid_x, mid_y, phi_az)
    size_counts = pd.Series([b['size'] for b in bins]).value_counts().sort_index(ascending=False)
    for size, count in size_counts.items():
        print(f'  {size*1000:.1f} m cells: {count:,}')
    print(f'  TOTAL bins retained: {len(bins):,} (of rays traced: {len(mid_x):,})')

    fig = make_figure(bins, len(mid_x))
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
