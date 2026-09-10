#!/usr/bin/env python3
"""
lqt_pykonal_tomography_inversion.py

Linear regularized SWS tomographic inversion for the LQT + PyKonal-FMM
production run -- the counterpart of lqt_pykonal_tomography_backprojection.py's
SIMPLE BACK-PROJECTION, but solving the full sparse system for all voxels
simultaneously (Tikhonov damping + 3D Laplacian smoothing, scipy.sparse.
linalg.lsqr) instead of taking each voxel's own ray-weighted average:

    c_i = dt_i * cos(2*phi_i) = sum_j A_ij * u_j        [E component]
    s_i = dt_i * sin(2*phi_i) = sum_j A_ij * v_j        [N component]

    min  ||[A 0; 0 A]*[u;v] - [c;s]||^2 + lam_d*||[u;v]||^2
         + lam_s*||Lap*[u;v]||^2

Data, QC, and kernel convention -- ALL matched to
lqt_pykonal_tomography_backprojection.py for direct comparability:
    Data source: lqt_pykonal_combined_results/ (all 6 stations) plus AXEC2's
    2015-2021 half from production_axec2_lqt_pykonal_results/ (497 batches).
    QC: quality (Q_w, Wustefeld 2010) > 0.7, phi_error < 20 deg,
        dt_error < 0.04 s, dt <= 0.8*max_dt (max_dt = 0.30 s -> dt <= 0.24 s).
    Kernel: A_ij is travel TIME [s] (path length [km] / local Vs [km/s] at
    voxel j's centre, from the Baillard model), NOT raw path length, so the
    recovered (u_j, v_j) are already dimensionless (DeltaVs/Vs components) --
    Vs is used ONCE to build the kernel, no post-hoc DeltaVs/Vs conversion
    step (same resolution as the "Vs double-counting" TODO in
    pykonal_raytracer.py::ray_to_voxels, applied locally here as it was in
    the back-projection script). Ray-coverage QC (MIN_PATH_KM) still uses
    physical path length. Voxel grid identical to the back-projection script.

Regularization: lambda_damp and lambda_smooth (= SMOOTH_RATIO * lambda_damp)
are selected DATA-ADAPTIVELY per LAMBDA_MODE='auto' -- run once on the
pre-eruption subset, scanning a log-spaced lambda_damp grid and picking by
the discrepancy principle (misfit crosses the noise-implied target) or the
L-curve corner when the target is unreachable. This selection logic (lcurve_
corner/discrepancy_index/select_lambda) is carried over from
sws_tomography_inversion.py, the (different-dataset) script this one
replaces for the LQT + PyKonal-FMM production run, with one addition: a
degenerate-misfit-curve fallback (DEGENERATE_MISFIT_REL_RANGE/_TOL) for when
the misfit-vs-lambda curve is nearly flat -- verified on this dataset's
quality>0.7 pre-eruption scan, where the standard curvature-based corner
picked up lsqr's iterative-solver numerical noise as a spurious "corner" at
near-zero damping, producing physically implausible (>100%) DeltaVs/Vs.

Each dt/phi PDF also gets an extra "indeterminate quality" page (same period
bins, same figure style): ALL 6 stations restricted to QW_INDETERMINATE_LO <
Q_w < QW_INDETERMINATE_HI (-0.2 to 0.2), same phi_error/dt_error/dt-cutoff
thresholds as the standard filter (matches rose_plots_lqt_pykonal_unweighted.
py's load_station_indeterminate() and the same variant added to
lqt_pykonal_tomography_backprojection.py). This is a disjoint subset from the
standard quality>QW_MIN build (fresh tracing), solved with the SAME
lambda_damp/lambda_smooth selected on the standard pre-eruption data (an
apples-to-apples comparison of quality categories under identical
regularization, not a fresh L-curve scan), and given its own colour scale
(its anisotropy range can differ substantially from the standard data). Every
dt/phi PDF also gets an analogous "Q_w > -0.5" page (quality > QW_ABOVE_NEG_
HALF, same phi_error/dt_error/dt-cutoff thresholds, same lambda_damp/
lambda_smooth, matches lqt_pykonal_tomography_backprojection.py's variant).

The 7-period dt/phi PDFs additionally get two more Q_w>-0.5 pages, reusing
the ALREADY-TRACED Q_w>-0.5 rays and solver (no re-tracing, no fresh lambda
scan) but a different depth binning and/or period binning:
  - "Q_w>-0.5, new depth bins": same 7 periods as the standard build, but
    depth rows are QW_ABOVE_NEG_HALF_DEPTH_BINS (0-0.5, 0.5-1.25, 1.25-1.75,
    >1.75 km) instead of the default DEPTH_SLICES.
  - "Q_w>-0.5, eruption-focused periods, new depth bins": same new depth
    bins, but periods from build_eruption_focused_periods() -- 3 equal-count
    syn-eruption bins, with pre- and post-eruption EACH re-binned into as
    many equal-count bins as needed to match that same per-bin event count,
    capped at 6 bins per window (literal count-matching alone gives ~16
    pre-eruption / ~29 post-eruption bins on this dataset -- too sparse per
    bin to be informative), instead of build_7_periods()'s single
    pre-eruption bin / fixed 5 post-eruption bins.

Produces in lqt_pykonal_combined_results/:
    lqt_pykonal_lcurve_regularisation.pdf
    lqt_pykonal_tomography_dt_7period_inversion.pdf    (standard + indeterminate + Q_w>-0.5 + 2 more Q_w>-0.5 pages)
    lqt_pykonal_tomography_phi_7period_inversion.pdf   (standard + indeterminate + Q_w>-0.5 + 2 more Q_w>-0.5 pages)
    lqt_pykonal_tomography_dt_annual_inversion.pdf     (standard + indeterminate + Q_w>-0.5 pages, each chunked)
    lqt_pykonal_tomography_phi_annual_inversion.pdf    (standard + indeterminate + Q_w>-0.5 pages, each chunked)

Run with:
    python3 lqt_pykonal_tomography_inversion.py
"""

import glob
import math
import os
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
from PIL import Image as PILImage
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
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

# ── QC constraints (quality > 0.7, stricter than the back-projection script's
# 0.5, per this task's request) ──────────────────────────────────────────────
QW_MIN = 0.7
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04
MAX_DT = 0.30
DT_CUTOFF = MAX_DT * 0.8

# "Indeterminate quality" variant: -0.2 < Q_w < 0.2, same phi_error/dt_error/
# dt-cutoff thresholds as the standard filter above (matches rose_plots_lqt_
# pykonal_unweighted.py's load_station_indeterminate() and the same band added
# to lqt_pykonal_tomography_backprojection.py).
QW_INDETERMINATE_LO = -0.2
QW_INDETERMINATE_HI = 0.2

# "Q_w > -0.5" variant: same phi_error/dt_error/dt-cutoff thresholds as the
# standard filter above, just a looser quality bound (matches
# rose_plots_lqt_pykonal_unweighted.py's load_station_above_neg_half() and
# the same variant added to lqt_pykonal_tomography_backprojection.py).
QW_ABOVE_NEG_HALF = -0.5

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

# ── Voxel grid (identical to lqt_pykonal_tomography_backprojection.py) ──────
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VXY = 0.30
VZ = 0.25
xn = np.arange(X_START, X_END + VXY * .5, VXY)
yn = np.arange(Y_START, Y_END + VXY * .5, VXY)
zn = np.arange(0., Z_MAX + VZ * .5, VZ)
NX, NY, NZ = len(xn), len(yn), len(zn)
J = NX * NY * NZ


def voxel_idx(ix, iy, iz):
    return ix * NY * NZ + iy * NZ + iz


DEPTH_SLICES = [0.3125, 0.9375, 1.5625, 2.1875]
DEPTH_LABELS = ['0.0–0.625 km', '0.625–1.25 km', '1.25–1.875 km', '1.875–2.5 km']
Z_HALF = 0.3125
COUNT_MIN = 8
N_RAY = 50
MIN_PATH_KM = 0.5  # km of cumulative ray length required per displayed voxel


def depth_slice_idx(z0):
    centres = zn + VZ / 2.
    sel = np.where((centres >= z0 - Z_HALF) & (centres < z0 + Z_HALF))[0]
    if len(sel) == 0:
        k = int(np.argmin(np.abs(centres - z0)))
        return k, k + 1
    return int(sel[0]), int(sel[-1]) + 1


# Alternate depth binning for the Q_w>-0.5 variant pages (matches
# axec1_rose_by_depth_lqt_pykonal.py's DEPTH_BINS convention and the same
# variant added to lqt_pykonal_tomography_backprojection.py): explicit
# [lo, hi) ranges rather than a slice centre + half-width.
QW_ABOVE_NEG_HALF_DEPTH_BINS = [(0, 0.5), (0.5, 1.25), (1.25, 1.75), (1.75, None)]
QW_ABOVE_NEG_HALF_DEPTH_LABELS = ['0–0.5 km', '0.5–1.25 km', '1.25–1.75 km', '>1.75 km']


def depth_bin_idx(lo, hi):
    centres = zn + VZ / 2.
    sel = np.where((centres >= lo) & (centres < hi))[0] if hi is not None else np.where(centres >= lo)[0]
    if len(sel) == 0:
        return None
    return int(sel[0]), int(sel[-1]) + 1


# (index_range, label, representative_depth_for_fault_traces) triples.
QW_ABOVE_NEG_HALF_DEPTH_ROWS = [
    (depth_bin_idx(lo, hi), lbl, (lo + hi) / 2 if hi is not None else lo + 0.375)
    for (lo, hi), lbl in zip(QW_ABOVE_NEG_HALF_DEPTH_BINS, QW_ABOVE_NEG_HALF_DEPTH_LABELS)
]


# ── Inversion parameters (lambda determined data-adaptively from the scan) ──
SMOOTH_RATIO = 0.5    # lambda_smooth = SMOOTH_RATIO * lambda_damp (fixed a-priori ratio)
LSQR_ITER = 500
N_LCURVE = 50         # number of lambda samples in the log-spaced scan
LAM_MIN, LAM_MAX = 1e-7, 1e1

LAMBDA_MODE = 'auto'
LAMBDA_FIXED = 0.20
LAMBDA_SCAN = []

# Degenerate-misfit-curve fallback (see lcurve_corner()): when the pre-eruption
# scan's misfit barely depends on lambda (model/discretization error dominates
# over picking-noise, so the noise-based discrepancy target is unreachable and
# curvature-based corner detection is numerically unreliable), pick the
# LARGEST lambda whose misfit is still within DEGENERATE_MISFIT_TOL of the
# achieved floor -- maximizes damping (keeps recovered DeltaVs/Vs physically
# plausible) for a negligible, explicit cost in data fit.
DEGENERATE_MISFIT_REL_RANGE = 0.20   # trigger if (max-min)/min misfit < 20%
DEGENERATE_MISFIT_TOL = 0.05         # accept up to 5% higher misfit than the floor


# ── 3D Laplacian (6-connected) ───────────────────────────────────────────────
def build_laplacian():
    rows, cols_, vals = [], [], []
    for ix in range(NX):
        for iy in range(NY):
            for iz in range(NZ):
                j = voxel_idx(ix, iy, iz)
                nb = 0
                for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                    ni, nj, nk = ix + di, iy + dj, iz + dk
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk < NZ:
                        rows.append(j)
                        cols_.append(voxel_idx(ni, nj, nk))
                        vals.append(1.)
                        nb += 1
                rows.append(j)
                cols_.append(j)
                vals.append(-nb)
    return sp.csr_matrix((vals, (rows, cols_)), shape=(J, J))


print('Building 3D Laplacian...')
LAP = build_laplacian()
print(f'  Laplacian: {LAP.shape}, {LAP.nnz:,} non-zeros')

# ── Pre-compute Vs per voxel from Baillard model (travel-time kernel) ───────
# Sampled at TRUE voxel centres (xn+VXY/2, yn+VXY/2, zn+VZ/2) -- xn/yn/zn
# themselves are each voxel's LEFT EDGE (see the "FIX 3" registration note on
# Xg/Yg below), so sampling at xn/yn/zn directly would offset every Vs
# lookup by half a voxel from the voxel it's meant to represent.
print('Querying Baillard Vs at each voxel centre...')
X3, Y3, Z3 = np.meshgrid(xn + VXY / 2., yn + VXY / 2., zn + VZ / 2., indexing='ij')
vs_voxels = vs_at(X3.ravel(), Y3.ravel(), Z3.ravel()).reshape(NX, NY, NZ)
vs_voxels = np.clip(vs_voxels, 0.3, 5.0)
vs_flat = vs_voxels.ravel()  # flat index order matches voxel_idx()
print(f'  Vs range: {vs_voxels.min():.3f}–{vs_voxels.max():.3f} km/s')

# ── Load LQT + PyKonal results, join event locations from the MLdd catalogs ─
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

print('Loading MLdd catalogs for event locations...')
_cat1 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'))
_cat2 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'))
_cat1['event_datetime'] = pd.to_datetime(_cat1['event_datetime'], utc=True, format='mixed')
_cat2['event_datetime'] = pd.to_datetime(_cat2['event_datetime'], utc=True, format='mixed')
CATALOG = pd.concat([_cat1, _cat2], ignore_index=True)


def load_station(sta, quality_mode='standard'):
    """Load and location-join the LQT + PyKonal results for one station.

    quality_mode='standard' (default): quality > QW_MIN, phi_error < PHI_ERR_MAX,
    dt_error < DT_ERR_MAX, dt <= DT_CUTOFF.
    quality_mode='indeterminate': QW_INDETERMINATE_LO < quality < QW_INDETERMINATE_HI,
    same phi_error/dt_error/dt-cutoff thresholds as 'standard'.
    quality_mode='above_neg_half': quality > QW_ABOVE_NEG_HALF, same
    phi_error/dt_error/dt-cutoff thresholds as 'standard'."""
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = glob.glob(os.path.join(
            AXEC2_2015_2021_DIR, 'splitting_results_mldd_2015_2021_axec2_batch_*.csv'))
        dfs.extend(pd.read_csv(f) for f in batch_files)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality', 'phi_error', 'dt_error'])
    if quality_mode == 'standard':
        df = df[(df['quality'] > QW_MIN) &
                (df['phi_error'] < PHI_ERR_MAX) &
                (df['dt_error'] < DT_ERR_MAX) &
                (df['dt'] <= DT_CUTOFF)]
    elif quality_mode == 'indeterminate':
        df = df[(df['quality'] > QW_INDETERMINATE_LO) &
                (df['quality'] < QW_INDETERMINATE_HI) &
                (df['phi_error'] < PHI_ERR_MAX) &
                (df['dt_error'] < DT_ERR_MAX) &
                (df['dt'] <= DT_CUTOFF)]
    elif quality_mode == 'above_neg_half':
        df = df[(df['quality'] > QW_ABOVE_NEG_HALF) &
                (df['phi_error'] < PHI_ERR_MAX) &
                (df['dt_error'] < DT_ERR_MAX) &
                (df['dt'] <= DT_CUTOFF)]
    else:
        raise ValueError(f'unknown quality_mode {quality_mode!r}')
    df['t'] = pd.to_datetime(df['datetime'], utc=True)
    df['phi_az'] = df['phi'] % 180.0

    sta_cat = CATALOG[CATALOG['station'] == sta].drop_duplicates(subset='event_datetime')
    df = df.merge(sta_cat[['event_datetime', 'event_lat', 'event_lon', 'event_depth']],
                  left_on='t', right_on='event_datetime', how='left')
    n_unmatched = df['event_lat'].isna().sum()
    if n_unmatched:
        print(f'  {sta}: dropping {n_unmatched:,} measurements with no catalog location match')
    df = df.dropna(subset=['event_lat', 'event_lon', 'event_depth'])

    df['x'], df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
    df['z'] = df['event_depth'].values
    return df


print(f'Loading events (quality > {QW_MIN}, phi_error < {PHI_ERR_MAX} deg, '
      f'dt_error < {DT_ERR_MAX} s, dt <= {DT_CUTOFF:.2f} s)...')
dfs = {}
for sta in STATIONS:
    df = load_station(sta)
    dfs[sta] = df
    print(f'  {sta}: {len(df):,}')
all_df = pd.concat(dfs.values(), ignore_index=True)
print(f'  TOTAL retained: {len(all_df):,}')

_sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                       engine='python').set_index('s')
_sta_df = _sta_df.loc[[s for s in STATIONS if s in _sta_df.index]]
_sta_df['x'], _sta_df['y'] = ((np.asarray(_sta_df['lon']) - INI_LON) * KM_PER_DEG_LON,
                               (np.asarray(_sta_df['lat']) - INI_LAT) * KM_PER_DEG_LAT)
sta_xy = {s: (float(_sta_df.loc[s, 'x']), float(_sta_df.loc[s, 'y']))
          for s in STATIONS if s in _sta_df.index}


# ── Period definitions (verbatim from lqt_pykonal_tomography_backprojection.py) ─
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


def _equal_count_bounds(times_sorted, n_bins, lo, hi):
    """n_bins consecutive equal-count boundaries spanning [lo, hi) over an
    already-time-sorted Series, returned as [lo, b1, b2, ..., hi]."""
    n = len(times_sorted)
    bounds = [lo]
    for i in range(1, n_bins):
        idx = min(int(round(i * n / n_bins)), n - 1)
        bounds.append(times_sorted.iloc[idx])
    bounds.append(hi)
    return bounds


def build_eruption_focused_periods(all_df, n_syn_bins=3, max_window_bins=6):
    """Eruption-focused period binning for the Q_w>-0.5 variant (matches
    lqt_pykonal_tomography_backprojection.py's build_eruption_focused_periods()):
    n_syn_bins (3) equal-count bins spanning the syn-eruption window, then
    pre- and post-eruption are EACH split into as many equal-count bins as
    would match that same per-bin event count, capped at max_window_bins (6)
    so the page stays legible -- literal count-matching alone would give far
    more (~16 pre-eruption / ~29 post-eruption) bins on this dataset, mostly
    too sparse per bin to be informative."""
    def fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else 'present'

    syn_df = all_df[(all_df['t'] >= ERUPTION_START) & (all_df['t'] < ERUPTION_END)]
    n_syn = len(syn_df)
    target_n = max(n_syn / n_syn_bins, 1)

    pre_df = all_df[all_df['t'] < ERUPTION_START].sort_values('t').reset_index(drop=True)
    n_pre = len(pre_df)
    n_pre_bins = min(max(1, int(round(n_pre / target_n))), max_window_bins)
    pre_bounds = _equal_count_bounds(pre_df['t'], n_pre_bins, None, ERUPTION_START)

    syn_sorted = syn_df.sort_values('t').reset_index(drop=True)
    syn_bounds = _equal_count_bounds(syn_sorted['t'], n_syn_bins, ERUPTION_START, ERUPTION_END)

    post_df = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n_post = len(post_df)
    n_post_bins = min(max(1, int(round(n_post / target_n))), max_window_bins)
    post_bounds = _equal_count_bounds(post_df['t'], n_post_bins, ERUPTION_END, None)

    print(f'  Eruption-focused periods: target ~{target_n:.0f} events/bin '
          f'(syn N={n_syn:,} / {n_syn_bins} bins) -> '
          f'{n_pre_bins} pre-eruption bins (N={n_pre:,}), {n_syn_bins} syn-eruption bins, '
          f'{n_post_bins} post-eruption bins (N={n_post:,})')

    pds = []
    for i in range(n_pre_bins):
        pds.append((f'Pre {i+1}/{n_pre_bins}\n{fmt(pre_bounds[i])}–{fmt(pre_bounds[i+1])}',
                     pre_bounds[i], pre_bounds[i+1]))
    for i in range(n_syn_bins):
        pds.append((f'Syn {i+1}/{n_syn_bins}\n{fmt(syn_bounds[i])}–{fmt(syn_bounds[i+1])}',
                     syn_bounds[i], syn_bounds[i+1]))
    for i in range(n_post_bins):
        pds.append((f'{fmt(post_bounds[i])}\n–{fmt(post_bounds[i+1])}',
                     post_bounds[i], post_bounds[i+1]))
    return pds


periods = build_7_periods(all_df)

# ── True eikonal ray tracing + global A matrix (travel-time kernel) ─────────
print('\nInitialising PyKonal FMM ray tracer...')
tracer = BaillardRayTracer(stride=5)
for sta in STATIONS:
    if sta in sta_xy:
        tracer.precompute_station(sta, *sta_xy[sta])

def trace_dataframe(df_by_sta, stations, label):
    """Trace rays for a {sta: df} mapping, returning the (A_km, A, ray_dt,
    ray_phi, ray_t, ray_dt_err, ray_phi_err, N_RAYS) tuple used to build/solve
    the inversion. A_km is path length [km] (coverage QC only); A is travel
    time [s] (the inversion kernel, path/Vs)."""
    print(f'\nTracing {sum(len(df_by_sta[s]) for s in stations if s in sta_xy):,} '
          f'{label} events through Baillard Vs model (FMM rays)...')
    t0_rt = time.time()
    coo_rows, coo_cols, coo_vals_km, coo_vals_s = [], [], [], []
    ray_dt, ray_phi, ray_t, ray_dt_err, ray_phi_err = [], [], [], [], []
    row_idx = 0
    done = 0
    trace_fail = 0
    total = sum(len(df_by_sta[s]) for s in stations if s in sta_xy)
    for sta in stations:
        if sta not in sta_xy:
            continue
        for _, row in df_by_sta[sta].iterrows():
            eq_x, eq_y, eq_z = float(row['x']), float(row['y']), float(row['z'])
            if eq_z < 0 or eq_z > Z_MAX:
                done += 1
                continue
            try:
                ray_xyz = tracer.trace(sta, eq_x, eq_y, eq_z, n_pts=N_RAY)
            except RuntimeError:
                trace_fail += 1
                done += 1
                continue
            vcols, vvals_km = tracer.ray_to_voxels(ray_xyz, xn, yn, zn)
            if len(vcols) == 0:
                done += 1
                continue
            coo_rows.extend([row_idx] * len(vcols))
            coo_cols.extend(vcols.tolist())
            coo_vals_km.extend(vvals_km.tolist())
            coo_vals_s.extend((vvals_km / vs_flat[vcols]).tolist())
            ray_dt.append(float(row['dt']))
            ray_phi.append(float(row['phi_az']))
            ray_t.append(row['t'])
            ray_dt_err.append(float(row['dt_error']))
            ray_phi_err.append(float(row['phi_error']))
            row_idx += 1
            done += 1
            if done % 10000 == 0:
                el = time.time() - t0_rt
                print(f'  {done:,}/{total:,}  {el:.0f}s  ~{el/done*(total-done):.0f}s remaining',
                      end='\r', flush=True)
    n_rays = row_idx
    print(f'\nDone in {time.time() - t0_rt:.0f}s  —  {n_rays:,} valid rays, {len(coo_vals_s):,} non-zeros')
    print(f'  Events skipped due to trace failure: {trace_fail:,}')
    a_km = sp.csr_matrix((coo_vals_km, (coo_rows, coo_cols)), shape=(n_rays, J))
    a_s = sp.csr_matrix((coo_vals_s, (coo_rows, coo_cols)), shape=(n_rays, J), dtype=np.float64)
    return (a_km, a_s, np.array(ray_dt), np.array(ray_phi), np.array(ray_t),
            np.array(ray_dt_err), np.array(ray_phi_err), n_rays)


A_km, A, ray_dt, ray_phi, ray_t, ray_dt_err, ray_phi_err, N_RAYS = trace_dataframe(dfs, STATIONS, 'standard')

phi_r = np.radians(2. * ray_phi)
c_data = ray_dt * np.cos(phi_r)   # dt*cos(2*phi)
s_data = ray_dt * np.sin(phi_r)   # dt*sin(2*phi)

# ── Build regularisation blocks ──────────────────────────────────────────────
LAP2 = sp.block_diag([LAP, LAP])   # 2J x 2J
Id2J = sp.eye(2 * J)


def build_augmented(A_sub, lam_damp, lam_smooth):
    G = sp.bmat([[A_sub, None], [None, A_sub]], format='csr')
    G_aug = sp.vstack([G,
                       np.sqrt(lam_damp) * Id2J,
                       np.sqrt(lam_smooth) * LAP2], format='csr')
    return G, G_aug


def solve_system(G_aug, d, d_aug):
    result = spla.lsqr(G_aug, d_aug, iter_lim=LSQR_ITER, show=False)
    x = result[0]
    misfit = float(np.linalg.norm(G_aug[:len(d)] @ x - d))
    mod_norm = float(np.linalg.norm(x))
    return x, misfit, mod_norm


def make_solve_period(A_use, A_km_use, c_use, s_use):
    """Factory for the regularized solve using the travel-time kernel A --
    recovered (u, v) are already dimensionless (DeltaVs/Vs components), no
    post-hoc Vs conversion. Ray-coverage QC (MIN_PATH_KM) uses the separate
    A_km matrix."""
    def solve_period(mask, label, lam_damp, lam_smooth):
        A_sub = A_use[mask]
        c_sub = c_use[mask]
        s_sub = s_use[mask]
        n_sub = int(mask.sum())
        if n_sub < COUNT_MIN * 10:
            print(f'  {label}: only {n_sub} rays — skipping')
            return np.zeros(J), np.zeros(J), np.zeros(J, dtype=bool)
        col_path_km = np.asarray(A_km_use[mask].sum(axis=0)).ravel()
        cov_mask = col_path_km >= MIN_PATH_KM
        d = np.concatenate([c_sub, s_sub])
        d_aug = np.concatenate([d, np.zeros(4 * J)])
        G, G_aug = build_augmented(A_sub, lam_damp, lam_smooth)
        t_s = time.time()
        x, misfit, mnorm = solve_system(G_aug, d, d_aug)
        print(f'  {label}: {n_sub:,} rays  {time.time() - t_s:.0f}s  '
              f'misfit={misfit:.3f}  ||m||={mnorm:.3f}  '
              f'covered={cov_mask.sum():,}/{J} voxels')
        return x[:J], x[J:], cov_mask
    return solve_period


solve_period = make_solve_period(A, A_km, c_data, s_data)


# ── lambda selection from the pre-eruption regularisation curve (base logic
# carried over from sws_tomography_inversion.py; degenerate-curve fallback
# added below, see DEGENERATE_MISFIT_REL_RANGE) ──────────────────────────────
def lcurve_corner(lambdas, misfits, mod_norms):
    n = len(misfits)
    if n < 3:
        return int(np.argmin(mod_norms))
    misfits = np.asarray(misfits, float)
    mod_norms = np.asarray(mod_norms, float)

    # Degenerate/near-flat misfit curve: when the full-scan relative misfit
    # range is small, lambda barely affects data fit at all (model/
    # discretization error dominates the noise floor here, not overfitting),
    # so the standard log-log curvature corner is numerically unreliable --
    # normalizing a near-zero misfit range amplifies lsqr's iterative-solver
    # convergence noise into a spurious "corner" near the lightly-damped end.
    # Verified on the LQT+PyKonal quality>0.7 pre-eruption scan: misfit only
    # spanned 6.17-6.77 (~10%) across 8 decades of lambda, and the standard
    # corner logic below picked the near-undamped end, giving physically
    # implausible (>100%) recovered DeltaVs/Vs. In that regime, use the
    # percent-degradation criterion instead: the LARGEST lambda whose misfit
    # is still within DEGENERATE_MISFIT_TOL of the achieved floor.
    floor = misfits.min()
    rel_range = (misfits.max() - floor) / floor if floor > 0 else np.inf
    if rel_range < DEGENERATE_MISFIT_REL_RANGE:
        ok = misfits <= floor * (1. + DEGENERATE_MISFIT_TOL)
        return int(np.max(np.where(ok)[0]))

    log_rho = np.log10(misfits)
    log_eta = np.log10(mod_norms + 1e-30)
    rng_rho = log_rho.max() - log_rho.min()
    rng_eta = log_eta.max() - log_eta.min()
    if rng_rho < 1e-12 or rng_eta < 1e-12:
        return int(np.argmin(mod_norms))
    lr_n = (log_rho - log_rho.min()) / rng_rho
    le_n = (log_eta - log_eta.min()) / rng_eta
    dr, de = np.gradient(lr_n), np.gradient(le_n)
    ddr, dde = np.gradient(dr), np.gradient(de)
    curv = np.abs(dr * dde - de * ddr) / (dr ** 2 + de ** 2 + 1e-30) ** 1.5
    span = misfits[-1] - misfits[0]
    if span > 0:
        moved = np.abs(misfits - misfits[0]) > 1e-4 * span
        flat_end = int(np.argmax(moved)) if moved.any() else 0
    else:
        flat_end = 0
    flat_end = min(max(flat_end, n // 4), n - 2)
    return flat_end + int(np.argmax(curv[flat_end:]))


def discrepancy_index(lambdas, misfits, target):
    diff = misfits - target
    crosses = np.where(np.diff(np.sign(diff)))[0]
    if len(crosses):
        k = int(crosses[0])
        denom = misfits[k + 1] - misfits[k]
        frac = (target - misfits[k]) / denom if abs(denom) > 1e-30 else 0.0
        frac = float(np.clip(frac, 0.0, 1.0))
        lam = float(lambdas[k] * (lambdas[k + 1] / lambdas[k]) ** frac)
        return k, lam
    k = int(np.argmin(np.abs(diff)))
    return k, float(lambdas[k])


def select_lambda(lambdas, misfits, mod_norms, target, mode):
    misfits = np.asarray(misfits, float)
    mod_norms = np.asarray(mod_norms, float)
    if mode == 'lcurve':
        idx = lcurve_corner(lambdas, misfits, mod_norms)
        return float(lambdas[idx]), idx, 'L-curve corner'
    if mode == 'discrepancy':
        idx, lam = discrepancy_index(lambdas, misfits, target)
        return lam, idx, 'Discrepancy principle'
    if target < misfits[0]:
        idx = lcurve_corner(lambdas, misfits, mod_norms)
        return float(lambdas[idx]), idx, 'L-curve corner'
    if target > misfits[-1]:
        idx = len(lambdas) - 1
        return float(lambdas[idx]), idx, 'Discrepancy (max λ; all models fit noise)'
    idx, lam = discrepancy_index(lambdas, misfits, target)
    return lam, idx, 'Discrepancy principle'


print('\nRegularisation-parameter (λ) selection...')
print(f'  mode = {LAMBDA_MODE!r}  (SMOOTH_RATIO={SMOOTH_RATIO}, so '
      f'λ_smooth = {SMOOTH_RATIO}·λ_damp)')

pre_mask = (ray_t < ERUPTION_START)
A_pre = A[pre_mask]
c_pre = c_data[pre_mask]
s_pre = s_data[pre_mask]
d_pre = np.concatenate([c_pre, s_pre])
d_aug_pre = np.concatenate([d_pre, np.zeros(4 * J)])

_pe = pre_mask
_phi_err_rad = np.radians(ray_phi_err[_pe])
_noise_var = ray_dt_err[_pe] ** 2 + 4. * _phi_err_rad ** 2 * ray_dt[_pe] ** 2
target_residual = float(np.sqrt(_noise_var.sum()))

if LAMBDA_SCAN:
    print('\n  λ justification scan (diagnostic, pre-eruption solve only)...')
    print(f'  {"λ_damp":>10s}  {"λ_smooth":>10s}  {"misfit":>10s}  {"||m||":>10s}')
    for _lam in LAMBDA_SCAN:
        _, _Gaug = build_augmented(A_pre, _lam, _lam * SMOOTH_RATIO)
        _x, _mis, _mn = solve_system(_Gaug, d_pre, d_aug_pre)
        flag = '  <-- LAMBDA_FIXED' if np.isclose(_lam, LAMBDA_FIXED) else ''
        print(f'  {_lam:10.4g}  {_lam*SMOOTH_RATIO:10.4g}  {_mis:10.4f}  {_mn:10.4f}{flag}')
    print('  (scan complete — chosen λ is unchanged)\n')

if LAMBDA_MODE == 'fixed':
    LAMBDA_DAMP = float(LAMBDA_FIXED)
    LAMBDA_SMOOTH = LAMBDA_DAMP * SMOOTH_RATIO
    SELECT_METHOD = 'fixed (not data-selected)'
    print(f'  FIXED: λ_damp={LAMBDA_DAMP:.2e}  λ_smooth={LAMBDA_SMOOTH:.2e}  '
          f'(scan skipped — value is NOT data-selected)')
    _, _Gaug = build_augmented(A_pre, LAMBDA_DAMP, LAMBDA_SMOOTH)
    _x, _mis, _mn = solve_system(_Gaug, d_pre, d_aug_pre)
    lambdas = np.array([LAMBDA_DAMP])
    misfits = np.array([_mis])
    mod_norms = np.array([_mn])
    marker_idx = 0
else:
    lambdas = np.logspace(np.log10(LAM_MIN), np.log10(LAM_MAX), N_LCURVE)
    print(f'  Discrepancy target = {target_residual:.4f}  (N_pre={int(_pe.sum()):,})')
    misfits, mod_norms = [], []
    for lam in lambdas:
        _, G_aug_pre = build_augmented(A_pre, lam, lam * SMOOTH_RATIO)
        _x, mis, mnorm = solve_system(G_aug_pre, d_pre, d_aug_pre)
        misfits.append(mis)
        mod_norms.append(mnorm)
        print(f'  λ={lam:.2e}  misfit={mis:.3f}  ||m||={mnorm:.3f}')
    misfits = np.array(misfits)
    mod_norms = np.array(mod_norms)

    LAMBDA_DAMP, marker_idx, SELECT_METHOD = select_lambda(
        lambdas, misfits, mod_norms, target_residual, LAMBDA_MODE)
    LAMBDA_SMOOTH = LAMBDA_DAMP * SMOOTH_RATIO
    print(f'  SELECTED via {SELECT_METHOD}: '
          f'λ_damp={LAMBDA_DAMP:.2e}  λ_smooth={LAMBDA_SMOOTH:.2e}  '
          f'(marker idx={marker_idx}, target={target_residual:.4f}, '
          f'misfit range [{misfits.min():.3f}, {misfits.max():.3f}])')

# ── Save regularisation figure ───────────────────────────────────────────────
fig_lc, ax_lc = plt.subplots(figsize=(6, 5))
if len(misfits) > 1:
    ax_lc.loglog(misfits, mod_norms, 'b.-', ms=8, lw=1.2, zorder=3)
    for m, n, lam in zip(misfits, mod_norms, lambdas):
        ax_lc.annotate(f'{lam:.0e}', (m, n), fontsize=6,
                       xytext=(4, 0), textcoords='offset points')
else:
    ax_lc.loglog(misfits, mod_norms, 'bo', ms=8, zorder=3)
ax_lc.loglog(misfits[marker_idx], mod_norms[marker_idx], 'r*', ms=16, zorder=5,
             label=f'{SELECT_METHOD}: λ_damp={LAMBDA_DAMP:.2e}')
if np.isfinite(target_residual) and LAMBDA_MODE != 'fixed':
    ax_lc.axvline(target_residual, color='green', ls='--', lw=1.2,
                  label=f'Discrepancy target = {target_residual:.2f}')
ax_lc.set_xlabel('Data misfit ||Gx-d||', fontsize=11)
ax_lc.set_ylabel('Model norm ||x||', fontsize=11)
ax_lc.set_title('Regularisation curve: pre-eruption data (LQT + PyKonal-FMM)\n'
                f'λ_damp={LAMBDA_DAMP:.2e}, λ_smooth={LAMBDA_SMOOTH:.2e}  '
                f'({SELECT_METHOD})',
                fontsize=10, fontweight='bold')
ax_lc.legend(fontsize=9)
ax_lc.grid(True, which='both', alpha=0.3)
fig_lc.tight_layout()
out_lc = os.path.join(OUT_DIR, 'lqt_pykonal_lcurve_regularisation.pdf')
fig_lc.savefig(out_lc, dpi=200, bbox_inches='tight')
plt.close(fig_lc)
print(f'Saved {out_lc}')

# ── Solve for each time period ────────────────────────────────────────────────
def run_periods(period_list, ray_t_use, n_rays_use, solve_fn):
    out = []
    for lbl, t_lo, t_hi in period_list:
        m = (ray_t_use >= t_lo) if t_lo is not None else np.ones(n_rays_use, dtype=bool)
        if t_hi is not None:
            m = m & (ray_t_use < t_hi)
        u, v, cov = solve_fn(m, lbl.replace('\n', ' '), LAMBDA_DAMP, LAMBDA_SMOOTH)
        a_str = np.sqrt(u ** 2 + v ** 2).reshape(NX, NY, NZ)   # = DeltaVs/Vs directly
        theta = (0.5 * np.degrees(np.arctan2(v, u)) % 180).reshape(NX, NY, NZ)
        cov3d = cov.reshape(NX, NY, NZ)
        a_str[~cov3d] = np.nan
        theta[~cov3d] = np.nan
        out.append(dict(label=lbl, n=int(m.sum()), a=a_str, theta=theta, cov=cov3d))
    return out


print(f'\nRunning inversions (λ_damp={LAMBDA_DAMP:.2e}, λ_smooth={LAMBDA_SMOOTH:.2e})...')
results = run_periods(periods, ray_t, N_RAYS, solve_period)

# ── "Indeterminate quality" variant: ALL 6 stations, -0.2 < Q_w < 0.2, same
# phi_error/dt_error/dt-cutoff thresholds as the standard filter (matches
# rose_plots_lqt_pykonal_unweighted.py's load_station_indeterminate() and the
# same variant added to lqt_pykonal_tomography_backprojection.py). Reuses the
# SAME lambda_damp/lambda_smooth selected above (apples-to-apples comparison
# of quality categories under identical regularization), rather than running
# a fresh L-curve scan on this disjoint, differently-noised subset. ─────────

print(f'\nLoading indeterminate-quality data ({QW_INDETERMINATE_LO} < Q_w < '
      f'{QW_INDETERMINATE_HI}, all 6 stations)...')
dfs_indeterminate = {}
for sta in STATIONS:
    dfi = load_station(sta, quality_mode='indeterminate')
    dfs_indeterminate[sta] = dfi
    print(f'  {sta}: {len(dfi):,}')

(A_km_indet, A_indet, ray_dt_indet, ray_phi_indet, ray_t_indet,
 ray_dt_err_indet, ray_phi_err_indet, N_RAYS_indet) = trace_dataframe(
    dfs_indeterminate, STATIONS, 'indeterminate-quality')

phi_r_indet = np.radians(2. * ray_phi_indet)
c_data_indet = ray_dt_indet * np.cos(phi_r_indet)
s_data_indet = ray_dt_indet * np.sin(phi_r_indet)

solve_period_indeterminate = make_solve_period(A_indet, A_km_indet, c_data_indet, s_data_indet)

print(f'\nRunning inversions (indeterminate-quality variant, same λ_damp={LAMBDA_DAMP:.2e}, '
      f'λ_smooth={LAMBDA_SMOOTH:.2e})...')
results_indeterminate = run_periods(periods, ray_t_indet, N_RAYS_indet, solve_period_indeterminate)

# ── "Q_w > -0.5" variant: ALL 6 stations, same phi_error/dt_error/dt-cutoff
# thresholds as the standard filter, just a looser quality bound (matches
# rose_plots_lqt_pykonal_unweighted.py's load_station_above_neg_half() and
# the same variant added to lqt_pykonal_tomography_backprojection.py). Reuses
# the SAME lambda_damp/lambda_smooth selected above. ─────────────────────────

print(f'\nLoading Q_w>{QW_ABOVE_NEG_HALF} data (all 6 stations)...')
dfs_above_neg_half = {}
for sta in STATIONS:
    dfa = load_station(sta, quality_mode='above_neg_half')
    dfs_above_neg_half[sta] = dfa
    print(f'  {sta}: {len(dfa):,}')

(A_km_anh, A_anh, ray_dt_anh, ray_phi_anh, ray_t_anh,
 ray_dt_err_anh, ray_phi_err_anh, N_RAYS_anh) = trace_dataframe(
    dfs_above_neg_half, STATIONS, 'Q_w>-0.5')

phi_r_anh = np.radians(2. * ray_phi_anh)
c_data_anh = ray_dt_anh * np.cos(phi_r_anh)
s_data_anh = ray_dt_anh * np.sin(phi_r_anh)

solve_period_above_neg_half = make_solve_period(A_anh, A_km_anh, c_data_anh, s_data_anh)

print(f'\nRunning inversions (Q_w>-0.5 variant, same λ_damp={LAMBDA_DAMP:.2e}, '
      f'λ_smooth={LAMBDA_SMOOTH:.2e})...')
results_above_neg_half = run_periods(periods, ray_t_anh, N_RAYS_anh, solve_period_above_neg_half)

# ── Q_w>-0.5, eruption-focused period binning (3 syn-eruption bins, pre-/
# post-eruption re-binned to match the same per-bin event count, capped at 6
# bins/window) -- reuses the already-traced Q_w>-0.5 rays and solver, just a
# different period mask, so no re-tracing/re-solving-for-lambda needed. ─────
ray_t_anh_df = pd.DataFrame({'t': ray_t_anh})
periods_erupt_focused = build_eruption_focused_periods(ray_t_anh_df)
print(f'\nRunning inversions (Q_w>-0.5, eruption-focused periods, same λ_damp={LAMBDA_DAMP:.2e}, '
      f'λ_smooth={LAMBDA_SMOOTH:.2e})...')
results_erupt_focused_above_neg_half = run_periods(
    periods_erupt_focused, ray_t_anh, N_RAYS_anh, solve_period_above_neg_half)

# ── Bathymetry ────────────────────────────────────────────────────────────────
BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
         'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS = None
_p = PILImage.open(BATHY)
_t = _p.tag_v2
_olon, _olat = _t[33922][3], _t[33922][4]
_pl, _pb = _t[33550][0], _t[33550][1]
_nc, _nr = _p.size
_p.close()
_c0 = max(0, int(((INI_LON + X_START / KM_PER_DEG_LON) - _olon) / _pl) - 2)
_c1 = min(_nc, int(((INI_LON + X_END / KM_PER_DEG_LON) - _olon) / _pl) + 2)
_r0 = max(0, int((_olat - (INI_LAT + Y_END / KM_PER_DEG_LAT)) / _pb) - 2)
_r1 = min(_nr, int((_olat - (INI_LAT + Y_START / KM_PER_DEG_LAT)) / _pb) + 2)
_rgb = tifffile.imread(BATHY)[_r0:_r1, _c0:_c1]
_ds = max(1, max(_rgb.shape[:2]) // 1024)
_rgb = _rgb[::_ds, ::_ds]
_gray = np.dot(_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
_ext = [(_olon + _c0 * _pl - INI_LON) * KM_PER_DEG_LON, (_olon + _c1 * _pl - INI_LON) * KM_PER_DEG_LON,
        (_olat - _r1 * _pb - INI_LAT) * KM_PER_DEG_LAT, (_olat - _r0 * _pb - INI_LAT) * KM_PER_DEG_LAT]


def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)


def _sta_markers(ax):
    for sta, row in _sta_df.iterrows():
        ax.plot(row['x'], row['y'], '^', ms=6, mfc='#FFD700', mec='k', mew=0.7, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


def _kidiwela(ax):
    for s in KIDIWELA:
        ax.plot(s['x'], s['y'], 'o', ms=4, mfc='red', mec='k', mew=0.5, zorder=14)
        ax.text(s['x'] + 0.12, s['y'] + 0.12, s['label'], fontsize=5, color='red',
                fontweight='bold', zorder=15)


COT66 = math.cos(math.radians(66)) / math.sin(math.radians(66))
DEPTH_TOP = 0.1
strike_rad = math.radians(330.)
L = 8.
s_hat = np.array([math.sin(strike_rad), math.cos(strike_rad)])


def _fault_traces(ax, z0):
    for cx, cy, dip_az in [(8.5, 4.0, 60.), (7.0, 4.0, 240.)]:
        horiz = (z0 - DEPTH_TOP) * COT66
        da = math.radians(dip_az)
        cx_z = cx + horiz * math.sin(da)
        cy_z = cy + horiz * math.cos(da)
        x1, y1 = cx_z - L / 2 * s_hat[0], cy_z - L / 2 * s_hat[1]
        x2, y2 = cx_z + L / 2 * s_hat[0], cy_z + L / 2 * s_hat[1]
        ax.plot([x1, x2], [y1, y2], '-', color='#cc0000', lw=0.9, alpha=0.75, zorder=11)


Xg, Yg = np.meshgrid(xn + VXY / 2., yn + VXY / 2., indexing='ij')

# ── Global colour scale (percent shear anisotropy, same convention as the
# back-projection script) ────────────────────────────────────────────────────
all_a = np.concatenate([r['a'][r['a'] > 0].ravel() * 100. for r in results if np.isfinite(r['a']).any()])
A_VMAX_DEFAULT = float(np.percentile(all_a, 99)) if len(all_a) else 5.0
a_vmin = 0.
print(f'\nAnisotropy range: 0 – {A_VMAX_DEFAULT:.2f}%')


def compute_a_vmax(results_list, default=5.0):
    """99th-percentile-based colour-scale max for a set of period results
    (used as-is for the standard 7-period/annual pages via A_VMAX_DEFAULT;
    recomputed locally for variant pages like indeterminate-quality, whose
    anisotropy range may differ substantially from the standard data)."""
    all_a = np.concatenate([r['a'][r['a'] > 0].ravel() * 100. for r in results_list
                             if np.isfinite(r['a']).any()]) if results_list else np.array([])
    return float(np.percentile(all_a, 99)) if len(all_a) else default


# ── Figure builder: anisotropy strength (% shear anisotropy, DeltaVs/Vs) ────
def make_strength_page(results_list, title, a_vmax_override=None, depth_rows=None):
    """depth_rows: optional override, list of (idx_range, label, z_for_faults)
    triples (see QW_ABOVE_NEG_HALF_DEPTH_ROWS); defaults to the standard
    DEPTH_SLICES/DEPTH_LABELS depth-slice binning."""
    if depth_rows is None:
        depth_rows = [(depth_slice_idx(z0), zlbl, z0) for z0, zlbl in zip(DEPTH_SLICES, DEPTH_LABELS)]
    n_dep = len(depth_rows)
    n_per = len(results_list)
    panel_w = 1.8 if n_per > 7 else 2.2
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.3))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.04, wspace=0.04)
    cmap = plt.colormaps['Blues']
    a_vmax = a_vmax_override if a_vmax_override is not None else A_VMAX_DEFAULT
    levels = np.linspace(a_vmin, a_vmax, 15)
    last_h = None
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(depth_rows):
        for ci, res in enumerate(results_list):
            ax = fig.add_subplot(gs[ri, ci])
            _bathy(ax)
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2) * 100.
            msk = np.isfinite(sl)
            if msk.sum() >= COUNT_MIN:
                sl_fill = np.where(msk, sl, 0.)
                sm = gaussian_filter(sl_fill, sigma=1.0)
                mp = np.ma.masked_where(~msk, sm)
                if not mp.mask.all():
                    ax.contourf(Xg, Yg, mp, levels=levels, cmap=cmap,
                                vmin=a_vmin, vmax=a_vmax, extend='neither')
                    last_h = ScalarMappable(cmap=cmap, norm=Normalize(a_vmin, a_vmax))
                    last_h.set_array([])
            _sta_markers(ax)
            _kidiwela(ax)
            _fault_traces(ax, z0)
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
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(a_vmin, a_vmax, 5),
                           format=FormatStrFormatter('%.1f'))
        cb.set_label('% shear anisotropy (ΔVs/Vs × 100, regularized inversion)', fontsize=9)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='OBS station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.subplots_adjust(bottom=0.07)
    return fig


# ── Figure builder: phi (fast direction sticks) ─────────────────────────────
def make_phi_page(results_list, title, depth_rows=None):
    """depth_rows: optional override, same convention as make_strength_page()."""
    if depth_rows is None:
        depth_rows = [(depth_slice_idx(z0), zlbl, z0) for z0, zlbl in zip(DEPTH_SLICES, DEPTH_LABELS)]
    n_dep = len(depth_rows)
    n_per = len(results_list)
    panel_w = 1.8 if n_per > 7 else 2.2
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.3))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.04, wspace=0.04)
    cmap_phi = plt.colormaps['hsv_r']
    last_h = None
    STEP = 3
    SLEN = 0.18
    LW = 0.7
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(depth_rows):
        for ci, res in enumerate(results_list):
            ax = fig.add_subplot(gs[ri, ci])
            _bathy(ax)
            a_sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            th_sl = np.nanmean(res['theta'][:, :, iz0:iz1], axis=2)
            msk = np.isfinite(a_sl)
            segs, colors = [], []
            ny_, nx_ = a_sl.shape
            for i in range(0, ny_, STEP):
                for j in range(0, nx_, STEP):
                    if not msk[i, j]:
                        continue
                    phi_r_ = np.radians(th_sl[i, j])
                    x0, y0 = Xg[i, j], Yg[i, j]
                    dx = SLEN * np.sin(phi_r_)
                    dy = SLEN * np.cos(phi_r_)
                    segs.append([(x0 - dx, y0 - dy), (x0 + dx, y0 + dy)])
                    colors.append(th_sl[i, j] / 180.)
            if segs:
                lc = LineCollection(segs, colors=cmap_phi(colors), linewidths=LW, zorder=6, alpha=0.9)
                ax.add_collection(lc)
                last_h = ScalarMappable(cmap=cmap_phi, norm=Normalize(0, 180))
                last_h.set_array([])
            _sta_markers(ax)
            _kidiwela(ax)
            _fault_traces(ax, z0)
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
        cb = plt.colorbar(last_h, cax=cax, ticks=np.arange(0, 181, 30),
                           format=FormatStrFormatter('%.0f'))
        cb.set_label('Fast direction θ [° from N]', fontsize=9)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='OBS station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.subplots_adjust(bottom=0.07)
    return fig


# ── Annual inversion (standard + indeterminate-quality + Q_w>-0.5 variants) ─
print('\nRunning annual inversions...')
periods_annual = build_annual()
results_annual = run_periods(periods_annual, ray_t, N_RAYS, solve_period)
results_annual_indeterminate = run_periods(periods_annual, ray_t_indet, N_RAYS_indet, solve_period_indeterminate)
results_annual_above_neg_half = run_periods(periods_annual, ray_t_anh, N_RAYS_anh, solve_period_above_neg_half)

# ── Write PDFs: standard page(s) + variant page(s) ──────────────────────────
tag = (f'(quality>{QW_MIN}, phi_err<{PHI_ERR_MAX}°, dt_err<{DT_ERR_MAX}s, '
       f'dt≤{DT_CUTOFF:.2f}s, regularized inversion (lsqr) -- λ_d={LAMBDA_DAMP:.2e}, '
       f'λ_s={LAMBDA_SMOOTH:.2e} ({SELECT_METHOD}), travel-time kernel A_ij=path/Vs, '
       f'a=ΔVs/Vs=dt/T directly)')

tag_indeterminate = (f'(ALL 6 stations: {QW_INDETERMINATE_LO}<Q_w<{QW_INDETERMINATE_HI} '
                     f'("indeterminate quality"), phi_err<{PHI_ERR_MAX}°, dt_err<{DT_ERR_MAX}s, '
                     f'dt≤{DT_CUTOFF:.2f}s, regularized inversion (lsqr) -- same λ_d={LAMBDA_DAMP:.2e}, '
                     f'λ_s={LAMBDA_SMOOTH:.2e} as the standard page, travel-time kernel A_ij=path/Vs, '
                     f'a=ΔVs/Vs=dt/T directly)')

tag_above_neg_half = (f'(ALL 6 stations: Q_w>{QW_ABOVE_NEG_HALF}, phi_err<{PHI_ERR_MAX}°, '
                     f'dt_err<{DT_ERR_MAX}s, dt≤{DT_CUTOFF:.2f}s, regularized inversion (lsqr) -- '
                     f'same λ_d={LAMBDA_DAMP:.2e}, λ_s={LAMBDA_SMOOTH:.2e} as the standard page, '
                     f'travel-time kernel A_ij=path/Vs, a=ΔVs/Vs=dt/T directly)')

tag_above_neg_half_depth = tag_above_neg_half.replace(
    ')', ', depth bins: 0-0.5/0.5-1.25/1.25-1.75/>1.75 km)')
tag_above_neg_half_erupt = tag_above_neg_half.replace(
    ')', ', depth bins: 0-0.5/0.5-1.25/1.25-1.75/>1.75 km, eruption-focused periods '
         '(3 syn-eruption bins, pre-/post-eruption re-binned to the same per-bin event count, '
         'capped at 6 bins/window))')

a_vmax_indet = compute_a_vmax(results_indeterminate)
a_vmax_indet_annual = compute_a_vmax(results_annual_indeterminate)
a_vmax_anh = compute_a_vmax(results_above_neg_half)
a_vmax_anh_annual = compute_a_vmax(results_annual_above_neg_half)
a_vmax_erupt_focused = compute_a_vmax(results_erupt_focused_above_neg_half)
print(f'Indeterminate-quality anisotropy range: 0 – {a_vmax_indet:.2f}% (7-period), '
      f'0 – {a_vmax_indet_annual:.2f}% (annual)')
print(f'Q_w>-0.5 anisotropy range: 0 – {a_vmax_anh:.2f}% (7-period), '
      f'0 – {a_vmax_anh_annual:.2f}% (annual), '
      f'0 – {a_vmax_erupt_focused:.2f}% (eruption-focused periods)')

CHUNK = 7


def write_strength_pdf(entries, out_path):
    """entries: list of (results_list, title, a_vmax_override) triples, or
    (results_list, title, a_vmax_override, depth_rows) quadruples to override
    the depth binning for that page. Each is paginated in groups of CHUNK and
    written as successive pages into ONE pdf."""
    print(f'\nWriting {os.path.basename(out_path)}...')
    with PdfPages(out_path) as pdf:
        for entry in entries:
            results_list, title, vmax = entry[0], entry[1], entry[2]
            depth_rows = entry[3] if len(entry) > 3 else None
            for start in range(0, len(results_list), CHUNK):
                fig = make_strength_page(results_list[start:start + CHUNK], title,
                                          a_vmax_override=vmax, depth_rows=depth_rows)
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
    print(f'Saved {out_path}')


def write_phi_pdf(entries, out_path):
    """entries: list of (results_list, title) pairs, or (results_list, title,
    depth_rows) triples to override the depth binning for that page. Same
    pagination as write_strength_pdf()."""
    print(f'\nWriting {os.path.basename(out_path)}...')
    with PdfPages(out_path) as pdf:
        for entry in entries:
            results_list, title = entry[0], entry[1]
            depth_rows = entry[2] if len(entry) > 2 else None
            for start in range(0, len(results_list), CHUNK):
                fig = make_phi_page(results_list[start:start + CHUNK], title, depth_rows=depth_rows)
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
    print(f'Saved {out_path}')


out_dt = os.path.join(OUT_DIR, 'lqt_pykonal_tomography_dt_7period_inversion.pdf')
write_strength_pdf([
    (results, f'LQT + PyKonal-FMM tomography — % shear anisotropy, regularized inversion (7 periods)  {tag}', None),
    (results_indeterminate,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, regularized inversion (7 periods, indeterminate quality)  '
     f'{tag_indeterminate}', a_vmax_indet),
    (results_above_neg_half,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, regularized inversion (7 periods, Q_w>-0.5)  '
     f'{tag_above_neg_half}', a_vmax_anh),
    (results_above_neg_half,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, regularized inversion (7 periods, Q_w>-0.5, new depth bins)  '
     f'{tag_above_neg_half_depth}', a_vmax_anh, QW_ABOVE_NEG_HALF_DEPTH_ROWS),
    (results_erupt_focused_above_neg_half,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, regularized inversion (Q_w>-0.5, eruption-focused periods, '
     f'new depth bins)  {tag_above_neg_half_erupt}', a_vmax_erupt_focused, QW_ABOVE_NEG_HALF_DEPTH_ROWS),
], out_dt)

out_phi = os.path.join(OUT_DIR, 'lqt_pykonal_tomography_phi_7period_inversion.pdf')
write_phi_pdf([
    (results, f'LQT + PyKonal-FMM tomography — fast direction θ, regularized inversion (7 periods)  {tag}'),
    (results_indeterminate,
     f'LQT + PyKonal-FMM tomography — fast direction θ, regularized inversion (7 periods, indeterminate quality)  '
     f'{tag_indeterminate}'),
    (results_above_neg_half,
     f'LQT + PyKonal-FMM tomography — fast direction θ, regularized inversion (7 periods, Q_w>-0.5)  '
     f'{tag_above_neg_half}'),
    (results_above_neg_half,
     f'LQT + PyKonal-FMM tomography — fast direction θ, regularized inversion (7 periods, Q_w>-0.5, new depth bins)  '
     f'{tag_above_neg_half_depth}', QW_ABOVE_NEG_HALF_DEPTH_ROWS),
    (results_erupt_focused_above_neg_half,
     f'LQT + PyKonal-FMM tomography — fast direction θ, regularized inversion (Q_w>-0.5, eruption-focused periods, '
     f'new depth bins)  {tag_above_neg_half_erupt}', QW_ABOVE_NEG_HALF_DEPTH_ROWS),
], out_phi)

out_dt_a = os.path.join(OUT_DIR, 'lqt_pykonal_tomography_dt_annual_inversion.pdf')
write_strength_pdf([
    (results_annual, f'LQT + PyKonal-FMM tomography annual — % shear anisotropy, regularized inversion  {tag}', None),
    (results_annual_indeterminate,
     f'LQT + PyKonal-FMM tomography annual — % shear anisotropy, regularized inversion (indeterminate quality)  '
     f'{tag_indeterminate}', a_vmax_indet_annual),
    (results_annual_above_neg_half,
     f'LQT + PyKonal-FMM tomography annual — % shear anisotropy, regularized inversion (Q_w>-0.5)  '
     f'{tag_above_neg_half}', a_vmax_anh_annual),
], out_dt_a)

out_phi_a = os.path.join(OUT_DIR, 'lqt_pykonal_tomography_phi_annual_inversion.pdf')
write_phi_pdf([
    (results_annual, f'LQT + PyKonal-FMM tomography annual — fast direction θ, regularized inversion  {tag}'),
    (results_annual_indeterminate,
     f'LQT + PyKonal-FMM tomography annual — fast direction θ, regularized inversion (indeterminate quality)  '
     f'{tag_indeterminate}'),
    (results_annual_above_neg_half,
     f'LQT + PyKonal-FMM tomography annual — fast direction θ, regularized inversion (Q_w>-0.5)  '
     f'{tag_above_neg_half}'),
], out_phi_a)

print('\nDone.')
