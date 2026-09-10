#!/usr/bin/env python3
"""
lqt_pykonal_tomography_backprojection.py

Spatial (dt-derived) percent-anisotropy back-projection tomography for the
LQT + PyKonal-FMM production run, using the TRUE eikonal ray tracer
(pykonal_raytracer.BaillardRayTracer, PyKonal FMM through the Baillard 3D
Vs model) -- follows sws_tomography_backprojection.py's SIMPLE
BACK-PROJECTION method (no damping, no smoothing, no inter-voxel coupling;
each voxel's (U,V) is the travel-time-weighted mean of the sampling rays'
own per-ray-average (u_ray, v_ray)):

    dt_i * cos(2*phi_i) = sum_j A_ij * u_j
    dt_i * sin(2*phi_i) = sum_j A_ij * v_j

Kernel convention (resolves the "Vs double-counting" TODO in
pykonal_raytracer.py::ray_to_voxels -- that geometric path-length kernel is
still used elsewhere and is left unchanged; this script converts its output
to a travel-time kernel locally):
    A_ij is now travel TIME [s] (path length [km] / local Vs [km/s] at voxel
    j's centre, from the Baillard model), not raw path length. L_ray = sum_j
    A_ij is then each ray's total predicted travel time, so
    u_ray = dt*cos(2*phi)/L_ray, v_ray = dt*sin(2*phi)/L_ray are already the
    dimensionless per-ray fractional-anisotropy components (weak-anisotropy
    limit a = dt/T, standard shear-wave-splitting convention) -- Vs is used
    ONCE, to build the kernel, not a second time in a post-hoc ΔVs/Vs
    conversion. Each voxel's (U, V) -- and hence a3d = sqrt(U^2+V^2) = ΔVs/Vs
    -- is read straight off the travel-time-weighted back-projection with no
    separate m_to_dvs_over_vs() conversion step. The ray-coverage QC gate
    (MIN_PATH_KM) still operates on physical path length, computed
    separately, so the coverage mask's meaning is unchanged.

Data source: lqt_pykonal_combined_results/ (all 6 stations) plus AXEC2's
2015-2021 half from production_axec2_lqt_pykonal_results/ (497 batches) --
lqt_pykonal_combined_results/ only has AXEC2's 2022-2026 half combined, same
gap as rose_plots_lqt_pykonal_unweighted.py.

That data has no event location columns, so each station's measurements are
joined against the MLdd catalogs (data/mldd_catalog_2015_2021.csv,
data/mldd_catalog_2022_2026.csv) on (station, event_datetime) to recover
event_lat/lon/depth for ray tracing.

Same QC constraints as the LQT + PyKonal rose plots
(rose_plots_lqt_pykonal_unweighted.py):
    quality (Q_w, Wustefeld 2010) > 0.5
    phi_error < 20 deg
    dt_error  < 0.04 s
    dt <= 0.8 * max_dt   (max_dt = 0.30 s -> dt <= 0.24 s; drops measurements
                          pinned near the grid-search's upper dt bound)

Voxel grid, depth slices, bathymetry background, and figure style are
identical to sws_tomography_backprojection.py's make_page() for direct
comparability.

Each PDF also gets three extra variant pages (same period bins, same figure
style):
  - "AXEC1 all-quality": AXEC1 keeps ALL measurements regardless of quality
    (baseline cleanup only -- success==True, dt>0, no Q_w/phi_error/dt_error/
    dt-cutoff threshold), while every other station still uses the standard
    quality filter above. Rays for the other 5 stations are reused from the
    standard build (no re-tracing); only AXEC1's larger baseline set is
    re-traced.
  - "indeterminate quality": ALL 6 stations restricted to
    QW_INDETERMINATE_LO < Q_w < QW_INDETERMINATE_HI (-0.2 to 0.2), same
    phi_error/dt_error/dt-cutoff thresholds as the standard filter (matches
    rose_plots_lqt_pykonal_unweighted.py's load_station_indeterminate()).
    Disjoint from the standard quality>QW_MIN build, so every station is
    re-traced.
  - "Q_w > -0.5": ALL 6 stations restricted to quality > QW_ABOVE_NEG_HALF
    (-0.5), same phi_error/dt_error/dt-cutoff thresholds as the standard
    filter (matches rose_plots_lqt_pykonal_unweighted.py's
    load_station_above_neg_half()). Every station is re-traced.

The 7-period PDF additionally gets two more Q_w>-0.5 pages, using the ALREADY
-TRACED Q_w>-0.5 rays (no re-tracing) but a different depth binning and/or
period binning:
  - "Q_w>-0.5, new depth bins": same 7 periods as the standard build, but
    depth rows are QW_ABOVE_NEG_HALF_DEPTH_BINS (0-0.5, 0.5-1.25, 1.25-1.75,
    >1.75 km) instead of the default DEPTH_SLICES.
  - "Q_w>-0.5, eruption-focused periods, new depth bins": same new depth
    bins, but periods from build_eruption_focused_periods() -- 3 equal-count
    syn-eruption bins, with pre- and post-eruption EACH re-binned into as
    many equal-count bins as needed to match that same per-bin event count,
    capped at 6 bins per window (literal count-matching alone gives ~16
    pre-eruption / ~29 post-eruption bins on this dataset -- too sparse per
    bin to be informative -- so it's capped, giving 6+3+6=15 periods here),
    instead of build_7_periods()'s single pre-eruption bin / fixed 5
    post-eruption bins.

Produces two PDFs in lqt_pykonal_combined_results/, each with the standard
quality-filtered page(s) (periods chunked 7-per-page) followed by the
AXEC1-all-quality, indeterminate-quality, and Q_w>-0.5 variant pages, same
chunking (the 7-period PDF also gets the two extra Q_w>-0.5 depth/period
variant pages above):
    lqt_pykonal_tomography_dt_7period_backprojection.pdf   (1+1+1+1 + 2 pages)
    lqt_pykonal_tomography_dt_annual_backprojection.pdf    (2+2+2+2 pages)

Run with:
    python3 lqt_pykonal_tomography_backprojection.py
"""

import glob
import math
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
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

# ── QC constraints (same as rose_plots_lqt_pykonal_unweighted.py) ───────────
QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04
MAX_DT = 0.30
DT_CUTOFF = MAX_DT * 0.8

# "Indeterminate quality" variant: -0.2 < Q_w < 0.2, same phi_error/dt_error/
# dt-cutoff thresholds as the standard filter above (see rose_plots_lqt_
# pykonal_unweighted.py's load_station_indeterminate() for the same band).
QW_INDETERMINATE_LO = -0.2
QW_INDETERMINATE_HI = 0.2

# "Q_w > -0.5" variant: same phi_error/dt_error/dt-cutoff thresholds as the
# standard filter above, just a looser quality bound (matches
# rose_plots_lqt_pykonal_unweighted.py's load_station_above_neg_half()).
QW_ABOVE_NEG_HALF = -0.5

# Coverage mask threshold -- require >= MIN_PATH_KM cumulative ray path length
# per displayed voxel (same as sws_tomography_backprojection.py).
MIN_PATH_KM = 0.5   # km

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

# ── Voxel grid (identical to sws_tomography_backprojection.py) ──────────────
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


def depth_slice_idx(z0):
    centres = zn + VZ / 2.
    sel = np.where((centres >= z0 - Z_HALF) & (centres < z0 + Z_HALF))[0]
    if len(sel) == 0:
        k = int(np.argmin(np.abs(centres - z0)))
        return k, k + 1
    return int(sel[0]), int(sel[-1]) + 1


# Alternate depth binning for the Q_w>-0.5 variant pages (matches
# axec1_rose_by_depth_lqt_pykonal.py's DEPTH_BINS convention: explicit
# [lo, hi) ranges rather than a slice centre + half-width).
QW_ABOVE_NEG_HALF_DEPTH_BINS = [(0, 0.5), (0.5, 1.25), (1.25, 1.75), (1.75, None)]
QW_ABOVE_NEG_HALF_DEPTH_LABELS = ['0–0.5 km', '0.5–1.25 km', '1.25–1.75 km', '>1.75 km']


def depth_bin_idx(lo, hi):
    centres = zn + VZ / 2.
    sel = np.where((centres >= lo) & (centres < hi))[0] if hi is not None else np.where(centres >= lo)[0]
    if len(sel) == 0:
        return None
    return int(sel[0]), int(sel[-1]) + 1


# (index_range, label, representative_depth_for_fault_traces) triples, in the
# make_page(depth_rows=...) format.
QW_ABOVE_NEG_HALF_DEPTH_ROWS = [
    (depth_bin_idx(lo, hi), lbl, (lo + hi) / 2 if hi is not None else lo + 0.375)
    for (lo, hi), lbl in zip(QW_ABOVE_NEG_HALF_DEPTH_BINS, QW_ABOVE_NEG_HALF_DEPTH_LABELS)
]


# ── Pre-compute mean Vs per voxel from Baillard model ────────────────────────
# Used to build the travel-time kernel (path length / Vs) below -- the ONLY
# use of Vs in this script, per the kernel-convention note above. Sampled at
# TRUE voxel centres (xn+VXY/2, yn+VXY/2, zn+VZ/2) -- xn/yn/zn themselves are
# each voxel's LEFT EDGE (see the "FIX 3" registration note on Xg/Yg below),
# so sampling at xn/yn/zn directly would offset every Vs lookup by half a
# voxel from the voxel it's meant to represent.
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

    quality_mode='standard' (default): the standard good-quality cut (quality >
    QW_MIN, phi_error < PHI_ERR_MAX, dt_error < DT_ERR_MAX, dt <= DT_CUTOFF).
    quality_mode='baseline': baseline cleanup only (success==True, dt>0) -- ALL
    measurements regardless of quality, matching the AXEC1 "baseline" convention
    used elsewhere this session (dt_vs_quality_lqt_pykonal.py,
    quality_vs_depth_axec1.py, axec1_rose_by_depth_lqt_pykonal.py).
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
    if quality_mode == 'standard':
        df = df.dropna(subset=['quality', 'phi_error', 'dt_error'])
        df = df[(df['quality'] > QW_MIN) &
                (df['phi_error'] < PHI_ERR_MAX) &
                (df['dt_error'] < DT_ERR_MAX) &
                (df['dt'] <= DT_CUTOFF)]
    elif quality_mode == 'indeterminate':
        df = df.dropna(subset=['quality', 'phi_error', 'dt_error'])
        df = df[(df['quality'] > QW_INDETERMINATE_LO) &
                (df['quality'] < QW_INDETERMINATE_HI) &
                (df['phi_error'] < PHI_ERR_MAX) &
                (df['dt_error'] < DT_ERR_MAX) &
                (df['dt'] <= DT_CUTOFF)]
    elif quality_mode == 'above_neg_half':
        df = df.dropna(subset=['quality', 'phi_error', 'dt_error'])
        df = df[(df['quality'] > QW_ABOVE_NEG_HALF) &
                (df['phi_error'] < PHI_ERR_MAX) &
                (df['dt_error'] < DT_ERR_MAX) &
                (df['dt'] <= DT_CUTOFF)]
    elif quality_mode != 'baseline':
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


# ── Period definitions (verbatim from sws_tomography_backprojection.py) ─────

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
    """Eruption-focused period binning for the Q_w>-0.5 variant: n_syn_bins
    (3) equal-count bins spanning the syn-eruption window, then pre- and
    post-eruption are EACH split into as many equal-count bins as would match
    that same per-bin event count (round(N_window / (N_syn/n_syn_bins))),
    capped at max_window_bins (6) so the page stays legible -- literal count-
    matching alone would give ~16 pre-eruption and ~29 post-eruption bins on
    this dataset, mostly too sparse per bin to be informative. Still more,
    count-matched bins before and during the eruption than build_7_periods()'s
    single pre-eruption bin / fixed 5 post-eruption bins, just bounded."""
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


def subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None:
        m = m & (df['t'] < t1)
    return df[m]


# ── True eikonal ray tracing + global A matrix ───────────────────────────────

tracer = BaillardRayTracer(stride=5)
for sta in STATIONS:
    if sta in sta_xy:
        tracer.precompute_station(sta, *sta_xy[sta])

def trace_dataframe(df_by_sta, stations):
    """Trace rays for a {sta: df} mapping, returning the (A_km, A, ray_dt,
    ray_phi, ray_t, ray_sta, N_RAYS) tuple used to build/solve the inversion."""
    t0_rt = time.time()
    coo_rows, coo_cols, coo_vals_km, coo_vals_s = [], [], [], []
    ray_dt, ray_phi, ray_t, ray_sta = [], [], [], []
    row_idx = 0
    done = 0
    trace_fail = 0
    total = sum(len(df_by_sta[s]) for s in stations if s in sta_xy)
    for sta in stations:
        if sta not in sta_xy:
            continue
        for _, row in df_by_sta[sta].iterrows():
            ex, ey, ez = float(row['x']), float(row['y']), float(row['z'])
            if ez < 0 or ez > Z_MAX:
                done += 1
                continue
            try:
                ray = tracer.trace(sta, ex, ey, ez, n_pts=N_RAY)
            except RuntimeError:
                trace_fail += 1
                done += 1
                continue
            vcols, vvals_km = tracer.ray_to_voxels(ray, xn, yn, zn)
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
            ray_sta.append(sta)
            row_idx += 1
            done += 1
            if done % 5000 == 0:
                el = time.time() - t0_rt
                print(f'  {done:,}/{total:,}  {el:.0f}s', end='\r', flush=True)
    n_rays = row_idx
    print(f'\nDone in {time.time() - t0_rt:.0f}s  {n_rays:,} rays')
    print(f'  Events skipped due to trace failure: {trace_fail:,}')
    a_km = sp.csr_matrix((coo_vals_km, (coo_rows, coo_cols)), shape=(n_rays, J))  # path length [km], coverage QC only
    a_s = sp.csr_matrix((coo_vals_s, (coo_rows, coo_cols)), shape=(n_rays, J))    # travel time [s], the inversion kernel
    return (a_km, a_s, np.array(ray_dt), np.array(ray_phi), np.array(ray_t),
            np.array(ray_sta), n_rays)


def make_solve_period(A_use, A_km_use, u_use, v_use):
    """Factory for SIMPLE BACK-PROJECTION solve_period(): each voxel's (U,V) is
    the travel-time-weighted mean of the sampling rays' own (u_ray, v_ray) --
    already dimensionless (ΔVs/Vs), no post-hoc Vs conversion. Ray-coverage QC
    (min_path_km) still uses physical path length, from the A_km matrix."""
    def solve_period(mask, label, count_min=COUNT_MIN, min_path_km=MIN_PATH_KM):
        A_sub = A_use[mask]
        col_path_km_p = np.asarray(A_km_use[mask].sum(axis=0)).ravel()
        cov_p = (col_path_km_p >= min_path_km).reshape(NX, NY, NZ)
        n = int(mask.sum())
        if n < count_min * 5:
            nan3d = np.full((NX, NY, NZ), np.nan)
            return nan3d, cov_p
        numer_u = A_sub.T.dot(u_use[mask])
        numer_v = A_sub.T.dot(v_use[mask])
        col_time_s_p = np.asarray(A_sub.sum(axis=0)).ravel()
        denom = np.maximum(col_time_s_p, 1e-9)
        U = (numer_u / denom).reshape(NX, NY, NZ)
        V = (numer_v / denom).reshape(NX, NY, NZ)
        U[~cov_p] = np.nan
        V[~cov_p] = np.nan
        a3d = np.sqrt(U ** 2 + V ** 2)  # = ΔVs/Vs directly (travel-time kernel)
        a3d[~cov_p] = np.nan
        print(f'  {label:28s}: {n:,} rays  covered={cov_p.sum():,}  '
              f'%anis range 0–{np.nanpercentile(a3d, 99) * 100:.1f}%')
        return a3d, cov_p
    return solve_period


def run_periods(periods, ray_t_use, n_rays_use, solve_fn):
    results = []
    for lbl, t0, t1 in periods:
        m = (ray_t_use >= t0) if t0 is not None else np.ones(n_rays_use, dtype=bool)
        if t1 is not None:
            m = m & (ray_t_use < t1)
        a3d, cov = solve_fn(m, lbl.replace('\n', ' '))
        results.append(dict(label=lbl, n=int(m.sum()), a=a3d, cov=cov))
    return results


print('\nBuilding global A matrix (true PyKonal FMM ray tracing)...')
A_km, A, ray_dt, ray_phi, ray_t, ray_sta, N_RAYS = trace_dataframe(dfs, STATIONS)
phi_r = np.radians(2. * ray_phi)
c_data = ray_dt * np.cos(phi_r)
s_data = ray_dt * np.sin(phi_r)

L_ray = np.asarray(A.sum(axis=1)).ravel()  # each ray's total travel time [s]
L_ray = np.maximum(L_ray, 1e-9)
u_ray = c_data / L_ray
v_ray = s_data / L_ray

solve_period = make_solve_period(A, A_km, u_ray, v_ray)

print('\nRunning back-projections...')
periods_7 = build_7_periods(all_df)
results_7 = run_periods(periods_7, ray_t, N_RAYS, solve_period)

periods_a = build_annual()
results_a = run_periods(periods_a, ray_t, N_RAYS, solve_period)

# ── AXEC1-all-quality variant: AXEC1 keeps ALL measurements (baseline cleanup
# only -- success==True, dt>0, no Q_w/phi_error/dt_error/dt-cutoff threshold),
# every other station keeps the standard quality filter above. Rays for the
# other 5 stations are reused from the standard build above (no re-tracing);
# only AXEC1's (larger) baseline set needs fresh tracing. ────────────────────

print(f'\nLoading AXEC1 baseline (all quality, N={len(dfs["AXEC1"]):,} -> ', end='')
axec1_all_df = load_station('AXEC1', quality_mode='baseline')
print(f'{len(axec1_all_df):,})...')

print('Tracing AXEC1 baseline rays...')
A_km_ax1, A_ax1, ray_dt_ax1, ray_phi_ax1, ray_t_ax1, ray_sta_ax1, n_ax1 = trace_dataframe(
    {'AXEC1': axec1_all_df}, ['AXEC1'])

non_axec1 = ray_sta != 'AXEC1'
A_km2 = sp.vstack([A_km[non_axec1], A_km_ax1]).tocsr()
A2 = sp.vstack([A[non_axec1], A_ax1]).tocsr()
ray_dt2 = np.concatenate([ray_dt[non_axec1], ray_dt_ax1])
ray_phi2 = np.concatenate([ray_phi[non_axec1], ray_phi_ax1])
ray_t2 = np.concatenate([ray_t[non_axec1], ray_t_ax1])
N_RAYS2 = len(ray_dt2)

phi_r2 = np.radians(2. * ray_phi2)
c_data2 = ray_dt2 * np.cos(phi_r2)
s_data2 = ray_dt2 * np.sin(phi_r2)
L_ray2 = np.maximum(np.asarray(A2.sum(axis=1)).ravel(), 1e-9)
u_ray2 = c_data2 / L_ray2
v_ray2 = s_data2 / L_ray2

solve_period_axec1_relaxed = make_solve_period(A2, A_km2, u_ray2, v_ray2)

print('\nRunning back-projections (AXEC1 all-quality variant)...')
results_7_axec1_relaxed = run_periods(periods_7, ray_t2, N_RAYS2, solve_period_axec1_relaxed)
results_a_axec1_relaxed = run_periods(periods_a, ray_t2, N_RAYS2, solve_period_axec1_relaxed)

# ── "Indeterminate quality" variant: ALL 6 stations, -0.2 < Q_w < 0.2, same
# phi_error/dt_error/dt-cutoff thresholds as the standard filter (matches
# rose_plots_lqt_pykonal_unweighted.py's load_station_indeterminate()). This
# is a disjoint subset from the standard quality>QW_MIN build, so every
# station needs fresh tracing (no rays reused). ──────────────────────────────

print(f'\nLoading indeterminate-quality data ({QW_INDETERMINATE_LO} < Q_w < '
      f'{QW_INDETERMINATE_HI}, all 6 stations)...')
dfs_indeterminate = {}
for sta in STATIONS:
    dfi = load_station(sta, quality_mode='indeterminate')
    dfs_indeterminate[sta] = dfi
    print(f'  {sta}: {len(dfi):,}')

print('Tracing indeterminate-quality rays...')
A_km_indet, A_indet, ray_dt_indet, ray_phi_indet, ray_t_indet, ray_sta_indet, N_RAYS_indet = trace_dataframe(
    dfs_indeterminate, STATIONS)

phi_r_indet = np.radians(2. * ray_phi_indet)
c_data_indet = ray_dt_indet * np.cos(phi_r_indet)
s_data_indet = ray_dt_indet * np.sin(phi_r_indet)
L_ray_indet = np.maximum(np.asarray(A_indet.sum(axis=1)).ravel(), 1e-9)
u_ray_indet = c_data_indet / L_ray_indet
v_ray_indet = s_data_indet / L_ray_indet

solve_period_indeterminate = make_solve_period(A_indet, A_km_indet, u_ray_indet, v_ray_indet)

print('\nRunning back-projections (indeterminate-quality variant)...')
results_7_indeterminate = run_periods(periods_7, ray_t_indet, N_RAYS_indet, solve_period_indeterminate)
results_a_indeterminate = run_periods(periods_a, ray_t_indet, N_RAYS_indet, solve_period_indeterminate)

# ── "Q_w > -0.5" variant: ALL 6 stations, same phi_error/dt_error/dt-cutoff
# thresholds as the standard filter, just a looser quality bound (matches
# rose_plots_lqt_pykonal_unweighted.py's load_station_above_neg_half()). ────

print(f'\nLoading Q_w>{QW_ABOVE_NEG_HALF} data (all 6 stations)...')
dfs_above_neg_half = {}
for sta in STATIONS:
    dfa = load_station(sta, quality_mode='above_neg_half')
    dfs_above_neg_half[sta] = dfa
    print(f'  {sta}: {len(dfa):,}')

print('Tracing Q_w>-0.5 rays...')
A_km_anh, A_anh, ray_dt_anh, ray_phi_anh, ray_t_anh, ray_sta_anh, N_RAYS_anh = trace_dataframe(
    dfs_above_neg_half, STATIONS)

phi_r_anh = np.radians(2. * ray_phi_anh)
c_data_anh = ray_dt_anh * np.cos(phi_r_anh)
s_data_anh = ray_dt_anh * np.sin(phi_r_anh)
L_ray_anh = np.maximum(np.asarray(A_anh.sum(axis=1)).ravel(), 1e-9)
u_ray_anh = c_data_anh / L_ray_anh
v_ray_anh = s_data_anh / L_ray_anh

solve_period_above_neg_half = make_solve_period(A_anh, A_km_anh, u_ray_anh, v_ray_anh)

print('\nRunning back-projections (Q_w>-0.5 variant)...')
results_7_above_neg_half = run_periods(periods_7, ray_t_anh, N_RAYS_anh, solve_period_above_neg_half)
results_a_above_neg_half = run_periods(periods_a, ray_t_anh, N_RAYS_anh, solve_period_above_neg_half)

# ── Q_w>-0.5, eruption-focused period binning (3 syn-eruption bins, pre-/
# post-eruption re-binned to match the same per-bin event count) -- reuses
# the already-traced Q_w>-0.5 rays (ray_t_anh/solve_period_above_neg_half),
# just a different period mask, so no re-tracing needed. ────────────────────
ray_t_anh_df = pd.DataFrame({'t': ray_t_anh})
periods_erupt_focused = build_eruption_focused_periods(ray_t_anh_df)
print('\nRunning back-projections (Q_w>-0.5, eruption-focused periods)...')
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


Xg, Yg = np.meshgrid(xn + VXY / 2., yn + VXY / 2., indexing='ij')


# ── Figure builder: percent anisotropy (same look as sws_tomography_backprojection.py) ─

def make_page(results, title, count_min=COUNT_MIN, depth_rows=None):
    """depth_rows: optional override, list of (idx_range, label, z_for_faults)
    triples (see QW_ABOVE_NEG_HALF_DEPTH_ROWS); defaults to the standard
    DEPTH_SLICES/DEPTH_LABELS depth-slice binning."""
    all_a = [r['a'][r['a'] > 0].ravel() * 100. for r in results if np.isfinite(r['a']).any()]
    a_vmax = float(np.nanpercentile(np.concatenate(all_a), 99)) if all_a else 30.
    cmap = plt.colormaps['Blues']
    levels = np.linspace(0., a_vmax, 15)
    n_per = len(results)
    if depth_rows is None:
        depth_rows = [(depth_slice_idx(z0), zlbl, z0) for z0, zlbl in zip(DEPTH_SLICES, DEPTH_LABELS)]
    n_dep = len(depth_rows)
    panel_w = 1.8 if n_per > 7 else 2.2
    fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.5))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.04, wspace=0.04)
    last_h = None
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(depth_rows):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            _bathy(ax)
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2) * 100.
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
        cb.set_label('% shear anisotropy (ΔVs/Vs × 100, back-projection)', fontsize=9)
        cb.ax.tick_params(labelsize=7)
    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='OBS station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.06)
    return fig


def write_pdf(entries, out_path, chunk=7):
    """entries: list of (results_list, title) pairs, or (results_list, title,
    depth_rows) triples to override the depth binning for that page. Each is
    paginated in groups of `chunk` and written as successive pages into ONE
    pdf at out_path."""
    print(f'\nWriting {os.path.basename(out_path)}...')
    with PdfPages(out_path) as pdf:
        for entry in entries:
            results_list, title = entry[0], entry[1]
            depth_rows = entry[2] if len(entry) > 2 else None
            for start in range(0, len(results_list), chunk):
                fig = make_page(results_list[start:start + chunk], title, depth_rows=depth_rows)
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
    print(f'Saved {out_path}')


tag = (f'(quality>{QW_MIN}, phi_err<{PHI_ERR_MAX}°, dt_err<{DT_ERR_MAX}s, '
       f'dt≤{DT_CUTOFF:.2f}s, true PyKonal FMM back-projection -- '
       f'no damping/smoothing, travel-time kernel A_ij=path/Vs, a=ΔVs/Vs=dt/T directly)')

tag_axec1_relaxed = (f'(AXEC1: ALL measurements regardless of quality -- baseline cleanup only, '
                     f'success==True & dt>0, no Q_w/phi_err/dt_err/dt-cutoff threshold; '
                     f'other 5 stations: quality>{QW_MIN}, phi_err<{PHI_ERR_MAX}°, dt_err<{DT_ERR_MAX}s, '
                     f'dt≤{DT_CUTOFF:.2f}s; true PyKonal FMM back-projection -- no damping/smoothing, '
                     f'travel-time kernel A_ij=path/Vs, a=ΔVs/Vs=dt/T directly)')

tag_indeterminate = (f'(ALL 6 stations: {QW_INDETERMINATE_LO}<Q_w<{QW_INDETERMINATE_HI} '
                     f'("indeterminate quality"), phi_err<{PHI_ERR_MAX}°, dt_err<{DT_ERR_MAX}s, '
                     f'dt≤{DT_CUTOFF:.2f}s; true PyKonal FMM back-projection -- no damping/smoothing, '
                     f'travel-time kernel A_ij=path/Vs, a=ΔVs/Vs=dt/T directly)')

tag_above_neg_half = (f'(ALL 6 stations: Q_w>{QW_ABOVE_NEG_HALF}, phi_err<{PHI_ERR_MAX}°, '
                     f'dt_err<{DT_ERR_MAX}s, dt≤{DT_CUTOFF:.2f}s; true PyKonal FMM back-projection -- '
                     f'no damping/smoothing, travel-time kernel A_ij=path/Vs, a=ΔVs/Vs=dt/T directly)')

tag_above_neg_half_depth = tag_above_neg_half.replace(
    ')', ', depth bins: 0-0.5/0.5-1.25/1.25-1.75/>1.75 km)')
tag_above_neg_half_erupt = tag_above_neg_half.replace(
    ')', ', depth bins: 0-0.5/0.5-1.25/1.25-1.75/>1.75 km, eruption-focused periods '
         '(3 syn-eruption bins, pre-/post-eruption re-binned to the same per-bin event count))')

out1 = os.path.join(OUT_DIR, 'lqt_pykonal_tomography_dt_7period_backprojection.pdf')
write_pdf([
    (results_7, f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (7 periods)  {tag}'),
    (results_7_axec1_relaxed,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (7 periods, AXEC1 all-quality)  '
     f'{tag_axec1_relaxed}'),
    (results_7_indeterminate,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (7 periods, indeterminate quality)  '
     f'{tag_indeterminate}'),
    (results_7_above_neg_half,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (7 periods, Q_w>-0.5)  '
     f'{tag_above_neg_half}'),
    (results_7_above_neg_half,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (7 periods, Q_w>-0.5, new depth bins)  '
     f'{tag_above_neg_half_depth}', QW_ABOVE_NEG_HALF_DEPTH_ROWS),
    (results_erupt_focused_above_neg_half,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (Q_w>-0.5, eruption-focused periods, '
     f'new depth bins)  {tag_above_neg_half_erupt}', QW_ABOVE_NEG_HALF_DEPTH_ROWS),
], out1, chunk=7)

out2 = os.path.join(OUT_DIR, 'lqt_pykonal_tomography_dt_annual_backprojection.pdf')
write_pdf([
    (results_a, f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (annual)  {tag}'),
    (results_a_axec1_relaxed,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (annual, AXEC1 all-quality)  '
     f'{tag_axec1_relaxed}'),
    (results_a_indeterminate,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (annual, indeterminate quality)  '
     f'{tag_indeterminate}'),
    (results_a_above_neg_half,
     f'LQT + PyKonal-FMM tomography — % shear anisotropy, back-projection (annual, Q_w>-0.5)  '
     f'{tag_above_neg_half}'),
], out2, chunk=7)

print('\nDone.')
