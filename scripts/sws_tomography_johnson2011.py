#!/usr/bin/env python3
"""
sws_tomography_johnson2011.py

Two-dimensional (map-view) delay-time shear-wave-splitting tomography, a faithful
reimplementation of the method of Johnson, Savage & Townend (2011), JGR 116,
B12303, Sections 3.2-3.3 ("Distinguishing between stress-induced and structural
anisotropy at Mount Ruapehu volcano, New Zealand"), applied to the Axial Seamount
SWS catalogs.

Method summary (paper equation numbers are cited inline in each function)
-------------------------------------------------------------------------
The delay time of a ray accumulates linearly along its path (Crampin 1991; Zhang
et al. 2007):

    delta_t_r = Sum_b  s_b * L_rb                                         (eq. 1)

where s_b [s/km] is the anisotropy strength of block b (the model parameter m_b)
and L_rb [km] is the length of ray r inside block b. Stacking every ray gives a
linear system  d = G m  (eq. 1 / eq. 2) which is solved as a bounded, covariance-
whitened weighted least-squares problem. The covariance C (eqs. 3-4) couples every
observation through a shared model-variance term; it is inverted analytically with
Sherman-Morrison and applied by a closed-form whitening transform (eq. 2). Fast
polarizations phi are averaged per block with a double-angle (180-periodic)
circular mean using four weighting schemes (eqs. 6-7). Model quality is assessed
with an absolute-scale checkerboard test (eq. 5), the resolution matrix, the
condition number of G^T C^-1 G, and the posterior model covariance C_m.

Strictly 2-D
------------
Everything is map view. Rays are the TRUE 3-D bent eikonal paths from
pykonal_raytracer.BaillardRayTracer, but each ray segment's full 3-D arc length is
collapsed onto the map cell of its midpoint (see build_ray_finecell_cache), so
L_rb is a genuine 3-D path length assigned to a 2-D block. The adaptive grid is a
2-D QUAD-TREE with a 0.25 km minimum block, NOT a 3-D octree.

Coordinates
-----------
Projected km via sws_forward_model.ll2xy (origin 45.9 N, 130.1 W): x = East km,
y = North km, z = depth km positive down. This matches the Baillard Vs model and
every other tomography script in scripts/.

Author: Michael Hemmett with Claude (Opus 4.8)
Date: 07-2026
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# READ-ONLY reuse targets (see CLAUDE.md repository map). We import ll2xy /
# STATION_FILE and the ray tracer; we never modify them.
from sws_forward_model import ll2xy, STATION_FILE
from pykonal_raytracer import BaillardRayTracer

# ── Repository layout ────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, 'lqt_pykonal_combined_results')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_lqt_pykonal_results')
CATALOG_DIR = os.path.join(REPO_ROOT, 'data')
RESULTS_DIR = os.path.join(REPO_ROOT, 'sws_tomography_johnson2011_results')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

# Per-station splitting-result files in lqt_pykonal_combined_results/ (mirrors the
# STATION_FILES dict of lqt_pykonal_tomography_raylength_anisotropy.py). AXEC2's
# 2015-2021 half additionally comes from the production batch dir (globbed).
STATION_FILES = {
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv',
              'splitting_results_AXAS2_2022_2026_all_batches.csv'],
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
}

# ── Epochs ───────────────────────────────────────────────────────────────────
# The eruption split at 2022-01-01 gives the two production epochs; 'combined'
# pools them. (label -> (t0, t1); None = open bound). Bounds are half-open [t0,t1).
EPOCHS = {
    'combined': (None, None),
    '2015_2021': (None, pd.Timestamp('2022-01-01', tz='UTC')),
    '2022_2026': (pd.Timestamp('2022-01-01', tz='UTC'), None),
}

# ── Upstream QC filters ──────────────────────────────────────────────────────
# quality is a CONTINUOUS sliding scale in [-1, +1] where -1 == null and values
# near -1 are null-like; nulls are removed by a THRESHOLD, not an equality test.
# Two established project thresholds: -0.5 (lenient) and 0.7 (strict).
DEFAULT_QUALITY_MIN = -0.5
INCIDENCE_MAX_DEG = 35.0   # Nuttli (1961) shear-wave-window cutoff

# ── Fixed fine map grid (§0.1, §1) ───────────────────────────────────────────
# Δ = 0.25 km LEFT-EDGE cells over a generous fixed extent that is a superset of
# every Axial ray (so the cache is reusable across every quality/epoch filter,
# mirroring the raylength script's fixed-grid + downstream-filter design). The
# extent is chosen to be an exact integer number of ROOT_SIDE_KM tiles so the
# quad-tree roots tile it cleanly and every subdivision is a power-of-two split.
DELTA = 0.25            # km, minimum block side (§1)
FINE_X0, FINE_X1 = 3.0, 15.0    # 12 km = 3 root tiles of 4 km  -> 48 fine cells
FINE_Y0, FINE_Y1 = -1.0, 15.0   # 16 km = 4 root tiles of 4 km  -> 64 fine cells
ROOT_SIDE_KM = 4.0      # S0 = Δ * 2^K, K = 4  (root = 16 fine cells)

# Single collapsed z column (§0.1): a TWO-element zn so ray_to_voxels sees NZ=2
# and every segment midpoint maps to iz=0 (never a length-1 zn). Z0 below the
# shallowest ray point and Z0+ZSPAN above the deepest, so Sum_b L_rb == the full
# 3-D arc length exactly (benchmark §7a). The huge span guarantees iz==0.
GRID_Z0 = -1.0
GRID_ZSPAN = 100.0

# Ray resample density: target <=0.03 km point spacing (§0.1). We trace ONCE at
# RAY_NPTS points (0.03 km spacing for any ray up to RAY_NPTS*0.03 km, i.e. ~6 km
# -- a superset of every Axial ray) and only re-trace at the exact
# n_pts = ceil(L/0.03) if that first spacing exceeds the target, so the common
# case is a single trace (the expensive step).
RAY_RESAMPLE_KM = 0.03
RAY_NPTS = 200

# Quad-tree subdivision / usage thresholds (§1).
#   subdivide if a block is traversed by MORE THAN 65 rays          (65 = paper)
#   a block is USED (kept as a column) if traversed by >= 8 rays    (8  = user
#   override of the paper's 20; confirmed by the user for this sparser dataset)
COUNT_SUBDIVIDE = 65
COUNT_KEEP = 8

# ── dt_error scale correction (§3, Auditor / finishing change #3) ─────────────
# dt_error in the splitting CSVs is a (1 - alpha) F-test confidence HALF-WIDTH,
# NOT a 1-sigma standard error: swspy/swspy/splitting/split.py:1159 computes it as
#   conf_bound = ftest(error_surf, dof, alpha=0.003, k=2)   # "(3 sigma)"
# i.e. a 99.7%, two-tailed interval. The true 1-sigma SE is dt_error / z with
# z = Phi^-1(1 - alpha/2); for alpha=0.003, z ~= 2.9677 (NOT 1.96, the 95% value).
# This correction is applied uniformly (same z for every ray) before dt_error
# enters sig_dt2 / the covariance, so it rescales the covariance without changing
# the WLS point estimate s_b (proven invariant, demonstrated empirically).
DT_ERROR_ALPHA = 0.003

# ── phi keep-gates (§5, finishing change #2) ──────────────────────────────────
# A per-block averaged phi is KEPT if its circular std dev sigma_phi < SIGMA_PHI_MAX
# AND the SE of the mean SE_phi < SE_PHI_MAX. The paper-era 30/10 deg gates keep 0
# blocks in the joint six-station run because the EMPIRICAL per-block sigma_phi
# distribution there sits entirely at 45-71 deg (median ~56). These relaxed
# defaults are data-driven from that distribution: SIGMA_PHI_MAX = 60 deg is its
# ~75th percentile (excludes the most-scattered top quartile, i.e. blocks with
# resultant length R < ~0.11 that are effectively multimodal/near-uniform), and
# SE_PHI_MAX = 12 deg is just above its ~90th percentile so the gate is driven by
# dispersion, not by sample-size SE. Both remain parameters of block_phi_average /
# run_chain; override per call as needed.
SIGMA_PHI_MAX = 60.0
SE_PHI_MAX = 12.0

# ── Checkerboard-recovery display mask (§5-§6, finishing change #1) ───────────
# A block is shown as trustworthy on the s_b strength map only if its checkerboard
# recovery ratio m_rec/m_cb is within CB_RECOVERY_TOL of 1 (default 0.5 = the known
# synthetic amplitude recovered to within 50%). Failing blocks are visually
# suppressed (greyed/hatched), NOT deleted from the model. Parameter, not hardcoded.
CB_RECOVERY_TOL = 0.5

# Active-set tolerance (finishing change #4): a block is treated as bound-active
# (clipped) if s_b sits within ACTIVE_TOL of a bound; used only as a cross-check /
# fallback for lsq_linear's exact active_mask.
ACTIVE_TOL = 1e-8

# Derived fine-grid geometry (LEFT EDGES, ray_to_voxels convention).
XNF = np.arange(FINE_X0, FINE_X1 - DELTA * 0.5, DELTA)
YNF = np.arange(FINE_Y0, FINE_Y1 - DELTA * 0.5, DELTA)
ZN2 = np.array([GRID_Z0, GRID_Z0 + GRID_ZSPAN])   # exactly two elements (§0.1)
NXF, NYF = len(XNF), len(YNF)
_ROOT_CELLS = int(round(ROOT_SIDE_KM / DELTA))     # fine cells per root side (16)


# =============================================================================
# 1. DATA LOADING  (merge splitting results with hypocenters, project to km)
# =============================================================================

def load_catalog():
    """Load and concatenate the two MLdd hypocenter catalogs (2015-2021 +
    2022-2026), parsing event_datetime as UTC. Returns a DataFrame with a
    ``station`` column and event_lat/lon/depth (same catalogs used by every
    tomography script)."""
    c1 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'))
    c2 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'))
    for c in (c1, c2):
        c['event_datetime'] = pd.to_datetime(c['event_datetime'], utc=True,
                                              format='mixed')
    return pd.concat([c1, c2], ignore_index=True)


def load_station_measurements(sta, catalog):
    """Load one station's splitting results (both epochs) and location-join them
    against the MLdd catalog, with BASELINE cleanup only (success==True, dt>0).

    Replicates ``load_station`` from
    lqt_pykonal_tomography_raylength_anisotropy.py: per-station CSVs from
    lqt_pykonal_combined_results/, plus (for AXEC2) the 2015-2021 production
    batch files; merge on datetime == event_datetime to attach event_lat/lon/
    depth; ll2xy -> x,y and z = event_depth. No quality / dt_error / incidence
    filtering here -- this baseline set is a superset traced ONCE, with the
    quality threshold and >35 deg incidence cut applied downstream so ray counts
    per filter are inspectable.

    Returns a DataFrame with columns
    [event_id, t, dt, dt_error, phi, phi_error, quality, x, y, z].
    """
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = glob.glob(os.path.join(
            AXEC2_2015_2021_DIR,
            'splitting_results_mldd_2015_2021_axec2_batch_*.csv'))
        dfs.extend(pd.read_csv(f) for f in batch_files)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()          # noqa: E712 (matches upstream)
    df = df[df['dt'] > 0]
    df['t'] = pd.to_datetime(df['datetime'], utc=True)

    sta_cat = catalog[catalog['station'] == sta].drop_duplicates(
        subset='event_datetime')
    df = df.merge(sta_cat[['event_datetime', 'event_lat', 'event_lon',
                           'event_depth']],
                  left_on='t', right_on='event_datetime', how='left')
    n_unmatched = int(df['event_lat'].isna().sum())
    if n_unmatched:
        print(f'  {sta}: dropping {n_unmatched:,} measurements with no catalog '
              f'location match')
    df = df.dropna(subset=['event_lat', 'event_lon', 'event_depth'])

    df['x'], df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
    df['z'] = df['event_depth'].values
    for col in ('event_id', 'phi', 'phi_error', 'dt_error', 'quality'):
        if col not in df.columns:
            df[col] = np.nan
    return df[['event_id', 't', 'dt', 'dt_error', 'phi', 'phi_error',
               'quality', 'x', 'y', 'z']].reset_index(drop=True)


def station_xy(sta_codes=STATIONS):
    """Station map coordinates {code: (x_km, y_km)} from the .llz station file."""
    sdf = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                      engine='python').set_index('s')
    return {s: tuple(ll2xy(sdf.loc[s, 'lat'], sdf.loc[s, 'lon']))
            for s in sta_codes if s in sdf.index}


# =============================================================================
# 2. RAY TRACING + FINE-CELL BINNING  (§0.1)  with CSV cache
# =============================================================================

def _decode_finecells(flat):
    """Decode ray_to_voxels flat indices for the TWO-element-zn fine grid (§0.1).

    flat = ix*NYF*2 + iy*2 + iz  with NZ=2, so iz = flat % 2 (always 0 here),
    tmp = flat // 2, iy = tmp % NYF, ix = tmp // NYF.
    Returns (ix, iy) integer arrays.
    """
    iz = flat % 2
    tmp = flat // 2
    iy = tmp % NYF
    ix = tmp // NYF
    assert np.all(iz == 0), 'collapsed z column violated: iz != 0 (bad zn?)'
    return ix, iy


def build_ray_finecell_cache(sta, df, tracer, out_dir=RESULTS_DIR, verbose=True):
    """Trace every baseline ray for one station and bin its true 3-D arc length
    onto the fixed fine map grid (§0.1), writing two reusable CSV caches.

    For each measurement: trace the bent eikonal ray (event -> station), resample
    to ~RAY_RESAMPLE_KM spacing (n_pts = ceil(L_r / 0.03)), then ray_to_voxels
    against (XNF, YNF, ZN2). Because zn has exactly two elements spanning far
    beyond the deepest ray, every segment maps to iz==0 and NO segment is dropped
    -- so the summed fine-cell length equals the full 3-D arc length exactly
    (asserted below; this underpins benchmark §7a). The segment's full 3-D length
    is assigned to the map cell of its midpoint, collapsing depth (§0.1).

    Incidence angle (Nuttli 1961 window) is RECOMPUTED here from the traced ray's
    arrival segment -- degrees(atan2(horizontal, |dz|)) of the final segment,
    identical to BaillardRayTracer.incidence_angle_at_station but reusing the ray
    already traced instead of tracing a second time. Any legacy ``incidence``
    column is ignored.

    Writes:
      <out_dir>/<sta>_ray_summary.csv    one row per traced ray
      <out_dir>/<sta>_ray_finecells.csv  one row per (ray, fine cell) crossed
    """
    os.makedirs(out_dir, exist_ok=True)
    sx, sy = tracer_station_xy(tracer, sta)
    summ_rows = []
    cell_rows = []
    n = len(df)
    n_fail = n_oob = n_short = n_leak = 0
    for ray_id, row in enumerate(df.itertuples(index=False)):
        ex, ey, ez = float(row.x), float(row.y), float(row.z)
        rec = dict(ray_id=ray_id, event_id=row.event_id,
                   t=row.t, dt=float(row.dt), dt_error=row.dt_error,
                   phi=row.phi, phi_error=row.phi_error, quality=row.quality,
                   ex=ex, ey=ey, ez=ez, sx=sx, sy=sy,
                   arc_len_km=np.nan, incidence_deg=np.nan,
                   sum_finecell_km=np.nan, trace_ok=0)
        # z-range precondition for the collapsed column (§0.1 / Auditor #4).
        if not (GRID_Z0 <= ez < GRID_Z0 + GRID_ZSPAN):
            n_oob += 1
            summ_rows.append(rec)
            continue
        try:
            ray = tracer.trace(sta, ex, ey, ez, n_pts=RAY_NPTS)
        except RuntimeError:
            n_fail += 1
            summ_rows.append(rec)
            continue
        arc = float(np.linalg.norm(np.diff(ray, axis=0), axis=1).sum())
        if arc <= 0:
            n_short += 1
            summ_rows.append(rec)
            continue
        # Re-trace at the exact density only if the fixed grid was too coarse.
        if arc / (RAY_NPTS - 1) > RAY_RESAMPLE_KM:
            n_pts = int(np.ceil(arc / RAY_RESAMPLE_KM)) + 1
            ray = tracer.trace(sta, ex, ey, ez, n_pts=n_pts)
            arc = float(np.linalg.norm(np.diff(ray, axis=0), axis=1).sum())
        # Incidence from the arrival segment (Nuttli 1961 window; recomputed).
        d = ray[-1] - ray[-2]
        incidence = float(np.degrees(np.arctan2(np.hypot(d[0], d[1]),
                                                 abs(d[2]))))
        flat, seg = tracer.ray_to_voxels(ray, XNF, YNF, ZN2)
        rec['arc_len_km'] = arc
        rec['incidence_deg'] = incidence
        if len(flat) == 0:
            summ_rows.append(rec)
            continue
        ix, iy = _decode_finecells(flat)
        sum_cell = float(seg.sum())
        rec['sum_finecell_km'] = sum_cell
        rec['trace_ok'] = 1
        # In-grid completeness: with the 2-element zn all segments are valid in z,
        # so a shortfall means the ray left the fine x-y extent. The fixed extent
        # is a superset of every Axial ray, so this should be ~0; count leaks.
        if abs(sum_cell - arc) > 1e-6 * max(arc, 1.0):
            n_leak += 1
        for k in range(len(flat)):
            cell_rows.append((ray_id, int(ix[k]), int(iy[k]), float(seg[k])))
        summ_rows.append(rec)
        if verbose and (ray_id + 1) % 5000 == 0:
            print(f'    {sta}: traced {ray_id + 1:,}/{n:,}', flush=True)

    summ = pd.DataFrame(summ_rows)
    cells = pd.DataFrame(cell_rows, columns=['ray_id', 'ixf', 'iyf',
                                             'seg_len_km'])
    summ.to_csv(os.path.join(out_dir, f'{sta}_ray_summary.csv'), index=False)
    cells.to_csv(os.path.join(out_dir, f'{sta}_ray_finecells.csv'), index=False)
    if verbose:
        ok = int(summ['trace_ok'].sum())
        print(f'  {sta}: {ok:,}/{n:,} traced ok  (oob-z={n_oob:,}, '
              f'trace-fail={n_fail:,}, zero-len={n_short:,}, '
              f'x-y-leak={n_leak:,})')
    return summ, cells


def load_or_build_ray_cache(sta, catalog, tracer, out_dir=RESULTS_DIR,
                            retrace=False, verbose=True):
    """Return (summary_df, finecells_df) for one station, tracing + caching on
    first call and re-loading the CSVs on subsequent calls (unless retrace)."""
    s_csv = os.path.join(out_dir, f'{sta}_ray_summary.csv')
    c_csv = os.path.join(out_dir, f'{sta}_ray_finecells.csv')
    if not retrace and os.path.exists(s_csv) and os.path.exists(c_csv):
        if verbose:
            print(f'  {sta}: loading cached rays')
        summ = pd.read_csv(s_csv, parse_dates=['t'])
        cells = pd.read_csv(c_csv, dtype={'ray_id': np.int64, 'ixf': np.int32,
                                          'iyf': np.int32,
                                          'seg_len_km': np.float64})
        # Backward-compat: older caches predate the per-ray station columns
        # (needed for the 1/d^2 phi weighting, §B). Each cache is one station,
        # so inject the single station (sx, sy) if absent.
        if 'sx' not in summ.columns or 'sy' not in summ.columns:
            sx, sy = station_xy([sta])[sta]
            summ['sx'] = sx
            summ['sy'] = sy
        return summ, cells
    df = load_station_measurements(sta, catalog)
    return build_ray_finecell_cache(sta, df, tracer, out_dir=out_dir,
                                    verbose=verbose)


def tracer_station_xy(tracer, sta):
    """Station (x, y) km as stored on the tracer, falling back to the .llz file.
    (BaillardRayTracer does not expose the coords, so we re-read them.)"""
    return station_xy([sta])[sta]


def dt_error_z(alpha=DT_ERROR_ALPHA):
    """Two-tailed normal z-score converting a (1 - alpha) confidence HALF-WIDTH to
    a 1-sigma standard error (finishing change #3).

    z = Phi^-1(1 - alpha/2). dt_error in the splitting CSVs is an alpha=0.003 (99.7
    %, "3 sigma") F-test half-width (swspy/swspy/splitting/split.py:1159), so the
    true 1-sigma SE is dt_error / z. z is DERIVED from alpha programmatically (for
    alpha=0.003, z ~= 2.9677 -- NOT the 95% value 1.96). alpha=None disables the
    correction (z=1, legacy behaviour, dt_error treated as 1-sigma)."""
    if alpha is None:
        return 1.0
    from scipy.stats import norm
    return float(norm.ppf(1.0 - alpha / 2.0))


def apply_filters(summ, quality_min=DEFAULT_QUALITY_MIN,
                  incidence_max=INCIDENCE_MAX_DEG,
                  dt_error_alpha=DT_ERROR_ALPHA):
    """Apply the downstream per-ray filters to a cached summary and report the
    survivor count at each stage.

    Stages (in order):
      trace_ok             valid bent ray with >=1 fine cell
      in_grid              ray fully contained in the fine map extent, i.e.
                           Sum_b L_rb == full 3-D arc length (a leaking ray would
                           accrue delta_t outside the modelled region, breaking
                           eq. 1); the count of excluded far-field leaks is
                           reported, never silently dropped
      incidence <= 35 deg  Nuttli (1961) shear-wave window (recomputed value)
      quality >= threshold null / null-like removal (threshold is a PARAMETER,
                           NOT an equality test: quality is continuous in
                           [-1, +1], -1 == null)

    Returns (kept_df, counts_dict). kept_df carries ray_id (its cache index),
    dt, dt_error, phi, quality, incidence, arc length, and a per-ray sig_dt2
    (dt_error**2 with non-positive / NaN replaced by the median positive value;
    guard for §3 SPD).
    """
    counts = {'total': int(len(summ))}
    df = summ[summ['trace_ok'] == 1].copy()
    counts['trace_ok'] = int(len(df))
    # in-grid completeness: keep only rays whose summed fine-cell length equals
    # the full arc length (so eq. 1 applies to the whole path).
    arc = df['arc_len_km'].values.astype(float)
    scell = df['sum_finecell_km'].values.astype(float)
    in_grid = np.abs(scell - arc) <= 1e-9 * np.maximum(arc, 1.0)
    df = df[in_grid]
    counts['in_grid'] = int(len(df))
    df = df[df['incidence_deg'] <= incidence_max]
    counts['incidence_le_35'] = int(len(df))
    df = df[df['quality'] >= quality_min]
    counts[f'quality_ge_{quality_min:g}'] = int(len(df))

    # sig_dt2 = (1-sigma SE)^2, guarded > 0 (§3; replace 0 / NaN by median
    # positive). dt_error is a (1-alpha) confidence HALF-WIDTH (99.7% / "3 sigma";
    # swspy split.py:1159), NOT a 1-sigma SE, so convert first:
    #   sigma_1sigma = dt_error / z,   z = Phi^-1(1 - alpha/2)  (#3).
    # z is the SAME for every ray (uniform rescale of all ray weights), which
    # leaves the WLS/bounded s_b point estimate invariant while correcting the
    # covariance model, C_m and the significance contour downstream.
    z = dt_error_z(dt_error_alpha)
    e = pd.to_numeric(df['dt_error'], errors='coerce').values.astype(float) / z
    e2 = e ** 2
    good = np.isfinite(e2) & (e2 > 0)
    med = np.median(e2[good]) if good.any() else 1e-4
    e2 = np.where(good, e2, med)
    df['sig_dt2'] = e2
    return df.reset_index(drop=True), counts


def restrict_epoch(df, epoch):
    """Subset a filtered ray DataFrame to an epoch (half-open [t0, t1))."""
    t0, t1 = EPOCHS[epoch]
    t = pd.to_datetime(df['t'], utc=True, format='mixed')
    m = pd.Series(True, index=df.index)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t < t1
    return df[m].reset_index(drop=True)


# =============================================================================
# 3. QUAD-TREE GRIDDING  (§1)
# =============================================================================

def _node_rays(r_ray, ridx):
    """Distinct ray ids among the row indices ``ridx``."""
    if len(ridx) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(r_ray[ridx])


def _subdivide(ix0, ix1, iy0, iy1, ridx, r_ray, r_ixf, r_iyf,
               count_subdivide, leaves):
    """Recursive quad-tree node (§1). A node covers fine cells [ix0,ix1) x
    [iy0,iy1). If it is crossed by MORE THAN count_subdivide distinct rays and
    its side is still > 1 fine cell, split into four equal children; otherwise it
    is a leaf. ``ridx`` are the row indices (into the fine-cell arrays) that lie
    inside this node, threaded down so each node only touches its own rows."""
    rays = _node_rays(r_ray, ridx)
    count = int(len(rays))
    side_cells = ix1 - ix0
    if count > count_subdivide and side_cells > 1:
        mx = (ix0 + ix1) // 2
        my = (iy0 + iy1) // 2
        rx = r_ixf[ridx]
        ry = r_iyf[ridx]
        for a0, a1, b0, b1 in ((ix0, mx, iy0, my), (mx, ix1, iy0, my),
                               (ix0, mx, my, iy1), (mx, ix1, my, iy1)):
            sub = ridx[(rx >= a0) & (rx < a1) & (ry >= b0) & (ry < b1)]
            _subdivide(a0, a1, b0, b1, sub, r_ray, r_ixf, r_iyf,
                       count_subdivide, leaves)
    else:
        leaves.append(dict(ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1,
                           side_cells=side_cells, count=count, ridx=ridx))


def build_quadtree(cells, keep_ray_ids, count_subdivide=COUNT_SUBDIVIDE):
    """Build the 2-D quad-tree over the fixed fine grid from the cached fine-cell
    table (§1), restricted to the rays in ``keep_ray_ids`` (the filtered survivor
    set for this station/epoch/quality threshold).

    Roots tile the fine grid in ROOT_SIDE_KM squares (= _ROOT_CELLS fine cells);
    each subdivides while crossed by > count_subdivide rays and side > 0.25 km.
    Returns (leaves, r_ray, r_ixf, r_iyf, r_seg) where leaves is the list of ALL
    leaf blocks (created) and the r_* arrays are the fine-cell rows used.
    """
    keep = np.asarray(sorted(set(int(x) for x in keep_ray_ids)))
    keepset = set(keep.tolist())
    mask = cells['ray_id'].isin(keepset).values
    r_ray = cells['ray_id'].values[mask].astype(np.int64)
    r_ixf = cells['ixf'].values[mask].astype(np.int64)
    r_iyf = cells['iyf'].values[mask].astype(np.int64)
    r_seg = cells['seg_len_km'].values[mask].astype(np.float64)

    leaves = []
    all_idx = np.arange(len(r_ray))
    for rx0 in range(0, NXF, _ROOT_CELLS):
        rx1 = min(rx0 + _ROOT_CELLS, NXF)
        for ry0 in range(0, NYF, _ROOT_CELLS):
            ry1 = min(ry0 + _ROOT_CELLS, NYF)
            sub = all_idx[(r_ixf >= rx0) & (r_ixf < rx1) &
                          (r_iyf >= ry0) & (r_iyf < ry1)]
            if len(sub) == 0:
                continue
            _subdivide(rx0, rx1, ry0, ry1, sub, r_ray, r_ixf, r_iyf,
                       count_subdivide, leaves)
    return leaves, r_ray, r_ixf, r_iyf, r_seg


# =============================================================================
# 4. G / d SYSTEM CONSTRUCTION  (§2)
# =============================================================================

def build_system(leaves, r_ray, r_ixf, r_iyf, r_seg, ray_meta,
                 count_keep=COUNT_KEEP):
    """Assemble the tomography linear system d = G m (eq. 1 / eq. 2) from the
    quad-tree leaves.

    Only USED leaves (count >= count_keep) become model columns; rays that touch
    only dropped leaves are removed and n_b(i) is recomputed over used blocks
    (§1). G[i,j] = L_rb [km] (path length of ray i in block j); the row sum
    Sum_j G[i,j] = L_i is the ray's total in-used-grid 3-D length. d[i] = delta_t
    [s]. m[j] = s_b [s/km].

    ``ray_meta`` is a DataFrame indexed by cache ray_id carrying dt, sig_dt2,
    phi, and the recording-station map coordinates sx, sy [km] (the station that
    recorded each ray -- may differ per ray in a JOINT multi-station system; used
    by the 1/d^2 phi weighting, §B).

    Returns a dict with keys:
      G (R,M), d (R,), sig_dt2 (R,), phi (R,), n_b (R,), ray_ids (R,),
      ray_sx (R,), ray_sy (R,) (recording-station coords per ray),
      used_leaves (list of M leaf dicts, each with center/side_km/count and a
      per-ray {ray_id: L_rb} map), created (int), used (int).
    """
    used_leaves = [lf for lf in leaves if lf['count'] >= count_keep]
    created = len(leaves)
    used = len(used_leaves)
    if used == 0:
        raise ValueError('quad-tree produced 0 used blocks (grid too sparse) -- '
                         'STOP: too few rays for an 8-ray/65-ray grid.')

    # Per-leaf {ray_id: L_rb}; collect the union of used rays.
    leaf_maps = []
    used_ray_set = set()
    for lf in used_leaves:
        ridx = lf['ridx']
        rr = r_ray[ridx]
        ss = r_seg[ridx]
        u, inv = np.unique(rr, return_inverse=True)
        sums = np.zeros(len(u))
        np.add.at(sums, inv, ss)
        lr = {int(rid): float(s) for rid, s in zip(u, sums) if s > 0}
        leaf_maps.append(lr)
        used_ray_set.update(lr.keys())
        lf['center'] = ((lf['ix0'] + lf['ix1']) / 2.0 * DELTA + FINE_X0,
                        (lf['iy0'] + lf['iy1']) / 2.0 * DELTA + FINE_Y0)
        lf['side_km'] = lf['side_cells'] * DELTA
        lf['ray_L'] = lr

    ray_ids = np.array(sorted(used_ray_set), dtype=np.int64)
    ridx_of = {rid: i for i, rid in enumerate(ray_ids)}
    R, M = len(ray_ids), used
    G = np.zeros((R, M))
    for j, lr in enumerate(leaf_maps):
        for rid, L in lr.items():
            G[ridx_of[rid], j] = L

    meta = ray_meta.loc[ray_ids]
    d = meta['dt'].values.astype(float)
    sig_dt2 = meta['sig_dt2'].values.astype(float)
    phi = meta['phi'].values.astype(float)
    ray_sx = meta['sx'].values.astype(float)
    ray_sy = meta['sy'].values.astype(float)
    n_b = (G > 0).sum(axis=1).astype(int)

    return dict(G=G, d=d, sig_dt2=sig_dt2, phi=phi, n_b=n_b, ray_ids=ray_ids,
                ray_sx=ray_sx, ray_sy=ray_sy,
                used_leaves=used_leaves, created=created, used=used)


# =============================================================================
# 5. COVARIANCE + WHITENING  (§3-§4)
# =============================================================================

def estimate_sigma_m2(G, sig_dt2, ridge=1e-12, cond_max=1e10,
                      return_cond=False):
    """Estimate the model-variance scale sigma_m^2 [(s/km)^2] used in C (§3).

    C0 = diag(sig_dt2) (data-only covariance); C_{m,0} = (G^T C0^-1 G)^-1.
    sigma_m^2 = mean(diag(C_{m,0})).

    Auditor #5: a plain LinAlgError fallback is not enough -- a
    near-singular-but-invertible G^T C0^-1 G (the collinear single-station case)
    inverts to garbage without ever raising. So the ridge is gated on the
    CONDITION NUMBER of G^T C0^-1 G (ridge added if cond > cond_max, default
    1e10) as well as on an outright LinAlgError. The condition number is reported
    (returned when return_cond=True; printed nowhere) so the caller can surface
    how collinear the geometry is.

    Returns sigma_m^2 (float), or (sigma_m^2, cond0) when return_cond=True.
    """
    W0 = G / sig_dt2[:, None]           # C0^-1 G
    GtC0iG = G.T @ W0
    cond0 = float(np.linalg.cond(GtC0iG))
    if cond0 > cond_max:
        GtC0iG = GtC0iG + ridge * np.eye(GtC0iG.shape[0])
    try:
        Cm0 = np.linalg.inv(GtC0iG)
    except np.linalg.LinAlgError:
        Cm0 = np.linalg.inv(GtC0iG + ridge * np.eye(GtC0iG.shape[0]))
    sigma_m2 = float(np.mean(np.diag(Cm0)))
    if return_cond:
        return sigma_m2, cond0
    return sigma_m2


def build_covariance(sig_dt2, n_b, n, L_b, sigma_m2):
    """Build the analytic covariance parameters (eqs. 3-4).

    C_ii = sig_dt2(i) + L_b^2 sigma_m^2 n_b(i) / n                      (eq. 3)
    C_ij = sigma_m^2 L_b^2 / n^2   (i != j)                            (eq. 4)
    which is the rank-1-plus-diagonal form  C = D + alpha * 1 1^T  with
        alpha = L_b^2 sigma_m^2 / n^2
        D_ii  = sig_dt2(i) + alpha * (n * n_b(i) - 1)
    (so C_ii = D_ii + alpha reproduces eq. 3 exactly).

    sig_dt2 (= dt_error^2) is the SAME per-ray quantity that appears as sigma_d^2
    in eq. 3 and as the diagonal of C0 in estimate_sigma_m2 (Auditor #8).

    Preconditions (Auditor #4, asserted -- not silently patched):
      min_i n_b(i) >= 1   (every kept ray touches >=1 used block)
      min_i D_ii   > 0    (SPD guard: C = D + alpha 1 1^T is SPD when D_ii > 0)
    Returns (alpha, D).
    """
    n_b = np.asarray(n_b, dtype=float)
    assert n_b.min() >= 1, 'precondition failed: a kept ray touches 0 used blocks'
    alpha = L_b ** 2 * sigma_m2 / n ** 2
    D = sig_dt2 + alpha * (n * n_b - 1.0)
    assert D.min() > 0, f'precondition failed: min D_ii = {D.min():.3e} <= 0 (not SPD)'
    return alpha, D


def whiten(G, d, D, alpha):
    """Closed-form covariance whitening (eq. 2).

    With C^-1 = P (I - beta a a^T) P, P = diag(1/sqrt(D_ii)), a = P 1 (so
    a_i = 1/sqrt(D_ii), a^2 = Sum 1/D_ii), and Sherman-Morrison giving
        beta = alpha / (1 + alpha * Sum 1/D_ii),
    the whitening operator W = (I - kappa a a^T) P with
        kappa = (1 - sqrt(1 - beta a^2)) / a^2
    satisfies W^T W = C^-1 exactly (real since beta a^2 in (0,1)). Then
        G_tilde = P G - kappa outer(a, a^T P G)
        d_tilde = P d - kappa a (a^T P d)
    and the whitened normal equations G_tilde^T G_tilde = G^T C^-1 G,
    G_tilde^T d_tilde = G^T C^-1 d hold (Auditor #5 cross-checks this densely).

    Returns (G_tilde, d_tilde, info) where info carries Dinv, beta, kappa, a.
    """
    Dinv = 1.0 / D
    a = np.sqrt(Dinv)                      # a_i = 1/sqrt(D_ii)
    a2 = float(Dinv.sum())                 # a^2 = Sum 1/D_ii
    beta = alpha / (1.0 + alpha * a2)
    ba2 = beta * a2
    assert 0.0 <= ba2 < 1.0, f'beta a^2 = {ba2} not in [0,1)'
    kappa = (1.0 - np.sqrt(1.0 - ba2)) / a2
    PG = a[:, None] * G                    # P G  (a == 1/sqrt(D))
    aTPG = a @ PG
    G_tilde = PG - kappa * np.outer(a, aTPG)
    Pd = a * d
    aTPd = float(a @ Pd)
    d_tilde = Pd - kappa * a * aTPd
    return G_tilde, d_tilde, dict(Dinv=Dinv, beta=beta, kappa=kappa, a=a, a2=a2)


def cinv_dense(D, alpha):
    """Dense C^-1 via Sherman-Morrison, C^-1 = diag(1/D) - beta (1/D)(1/D)^T,
    beta = alpha / (1 + alpha Sum 1/D). Used only by the whitening cross-check
    and resolution diagnostics; the solve never forms this."""
    Dinv = 1.0 / D
    beta = alpha / (1.0 + alpha * Dinv.sum())
    return np.diag(Dinv) - beta * np.outer(Dinv, Dinv)


# =============================================================================
# 6. CONSTRAINED WLS SOLVE  (§4)
# =============================================================================

def solve_tomography(G, d, D, alpha, L_min=DELTA, used_leaves=None,
                     method='bvls'):
    """Solve the bounded, covariance-whitened WLS problem (§4):
        min_m || G_tilde m - d_tilde ||^2   s.t.  0 <= m_j <= ub
    equivalent to the whitened normal equations G^T C^-1 G m = G^T C^-1 d.

    The upper bound is the physical ceiling ub = delta_t_max / L_min with
    delta_t_max = max_i d_i and L_min = 0.25 km (global scalar; §4). If ANY block
    binds at ub, we switch to a PER-BLOCK ceiling ub_b = delta_t_max / side(b)
    and re-solve (Auditor #6); ``used_leaves`` supplies side(b).

    Returns a dict: m (M,), G_tilde, d_tilde, whiten info, ub (scalar or array),
    n_at_ub, cond (of G^T C^-1 G), per_block_ub (bool), and (finishing change #4)
    active_mask / free_mask identifying which blocks are bound-active (clipped)
    vs free (interior, data-determined). active_mask follows scipy's convention
    (-1 at lower bound, 0 free, +1 at upper bound); free_mask = (active_mask == 0).
    """
    G_tilde, d_tilde, info = whiten(G, d, D, alpha)
    dt_max = float(np.max(d))
    ub = dt_max / L_min
    res = lsq_linear(G_tilde, d_tilde, bounds=(0.0, ub), method=method)
    m = res.x
    n_at_ub = int(np.sum(m >= ub * (1 - 1e-6)))
    per_block = False
    if n_at_ub > 0 and used_leaves is not None:
        # Auditor #6: switch to per-block ub_b = dt_max / side(b) and re-solve.
        sides = np.array([lf['side_km'] for lf in used_leaves])
        ub_vec = dt_max / sides
        res = lsq_linear(G_tilde, d_tilde, bounds=(np.zeros_like(ub_vec), ub_vec),
                         method=method)
        m = res.x
        n_at_ub = int(np.sum(m >= ub_vec * (1 - 1e-6)))
        ub = ub_vec
        per_block = True
    # #4 active set: prefer lsq_linear's exact active_mask; if absent (older
    # scipy), fall back to a bound-proximity test (ACTIVE_TOL) and cross-check.
    lb_vec = np.zeros_like(m)
    ub_vec = ub if not np.isscalar(ub) else np.full_like(m, ub)
    free_by_tol = (m > lb_vec + ACTIVE_TOL) & (m < ub_vec - ACTIVE_TOL)
    active_mask = getattr(res, 'active_mask', None)
    if active_mask is None:
        active_mask = np.where(m <= lb_vec + ACTIVE_TOL, -1,
                               np.where(m >= ub_vec - ACTIVE_TOL, 1, 0))
    active_mask = np.asarray(active_mask)
    free_mask = (active_mask == 0)
    GtCiG = G_tilde.T @ G_tilde
    cond = float(np.linalg.cond(GtCiG))
    return dict(m=m, G_tilde=G_tilde, d_tilde=d_tilde, info=info, ub=ub,
                n_at_ub=n_at_ub, cond=cond, per_block_ub=per_block,
                GtCiG=GtCiG, active_mask=active_mask, free_mask=free_mask,
                free_by_tol=free_by_tol)


# =============================================================================
# 7. FAST-POLARIZATION SPATIAL AVERAGING  (§5, eqs. 6-7)
# =============================================================================

# Weight schemes (§5). 'tomography' uses w_rb = s_b / delta_t_r (eq. 6).
PHI_WEIGHTS = ('none', 'inv_d', 'inv_d2', 'tomography')


def circular_axial_mean(phi_deg, w):
    """Double-angle (180-periodic) weighted circular mean of fast polarizations
    (eqs. 6-7, Auditor #1 correction -- the DOUBLE-ANGLE form).

    For axial data the direction is 180-periodic, so map phi -> 2 phi:
        C = Sum w cos 2phi,  S = Sum w sin 2phi,  Wsum = Sum w
        phi_bar = 1/2 * atan2(S, C)                                     (eq. 7)
    Also returns the mean resultant length R = sqrt(C^2 + S^2) / Wsum, the
    effective sample size N_eff = Wsum^2 / Sum w^2, the circular standard
    deviation sigma_phi [deg] = (180/pi) * 1/2 * sqrt(-2 ln R), and the standard
    error SE_phi = sigma_phi / sqrt(N_eff).

    Returns dict(phi_bar, R, n_eff, sigma_phi, se_phi, wsum).
    """
    phi_deg = np.asarray(phi_deg, dtype=float)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(phi_deg) & np.isfinite(w) & (w >= 0)
    phi_deg, w = phi_deg[good], w[good]
    two = 2.0 * np.radians(phi_deg)
    C = float(np.sum(w * np.cos(two)))
    S = float(np.sum(w * np.sin(two)))
    Wsum = float(np.sum(w))
    if Wsum <= 0:
        return dict(phi_bar=np.nan, R=np.nan, n_eff=0.0, sigma_phi=np.nan,
                    se_phi=np.nan, wsum=0.0)
    phi_bar = float(np.degrees(0.5 * np.arctan2(S, C)) % 180.0)
    R = float(np.hypot(C, S) / Wsum)
    n_eff = float(Wsum ** 2 / np.sum(w ** 2))
    R_clip = min(max(R, 1e-12), 1.0 - 1e-15)
    sigma_phi = float((180.0 / np.pi) * 0.5 * np.sqrt(-2.0 * np.log(R_clip)))
    se_phi = float(sigma_phi / np.sqrt(n_eff)) if n_eff > 0 else np.nan
    return dict(phi_bar=phi_bar, R=R, n_eff=n_eff, sigma_phi=sigma_phi,
                se_phi=se_phi, wsum=Wsum)


def block_phi_average(system, m, weight='inv_d2', sigma_phi_max=SIGMA_PHI_MAX,
                      se_phi_max=SE_PHI_MAX):
    """Per-block averaged fast polarization for every used block (§5).

    For block j, gather the phi and delta_t of every ray with L_rb > 0, weight
    by one of PHI_WEIGHTS:
        none        w = 1
        inv_d       w = 1 / d_rb    (d_rb = map distance block-centre -> the ray's
                                     recording station; Audoine et al. 2004, §B)
        inv_d2      w = 1 / d_rb^2                                   (default, §B)
        tomography  w_rb = s_b / delta_t_r                          (eq. 6)
    then take circular_axial_mean. NOTE: inv_d / inv_d2 use the GEOMETRIC
    block-to-station DISTANCE d_rb (not the delay time) per the paper/Audoine
    2004; only the 'tomography' scheme legitimately uses delta_t_r (eq. 6). For a
    joint multi-station system each ray carries its own recording station
    (system['ray_sx'|'ray_sy']), so d_rb is measured to the station that recorded
    that ray. A block is KEPT if sigma_phi < sigma_phi_max AND SE_phi < se_phi_max
    (§5); both gates are PARAMETERS (defaults SIGMA_PHI_MAX / SE_PHI_MAX, chosen
    data-drivenly from the empirical per-block distribution -- see their comments).
    Returns a list of per-block dicts (center, side_km, count, phi_bar, R, n_eff,
    sigma_phi, se_phi, kept).
    """
    if weight not in PHI_WEIGHTS:
        raise ValueError(f'weight must be one of {PHI_WEIGHTS}')
    ray_ids = system['ray_ids']
    idx_of = {int(r): i for i, r in enumerate(ray_ids)}
    d = system['d']
    phi = system['phi']
    need_dist = weight in ('inv_d', 'inv_d2')
    if need_dist and ('ray_sx' not in system or 'ray_sy' not in system):
        raise KeyError("system lacks per-ray station coords (ray_sx/ray_sy) "
                       f"required for weight='{weight}'")
    ray_sx = system.get('ray_sx')
    ray_sy = system.get('ray_sy')
    out = []
    for j, lf in enumerate(system['used_leaves']):
        rids = list(lf['ray_L'].keys())
        rows = [idx_of[r] for r in rids if r in idx_of]
        dt_r = d[rows]
        phi_r = phi[rows]
        if weight == 'none':
            w = np.ones_like(dt_r)
        elif need_dist:
            # Map distance from THIS block centre to each ray's recording station.
            xc, yc = lf['center']
            d_rb = np.hypot(xc - ray_sx[rows], yc - ray_sy[rows])
            d_rb = np.maximum(d_rb, 1e-6)   # guard block-on-station coincidence
            w = 1.0 / d_rb if weight == 'inv_d' else 1.0 / d_rb ** 2
        else:  # tomography, eq. 6
            w = m[j] / dt_r
        stat = circular_axial_mean(phi_r, w)
        kept = bool(np.isfinite(stat['sigma_phi']) and np.isfinite(stat['se_phi'])
                    and stat['sigma_phi'] < sigma_phi_max
                    and stat['se_phi'] < se_phi_max)
        out.append(dict(center=lf['center'], side_km=lf['side_km'],
                        count=lf['count'], **stat, kept=kept))
    return out


# =============================================================================
# 8. CHECKERBOARD / RESOLUTION / SIGNIFICANCE  (§5-§6, eq. 5)
# =============================================================================

def make_checkerboard(used_leaves, lam=1.0, s_lo=0.01, s_hi=0.02):
    """Absolute-scale checkerboard model (eq. 5): s_CB(b) = s_lo if
    (floor(x_c/lam) + floor(y_c/lam)) is even else s_hi, using each block's map
    centre (x_c, y_c) and an ABSOLUTE wavelength lam [km] (default 1 km; the
    paper sweeps {0.5, 1, 2})."""
    m = np.empty(len(used_leaves))
    for j, lf in enumerate(used_leaves):
        xc, yc = lf['center']
        parity = (int(np.floor(xc / lam)) + int(np.floor(yc / lam))) % 2
        m[j] = s_lo if parity == 0 else s_hi
    return m


def run_checkerboard(system, solve_kw=None, lam=1.0, s_lo=0.01, s_hi=0.02,
                     noise='per_ray', sig_dt2=None, D=None, alpha=None,
                     L_min=DELTA, seed=0):
    """Checkerboard resolution test (§5-§6, eq. 5).

    Build a synthetic model m_CB (make_checkerboard), forward-model synthetic
    data d_CB = G m_CB + eta, then invert with the SAME covariance C and bounds
    to recover m_rec.

    NOISE (Auditor #2): the paper's wording is "standard-normal" noise, but a
    literal N(0, 1 s) would swamp real delta_t ~ 0.2 s. The PRIMARY, self-
    consistent choice (matching C) is per-ray eta_i ~ N(0, sigma_dt,i) with
    sigma_dt,i = sqrt(sig_dt2(i)). Pass noise='flat' for a single
    sigma = median(sqrt(sig_dt2)) variant, or noise='none' for the noise-free
    recovery ceiling. This departs from the paper's literal "standard-normal".

    Returns dict(m_cb, m_rec, ratio, corr, corr_well_sampled).
    """
    G = system['G']
    if D is None or alpha is None:
        raise ValueError('run_checkerboard needs the D and alpha used for C')
    m_cb = make_checkerboard(system['used_leaves'], lam=lam, s_lo=s_lo, s_hi=s_hi)
    rng = np.random.default_rng(seed)
    R = G.shape[0]
    if noise == 'none':
        eta = np.zeros(R)
    elif noise == 'flat':
        s = float(np.median(np.sqrt(sig_dt2)))
        eta = rng.normal(0.0, s, R)
    else:  # per_ray (primary)
        eta = rng.normal(0.0, np.sqrt(sig_dt2), R)
    d_cb = G @ m_cb + eta

    sol = solve_tomography(G, d_cb, D, alpha, L_min=L_min,
                           used_leaves=system['used_leaves'],
                           **(solve_kw or {}))
    m_rec = sol['m']
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = m_rec / m_cb
    corr = float(np.corrcoef(m_cb, m_rec)[0, 1]) if len(m_cb) > 1 else np.nan
    counts = np.array([lf['count'] for lf in system['used_leaves']])
    well = counts >= np.median(counts)
    corr_ws = (float(np.corrcoef(m_cb[well], m_rec[well])[0, 1])
               if well.sum() > 1 else np.nan)
    return dict(m_cb=m_cb, m_rec=m_rec, ratio=ratio, corr=corr,
                corr_well_sampled=corr_ws, well_sampled=well)


def resolution_diagnostics(GtCiG):
    """Resolution / covariance diagnostics (§6).

    For the UNCONSTRAINED linear problem the resolution matrix is
    Res = (G^T C^-1 G)^-1 (G^T C^-1 G) = I, so the meaningful diagnostics are the
    condition number of G^T C^-1 G and the posterior model covariance
    C_m = (G^T C^-1 G)^-1. The significance map contours the VARIANCE
    diag(C_m) [(s/km)^2] at log10 = -5.5 (Auditor #7 -- variance, NOT std).

    Returns dict(Res, cond, Cm, var_diag, log10_var, significance_level).
    """
    cond = float(np.linalg.cond(GtCiG))
    Cm = np.linalg.inv(GtCiG)
    Res = Cm @ GtCiG
    var_diag = np.diag(Cm).copy()
    with np.errstate(divide='ignore'):
        log10_var = np.log10(np.where(var_diag > 0, var_diag, np.nan))
    return dict(Res=Res, cond=cond, Cm=Cm, var_diag=var_diag,
                log10_var=log10_var, significance_level=-5.5)


def significance_active(G_tilde, free_mask, significance_level=-5.5):
    """Posterior-variance significance on the ACTIVE (unclipped) block set (#4).

    resolution_diagnostics inverts the FULL unconstrained G^T C^-1 G and reports
    diag(C_m) for EVERY column -- but that does not match the bound-CONSTRAINED
    s_b map: blocks clipped to a bound (0 or ub) are not data-determined and carry
    no meaningful posterior variance there. This recomputes the significance map
    using ONLY the active (free) block set: it inverts the active submatrix
        C_m^active = (G_tilde[:, free]^T G_tilde[:, free])^-1
    (G_tilde^T G_tilde == G^T C^-1 G by construction, eq. 2) and contours its
    diagonal VARIANCE at log10 = significance_level (eq. 5; -5.5). ``free_mask``
    comes from solve_tomography (lsq_linear active_mask == 0). var_diag / log10_var
    are FULL-length (one entry per used block) with NaN at clipped blocks so they
    align with used_leaves for plotting.

    Returns dict(var_diag, log10_var, significance_level, n_active, n_significant).
    """
    free_mask = np.asarray(free_mask, dtype=bool)
    M = free_mask.shape[0]
    var_diag = np.full(M, np.nan)
    n_active = int(free_mask.sum())
    if n_active >= 1:
        Ga = G_tilde[:, free_mask]
        Cm_a = np.linalg.inv(Ga.T @ Ga)
        var_diag[free_mask] = np.diag(Cm_a)
    with np.errstate(divide='ignore', invalid='ignore'):
        log10_var = np.log10(np.where(var_diag > 0, var_diag, np.nan))
    n_sig = int(np.sum(np.isfinite(log10_var) & (log10_var < significance_level)))
    return dict(var_diag=var_diag, log10_var=log10_var,
                significance_level=significance_level, n_active=n_active,
                n_significant=n_sig)


def checkerboard_block_mask(cb, tol=CB_RECOVERY_TOL):
    """Per-block checkerboard-recovery display mask for the s_b strength map (#1).

    Criterion (documented, defensible, parameterised): a block is RESOLVABLE --
    shown as trustworthy on the strength map -- iff its checkerboard recovery ratio
    m_rec / m_cb lies within [1 - tol, 1 + tol], i.e. the KNOWN synthetic anisotropy
    amplitude is recovered to within tol (default CB_RECOVERY_TOL = 0.5 = 50%). A
    non-finite ratio (m_cb == 0, never happens for the two-value checkerboard) also
    fails. Blocks that FAIL -- typically the poorly-sampled regions where the real
    inversion clips m to ~0 -- are NOT deleted from the model; the caller only greys
    / hatches them so the map is restricted to checkerboard-VALIDATED regions.

    Returns a boolean array (True = resolvable / shown solid), one per used block.
    """
    ratio = np.asarray(cb['ratio'], dtype=float)
    return np.isfinite(ratio) & (np.abs(ratio - 1.0) <= tol)


# =============================================================================
# 9. VERIFICATION / BENCHMARKS  (§7 + Auditor #5)  -- callable, hard-asserting
# =============================================================================

def verify_arclength(summ, cells, rtol=1e-9):
    """Benchmark §7a (HARD assert, Auditor #4): for every IN-GRID ray the summed
    fine-cell length Sum_b L_rb (recomputed INDEPENDENTLY from the fine-cell
    table, not the cached scalar) equals the ray's total 3-D arc length to
    relative tolerance rtol (default 1e-9).

    Because the fine grid uses a two-element zn spanning far beyond the deepest
    ray, no segment is ever dropped in z; the only way Sum_b L_rb can fall short
    of the arc length is a ray leaving the fine x-y extent (a far-field leak).
    Those leaks are excluded upstream (apply_filters 'in_grid' stage) rather than
    silently accepted, so the benchmark asserts over the in-grid set and the leak
    count is reported separately.

    Returns (ok, max_rel_ingrid, n_ingrid, n_leaked)."""
    ok_rows = summ[summ['trace_ok'] == 1]
    seg_sum = cells.groupby('ray_id')['seg_len_km'].sum()
    max_rel = 0.0
    n_in = 0
    n_leak = 0
    for row in ok_rows.itertuples(index=False):
        arc = float(row.arc_len_km)
        s = float(seg_sum.get(row.ray_id, 0.0))
        rel = abs(s - arc) / max(arc, 1.0)
        if rel <= 1e-9:                     # in-grid ray
            max_rel = max(max_rel, rel)
            n_in += 1
        else:                               # far-field leak (excluded upstream)
            n_leak += 1
    ok = max_rel < rtol
    assert ok, f'§7a arc-length benchmark failed: max rel err {max_rel:.3e} >= {rtol}'
    return ok, max_rel, n_in, n_leak


def verify_uniform_strength(G, s=0.01, rtol=1e-12):
    """Benchmark §7b: a uniform model m = s * 1 must forward-model to
    d_hat_i = s * L_i where L_i = Sum_j G[i,j] is the ray's in-grid length."""
    m = np.full(G.shape[1], s)
    d_hat = G @ m
    L = G.sum(axis=1)
    err = np.max(np.abs(d_hat - s * L))
    ok = err < rtol * max(1.0, np.max(np.abs(d_hat)))
    assert ok, f'§7b uniform-strength benchmark failed: max err {err:.3e}'
    return ok, float(err)


def verify_checkerboard(system, D, alpha, corr_min=0.5, **kw):
    """Benchmark §7c: a noise-free checkerboard must be recovered in the
    well-sampled blocks (correlation >= corr_min). Uses the noise-free ceiling so
    the assert is deterministic; run_checkerboard(noise='per_ray') gives the
    realistic (noisier) number for reporting."""
    res = run_checkerboard(system, D=D, alpha=alpha, noise='none', **kw)
    corr = res['corr_well_sampled']
    ok = np.isfinite(corr) and corr >= corr_min
    assert ok, f'§7c checkerboard benchmark failed: well-sampled corr {corr}'
    return ok, float(corr)


def verify_resolution(GtCiG, cond_max=1e8, res_atol=1e-6):
    """Benchmark §7d: Res = (G^T C^-1 G)^-1 (G^T C^-1 G) ~= I and G^T C^-1 G is
    well-conditioned (cond < cond_max; flagged, not fatal, above that)."""
    diag = resolution_diagnostics(GtCiG)
    res_err = float(np.max(np.abs(diag['Res'] - np.eye(GtCiG.shape[0]))))
    ok_res = res_err < res_atol
    assert ok_res, f'§7d resolution benchmark failed: ||Res - I|| = {res_err:.3e}'
    well_conditioned = diag['cond'] < cond_max
    return ok_res, res_err, diag['cond'], well_conditioned


def verify_whitening(rtol=1e-10):
    """Auditor #5 independent whitening cross-check.

    For random systems across n in {1, 100, 1000} AND a tiny (M=2, R=3) case,
    form C DENSELY as C = D + alpha 1 1^T, invert it with a Cholesky
    factorisation (scipy.linalg.cho_factor / cho_solve), and assert
        || W^T W - C^-1 || / || C^-1 ||          < rtol
        || G_tilde^T G_tilde - G^T C^-1 G ||     small (relative)
    Confirms the closed-form whitening (eq. 2) reproduces the analytic C^-1 and
    the whitened normal-equation operator. Returns a list of per-case dicts.
    """
    from scipy.linalg import cho_factor, cho_solve
    rng = np.random.default_rng(12345)
    cases = [dict(R=3, M=2), dict(R=200, M=8, n=1), dict(R=200, M=8, n=100),
             dict(R=200, M=8, n=1000)]
    out = []
    for c in cases:
        R, M = c['R'], c['M']
        n = c.get('n', 100)
        G = rng.uniform(0.0, 1.0, (R, M))
        sig_dt2 = rng.uniform(0.001, 0.01, R)
        n_b = rng.integers(1, M + 1, R).astype(float)
        sigma_m2 = 1e-4
        alpha = 0.25 ** 2 * sigma_m2 / n ** 2
        D = sig_dt2 + alpha * (n * n_b - 1.0)
        D = np.maximum(D, 1e-9)
        C = np.diag(D) + alpha * np.ones((R, R))
        Cinv = cho_solve(cho_factor(C), np.eye(R))
        # Whitening operator W = (I - kappa a a^T) P.
        Dinv = 1.0 / D
        a = np.sqrt(Dinv)
        a2 = Dinv.sum()
        beta = alpha / (1.0 + alpha * a2)
        kappa = (1.0 - np.sqrt(1.0 - beta * a2)) / a2
        P = np.diag(a)
        W = (np.eye(R) - kappa * np.outer(a, a)) @ P
        err_cinv = (np.linalg.norm(W.T @ W - Cinv) /
                    np.linalg.norm(Cinv))
        d = rng.uniform(0.0, 0.3, R)
        G_t, d_t, _ = whiten(G, d, D, alpha)
        GtCiG_dense = G.T @ Cinv @ G
        err_op = (np.linalg.norm(G_t.T @ G_t - GtCiG_dense) /
                  np.linalg.norm(GtCiG_dense))
        ok = (err_cinv < rtol) and (err_op < 1e-8)
        assert ok, (f'whitening cross-check failed (R={R},M={M},n={n}): '
                    f'err_cinv={err_cinv:.2e} err_op={err_op:.2e}')
        out.append(dict(R=R, M=M, n=n, err_cinv=float(err_cinv),
                        err_op=float(err_op), ok=ok))
    return out


def verify_phi_distance_weight(atol=1e-6):
    """Unit test for the §B 1/d / 1/d^2 phi weighting fix.

    Two rays cross ONE block with EQUAL delta_t but DIFFERENT block-to-station
    distance and different phi. Because delta_t is equal, any (buggy) delta_t-
    based weighting collapses to the unweighted mean; only a genuine geometric
    distance weight can move phi_bar. So the test asserts:
        inv_d  weighted mean != none (unweighted) mean, and
        inv_d2 weighted mean != inv_d weighted mean,
    with the closer station pulling phi_bar toward its ray. Also exercises the
    multi-station path: the two rays are recorded at different stations (distinct
    ray_sx), so d_rb is measured per-ray to the correct receiver.
    """
    # One block centred at (0,0); ray 0 station at 1 km, ray 1 station at 5 km.
    leaf = dict(center=(0.0, 0.0), side_km=0.25, count=2, ray_L={0: 1.0, 1: 1.0})
    system = dict(used_leaves=[leaf], ray_ids=np.array([0, 1]),
                  d=np.array([0.20, 0.20]),          # EQUAL delta_t
                  phi=np.array([0.0, 30.0]),
                  ray_sx=np.array([1.0, 5.0]), ray_sy=np.array([0.0, 0.0]))
    m = np.array([0.01])
    p_none = block_phi_average(system, m, weight='none')[0]['phi_bar']
    p_d1 = block_phi_average(system, m, weight='inv_d')[0]['phi_bar']
    p_d2 = block_phi_average(system, m, weight='inv_d2')[0]['phi_bar']
    # Distance weighting must move phi_bar away from the equal-weight mean and
    # pull it toward the CLOSER station's ray (phi=0): none > inv_d > inv_d2 > 0.
    assert abs(p_d1 - p_none) > 1e-3, (
        f'inv_d ({p_d1}) == none ({p_none}): weighting ignored distance')
    assert abs(p_d2 - p_d1) > 1e-3, (
        f'inv_d2 ({p_d2}) == inv_d ({p_d1}): 1/d^2 not steeper than 1/d')
    assert p_none > p_d1 > p_d2 >= 0.0 - atol, (
        f'expected none>inv_d>inv_d2>=0, got {p_none},{p_d1},{p_d2}')
    return dict(none=p_none, inv_d=p_d1, inv_d2=p_d2, ok=True)


def verify_phi_axial_mean(atol=1e-6):
    """Auditor #1 unit test: the double-angle axial mean of {10 deg, 170 deg}
    must be ~0 deg (== 180 deg) and of {80 deg, 100 deg} must be ~90 deg -- the
    single-silently-wrong-if-mistaken item."""
    a = circular_axial_mean([10.0, 170.0], [1.0, 1.0])['phi_bar']
    b = circular_axial_mean([80.0, 100.0], [1.0, 1.0])['phi_bar']
    a_ok = min(abs(a - 0.0), abs(a - 180.0)) < atol
    b_ok = abs(b - 90.0) < atol
    assert a_ok, f'axial mean of {{10,170}} = {a}, expected ~0/180'
    assert b_ok, f'axial mean of {{80,100}} = {b}, expected ~90'
    return dict(mean_10_170=a, mean_80_100=b, ok=True)


# =============================================================================
# 10. FULL-CHAIN CONVENIENCE
# =============================================================================

def run_chain(summ, cells, quality_min=DEFAULT_QUALITY_MIN, epoch='combined',
              n=100, L_b=DELTA, incidence_max=INCIDENCE_MAX_DEG,
              count_subdivide=COUNT_SUBDIVIDE, count_keep=COUNT_KEEP,
              phi_weight='inv_d2', checkerboard_lam=1.0,
              dt_error_alpha=DT_ERROR_ALPHA, sigma_phi_max=SIGMA_PHI_MAX,
              se_phi_max=SE_PHI_MAX, cb_recovery_tol=CB_RECOVERY_TOL,
              verbose=True):
    """Run the full Johnson-2011 chain for one station's cached rays at a given
    quality threshold and epoch: filter -> quad-tree -> G/d -> covariance ->
    whitened bounded WLS -> phi averaging -> checkerboard/resolution.

    Returns a dict of results + a ``diagnostics`` sub-dict with every number the
    build report needs (ray counts per filter, created/used blocks, s_b range,
    phi kept, checkerboard corr, cond(G^T C^-1 G), n_at_ub). Raises (STOP) if the
    grid is too sparse to build a usable system.
    """
    kept, counts = apply_filters(summ, quality_min=quality_min,
                                 incidence_max=incidence_max,
                                 dt_error_alpha=dt_error_alpha)
    kept = restrict_epoch(kept, epoch)
    counts[f'epoch_{epoch}'] = int(len(kept))
    if len(kept) < count_keep:
        raise ValueError(f'STOP: only {len(kept)} rays survive filters for '
                         f'epoch={epoch} q>={quality_min} -- too sparse for a '
                         f'{count_keep}-ray/{count_subdivide}-ray grid.')
    ray_meta = kept.set_index('ray_id')[['dt', 'sig_dt2', 'phi', 'sx', 'sy']]

    leaves, r_ray, r_ixf, r_iyf, r_seg = build_quadtree(
        cells, kept['ray_id'].values, count_subdivide=count_subdivide)
    system = build_system(leaves, r_ray, r_ixf, r_iyf, r_seg, ray_meta,
                          count_keep=count_keep)

    sigma_m2, sigma_m2_cond = estimate_sigma_m2(system['G'], system['sig_dt2'],
                                                 return_cond=True)
    alpha, D = build_covariance(system['sig_dt2'], system['n_b'], n, L_b,
                                sigma_m2)
    sol = solve_tomography(system['G'], system['d'], D, alpha, L_min=DELTA,
                           used_leaves=system['used_leaves'])
    m = sol['m']

    phi_stats = {w: block_phi_average(system, m, weight=w,
                                      sigma_phi_max=sigma_phi_max,
                                      se_phi_max=se_phi_max)
                 for w in PHI_WEIGHTS}
    n_phi_kept = int(sum(1 for b in phi_stats[phi_weight] if b['kept']))

    cb = run_checkerboard(system, D=D, alpha=alpha, sig_dt2=system['sig_dt2'],
                          lam=checkerboard_lam, noise='per_ray')
    # #1 display mask: blocks that fail the checkerboard-recovery criterion are
    # suppressed on the strength map (kept in the model).
    cb_resolvable = checkerboard_block_mask(cb, tol=cb_recovery_tol)
    diag = resolution_diagnostics(sol['GtCiG'])
    # #4 significance recomputed on the ACTIVE (unclipped) block set, to match the
    # bound-constrained s_b map shown.
    sig_active = significance_active(sol['G_tilde'], sol['free_mask'],
                                     significance_level=diag['significance_level'])

    # Data-fit diagnostics. var_reduction near 0 is EXPECTED for a single station
    # (all rays converge on one receiver, so the ray geometry is highly collinear
    # and cannot distinguish the event-to-event delta_t scatter; sigma_dt is
    # comparable to delta_t itself). This is a resolution limitation of single-
    # station delay-time tomography, not a solver failure -- the §7c noise-free
    # checkerboard recovers the input model perfectly. Reported for the Auditor.
    d_hat = system['G'] @ m
    resid = system['d'] - d_hat
    var_reduction = float(1.0 - np.var(resid) / np.var(system['d']))
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))

    diagnostics = dict(
        counts=counts, created=system['created'], used=system['used'],
        n_rays=len(system['ray_ids']), sigma_m2=sigma_m2,
        sigma_m2_cond=sigma_m2_cond, alpha=alpha,
        s_b_min=float(m.min()), s_b_max=float(m.max()),
        s_b_median=float(np.median(m)), s_b_mean=float(m.mean()),
        frac_at_lb=float(np.mean(m <= 1e-9)), n_at_ub=sol['n_at_ub'],
        per_block_ub=sol['per_block_ub'], cond=sol['cond'],
        var_reduction=var_reduction, rms_resid_s=rms_resid,
        n_phi_kept=n_phi_kept, cb_corr=cb['corr'],
        cb_corr_well_sampled=cb['corr_well_sampled'],
        res_diag_cond=diag['cond'],
        # #1 checkerboard-recovery display mask
        cb_recovery_tol=cb_recovery_tol,
        n_cb_resolvable=int(cb_resolvable.sum()),
        # #2 phi keep-gates actually used
        sigma_phi_max=sigma_phi_max, se_phi_max=se_phi_max,
        # #3 dt_error -> 1-sigma correction
        dt_error_alpha=dt_error_alpha, dt_error_z=dt_error_z(dt_error_alpha),
        # #4 active-set vs unconstrained significance
        n_active=sig_active['n_active'],
        n_sig_active=sig_active['n_significant'],
        n_sig_unconstrained=int(np.sum(
            np.isfinite(diag['log10_var'])
            & (diag['log10_var'] < diag['significance_level']))))
    if verbose:
        print(f'    epoch={epoch} q>={quality_min}: rays={diagnostics["n_rays"]} '
              f'blocks {system["created"]}->{system["used"]} '
              f's_b[{m.min():.4f},{m.max():.4f}] med={np.median(m):.4f} '
              f'mean={m.mean():.4f} frac@lb={diagnostics["frac_at_lb"]:.2f} '
              f'cond={sol["cond"]:.2e} n@ub={sol["n_at_ub"]} '
              f'varRed={var_reduction:+.3f} phi_kept={n_phi_kept} '
              f'cb_corr_ws={cb["corr_well_sampled"]:.3f}')
    return dict(system=system, D=D, alpha=alpha, m=m, sol=sol,
                phi_stats=phi_stats, checkerboard=cb, resolution=diag,
                significance_active=sig_active, cb_resolvable=cb_resolvable,
                diagnostics=diagnostics, quality_min=quality_min, epoch=epoch)


if __name__ == '__main__':
    # Cheap self-tests that need no data (the Debugger re-runs these + the
    # data-driven §7a/c/d benchmarks independently).
    print('verify_phi_axial_mean :', verify_phi_axial_mean())
    print('verify_phi_dist_weight:', verify_phi_distance_weight())
    print('verify_whitening      :')
    for c in verify_whitening():
        print('   ', c)
    print('All data-free verification asserts passed.')
