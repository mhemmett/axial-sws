#!/usr/bin/env python3
"""
sws_tomography_resolution_tests.py

Standalone resolution / uncertainty test suite for the SWS tomographic
inversion defined in sws_tomography_inversion.py.

This addresses the audit gap: the main inversion ships *no* checkerboard /
restitution tests, *no* ray-density maps, and *no* demonstration that depth or
temporal structure is actually resolved (rather than manufactured by the
regularisation acting on near-vertical rays from a compact cluster).

Three tests are implemented, all using the SAME real ray geometry (global
sparse matrix A built from PyKonal FMM rays), the SAME 2-component forward
model (c = δt·cos2φ → A·u ;  s = δt·sin2φ → A·v), and the SAME augmented
[A; √λ_d I; √λ_s Lap] LSQR solve as the production inversion:

  TEST 1 — Checkerboard / spike restitution
      Build synthetic (u_true,v_true) input fields (checkerboards at a couple
      of block sizes + a single spike), forward-model them through the GLOBAL
      A, optionally add noise scaled to typical δt/φ errors, invert, recover
      m,θ.  Quantify amplitude recovery, angular error, and — the headline
      number — HORIZONTAL vs VERTICAL smearing length of a point spike.

  TEST 2 — Ray density / coverage diagnostics
      Per-voxel ray count, summed path length, and an azimuthal-coverage
      diagnostic (spread of ray-segment orientations sampling each voxel).
      Anisotropy needs crossing rays at varied azimuths to separate φ from δt
      (Chevrot 2006: φ resolvable only where m is constrained).

  TEST 3 — Per-period geometry test (spatial-vs-temporal ambiguity)
      Period boundaries are event-count quantiles, so each period samples a
      DIFFERENT set of rays.  Run ONE fixed synthetic input through each
      period's ray sub-geometry A[mask] and recover.  Any difference between
      per-period recoveries of the IDENTICAL input is a pure geometry artifact
      — the floor that a real temporal signal must exceed.

GRID / LAPLACIAN / build_augmented below are COPIED from
sws_tomography_inversion.py (kept byte-for-byte where it matters) so that
importing this file does not trigger the main script's multi-minute full run.
If the main script's grid changes, update these constants to match.

Run:
    # quick smoke / development (subsamples events, runs in a few minutes)
    /opt/anaconda3/envs/seismo/bin/python sws_tomography_resolution_tests.py --quick
    # full resolution (set SUBSAMPLE=1 below, or pass --full)
    /opt/anaconda3/envs/seismo/bin/python sws_tomography_resolution_tests.py --full

Outputs (in results/):
    sws_tomography_resolution_checkerboard.pdf
    sws_tomography_ray_density.pdf
    sws_tomography_period_geometry_test.pdf
Quantitative summaries are printed to stdout.
"""

import sys, os, time, argparse, warnings
sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pykonal_raytracer import BaillardRayTracer

# ──────────────────────────────────────────────────────────────────────────────
#  Performance control
# ──────────────────────────────────────────────────────────────────────────────
# SUBSAMPLE = N  → trace every Nth event (across the full concatenated catalogue).
# Default 30 runs the smoke / development pass in a few minutes.
# The FULL-resolution run (matching the production inversion) sets SUBSAMPLE = 1.
SUBSAMPLE = 30

# ──────────────────────────────────────────────────────────────────────────────
#  Paths / constants  (mirror sws_tomography_inversion.py)
# ──────────────────────────────────────────────────────────────────────────────
BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'

OUT_CHK    = BASE + 'sws_tomography_resolution_checkerboard.pdf'
OUT_DENS   = BASE + 'sws_tomography_ray_density.pdf'
OUT_PERIOD = BASE + 'sws_tomography_period_geometry_test.pdf'

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_PER_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_PER_DEG_LAT)

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')
STATIONS    = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
PHI_ERR_MAX = 10.0
DT_ERR_MAX  = 0.01

# ── Voxel grid (mirror of sws_tomography_inversion.py) ─────────────────────────
X_START,X_END = 4.0,12.0; Y_START,Y_END = 0.0,12.0; Z_MAX = 4.0
VXY = 0.30; VZ = 0.25
xn = np.arange(X_START, X_END+VXY*.5, VXY)
yn = np.arange(Y_START, Y_END+VXY*.5, VXY)
zn = np.arange(0.,       Z_MAX+VZ*.5,   VZ)
NX,NY,NZ = len(xn),len(yn),len(zn)
J = NX*NY*NZ

def voxel_idx(ix,iy,iz): return ix*NY*NZ + iy*NZ + iz

# Voxel CENTRE coordinates (xn/yn/zn are lower edges as used by ray_to_voxels)
xc = xn + VXY/2
yc = yn + VXY/2
zc = zn + VZ/2

DEPTH_SLICES = [0.3125, 0.9375, 1.5625, 2.1875]
Z_HALF       = 0.3125
DEPTH_LABELS = ['0.0–0.625 km','0.625–1.25 km','1.25–1.875 km','1.875–2.5 km']
# FIX 7: coverage mask threshold (mirror of sws_tomography_inversion.py). Display
# only voxels whose summed ray path length is at least MIN_PATH_KM. Replaces the
# dimensionally-odd `col_norms_sq >= 10*λ_damp` test.
MIN_PATH_KM  = 0.5   # km of cumulative ray length required per displayed voxel

# ── Inversion parameters (mirror of sws_tomography_inversion.py) ───────────────
SMOOTH_RATIO  = 0.5
LSQR_ITER     = 500
LAMBDA_FIXED  = 0.20
LAMBDA_DAMP   = LAMBDA_FIXED
LAMBDA_SMOOTH = LAMBDA_FIXED * SMOOTH_RATIO
N_RAY         = 50

# ── 3D Laplacian (6-connected) — copied from main script ───────────────────────
def build_laplacian():
    rows,cols_,vals=[],[],[]
    for ix in range(NX):
        for iy in range(NY):
            for iz in range(NZ):
                j=voxel_idx(ix,iy,iz); nb=0
                for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ni,nj,nk=ix+di,iy+dj,iz+dk
                    if 0<=ni<NX and 0<=nj<NY and 0<=nk<NZ:
                        rows.append(j); cols_.append(voxel_idx(ni,nj,nk))
                        vals.append(1.); nb+=1
                rows.append(j); cols_.append(j); vals.append(-nb)
    return sp.csr_matrix((vals,(rows,cols_)),shape=(J,J))

# ── Augmented-system builders — copied from main script ────────────────────────
def build_augmented(A_sub, lam_damp, lam_smooth, LAP2, Id2J):
    G = sp.bmat([[A_sub, None],[None, A_sub]], format='csr')
    G_aug = sp.vstack([G,
                       np.sqrt(lam_damp)   * Id2J,
                       np.sqrt(lam_smooth) * LAP2], format='csr')
    return G, G_aug

def solve_uv(A_sub, c_sub, s_sub, lam_damp, lam_smooth, LAP2, Id2J):
    """Run the production augmented LSQR solve and return (u, v)."""
    d     = np.concatenate([c_sub, s_sub])
    d_aug = np.concatenate([d, np.zeros(4*J)])
    _, G_aug = build_augmented(A_sub, lam_damp, lam_smooth, LAP2, Id2J)
    x = spla.lsqr(G_aug, d_aug, iter_lim=LSQR_ITER, show=False)[0]
    return x[:J], x[J:]

def coverage_mask(A_sub, lam_damp=None):
    """FIX 7: physical path-length coverage mask the production script uses for
    display. Require at least MIN_PATH_KM of cumulative ray path length per voxel.
    A_sub.sum(axis=0) sums the (positive) per-ray path lengths through each voxel.
    Old behaviour (dimensionally odd, kept for reference):
        col_norms_sq = np.asarray(A_sub.power(2).sum(axis=0)).ravel()
        return col_norms_sq >= 10.*lam_damp
    """
    col_path_km = np.asarray(A_sub.sum(axis=0)).ravel()
    return col_path_km >= MIN_PATH_KM

# ──────────────────────────────────────────────────────────────────────────────
#  Data + global ray matrix
# ──────────────────────────────────────────────────────────────────────────────
FILES = {
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv','splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv','splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv','splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv','splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv','splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv','splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

def load_stations():
    _sta = pd.read_csv(STATION_FILE, sep=r'\s+',
                       names=['lon','lat','elev_km','station'],
                       engine='python').set_index('station')
    _sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
    _sta['x'],_sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)
    sta_xy = {s:(float(_sta.loc[s,'x']),float(_sta.loc[s,'y']))
              for s in STATIONS if s in _sta.index}
    return _sta, sta_xy

def load_events():
    dfs={}
    for sta,(f1,f2) in FILES.items():
        def _load(f):
            d = pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
            return d[(d['dt']>0)&(d['phi_error']<PHI_ERR_MAX)&(d['dt_error']<DT_ERR_MAX)]
        df = pd.concat([_load(f1),_load(f2)], ignore_index=True)
        df['x'],df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
        df['z'] = df['event_depth'].values
        df['t'] = pd.to_datetime(df['event_datetime'], utc=True)
        # FIX 1: `phi` is the swspy fast azimuth, ALREADY in degrees clockwise
        # from North. The previous `+90.` rotated every fast direction by 90°.
        # Only wrap into the 0–180° orientation half-circle (matches production).
        df['phi_az'] = df['phi']%180.
        df['sta'] = sta
        dfs[sta]=df
    return dfs

def build_global_A(dfs, sta_xy, subsample):
    """
    Build the global sparse ray matrix A (N_RAYS × J) using the SAME PyKonal FMM
    tracer and ray_to_voxels machinery as the production inversion.  Also return
    per-ray metadata needed for the tests (times, station, back-azimuth, and the
    per-ray voxel cols/vals so we can build azimuthal-coverage diagnostics).
    """
    tracer = BaillardRayTracer(stride=5)
    for sta in STATIONS:
        if sta in sta_xy:
            sx,sy = sta_xy[sta]
            tracer.precompute_station(sta, sx, sy, sz=0.0)

    coo_rows, coo_cols, coo_vals = [], [], []
    ray_t, ray_sta = [], []
    # per-ray dominant horizontal azimuth (deg) of the ray path through the medium
    ray_path_az = []
    # accumulate for azimuthal coverage: for each voxel, list of segment azimuths
    az_rows, az_cols = [], []   # (voxel, segment-azimuth-deg)

    row_idx = 0
    trace_fail = 0
    t0 = time.time()
    # concatenate across stations, then subsample by stride
    all_rows = []
    for sta in STATIONS:
        if sta not in sta_xy: continue
        for _, r in dfs[sta].iterrows():
            all_rows.append((sta, float(r['x']), float(r['y']), float(r['z']), r['t']))
    all_rows = all_rows[::subsample]
    total = len(all_rows)
    print(f'  Tracing {total:,} events (SUBSAMPLE={subsample})...')

    for sta, eq_x, eq_y, eq_z, t in all_rows:
        if eq_z < 0 or eq_z > Z_MAX:
            continue
        try:
            ray_xyz = tracer.trace(sta, eq_x, eq_y, eq_z, n_pts=N_RAY)
        except RuntimeError:
            trace_fail += 1
            continue
        vcols, vvals = tracer.ray_to_voxels(ray_xyz, xn, yn, zn)
        if len(vcols) == 0:
            continue
        coo_rows.extend([row_idx]*len(vcols))
        coo_cols.extend(vcols.tolist())
        coo_vals.extend(vvals.tolist())

        # ray-path horizontal azimuth (source→receiver), for coverage diagnostic
        d_horiz = ray_xyz[-1,:2] - ray_xyz[0,:2]
        az = np.degrees(np.arctan2(d_horiz[0], d_horiz[1])) % 180.  # 0–180 (axial)
        ray_path_az.append(az)
        # assign this azimuth to every voxel the ray touches
        az_rows.extend(vcols.tolist())
        az_cols.extend([az]*len(vcols))

        ray_t.append(t); ray_sta.append(sta)
        row_idx += 1
        if row_idx % 5000 == 0:
            el = time.time()-t0
            print(f'    {row_idx:,} rays  {el:.0f}s', end='\r', flush=True)

    N_RAYS = row_idx
    print(f'\n  Done: {N_RAYS:,} rays, {len(coo_vals):,} non-zeros in {time.time()-t0:.0f}s')
    print(f'  Events skipped due to trace failure: {trace_fail}')
    A = sp.csr_matrix((coo_vals,(coo_rows,coo_cols)), shape=(N_RAYS,J), dtype=np.float64)
    meta = dict(
        ray_t   = np.array(ray_t),
        ray_sta = np.array(ray_sta),
        ray_path_az = np.array(ray_path_az),
        az_rows = np.array(az_rows, dtype=int),
        az_cols = np.array(az_cols, dtype=float),
    )
    return A, meta

# ──────────────────────────────────────────────────────────────────────────────
#  Synthetic input fields
# ──────────────────────────────────────────────────────────────────────────────
def make_checkerboard(block, m_amp=0.05, theta_deg=30., alternate_theta=False):
    """
    Alternating-sign anisotropy-strength checkerboard.
    block : block size in voxels (applied independently in x,y,z).
    Returns (u_true, v_true) flat arrays length J.
    Sign(+/-) of the strength flips per block; θ is fixed (or flips with sign
    if alternate_theta) — a sign flip of m at fixed θ is equivalent to a 90°
    flip in fast-axis, which is the most demanding pattern for SWS recovery.
    """
    ix = np.arange(NX)//block
    iy = np.arange(NY)//block
    iz = np.arange(NZ)//block
    sign = (((ix[:,None,None]+iy[None,:,None]+iz[None,None,:]) % 2)*2 - 1).astype(float)
    m = m_amp*sign
    if alternate_theta:
        th = np.where(sign>0, theta_deg, theta_deg+90.)
        m  = m_amp*np.ones_like(sign)
    else:
        th = theta_deg*np.ones_like(sign)
    th_r = np.radians(2.*th)
    u = (m*np.cos(th_r)).ravel()
    v = (m*np.sin(th_r)).ravel()
    return u, v

def make_spike(ix0, iy0, iz0, m_amp=0.05, theta_deg=30.):
    u = np.zeros(J); v = np.zeros(J)
    j = voxel_idx(ix0,iy0,iz0)
    th_r = np.radians(2.*theta_deg)
    u[j] = m_amp*np.cos(th_r)
    v[j] = m_amp*np.sin(th_r)
    return u, v

def forward(A, u, v, noise_sigma=0.0, rng=None):
    """c = A·u, s = A·v, with optional Gaussian noise (in c/s units = s)."""
    c = A @ u; s = A @ v
    if noise_sigma > 0:
        rng = rng or np.random.default_rng(0)
        c = c + rng.normal(0, noise_sigma, c.shape)
        s = s + rng.normal(0, noise_sigma, s.shape)
    return c, s

def recover_m_theta(u, v):
    m  = np.sqrt(u**2+v**2)
    th = (0.5*np.degrees(np.arctan2(v,u))) % 180.
    return m, th

def ang_diff_deg(a, b):
    """Smallest angular separation of two 0–180° axial directions."""
    d = np.abs(a-b) % 180.
    return np.minimum(d, 180.-d)

# ──────────────────────────────────────────────────────────────────────────────
#  Smearing-length estimation
# ──────────────────────────────────────────────────────────────────────────────
def smearing_lengths(m_rec3d, ix0, iy0, iz0):
    """
    Estimate horizontal (x,y) and vertical (z) smearing of a recovered spike as
    the amplitude-weighted RMS distance from the input voxel, computed along 1-D
    profiles through the spike location.  Returns (Lx, Ly, Lz) in km, plus the
    full-width-at-half-max along each axis (km) as a second tuple.
    """
    def rms_and_fwhm(profile, centres, c0):
        p = np.clip(profile, 0, None)
        if p.sum() <= 0:
            return np.nan, np.nan
        d = centres - c0
        rms = np.sqrt(np.sum(p*d**2)/p.sum())
        # FWHM: extent where profile ≥ half its peak
        half = p.max()/2.
        above = np.where(p >= half)[0]
        if len(above) >= 1:
            fwhm = centres[above[-1]] - centres[above[0]]
        else:
            fwhm = np.nan
        return rms, fwhm
    Lx, Fx = rms_and_fwhm(m_rec3d[:, iy0, iz0], xc, xc[ix0])
    Ly, Fy = rms_and_fwhm(m_rec3d[ix0, :, iz0], yc, yc[iy0])
    Lz, Fz = rms_and_fwhm(m_rec3d[ix0, iy0, :], zc, zc[iz0])
    return (Lx, Ly, Lz), (Fx, Fy, Fz)

# ──────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def depth_slice_idx(z0):
    """FIX 3: select z-voxels whose CENTRE (zn[k]+VZ/2) lies in the band
    [z0-Z_HALF, z0+Z_HALF). Centre-based selection (mirror of production's
    depth_slice_idx) keeps depth panels registered with the centre-shifted
    horizontal field."""
    centres = zn + VZ/2.
    sel = np.where((centres >= z0-Z_HALF) & (centres < z0+Z_HALF))[0]
    if len(sel) == 0:
        k = int(np.argmin(np.abs(centres - z0)))
        return k, k+1
    return int(sel[0]), int(sel[-1])+1

def depth_layer_indices():
    """For each display depth slice, the iz range (centre-based, mirrors main
    script's depth_slice_idx).
    Old edge-based behaviour (kept for reference):
        iz0=max(0,int((z0-Z_HALF)/VZ)); iz1=min(NZ,int((z0+Z_HALF)/VZ)+1)
    """
    return [depth_slice_idx(z0) for z0 in DEPTH_SLICES]

def slice_mean(field3d, iz0, iz1):
    return np.nanmean(field3d[:,:,iz0:iz1], axis=2)

def panel_map(ax, sl, vmin, vmax, cmap, sta=None, extent=None):
    # FIX 3: half-voxel registration. ray_to_voxels treats xn[j]/yn[j] as the
    # LEFT EDGE of voxel j, so the value for voxel j physically sits at its CENTRE
    # xn[j]+VXY/2. With imshow the extent must therefore span the true voxel EDGES
    # ([xn[0], xn[-1]+VXY] etc.) so pixel centres coincide with voxel centres —
    # the imshow equivalent of production's meshgrid(xn+VXY/2, yn+VXY/2). Stations
    # stay in the absolute-km frame (NOT shifted). Display window still cropped to
    # [X_START,X_END]×[Y_START,Y_END] via set_xlim/set_ylim below.
    extent = extent or [xn[0], xn[-1]+VXY, yn[0], yn[-1]+VXY]
    im = ax.imshow(sl.T, origin='lower', extent=extent, aspect='equal',
                   cmap=cmap, vmin=vmin, vmax=vmax)
    if sta is not None:
        for s,row in sta.iterrows():
            ax.plot(row['x'],row['y'],'^',ms=5,mfc='#FFD700',mec='k',mew=0.6,zorder=10)
    ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
    ax.tick_params(labelsize=5)
    return im

# ──────────────────────────────────────────────────────────────────────────────
#  TEST 1 — Checkerboard / spike restitution
# ──────────────────────────────────────────────────────────────────────────────
def test1_checkerboard(A, sta, LAP2, Id2J, noise_sigma):
    print('\n=== TEST 1: Checkerboard / spike restitution ===')
    layers = depth_layer_indices()
    cov = coverage_mask(A, LAMBDA_DAMP).reshape(NX,NY,NZ)

    # Spike location: the single best-covered voxel that lies above the
    # MIN_PATH_KM coverage threshold AND within the displayed depth range, so the
    # spike sits where rays actually sample and the recovery is meaningful.
    col_norms = np.asarray(A.power(2).sum(axis=0)).ravel()
    col_norms_masked = np.where(cov.ravel(), col_norms, -np.inf)
    cn3 = col_norms_masked.reshape(NX,NY,NZ)
    # restrict to depths shown in the production figures (≤ ~2.5 km).
    # FIX 3: centre-based upper bound (exclusive iz1 of the deepest display slice).
    # Old edge-based: iz_max = min(NZ, int((DEPTH_SLICES[-1]+Z_HALF)/VZ)+1)
    iz_max = min(NZ, depth_slice_idx(DEPTH_SLICES[-1])[1])
    cn3[:,:,iz_max:] = -np.inf
    ix0, iy0, iz0 = np.unravel_index(np.argmax(cn3), cn3.shape)
    print(f'  Spike voxel: (ix,iy,iz)=({ix0},{iy0},{iz0}) '
          f'= ({xc[ix0]:.2f},{yc[iy0]:.2f},{zc[iz0]:.2f}) km')

    M_AMP = 0.05; THETA = 30.

    cases = [
        ('Checkerboard 2-voxel', *make_checkerboard(2, M_AMP, THETA)),
        ('Checkerboard 4-voxel', *make_checkerboard(4, M_AMP, THETA)),
        ('Single spike',         *make_spike(ix0, iy0, iz0, M_AMP, THETA)),
    ]

    rng = np.random.default_rng(42)
    results = []
    for name, u_t, v_t in cases:
        c, s = forward(A, u_t, v_t, noise_sigma, rng)
        u_r, v_r = solve_uv(A, c, s, LAMBDA_DAMP, LAMBDA_SMOOTH, LAP2, Id2J)
        m_t, th_t = recover_m_theta(u_t, v_t)
        m_r, th_r = recover_m_theta(u_r, v_r)
        m_t3 = m_t.reshape(NX,NY,NZ); m_r3 = m_r.reshape(NX,NY,NZ)
        th_t3 = th_t.reshape(NX,NY,NZ); th_r3 = th_r.reshape(NX,NY,NZ)
        # mask to covered voxels for metrics
        covf = cov.ravel()
        live = covf & (m_t > 1e-9)
        amp_ratio = np.full(J, np.nan)
        amp_ratio[live] = m_r[live]/m_t[live]
        ang_err = np.full(J, np.nan)
        ang_err[live] = ang_diff_deg(th_r[live], th_t[live])
        med_ratio = np.nanmedian(amp_ratio[live]) if live.any() else np.nan
        med_ang   = np.nanmedian(ang_err[live])   if live.any() else np.nan
        print(f'  [{name}] covered voxels with input={int(live.sum())}  '
              f'median recovered/true m = {med_ratio:.2f}  '
              f'median |Δθ| = {med_ang:.1f}°')
        results.append(dict(name=name, m_t3=m_t3, m_r3=m_r3,
                            th_t3=th_t3, th_r3=th_r3,
                            amp_ratio=med_ratio, ang_err=med_ang,
                            u_r=u_r, v_r=v_r))
        if name == 'Single spike':
            (Lx,Ly,Lz),(Fx,Fy,Fz) = smearing_lengths(m_r3, ix0, iy0, iz0)
            print('  ── SPIKE SMEARING (headline) ──')
            print(f'     RMS smear  Lx={Lx:.2f}  Ly={Ly:.2f} km (horizontal)   '
                  f'Lz={Lz:.2f} km (VERTICAL)')
            print(f'     FWHM       x={Fx:.2f}   y={Fy:.2f} km                  '
                  f'z={Fz:.2f} km')
            print(f'     Vertical voxel size VZ={VZ} km, depth-slice spacing '
                  f'≈{DEPTH_SLICES[1]-DEPTH_SLICES[0]:.3f} km')
            if np.isfinite(Lz):
                indep = Lz < (DEPTH_SLICES[1]-DEPTH_SLICES[0])
                print(f'     → vertical RMS smear {"<" if indep else "≥"} slice '
                      f'spacing: depth slices {"appear independently resolved" if indep else "are SMEARED / not independent"}')
            results[-1]['smear'] = dict(Lx=Lx,Ly=Ly,Lz=Lz,Fx=Fx,Fy=Fy,Fz=Fz,
                                        ix0=ix0,iy0=iy0,iz0=iz0)

    # ── plot ──
    vmax = M_AMP
    cmap = 'RdBu_r'
    with PdfPages(OUT_CHK) as pdf:
        for res in results:
            n_dep = len(layers)
            fig = plt.figure(figsize=(2.2*2+0.6, 2.4*n_dep+0.6))
            gs = GridSpec(n_dep, 3, width_ratios=[1,1,0.06], wspace=0.08, hspace=0.12)
            # signed strength = m * cos(2θ) just for a signed checkerboard look:
            # show recovered/true m (unsigned) which is what the inversion reports
            for ri,(iz0_,iz1_) in enumerate(layers):
                axT = fig.add_subplot(gs[ri,0])
                axR = fig.add_subplot(gs[ri,1])
                slT = slice_mean(res['m_t3'], iz0_, iz1_)
                slR = slice_mean(res['m_r3'], iz0_, iz1_)
                panel_map(axT, slT, 0, vmax, 'magma', sta)
                im = panel_map(axR, slR, 0, vmax, 'magma', sta)
                axT.set_ylabel(DEPTH_LABELS[ri], fontsize=6)
                if ri==0:
                    axT.set_title('INPUT m', fontsize=8, fontweight='bold')
                    axR.set_title('RECOVERED m', fontsize=8, fontweight='bold')
                axT.set_xticklabels([]); axR.set_xticklabels([]); axR.set_yticklabels([])
                cax = fig.add_subplot(gs[ri,2])
                plt.colorbar(im, cax=cax).ax.tick_params(labelsize=5)
            sub = (f'{res["name"]}   median recovered/true m={res["amp_ratio"]:.2f}, '
                   f'median |Δθ|={res["ang_err"]:.1f}°')
            if 'smear' in res:
                sm = res['smear']
                sub += (f'\nspike smear: horiz Lx={sm["Lx"]:.2f}/Ly={sm["Ly"]:.2f} km, '
                        f'VERTICAL Lz={sm["Lz"]:.2f} km (FWHM_z={sm["Fz"]:.2f} km)')
            fig.suptitle(f'Checkerboard / spike restitution — {sub}',
                         fontsize=9, fontweight='bold')
            fig.subplots_adjust(top=0.90)
            pdf.savefig(fig, dpi=200, bbox_inches='tight'); plt.close(fig)

        # dedicated vertical cross-section through the spike (the headline plot)
        spike = next(r for r in results if 'smear' in r)
        sm = spike['smear']
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        # x-z slice at iy0
        ax = axes[0]
        # FIX 3: span true voxel EDGES so imshow pixel centres land on voxel
        # centres in both x ([xn[0],xn[-1]+VXY]) and z ([zn[0],zn[-1]+VZ]); the
        # input-spike marker is already at voxel centres (xc[ix0],zc[iz0]).
        im = ax.imshow(spike['m_r3'][:, sm['iy0'], :].T, origin='lower',
                       extent=[xn[0],xn[-1]+VXY,zn[0],zn[-1]+VZ], aspect='auto',
                       cmap='magma', vmin=0, vmax=vmax)
        ax.plot(xc[sm['ix0']], zc[sm['iz0']], 'c+', ms=14, mew=2)
        ax.invert_yaxis(); ax.set_xlabel('x [km]'); ax.set_ylabel('depth [km]')
        ax.set_title(f'Recovered m, x–z slice at y={yc[sm["iy0"]]:.1f} km\n'
                     f'(+ = input spike)', fontsize=8)
        plt.colorbar(im, ax=ax)
        # vertical profile through spike
        ax = axes[1]
        prof = spike['m_r3'][sm['ix0'], sm['iy0'], :]
        ax.plot(prof, zc, 'o-')
        ax.axhline(zc[sm['iz0']], color='r', ls='--', label='input depth')
        ax.invert_yaxis(); ax.set_xlabel('recovered m [s/km]'); ax.set_ylabel('depth [km]')
        ax.set_title(f'Vertical profile through spike\nRMS smear Lz={sm["Lz"]:.2f} km, '
                     f'FWHM={sm["Fz"]:.2f} km', fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {OUT_CHK}')
    return results

# ──────────────────────────────────────────────────────────────────────────────
#  TEST 2 — Ray density / coverage diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def azimuthal_spread(meta):
    """
    Per-voxel azimuthal-coverage metric: circular spread (on the 0–180° axial
    circle) of the path-azimuths of all rays touching the voxel.  Returns
    R = 1 - mean-resultant-length on the doubled-angle circle, so R∈[0,1];
    R≈0 ⇒ all rays one azimuth (bad for separating φ); R≈1 ⇒ well spread.
    Also returns number of distinct rays per voxel.
    """
    az_rows = meta['az_rows']; az_cols = meta['az_cols']
    # doubled angle so 0° and 180° (same axial dir) coincide
    ang = np.radians(2.*az_cols)
    R = np.full(J, np.nan)
    nray = np.zeros(J, dtype=int)
    order = np.argsort(az_rows)
    rs = az_rows[order]; cs = np.cos(ang)[order]; sn = np.sin(ang)[order]
    # group by voxel
    uniq, start = np.unique(rs, return_index=True)
    start = list(start) + [len(rs)]
    for k, v in enumerate(uniq):
        a,b = start[k], start[k+1]
        n = b-a
        nray[v] = n
        mrl = np.sqrt(cs[a:b].sum()**2 + sn[a:b].sum()**2)/n
        R[v] = 1.0 - mrl
    return R, nray

def test2_ray_density(A, meta, sta):
    print('\n=== TEST 2: Ray density / coverage diagnostics ===')
    ray_count = np.asarray((A > 0).sum(axis=0)).ravel().reshape(NX,NY,NZ)
    path_len  = np.asarray(A.sum(axis=0)).ravel().reshape(NX,NY,NZ)
    az_R, az_nray = azimuthal_spread(meta)
    az_R3 = az_R.reshape(NX,NY,NZ)

    cov = coverage_mask(A, LAMBDA_DAMP)
    covered = cov.sum()
    print(f'  Voxels with ANY ray: {(ray_count>0).sum():,}/{J}')
    print(f'  Voxels above coverage threshold (≥{MIN_PATH_KM} km path): {covered:,}/{J}')
    print(f'  Max ray count in a voxel: {ray_count.max()}   '
          f'median (touched): {np.median(ray_count[ray_count>0]):.0f}')
    # azimuthal coverage stats over covered voxels
    covR = az_R[cov & np.isfinite(az_R)]
    if covR.size:
        poor = (covR < 0.25).sum()
        print(f'  Azimuthal-spread R over covered voxels: '
              f'median={np.median(covR):.2f}  '
              f'frac poorly-sampled (R<0.25)={poor/covR.size:.2f}')
        print(f'    (R≈0 ⇒ rays all one azimuth ⇒ φ poorly separable from δt; '
              f'Chevrot 2006)')

    layers = depth_layer_indices()
    with PdfPages(OUT_DENS) as pdf:
        for field3d, label, cmap, logn in [
            (ray_count, 'Ray count per voxel', 'viridis', True),
            (path_len,  'Summed path length [km]', 'viridis', True),
            (az_R3,     'Azimuthal spread R (1=well-crossed, 0=single az)', 'cividis', False),
        ]:
            n_dep = len(layers)
            fig = plt.figure(figsize=(2.3*n_dep+0.8, 2.6))
            gs = GridSpec(1, n_dep+1, width_ratios=[1]*n_dep+[0.05], wspace=0.1)
            if logn:
                disp = np.where(field3d>0, field3d, np.nan)
                vmax = np.nanpercentile(disp, 99) if np.isfinite(disp).any() else 1
                vmin = 0
            else:
                disp = field3d; vmin, vmax = 0, 1
            for ri,(iz0_,iz1_) in enumerate(layers):
                ax = fig.add_subplot(gs[0,ri])
                sl = np.nanmean(disp[:,:,iz0_:iz1_], axis=2)
                im = panel_map(ax, sl, vmin, vmax, cmap, sta)
                ax.set_title(DEPTH_LABELS[ri], fontsize=6)
                if ri>0: ax.set_yticklabels([])
            cax = fig.add_subplot(gs[0,n_dep])
            plt.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
            fig.suptitle(label, fontsize=9, fontweight='bold')
            fig.subplots_adjust(top=0.82)
            pdf.savefig(fig, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {OUT_DENS}')
    return dict(ray_count=ray_count, path_len=path_len, az_R=az_R3)

# ──────────────────────────────────────────────────────────────────────────────
#  TEST 3 — Per-period geometry test
# ──────────────────────────────────────────────────────────────────────────────
def build_7_periods(ray_t):
    post = np.sort(ray_t[ray_t >= ERUPTION_END])
    n = len(post); bounds=[ERUPTION_END]
    for i in range(1,5):
        bounds.append(post[min(int(round(i*n/5)), n-1)])
    bounds.append(None)
    def fmt(ts): return pd.Timestamp(ts).strftime('%b %Y') if ts is not None else 'present'
    pds=[('Pre',None,ERUPTION_START),('Syn',ERUPTION_START,ERUPTION_END)]
    for i in range(5):
        pds.append((f'{fmt(bounds[i])}–{fmt(bounds[i+1])}', bounds[i], bounds[i+1]))
    return pds

def test3_period_geometry(A, meta, sta, LAP2, Id2J, noise_sigma):
    print('\n=== TEST 3: Per-period geometry test (spatial-vs-temporal) ===')
    ray_t = meta['ray_t']
    periods = build_7_periods(ray_t)

    # ONE fixed synthetic input field for all periods
    u_t, v_t = make_checkerboard(4, m_amp=0.05, theta_deg=30.)
    m_t = np.sqrt(u_t**2+v_t**2).reshape(NX,NY,NZ)

    rng = np.random.default_rng(7)
    recov = []
    for lbl, t0, t1 in periods:
        if t0 is not None:
            mask = ray_t >= t0
        else:
            mask = np.ones(len(ray_t), dtype=bool)
        if t1 is not None:
            mask = mask & (ray_t < t1)
        n = int(mask.sum())
        if n < 50:
            print(f'  {lbl}: only {n} rays — skipping')
            recov.append(dict(label=lbl, n=n, m_r3=np.full((NX,NY,NZ),np.nan), cov=np.zeros(J,bool)))
            continue
        A_sub = A[mask]
        c, s = forward(A_sub, u_t, v_t, noise_sigma, rng)
        u_r, v_r = solve_uv(A_sub, c, s, LAMBDA_DAMP, LAMBDA_SMOOTH, LAP2, Id2J)
        m_r = np.sqrt(u_r**2+v_r**2)
        cov = coverage_mask(A_sub, LAMBDA_DAMP)
        recov.append(dict(label=lbl, n=n, m_r3=m_r.reshape(NX,NY,NZ), cov=cov, m_r=m_r))
        print(f'  {lbl}: {n:,} rays  covered={cov.sum():,} voxels')

    # geometry floor: per-voxel variation across periods of the IDENTICAL input.
    # Measured only among periods that actually COVER a given voxel (data-
    # dominance mask per period), over voxels covered by ≥2 periods.  Using the
    # "covered in ≥2 periods" set rather than "covered in ALL periods" keeps a
    # meaningful floor under aggressive --quick subsampling; at --full nearly
    # all displayed voxels are covered in every period anyway.
    live = [r for r in recov if 'm_r' in r]
    spread = None; n_cov = None
    if len(live) >= 2:
        stack = np.array([r['m_r'] for r in live])          # (P, J)
        covst = np.array([r['cov'] for r in live])          # (P, J) bool
        n_cov = covst.sum(axis=0)                           # periods covering each voxel
        masked = np.where(covst, stack, np.nan)
        spread = np.nanstd(masked, axis=0)                  # inter-period std where covered
        inp = np.sqrt(u_t**2+v_t**2)
        good = (n_cov >= 2) & (inp > 1e-9)
        print('  ── GEOMETRY FLOOR (identical input, different period rays) ──')
        if good.any():
            sp = spread[good]; rel = sp/inp[good]
            print(f'     voxels covered by ≥2 periods (with input): {int(good.sum())}')
            print(f'     inter-period std of recovered m: median={np.median(sp):.4f} s/km, '
                  f'max={np.nanmax(sp):.4f} s/km')
            print(f'     relative to input amp: median={np.median(rel)*100:.1f}%, '
                  f'max={np.nanmax(rel)*100:.1f}%')
            print(f'     → any REAL temporal change in m must exceed ~{np.median(sp):.4f} s/km '
                  f'(median) to be distinguishable from geometry alone')
            rng_field = (np.nanmax(masked[:,good],axis=0)
                         - np.nanmin(masked[:,good],axis=0))
            print(f'     max inter-period range over those voxels: '
                  f'{np.nanmax(rng_field):.4f} s/km')
        else:
            print('     (no voxel is covered by ≥2 periods at this subsampling — '
                  'run --full or a smaller SUBSAMPLE for a meaningful floor)')

    # ── plot: per-period recovery of identical input ──
    layers = depth_layer_indices()
    iz0_, iz1_ = layers[1]   # show one representative mid-depth layer
    vmax = 0.05
    with PdfPages(OUT_PERIOD) as pdf:
        n_per = len(recov)
        fig = plt.figure(figsize=(1.9*(n_per+1)+0.6, 2.6))
        gs = GridSpec(1, n_per+2, width_ratios=[1]+[1]*n_per+[0.05], wspace=0.1)
        ax0 = fig.add_subplot(gs[0,0])
        im = panel_map(ax0, slice_mean(m_t, iz0_, iz1_), 0, vmax, 'magma', sta)
        ax0.set_title('INPUT', fontsize=7, fontweight='bold')
        for ci,r in enumerate(recov):
            ax = fig.add_subplot(gs[0,ci+1])
            sl = slice_mean(r['m_r3'], iz0_, iz1_)
            im = panel_map(ax, sl, 0, vmax, 'magma', sta)
            ax.set_title(f'{r["label"]}\nN={r["n"]:,}', fontsize=5.5)
            ax.set_yticklabels([])
        cax = fig.add_subplot(gs[0,n_per+1])
        plt.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
        fig.suptitle(f'Per-period recovery of IDENTICAL checkerboard input '
                     f'({DEPTH_LABELS[1]}) — differences = geometry artifact',
                     fontsize=8, fontweight='bold')
        fig.subplots_adjust(top=0.80)
        pdf.savefig(fig, dpi=200, bbox_inches='tight'); plt.close(fig)

        # geometry-floor map (inter-period std) per depth layer
        if spread is not None:
            spread_disp = spread.copy()
            spread_disp[n_cov < 2] = np.nan   # only where ≥2 periods sample
            spread3 = spread_disp.reshape(NX,NY,NZ)
            n_dep = len(layers)
            fig = plt.figure(figsize=(2.3*n_dep+0.8, 2.6))
            gs = GridSpec(1, n_dep+1, width_ratios=[1]*n_dep+[0.05], wspace=0.1)
            vmx = np.nanpercentile(spread3, 99) if np.isfinite(spread3).any() else 0.01
            for ri,(a,b) in enumerate(layers):
                ax = fig.add_subplot(gs[0,ri])
                sl = np.nanmean(spread3[:,:,a:b], axis=2)
                im = panel_map(ax, sl, 0, vmx, 'inferno', sta)
                ax.set_title(DEPTH_LABELS[ri], fontsize=6)
                if ri>0: ax.set_yticklabels([])
            cax = fig.add_subplot(gs[0,n_dep])
            plt.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
            fig.suptitle('Geometry floor: inter-period std of recovered m [s/km] '
                         '(identical input)', fontsize=8, fontweight='bold')
            fig.subplots_adjust(top=0.82)
            pdf.savefig(fig, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {OUT_PERIOD}')
    return recov

# ──────────────────────────────────────────────────────────────────────────────
#  Small synthetic A (logic validation without full geometry)
# ──────────────────────────────────────────────────────────────────────────────
def build_fake_A(n_rays=400, seed=0):
    """
    Build a small synthetic ray matrix: near-vertical rays from random shallow
    sources to a handful of surface 'stations', to validate the
    checkerboard/restitution + smearing logic without tracing the full
    catalogue.  Mimics the real geometry's poor vertical resolution.
    """
    rng = np.random.default_rng(seed)
    stations = np.array([[xc[NX//2], yc[NY//2]],
                         [xc[NX//3], yc[NY//3]],
                         [xc[2*NX//3], yc[2*NY//3]]])
    rows, cols, vals = [], [], []
    az_rows, az_cols = [], []
    for r in range(n_rays):
        sx, sy = stations[rng.integers(len(stations))]
        ex = sx + rng.normal(0, 0.6); ey = sy + rng.normal(0, 0.6)
        ez = rng.uniform(0.5, Z_MAX*0.8)
        ray = np.column_stack([np.linspace(ex, sx, 40),
                               np.linspace(ey, sy, 40),
                               np.linspace(ez, 0., 40)])
        vc, vv = ray_to_voxels_local(ray)
        if len(vc)==0: continue
        rows.extend([r]*len(vc)); cols.extend(vc.tolist()); vals.extend(vv.tolist())
        az = np.degrees(np.arctan2(sx-ex, sy-ey)) % 180.
        az_rows.extend(vc.tolist()); az_cols.extend([az]*len(vc))
    A = sp.csr_matrix((vals,(rows,cols)), shape=(max(rows)+1, J))
    meta = dict(ray_t=np.array([]), ray_sta=np.array([]),
                ray_path_az=np.array([]),
                az_rows=np.array(az_rows,int), az_cols=np.array(az_cols,float))
    return A, meta

def ray_to_voxels_local(ray):
    mid = 0.5*(ray[:-1]+ray[1:]); ds = np.linalg.norm(np.diff(ray,axis=0),axis=1)
    ix = np.floor((mid[:,0]-xn[0])/VXY).astype(int)
    iy = np.floor((mid[:,1]-yn[0])/VXY).astype(int)
    iz = np.floor((mid[:,2]-zn[0])/VZ ).astype(int)
    valid=(ix>=0)&(ix<NX)&(iy>=0)&(iy<NY)&(iz>=0)&(iz<NZ)
    fi=(ix*NY*NZ+iy*NZ+iz)[valid]; sl=ds[valid]
    if len(fi)==0: return np.array([],int), np.array([])
    u,inv=np.unique(fi,return_inverse=True); m=np.zeros(len(u)); np.add.at(m,inv,sl)
    return u,m

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    global SUBSAMPLE
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help=f'subsample events (SUBSAMPLE={SUBSAMPLE}) for a fast pass')
    ap.add_argument('--full', action='store_true',
                    help='trace every event (SUBSAMPLE=1) — matches production inversion')
    ap.add_argument('--fake', action='store_true',
                    help='validate test logic on a small synthetic A (no FMM tracing)')
    ap.add_argument('--noise', type=float, default=0.0,
                    help='Gaussian noise sigma added to c/s in seconds '
                         '(e.g. 0.002 ≈ typical δt/φ error level)')
    args = ap.parse_args()

    if args.full:  SUBSAMPLE = 1
    elif args.quick: pass  # keep default
    os.makedirs(BASE, exist_ok=True)

    print('Building 3D Laplacian...')
    LAP  = build_laplacian()
    LAP2 = sp.block_diag([LAP,LAP])
    Id2J = sp.eye(2*J)
    print(f'  Laplacian {LAP.shape}, {LAP.nnz:,} nnz;  J={J} voxels ({NX}×{NY}×{NZ})')

    sta, sta_xy = load_stations()

    if args.fake:
        print('\n[FAKE-A MODE] Validating test logic on a small synthetic ray '
              'matrix (NOT the real geometry).')
        A, meta = build_fake_A(n_rays=600)
        # synthesize plausible times so the period test has something to split
        meta['ray_t'] = np.array(
            pd.date_range('2015-06-01', '2025-12-31', periods=A.shape[0], tz='UTC'))
        print(f'  Fake A: {A.shape[0]} rays × {A.shape[1]} voxels')
    else:
        print(f'\nInitialising PyKonal FMM ray tracer + tracing rays...')
        dfs = load_events()
        A, meta = build_global_A(dfs, sta_xy, SUBSAMPLE)

    noise = args.noise
    test1_checkerboard(A, sta, LAP2, Id2J, noise)
    test2_ray_density(A, meta, sta)
    test3_period_geometry(A, meta, sta, LAP2, Id2J, noise)

    print('\nAll tests complete. PDFs written to:')
    print(f'  {OUT_CHK}')
    print(f'  {OUT_DENS}')
    print(f'  {OUT_PERIOD}')

if __name__ == '__main__':
    main()
