#!/usr/bin/env python3
"""
sws_tomography_inversion.py

Linear SWS tomographic inversion using bent rays through the Baillard Vs model.

Forward model (2-component φ parameterisation):
    c_i = δt_i · cos(2φ_i) = Σ_j  A_ij · u_j        [E component]
    s_i = δt_i · sin(2φ_i) = Σ_j  A_ij · v_j        [N component]

where:
    A_ij  = path length of ray i through voxel j  [km]
    u_j   = m_j · cos(2θ_j)  [s/km]
    v_j   = m_j · sin(2θ_j)  [s/km]

Recovery:
    m_j = √(u_j² + v_j²)          local anisotropy strength  [s/km]
    θ_j = ½ arctan2(v_j, u_j)     local fast direction       [°]

Objective:
    min  ‖[A 0; 0 A]·[u;v] − [c;s]‖²  +  λ_d‖[u;v]‖²  +  λ_s‖Lap·[u;v]‖²

Solved with scipy.sparse.linalg.lsqr per time period.
"""

import sys, warnings
sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, tifffile, time
from PIL import Image as PILImage
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from pykonal_raytracer import BaillardRayTracer

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT_DT  = BASE + 'sws_tomography_dt_7period_phi10_dt001_CORRECTED.pdf'
OUT_PHI = BASE + 'sws_tomography_phi_7period_phi10_dt001_CORRECTED.pdf'

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_PER_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_PER_DEG_LAT)

KIDIWELA = [dict(x=7.57,y=4.55,label='S1'),dict(x=7.53,y=6.60,label='S2')]
ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')
STATIONS    = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET = {'AXAS1':(-0.20,-0.55),'AXAS2':(-0.45,-0.55),
                'AXCC1':(-0.75,0.25),'AXEC1':(0.15,0.30),
                'AXEC2':(0.15,0.05),'AXEC3':(0.30,-0.28)}
# FIX 6: harmonized QC cuts — SINGLE place to change them. The same
# PHI_ERR_MAX/DT_ERR_MAX defaults are used across all four tomography scripts.
PHI_ERR_MAX = 10.0    # keep splits with phi_error < PHI_ERR_MAX  [deg]
DT_ERR_MAX  = 0.01    # keep splits with dt_error  < DT_ERR_MAX   [s]

# ── Voxel grid ─────────────────────────────────────────────────────────────────
X_START,X_END = 4.0,12.0; Y_START,Y_END = 0.0,12.0; Z_MAX = 4.0
VXY = 0.30; VZ = 0.25
xn = np.arange(X_START, X_END+VXY*.5, VXY)
yn = np.arange(Y_START, Y_END+VXY*.5, VXY)
zn = np.arange(0.,       Z_MAX+VZ*.5,   VZ)
NX,NY,NZ = len(xn),len(yn),len(zn)
J = NX*NY*NZ    # total voxels

def voxel_idx(ix,iy,iz): return ix*NY*NZ + iy*NZ + iz

DEPTH_SLICES = [0.3125, 0.9375, 1.5625, 2.1875]
Z_HALF       = 0.3125
DEPTH_LABELS = ['0.0–0.625 km','0.625–1.25 km','1.25–1.875 km','1.875–2.5 km']
COUNT_MIN    = 8
# FIX 7: coverage mask threshold. Display only voxels whose summed ray path length
# is at least MIN_PATH_KM (≈ half a voxel of cumulative ray length). This replaces
# the dimensionally-odd `col_norms_sq >= 10*λ_damp` test (sum of SQUARED segment
# lengths compared against a regularisation weight). Single place to tune coverage.
MIN_PATH_KM  = 0.5   # km of cumulative ray length required per displayed voxel

def depth_slice_idx(z0):
    """FIX 3: select z-voxels whose CENTRE (zn[k]+VZ/2) lies in the band
    [z0-Z_HALF, z0+Z_HALF). Centre-based selection keeps depth panels registered
    consistently with the centre-shifted horizontal field."""
    centres = zn + VZ/2.
    sel = np.where((centres >= z0-Z_HALF) & (centres < z0+Z_HALF))[0]
    if len(sel) == 0:
        k = int(np.argmin(np.abs(centres - z0)))
        return k, k+1
    return int(sel[0]), int(sel[-1])+1

# ── Inversion parameters (λ determined data-adaptively from the scan below) ───
# SMOOTH_RATIO fixes the damping:smoothing coupling a-priori: λ_smooth =
# SMOOTH_RATIO × λ_damp. This is a fixed 1-D assumption, NOT a rigorous choice —
# the more defensible approach is a full 2-D (λ_damp, λ_smooth) grid search (e.g.
# a generalised-cross-validation surface or an L-hypersurface corner). We keep the
# 1-D ratio here (cheaper, and adequate given the two terms are correlated) but the
# ratio is a knob the analyst should justify / sensitivity-test.
SMOOTH_RATIO  = 0.5    # λ_smooth = SMOOTH_RATIO × λ_damp  (FIXED a-priori ratio)
LSQR_ITER     = 500
N_LCURVE      = 50     # number of λ samples in the log-spaced scan
LAM_MIN, LAM_MAX = 1e-7, 1e1

# ── Regularisation-parameter (λ) selection mode ──────────────────────────────
# LAMBDA_MODE routes how λ_damp is chosen (λ_smooth = SMOOTH_RATIO·λ_damp):
#   'auto'        → run the pre-eruption scan and pick λ automatically. The chosen
#                   method (L-curve corner vs discrepancy interpolation vs
#                   over/under-regularised end) is DECIDED BY THE DATA and reported
#                   honestly on the figure. THIS IS THE DEFAULT.
#   'lcurve'      → force the L-curve max-curvature (Hansen) corner, ignoring the
#                   discrepancy target.
#   'discrepancy' → force the discrepancy-principle crossing (needs the target to
#                   lie within the scanned misfit range; falls back to nearest λ).
#   'fixed'       → escape hatch: use LAMBDA_FIXED, do NOT run the scan, and label
#                   the figure honestly as "fixed (not data-selected)". Must be set
#                   deliberately; it is NOT the default and never fabricates a curve.
LAMBDA_MODE   = 'auto'
# Only consulted when LAMBDA_MODE == 'fixed'. Kept documented so a fixed λ can be
# forced for reproducibility of a specific published figure, but the default path
# does NOT touch it.
LAMBDA_FIXED  = 0.20
# Optional λ justification scan (diagnostic only, does NOT change the chosen λ).
# Populate e.g. LAMBDA_SCAN = [0.05, 0.1, 0.2, 0.4, 0.8] to print the pre-eruption
# misfit/‖m‖ for each λ so a hand-picked value can be placed on the scan.
LAMBDA_SCAN   = []

# ── PyKonal FMM ray tracer ─────────────────────────────────────────────────────
# Pre-compute one travel-time field per station (station as FMM source).
# Each subsequent trace() call just follows the gradient — no re-solve needed.
N_RAY = 50   # resampled points per ray

# ── 3D Laplacian (6-connected) ────────────────────────────────────────────────
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

print('Building 3D Laplacian...')
LAP = build_laplacian()
print(f'  Laplacian: {LAP.shape}, {LAP.nnz:,} non-zeros')

# ── Load data ──────────────────────────────────────────────────────────────────
FILES = {
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv','splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv','splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv','splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv','splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv','splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv','splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}
_sta=pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','elev_km','station'],
                 engine='python').set_index('station')
_sta=_sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'],_sta['y']=ll2xy(_sta['lat'].values,_sta['lon'].values)
sta_xy={sta:(float(_sta.loc[sta,'x']),float(_sta.loc[sta,'y'])) for sta in STATIONS if sta in _sta.index}

print(f'Loading events (phi_error < {PHI_ERR_MAX}°, dt_error < {DT_ERR_MAX}s)...')
dfs={}
for sta,(f1,f2) in FILES.items():
    def _load(f):
        d=pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
        return d[(d['dt']>0)&(d['phi_error']<PHI_ERR_MAX)&(d['dt_error']<DT_ERR_MAX)]
    df=pd.concat([_load(f1),_load(f2)],ignore_index=True)
    df['x'],df['y']=ll2xy(df['event_lat'].values,df['event_lon'].values)
    df['z']=df['event_depth'].values
    df['t']=pd.to_datetime(df['event_datetime'],utc=True)
    # FIX 1: `phi` is the swspy `fast` azimuth, ALREADY in degrees clockwise from
    # North (geographic). The previous `+90.` rotated every recovered fast direction
    # by 90°. The only operation needed is to wrap into the 0–180° half-circle of an
    # orientation; wrapping is harmless for cos/sin(2φ) but keeps phi_az interpretable.
    df['phi_az']=df['phi']%180.
    dfs[sta]=df
    print(f'  {sta}: {len(df):,}')
all_df=pd.concat(dfs.values(),ignore_index=True)
print(f'  TOTAL retained: {len(all_df):,}')   # FIX 6: report total N after QC

# ── Time periods ───────────────────────────────────────────────────────────────
def build_annual_periods():
    pds=[('Pre-eruption\n2015',None,ERUPTION_START),
         ('Syn-eruption\n2015',ERUPTION_START,ERUPTION_END),
         ('Post-eruption\n2015',ERUPTION_END,pd.Timestamp('2016-01-01',tz='UTC'))]
    for yr in range(2016,2027):
        t0=pd.Timestamp(f'{yr}-01-01',tz='UTC')
        t1=pd.Timestamp(f'{yr+1}-01-01',tz='UTC') if yr<2026 else None
        pds.append((str(yr),t0,t1))
    return pds

def build_7_periods(all_df):
    post=all_df[all_df['t']>=ERUPTION_END].sort_values('t').reset_index(drop=True)
    n=len(post); bounds=[ERUPTION_END]
    for i in range(1,5):
        bounds.append(post['t'].iloc[min(int(round(i*n/5)),n-1)])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds=[('Pre-eruption',None,ERUPTION_START),('Syn-eruption',ERUPTION_START,ERUPTION_END)]
    for i in range(5): pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}',bounds[i],bounds[i+1]))
    return pds

def subset(df,t0,t1):
    m=(df['t']>=t0) if t0 is not None else pd.Series(True,index=df.index)
    if t1 is not None: m=m&(df['t']<t1)
    return df[m]

periods=build_7_periods(all_df)

# ── Build global sparse ray matrix via PyKonal FMM ─────────────────────────────
print(f'\nInitialising PyKonal FMM ray tracer...')
tracer = BaillardRayTracer(stride=5)   # 0.25 km node spacing

print('Pre-computing per-station travel-time fields (FMM)...')
for sta in STATIONS:
    if sta in sta_xy:
        sx, sy = sta_xy[sta]
        tracer.precompute_station(sta, sx, sy, sz=0.0)

print(f'\nTracing {len(all_df):,} events through Baillard Vs model (FMM rays)...')
t0_rt = time.time()

coo_rows, coo_cols, coo_vals = [], [], []
ray_dt      = []
ray_phi     = []
ray_t       = []
ray_dt_err  = []
ray_phi_err = []

row_idx = 0; done = 0; trace_fail = 0
total = sum(len(dfs[sta]) for sta in STATIONS if sta in sta_xy)

for sta in STATIONS:
    if sta not in sta_xy: continue
    for _, row in dfs[sta].iterrows():
        eq_x, eq_y, eq_z = float(row['x']), float(row['y']), float(row['z'])
        if eq_z < 0 or eq_z > Z_MAX: done += 1; continue
        # True eikonal ray via PyKonal
        try:
            ray_xyz = tracer.trace(sta, eq_x, eq_y, eq_z, n_pts=N_RAY)
        except RuntimeError:
            trace_fail += 1; done += 1; continue
        vcols, vvals = tracer.ray_to_voxels(ray_xyz, xn, yn, zn)
        if len(vcols) == 0: done += 1; continue
        coo_rows.extend([row_idx]*len(vcols))
        coo_cols.extend(vcols.tolist())
        coo_vals.extend(vvals.tolist())
        ray_dt.append(float(row['dt']))
        ray_phi.append(float(row['phi_az']))
        ray_t.append(row['t'])
        ray_dt_err.append(float(row.get('dt_error', 0.0)))
        ray_phi_err.append(float(row.get('phi_error', 0.0)))
        row_idx += 1; done += 1
        if done % 10000 == 0:
            el = time.time()-t0_rt
            print(f'  {done:,}/{total:,}  {el:.0f}s  ~{el/done*(total-done):.0f}s remaining',
                  end='\r', flush=True)

N_RAYS = row_idx
print(f'\nDone in {time.time()-t0_rt:.0f}s  —  {N_RAYS:,} valid rays, {len(coo_vals):,} non-zeros')
print(f'  Events skipped due to trace failure: {trace_fail:,}')

# CSR sparse matrix A (N_RAYS × J)
A = sp.csr_matrix((coo_vals,(coo_rows,coo_cols)),shape=(N_RAYS,J),dtype=np.float64)
ray_dt      = np.array(ray_dt)
ray_phi     = np.array(ray_phi)       # degrees
ray_t       = np.array(ray_t)
ray_dt_err  = np.array(ray_dt_err)   # s
ray_phi_err = np.array(ray_phi_err)  # degrees

# 2-component data
phi_r = np.radians(2.*ray_phi)
c_data = ray_dt * np.cos(phi_r)   # δt·cos(2φ)
s_data = ray_dt * np.sin(phi_r)   # δt·sin(2φ)

# ── Build regularisation blocks ────────────────────────────────────────────────
# Full system: G = block_diag(A,A)  size 2N × 2J
# Augmented:   [G; √λ_d I_{2J}; √λ_s Lap_{2J}]
LAP2  = sp.block_diag([LAP,LAP])   # 2J × 2J
Id2J  = sp.eye(2*J)

def build_augmented(A_sub, lam_damp, lam_smooth):
    G = sp.bmat([[A_sub, None],[None, A_sub]], format='csr')
    G_aug = sp.vstack([G,
                       np.sqrt(lam_damp)   * Id2J,
                       np.sqrt(lam_smooth) * LAP2], format='csr')
    return G, G_aug

def solve_system(G_aug, d, d_aug):
    result = spla.lsqr(G_aug, d_aug, iter_lim=LSQR_ITER, show=False)
    x = result[0]
    misfit    = float(np.linalg.norm(G_aug[:len(d)] @ x - d))
    mod_norm  = float(np.linalg.norm(x))
    return x, misfit, mod_norm

def solve_period(mask, label, lam_damp, lam_smooth):
    A_sub = A[mask]
    c_sub = c_data[mask]; s_sub = s_data[mask]
    n_sub = int(mask.sum())
    if n_sub < COUNT_MIN*10:
        print(f'  {label}: only {n_sub} rays — skipping')
        return np.zeros(J), np.zeros(J), np.zeros(J, dtype=bool)
    # FIX 7: physically interpretable coverage mask.
    # Old behaviour (dimensionally odd) was:
    #   col_norms_sq = np.asarray(A_sub.power(2).sum(axis=0)).ravel()
    #   cov_mask = col_norms_sq >= 10. * lam_damp
    # New: require at least MIN_PATH_KM of cumulative ray path length per voxel.
    # A_sub.sum(axis=0) sums the (positive) per-ray path lengths through each voxel.
    col_path_km = np.asarray(A_sub.sum(axis=0)).ravel()
    cov_mask = col_path_km >= MIN_PATH_KM
    d     = np.concatenate([c_sub, s_sub])
    d_aug = np.concatenate([d, np.zeros(4*J)])
    G, G_aug = build_augmented(A_sub, lam_damp, lam_smooth)
    t_s = time.time()
    x, misfit, mnorm = solve_system(G_aug, d, d_aug)
    print(f'  {label}: {n_sub:,} rays  {time.time()-t_s:.0f}s  '
          f'misfit={misfit:.3f}  ‖m‖={mnorm:.3f}  '
          f'covered={cov_mask.sum():,}/{J} voxels')
    return x[:J], x[J:], cov_mask

# ── λ selection from the pre-eruption regularisation curve ────────────────────
def lcurve_corner(lambdas, misfits, mod_norms):
    """Hansen max-curvature corner of the log-log L-curve.

    Returns the index of the maximum-curvature point of the (log‖Gx−d‖,
    log‖x‖) trade-off curve, computed on axis-normalised coordinates so the
    curvature is scale-invariant. Guards against zero-range (a flat/degenerate
    curve) and skips the under-regularised flat foot of the L where the misfit
    barely moves (that region has spurious high curvature from noise).

    Assumes `lambdas` is sorted ascending, so misfit increases and model norm
    decreases with index.
    """
    n = len(misfits)
    if n < 3:                      # not enough points for a curvature estimate
        return int(np.argmin(mod_norms))
    log_rho = np.log10(misfits)
    log_eta = np.log10(mod_norms + 1e-30)
    rng_rho = log_rho.max() - log_rho.min()
    rng_eta = log_eta.max() - log_eta.min()
    if rng_rho < 1e-12 or rng_eta < 1e-12:   # degenerate: no real trade-off
        return int(np.argmin(mod_norms))
    lr_n = (log_rho - log_rho.min()) / rng_rho
    le_n = (log_eta - log_eta.min()) / rng_eta
    dr, de   = np.gradient(lr_n), np.gradient(le_n)
    ddr, dde = np.gradient(dr),   np.gradient(de)
    curv = np.abs(dr * dde - de * ddr) / (dr**2 + de**2 + 1e-30)**1.5
    # Skip the flat under-regularised foot: first index where the misfit has
    # risen appreciably from its minimum. `flat_end` is clamped to a valid,
    # non-degenerate window so the curvature argmax always has ≥1 candidate.
    span = misfits[-1] - misfits[0]
    if span > 0:
        moved = np.abs(misfits - misfits[0]) > 1e-4 * span
        flat_end = int(np.argmax(moved)) if moved.any() else 0
    else:
        flat_end = 0
    flat_end = min(max(flat_end, n // 4), n - 2)   # keep window ≥1 pt, in range
    return flat_end + int(np.argmax(curv[flat_end:]))


def discrepancy_index(lambdas, misfits, target):
    """Discrepancy-principle λ: pick λ where misfit(λ) first rises through
    `target` (the expected noise norm). Returns (idx, interp_lambda) where idx
    is the sample just below the crossing and interp_lambda is the log-linear
    interpolant in λ. Assumes ascending λ (ascending misfit). Falls back to the
    nearest-misfit sample when there is no sign change, and is index-safe at the
    array end.
    """
    diff = misfits - target
    crosses = np.where(np.diff(np.sign(diff)))[0]
    if len(crosses):
        k = int(crosses[0])                       # guaranteed k+1 in range
        denom = misfits[k+1] - misfits[k]
        frac = (target - misfits[k]) / denom if abs(denom) > 1e-30 else 0.0
        frac = float(np.clip(frac, 0.0, 1.0))
        lam = float(lambdas[k] * (lambdas[k+1] / lambdas[k])**frac)
        return k, lam
    k = int(np.argmin(np.abs(diff)))              # no crossing → nearest sample
    return k, float(lambdas[k])


def select_lambda(lambdas, misfits, mod_norms, target, mode):
    """Route λ selection and report the branch that actually fired.

    Returns (lam_damp, marker_idx, method_label) where marker_idx points at the
    scan sample to star on the figure and method_label is the HONEST name of the
    method used. `lam_damp` may lie between samples (discrepancy interpolation);
    marker_idx is then the sample just below it.
    """
    misfits   = np.asarray(misfits, float)
    mod_norms = np.asarray(mod_norms, float)
    if mode == 'lcurve':
        idx = lcurve_corner(lambdas, misfits, mod_norms)
        return float(lambdas[idx]), idx, 'L-curve corner'
    if mode == 'discrepancy':
        idx, lam = discrepancy_index(lambdas, misfits, target)
        return lam, idx, 'Discrepancy principle'
    # 'auto': let the data decide which branch is appropriate.
    if target < misfits[0]:
        # Target unreachable (over-optimistic noise estimate): fall back to the
        # geometric L-curve corner rather than the noisiest end.
        idx = lcurve_corner(lambdas, misfits, mod_norms)
        return float(lambdas[idx]), idx, 'L-curve corner'
    if target > misfits[-1]:
        # Even the most-damped model fits within noise: take the largest λ.
        idx = len(lambdas) - 1
        return float(lambdas[idx]), idx, 'Discrepancy (max λ; all models fit noise)'
    idx, lam = discrepancy_index(lambdas, misfits, target)
    return lam, idx, 'Discrepancy principle'


print('\nRegularisation-parameter (λ) selection...')
print(f'  mode = {LAMBDA_MODE!r}  (SMOOTH_RATIO={SMOOTH_RATIO}, so '
      f'λ_smooth = {SMOOTH_RATIO}·λ_damp)')

pre_mask = (ray_t < ERUPTION_START)
A_pre    = A[pre_mask]
c_pre    = c_data[pre_mask]; s_pre = s_data[pre_mask]
d_pre    = np.concatenate([c_pre, s_pre])
d_aug_pre = np.concatenate([d_pre, np.zeros(4*J)])

# Discrepancy target: expected data-misfit norm from measurement noise on the
# pre-eruption picks. Var(δt·cos2φ) etc. → σ² = σ_δt² + 4·σ_φ²·δt² per datum,
# summed over both components (identical noise on the s-component), so the total
# target is √(Σ (σ_δt² + 4 σ_φ² δt²)).  (Independent of λ.)
_pe          = pre_mask
_phi_err_rad = np.radians(ray_phi_err[_pe])
_noise_var   = ray_dt_err[_pe]**2 + 4.*_phi_err_rad**2 * ray_dt[_pe]**2
target_residual = float(np.sqrt(_noise_var.sum()))

# Optional diagnostic scan (does NOT change the chosen λ). Prints where any
# hand-picked λ would sit on the pre-eruption misfit/‖m‖ trade-off.
if LAMBDA_SCAN:
    print('\n  λ justification scan (diagnostic, pre-eruption solve only)...')
    print(f'  {"λ_damp":>10s}  {"λ_smooth":>10s}  {"misfit":>10s}  {"‖m‖":>10s}')
    for _lam in LAMBDA_SCAN:
        _, _Gaug = build_augmented(A_pre, _lam, _lam * SMOOTH_RATIO)
        _x, _mis, _mn = solve_system(_Gaug, d_pre, d_aug_pre)
        flag = '  <-- LAMBDA_FIXED' if np.isclose(_lam, LAMBDA_FIXED) else ''
        print(f'  {_lam:10.4g}  {_lam*SMOOTH_RATIO:10.4g}  {_mis:10.4f}  {_mn:10.4f}{flag}')
    print('  (scan complete — chosen λ is unchanged)\n')

# ── Fixed escape hatch: no scan, honest single-point figure ───────────────────
if LAMBDA_MODE == 'fixed':
    LAMBDA_DAMP   = float(LAMBDA_FIXED)
    LAMBDA_SMOOTH = LAMBDA_DAMP * SMOOTH_RATIO
    SELECT_METHOD = 'fixed (not data-selected)'
    print(f'  FIXED: λ_damp={LAMBDA_DAMP:.2e}  λ_smooth={LAMBDA_SMOOTH:.2e}  '
          f'(scan skipped — value is NOT data-selected)')
    # Solve pre-eruption once at the fixed λ so the figure shows one HONEST point.
    _, _Gaug = build_augmented(A_pre, LAMBDA_DAMP, LAMBDA_SMOOTH)
    _x, _mis, _mn = solve_system(_Gaug, d_pre, d_aug_pre)
    lambdas    = np.array([LAMBDA_DAMP])
    misfits    = np.array([_mis])
    mod_norms  = np.array([_mn])
    marker_idx = 0
else:
    # ── Real scan: build the regularisation curve over the log-spaced grid ────
    lambdas = np.logspace(np.log10(LAM_MIN), np.log10(LAM_MAX), N_LCURVE)
    print(f'  Discrepancy target = {target_residual:.4f}  (N_pre={int(_pe.sum()):,})')
    misfits, mod_norms = [], []
    for lam in lambdas:
        _, G_aug_pre = build_augmented(A_pre, lam, lam * SMOOTH_RATIO)
        _x, mis, mnorm = solve_system(G_aug_pre, d_pre, d_aug_pre)
        misfits.append(mis); mod_norms.append(mnorm)
        print(f'  λ={lam:.2e}  misfit={mis:.3f}  ‖m‖={mnorm:.3f}')
    misfits = np.array(misfits); mod_norms = np.array(mod_norms)

    LAMBDA_DAMP, marker_idx, SELECT_METHOD = select_lambda(
        lambdas, misfits, mod_norms, target_residual, LAMBDA_MODE)
    LAMBDA_SMOOTH = LAMBDA_DAMP * SMOOTH_RATIO
    print(f'  SELECTED via {SELECT_METHOD}: '
          f'λ_damp={LAMBDA_DAMP:.2e}  λ_smooth={LAMBDA_SMOOTH:.2e}  '
          f'(marker idx={marker_idx}, target={target_residual:.4f}, '
          f'misfit range [{misfits.min():.3f}, {misfits.max():.3f}])')

# ── Save regularisation figure — honest label, no NaN arrays ──────────────────
# The figure is only ever built from a real solve: either the full scan or, in
# fixed mode, a single honest (misfit,‖m‖) point at the forced λ. It never plots
# fabricated NaN arrays, and the star/label reflect the method that actually ran.
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
ax_lc.set_xlabel('Data misfit ‖Gx−d‖', fontsize=11)
ax_lc.set_ylabel('Model norm ‖x‖', fontsize=11)
ax_lc.set_title('Regularisation curve: pre-eruption data\n'
                f'λ_damp={LAMBDA_DAMP:.2e}, λ_smooth={LAMBDA_SMOOTH:.2e}  '
                f'({SELECT_METHOD})',
                fontsize=10, fontweight='bold')
ax_lc.legend(fontsize=9); ax_lc.grid(True, which='both', alpha=0.3)
fig_lc.tight_layout()
fig_lc.savefig(BASE + 'lcurve_regularisation_CORRECTED.pdf', dpi=200, bbox_inches='tight')
plt.close(fig_lc)
print('Saved lcurve_regularisation_CORRECTED.pdf')

# ── Solve for each time period ─────────────────────────────────────────────────
print(f'\nRunning inversions (λ_damp={LAMBDA_DAMP:.2e}, λ_smooth={LAMBDA_SMOOTH:.2e})...')
results = []   # list of (u, v, n_rays) per period
for lbl,t_lo,t_hi in periods:
    m = (ray_t >= t_lo) if t_lo is not None else np.ones(N_RAYS,dtype=bool)
    if t_hi is not None: m = m & (ray_t < t_hi)
    u,v,cov = solve_period(m, lbl.replace('\n',' '), LAMBDA_DAMP, LAMBDA_SMOOTH)
    m_str = np.sqrt(u**2+v**2).reshape(NX,NY,NZ)
    theta = (0.5*np.degrees(np.arctan2(v,u)) % 180).reshape(NX,NY,NZ)
    cov3d = cov.reshape(NX,NY,NZ)
    # Zero out voxels with no ray coverage
    m_str[~cov3d] = np.nan
    theta[~cov3d] = np.nan
    results.append(dict(label=lbl, n=int(m.sum()),
                        m=m_str, theta=theta, cov=cov3d))

# ── Recompute tomographic centroid from CORRECTED pre-eruption 7-period model ────
# Definition: (x,y,z) of the top-10% anisotropy-strength (m_str) covered voxels of
# the pre-eruption period, reported at voxel CENTRES (half-voxel registration).
_pre_m = results[0]['m']                       # (NX,NY,NZ), NaN where uncovered
_finite = np.isfinite(_pre_m)
if _finite.sum() == 0:
    print('\n*** TOMO_CENTROID: no covered voxels in pre-eruption model — cannot recompute ***')
    TOMO_CENTROID_NEW = None
else:
    _vals = _pre_m[_finite]
    _thr  = np.nanpercentile(_pre_m, 90.0)     # top 10% strength threshold
    _sel  = _finite & (_pre_m >= _thr)
    _ixg, _iyg, _izg = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
    _xc = (xn[_ixg[_sel]] + VXY/2.)
    _yc = (yn[_iyg[_sel]] + VXY/2.)
    _zc = (zn[_izg[_sel]] + VZ/2.)
    _w  = _pre_m[_sel]                          # strength-weighted centroid
    _cx = float(np.average(_xc, weights=_w))
    _cy = float(np.average(_yc, weights=_w))
    _cz = float(np.average(_zc, weights=_w))
    TOMO_CENTROID_NEW = (round(_cx,2), round(_cy,2), round(_cz,2))
    print('\n' + '='*72)
    print('  RECOMPUTED TOMO_CENTROID (corrected phi%180 pre-eruption 7-period model)')
    print(f'  top-10% strength threshold m_str >= {_thr:.4f} s/km, {int(_sel.sum())} voxels')
    print(f'  NEW TOMO_CENTROID = ({_cx:.2f}, {_cy:.2f}, {_cz:.2f})  (x_km, y_km, z_km)')
    print(f'  (old stale value was (8.31, 5.08, 0.86) from buggy phi+90 runs)')
    print('='*72)

# ── Bathymetry ─────────────────────────────────────────────────────────────────
BATHY=('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
       'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS=None
_p=PILImage.open(BATHY); _t=_p.tag_v2
_olon,_olat=_t[33922][3],_t[33922][4]; _pl,_pb=_t[33550][0],_t[33550][1]
_nc,_nr=_p.size; _p.close()
_c0=max(0,int(((INI_LON+X_START/KM_PER_DEG_LON)-_olon)/_pl)-2)
_c1=min(_nc,int(((INI_LON+X_END/KM_PER_DEG_LON)-_olon)/_pl)+2)
_r0=max(0,int((_olat-(INI_LAT+Y_END/KM_PER_DEG_LAT))/_pb)-2)
_r1=min(_nr,int((_olat-(INI_LAT+Y_START/KM_PER_DEG_LAT))/_pb)+2)
_rgb=tifffile.imread(BATHY)[_r0:_r1,_c0:_c1]
_ds=max(1,max(_rgb.shape[:2])//1024); _rgb=_rgb[::_ds,::_ds]
_gray=np.dot(_rgb[...,:3].astype(np.float32),[0.299,0.587,0.114]).astype(np.uint8)
_ext=[(_olon+_c0*_pl-INI_LON)*KM_PER_DEG_LON,(_olon+_c1*_pl-INI_LON)*KM_PER_DEG_LON,
      (_olat-_r1*_pb-INI_LAT)*KM_PER_DEG_LAT,(_olat-_r0*_pb-INI_LAT)*KM_PER_DEG_LAT]

def _bathy(ax):
    ax.imshow(_gray,origin='upper',extent=_ext,aspect='auto',cmap='gray',zorder=0)
def _sta_markers(ax):
    for sta,row in _sta.iterrows():
        ax.plot(row['x'],row['y'],'^',ms=6,mfc='#FFD700',mec='k',mew=0.7,zorder=12)
        dx,dy=LABEL_OFFSET.get(sta,(0.12,0.12))
        ax.text(row['x']+dx,row['y']+dy,STA_DISPLAY.get(sta,sta),fontsize=5.5,zorder=13)
def _kidiwela(ax):
    for s in KIDIWELA:
        ax.plot(s['x'],s['y'],'o',ms=4,mfc='red',mec='k',mew=0.5,zorder=14)
        ax.text(s['x']+0.12,s['y']+0.12,s['label'],fontsize=5,color='red',
                fontweight='bold',zorder=15)

# ── Additional source markers ──────────────────────────────────────────────────
# Slead sources (converted from lon/lat)
INI_LON_M,INI_LAT_M=-130.1,45.9
KDN_M=111.32*np.cos(np.radians(INI_LAT_M)); KDL_M=111.32
def _ll2xy_m(lat,lon): return (lon-INI_LON_M)*KDN_M,(lat-INI_LAT_M)*KDL_M

SLEAD_SOURCES=[
    ('2a',      *_ll2xy_m(45.9637,-130.0100), 2.666),
    ('2b.1',    *_ll2xy_m(45.9968,-130.0249), 2.241),
    ('2b.2',    *_ll2xy_m(45.9543,-130.0110), 1.712),
    ('2b.3',    *_ll2xy_m(45.9265,-129.9850), 1.985),
]
# Chadwick & Nooner prolate spheroid centroid
CHADWICK_NOONER=(8.84, 5.38, 3.81)   # (x, y, depth) km
# Hefner et al. 2020: 2110 m S64°W of Nooner & Chadwick → az=244°
import math as _math
_az=_math.radians(244.); _d=2.110
HEFNER=(8.84+_d*_math.sin(_az), 5.38+_d*_math.cos(_az), 3.01)
# Tomographic centroid (top 10% pre-eruption, 7-period model).
# Recomputed above (TOMO_CENTROID_NEW) from the CORRECTED phi%180 inversion. The
# stale buggy-phi+90 fallback (8.31, 5.08, 0.86) is only used if recomputation
# somehow failed (no covered voxels), which would itself be reported loudly.
TOMO_CENTROID = TOMO_CENTROID_NEW if TOMO_CENTROID_NEW is not None else (8.31, 5.08, 0.86)

def _fault_traces(ax, depth_km=0.0):
    """
    Draw fault intersection lines at a given depth.
    Both ring faults strike N30°W (330°), dip 66° outward from caldera.
    Eastern fault (center 8.5,4.0) dips ENE (az=60°).
    Western fault (center 7.0,4.0) dips WSW (az=240°).
    Horizontal shift per km of depth = cot(66°) = 0.445 km.
    """
    import math as _m
    DEPTH_TOP=0.1; COT66=_m.cos(_m.radians(66))/_m.sin(_m.radians(66))
    horiz=(depth_km-DEPTH_TOP)*COT66  # horizontal shift along dip direction
    faults=[
        (8.5, 4.0, 330., 60.),   # east: cx,cy,strike,dip_azimuth
        (7.0, 4.0, 330., 240.),  # west
    ]
    for cx,cy,strike,dip_az in faults:
        # Shift centre to depth
        da=_m.radians(dip_az)
        cx_z=cx+horiz*_m.sin(da); cy_z=cy+horiz*_m.cos(da)
        # Draw strike line
        st=_m.radians(strike); L=8.
        x1,y1=cx_z-L/2*_m.sin(st),cy_z-L/2*_m.cos(st)
        x2,y2=cx_z+L/2*_m.sin(st),cy_z+L/2*_m.cos(st)
        ax.plot([x1,x2],[y1,y2],'-',color='#cc0000',lw=0.9,alpha=0.75,zorder=11)

def _extra_sources(ax):
    MS=3; FS=4.0; MEW=0.5  # dot size, font size, edge width
    # Slead — green dots
    for lbl,sx,sy,dep in SLEAD_SOURCES:
        ax.plot(sx,sy,'o',ms=MS,mfc='limegreen',mec='darkgreen',mew=MEW,zorder=16)
        ax.text(sx+0.10,sy+0.08,lbl,fontsize=FS,color='darkgreen',
                fontweight='bold',zorder=17)
    # Chadwick & Nooner — purple dot
    ax.plot(CHADWICK_NOONER[0],CHADWICK_NOONER[1],'o',ms=MS,
            mfc='purple',mec='indigo',mew=MEW,zorder=16)
    ax.text(CHADWICK_NOONER[0]+0.10,CHADWICK_NOONER[1]+0.08,
            'C&N',fontsize=FS,color='purple',fontweight='bold',zorder=17)
    # Hefner et al. 2020 — orange dot
    ax.plot(HEFNER[0],HEFNER[1],'o',ms=MS,
            mfc='darkorange',mec='saddlebrown',mew=MEW,zorder=16)
    ax.text(HEFNER[0]+0.10,HEFNER[1]+0.08,
            'Hef',fontsize=FS,color='darkorange',fontweight='bold',zorder=17)
    # This study (tomographic centroid) — dark blue dot (phi page only, not dt)
    # Added conditionally via _add_tomo_centroid(ax) where needed

# FIX 3: half-voxel registration. The ray-to-voxel sampler treats xn[j]/yn[j] as the
# LEFT EDGE of voxel j (floor((mid-xn[0])/VXY)), so the value attributed to voxel j
# physically sits at its CENTRE xn[j]+VXY/2, yn[j]+VXY/2. Plot at cell centres so the
# field is not shifted half a voxel relative to bathymetry/stations (those stay in the
# absolute-km frame and are NOT shifted). 'ij' indexing is preserved.
Xg,Yg=np.meshgrid(xn+VXY/2., yn+VXY/2., indexing='ij')

# ── Global colour scales ───────────────────────────────────────────────────────
all_m = np.concatenate([r['m'][r['m']>0].ravel() for r in results])
m_vmax= float(np.percentile(all_m,99)) if len(all_m) else 0.05
m_vmin= 0.
print(f'\nAnisotropy strength range: 0 – {m_vmax:.4f} s/km')

# ── Figure builder: dt (anisotropy strength) ───────────────────────────────────
def make_strength_page(title):
    n_dep=len(DEPTH_SLICES); n_per=len(periods)
    panel_w=2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.3))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    cmap=plt.colormaps['Blues']
    levels=np.linspace(m_vmin,m_vmax,15)
    last_h=None
    for ri,(z0,zlbl) in enumerate(zip(DEPTH_SLICES,DEPTH_LABELS)):
        iz0,iz1=depth_slice_idx(z0)   # FIX 3: centre-based depth selection
        for ci,res in enumerate(results):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            sl=np.nanmean(res['m'][:,:,iz0:iz1],axis=2)
            msk=np.isfinite(sl)
            if msk.sum()>=COUNT_MIN:
                sl_fill=np.where(msk,sl,0.)
                sm=gaussian_filter(sl_fill,sigma=1.0)
                mp=np.ma.masked_where(~msk,sm)
                if not mp.mask.all():
                    ax.contourf(Xg,Yg,mp,levels=levels,cmap=cmap,
                                vmin=m_vmin,vmax=m_vmax,extend='neither')
                    last_h=ScalarMappable(cmap=cmap,norm=Normalize(m_vmin,m_vmax))
                    last_h.set_array([])
            _sta_markers(ax); _kidiwela(ax); _extra_sources(ax); _fault_traces(ax,z0)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{res["label"]}\nN={res["n"]:,}',
                                    fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'{zlbl}',fontsize=7)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        cb=plt.colorbar(last_h,cax=cax,ticks=np.linspace(m_vmin,m_vmax,5),
                        format=FormatStrFormatter('%.3f'))
        cb.set_label('Anisotropy m [s/km]',fontsize=9); cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.99)
    # External legend
    import matplotlib.lines as mlines
    legend_handles=[
        mlines.Line2D([],[],marker='o',color='w',mfc='red',mec='k',ms=5,label='Kidiwela'),
        mlines.Line2D([],[],marker='o',color='w',mfc='limegreen',mec='darkgreen',ms=5,label='Slead'),
        mlines.Line2D([],[],marker='o',color='w',mfc='purple',mec='indigo',ms=5,label='Chadwick & Nooner'),
        mlines.Line2D([],[],marker='o',color='w',mfc='darkorange',mec='saddlebrown',ms=5,label='Hefner'),
        mlines.Line2D([],[],marker='^',color='w',mfc='#FFD700',mec='k',ms=6,label='OBS station'),
    ]
    fig.legend(handles=legend_handles,loc='lower center',ncol=4,
               fontsize=7,framealpha=0.9,bbox_to_anchor=(0.45,0.0))
    fig.subplots_adjust(bottom=0.07)
    return fig

# ── Figure builder: phi (fast direction sticks) ────────────────────────────────
def make_phi_page(title):
    n_dep=len(DEPTH_SLICES); n_per=len(periods)
    panel_w=2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.3))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    cmap_phi=plt.colormaps['hsv_r']
    last_h=None
    STEP=3; SLEN=0.18; LW=0.7
    for ri,(z0,zlbl) in enumerate(zip(DEPTH_SLICES,DEPTH_LABELS)):
        iz0,iz1=depth_slice_idx(z0)   # FIX 3: centre-based depth selection
        for ci,res in enumerate(results):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            m_sl  =np.nanmean(res['m'][:,:,iz0:iz1],axis=2)
            th_sl =np.nanmean(res['theta'][:,:,iz0:iz1],axis=2)
            msk=np.isfinite(m_sl)
            segs,colors=[],[]
            ny_,nx_=m_sl.shape
            for i in range(0,ny_,STEP):
                for j in range(0,nx_,STEP):
                    if not msk[i,j]: continue
                    phi_r=np.radians(th_sl[i,j])
                    x0,y0=Xg[i,j],Yg[i,j]
                    dx=SLEN*np.sin(phi_r); dy=SLEN*np.cos(phi_r)
                    segs.append([(x0-dx,y0-dy),(x0+dx,y0+dy)])
                    colors.append(th_sl[i,j]/180.)
            if segs:
                lc=LineCollection(segs,colors=cmap_phi(colors),
                                  linewidths=LW,zorder=6,alpha=0.9)
                ax.add_collection(lc)
                last_h=ScalarMappable(cmap=cmap_phi,norm=Normalize(0,180))
                last_h.set_array([])
            _sta_markers(ax); _kidiwela(ax); _extra_sources(ax); _fault_traces(ax,z0)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{res["label"]}\nN={res["n"]:,}',
                                    fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'{zlbl}',fontsize=7)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        cb=plt.colorbar(last_h,cax=cax,ticks=np.arange(0,181,30),
                        format=FormatStrFormatter('%.0f'))
        cb.set_label('Fast direction θ [° from N]',fontsize=9); cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.99)
    # External legend
    import matplotlib.lines as mlines
    legend_handles=[
        mlines.Line2D([],[],marker='o',color='w',mfc='red',mec='k',ms=5,label='Kidiwela'),
        mlines.Line2D([],[],marker='o',color='w',mfc='limegreen',mec='darkgreen',ms=5,label='Slead'),
        mlines.Line2D([],[],marker='o',color='w',mfc='purple',mec='indigo',ms=5,label='Chadwick & Nooner'),
        mlines.Line2D([],[],marker='o',color='w',mfc='darkorange',mec='saddlebrown',ms=5,label='Hefner'),
        mlines.Line2D([],[],marker='o',color='w',mfc='darkblue',mec='navy',ms=5,label='This study'),
        mlines.Line2D([],[],marker='^',color='w',mfc='#FFD700',mec='k',ms=6,label='OBS station'),
    ]
    fig.legend(handles=legend_handles,loc='lower center',ncol=4,
               fontsize=7,framealpha=0.9,bbox_to_anchor=(0.45,0.0))
    fig.subplots_adjust(bottom=0.07)
    return fig

# ── Write PDFs ─────────────────────────────────────────────────────────────────
tag=f'(φ_err≤{PHI_ERR_MAX}°, λ_d={LAMBDA_DAMP}, λ_s={LAMBDA_SMOOTH})'

print(f'\nWriting {os.path.basename(OUT_DT)}...')
with PdfPages(OUT_DT) as pdf:
    fig=make_strength_page(f'SWS tomography — anisotropy strength m [s/km]  {tag}')
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
print(f'Saved {OUT_DT}')

print(f'Writing {os.path.basename(OUT_PHI)}...')
with PdfPages(OUT_PHI) as pdf:
    fig=make_phi_page(f'SWS tomography — fast direction θ  {tag}')
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
print(f'Saved {OUT_PHI}')

# ── Annual inversion ───────────────────────────────────────────────────────────
print('\nRunning annual inversions...')
periods_annual = build_annual_periods()
results_annual = []
for lbl,t_lo,t_hi in periods_annual:
    m = (ray_t >= t_lo) if t_lo is not None else np.ones(N_RAYS,dtype=bool)
    if t_hi is not None: m = m & (ray_t < t_hi)
    u,v,cov = solve_period(m, lbl.replace('\n',' '), LAMBDA_DAMP, LAMBDA_SMOOTH)
    m_str = np.sqrt(u**2+v**2).reshape(NX,NY,NZ)
    theta  = (0.5*np.degrees(np.arctan2(v,u))%180).reshape(NX,NY,NZ)
    cov3d  = cov.reshape(NX,NY,NZ)
    m_str[~cov3d] = np.nan; theta[~cov3d] = np.nan
    results_annual.append(dict(label=lbl,n=int(m.sum()),m=m_str,theta=theta,cov=cov3d))

# Swap results so make_strength_page/make_phi_page use annual periods
_results_orig = results
results = results_annual

OUT_DT_A  = BASE + 'sws_tomography_dt_annual_phi10_dt001_CORRECTED.pdf'
OUT_PHI_A = BASE + 'sws_tomography_phi_annual_phi10_dt001_CORRECTED.pdf'

all_m_a=[r['m'][r['m']>0].ravel() for r in results if r['m'][np.isfinite(r['m'])].size>0]
if all_m_a:
    import numpy as _np
    dt_vmax=float(_np.percentile(_np.concatenate([x for x in all_m_a if len(x)]),99))
    print(f'Annual anisotropy range: 0 – {dt_vmax:.4f} s/km')

print(f'\nWriting {os.path.basename(OUT_DT_A)}...')
with PdfPages(OUT_DT_A) as pdf:
    # Split annual into two pages: 2015 rows and 2016+ rows
    CHUNK = 7   # panels per page
    for start in range(0, len(results), CHUNK):
        chunk = results[start:start+CHUNK]
        _orig = results[:]
        results[:] = chunk
        fig = make_strength_page(f'SWS tomography annual — anisotropy strength m [s/km]  {tag}')
        pdf.savefig(fig, dpi=300, bbox_inches='tight'); plt.close(fig)
        results[:] = _orig
print(f'Saved {OUT_DT_A}')

print(f'Writing {os.path.basename(OUT_PHI_A)}...')
with PdfPages(OUT_PHI_A) as pdf:
    for start in range(0, len(results), CHUNK):
        chunk = results[start:start+CHUNK]
        _orig = results[:]
        results[:] = chunk
        fig = make_phi_page(f'SWS tomography annual — fast direction θ  {tag}')
        pdf.savefig(fig, dpi=300, bbox_inches='tight'); plt.close(fig)
        results[:] = _orig
print(f'Saved {OUT_PHI_A}')

results = _results_orig   # restore
print('\nDone.')
