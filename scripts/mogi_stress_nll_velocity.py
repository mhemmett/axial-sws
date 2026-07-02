#!/usr/bin/env python3
"""
mogi_stress_nll_velocity.py

Re-runs the Mogi stress comparison using Baillard's 3D S-wave velocity model
(AXIAL_MODEL_3P_VELOCITY.S.mod) for ray-path length calculation instead of
the simple AMC sphere routing.

For each earthquake-station ray:
  1. Sample the 3D Vs model along the straight-line path (N points)
  2. Identify the fraction of path through solid rock (Vs > VS_THRESHOLD)
  3. Effective path length L_eff = L_total × solid_fraction
     (excludes partial-melt / AMC segments from the splitting accumulation)

Mogi sources: Kidiwela two-sphere model.
No dike intrusion (pure Mogi comparison).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import map_coordinates

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
DATA_DIR     = '/Users/mhemmett/Seismology/axial-splitting-ml/data/'
STATION_FILE = DATA_DIR + 'stations_axial.llz'
NLL_BUF      = DATA_DIR + 'AXIAL_MODEL_3P_VELOCITY.S.mod.buf'
OUT_DIR      = BASE

# ── NLL velocity model parameters ─────────────────────────────────────────────
# From AXIAL_MODEL_3P_VELOCITY.S.mod.hdr
NLL_NX, NLL_NY, NLL_NZ = 302, 302, 92
NLL_OX, NLL_OY, NLL_OZ = 0.0, 0.0, -0.5   # km (NLL grid origin)
NLL_DX, NLL_DY, NLL_DZ = 0.05, 0.05, 0.05  # km grid spacing
NLL_LAT_ORIG = 45.9
NLL_LON_ORIG = -130.1
MEAN_SEAFLOOR_KM = 1.660   # mean depth to seafloor below sea level (km)

# Vs threshold: below this the material is treated as partial melt / AMC
# and is excluded from the anisotropy-accumulating path length
VS_THRESHOLD = 1.5    # km/s
N_RAY_SAMPLES = 100   # sample points per ray

# ── Coordinate system ─────────────────────────────────────────────────────────
# Our maps use INI_LON=-130.1, INI_LAT=45.9 — same as NLL origin, so:
#   our_x (km E) = NLL x   |   our_y (km N) = NLL y
# Earthquake depth z_sf (km below seafloor, positive down):
#   NLL z = MEAN_SEAFLOOR_KM + z_sf
# NLL grid index: ix = x/DX, iy = y/DY, iz = (NLL_z - NLL_OZ)/DZ

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_PER_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_PER_DEG_LAT)

# ── Mogi source parameters (Kidiwela) ─────────────────────────────────────────
MU, NU = 30e9, 0.25
SPHERES = [
    dict(x0=7.57, y0=4.55, d=3.33, R=0.43, dP=0.05e9, label='S1'),
    dict(x0=7.53, y0=6.60, d=1.25, R=0.20, dP=0.05e9, label='S2'),
]

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

GEODETIC = [
    (pd.Timestamp('2015-01-01'), 3.50),
    (pd.Timestamp('2015-04-24'), 1.00),
    (pd.Timestamp('2018-01-01'), 2.20),
    (pd.Timestamp('2019-01-01'), 2.60),
    (pd.Timestamp('2021-01-01'), 3.00),
    (pd.Timestamp('2022-01-01'), 3.20),
    (pd.Timestamp('2023-01-01'), 3.30),
    (pd.Timestamp('2024-01-01'), 3.34),
    (pd.Timestamp('2025-01-01'), 3.59),
    (pd.Timestamp('2026-01-01'), 3.75),
    (pd.Timestamp('2026-05-01'), 3.77),
]
GEO_T = pd.to_datetime([g[0] for g in GEODETIC])
GEO_Z = np.array([g[1] for g in GEODETIC])
ELEV_BASELINE, ELEV_KIDIWELA = 1.0, 3.2
SCALE_DENOM = ELEV_KIDIWELA - ELEV_BASELINE

def elevation_at(t):
    return float(np.interp(pd.Timestamp(t).value, GEO_T.asi8.astype(float), GEO_Z))

def source_scale(t):
    return (elevation_at(t) - ELEV_BASELINE) / SCALE_DENOM

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

# ── Load S-wave velocity model ────────────────────────────────────────────────

print('Loading 3D S-wave velocity model...')
vel_3d = np.frombuffer(open(NLL_BUF, 'rb').read(),
                       dtype=np.float32).reshape(NLL_NX, NLL_NY, NLL_NZ).copy()
print(f'  Shape: {vel_3d.shape}  Vs range: {vel_3d.min():.3f}–{vel_3d.max():.3f} km/s')
print(f'  Vs < {VS_THRESHOLD} km/s: {100*(vel_3d<VS_THRESHOLD).mean():.1f}% of nodes (AMC + low-vel)')

def sample_vs_along_ray(eq_x, eq_y, eq_z_sf, sta_x, sta_y):
    """
    Sample Vs at N_RAY_SAMPLES points along the straight ray from earthquake
    to station (surface). Returns array of Vs values.

    eq_z_sf : depth below seafloor (km, positive down)
    """
    # Station is at seafloor: NLL z = MEAN_SEAFLOOR_KM
    # Earthquake: NLL z = MEAN_SEAFLOOR_KM + eq_z_sf
    eq_z_nll  = MEAN_SEAFLOOR_KM + eq_z_sf
    sta_z_nll = MEAN_SEAFLOOR_KM

    t  = np.linspace(0, 1, N_RAY_SAMPLES)
    rx = eq_x  + t * (sta_x  - eq_x)
    ry = eq_y  + t * (sta_y  - eq_y)
    rz = eq_z_nll + t * (sta_z_nll - eq_z_nll)

    # NLL grid indices (continuous, for interpolation)
    ix = (rx - NLL_OX) / NLL_DX
    iy = (ry - NLL_OY) / NLL_DY
    iz = (rz - NLL_OZ) / NLL_DZ

    coords = np.array([ix, iy, iz])
    return map_coordinates(vel_3d, coords, order=1, mode='nearest')


def effective_ray_length(eq_x, eq_y, eq_z_sf, sta_x, sta_y):
    """
    Straight-line distance × fraction of path through solid rock (Vs >= threshold).
    """
    L_total = float(np.sqrt((sta_x-eq_x)**2 + (sta_y-eq_y)**2 + eq_z_sf**2))
    vs_samples = sample_vs_along_ray(eq_x, eq_y, eq_z_sf, sta_x, sta_y)
    solid_frac = float(np.mean(vs_samples >= VS_THRESHOLD))
    return L_total * solid_frac, solid_frac


# ── Load stations ──────────────────────────────────────────────────────────────

_sta = pd.read_csv(STATION_FILE, sep=r'\s+',
                   names=['lon','lat','elev_km','station'],
                   engine='python').set_index('station')
_sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'], _sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)

# ── Load splitting data ────────────────────────────────────────────────────────

FILES = {
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv',
             'splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv',
             'splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv',
             'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv',
             'splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv',
             'splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv',
             'splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

print('\nLoading splitting data and computing effective ray lengths...')
dfs = {}
mean_L_nll = {}
mean_solid_frac = {}

for sta, (f1, f2) in FILES.items():
    df = pd.concat([pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
                    for f in [f1, f2]], ignore_index=True)
    df = df[df['dt'] > 0]
    df['x'], df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
    df['z'] = df['event_depth'].values
    df['t'] = pd.to_datetime(df['event_datetime'], utc=True)
    df['phi_az'] = df['phi'] + 90.0

    # Sample 500 events to estimate mean effective path length
    srow = _sta.loc[sta]
    sx, sy = float(srow['x']), float(srow['y'])
    sample = df.sample(min(500, len(df)), random_state=42)
    L_effs, fracs = [], []
    for _, row in sample.iterrows():
        L_eff, frac = effective_ray_length(
            float(row['x']), float(row['y']), float(row['z']), sx, sy)
        L_effs.append(L_eff)
        fracs.append(frac)

    mean_L_nll[sta]      = float(np.mean(L_effs))
    mean_solid_frac[sta] = float(np.mean(fracs))
    dfs[sta] = df
    print(f'  {sta}: {len(df):,} events | '
          f'mean L_eff={mean_L_nll[sta]:.2f} km | '
          f'solid fraction={mean_solid_frac[sta]:.2%}')

# ── Mogi stress field ─────────────────────────────────────────────────────────

def T_sphere_fast(xn, yn, X, Y, sph, sc=1.):
    lam = 2*NU*MU/(1-2*NU)
    dV  = np.pi*(sph['R']*1e3)**3*sph['dP']*sc/MU
    C   = dV*(1-NU)/np.pi
    dx  = (X-sph['x0'])*1e3; dy = (Y-sph['y0'])*1e3; d = sph['d']*1e3
    R3  = (dx**2+dy**2+d**2)**1.5
    Ux  = C*dx/R3; Uy = C*dy/R3
    ny, nx = Ux.shape; dm = (xn[1]-xn[0])*1e3; dm2 = (yn[1]-yn[0])*1e3
    T = np.zeros((ny, nx, 2, 2))
    e11 = (Ux[1:-1,2:]-Ux[1:-1,:-2])/(2*dm)
    e22 = (Uy[2:,1:-1]-Uy[:-2,1:-1])/(2*dm2)
    e12 = 0.5*((Ux[2:,1:-1]-Ux[:-2,1:-1])/(2*dm2)
              +(Uy[1:-1,2:]-Uy[1:-1,:-2])/(2*dm))
    tr  = e11+e22
    T[1:-1,1:-1,0,0]=lam*tr+2*MU*e11; T[1:-1,1:-1,0,1]=T[1:-1,1:-1,1,0]=2*MU*e12
    T[1:-1,1:-1,1,1]=lam*tr+2*MU*e22
    return T

def phi_at(T, xn, yn, sx, sy):
    iy=int(np.argmin(abs(yn-sy))); ix=int(np.argmin(abs(xn-sx)))
    ev,evec=np.linalg.eigh(T[iy,ix]); v=abs(ev[1]-ev[0])*evec[:,0]
    return float(np.degrees(np.arctan2(v[0],v[1]))%180)

def dphi(a,b):
    if np.isnan(a) or np.isnan(b): return float('nan')
    return float((a-b+90)%180-90)

xn=np.arange(4,12.01,0.1); yn=np.arange(0,12.01,0.1); X,Y=np.meshgrid(xn,yn)
sp=(3.5-1.)/2.2

print('\nComputing Mogi stress fields...')
T1=T_sphere_fast(xn,yn,X,Y,SPHERES[0]); T2=T_sphere_fast(xn,yn,X,Y,SPHERES[1])
T_pre=sp*(T1+T2)

# Optimal Mogi-only deflation (S1=5%, S2=100%)
T_syn_m=sp*0.95*T1+sp*0.0*T2

# ── Pre vs syn observed phi ───────────────────────────────────────────────────

obs_pre={}; obs_syn={}
for sta, df in dfs.items():
    pre=df[df['t']<ERUPTION_START]; syn=df[(df['t']>=ERUPTION_START)&(df['t']<ERUPTION_END)]
    obs_pre[sta]=float(pre['phi_az'].median()) if len(pre)>=10 else float('nan')
    obs_syn[sta]=float(syn['phi_az'].median()) if len(syn)>=10 else float('nan')

# ── Results table ──────────────────────────────────────────────────────────────

print('\n' + '─'*95)
print(f'  Baillard 3D Vs model ray tracing  |  Vs threshold = {VS_THRESHOLD} km/s  |  '
      f'Mogi only (S1=5%, S2=100% deflation)')
print('─'*95)
print(f"{'Station':>8} | {'phi_obs':^14} | {'Dphi':>8} | {'phi_pred':^14} | {'Dphi':>8} | "
      f"{'L_eff':>7} | {'solid%':>6}")
print(f"{'':>8} | {'PRE':>6} {'SYN':>6} | {'obs':>8} | {'PRE':>6} {'SYN':>6} | {'pred':>8} | "
      f"{'km':>7} | {'':>6}")
print('─'*95)

for sta in STATIONS:
    sx, sy = float(_sta.loc[sta,'x']), float(_sta.loc[sta,'y'])
    pp  = phi_at(T_pre,   xn, yn, sx, sy)
    ps  = phi_at(T_syn_m, xn, yn, sx, sy)
    do  = dphi(obs_syn[sta], obs_pre[sta])
    dp  = dphi(ps, pp)
    L   = mean_L_nll[sta]
    sf  = mean_solid_frac[sta]
    print(f"{sta[2:]:>8} | {obs_pre[sta]:>6.1f} {obs_syn[sta]:>6.1f} | {do:>+8.1f} | "
          f"{pp:>6.1f} {ps:>6.1f} | {dp:>+8.1f} | {L:>7.2f} | {sf:>6.1%}")

print('─'*95)

print('\nComparison of ray lengths: NLL model vs old AMC sphere')
print(f"{'Station':>8} | {'L_eff NLL (km)':>15} | {'solid frac':>11}")
for sta in STATIONS:
    print(f"{sta[2:]:>8} | {mean_L_nll[sta]:>15.2f} | {mean_solid_frac[sta]:>11.1%}")

print('\nDone.')
