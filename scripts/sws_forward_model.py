#!/usr/bin/env python3
"""
sws_forward_model.py

Full forward model connecting:
  Mogi (S1+S2) + Okada dike  →  surface stress field
  + lithostatic background    →  3D stress tensor along ray paths
  Hudson random crack model   →  effective anisotropic stiffness
  Christoffel + integration   →  predicted φ and δt per station × period

MCMC parameters θ (11 total):
  [0]  dP1      : S1 pressure change [GPa]
  [1]  dP2      : S2 pressure change [GPa]
  [2]  dike_x0  : dike east position [km]  (syn-eruption only)
  [3]  dike_ys  : dike south tip y [km]
  [4]  dike_yn  : dike north tip y [km]
  [5]  dike_dt  : dike top depth [km]
  [6]  dike_db  : dike bottom depth [km]
  [7]  dike_U3  : dike opening [m]
  [8]  epsilon0 : isotropic crack density
  [9]  r_crack  : crack aspect ratio
  [10] sigma_c  : crack closure pressure [MPa]

Fixed: ν=0.25, ρ=2750 kg/m³, P_litho (background pressure, set separately)
"""

import numpy as np
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator
from hudson_crack_model import (effective_stiffness, christoffel_splitting,
                                 fibonacci_sphere, K_WATER, RHO, NU)
from baillard_velocity import ray_vs

# ── Constants ─────────────────────────────────────────────────────────────────
MU_BG  = 30e9       # Pa  background shear modulus
NU_FIX = 0.25       # fixed Poisson's ratio
LAM_BG = 2*NU_FIX*MU_BG/(1-2*NU_FIX)   # = MU_BG for ν=0.25

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_PER_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_PER_DEG_LAT)

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'

# Kidiwela source geometry (fixed)
SPHERES = [
    dict(x0=7.57, y0=4.55, d=3.33, R=0.43, label='S1'),
    dict(x0=7.53, y0=6.60, d=1.25, R=0.20, label='S2'),
]

# Geodetic elevation record → source scale
_GEO_T = pd.to_datetime([
    '2015-01-01','2015-04-24','2018-01-01','2019-01-01','2021-01-01',
    '2022-01-01','2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-05-01'])
_GEO_T = _GEO_T.tz_localize('UTC')
_GEO_Z = np.array([3.50,1.00,2.20,2.60,3.00,3.20,3.30,3.34,3.59,3.75,3.77])
_ELEV_BASE    = 1.00   # post-eruption minimum [m]
_ELEV_KIDIWELA= 3.20   # Kidiwela reference [m]
_SCALE_DENOM  = _ELEV_KIDIWELA - _ELEV_BASE

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

STATIONS = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']

# Ray sampling
N_RAY_PTS   = 20    # sample points per ray
N_CRACKS    = 200   # crack normal samples (faster than 300 for MCMC)
P_HAT_VERT  = np.array([0., 0., 1.])  # near-vertical ray direction

# ── Geodetic scaling ──────────────────────────────────────────────────────────

def source_scale(t):
    t_num = pd.Timestamp(t).value
    z     = float(np.interp(t_num, _GEO_T.asi8.astype(float), _GEO_Z))
    return (z - _ELEV_BASE) / _SCALE_DENOM

def period_midpoint(t0, t1):
    t0 = pd.Timestamp('2015-01-01', tz='UTC') if t0 is None else t0
    t1 = pd.Timestamp('2026-05-01', tz='UTC') if t1 is None else t1
    return t0 + (t1-t0)/2

# ── Mogi surface stress field ─────────────────────────────────────────────────

def _mogi_disp_grid(X, Y, sph, dP_Pa, mu=MU_BG, nu=NU_FIX):
    """Surface displacement (Ux, Uy) from one Mogi sphere."""
    dV  = np.pi * (sph['R']*1e3)**3 * dP_Pa / mu
    C   = dV*(1-nu)/np.pi
    dx  = (X-sph['x0'])*1e3;  dy = (Y-sph['y0'])*1e3;  d = sph['d']*1e3
    R3  = (dx**2+dy**2+d**2)**1.5
    return C*dx/R3, C*dy/R3

def mogi_stress_field(X, Y, xn, yn, dP1_Pa, dP2_Pa, mu=MU_BG, nu=NU_FIX):
    """
    2D surface stress tensor field from both Mogi spheres.

    Returns sigma: (ny, nx, 3) where sigma[i,j] = [σ_xx, σ_yy, σ_xy] in Pa.
    At the free surface σ_zz=σ_xz=σ_yz=0.
    """
    lam = 2*nu*mu/(1-2*nu)
    Ux  = np.zeros_like(X);  Uy = np.zeros_like(X)
    for sph, dP in zip(SPHERES, [dP1_Pa, dP2_Pa]):
        ux, uy = _mogi_disp_grid(X, Y, sph, dP, mu, nu)
        Ux += ux;  Uy += uy

    dm  = (xn[1]-xn[0])*1e3;  dm2 = (yn[1]-yn[0])*1e3
    ny, nx = Ux.shape
    sig = np.zeros((ny, nx, 3))   # [σ_xx, σ_yy, σ_xy]

    # Central differences (interior only)
    e11 = (Ux[1:-1,2:]-Ux[1:-1,:-2])/(2*dm)
    e22 = (Uy[2:,1:-1]-Uy[:-2,1:-1])/(2*dm2)
    e12 = 0.5*((Ux[2:,1:-1]-Ux[:-2,1:-1])/(2*dm2) +
               (Uy[1:-1,2:]-Uy[1:-1,:-2])/(2*dm))
    tr  = e11+e22
    sig[1:-1,1:-1,0] = lam*tr + 2*mu*e11   # σ_xx
    sig[1:-1,1:-1,1] = lam*tr + 2*mu*e22   # σ_yy
    sig[1:-1,1:-1,2] = 2*mu*e12             # σ_xy
    return sig

# ── Analytical 3D Mogi stress at arbitrary depth ─────────────────────────────

def mogi_stress_tensor_3d(x_km, y_km, z_km, sph, dP_Pa, mu=MU_BG, nu=NU_FIX):
    """
    Full 3D stress tensor from one Mogi sphere at an arbitrary interior point.

    Uses the method of images: direct source at depth d plus a mirror source
    above the free surface at depth -d, weighted so that the horizontal
    displacement recovers the standard Mogi (1958) surface formula at z=0.

    The stress is deviatoric (trace-free) from each term; trace contributions
    from the free-surface correction are small compared with lithostatic and
    are handled by the separate P_litho term.

    Parameters
    ----------
    x_km, y_km : horizontal observation position [km]
    z_km       : observation depth below seafloor [km], positive downward
    sph        : Mogi sphere dict with keys x0, y0 [km], d [km], R [km]
    dP_Pa      : pressure change [Pa]

    Returns
    -------
    sigma : (3,3) ndarray [Pa], tensile positive  (x=E, y=N, z=down)
    """
    dV = np.pi * (sph['R'] * 1e3)**3 * dP_Pa / mu   # volume change [m³]
    A  = dV * (1.0 - nu) / np.pi                     # Mogi amplitude [m⁴]

    dx   = (x_km - sph['x0']) * 1e3                  # [m]
    dy   = (y_km - sph['y0']) * 1e3
    dz_d = (z_km - sph['d'])  * 1e3   # obs relative to source  (+ = below src)
    dz_i = (z_km + sph['d'])  * 1e3   # obs relative to image   (always +)

    R_d = max(np.sqrt(dx**2 + dy**2 + dz_d**2), 1.0)   # [m], floor 1 m
    R_i = max(np.sqrt(dx**2 + dy**2 + dz_i**2), 1.0)

    def _kelvin_sig(rx, ry, rz, R):
        """Deviatoric stress tensor from a Kelvin centre of dilatation."""
        r = np.array([rx, ry, rz])
        sig = np.zeros((3, 3))
        for ii in range(3):
            for jj in range(3):
                sig[ii, jj] = 2.0 * mu * A * (
                    (1.0 if ii == jj else 0.0) / R**3
                    - 3.0 * r[ii] * r[jj] / R**5
                )
        return sig

    # Half weight on each term — recovers Mogi surface displacement at z_km=0
    return 0.5 * _kelvin_sig(dx, dy, dz_d, R_d) \
         + 0.5 * _kelvin_sig(dx, dy, dz_i, R_i)


def mogi_stress_3d_all_sources(x_km, y_km, z_km, dP_list, mu=MU_BG, nu=NU_FIX):
    """
    Combined 3D stress at (x_km, y_km, z_km) from all SPHERES.

    dP_list : iterable of pressure changes [Pa], one per entry in SPHERES.
    Returns (3,3) stress tensor [Pa].
    """
    sig = np.zeros((3, 3))
    for sph, dP in zip(SPHERES, dP_list):
        if dP != 0.0:
            sig += mogi_stress_tensor_3d(x_km, y_km, z_km, sph, dP, mu, nu)
    return sig


# ── Okada vertical dike surface stress ───────────────────────────────────────

def okada_dike_stress(X, Y, xn, yn, x0, y_start, y_end,
                      depth_top, depth_bot, opening_m,
                      mu=MU_BG, nu=NU_FIX):
    """
    Surface stress from a vertical tensile dike (Okada 1985).
    Returns (ny, nx, 3) stress tensor [σ_xx, σ_yy, σ_xy].
    Derived from surface displacements via finite differences.
    """
    lam = 2*nu*mu/(1-2*nu)
    L   = abs(y_end-y_start)*1e3; W = (depth_bot-depth_top)*1e3
    y_c = 0.5*(y_start+y_end)
    xi  = (Y-y_c)*1e3; eta = (X-x0)*1e3; q = eta

    def R(xi_,e_,p_): return np.sqrt(xi_**2+e_**2+p_**2)+1e-10
    def fux(xi_,e_,p_):
        r=R(xi_,e_,p_); rp=r+p_+1e-10; re=r+e_+1e-10
        return -(1-2*nu)*xi_/re - xi_*q/r/rp
    def fuy(xi_,e_,p_):
        r=R(xi_,e_,p_); rp=r+p_+1e-10; re=r+e_+1e-10
        th=np.arctan2(xi_*e_,q*r+1e-10)
        return xi_**2/r/rp-(1-2*nu)/re+(1-2*nu)*th
    def chinnery(f):
        x1,x2=xi-L/2,xi+L/2; e1,e2=depth_top*1e3,depth_bot*1e3
        return f(x2,e2,q)-f(x2,e1,q)-f(x1,e2,q)+f(x1,e1,q)

    sc = opening_m/(2*np.pi)
    Ux = sc*chinnery(fuy)   # E component (perpendicular to strike)
    Uy = sc*chinnery(fux)   # N component (along strike)

    dm=xn[1]-xn[0]; dm2=yn[1]-yn[0]
    ny,nx=Ux.shape; sig=np.zeros((ny,nx,3))
    dm*=1e3; dm2*=1e3
    e11=(Ux[1:-1,2:]-Ux[1:-1,:-2])/(2*dm)
    e22=(Uy[2:,1:-1]-Uy[:-2,1:-1])/(2*dm2)
    e12=0.5*((Ux[2:,1:-1]-Ux[:-2,1:-1])/(2*dm2)+
             (Uy[1:-1,2:]-Uy[1:-1,:-2])/(2*dm))
    tr=e11+e22
    sig[1:-1,1:-1,0]=lam*tr+2*mu*e11
    sig[1:-1,1:-1,1]=lam*tr+2*mu*e22
    sig[1:-1,1:-1,2]=2*mu*e12
    return sig

# ── 2D stress → 3D tensor (free surface + lithostatic) ───────────────────────

def stress_3d(sig2d_vec, P_litho):
    """
    Convert 2D surface stress [σ_xx, σ_yy, σ_xy] to full 3D tensor
    and add isotropic lithostatic pressure.

    At free surface: σ_zz = σ_xz = σ_yz = 0.
    Lithostatic: adds -P_litho to all diagonal components
    (tensile-positive convention: compression is negative).
    """
    sxx, syy, sxy = sig2d_vec[0], sig2d_vec[1], sig2d_vec[2]
    sigma = np.array([[sxx, sxy, 0.],
                      [sxy, syy, 0.],
                      [0.,  0.,  0.]])
    sigma[0,0] -= P_litho   # add compression (negative in tensile-positive)
    sigma[1,1] -= P_litho
    sigma[2,2] -= P_litho
    return sigma

# ── Build fast stress interpolators ──────────────────────────────────────────

def build_interpolators(sig_field, xn, yn):
    """
    Build scipy RegularGridInterpolator for each stress component.
    sig_field: (ny, nx, 3)
    Returns list of 3 interpolators.
    """
    interps = []
    for k in range(3):
        interps.append(
            RegularGridInterpolator((yn, xn), sig_field[:,:,k],
                                    method='linear', bounds_error=False,
                                    fill_value=0.))
    return interps

def eval_stress_at(interps, x_km, y_km):
    """Evaluate [σ_xx, σ_yy, σ_xy] at point (x_km, y_km)."""
    pt = np.array([[y_km, x_km]])
    return np.array([ip(pt)[0] for ip in interps])

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(dt_error_max=None):
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
    dfs = {}
    for sta, (f1, f2) in FILES.items():
        df = pd.concat([pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
                        for f in [f1,f2]], ignore_index=True)
        df = df[df['dt'] > 0]
        if dt_error_max: df = df[df['dt_error'] < dt_error_max]
        df['x'],df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
        df['z'] = df['event_depth'].values
        df['t'] = pd.to_datetime(df['event_datetime'], utc=True)
        df['phi_az'] = df['phi'] + 90.
        dfs[sta] = df
    return dfs

def build_time_periods(all_df):
    post = all_df[all_df['t']>=ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post); bounds=[ERUPTION_END]
    for i in range(1,5):
        idx=min(int(round(i*n/5)),n-1); bounds.append(post['t'].iloc[idx])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds=[('Pre-eruption',None,ERUPTION_START),('Syn-eruption',ERUPTION_START,ERUPTION_END)]
    for i in range(5): pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}',bounds[i],bounds[i+1]))
    return pds

def subset(df, t0, t1):
    m = (df['t']>=t0) if t0 is not None else pd.Series(True,index=df.index)
    if t1 is not None: m = m&(df['t']<t1)
    return df[m]

# ── Observation summary ───────────────────────────────────────────────────────

def build_observations(dfs, time_periods, min_events=15):
    """
    Build observed (phi_med, dt_med) arrays: shape (n_sta, n_per).
    NaN where insufficient data.
    """
    n_sta, n_per = len(STATIONS), len(time_periods)
    obs_phi = np.full((n_sta, n_per), np.nan)
    obs_dt  = np.full((n_sta, n_per), np.nan)
    obs_N   = np.zeros((n_sta, n_per), dtype=int)
    for j, (_, t0, t1) in enumerate(time_periods):
        for i, sta in enumerate(STATIONS):
            sub = subset(dfs[sta], t0, t1)
            if len(sub) >= min_events:
                obs_phi[i,j] = sub['phi_az'].median()
                obs_dt[i,j]  = sub['dt'].median()
                obs_N[i,j]   = len(sub)
    return obs_phi, obs_dt, obs_N

def build_median_ray_origins(dfs, time_periods):
    """
    Median (x, y, z) event location per station per period.
    Used as the earthquake end of each ray.
    """
    n_sta, n_per = len(STATIONS), len(time_periods)
    med_x = np.zeros((n_sta, n_per)); med_y = np.zeros_like(med_x)
    med_z = np.zeros_like(med_x)
    for j, (_, t0, t1) in enumerate(time_periods):
        for i, sta in enumerate(STATIONS):
            sub = subset(dfs[sta], t0, t1)
            if len(sub) > 0:
                med_x[i,j] = sub['x'].median()
                med_y[i,j] = sub['y'].median()
                med_z[i,j] = sub['z'].median()
            else:
                # Use overall median as fallback
                med_x[i,j] = dfs[sta]['x'].median()
                med_y[i,j] = dfs[sta]['y'].median()
                med_z[i,j] = dfs[sta]['z'].median()
    return med_x, med_y, med_z

# ── Full forward model ────────────────────────────────────────────────────────

def forward(theta, time_periods, stations_xy, med_x, med_y, med_z,
            xn, yn, X, Y, P_litho=65e6, n_hats=None):
    """
    Forward model: parameters → predicted (phi, dt) at each station × period.

    Parameters
    ----------
    theta : 11-element array
        [dP1_GPa, dP2_GPa,
         dike_x0, dike_ys, dike_yn, dike_dt, dike_db, dike_U3,
         epsilon0, r_crack, sigma_c_MPa]
    time_periods : list of (label, t0, t1)
    stations_xy  : (n_sta, 2) station (x, y) in km
    med_x/y/z    : (n_sta, n_per) median event locations per station×period
    xn, yn       : 1D grid arrays [km]
    X, Y         : 2D meshgrid [km]
    P_litho      : lithostatic pressure [Pa] (positive = compressive)
    n_hats       : crack normal samples; uses default if None

    Returns
    -------
    phi_pred : (n_sta, n_per)  [° from N]
    dt_pred  : (n_sta, n_per)  [s]
    """
    if n_hats is None:
        n_hats = fibonacci_sphere(N_CRACKS)

    # Unpack θ
    dP1   = theta[0]*1e9;  dP2   = theta[1]*1e9   # GPa → Pa
    dike  = dict(x0=theta[2], y_start=theta[3], y_end=theta[4],
                 depth_top=theta[5], depth_bot=theta[6], opening=theta[7])
    eps0  = theta[8]; r_cr = theta[9]; sig_c = theta[10]*1e6  # MPa → Pa
    B_bg  = theta[11]*1e6                          # background deviatoric [Pa]
    phi1  = np.radians(theta[12])                  # σ₁ azimuth [rad]

    # Background tectonic stress tensor (horizontal deviatoric + lithostatic)
    # σ₁ (compressive) along phi1, σ₃ (tensile) along phi1+90°
    c1 = np.array([np.sin(phi1), np.cos(phi1)])    # σ₁ direction (x=E, y=N)
    e1 = np.array([-np.cos(phi1), np.sin(phi1)])   # σ₃ direction (perpendicular)
    sigma_bg = np.zeros((3, 3))
    for ii in range(2):
        for jj in range(2):
            sigma_bg[ii, jj] = B_bg*(e1[ii]*e1[jj] - c1[ii]*c1[jj])
    sigma_bg[0,0] -= P_litho
    sigma_bg[1,1] -= P_litho
    sigma_bg[2,2] -= P_litho

    n_sta, n_per = len(STATIONS), len(time_periods)
    phi_pred = np.full((n_sta, n_per), np.nan)
    dt_pred  = np.full((n_sta, n_per), np.nan)

    # Precompute dike stress once (only geometry changes between calls)
    dike_stress = okada_dike_stress(X, Y, xn, yn,
                                    dike['x0'], dike['y_start'], dike['y_end'],
                                    dike['depth_top'], dike['depth_bot'],
                                    dike['opening'])
    dike_interps = build_interpolators(dike_stress, xn, yn)

    for j, (label, t0, t1) in enumerate(time_periods):
        t_mid  = period_midpoint(t0, t1)
        scale  = source_scale(t_mid)
        is_syn = (label == 'Syn-eruption')

        # Dike stress: 2D surface interpolator (syn-eruption only)
        if is_syn:
            dike_interps_j = dike_interps
        else:
            dike_interps_j = None

        for i, (sx, sy) in enumerate(stations_xy):
            eq_x = med_x[i, j]; eq_y = med_y[i, j]; eq_z = med_z[i, j]

            t_s   = np.linspace(0., 1., N_RAY_PTS)
            ray_x = eq_x + t_s*(sx - eq_x)
            ray_y = eq_y + t_s*(sy - eq_y)
            ray_z = eq_z * (1.0 - t_s)          # depth: eq_z → 0 at station

            L_total = float(np.linalg.norm([(sx-eq_x)*1e3,
                                             (sy-eq_y)*1e3,
                                             eq_z*1e3]))
            ds = np.full(N_RAY_PTS, L_total / N_RAY_PTS)

            # 3D stress at each ray sample
            sigma_stack = np.zeros((N_RAY_PTS, 3, 3))
            for k in range(N_RAY_PTS):
                sig3d = sigma_bg.copy()
                sig3d += mogi_stress_3d_all_sources(
                    ray_x[k], ray_y[k], ray_z[k],
                    [dP1*scale, dP2*scale])
                if dike_interps_j is not None:
                    sd = eval_stress_at(dike_interps_j, ray_x[k], ray_y[k])
                    sig3d[0,0] += sd[0]; sig3d[1,1] += sd[1]
                    sig3d[0,1] += sd[2]; sig3d[1,0] += sd[2]
                sigma_stack[k] = sig3d

            # Baillard Vs along the ray — drives both path-length weighting and δt
            vs_arr, ds_km = ray_vs(eq_x, eq_y, eq_z,
                                    sx,   sy,   0.,
                                    n=N_RAY_PTS)
            vs_ms   = vs_arr * 1e3                  # km/s → m/s
            ds_m    = ds_km  * 1e3                  # km   → m per segment

            # Integrate Hudson splitting along ray
            phi_k = np.zeros(N_RAY_PTS); dV_k = np.zeros(N_RAY_PTS)
            for k in range(N_RAY_PTS):
                C_eff, _ = effective_stiffness(
                    sigma_stack[k], MU_BG, eps0, K_WATER,
                    r_cr, sig_c, n_hats, NU_FIX)
                phi_k[k], dV_k[k], _, _ = christoffel_splitting(
                    C_eff, P_HAT_VERT, RHO)

            # Path-weighted circular mean φ (weight by local splitting × segment)
            w = dV_k * ds_m; w_sum = w.sum()
            if w_sum < 1e-30:
                phi_pred[i, j] = float(np.mean(phi_k))
            else:
                ang2 = 2.*np.radians(phi_k)
                phi_pred[i, j] = float(np.degrees(
                    np.arctan2((w*np.sin(ang2)).sum(),
                               (w*np.cos(ang2)).sum())) / 2. % 180.)

            # δt: integrate dV/VS² × ds along ray using Baillard VS at each point
            vs_safe = np.maximum(vs_ms, 1.)
            dt_pred[i, j] = float(np.sum(dV_k / vs_safe**2 * ds_m))

    return phi_pred, dt_pred

# ── Log-likelihood and log-posterior ─────────────────────────────────────────

def circular_phi_misfit(phi_pred_deg, obs_phi_deg):
    """
    Geodesic misfit for axial (180°-periodic) fast-direction data.

    Uses the doubled-angle metric: map φ → 2φ on the unit circle, compute
    the arc-length misfit, then halve back to degrees.  This is the correct
    circular distance for axial (undirected) data and avoids the ~30° artifact
    that Euclidean % 180 introduces near the wrap-around.

    Returns residuals in degrees, range (−90, 90].
    """
    drad = np.radians(2. * (phi_pred_deg - obs_phi_deg))
    return np.degrees(np.arctan2(np.sin(drad), np.cos(drad))) / 2.


def log_likelihood(phi_pred, dt_pred, obs_phi, obs_dt,
                   sigma_phi=15., sigma_dt=0.02):
    """
    Gaussian log-likelihood for (phi, dt) observations.

    φ uses the proper circular (doubled-angle) metric — not Euclidean % 180.
    δt uses standard Euclidean misfit; it is the amplitude-bearing observable
    that constrains source pressure even when φ has saturated.

    sigma_phi in degrees, sigma_dt in seconds.
    """
    ll = 0.
    mask_phi = ~np.isnan(obs_phi) & ~np.isnan(phi_pred)
    mask_dt  = ~np.isnan(obs_dt)  & ~np.isnan(dt_pred)

    if mask_phi.any():
        dphi = circular_phi_misfit(phi_pred[mask_phi], obs_phi[mask_phi])
        ll  -= 0.5 * np.sum(dphi**2 / sigma_phi**2)

    if mask_dt.any():
        ddt  = dt_pred[mask_dt] - obs_dt[mask_dt]
        ll  -= 0.5 * np.sum(ddt**2 / sigma_dt**2)

    return ll

# Prior bounds: [lo, hi] for each parameter
PRIOR_BOUNDS = [
    [0.005, 0.5],     #  0  dP1 [GPa]
    [0.005, 0.5],     #  1  dP2 [GPa]
    [8.0,  11.0],     #  2  dike x0 [km]
    [3.0,   7.0],     #  3  dike y_start [km]
    [7.0,  12.0],     #  4  dike y_end [km]
    [0.0,   2.0],     #  5  dike depth_top [km]
    [0.1,   5.0],     #  6  dike depth_bot [km]
    [0.1,   8.0],     #  7  dike opening [m]
    [0.005, 0.50],    #  8  epsilon0  (wider: volcanic rock can be highly fractured)
    [0.001, 0.10],    #  9  r_crack
    [1.0, 200.0],     # 10  sigma_c [MPa]
    [0.1,  50.0],     # 11  B_bg [MPa]  background deviatoric magnitude
    [0.,  180.],      # 12  phi_1 [°]   azimuth of σ₁ (most compressive), CW from N
]
_LO = np.array([b[0] for b in PRIOR_BOUNDS])
_HI = np.array([b[1] for b in PRIOR_BOUNDS])

# Starting point — Kidiwela sources + E-W caldera compression from scan results
THETA0 = np.array([
    0.05,    #  0  dP1 [GPa] — Kidiwela deep source
    0.05,    #  1  dP2 [GPa] — Kidiwela shallow source
    10.01,   #  2  dike x0
    3.90,    #  3  dike y_start
    6.82,    #  4  dike y_end
    0.31,    #  5  dike depth_top
    0.41,    #  6  dike depth_bot
    2.62,    #  7  dike opening
    0.05,    #  8  epsilon0
    0.01,    #  9  r_crack
    40.0,    # 10  sigma_c [MPa]
    3.0,     # 11  B_bg [MPa]  — best fit from background scan
    80.0,    # 12  phi_1 [°]   — E-W compression (caldera inflation)
])

def log_prior(theta):
    """Uniform prior within bounds; -inf outside."""
    if np.any(theta < _LO) or np.any(theta > _HI): return -np.inf
    if theta[6] <= theta[5] + 0.05: return -np.inf   # depth_bot > depth_top
    if theta[4] <= theta[3] + 0.5:  return -np.inf   # y_end > y_start + 0.5 km
    return 0.

def log_posterior(theta, time_periods, stations_xy, med_x, med_y, med_z,
                  xn, yn, X, Y, obs_phi, obs_dt,
                  P_litho=65e6, sigma_phi=15., sigma_dt=0.02, n_hats=None):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    try:
        phi_pred, dt_pred = forward(theta, time_periods, stations_xy,
                                     med_x, med_y, med_z,
                                     xn, yn, X, Y, P_litho, n_hats)
        return lp + log_likelihood(phi_pred, dt_pred, obs_phi, obs_dt,
                                    sigma_phi, sigma_dt)
    except Exception:
        return -np.inf

# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    print('Loading data...')
    dfs      = load_data(dt_error_max=0.1)
    all_df   = pd.concat(dfs.values(), ignore_index=True)
    periods  = build_time_periods(all_df)

    _sta = pd.read_csv(STATION_FILE, sep=r'\s+',
                       names=['lon','lat','elev_km','station'],
                       engine='python').set_index('station')
    _sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
    _sta['x'],_sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)
    sta_xy = np.column_stack([_sta['x'].values, _sta['y'].values])

    obs_phi, obs_dt, obs_N = build_observations(dfs, periods)
    med_x, med_y, med_z    = build_median_ray_origins(dfs, periods)

    xn=np.arange(4,12.01,0.1); yn=np.arange(0,12.01,0.1)
    X,Y=np.meshgrid(xn,yn)
    n_hats = fibonacci_sphere(N_CRACKS)

    print(f'\nRunning forward model at starting point (θ₀)...')
    t0 = time.time()
    phi_p, dt_p = forward(THETA0, periods, sta_xy, med_x, med_y, med_z,
                           xn, yn, X, Y, P_litho=65e6, n_hats=n_hats)
    elapsed = time.time()-t0
    print(f'  Time: {elapsed:.1f} s')

    ll = log_likelihood(phi_p, dt_p, obs_phi, obs_dt)
    print(f'  Log-likelihood: {ll:.1f}')

    sep = '─'*80
    print(f'\n{sep}')
    print(f"{'Station':>8} | {'phi_obs':^14} | {'Dphi_obs':>9} | "
          f"{'phi_pred':^14} | {'Dphi_pred':>10}")
    print(f"{'':>8} | {'PRE':>6} {'SYN':>6} | {'':>9} | {'PRE':>6} {'SYN':>6} | {'':>10}")
    print(sep)
    for i,sta in enumerate(STATIONS):
        po=obs_phi[i,0]; ps=obs_phi[i,1]
        pp=phi_p[i,0];   psp=phi_p[i,1]
        do=((ps-po+90)%180-90) if not np.isnan(po+ps) else float('nan')
        dp=((psp-pp+90)%180-90) if not np.isnan(pp+psp) else float('nan')
        print(f"{sta[2:]:>8} | {po:>6.1f} {ps:>6.1f} | {do:>+9.1f} | "
              f"{pp:>6.1f} {psp:>6.1f} | {dp:>+10.1f}")
    print(sep)
    print(f'\nP_litho = 65 MPa  |  epsilon0 = {THETA0[8]}  |  '
          f'sigma_c = {THETA0[10]} MPa  |  r_crack = {THETA0[9]}')
    print(f'\nForward model ready for MCMC.')
