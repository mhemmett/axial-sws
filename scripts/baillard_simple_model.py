#!/usr/bin/env python3
"""
baillard_simple_model.py

Replicates Baillard's (2019) approach:
  1. Surface stress from pressurized source + dike
  2. Eigendecompose 2-D horizontal stress tensor at each station
  3. φ_pred = azimuth of σ₁ (most compressive eigenvector)
  4. Compare Δφ = φ_syn − φ_pre (modelled vs. observed) at 6 stations

Source geometry (as specified):
  Origin: 45.9°N, 130.1°W
  Prolate spheroid: centre (8.84, 5.38) km, depth 3.81 km,
                    major axis 2.2 km, minor axis 0.38 km,
                    major axis dips at 77° toward azimuth 286°.
  Yang et al. (1988) half-space solution (numerical integration along
  major axis using Gauss-Legendre quadrature with Eshelby weight function).

Dike (syn-eruption):
  Fissure (8.29, 6.27) → (8.19, 8.20) km, azimuth 353°, length 1.93 km.
  Okada (1985) vertical tensile dike rotated to 353° strike.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
from sws_forward_model import (
    load_data, build_time_periods, build_observations,
    build_interpolators, eval_stress_at,
    STATIONS, STATION_FILE, ll2xy,
    MU_BG, NU_FIX,
)

# ── Source geometry ───────────────────────────────────────────────────────────

# Prolate spheroid
SPH_X0    =  8.84    # km East from origin
SPH_Y0    =  5.38    # km North from origin
SPH_DEPTH =  3.81    # km depth to centre
SPH_A     =  2.20    # km  semi-major axis  (at 77° dip → ~4.28 km vertical extent)
SPH_B     =  0.38    # km  semi-minor axis
SPH_DIP   = 77.      # degrees from horizontal
SPH_AZIMUTH = 286.   # degrees CW from N  (WNW)

# Dike (syn-eruption)
DIKE_START    = (8.29, 6.27)   # (x, y) km
DIKE_END      = (8.19, 8.20)   # (x, y) km
DIKE_AZIMUTH  = 353.           # degrees CW from N  (slightly W of N)
DIKE_DEPTH_TOP = 0.1           # km
DIKE_DEPTH_BOT = 2.0           # km
DIKE_OPENING   = 2.0           # m
DIKE_DEPTH_TOP = 0.0           # km (surface)
DIKE_DEPTH_BOT = 0.5           # km

# Elastic constants
MU, NU = MU_BG, NU_FIX
LAM = 2*NU*MU/(1-2*NU)

# Source pressures
DP_INFLATE =  0.05e9    # Pa  pre-eruption inflation
DP_DEFLATE = -0.05e9    # Pa  syn-eruption deflation

# ── Yang et al. (1988) prolate spheroid ──────────────────────────────────────

def yang_surface_displacement(X_km, Y_km,
                               x0, y0, depth, a, b,
                               dip_deg, az_deg,
                               dP, mu, nu, N_gauss=80):
    """
    Surface displacement (Ux, Uy) [m] from a pressurized prolate spheroid.

    Uses Gauss–Legendre integration along the major axis between the two
    focal points.  Each element is treated as a Mogi (1958) half-space
    point source with volume dV given by the Eshelby (1957) equivalent
    line distribution:

        dV/ds  ∝  sqrt(c² − s²)    (−c ≤ s ≤ c)

    The distribution is calibrated so that the total ΔV matches the
    Eshelby inclusion result for a pressurized prolate spheroid:

        ΔV_total = (4π/3) · a · b² · dP / μ

    Parameters
    ----------
    X_km, Y_km : (ny, nx) ndarray  surface grid [km]
    x0, y0, depth : float          spheroid centre [km]
    a, b : float                   semi-major / semi-minor axes [km]
    dip_deg : float                dip from horizontal [°]
    az_deg  : float                azimuth of dip direction, CW from N [°]
    dP      : float                excess internal pressure [Pa]
    N_gauss : int                  quadrature points along major axis
    """
    # Convert to metres
    x0m  = x0    * 1e3;  y0m  = y0    * 1e3;  dm   = depth * 1e3
    am   = a     * 1e3;  bm   = b     * 1e3
    X_m  = X_km  * 1e3;  Y_m  = Y_km  * 1e3

    # Focal distance and eccentricity
    e_f = np.sqrt(max(1 - (bm/am)**2, 0))
    c_m = am * e_f                              # focal distance [m]

    # Major-axis unit vector (dip direction, positive downward)
    dip_r = np.radians(dip_deg)
    az_r  = np.radians(az_deg)
    ux_hat =  np.sin(dip_r) * np.sin(az_r)    # East
    uy_hat =  np.sin(dip_r) * np.cos(az_r)    # North
    uz_hat =  np.cos(dip_r)                   # Down (positive)

    # Eshelby total volume change for pressurised prolate spheroid
    # ΔV = (4π/3)·a·b²·ΔP / μ  (leading-order Eshelby result)
    DV_total = (4*np.pi/3.0) * am * bm**2 * dP / mu

    # Weight function: dV/ds = K·sqrt(c²−s²), integral over [−c,+c] = K·π·c²/2
    K_weight = 2.0 * DV_total / (np.pi * c_m**2) if c_m > 1.0 else DV_total

    # Gauss–Legendre nodes on [−c, +c]
    xi, wi = np.polynomial.legendre.leggauss(N_gauss)
    s_pts  = c_m * xi
    w_pts  = c_m * wi                          # Jacobian

    Ux = np.zeros_like(X_m)
    Uy = np.zeros_like(X_m)

    for k in range(N_gauss):
        s   = s_pts[k]
        dV  = K_weight * np.sqrt(max(c_m**2 - s**2, 0)) * w_pts[k]

        # 3-D position of this source element [m]
        xs = x0m + s * ux_hat
        ys = y0m + s * uy_hat
        zs = dm  + s * uz_hat          # depth (positive down)

        if zs <= 0:
            continue                    # skip elements above free surface

        dx = X_m - xs;  dy = Y_m - ys
        R3 = (dx**2 + dy**2 + zs**2)**1.5 + 1e-6

        # Mogi (1958) half-space surface displacement from point dilatation
        A   = dV * (1 - nu) / np.pi / R3
        Ux += A * dx
        Uy += A * dy

    return Ux, Uy


def yang_surface_stress(X_km, Y_km, x0, y0, depth, a, b,
                         dip_deg, az_deg, dP, mu, nu):
    """
    Surface stress (σ_xx, σ_yy, σ_xy) from Yang spheroid displacement.
    Uses central finite differences on the displacement field.
    """
    Ux, Uy = yang_surface_displacement(X_km, Y_km, x0, y0, depth, a, b,
                                        dip_deg, az_deg, dP, mu, nu)
    dx_km = X_km[0, 1] - X_km[0, 0]
    dy_km = Y_km[1, 0] - Y_km[0, 0]
    dm    = dx_km * 1e3;  dm2 = dy_km * 1e3
    ny, nx = Ux.shape
    sig = np.zeros((ny, nx, 3))
    e11 = (Ux[1:-1, 2:] - Ux[1:-1, :-2]) / (2*dm)
    e22 = (Uy[2:, 1:-1] - Uy[:-2, 1:-1]) / (2*dm2)
    e12 = 0.5*((Ux[2:, 1:-1] - Ux[:-2, 1:-1]) / (2*dm2) +
               (Uy[1:-1, 2:] - Uy[1:-1, :-2]) / (2*dm))
    tr  = e11 + e22
    sig[1:-1, 1:-1, 0] = LAM*tr + 2*mu*e11
    sig[1:-1, 1:-1, 1] = LAM*tr + 2*mu*e22
    sig[1:-1, 1:-1, 2] = 2*mu*e12
    return sig

# ── Okada (1985) dike with arbitrary azimuth ──────────────────────────────────

def okada_dike_azimuth(X_km, Y_km, xn, yn,
                        x_start, y_start, x_end, y_end,
                        az_deg, depth_top, depth_bot, opening, mu, nu):
    """
    Surface stress from a vertical tensile dike with arbitrary azimuth.

    Strategy:
      1. Transform each geographic grid point to the dike-aligned frame
         (along-strike = Yr, perpendicular = Xr).
      2. Evaluate the Okada (1985) N-S dike formula at those local coords.
      3. Rotate the resulting displacement vectors back to geographic (E, N).
      4. Compute stress via central finite differences on the geographic grid.

    This avoids the re-interpolation issues of working on a rotated sub-grid.
    """
    lam = 2*nu*mu/(1-2*nu)

    # Dike geometry
    L     = np.sqrt((x_end-x_start)**2 + (y_end-y_start)**2)   # km
    x_mid = 0.5*(x_start + x_end)
    y_mid = 0.5*(y_start + y_end)

    # Unit vectors in geographic (East, North) frame
    az_r  = np.radians(az_deg)
    s_hat = np.array([ np.sin(az_r),  np.cos(az_r)])   # along-strike
    n_hat = np.array([ np.cos(az_r), -np.sin(az_r)])   # perpendicular (right-hand)

    # Relative position of every grid point from dike mid-point
    dX = X_km - x_mid
    dY = Y_km - y_mid

    # Project onto dike frame
    Xr = dX*n_hat[0] + dY*n_hat[1]    # perpendicular [km]  (= "East"  in Okada)
    Yr = dX*s_hat[0] + dY*s_hat[1]    # along-strike  [km]  (= "North" in Okada)

    # Okada (1985) Chinnery notation for vertical tensile dike
    # Dike at Xr=0, extending from Yr = -L/2 to Yr = +L/2
    def Rok(xi, e, p): return np.sqrt(xi**2 + e**2 + p**2) + 1e-10
    def fux(xi, e, p):            # along-normal displacement kernel
        r  = Rok(xi, e, p); rp = r+p+1e-10; re = r+e+1e-10
        return -(1-2*nu)*xi/re - xi*e/(r*rp)
    def fuy(xi, e, p):            # along-strike displacement kernel
        r  = Rok(xi, e, p); rp = r+p+1e-10; re = r+e+1e-10
        th = np.arctan2(xi*e, e*r+1e-10)
        return xi**2/(r*rp) - (1-2*nu)/re + (1-2*nu)*th

    def chinnery(f, xi_m, q_m, d_top, d_bot):
        e1, e2 = d_top*1e3, d_bot*1e3
        x1 = xi_m - (-L/2)*1e3    # xi relative to south end
        x2 = xi_m - ( L/2)*1e3    # xi relative to north end
        return f(x2,e2,q_m) - f(x2,e1,q_m) - f(x1,e2,q_m) + f(x1,e1,q_m)

    sc    = opening / (2*np.pi)
    xi_m  = Yr * 1e3    # along-strike [m]
    q_m   = Xr * 1e3    # perpendicular [m]

    Ux_df = sc * chinnery(fuy, xi_m, q_m, depth_top, depth_bot)   # perp displacement
    Uy_df = sc * chinnery(fux, xi_m, q_m, depth_top, depth_bot)   # strike displacement

    # Rotate displacement back to geographic (East, North)
    Ux_geo = Ux_df * n_hat[0] + Uy_df * s_hat[0]    # East
    Uy_geo = Ux_df * n_hat[1] + Uy_df * s_hat[1]    # North

    # Stress via central finite differences on geographic grid
    dm  = (xn[1]-xn[0])*1e3
    dm2 = (yn[1]-yn[0])*1e3
    ny_, nx_ = Ux_geo.shape
    sig = np.zeros((ny_, nx_, 3))
    e11 = (Ux_geo[1:-1,2:] - Ux_geo[1:-1,:-2]) / (2*dm)
    e22 = (Uy_geo[2:,1:-1] - Uy_geo[:-2,1:-1]) / (2*dm2)
    e12 = 0.5*((Ux_geo[2:,1:-1]-Ux_geo[:-2,1:-1])/(2*dm2) +
               (Uy_geo[1:-1,2:]-Uy_geo[1:-1,:-2])/(2*dm))
    tr  = e11 + e22
    sig[1:-1,1:-1,0] = lam*tr + 2*mu*e11
    sig[1:-1,1:-1,1] = lam*tr + 2*mu*e22
    sig[1:-1,1:-1,2] = 2*mu*e12
    return sig

# ── Build stress fields ───────────────────────────────────────────────────────
xn = np.arange(4., 13.01, 0.10)
yn = np.arange(1., 11.01, 0.10)
X, Y = np.meshgrid(xn, yn)

print('Computing Yang spheroid stress fields...')
sig_pre = yang_surface_stress(X, Y,
                               SPH_X0, SPH_Y0, SPH_DEPTH,
                               SPH_A, SPH_B, SPH_DIP, SPH_AZIMUTH,
                               DP_INFLATE, MU, NU)
sig_syn_sph = yang_surface_stress(X, Y,
                                   SPH_X0, SPH_Y0, SPH_DEPTH,
                                   SPH_A, SPH_B, SPH_DIP, SPH_AZIMUTH,
                                   DP_DEFLATE, MU, NU)
interps_pre = build_interpolators(sig_pre, xn, yn)

print('Computing Okada dike stress (az=353°, exact geometry)...')
sig_dike = okada_dike_azimuth(X, Y, xn, yn,
                               DIKE_START[0], DIKE_START[1],
                               DIKE_END[0],   DIKE_END[1],
                               DIKE_AZIMUTH,
                               DIKE_DEPTH_TOP, DIKE_DEPTH_BOT,
                               DIKE_OPENING, MU, NU)
interps_dike = build_interpolators(sig_dike, xn, yn)


# Combine syn stress: evaluate dike interpolators back onto grid, then add
pts = np.column_stack([Y.ravel(), X.ravel()])
dike_on_grid = np.column_stack([ip(pts) for ip in interps_dike]).reshape(
    len(yn), len(xn), 3)
sig_syn_combined = sig_syn_sph + dike_on_grid
interps_syn = build_interpolators(sig_syn_combined, xn, yn)

print('Done.')

# ── Station locations ─────────────────────────────────────────────────────────
_sta = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon','lat','e','s'],
                   engine='python').set_index('s')
_sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'], _sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)

# ── σ₁ azimuth from stress tensor ────────────────────────────────────────────
def sigma1_azimuth(s2d):
    """
    Baillard's approach: eigendecompose 2×2 horizontal stress,
    return azimuth of most compressive eigenvector (°CW from N, 0–180).
    s2d = [σ_xx, σ_yy, σ_xy]
    """
    S = np.array([[s2d[0], s2d[2]],
                  [s2d[2], s2d[1]]])
    eigvals, eigvecs = np.linalg.eigh(S)   # ascending order
    # Most compressive = smallest eigenvalue = eigvals[0]
    v = eigvecs[:, 0]                      # (East, North) components
    return float(np.degrees(np.arctan2(v[0], v[1])) % 180.)

phi_pre_model = {}
phi_syn_model = {}

for sta in STATIONS:
    if sta not in _sta.index:
        continue
    sx, sy = _sta.loc[sta, 'x'], _sta.loc[sta, 'y']
    phi_pre_model[sta] = sigma1_azimuth(eval_stress_at(interps_pre, sx, sy))
    phi_syn_model[sta] = sigma1_azimuth(eval_stress_at(interps_syn, sx, sy))

# ── Observed φ ────────────────────────────────────────────────────────────────
print('Loading observed SWS data...')
dfs     = load_data(dt_error_max=0.1)
all_df  = pd.concat(dfs.values(), ignore_index=True)
periods = build_time_periods(all_df)
obs_phi, obs_dt, obs_N = build_observations(dfs, periods)
phi_pre_obs = {STATIONS[i]: obs_phi[i, 0] for i in range(len(STATIONS))}
phi_syn_obs = {STATIONS[i]: obs_phi[i, 1] for i in range(len(STATIONS))}

# ── Δφ table ──────────────────────────────────────────────────────────────────
print(f'\n{"Station":8s} {"φ_pre mod":>10} {"φ_pre obs":>10} '
      f'{"φ_syn mod":>10} {"φ_syn obs":>10} '
      f'{"Δφ mod":>9} {"Δφ obs":>9} {"diff":>8}')
print('─' * 85)

dphi_model_all = []
dphi_obs_all   = []

for sta in STATIONS:
    if sta not in phi_pre_model:
        continue
    p_pre_m = phi_pre_model[sta]; p_syn_m = phi_syn_model[sta]
    p_pre_o = phi_pre_obs[sta];   p_syn_o = phi_syn_obs[sta]

    dphi_m = float(np.degrees(np.arctan2(
        np.sin(2*np.radians(p_syn_m - p_pre_m)),
        np.cos(2*np.radians(p_syn_m - p_pre_m)))) / 2.)
    dphi_o = float(np.degrees(np.arctan2(
        np.sin(2*np.radians(p_syn_o - p_pre_o)),
        np.cos(2*np.radians(p_syn_o - p_pre_o)))) / 2.)

    dphi_model_all.append(dphi_m)
    dphi_obs_all.append(dphi_o)

    print(f'{sta[2:]:8s} {p_pre_m:>10.1f}° {p_pre_o:>10.1f}° '
          f'{p_syn_m:>10.1f}° {p_syn_o:>10.1f}° '
          f'{dphi_m:>+9.1f}° {dphi_o:>+9.1f}° {dphi_m-dphi_o:>+8.1f}°')

rms = float(np.sqrt(np.mean((np.array(dphi_model_all) -
                               np.array(dphi_obs_all))**2)))
print(f'\nΔφ RMS: {rms:.1f}°')
print(f'Spheroid: centre ({SPH_X0},{SPH_Y0}) km, depth {SPH_DEPTH} km, '
      f'a={SPH_A} km, b={SPH_B} km, dip={SPH_DIP}° toward {SPH_AZIMUTH}°')
print(f'Dike:     ({DIKE_START})→({DIKE_END}) km, az={DIKE_AZIMUTH}°, '
      f'depth=[{DIKE_DEPTH_TOP},{DIKE_DEPTH_BOT}] km, opening={DIKE_OPENING} m')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
sta_labels = [s[2:] for s in STATIONS if s in phi_pre_model]
n = len(sta_labels); x = np.arange(n)

for ax_i, (title, m_vals, o_vals) in enumerate([
    ('Pre-eruption φ',
     [phi_pre_model[s] for s in STATIONS if s in phi_pre_model],
     [phi_pre_obs[s]   for s in STATIONS if s in phi_pre_model]),
    ('Syn-eruption φ',
     [phi_syn_model[s] for s in STATIONS if s in phi_syn_model],
     [phi_syn_obs[s]   for s in STATIONS if s in phi_syn_model]),
    (f'Δφ  (RMS={rms:.1f}°)', dphi_model_all, dphi_obs_all),
]):
    ax = fig.add_subplot(gs[ax_i])
    ax.bar(x-0.2, m_vals, 0.35, color='steelblue', alpha=0.85, label='Model')
    ax.bar(x+0.2, o_vals, 0.35, color='tomato',    alpha=0.85, label='Observed')
    if ax_i < 2:
        ax.set_ylim(0, 180)
    else:
        ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(sta_labels)
    ax.set_ylabel('φ (°)' if ax_i < 2 else 'Δφ (°)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

fig.suptitle(
    f'Baillard-style model: σ₁ azimuth = predicted φ\n'
    f'Yang (1988) prolate spheroid (a={SPH_A} km, b={SPH_B} km, '
    f'dip={SPH_DIP}° @ {SPH_AZIMUTH}°) + Okada dike (az={DIKE_AZIMUTH}°, '
    f'{DIKE_OPENING} m opening)',
    fontsize=10, fontweight='bold')

out = '/Users/mhemmett/Seismology/axial-splitting-ml/results/baillard_simple_model.pdf'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'\nSaved {out}')
