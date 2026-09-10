#!/usr/bin/env python3
"""
synthetic_prolate_spheroid_axas1_axas2_stress_field.py

Predicted surface stress field from a RAPIDLY INFLATING VERTICAL prolate spheroid source
between stations AXAS1 and AXAS2, spanning 3.0-5.0 km depth, per explicit user request.

Physics: Yang, Davis & Dieterich (1988) half-space solution for a pressurized prolate
spheroid, via the Eshelby (1957) equivalent line-distribution of Mogi (1958) point sources
along the major axis (Gauss-Legendre quadrature) -- the SAME formulation already implemented
in this repo's baillard_simple_model.py (yang_surface_displacement/yang_surface_stress),
reused here (adapted to be self-contained and import this repo's own sws_forward_model.py,
not the external axial-splitting-ml path that reference script used).

Source geometry:
  Location (x0, y0): the midpoint of AXAS1 and AXAS2 (per explicit user request) --
    computed from the station file, not hardcoded.
  Depth to centre: 4.0 km (midpoint of the requested 3.0-5.0 km depth range).
  Semi-major axis a: 1.0 km (half of the requested 2.0 km depth span; VERTICAL orientation --
    dip=0 in this function's own convention, where uz_hat=cos(dip_deg), so the major axis
    points straight down and the full vertical extent is 2*a = 2.0 km, exactly 3.0-5.0 km).
  Semi-minor axis b: NOT specified by the user -- ASSUMED 0.3 km (aspect ratio a:b ~ 3.3:1,
    a moderately elongated body) so the model can run; flagged here and in the figure caption
    since it directly sets the source's radial footprint and stress-perturbation magnitude --
    change SPH_B below and re-run if a different aspect ratio is wanted.
  Inflation pressure dP: NOT specified ("rapidly inflating") -- ASSUMED +200 MPa (matching the
    SILL_DP_PA convention used for the other inflating sources in
    synthetic_hudson_crack_scenarios.py this session, for consistency), positive = inflation.
    Also flagged -- change SPH_DP_PA below and re-run for a different magnitude.

Elastic constants: MU_BG, NU_FIX from sws_forward_model.py (same as every other synthetic
source in this project).

Output: ONE map, grayscale bathymetry + structures (spheroid's plan-view footprint circle,
radius b, since a purely vertical major axis projects to a circle in map view; ring faults for
context; AXAS1/AXAS2 highlighted as the bounding stations, other stations shown white) +
a sequential-colormap background of the predicted horizontal DIFFERENTIAL stress magnitude
(sigma1 - sigma2, the two horizontal principal stresses) + a quiver of the predicted
MOST-COMPRESSIVE horizontal principal stress (sigma1) orientation on top -- the orientation
crack-induced shear-wave splitting fast directions would be expected to align with, per the
EDA (Crampin) stress-induced-anisotropy model.

Produces:
    synthetic_prolate_spheroid_axas1_axas2_stress_field.pdf
(1 page, 1 panel) -- a NEW file, does not touch any existing output.

Run with:
    python3 synthetic_prolate_spheroid_axas1_axas2_stress_field.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

from sws_forward_model import MU_BG, NU_FIX
import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'synthetic_prolate_spheroid_axas1_axas2_stress_field.pdf')

# ── Source geometry (see module docstring for how each value was chosen) ───────────────────
SPH_DEPTH = 4.0     # km, centre depth = midpoint of requested 3.0-5.0 km range
SPH_A = 1.0         # km, semi-major axis = half of the requested 2.0 km depth span
SPH_B = 0.3         # km, semi-minor axis -- ASSUMED (not specified by user)
SPH_DIP = 0.0       # deg -- 0 in this function's own convention = fully VERTICAL major axis
SPH_AZIMUTH = 0.0   # deg -- irrelevant when dip=0 (horizontal components vanish)
SPH_DP_PA = 200e6   # Pa -- ASSUMED "rapid" inflation pressure (not specified by user)

MU, NU = MU_BG, NU_FIX
LAM = 2 * NU * MU / (1 - 2 * NU)

GRID_STEP_KM = 0.1
N_GAUSS = 80


def yang_surface_displacement(X_km, Y_km, x0, y0, depth, a, b, dip_deg, az_deg, dP, mu, nu,
                              n_gauss=N_GAUSS):
    """Yang et al. (1988) prolate-spheroid half-space surface displacement (Ux, Uy) [m], via
    Gauss-Legendre integration of Eshelby-weighted Mogi (1958) point sources along the major
    axis. See baillard_simple_model.py's yang_surface_displacement for the original derivation
    this is adapted from (self-contained here, no external-repo imports)."""
    x0m, y0m, dm = x0 * 1e3, y0 * 1e3, depth * 1e3
    am, bm = a * 1e3, b * 1e3
    X_m, Y_m = X_km * 1e3, Y_km * 1e3

    e_f = np.sqrt(max(1 - (bm / am) ** 2, 0))
    c_m = am * e_f

    dip_r, az_r = np.radians(dip_deg), np.radians(az_deg)
    ux_hat = np.sin(dip_r) * np.sin(az_r)
    uy_hat = np.sin(dip_r) * np.cos(az_r)
    uz_hat = np.cos(dip_r)

    DV_total = (4 * np.pi / 3.0) * am * bm ** 2 * dP / mu
    K_weight = 2.0 * DV_total / (np.pi * c_m ** 2) if c_m > 1.0 else DV_total

    xi, wi = np.polynomial.legendre.leggauss(n_gauss)
    s_pts, w_pts = c_m * xi, c_m * wi

    Ux = np.zeros_like(X_m)
    Uy = np.zeros_like(X_m)
    for k in range(n_gauss):
        s = s_pts[k]
        dV = K_weight * np.sqrt(max(c_m ** 2 - s ** 2, 0)) * w_pts[k]
        xs, ys, zs = x0m + s * ux_hat, y0m + s * uy_hat, dm + s * uz_hat
        if zs <= 0:
            continue
        dx, dy = X_m - xs, Y_m - ys
        R3 = (dx ** 2 + dy ** 2 + zs ** 2) ** 1.5 + 1e-6
        A = dV * (1 - nu) / np.pi / R3
        Ux += A * dx
        Uy += A * dy
    return Ux, Uy


def yang_surface_stress(X_km, Y_km, x0, y0, depth, a, b, dip_deg, az_deg, dP, mu, nu):
    """Surface stress (sigma_xx, sigma_yy, sigma_xy) [Pa] from the Yang spheroid
    displacement field, via central finite differences."""
    Ux, Uy = yang_surface_displacement(X_km, Y_km, x0, y0, depth, a, b, dip_deg, az_deg, dP, mu, nu)
    dx_km = X_km[0, 1] - X_km[0, 0]
    dy_km = Y_km[1, 0] - Y_km[0, 0]
    dm, dm2 = dx_km * 1e3, dy_km * 1e3
    ny, nx = Ux.shape
    sig = np.zeros((ny, nx, 3))
    e11 = (Ux[1:-1, 2:] - Ux[1:-1, :-2]) / (2 * dm)
    e22 = (Uy[2:, 1:-1] - Uy[:-2, 1:-1]) / (2 * dm2)
    e12 = 0.5 * ((Ux[2:, 1:-1] - Ux[:-2, 1:-1]) / (2 * dm2) +
                (Uy[1:-1, 2:] - Uy[1:-1, :-2]) / (2 * dm))
    tr = e11 + e22
    sig[1:-1, 1:-1, 0] = LAM * tr + 2 * mu * e11
    sig[1:-1, 1:-1, 1] = LAM * tr + 2 * mu * e22
    sig[1:-1, 1:-1, 2] = 2 * mu * e12
    return sig


def principal_stress_2d(sxx, syy, sxy):
    """Eigen-decompose the 2x2 horizontal stress tensor at every grid point.
    Returns (sigma1, sigma2, azimuth1_deg) -- sigma1 >= sigma2 (sigma1 = most compressive
    if using a compression-positive sign convention; here dP>0/inflation gives tension in
    the far field per the elastic convention used, so azimuth1 is reported as the direction
    of the numerically LARGER-magnitude compressive eigenvalue for EDA-style interpretation --
    see note in main())."""
    ny, nx = sxx.shape
    sigma1 = np.full((ny, nx), np.nan)
    sigma2 = np.full((ny, nx), np.nan)
    az1 = np.full((ny, nx), np.nan)
    for iy in range(ny):
        for ix in range(nx):
            S = np.array([[sxx[iy, ix], sxy[iy, ix]], [sxy[iy, ix], syy[iy, ix]]])
            if not np.all(np.isfinite(S)):
                continue
            eigvals, eigvecs = np.linalg.eigh(S)
            # eigh returns ascending order; take the MOST COMPRESSIVE (algebraically smallest,
            # i.e. most negative under a tension-positive convention) as sigma1/az1 -- the
            # EDA-relevant direction (crack planes/fast directions align parallel to this).
            sigma2[iy, ix] = eigvals[1]
            sigma1[iy, ix] = eigvals[0]
            v = eigvecs[:, 0]
            az1[iy, ix] = np.degrees(np.arctan2(v[0], v[1])) % 180.0
    return sigma1, sigma2, az1


def main():
    T._setup_environment(need_tracer=False)
    x0 = float((T._sta_df.loc['AXAS1', 'x'] + T._sta_df.loc['AXAS2', 'x']) / 2.0)
    y0 = float((T._sta_df.loc['AXAS1', 'y'] + T._sta_df.loc['AXAS2', 'y']) / 2.0)
    print(f'Source location (midpoint AXAS1-AXAS2): ({x0:.3f}, {y0:.3f}) km')
    print(f'Depth to centre: {SPH_DEPTH} km, a={SPH_A} km, b={SPH_B} km (ASSUMED), '
          f'dP={SPH_DP_PA/1e6:.0f} MPa (ASSUMED, inflating)')

    xn = np.arange(T.X_START, T.X_END + 1e-9, GRID_STEP_KM)
    yn = np.arange(T.Y_START, T.Y_END + 1e-9, GRID_STEP_KM)
    X, Y = np.meshgrid(xn, yn)

    print('Computing Yang (1988) prolate-spheroid surface stress field...')
    sig = yang_surface_stress(X, Y, x0, y0, SPH_DEPTH, SPH_A, SPH_B, SPH_DIP, SPH_AZIMUTH,
                              SPH_DP_PA, MU, NU)
    sxx, syy, sxy = sig[..., 0], sig[..., 1], sig[..., 2]
    sigma1, sigma2, az1 = principal_stress_2d(sxx, syy, sxy)
    diff_stress_mpa = (sigma2 - sigma1) / 1e6   # magnitude of horizontal stress anisotropy

    print(f'  Differential horizontal stress range: '
          f'{np.nanmin(diff_stress_mpa):.3f} to {np.nanmax(diff_stress_mpa):.3f} MPa')

    fig, ax = plt.subplots(figsize=(8, 9))
    T._bathy(ax)

    vmax = float(np.nanpercentile(diff_stress_mpa, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    cs = ax.contourf(X, Y, diff_stress_mpa, levels=np.linspace(0, vmax, 15),
                     cmap='Blues', alpha=0.75, zorder=4, extend='max')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, format=FormatStrFormatter('%.2f'))
    cb.set_label('Horizontal differential stress $\\sigma_2-\\sigma_1$ (MPa)', fontsize=8)

    # Quiver: predicted most-compressive horizontal principal-stress orientation (the EDA
    # fast-direction proxy), subsampled for a legible tick density.
    step = max(1, int(round(0.5 / GRID_STEP_KM)))
    Xs, Ys, azs = X[::step, ::step], Y[::step, ::step], az1[::step, ::step]
    mask = np.isfinite(azs)
    r = np.radians(azs[mask])
    u, v = np.sin(r), np.cos(r)
    tick_kw = dict(scale=22, width=0.0035, headlength=0, headaxislength=0, headwidth=0,
                  pivot='middle', color='k', alpha=0.85, zorder=9)
    ax.quiver(Xs[mask], Ys[mask], u, v, **tick_kw)
    ax.quiver(Xs[mask], Ys[mask], -u, -v, **tick_kw)

    # Structures: spheroid plan-view footprint (a vertical major axis projects to a circle of
    # radius b), ring faults for context.
    circ = plt.Circle((x0, y0), SPH_B, fill=False, color='0.1', lw=2.0, zorder=10)
    ax.add_patch(circ)
    T._faults(ax, SPH_DEPTH)

    for sta, row in T._sta_df.iterrows():
        highlight = sta in ('AXAS1', 'AXAS2')
        color = '#FFD700' if highlight else 'white'
        ax.plot(row['x'], row['y'], '^', ms=8 if highlight else 6, mfc=color, mec='k',
               mew=0.9, zorder=13)
        dx, dy = T.LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, T.STA_DISPLAY.get(sta, sta), fontsize=7,
               fontweight='bold', zorder=13)

    ax.set_xlim(T.X_START, T.X_END)
    ax.set_ylim(T.Y_START, T.Y_END)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('East [km]', fontsize=9)
    ax.set_ylabel('North [km]', fontsize=9)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8,
                      label='AXAS1 / AXAS2 (bounding stations)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='white', mec='k', ms=6,
                      label='Other stations'),
        mlines.Line2D([], [], color='0.1', lw=2.0, label='Spheroid footprint (radius=b)'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
        mlines.Line2D([], [], color='k', lw=1.2, label='Predicted $\\sigma_1$ orientation'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=7.5, framealpha=0.9)

    ax.set_title(
        'Predicted surface stress field: rapidly inflating VERTICAL prolate spheroid\n'
        f'between AXAS1/AXAS2, depth {SPH_DEPTH-SPH_A:.1f}-{SPH_DEPTH+SPH_A:.1f} km '
        f'(centre {SPH_DEPTH:.1f} km, a={SPH_A:.1f} km), b={SPH_B:.1f} km (assumed), '
        f'dP=+{SPH_DP_PA/1e6:.0f} MPa (assumed)',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
