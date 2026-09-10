#!/usr/bin/env python3
"""
synthetic_hudson_crack_scenarios.py

Physics/geometry module for the synthetic Hudson-crack-model splitting task (see the plan
this was built from). Defines three stress/crack scenarios sharing a common background:

  Scenario A (baseline): a regional background deviatoric stress whose TENSILE axis sits at
    azimuth 350deg/170deg (axial, same line) -- drives the stress-weighted opening of an
    isotropic (Fibonacci-sphere) Hudson crack population via hudson_crack_model.
    effective_stiffness (reused unmodified). PLUS a second, spatially-localized "structural"
    crack set fixed at the eastern ring-fault's strike (330deg), active only within
    FAULT_DAMAGE_HALFWIDTH_KM of the fault's surface trace -- representing the fault damage
    zone's own fabric, not stress-driven.
  Scenario B: Scenario A + a Mogi-type spheroidal sill stress source at 1.5 km depth, 0.5 km
    west of AXEC2 and 0.5 km north of AXEC3 -- radial crack opening above it is an EMERGENT
    property of feeding this local stress into the same Hudson physics, not a hardcoded rule.
  Scenario C: Scenario A + the eastern-wall dike's inflation stress -- a rectangular,
    2 m-opening tensile dislocation following PyLith's 5-waypoint/4-segment trace
    (pylith_axial/mesh/generate_mesh.py's DIKE_WAYPOINTS, same (East,North) km coordinate
    frame, confirmed by cross-checking P0 against AXEC1's own coordinates), but injected at
    1.5 km depth (not PyLith's shallow 0-0.5 km range) to match the sill's comparison depth.
    PLUS: per explicit user request, the sill's crack population is NOT removed here -- it
    stays present, but DEFLATED (SILL_DEFLATE_DP_PA = -SILL_DP_PA) rather than inflated, so
    the same stress-weighted sigmoid physics opens a different subset of the same cracks
    (rather than dropping the sill's contribution to zero, as the original Scenario C did).

Background-stress-azimuth convention: per explicit user correction, cracks "point" (fast S-
wave direction, crack face orientation) ALONG the direction of MAXIMUM COMPRESSION (sigma_1),
and OPEN (crack normal, tensile/extension direction) ORTHOGONAL to that (sigma_1 +/- 90) --
the standard EDA (extensive-dilatancy-anisotropy) convention: fast direction parallel to
sigma_Hmax, not to the tensile axis. So to get a fast/crack-pointing axis at azimuth A, the
background stress's COMPRESSIVE axis (phi1 in sws_forward_model.py's `forward()` formula,
reused verbatim here) is set directly to phi1=A. For A=350/170 (axial), BG_PHI1_DEG=350.0 --
compression at 350/170, extension/opening at 80/260. **Verify this empirically** via this
module's own `__main__` smoke test (Scenario A check) before trusting scenarios B/C -- if the
Hudson-model code's actual fast-vs-stress-axis behavior doesn't match this stated convention,
that's a real finding to flag, not something to silently sign-flip around.

Single-crack-set fast-axis convention (same reasoning, ALSO to be empirically verified): a
single fixed vertical crack set represents cracks whose FACE (fast direction) points along
the target azimuth A; since a crack's face is the plane perpendicular to its normal, the
crack-set's normal azimuth is set to A+90 (mod 180, symmetric either sign). For the eastern-
fault-damage-zone set (fast/pointing axis = fault strike 330 deg), the crack normal azimuth is
330+90 = 60 deg mod 180 (= 240 deg) -- which is also the eastern fault's own dip azimuth (60
deg), i.e. cracks whose face parallels the fault plane, opening in the fault's dip direction.

Dike stress: reuses sws_forward_model.py's Okada tensile-dislocation math (fux/fuy/chinnery)
exactly, generalized from that module's fixed-N-S-strike okada_dike_stress() to arbitrary
strike via a coordinate-rotation wrapper (rotate the observation grid into the segment's own
along-strike/cross-strike frame, run the unmodified Okada formulas, rotate the resulting
displacement field back to (East,North) before differencing) -- since the 4 PyLith segments
have genuinely different strikes (180, 350, 270, 30 deg), the original single-strike function
cannot represent the bent trace directly. Depth dependence is NOT modeled (same simplification
sws_forward_model.py's forward() already makes for its own dike term: a 2D surface-proxy
stress applied uniformly regardless of ray depth) -- flagged here, not silently different.
"""

import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from hudson_crack_model import (
    effective_stiffness, christoffel_splitting, fibonacci_sphere,
    _hudson_4d, _to_voigt, K_WATER, RHO, NU,
)
from sws_forward_model import mogi_stress_tensor_3d, MU_BG, NU_FIX

# ── Established local (East,North) km coordinates, this session's standard grid ──
# (INI_LON=-130.1, INI_LAT=45.9; same origin used throughout scripts/ this session and
# confirmed identical to pylith_axial/mesh/generate_mesh.py's origin via its own P0
# waypoint comment, "0.3 km east of AXEC1 (9.320, 5.519)", matching AXEC1's coords below.)
STATION_XY = {
    'AXAS1': (7.80, 3.74),
    'AXAS2': (6.65, 3.76),   # added once AXAS2's new-data 2015-2021 rerun completed
    'AXCC1': (7.06, 6.09),
    'AXEC1': (9.32, 5.52),
    'AXEC2': (9.78, 4.42),
    'AXEC3': (9.41, 4.02),
}

# ── Ring-fault geometry, both east and west (same convention as rose_map_by_station_
# 5stations_newdata_7period.py / spatial_map_fast_direction_5stations_newdata_7period.py /
# traveltime_anisotropy_7period_5stations_newdata.py) ────────────────────────────
EASTERN_FAULT_CENTER = (8.5, 4.0)   # km, surface-trace (z=FAULT_DEPTH_TOP_KM) position
WESTERN_FAULT_CENTER = (7.0, 4.0)   # km, surface-trace (z=FAULT_DEPTH_TOP_KM) position
FAULT_STRIKE_DEG = 330.0        # N30W -- SAME strike for both faults (only dip direction differs:
                                # east dips az 60 (toward E), west dips az 240 (toward W) -- doesn't
                                # change the crack-set's fast-direction target, which depends only
                                # on strike, mod 180)
FAULT_DIP_DEG = 66.0
EASTERN_FAULT_DIP_AZ_DEG = 60.0    # dips outward, toward the east
WESTERN_FAULT_DIP_AZ_DEG = 240.0   # dips outward, toward the west
FAULT_DEPTH_TOP_KM = 0.1        # depth (below seafloor) of the fault's surface trace -- same
                                # convention as rose_map_by_station_5stations_newdata_7period.py's
                                # DEPTH_TOP / COT66 depth-projection
FAULT_HALF_LEN_KM = 4.0         # along-strike half-length
FAULT_DAMAGE_HALFWIDTH_KM = 0.35   # perpendicular half-width of the "damage zone" gate --
                                    # thinned from 0.75 km per explicit user request

_FAULT_STRIKE_RAD = math.radians(FAULT_STRIKE_DEG)
_FAULT_S_HAT = np.array([math.sin(_FAULT_STRIKE_RAD), math.cos(_FAULT_STRIKE_RAD)])  # along strike
_FAULT_COT_DIP = math.cos(math.radians(FAULT_DIP_DEG)) / math.sin(math.radians(FAULT_DIP_DEG))

# Fault-damage-zone crack set: normal azimuth = strike - 90 (mod 180) so the resulting fast
# axis sits at the fault strike itself (see module docstring convention). Same for both faults
# since they share the same strike -- only the damage-zone GATING location differs (east vs.
# west fault trace), not the crack orientation itself.
FAULT_CRACK_NORMAL_AZ_DEG = FAULT_STRIKE_DEG - 90.0   # = 240 (= 60 mod 180)
_fcaz = math.radians(FAULT_CRACK_NORMAL_AZ_DEG)
FAULT_CRACK_NHAT = np.array([math.sin(_fcaz), math.cos(_fcaz), 0.0])
FAULT_CRACK_EPS = 0.02   # extra crack density for the damage-zone set (both faults) -- reduced
                         # from 0.05 per explicit user request (less dominant vs. sill/dike);
                         # western fault uses the SAME density, per explicit user request

# ── Background (regional-extension) crack density: decreases LINEARLY with depth ─
# per explicit user request. Reaches 0 at BACKGROUND_EPS_ZERO_DEPTH_KM (chosen to match the
# Z_MAX=4 km ray-validity cap used throughout the driver scripts) -- not pinned anywhere in
# the repo otherwise, a tunable default like the other magnitude choices in this module.
BACKGROUND_EPS_SURFACE = 0.02   # epsilon0 at the seafloor (z=0) -- previous fixed default
BACKGROUND_EPS_ZERO_DEPTH_KM = 4.0   # depth at which background crack density reaches 0


def background_epsilon0_at_depth(z_km):
    """Linear decay: BACKGROUND_EPS_SURFACE at z=0, down to 0 at
    BACKGROUND_EPS_ZERO_DEPTH_KM (clipped so it never goes negative below that)."""
    frac = max(0.0, 1.0 - z_km / BACKGROUND_EPS_ZERO_DEPTH_KM)
    return BACKGROUND_EPS_SURFACE * frac

# ── Sill (Scenario B): 0.5 km west of AXEC2, 0.5 km north of AXEC3, 1.5 km depth ──
SILL = dict(
    # Per explicit user request, the "v2" sill1/dike geometry correction chain is now the
    # ONLY geometry -- baked in here directly rather than kept as a separate override in
    # *_v2.py scripts. Original placement was (AXEC2.x-0.5, AXEC3.y+0.5) = (9.28, 4.52);
    # x0 here already reflects the net 0.05 km west shift from that original position (0.25
    # km west, then 0.2 km back east) applied identically to the dike's P0 below.
    x0=9.23,
    y0=4.52,
    d=1.5,
    R=0.25,   # km, source radius -- tunable default, similar scale to existing Kidiwela sources
)
SILL_DP_PA = 200e6   # Pa, inflation pressure (Scenario B, pre-eruption) -- bumped 4x (per
                     # explicit user request) so the sill's signal more substantially
                     # dominates over the weak background term
SILL_DEFLATE_DP_PA = -SILL_DP_PA   # Pa, Scenario C (syn-eruption): same sill crack population
                                   # stays present/active (not removed), but now DEFLATED
                                   # (negative pressure) -- per explicit user request, this
                                   # opens a DIFFERENT subset of the same crack population via
                                   # the same stress-weighted sigmoid physics, rather than
                                   # dropping the sill's contribution to zero

# ── Dike (Scenario C): PyLith's 5-waypoint/4-segment trace, reused verbatim ──────
DIKE_WAYPOINTS = [
    # Final geometry after the full correction chain (formerly kept as a separate "_v2"
    # override; per explicit user request now baked in here as the ONE dike geometry, used
    # by every scenario script -- single-sill and two-sills no longer differ in geometry, only
    # in whether a second sill is present):
    #   1. Re-anchored to sill 1's own location, dike starts there and moves south.
    #   2. West-traveling segment (P2->P3) start moved 0.75 km farther north, segment
    #      shortened 0.25 km (2.25 -> 2.0 km).
    #   3. Whole chain shifted 0.25 km west, then corrected 0.2 km back east (net 0.05 km
    #      west of the pre-shift position) -- applied to sill 1 too (same x0).
    #   4. West segment lengthened again, net +0.45 km total (2.0 -> 2.45 km... see P2/P3
    #      below for the actual resulting coordinates).
    #   5. Longest ("continues north") segment: net 5 deg WEST of its original azimuth,
    #      after several intermediate corrections (+10, then +5/-5/-5 adjustments) -- same
    #      length throughout (6.777 km).
    (9.230, 4.520),    # P0 = sill 1's location
    (9.230, 3.270),    # P1: south of P0, 1.25 km (extended 0.25 km from the original 1.0 km)
    (7.6025, 9.8482),  # P2: longest segment, 6.777 km at az ~341 (5 deg west of original ~346)
    (5.4025, 9.8482),  # P3: west segment, ~2.45 km total (see history above)
    (6.4375, 13.7122), # P4: unchanged 4 km length at az 15, recomputed from the new P3
]
# NOTE: this is now a MODIFIED copy of pylith_axial/mesh/generate_mesh.py's DIKE_WAYPOINTS --
# that file's own geometry is UNCHANGED (separate PyLith modeling effort); only this
# synthetic-splitting model's dike trace has been adjusted per the above user requests.
DIKE_OPENING_M = 6.0   # bumped from the physically-estimated 2 m (per explicit user request)
                       # so the dike's signal more substantially dominates the synthetic
                       # splitting -- a deliberate dominance amplification, not a revised
                       # physical estimate of the real dike's opening width; slightly reduced
                       # from an earlier 8.0 m per follow-up user request
DIKE_DEPTH_TOP_KM = 1.0     # centered at 1.5 km depth (user-specified), 1 km width -- tunable
DIKE_DEPTH_BOT_KM = 2.0

# ── Sill/dike "overprint" gate -- suppresses the fault's fixed crack set ─────────
# Per explicit user request: near the sill (Scenario B) or dike (Scenario C), the sill/dike's
# OWN stress-driven crack pattern should dominate/overprint the fault's fixed crack set, not
# be diluted by it -- otherwise the fault's strong deterministic contribution drowns out the
# comparatively weak sill/dike stress signal wherever a ray sample point falls in both zones.
# Same margin/halfwidth values as synthetic_hudson_structure_map.py's SILL_GATE_MARGIN_KM/
# DIKE_GATE_HALFWIDTH_KM (kept here too since this is now core physics, not just plotting).
SILL_OVERPRINT_MARGIN_KM = SILL['R'] * 2.5
DIKE_OVERPRINT_HALFWIDTH_KM = 0.5


def _dist_to_dike_km(x, y):
    """Minimum distance from (x,y) to the dike's 4-segment polyline."""
    best = np.inf
    for (x0, y0), (x1, y1) in zip(DIKE_WAYPOINTS[:-1], DIKE_WAYPOINTS[1:]):
        seg = np.array([x1 - x0, y1 - y0])
        seg_len2 = seg[0]**2 + seg[1]**2
        t = 0.0 if seg_len2 == 0 else max(0.0, min(1.0, ((x - x0) * seg[0] + (y - y0) * seg[1]) / seg_len2))
        px, py = x0 + t * seg[0], y0 + t * seg[1]
        best = min(best, math.hypot(x - px, y - py))
    return best

# ── Background deviatoric stress (reused verbatim from sws_forward_model.forward()) ──
P_LITHO_PA = 65e6      # matches sws_forward_model.py's forward() default
B_BG_PA = 3.0e6        # matches sws_forward_model.py's THETA0[11] (3.0 MPa)
BG_PHI1_DEG = 350.0    # compressive axis (=fast/crack-pointing axis, per user's EDA
                       # convention correction); extension/opening axis at 350+90=80 deg


def background_stress_tensor(phi1_deg=BG_PHI1_DEG, B_bg_pa=B_BG_PA, p_litho_pa=P_LITHO_PA):
    """Regional deviatoric + lithostatic background stress, tensile axis at phi1+90 deg.
    Identical formula to sws_forward_model.py's forward() (verbatim, not re-derived)."""
    phi1 = np.radians(phi1_deg)
    c1 = np.array([np.sin(phi1), np.cos(phi1)])     # compressive axis direction
    e1 = np.array([-np.cos(phi1), np.sin(phi1)])    # tensile axis direction (perpendicular)
    sigma = np.zeros((3, 3))
    for ii in range(2):
        for jj in range(2):
            sigma[ii, jj] = B_bg_pa * (e1[ii] * e1[jj] - c1[ii] * c1[jj])
    sigma[0, 0] -= p_litho_pa
    sigma[1, 1] -= p_litho_pa
    sigma[2, 2] -= p_litho_pa
    return sigma


def fault_perp_distance_km(x_km, y_km, z_km, center, dip_az_deg):
    """Perpendicular horizontal distance from (x,y) to a fault's PLANE at depth z_km -- i.e.
    the fault's surface trace, depth-projected outward along its own dip azimuth as depth
    increases (same COT66 convention as rose_map_by_station_5stations_newdata_7period.py's
    _faults(): horiz = (z - FAULT_DEPTH_TOP_KM) * cot(dip), shifted along dip_az_deg), NOT
    just the fixed surface trace applied uniformly at every depth (these faults are outward-
    dipping at 66 deg, not vertical -- ignoring this made the damage-zone gate wrong by up to
    ~1.3 km at 3 km depth, more than 3x the gate's own half-width).

    Returns +inf if the along-strike projection falls outside the finite segment (length
    2*FAULT_HALF_LEN_KM, centered at `center`, strike FAULT_STRIKE_DEG -- shared by both
    faults, only dip_az_deg differs)."""
    horiz = max(z_km - FAULT_DEPTH_TOP_KM, 0.0) * _FAULT_COT_DIP
    da = math.radians(dip_az_deg)
    cx_z = center[0] + horiz * math.sin(da)
    cy_z = center[1] + horiz * math.cos(da)

    dx = x_km - cx_z
    dy = y_km - cy_z
    along = dx * _FAULT_S_HAT[0] + dy * _FAULT_S_HAT[1]
    if abs(along) > FAULT_HALF_LEN_KM:
        return np.inf
    perp = dx * _FAULT_S_HAT[1] - dy * _FAULT_S_HAT[0]   # 2D cross product
    return abs(perp)


def in_eastern_fault_damage_zone(x_km, y_km, z_km):
    return fault_perp_distance_km(x_km, y_km, z_km, EASTERN_FAULT_CENTER,
                                  EASTERN_FAULT_DIP_AZ_DEG) <= FAULT_DAMAGE_HALFWIDTH_KM


def in_western_fault_damage_zone(x_km, y_km, z_km):
    return fault_perp_distance_km(x_km, y_km, z_km, WESTERN_FAULT_CENTER,
                                  WESTERN_FAULT_DIP_AZ_DEG) <= FAULT_DAMAGE_HALFWIDTH_KM


# ── Sill (Mogi-type) stress at an arbitrary 3D point ─────────────────────────────

def sill_stress_tensor(x_km, y_km, z_km, dp_pa=SILL_DP_PA):
    """Reuses sws_forward_model.py's Kelvin-image Mogi stress solution unmodified."""
    return mogi_stress_tensor_3d(x_km, y_km, z_km, SILL, dp_pa, MU_BG, NU_FIX)


# ── Dike (Okada tensile dislocation) stress, arbitrary strike per segment ────────

def _okada_seg_grid_stress(X, Y, xn, yn, xc, yc, strike_deg, length_km,
                           depth_top_km, depth_bot_km, opening_m, mu=MU_BG, nu=NU_FIX):
    """One PyLith dike segment's surface-proxy stress on a (ny,nx,3) grid, generalized
    from sws_forward_model.okada_dike_stress() to an arbitrary strike via a coordinate
    rotation wrapper -- the underlying fux/fuy/chinnery formulas are unchanged."""
    lam = 2 * nu * mu / (1 - 2 * nu)
    theta = math.radians(strike_deg)
    s_hat = np.array([math.sin(theta), math.cos(theta)])    # along strike
    n_hat = np.array([math.cos(theta), -math.sin(theta)])   # cross strike

    dX = X - xc; dY = Y - yc
    xi = dX * s_hat[0] + dY * s_hat[1]     # along strike [km]
    eta = dX * n_hat[0] + dY * n_hat[1]    # cross strike [km]
    xi_m, eta_m = xi * 1e3, eta * 1e3
    q = eta_m

    def R(xi_, e_, p_): return np.sqrt(xi_**2 + e_**2 + p_**2) + 1e-10
    def fux(xi_, e_, p_):
        r = R(xi_, e_, p_); rp = r + p_ + 1e-10; re = r + e_ + 1e-10
        return -(1 - 2 * nu) * xi_ / re - xi_ * q / r / rp
    def fuy(xi_, e_, p_):
        r = R(xi_, e_, p_); rp = r + p_ + 1e-10; re = r + e_ + 1e-10
        th = np.arctan2(xi_ * e_, q * r + 1e-10)
        return xi_**2 / r / rp - (1 - 2 * nu) / re + (1 - 2 * nu) * th
    def chinnery(f):
        L = length_km * 1e3
        x1, x2 = xi_m - L / 2, xi_m + L / 2
        e1, e2 = depth_top_km * 1e3, depth_bot_km * 1e3
        return f(x2, e2, q) - f(x2, e1, q) - f(x1, e2, q) + f(x1, e1, q)

    sc = opening_m / (2 * np.pi)
    U_cross = sc * chinnery(fuy)   # perpendicular-to-strike displacement (local frame)
    U_along = sc * chinnery(fux)   # along-strike displacement (local frame)

    # Rotate local (cross-strike, along-strike) displacement back to global (East, North)
    Ux = U_cross * n_hat[0] + U_along * s_hat[0]
    Uy = U_cross * n_hat[1] + U_along * s_hat[1]

    dm = (xn[1] - xn[0]) * 1e3; dm2 = (yn[1] - yn[0]) * 1e3
    ny, nx = Ux.shape
    sig = np.zeros((ny, nx, 3))
    e11 = (Ux[1:-1, 2:] - Ux[1:-1, :-2]) / (2 * dm)
    e22 = (Uy[2:, 1:-1] - Uy[:-2, 1:-1]) / (2 * dm2)
    e12 = 0.5 * ((Ux[2:, 1:-1] - Ux[:-2, 1:-1]) / (2 * dm2) +
                (Uy[1:-1, 2:] - Uy[1:-1, :-2]) / (2 * dm))
    tr = e11 + e22
    sig[1:-1, 1:-1, 0] = lam * tr + 2 * mu * e11
    sig[1:-1, 1:-1, 1] = lam * tr + 2 * mu * e22
    sig[1:-1, 1:-1, 2] = 2 * mu * e12
    return sig


def build_dike_stress_interpolators(xn, yn, opening_m=DIKE_OPENING_M,
                                    depth_top_km=DIKE_DEPTH_TOP_KM,
                                    depth_bot_km=DIKE_DEPTH_BOT_KM):
    """Sum all 4 PyLith waypoint segments' rotated Okada stress onto one grid, then build
    3 RegularGridInterpolators (sigma_xx, sigma_yy, sigma_xy), same pattern as
    sws_forward_model.build_interpolators(). Depth-independent surface-proxy stress, applied
    uniformly regardless of ray depth (same simplification as sws_forward_model.forward())."""
    X, Y = np.meshgrid(xn, yn)
    total = np.zeros((len(yn), len(xn), 3))
    for (x0, y0), (x1, y1) in zip(DIKE_WAYPOINTS[:-1], DIKE_WAYPOINTS[1:]):
        xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        length_km = math.hypot(x1 - x0, y1 - y0)
        strike_deg = math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0
        total += _okada_seg_grid_stress(X, Y, xn, yn, xc, yc, strike_deg, length_km,
                                        depth_top_km, depth_bot_km, opening_m)
    interps = []
    for k in range(3):
        interps.append(RegularGridInterpolator((yn, xn), total[:, :, k],
                                               method='linear', bounds_error=False,
                                               fill_value=0.))
    return interps


def eval_dike_stress(interps, x_km, y_km):
    """[sigma_xx, sigma_yy, sigma_xy] at (x_km, y_km); sigma_zz/xz/yz = 0 (surface-proxy
    simplification, same as sws_forward_model.py's dike term)."""
    pt = np.array([[y_km, x_km]])
    sxx, syy, sxy = (float(ip(pt)[0]) for ip in interps)
    sigma = np.zeros((3, 3))
    sigma[0, 0], sigma[1, 1] = sxx, syy
    sigma[0, 1] = sigma[1, 0] = sxy
    return sigma


# ── Scenario dispatch ─────────────────────────────────────────────────────────

SCENARIOS = ('A', 'B', 'C')


def total_stress_and_crack_sets(scenario, x_km, y_km, z_km, dike_interps=None):
    """Returns (sigma_total_3x3_Pa, extra_crack_sets) where extra_crack_sets is a list of
    (n_hat, epsilon) tuples for fixed (non-stress-weighted) crack sets to add on top of the
    background isotropic (Fibonacci-sphere) population at this point."""
    sigma = background_stress_tensor()

    if scenario == 'B':
        sigma = sigma + sill_stress_tensor(x_km, y_km, z_km, dp_pa=SILL_DP_PA)
    elif scenario == 'C':
        if dike_interps is None:
            raise ValueError('Scenario C requires dike_interps (build_dike_stress_interpolators())')
        sigma = sigma + eval_dike_stress(dike_interps, x_km, y_km)
        # Sill crack population stays present (not removed) but now DEFLATED -- same stress-
        # weighted sigmoid physics opens a different subset of the same cracks (see
        # SILL_DEFLATE_DP_PA's comment).
        sigma = sigma + sill_stress_tensor(x_km, y_km, z_km, dp_pa=SILL_DEFLATE_DP_PA)
    elif scenario != 'A':
        raise ValueError(f'Unknown scenario {scenario!r}')

    # Sill/dike overprint: within SILL_OVERPRINT_MARGIN_KM of the sill (Scenario B) or
    # DIKE_OVERPRINT_HALFWIDTH_KM of the dike trace (Scenario C), the sill/dike's own
    # stress-driven pattern overprints the fault's fixed crack set -- so that set is
    # suppressed here, even inside a fault damage zone (see constants' comment above).
    sill_overprint = (scenario == 'B' and
                      math.hypot(x_km - SILL['x0'], y_km - SILL['y0'])
                      <= SILL['R'] + SILL_OVERPRINT_MARGIN_KM)
    dike_overprint = (scenario == 'C' and
                      _dist_to_dike_km(x_km, y_km) <= DIKE_OVERPRINT_HALFWIDTH_KM)

    extra_crack_sets = []
    if not (sill_overprint or dike_overprint):
        if (in_eastern_fault_damage_zone(x_km, y_km, z_km) or
                in_western_fault_damage_zone(x_km, y_km, z_km)):
            extra_crack_sets.append((FAULT_CRACK_NHAT, FAULT_CRACK_EPS))

    return sigma, extra_crack_sets


def effective_stiffness_with_extra_cracks(sigma, extra_crack_sets, mu=MU_BG,
                                          epsilon0=0.02, K_f=K_WATER, r_crack=0.01,
                                          sigma_c=40e6, n_hats=None, nu=NU):
    """hudson_crack_model.effective_stiffness() for the stress-weighted background
    population, PLUS additive fixed-orientation crack-set contributions (fault damage zone)
    -- valid because Hudson's first-order stiffness correction is additive across crack sets
    (see hudson_crack_model.py's own single-crack-set unit test)."""
    if n_hats is None:
        n_hats = fibonacci_sphere(200)
    C_eff, open_frac = effective_stiffness(sigma, mu, epsilon0, K_f, r_crack, sigma_c, n_hats, nu)
    if extra_crack_sets:
        DC4 = np.zeros((3, 3, 3, 3))
        for n_hat, eps in extra_crack_sets:
            DC4 += _hudson_4d(n_hat, eps, mu, nu, K_f, r_crack)
        C_eff = C_eff + _to_voigt(DC4)
    return C_eff, open_frac


def integrate_synthetic_splitting(scenario, ray_xyz, mu=MU_BG, rho=RHO,
                                  dike_interps=None, n_hats=None, **hudson_kwargs):
    """Accumulate synthetic (phi, dt) along a traced ray (N_pts,3) for the given scenario --
    mirrors hudson_crack_model.integrate_splitting's path-weighted-circular-mean/summed-dt
    logic exactly, just using effective_stiffness_with_extra_cracks() per point instead of
    the plain stress-only effective_stiffness()."""
    mid = 0.5 * (ray_xyz[:-1] + ray_xyz[1:])
    ds_km = np.linalg.norm(np.diff(ray_xyz, axis=0), axis=1)
    ds_m = ds_km * 1e3
    p_hat = ray_xyz[-1] - ray_xyz[0]
    p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-30)

    N = len(mid)
    phis = np.zeros(N); dVs = np.zeros(N); VSs = np.zeros(N)
    for i in range(N):
        x_km, y_km, z_km = mid[i]
        sigma, extra = total_stress_and_crack_sets(scenario, x_km, y_km, z_km, dike_interps)
        # Background crack density decreases linearly with depth (per explicit user
        # request) -- only applied if the caller hasn't explicitly overridden epsilon0.
        point_kwargs = dict(hudson_kwargs)
        point_kwargs.setdefault('epsilon0', background_epsilon0_at_depth(z_km))
        C_eff, _ = effective_stiffness_with_extra_cracks(sigma, extra, mu, n_hats=n_hats,
                                                         **point_kwargs)
        phis[i], dVs[i], VSs[i], _ = christoffel_splitting(C_eff, p_hat, rho)

    w = dVs * ds_m; w_sum = w.sum()
    if w_sum < 1e-30:
        phi_pred = float(np.mean(phis))
    else:
        ang2 = 2. * np.radians(phis)
        phi_pred = float(np.degrees(
            np.arctan2((w * np.sin(ang2)).sum(), (w * np.cos(ang2)).sum())) / 2. % 180.)

    V_bar = VSs.mean() if VSs.mean() > 0 else 3500.
    dt_pred = float(np.sum(dVs / max(V_bar**2, 1.) * ds_m))
    return phi_pred, dt_pred


if __name__ == '__main__':
    # ── Smoke test: verify the sign/rotation conventions before trusting B/C ────
    print('=' * 70)
    print('Convention checks (vertical ray, p_hat=[0,0,1])')
    print('=' * 70)

    p_vert = np.array([0., 0., 1.])
    n_hats200 = fibonacci_sphere(200)

    # A. Background-only, away from the fault damage zone -> expect phi ~= 350/170
    sigma_bg = background_stress_tensor()
    C_bg, _ = effective_stiffness(sigma_bg, MU_BG, epsilon0=0.02, sigma_c=40e6, n_hats=n_hats200)
    phi_bg, dV_bg, VS1, VS2 = christoffel_splitting(C_bg, p_vert, RHO)
    print(f'\nA. Background only: phi={phi_bg:.1f} deg (target ~350/170), '
          f'dV={dV_bg:.1f} m/s, VS1={VS1:.0f} VS2={VS2:.0f}')

    # A (fault set). Fault-damage-zone crack set alone (on top of background) -> expect
    # phi pulled toward 330 relative to background-only.
    C_flt, _ = effective_stiffness_with_extra_cracks(
        sigma_bg, [(FAULT_CRACK_NHAT, FAULT_CRACK_EPS)], n_hats=n_hats200, sigma_c=40e6)
    phi_flt, dV_flt, _, _ = christoffel_splitting(C_flt, p_vert, RHO)
    print(f'B. Background + fault-damage-zone crack set: phi={phi_flt:.1f} deg '
          f'(target ~330, pulled from {phi_bg:.1f}), dV={dV_flt:.1f} m/s')

    # C. Sill stress alone at a point directly above the sill (radial pattern check at 4
    # azimuths around the sill, same depth as the sill's own -0.5 km overburden)
    print(f'\nC. Sill-only stress at 4 azimuths around the sill (radius 0.6 km, z=1.0 km):')
    for az_deg in [0, 90, 180, 270]:
        az = math.radians(az_deg)
        x_km = SILL['x0'] + 0.6 * math.sin(az)
        y_km = SILL['y0'] + 0.6 * math.cos(az)
        sigma_sill = sigma_bg + sill_stress_tensor(x_km, y_km, 1.0)
        C_s, _ = effective_stiffness(sigma_sill, MU_BG, epsilon0=0.02, sigma_c=40e6, n_hats=n_hats200)
        phi_s, dV_s, _, _ = christoffel_splitting(C_s, p_vert, RHO)
        print(f'   az={az_deg:>3} deg from sill: phi={phi_s:.1f} deg, dV={dV_s:.1f} m/s')

    # D. Dike stress alone at a point near the dike trace
    print(f'\nD. Dike-only stress near the P1-P2 segment midpoint:')
    xn = np.arange(4., 12.01, 0.1); yn = np.arange(0., 15.01, 0.1)
    dike_interps = build_dike_stress_interpolators(xn, yn)
    x_km, y_km = 9.0, 7.5   # near P1(9.62,4.52)->P2(8.57,10.46) segment
    sigma_dike = sigma_bg + eval_dike_stress(dike_interps, x_km, y_km)
    C_d, _ = effective_stiffness(sigma_dike, MU_BG, epsilon0=0.02, sigma_c=40e6, n_hats=n_hats200)
    phi_d, dV_d, _, _ = christoffel_splitting(C_d, p_vert, RHO)
    print(f'   ({x_km},{y_km}): phi={phi_d:.1f} deg, dV={dV_d:.1f} m/s '
          f'(background-only there would be phi={phi_bg:.1f})')

    print('\nDone. Compare phi_bg/phi_flt/sill-azimuth-pattern/phi_d against the intended '
          'targets before trusting the full driver script.')
