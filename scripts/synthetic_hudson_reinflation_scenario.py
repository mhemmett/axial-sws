#!/usr/bin/env python3
"""
synthetic_hudson_reinflation_scenario.py

Adds a 4th synthetic Hudson-crack scenario, "D" (Re-Inflation), per explicit user request, for
the post-eruption re-inflation period: background regional-extension cracks + both ring-fault
damage zones + ONE inflating sill located where the "second sill" (Kidiwela S2 location,
sws_forward_model.py's SPHERES: x0=7.53, y0=6.60, d=1.25 km) sat in the now-retired two-sills
model -- NOT the original sill's location, and with NO dike and NO second/original sill present
at all (unlike Scenario C, which keeps the original sill present but deflated). Same size (R)
and inflation pressure as the original sill.

This module follows the same monkeypatch-override pattern as run_synthetic_hudson_splitting_
two_sills.py: it wraps (not edits in place) synthetic_hudson_crack_scenarios.
total_stress_and_crack_sets so scenarios 'A'/'B'/'C' are completely unchanged (still the
single-sill physics), and only 'D' is new.

Also adds a localized ISOTROPIC crack-density boost around AXAS2, active in Scenario D only,
per explicit user request: "cracks going all directions around AXAS2 ... to simulate
hydrothermal pathways". Implemented as an increased epsilon0 (crack density) for the SAME
isotropic (Fibonacci-sphere-oriented) background population already used everywhere else in
this model -- not a new stress source, not a new orientation set -- so within
AXAS2_HYDROTHERMAL_RADIUS_KM of AXAS2, more of that all-directions crack population is open,
representing a locally intensified, non-single-fabric fracture network (hydrothermal pathways)
rather than a coherent, stress-aligned crack set.

Exposes:
    SILL2                                     -- the re-inflation sill's location dict
    total_stress_and_crack_sets_reinflation   -- scenario-D-aware wrapper (installs itself onto
                                                  synthetic_hudson_crack_scenarios at import time)
    integrate_synthetic_splitting_reinflation -- scenario-D-aware ray integrator (adds the
                                                  AXAS2 epsilon0 boost); use THIS instead of
                                                  synthetic_hudson_crack_scenarios.
                                                  integrate_synthetic_splitting for scenario D
"""

import math

import numpy as np

import synthetic_hudson_crack_scenarios as shc
from sws_forward_model import SPHERES

_KIDIWELA_S2 = next(s for s in SPHERES if s['label'] == 'S2')
SILL2 = dict(x0=_KIDIWELA_S2['x0'], y0=_KIDIWELA_S2['y0'], d=_KIDIWELA_S2['d'], R=shc.SILL['R'])

SILL2_OVERPRINT_MARGIN_KM = SILL2['R'] * 2.5   # same convention as shc.SILL_OVERPRINT_MARGIN_KM

# ── AXAS2 hydrothermal-pathway crack-density boost (Scenario D only) ────────────
AXAS2_XY = shc.STATION_XY['AXAS2']
AXAS2_HYDROTHERMAL_RADIUS_KM = 1.0    # tunable default, ~4x the sill radius
AXAS2_HYDROTHERMAL_EPS_BOOST = 0.04   # additive epsilon0 boost -- same order of magnitude as
                                      # FAULT_CRACK_EPS/BACKGROUND_EPS_SURFACE, doubling the
                                      # surface background density within the radius

_orig_total_stress_and_crack_sets = shc.total_stress_and_crack_sets


def total_stress_and_crack_sets_reinflation(scenario, x_km, y_km, z_km, dike_interps=None):
    if scenario != 'D':
        return _orig_total_stress_and_crack_sets(scenario, x_km, y_km, z_km, dike_interps)

    sigma = shc.background_stress_tensor()
    sigma = sigma + shc.mogi_stress_tensor_3d(x_km, y_km, z_km, SILL2, shc.SILL_DP_PA,
                                              shc.MU_BG, shc.NU_FIX)

    # Same "sill overprints fault" rule as Scenario B applies to the original sill -- suppress
    # the fault's fixed crack set within SILL2's own overprint radius, so SILL2 shows its own
    # pure radial pattern there rather than a blend diluted by the fault's fixed set.
    sill2_overprint = (math.hypot(x_km - SILL2['x0'], y_km - SILL2['y0'])
                       <= SILL2['R'] + SILL2_OVERPRINT_MARGIN_KM)
    extra_crack_sets = []
    if not sill2_overprint:
        if (shc.in_eastern_fault_damage_zone(x_km, y_km, z_km) or
                shc.in_western_fault_damage_zone(x_km, y_km, z_km)):
            extra_crack_sets.append((shc.FAULT_CRACK_NHAT, shc.FAULT_CRACK_EPS))

    return sigma, extra_crack_sets


shc.total_stress_and_crack_sets = total_stress_and_crack_sets_reinflation


def _axas2_epsilon0_boost(x_km, y_km, z_km):
    """Background epsilon0 at this depth, PLUS AXAS2_HYDROTHERMAL_EPS_BOOST if (x,y) is within
    AXAS2_HYDROTHERMAL_RADIUS_KM of AXAS2 (Scenario D only -- caller only invokes this for 'D')."""
    eps0 = shc.background_epsilon0_at_depth(z_km)
    if math.hypot(x_km - AXAS2_XY[0], y_km - AXAS2_XY[1]) <= AXAS2_HYDROTHERMAL_RADIUS_KM:
        eps0 += AXAS2_HYDROTHERMAL_EPS_BOOST
    return eps0


def integrate_synthetic_splitting_reinflation(scenario, ray_xyz, mu=shc.MU_BG, rho=shc.RHO,
                                              dike_interps=None, n_hats=None, **hudson_kwargs):
    """Same accumulation logic as synthetic_hudson_crack_scenarios.integrate_synthetic_splitting
    (path-weighted circular-mean phi, summed dt), duplicated here (not merged into the shared
    function) so only Scenario D gets the AXAS2 hydrothermal epsilon0 boost -- Scenarios A/B/C
    still use the shared function unchanged."""
    mid = 0.5 * (ray_xyz[:-1] + ray_xyz[1:])
    ds_km = np.linalg.norm(np.diff(ray_xyz, axis=0), axis=1)
    ds_m = ds_km * 1e3
    p_hat = ray_xyz[-1] - ray_xyz[0]
    p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-30)

    N = len(mid)
    phis = np.zeros(N); dVs = np.zeros(N); VSs = np.zeros(N)
    for i in range(N):
        x_km, y_km, z_km = mid[i]
        sigma, extra = shc.total_stress_and_crack_sets(scenario, x_km, y_km, z_km, dike_interps)
        point_kwargs = dict(hudson_kwargs)
        if scenario == 'D':
            point_kwargs.setdefault('epsilon0', _axas2_epsilon0_boost(x_km, y_km, z_km))
        else:
            point_kwargs.setdefault('epsilon0', shc.background_epsilon0_at_depth(z_km))
        C_eff, _ = shc.effective_stiffness_with_extra_cracks(sigma, extra, mu, n_hats=n_hats,
                                                              **point_kwargs)
        phis[i], dVs[i], VSs[i], _ = shc.christoffel_splitting(C_eff, p_hat, rho)

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
