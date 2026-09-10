#!/usr/bin/env python3
"""
synthetic_hudson_structure_map.py

Map of the synthetic Hudson-crack-model geometry, pre- and syn-eruption, over grayscale
bathymetry with yellow-triangle labeled stations (same conventions as this session's
spatial_map_fast_direction_5stations_newdata_7period.py). Two panels:

  Pre-eruption (Scenario B): background regional-extension cracks + both ring-fault damage
    zones + the sill and its (emergent, not hardcoded) radial crack pattern.
  Syn-eruption (Scenario C): background regional-extension cracks + both ring-fault damage
    zones + the eastern-wall dike trace and its (emergent) crack pattern.

The "cracks" for ALL FOUR populations (background, both fault-damage-zones, sill, dike) are
drawn as ONE tick-mark field per panel: at each grid point, the actual scenario physics
(synthetic_hudson_crack_scenarios.total_stress_and_crack_sets + hudson_crack_model's
effective_stiffness/christoffel_splitting -- the exact same functions the real driver script
uses, not a separate illustrative approximation) is evaluated at 1.5 km depth (matching the
sill/dike depth), so the map is a direct, physically-consistent visualization of the model,
not an artist's schematic. Ring-fault traces and the dike waypoint trace are ALSO drawn
explicitly as reference lines on top of the tick-mark field.

Produces: synthetic_hudson_structure_map.pdf (1 page, 2 panels)

Run with:
    python3 synthetic_hudson_structure_map.py
"""

import math
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import tifffile
from PIL import Image as PILImage
import os

from hudson_crack_model import christoffel_splitting, fibonacci_sphere, RHO, effective_stiffness
from sws_forward_model import MU_BG
from synthetic_hudson_crack_scenarios import (
    STATION_XY, EASTERN_FAULT_CENTER, WESTERN_FAULT_CENTER, FAULT_STRIKE_DEG,
    FAULT_HALF_LEN_KM, SILL, DIKE_WAYPOINTS,
    total_stress_and_crack_sets, effective_stiffness_with_extra_cracks,
    build_dike_stress_interpolators, background_stress_tensor,
    in_eastern_fault_damage_zone, in_western_fault_damage_zone,
    background_epsilon0_at_depth,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'synthetic_hudson_structure_map.pdf')

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 15.0   # extended north to cover the dike's full trace (y up to 13.93)
EVAL_DEPTH_KM = 1.5          # matches sill/dike depth

GRID_STEP_KM = 0.4
N_CRACKS = 100
n_hats = fibonacci_sphere(N_CRACKS)
p_vert = np.array([0., 0., 1.])

ALL_STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
NEWDATA_STATIONS = {'AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3'}
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


def load_bathymetry():
    bathy = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
             'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
    PILImage.MAX_IMAGE_PIXELS = None
    _p = PILImage.open(bathy)
    _tag = _p.tag_v2
    _olon, _olat = _tag[33922][3], _tag[33922][4]
    _pl, _pb = _tag[33550][0], _tag[33550][1]
    _nc, _nr = _p.size
    _p.close()
    _c0 = max(0, int(((INI_LON + X_START / KM_PER_DEG_LON) - _olon) / _pl) - 2)
    _c1 = min(_nc, int(((INI_LON + X_END / KM_PER_DEG_LON) - _olon) / _pl) + 2)
    _r0 = max(0, int((_olat - (INI_LAT + Y_END / KM_PER_DEG_LAT)) / _pb) - 2)
    _r1 = min(_nr, int((_olat - (INI_LAT + Y_START / KM_PER_DEG_LAT)) / _pb) + 2)
    _rgb = tifffile.imread(bathy)[_r0:_r1, _c0:_c1]
    _ds = max(1, max(_rgb.shape[:2]) // 1024)
    _rgb = _rgb[::_ds, ::_ds]
    _gray = np.dot(_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
    _ext = [(_olon + _c0 * _pl - INI_LON) * KM_PER_DEG_LON,
            (_olon + _c1 * _pl - INI_LON) * KM_PER_DEG_LON,
            (_olat - _r1 * _pb - INI_LAT) * KM_PER_DEG_LAT,
            (_olat - _r0 * _pb - INI_LAT) * KM_PER_DEG_LAT]
    return _gray, _ext


def _bathy(ax, gray, ext):
    ax.imshow(gray, origin='upper', extent=ext, aspect='auto', cmap='gray', zorder=0)


def _stations(ax, sta_df):
    for sta, row in sta_df.iterrows():
        color = '#FFD700' if sta in NEWDATA_STATIONS else '#FFFFFF'
        ax.plot(row['x'], row['y'], '^', ms=7, mfc=color, mec='k', mew=0.8, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=6.5,
                fontweight='bold', zorder=13)


def _fault_line(ax, center, label):
    strike_rad = math.radians(FAULT_STRIKE_DEG)
    s_hat = np.array([math.sin(strike_rad), math.cos(strike_rad)])
    p1 = np.array(center) - FAULT_HALF_LEN_KM * s_hat
    p2 = np.array(center) + FAULT_HALF_LEN_KM * s_hat
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='#00CFFF', lw=2.0, alpha=0.9,
            zorder=10, solid_capstyle='round')


def _dike_line(ax):
    xs = [p[0] for p in DIKE_WAYPOINTS]
    ys = [p[1] for p in DIKE_WAYPOINTS]
    ax.plot(xs, ys, '-', color='lime', lw=2.2, alpha=0.95, zorder=10,
            solid_capstyle='round', label='Dike trace')


def _sill_circle(ax, deflated=False):
    circ = plt.Circle((SILL['x0'], SILL['y0']), SILL['R'], fill=False, color='orange',
                      lw=2.2, zorder=10, linestyle='--' if deflated else '-')
    ax.add_patch(circ)


SILL_GATE_MARGIN_KM = SILL['R'] * 2.5   # overprint radius around the sill (beyond its own R)
DIKE_GATE_HALFWIDTH_KM = 0.5            # overprint corridor half-width around the dike trace


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


def compute_background_field():
    """Background (regional-extension) crack field -- background stress ONLY, no fault/
    sill/dike terms -- evaluated EVERYWHERE (this population exists across the whole map,
    per explicit user request), drawn as the base layer that structures overprint."""
    xn = np.arange(X_START, X_END + 1e-9, GRID_STEP_KM)
    yn = np.arange(Y_START, Y_END + 1e-9, GRID_STEP_KM)
    sigma_bg = background_stress_tensor()
    eps0_at_depth = background_epsilon0_at_depth(EVAL_DEPTH_KM)
    C_bg, _ = effective_stiffness(sigma_bg, MU_BG, epsilon0=eps0_at_depth, sigma_c=40e6, n_hats=n_hats)
    phi_bg, dV_bg, _, _ = christoffel_splitting(C_bg, p_vert, RHO)
    PHI = np.full((len(xn), len(yn)), phi_bg)
    DV = np.full((len(xn), len(yn)), dV_bg)
    return xn, yn, PHI, DV


def compute_fault_only_field(xn, yn):
    """Fault-damage-zone-only physics (background + fault crack sets, NO sill/dike --
    scenario 'A' never adds either) -- drawn BEFORE the sill/dike layer, so that layer can
    overprint it wherever the two overlap (per explicit user request: the sill should
    overprint the eastern fault region, not the other way around)."""
    PHI = np.full((len(xn), len(yn)), np.nan)
    DV = np.zeros((len(xn), len(yn)))
    GATE = np.zeros((len(xn), len(yn)), dtype=bool)
    for ix, x in enumerate(xn):
        for iy, y in enumerate(yn):
            gate = (in_eastern_fault_damage_zone(x, y, EVAL_DEPTH_KM) or
                   in_western_fault_damage_zone(x, y, EVAL_DEPTH_KM))
            GATE[ix, iy] = gate
            if not gate:
                continue
            sigma, extra = total_stress_and_crack_sets('A', x, y, EVAL_DEPTH_KM)
            C_eff, _ = effective_stiffness_with_extra_cracks(
                sigma, extra, n_hats=n_hats, epsilon0=background_epsilon0_at_depth(EVAL_DEPTH_KM))
            phi, dV, _, _ = christoffel_splitting(C_eff, p_vert, RHO)
            PHI[ix, iy] = phi
            DV[ix, iy] = dV
    return PHI, DV, GATE


def compute_sill_dike_field(scenario, xn, yn, dike_interps=None):
    """Sill (Scenario B) / dike (Scenario C) physics -- background + sill/dike stress ONLY
    (the fault's fixed crack-set contribution is deliberately DROPPED here, even inside a
    fault damage zone, per explicit user request: the sill/dike layer should show its own
    pure radially/dike-organized crack pattern in the overlap region, not a blend diluted by
    the fault's fixed set -- otherwise the fault's crack set, being a strong deterministic
    contribution, drowns out the sill's comparatively weak stress-driven signal). Only drawn
    where the sill/dike overprint gate applies. Drawn AFTER (higher zorder than)
    compute_fault_only_field's layer, so it overprints the fault-only ticks in any overlap
    region, both visually AND physically (this layer's own physics, not a shared blend)."""
    PHI = np.full((len(xn), len(yn)), np.nan)
    DV = np.zeros((len(xn), len(yn)))
    GATE = np.zeros((len(xn), len(yn)), dtype=bool)
    for ix, x in enumerate(xn):
        for iy, y in enumerate(yn):
            sill_gate = (scenario == 'B' and
                        math.hypot(x - SILL['x0'], y - SILL['y0']) <= SILL['R'] + SILL_GATE_MARGIN_KM)
            dike_gate = (scenario == 'C' and _dist_to_dike_km(x, y) <= DIKE_GATE_HALFWIDTH_KM)
            gate = sill_gate or dike_gate
            GATE[ix, iy] = gate
            if not gate:
                continue
            sigma, _extra_fault_ignored = total_stress_and_crack_sets(
                scenario, x, y, EVAL_DEPTH_KM, dike_interps=dike_interps)
            C_eff, _ = effective_stiffness_with_extra_cracks(
                sigma, [], n_hats=n_hats, epsilon0=background_epsilon0_at_depth(EVAL_DEPTH_KM))
            phi, dV, _, _ = christoffel_splitting(C_eff, p_vert, RHO)
            PHI[ix, iy] = phi
            DV[ix, iy] = dV
    return PHI, DV, GATE


AZ_CMAP = plt.colormaps['hsv']
AZ_NORM = Normalize(vmin=0.0, vmax=180.0)


def _draw_ticks(ax, Xg, Yg, PHI, mask, alpha, lw, scale, zorder):
    ro = np.radians(PHI[mask])
    uo, vo = np.sin(ro), np.cos(ro)
    tick_kw = dict(scale=scale, width=lw, headlength=0, headaxislength=0, headwidth=0,
                  pivot='middle', cmap=AZ_CMAP, norm=AZ_NORM, alpha=alpha, zorder=zorder)
    ax.quiver(Xg[mask], Yg[mask], uo, vo, PHI[mask], **tick_kw)
    ax.quiver(Xg[mask], Yg[mask], -uo, -vo, PHI[mask], **tick_kw)


def draw_panel(ax, gray, ext, sta_df, scenario, title, dike_interps=None):
    _bathy(ax, gray, ext)

    # ── Background field: drawn everywhere (base layer) ─────────────────────
    xn, yn, PHI_bg, DV_bg = compute_background_field()
    Xg, Yg = np.meshgrid(xn, yn, indexing='ij')
    bg_mask = np.ones_like(PHI_bg, dtype=bool)
    _draw_ticks(ax, Xg, Yg, PHI_bg, bg_mask, alpha=0.55, lw=0.0035, scale=22, zorder=5)

    # ── Fault-damage-zone field: overprints the background near either fault ────
    PHI_ft, DV_ft, GATE_ft = compute_fault_only_field(xn, yn)
    ft_mask = GATE_ft & np.isfinite(PHI_ft)
    _draw_ticks(ax, Xg, Yg, PHI_ft, ft_mask, alpha=0.95, lw=0.006, scale=22, zorder=7)

    # ── Sill/dike field: drawn LAST (highest zorder), so it overprints the fault-only
    # layer wherever the two overlap (per explicit user request: sill overprints fault) ──
    PHI_sd, DV_sd, GATE_sd = compute_sill_dike_field(scenario, xn, yn, dike_interps=dike_interps)
    sd_mask = GATE_sd & np.isfinite(PHI_sd)
    _draw_ticks(ax, Xg, Yg, PHI_sd, sd_mask, alpha=0.95, lw=0.006, scale=22, zorder=9)

    _fault_line(ax, EASTERN_FAULT_CENTER, 'Eastern fault')
    _fault_line(ax, WESTERN_FAULT_CENTER, 'Western fault')

    if scenario == 'B':
        _sill_circle(ax, deflated=False)
    elif scenario == 'C':
        _dike_line(ax)
        _sill_circle(ax, deflated=True)   # still present, now deflated (see scenario physics)

    _stations(ax, sta_df)

    ax.set_xlim(X_START, X_END)
    ax.set_ylim(Y_START, Y_END)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('East [km]', fontsize=8)
    ax.set_ylabel('North [km]', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=10, fontweight='bold')


def main():
    print('Loading bathymetry...')
    gray, ext = load_bathymetry()

    # STATION_XY only has the 5 new-data stations' LOCAL (x,y) already, not lon/lat -- use
    # those directly; AXAS2 isn't in STATION_XY (excluded throughout this session), so read
    # its lon/lat from the station file and convert, matching every other script's pattern.
    xs, ys = [], []
    names = []
    for s in ALL_STATIONS:
        if s in STATION_XY:
            xs.append(STATION_XY[s][0]); ys.append(STATION_XY[s][1])
        else:
            sta_file = pd.read_csv(os.path.join(HERE, '..', 'data', 'stations_axial.llz'),
                                   sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                                   engine='python').set_index('s')
            x, y = ll2xy(sta_file.loc[s, 'lat'], sta_file.loc[s, 'lon'])
            xs.append(float(x)); ys.append(float(y))
        names.append(s)
    sta_df = pd.DataFrame({'x': xs, 'y': ys}, index=names)

    print('Building dike stress interpolators (Scenario C)...')
    dike_xn = np.arange(4.0, 12.01, 0.1)
    dike_yn = np.arange(0.0, 15.01, 0.1)
    dike_interps = build_dike_stress_interpolators(dike_xn, dike_yn)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    print('Computing pre-eruption (Scenario B, sill) crack field...')
    draw_panel(axes[0], gray, ext, sta_df, 'B',
              'Pre-eruption (Scenario B): background + fault damage zones + sill',
              dike_interps=None)

    print('Computing syn-eruption (Scenario C, dike) crack field...')
    draw_panel(axes[1], gray, ext, sta_df, 'C',
              'Syn-eruption (Scenario C): background + fault damage zones + dike',
              dike_interps=dike_interps)

    legend_handles = [
        mlines.Line2D([], [], color='#00CFFF', lw=2.0, label='Ring faults (E + W)'),
        mlines.Line2D([], [], color='orange', lw=2.0, label='Sill, inflated (pre-eruption)'),
        mlines.Line2D([], [], color='orange', lw=2.0, ls='--', label='Sill, deflated (syn-eruption)'),
        mlines.Line2D([], [], color='lime', lw=2.0, label='Dike trace (syn-eruption only)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8,
                      label='Station (new data, all 6)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=8,
              framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    sm = ScalarMappable(cmap=AZ_CMAP, norm=AZ_NORM)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Synthetic fast direction $\\phi$ (deg, axial 0-180)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(f'Synthetic Hudson crack-model structure map (evaluated at {EVAL_DEPTH_KM} km '
                 'depth, tick marks colored by fast direction)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 0.91, 0.95])
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
