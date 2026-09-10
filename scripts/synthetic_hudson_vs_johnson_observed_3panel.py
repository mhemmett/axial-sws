#!/usr/bin/env python3
"""
synthetic_hudson_vs_johnson_observed_3panel.py

Modeled-vs-observed stress-orientation comparison, per explicit user request: overlay the
synthetic Hudson crack-model's three canonical stress scenarios (B: pre-eruption sill, C:
syn-eruption dike, D: post-eruption re-inflation -- see synthetic_hudson_structure_map.py and
synthetic_hudson_reinflation_scenario.py for the underlying physics, reused here directly, not
duplicated) against the OBSERVED fast direction from the Johnson et al. (2011) 7-period
tomography re-run (sws_tomography_johnson2011_newdata_grade3_7period.py), for the three
matching real time periods: panel 1 = Pre-eruption, panel 2 = Syn-eruption, panel 3 = Jul
2025-present (period 7 of 7, the most recent -- i.e. the real-world analog of the Scenario-D
re-inflation state).

Per explicit user color/layer convention (all three panels):
  - STRUCTURES (ring faults, sill outline(s), dike trace) -- GRAYSCALE reference lines/circles
    over the grayscale bathymetry (not the colorful cyan/orange/lime of the original structure
    map script).
  - MODELED stress (the crack model's own predicted phi, evaluated at 1.5 km depth on the same
    background+fault-damage-zone+sill/dike/reinflation grid as synthetic_hudson_structure_map.py)
    -- BLUE double-headed tick marks (single color, not the hsv-by-phi coloring used elsewhere).
  - OBSERVED stress (Johnson-2011 per-block phi_bar, same DISPLAY_COUNT_MIN=20 +
    SIGNIFICANCE_LEVEL=-5.5 gates as sws_tomography_johnson2011_newdata_grade3_7period.py's own
    phi-quiver figure) -- RED double-headed tick marks, drawn ON TOP of the blue modeled field.

Produces:
    synthetic_hudson_vs_johnson_observed_3panel.pdf
(1 page, 3 panels side by side) -- a NEW file, does not touch any existing output.

Run with:
    python3 synthetic_hudson_vs_johnson_observed_3panel.py
"""

import math
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import synthetic_hudson_structure_map as SM
import synthetic_hudson_crack_scenarios as shc
import synthetic_hudson_reinflation_scenario as shr   # noqa: F401  (import installs the 'D' monkeypatch)
from hudson_crack_model import christoffel_splitting, RHO, effective_stiffness

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
import sws_tomography_johnson2011_newdata_grade3_7period as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'synthetic_hudson_vs_johnson_observed_3panel.pdf')

MODEL_COLOR = 'tab:blue'
OBS_COLOR = 'tab:red'
STRUCT_COLOR = '0.15'   # near-black, over the already-grayscale bathy


# ── Modeled crack-field computation for the reinflation (Scenario D) panel ─────────────────
# Mirrors synthetic_hudson_structure_map.compute_sill_dike_field's B/C gating pattern, but for
# SILL2 (shr.SILL2) with the AXAS2 epsilon0 boost (shr._axas2_epsilon0_boost) that Scenario D
# alone uses (see synthetic_hudson_reinflation_scenario.py's own docstring).

def compute_sill2_field(xn, yn):
    PHI = np.full((len(xn), len(yn)), np.nan)
    GATE = np.zeros((len(xn), len(yn)), dtype=bool)
    for ix, x in enumerate(xn):
        for iy, y in enumerate(yn):
            gate = (math.hypot(x - shr.SILL2['x0'], y - shr.SILL2['y0'])
                   <= shr.SILL2['R'] + shr.SILL2_OVERPRINT_MARGIN_KM)
            GATE[ix, iy] = gate
            if not gate:
                continue
            sigma, _extra_ignored = shc.total_stress_and_crack_sets('D', x, y, SM.EVAL_DEPTH_KM)
            eps0 = shr._axas2_epsilon0_boost(x, y, SM.EVAL_DEPTH_KM)
            C_eff, _ = shc.effective_stiffness_with_extra_cracks(
                sigma, [], n_hats=SM.n_hats, epsilon0=eps0)
            phi, _dV, _, _ = christoffel_splitting(C_eff, SM.p_vert, RHO)
            PHI[ix, iy] = phi
    return PHI, GATE


def compute_modeled_phi_grid(scenario, dike_interps=None):
    """Background + fault-damage-zone + sill/dike/reinflation modeled phi, on the SAME grid
    convention as synthetic_hudson_structure_map.py (sill/dike/reinflation layer overprints the
    fault-only layer wherever they overlap)."""
    xn, yn, PHI_bg, _ = SM.compute_background_field()
    PHI = PHI_bg.copy()

    PHI_ft, _dv, GATE_ft = SM.compute_fault_only_field(xn, yn)
    ft_mask = GATE_ft & np.isfinite(PHI_ft)
    PHI[ft_mask] = PHI_ft[ft_mask]

    if scenario == 'D':
        PHI_sd, GATE_sd = compute_sill2_field(xn, yn)
    else:
        PHI_sd, _dv, GATE_sd = SM.compute_sill_dike_field(scenario, xn, yn, dike_interps=dike_interps)
    sd_mask = GATE_sd & np.isfinite(PHI_sd)
    PHI[sd_mask] = PHI_sd[sd_mask]

    Xg, Yg = np.meshgrid(xn, yn, indexing='ij')
    return Xg, Yg, PHI


def _draw_monochrome_ticks(ax, Xg, Yg, PHI, mask, color, alpha, lw, scale, zorder, label=None):
    ro = np.radians(PHI[mask])
    uo, vo = np.sin(ro), np.cos(ro)
    tick_kw = dict(scale=scale, width=lw, headlength=0, headaxislength=0, headwidth=0,
                  pivot='middle', color=color, alpha=alpha, zorder=zorder)
    ax.quiver(Xg[mask], Yg[mask], uo, vo, **tick_kw)
    q = ax.quiver(Xg[mask], Yg[mask], -uo, -vo, **tick_kw)
    if label:
        q.set_label(label)


def _draw_structures_grayscale(ax, scenario):
    """Ring faults + sill/dike/reinflation-sill outlines, all in a single dark-gray tone
    (structures only -- no crack ticks here)."""
    SM._fault_line(ax, shc.EASTERN_FAULT_CENTER, 'Eastern fault')
    SM._fault_line(ax, shc.WESTERN_FAULT_CENTER, 'Western fault')
    for ln in ax.lines[-2:]:
        ln.set_color(STRUCT_COLOR)
        ln.set_linewidth(1.6)
        ln.set_alpha(0.9)

    if scenario == 'B':
        circ = plt.Circle((shc.SILL['x0'], shc.SILL['y0']), shc.SILL['R'], fill=False,
                          color=STRUCT_COLOR, lw=1.8, zorder=10, linestyle='-')
        ax.add_patch(circ)
    elif scenario == 'C':
        xs = [p[0] for p in shc.DIKE_WAYPOINTS]
        ys = [p[1] for p in shc.DIKE_WAYPOINTS]
        ax.plot(xs, ys, '-', color=STRUCT_COLOR, lw=1.8, alpha=0.95, zorder=10,
               solid_capstyle='round')
        circ = plt.Circle((shc.SILL['x0'], shc.SILL['y0']), shc.SILL['R'], fill=False,
                          color=STRUCT_COLOR, lw=1.8, zorder=10, linestyle='--')
        ax.add_patch(circ)
    elif scenario == 'D':
        circ = plt.Circle((shr.SILL2['x0'], shr.SILL2['y0']), shr.SILL2['R'], fill=False,
                          color=STRUCT_COLOR, lw=1.8, zorder=10, linestyle='-')
        ax.add_patch(circ)


def draw_panel(ax, gray, ext, sta_df, scenario, title, johnson_res, dike_interps=None):
    SM._bathy(ax, gray, ext)

    # ── Modeled stress (blue) ────────────────────────────────────────────────
    Xg, Yg, PHI = compute_modeled_phi_grid(scenario, dike_interps=dike_interps)
    mask = np.isfinite(PHI)
    _draw_monochrome_ticks(ax, Xg, Yg, PHI, mask, MODEL_COLOR, alpha=0.55, lw=0.0035,
                          scale=22, zorder=5, label='Modeled stress (Hudson crack model)')

    # ── Structures (grayscale) ───────────────────────────────────────────────
    _draw_structures_grayscale(ax, scenario)

    # ── Observed stress (red), from the Johnson-2011 newdata grade-3 result ─────────────────
    if johnson_res is not None:
        stats = johnson_res['phi_stats'][S.PHI_WEIGHT]
        significant = S._significant_mask(johnson_res)
        kept_blocks = [b for b, sig in zip(stats, significant)
                      if b['kept'] and np.isfinite(b['phi_bar'])
                      and b['count'] >= S.DISPLAY_COUNT_MIN and sig]
        if kept_blocks:
            cx = np.array([b['center'][0] for b in kept_blocks])
            cy = np.array([b['center'][1] for b in kept_blocks])
            sz = np.array([b['side_km'] for b in kept_blocks])
            phi_vals = np.array([b['phi_bar'] for b in kept_blocks])
            r = np.radians(phi_vals)
            length = 0.8 * sz
            u, v = length * np.sin(r), length * np.cos(r)
            tick_kw = dict(scale=1, scale_units='xy', width=0.006, headlength=0,
                          headaxislength=0, headwidth=0, pivot='middle', color=OBS_COLOR,
                          alpha=0.95, zorder=11)
            ax.quiver(cx, cy, u, v, **tick_kw)
            ax.quiver(cx, cy, -u, -v, **tick_kw)

    SM._stations(ax, sta_df)

    ax.set_xlim(SM.X_START, SM.X_END)
    ax.set_ylim(SM.Y_START, SM.Y_END)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('East [km]', fontsize=8)
    ax.set_ylabel('North [km]', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=10, fontweight='bold')


def get_johnson_results():
    """Panels 1, 2, 7 of the Johnson-2011 newdata grade-3 7-period run: Pre-eruption,
    Syn-eruption, Jul 2025-present. Reuses the cached rays -- no re-tracing."""
    T._setup_environment(need_tracer=False)
    print('Loading JOINT six-station grade-3 ray cache (reusing existing cache, no re-tracing)...')
    joint_summ, joint_cells, station_index, sta_xy = S.build_joint_cache()
    print(f'Joint cache: {len(joint_summ):,} rays total across {len(S.STATIONS)} stations')

    t_parsed = pd.to_datetime(joint_summ['t'], utc=True, format='mixed')
    periods = T.build_7_periods(pd.DataFrame({'t': t_parsed}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]

    wanted = {0: 'B', 1: 'C', 6: 'D'}   # Pre-eruption, Syn-eruption, Jul 2025-present
    results = {}
    print('\n=== Running Johnson et al. (2011) chain for panels 1, 2, 7 ===')
    for pi, scenario in wanted.items():
        lbl, t0, t1 = periods[pi]
        print(f'\n-- panel {pi + 1}: {lbl.replace(chr(10), " ")} (Scenario {scenario}) --')
        try:
            res = S.run_chain_grade3_period(joint_summ, joint_cells, t0, t1)
        except ValueError as exc:
            print(f'  [SKIP] {exc}')
            res = None
        results[scenario] = (res, lbl.replace('\n', ' '))
    return results


def main():
    print('Loading bathymetry...')
    gray, ext = SM.load_bathymetry()

    xs, ys, names = [], [], []
    for s in SM.ALL_STATIONS:
        if s in shc.STATION_XY:
            xs.append(shc.STATION_XY[s][0]); ys.append(shc.STATION_XY[s][1])
        else:
            sta_file = pd.read_csv(os.path.join(HERE, '..', 'data', 'stations_axial.llz'),
                                   sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                                   engine='python').set_index('s')
            x, y = SM.ll2xy(sta_file.loc[s, 'lat'], sta_file.loc[s, 'lon'])
            xs.append(float(x)); ys.append(float(y))
        names.append(s)
    sta_df = pd.DataFrame({'x': xs, 'y': ys}, index=names)

    print('Building dike stress interpolators (Scenario C)...')
    dike_xn = np.arange(4.0, 12.01, 0.1)
    dike_yn = np.arange(0.0, 15.01, 0.1)
    dike_interps = shc.build_dike_stress_interpolators(dike_xn, dike_yn)

    johnson = get_johnson_results()

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    print('\nComputing pre-eruption (Scenario B, sill) modeled field...')
    res_b, lbl_b = johnson['B']
    draw_panel(axes[0], gray, ext, sta_df, 'B',
              f'{lbl_b}: Scenario B (sill)', res_b, dike_interps=None)

    print('Computing syn-eruption (Scenario C, dike) modeled field...')
    res_c, lbl_c = johnson['C']
    draw_panel(axes[1], gray, ext, sta_df, 'C',
              f'{lbl_c}: Scenario C (dike)', res_c, dike_interps=dike_interps)

    print('Computing post-eruption re-inflation (Scenario D) modeled field...')
    res_d, lbl_d = johnson['D']
    draw_panel(axes[2], gray, ext, sta_df, 'D',
              f'{lbl_d} (Re-inflation): Scenario D (sill2)', res_d, dike_interps=None)

    legend_handles = [
        mlines.Line2D([], [], color=MODEL_COLOR, lw=2.0, label='Modeled stress (Hudson crack model)'),
        mlines.Line2D([], [], color=OBS_COLOR, lw=2.0, label='Observed stress (Johnson et al. 2011)'),
        mlines.Line2D([], [], color=STRUCT_COLOR, lw=1.8, label='Structures (faults / sill / dike)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8, label='Station'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=8,
              framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Synthetic Hudson crack-model (blue) vs. observed Johnson et al. (2011) '
                 'tomography (red) fast-direction comparison', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
