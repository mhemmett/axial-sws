#!/usr/bin/env python3
"""
synthetic_hudson_structure_map_two_sills.py

Variant of synthetic_hudson_structure_map.py with a SECOND sill added, per explicit user
request: same size (R) and inflation pressure as the original sill, located at the Kidiwela
S2 source location (x0=7.53, y0=6.60, d=1.25 km -- sws_forward_model.py's SPHERES list;
superseding an earlier AXCC1-relative-offset placement per user correction). Active only in
the pre-eruption (Scenario B) panel, alongside the original sill -- both sills present and
inflated together there. The syn-eruption (Scenario C) panel is unchanged from the base
structure map (dike + the ORIGINAL sill deflated; SILL2 is NOT added to Scenario C, since the
user's request didn't specify syn-eruption behavior for it).

Everything else (bathymetry, stations, fault lines, dike trace, tick-mark field method) is
identical to synthetic_hudson_structure_map.py -- imported directly, not duplicated.

Produces: synthetic_hudson_structure_map_two_sills.pdf (1 page, 2 panels)

Run with:
    python3 synthetic_hudson_structure_map_two_sills.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable

import synthetic_hudson_structure_map as sm
from synthetic_hudson_crack_scenarios import (
    SILL, mogi_stress_tensor_3d, MU_BG, NU_FIX,
)
from sws_forward_model import SPHERES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'synthetic_hudson_structure_map_two_sills.pdf')

# Second sill: same size/pressure as the original, at the Kidiwela S2 source location.
_KIDIWELA_S2 = next(s for s in SPHERES if s['label'] == 'S2')
SILL2 = dict(
    x0=_KIDIWELA_S2['x0'],
    y0=_KIDIWELA_S2['y0'],
    d=_KIDIWELA_S2['d'],
    R=SILL['R'],
)
print(f'SILL2 (Kidiwela S2 location): x0={SILL2["x0"]:.2f}, y0={SILL2["y0"]:.2f}, '
      f'd={SILL2["d"]:.2f}, R={SILL2["R"]:.2f} km')

_orig_total_stress_and_crack_sets = sm.total_stress_and_crack_sets


def total_stress_and_crack_sets_two_sills(scenario, x_km, y_km, z_km, dike_interps=None):
    sigma, extra = _orig_total_stress_and_crack_sets(scenario, x_km, y_km, z_km, dike_interps)
    from synthetic_hudson_crack_scenarios import SILL_DP_PA
    if scenario == 'B':
        sigma = sigma + mogi_stress_tensor_3d(x_km, y_km, z_km, SILL2, SILL_DP_PA, MU_BG, NU_FIX)
    elif scenario == 'C':
        # SILL2's crack population stays present (not removed) but now DEFLATED, per explicit
        # user request -- same treatment sill1 already gets in Scenario C.
        sigma = sigma + mogi_stress_tensor_3d(x_km, y_km, z_km, SILL2, -SILL_DP_PA, MU_BG, NU_FIX)
    return sigma, extra


def compute_sill_dike_field_two_sills(scenario, xn, yn, dike_interps=None):
    """Same as sm.compute_sill_dike_field(), but ALSO gates/overprints around SILL2 (so it
    gets its own crack-tick population, not just a stress contribution) -- per explicit user
    request. This is the layer drawn LAST (highest zorder) in draw_panel, so it (and SILL2)
    still overprints the fault-only layer wherever they overlap. The fault's fixed crack set
    is deliberately dropped here (same fix as sm.compute_sill_dike_field), so the overlap
    region shows the sill(s)'/dike's own pure pattern rather than one diluted by the fault."""
    import numpy as np
    PHI = np.full((len(xn), len(yn)), np.nan)
    DV = np.zeros((len(xn), len(yn)))
    GATE = np.zeros((len(xn), len(yn)), dtype=bool)
    for ix, x in enumerate(xn):
        for iy, y in enumerate(yn):
            sill_gate = (scenario == 'B' and
                        sm.math.hypot(x - SILL['x0'], y - SILL['y0'])
                        <= SILL['R'] + sm.SILL_GATE_MARGIN_KM)
            sill2_gate = (scenario in ('B', 'C') and
                         sm.math.hypot(x - SILL2['x0'], y - SILL2['y0'])
                         <= SILL2['R'] + sm.SILL_GATE_MARGIN_KM)
            dike_gate = (scenario == 'C' and sm._dist_to_dike_km(x, y) <= sm.DIKE_GATE_HALFWIDTH_KM)
            gate = sill_gate or sill2_gate or dike_gate
            GATE[ix, iy] = gate
            if not gate:
                continue
            sigma, _extra_fault_ignored = sm.total_stress_and_crack_sets(
                scenario, x, y, sm.EVAL_DEPTH_KM, dike_interps=dike_interps)
            C_eff, _ = sm.effective_stiffness_with_extra_cracks(sigma, [], n_hats=sm.n_hats)
            phi, dV, _, _ = sm.christoffel_splitting(C_eff, sm.p_vert, sm.RHO)
            PHI[ix, iy] = phi
            DV[ix, iy] = dV
    return PHI, DV, GATE


sm.compute_sill_dike_field = compute_sill_dike_field_two_sills


# Monkeypatch: sm.compute_crack_field looks up total_stress_and_crack_sets from sm's own
# module namespace at call time, so reassigning this name here takes effect.
sm.total_stress_and_crack_sets = total_stress_and_crack_sets_two_sills


def _sill2_circle(ax, deflated=False):
    circ = plt.Circle((SILL2['x0'], SILL2['y0']), SILL2['R'], fill=False, color='deepskyblue',
                      lw=2.2, zorder=10, linestyle='--' if deflated else '-')
    ax.add_patch(circ)


def main():
    print('Loading bathymetry...')
    gray, ext = sm.load_bathymetry()

    xs, ys, names = [], [], []
    for s in sm.ALL_STATIONS:
        if s in sm.STATION_XY:
            xs.append(sm.STATION_XY[s][0]); ys.append(sm.STATION_XY[s][1])
        else:
            import pandas as pd
            sta_file = pd.read_csv(os.path.join(HERE, '..', 'data', 'stations_axial.llz'),
                                   sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                                   engine='python').set_index('s')
            x, y = sm.ll2xy(sta_file.loc[s, 'lat'], sta_file.loc[s, 'lon'])
            xs.append(float(x)); ys.append(float(y))
        names.append(s)
    import pandas as pd
    sta_df = pd.DataFrame({'x': xs, 'y': ys}, index=names)

    print('Building dike stress interpolators (Scenario C)...')
    dike_xn = np.arange(4.0, 12.01, 0.1)
    dike_yn = np.arange(0.0, 15.01, 0.1)
    dike_interps = sm.build_dike_stress_interpolators(dike_xn, dike_yn)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    print('Computing pre-eruption (Scenario B, TWO sills) crack field...')
    sm.draw_panel(axes[0], gray, ext, sta_df, 'B',
                 'Pre-eruption (Scenario B): background + fault damage zones + 2 sills',
                 dike_interps=None)
    _sill2_circle(axes[0], deflated=False)

    print('Computing syn-eruption (Scenario C, dike) crack field...')
    sm.draw_panel(axes[1], gray, ext, sta_df, 'C',
                 'Syn-eruption (Scenario C): background + fault damage zones + dike '
                 '+ both sills deflated',
                 dike_interps=dike_interps)
    _sill2_circle(axes[1], deflated=True)

    legend_handles = [
        mlines.Line2D([], [], color='#00CFFF', lw=2.0, label='Ring faults (E + W)'),
        mlines.Line2D([], [], color='orange', lw=2.0, label='Sill 1, inflated (pre-eruption)'),
        mlines.Line2D([], [], color='deepskyblue', lw=2.0,
                      label='Sill 2 (Kidiwela S2 location), inflated (pre-eruption)'),
        mlines.Line2D([], [], color='orange', lw=2.0, ls='--', label='Sill 1, deflated (syn-eruption)'),
        mlines.Line2D([], [], color='deepskyblue', lw=2.0, ls='--',
                      label='Sill 2, deflated (syn-eruption)'),
        mlines.Line2D([], [], color='lime', lw=2.0, label='Dike trace (syn-eruption only)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8,
                      label='Station (new data, all 6)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6, fontsize=8,
              framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    sm_map = ScalarMappable(cmap=sm.AZ_CMAP, norm=sm.AZ_NORM)
    sm_map.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm_map, cax=cax)
    cb.set_label('Synthetic fast direction $\\phi$ (deg, axial 0-180)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(f'Synthetic Hudson crack-model structure map, TWO SILLS variant '
                 f'(evaluated at {sm.EVAL_DEPTH_KM} km depth)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.07, 0.91, 0.95])
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
