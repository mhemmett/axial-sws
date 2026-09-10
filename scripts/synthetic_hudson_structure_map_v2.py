#!/usr/bin/env python3
"""
synthetic_hudson_structure_map_v2.py

NEW v2 geometry variant (supersedes the earlier "_v2" iteration chain, which has been fully
baked into synthetic_hudson_crack_scenarios.py's base SILL/DIKE_WAYPOINTS -- this script now
starts fresh from that consolidated base), per explicit user request:

    - Sill 1 moves to 0.25 km WEST of AXEC1 (x0=9.07, y0=5.52 -- same latitude as AXEC1).
    - ALL other dike waypoints (P1, P2, P3, P4 -- the longest segment and everything north of
      it) stay at EXACTLY their current base positions, unchanged.
    - Only the south-going segment (P0->P1) is redrawn: a straight connector from the NEW
      sill 1 location to P1 (the existing, unmoved bottom of the longest segment) -- not a
      rigid shift/same-length-preserving move like the earlier v2 chain, just a fresh
      straight line between those two fixed points.

Produces: synthetic_hudson_structure_map_v2.pdf (1 page, 2 panels)

Run with:
    python3 synthetic_hudson_structure_map_v2.py
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import synthetic_hudson_crack_scenarios as shc

print(f'Base sill1: x0={shc.SILL["x0"]:.3f}, y0={shc.SILL["y0"]:.3f}')
print(f'Base dike waypoints: {shc.DIKE_WAYPOINTS}')

_, _P1, _P2, _P3, _P4 = shc.DIKE_WAYPOINTS

_NEW_P0 = (shc.STATION_XY['AXEC1'][0] - 0.25, shc.STATION_XY['AXEC1'][1])
_NEW_WAYPOINTS = [_NEW_P0, _P1, _P2, _P3, _P4]

shc.SILL['x0'] = _NEW_P0[0]
shc.SILL['y0'] = _NEW_P0[1]
shc.DIKE_WAYPOINTS = _NEW_WAYPOINTS

print(f'\nNew sill1 (0.25 km west of AXEC1): x0={shc.SILL["x0"]:.3f}, y0={shc.SILL["y0"]:.3f}')
print(f'New dike waypoints: {[(round(x,4), round(y,4)) for x,y in _NEW_WAYPOINTS]}')

import synthetic_hudson_structure_map as sm

# sm's own separately-imported DIKE_WAYPOINTS/SILL names must ALSO be patched (a plain
# `from module import NAME` does not stay linked to later reassignment in the source module).
sm.DIKE_WAYPOINTS = _NEW_WAYPOINTS
sm.OUT_PDF = sm.OUT_PDF.replace('synthetic_hudson_structure_map.pdf',
                                'synthetic_hudson_structure_map_v2.pdf')

if __name__ == '__main__':
    print('Loading bathymetry...')
    gray, ext = sm.load_bathymetry()

    import os
    import pandas as pd
    xs, ys, names = [], [], []
    for s in sm.ALL_STATIONS:
        if s in sm.STATION_XY:
            xs.append(sm.STATION_XY[s][0]); ys.append(sm.STATION_XY[s][1])
        else:
            sta_file = pd.read_csv(os.path.join(sm.HERE, '..', 'data', 'stations_axial.llz'),
                                   sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                                   engine='python').set_index('s')
            x, y = sm.ll2xy(sta_file.loc[s, 'lat'], sta_file.loc[s, 'lon'])
            xs.append(float(x)); ys.append(float(y))
        names.append(s)
    sta_df = pd.DataFrame({'x': xs, 'y': ys}, index=names)

    print('Building dike stress interpolators (Scenario C, v2 geometry)...')
    dike_xn = np.arange(4.0, 12.01, 0.1)
    dike_yn = np.arange(0.0, 15.01, 0.1)
    dike_interps = sm.build_dike_stress_interpolators(dike_xn, dike_yn)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    print('Computing pre-eruption (Scenario B, sill, v2 geometry) crack field...')
    sm.draw_panel(axes[0], gray, ext, sta_df, 'B',
                 'Pre-eruption v2 (Scenario B): background + fault damage zones + sill',
                 dike_interps=None)

    print('Computing syn-eruption (Scenario C, dike, v2 geometry) crack field...')
    sm.draw_panel(axes[1], gray, ext, sta_df, 'C',
                 'Syn-eruption v2 (Scenario C): background + fault damage zones + dike',
                 dike_interps=dike_interps)

    import matplotlib.lines as mlines
    from matplotlib.cm import ScalarMappable
    legend_handles = [
        mlines.Line2D([], [], color='#00CFFF', lw=2.0, label='Ring faults (E + W)'),
        mlines.Line2D([], [], color='orange', lw=2.0, label='Sill 1, inflated (pre-eruption)'),
        mlines.Line2D([], [], color='orange', lw=2.0, ls='--', label='Sill 1, deflated (syn-eruption)'),
        mlines.Line2D([], [], color='lime', lw=2.0, label='Dike trace (syn-eruption only)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8,
                      label='Station (new data, all 6)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=8,
              framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    sm_map = ScalarMappable(cmap=sm.AZ_CMAP, norm=sm.AZ_NORM)
    sm_map.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm_map, cax=cax)
    cb.set_label('Synthetic fast direction $\\phi$ (deg, axial 0-180)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle('Synthetic Hudson crack-model structure map, v2 geometry (sill1 moved 0.25km '
                 'west of AXEC1; only the south-going dike segment redrawn to connect to it, '
                 'everything north unchanged) '
                 f'(evaluated at {sm.EVAL_DEPTH_KM} km depth)', fontsize=10, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 0.91, 0.95])
    fig.savefig(sm.OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {sm.OUT_PDF}')
