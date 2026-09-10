#!/usr/bin/env python3
"""
synthetic_hudson_structure_map_two_sills_v2.py

Two-sills structure map combined with the NEW v2 geometry (see synthetic_hudson_structure_
map_v2.py for the full derivation): sill 1 moves to 0.25 km WEST of AXEC1; all other dike
waypoints (P1-P4, the longest segment and everything north of it) stay exactly where they
are in the base geometry; only the south-going segment (P0->P1) is redrawn as a straight
connector from the new sill 1 location to the unmoved P1. SILL2 (Kidiwela S2 location) is
UNCHANGED -- only sill 1 and the dike's south segment move in this variant.

Everything else (SILL2 addition/overprint, bathymetry, stations, fault lines, tick-mark field
method) is identical to synthetic_hudson_structure_map_two_sills.py -- imported directly
(which itself imports synthetic_hudson_structure_map), not duplicated, except for the
geometry overrides applied on top before running.

Produces: synthetic_hudson_structure_map_two_sills_v2.pdf (1 page, 2 panels)

Run with:
    python3 synthetic_hudson_structure_map_two_sills_v2.py
"""

import warnings

warnings.filterwarnings('ignore')

import synthetic_hudson_crack_scenarios as shc
import synthetic_hudson_structure_map_two_sills as base2

sm = base2.sm   # the underlying synthetic_hudson_structure_map module

print(f'Base sill1: x0={shc.SILL["x0"]:.3f}, y0={shc.SILL["y0"]:.3f}')
print(f'Base dike waypoints: {shc.DIKE_WAYPOINTS}')

_, _P1, _P2, _P3, _P4 = shc.DIKE_WAYPOINTS

_NEW_P0 = (shc.STATION_XY['AXEC1'][0] - 0.25, shc.STATION_XY['AXEC1'][1])
_NEW_WAYPOINTS = [_NEW_P0, _P1, _P2, _P3, _P4]

shc.SILL['x0'] = _NEW_P0[0]
shc.SILL['y0'] = _NEW_P0[1]
shc.DIKE_WAYPOINTS = _NEW_WAYPOINTS
sm.DIKE_WAYPOINTS = _NEW_WAYPOINTS   # sm's own separately-imported name, see v2 map docstring

print(f'\nNew sill1 (0.25 km west of AXEC1): x0={shc.SILL["x0"]:.3f}, y0={shc.SILL["y0"]:.3f}')
print(f'New dike waypoints: {[(round(x,4), round(y,4)) for x,y in _NEW_WAYPOINTS]}')
print(f'SILL2 (Kidiwela S2 location, unchanged): x0={base2.SILL2["x0"]:.2f}, '
      f'y0={base2.SILL2["y0"]:.2f}')

base2.OUT_PDF = base2.OUT_PDF.replace('synthetic_hudson_structure_map_two_sills.pdf',
                                      'synthetic_hudson_structure_map_two_sills_v2.pdf')

if __name__ == '__main__':
    base2.main()
