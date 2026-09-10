#!/usr/bin/env python3
"""
convert_deformation_spatialdb.py

Converts Christian Baillard's DMODELS surface/near-surface deformation grids
(../../Axial_Deformation/def_*.xyzuvw -- ASCII text despite the extension, NOT
a binary MATLAB .mat file; written by axial_comb_V0.m/axial_dike_V0.m) into
PyLith SimpleGridDB ASCII spatial databases (displacement_x/y/z, in metres),
so they can be prescribed as a boundary or initial condition in a PyLith run.

Independent of scripts/deformation_util.py in the main axial-sws repo (per
CLAUDE.md, pylith_axial/ shares no imports/data with scripts/) -- the .xyzuvw
parser below is a small, self-contained re-implementation of the same ASCII
grid format (header "nx N ny N nz N", then one "X Y Z Ux Uy Uz" row per grid
point, Fortran/column-major order -- see read_disp_file() in
scripts/deformation_util.py for the sibling parser in the main repo).

COORDINATE SCALE: written with cs-data.to-meters = 1.0e+3 (coordinates in the
file are in km), matching cfg/pylithapp.cfg's `reader.coordsys.units = km`
fix (2026-07-22) and every other file in spatialdb/. Do NOT use to-meters=1.0
here -- that would only have matched the mesh reader's PRE-FIX behavior (see
pylithapp.cfg's coordinate-scaling comment), which has since been corrected.

Grid point order in the output file does not matter (confirmed from PyLith's
own examples, e.g. box-3d/sheardisp_ic.spatialdb: "Order of points does not
matter") -- each row is self-describing (carries its own x/y/z), so no
axis-ordering convention needs to be replicated here.

Produces one .spatialdb per input file in spatialdb/deformation/, e.g.:
    spatialdb/deformation/def_pre_1.spatialdb
    spatialdb/deformation/def_syn_6.spatialdb
    ...

Run with:
    python3 convert_deformation_spatialdb.py [--files def_pre_1 def_syn_6 ...]
"""

import argparse
import glob
import os

import numpy as np
from scipy.interpolate import NearestNDInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
DEFORM_DIR = os.path.join(HERE, '..', '..', 'Axial_Deformation')
OUT_DIR = os.path.join(HERE, '..', 'spatialdb', 'deformation')

TO_METERS = 1.0e+3   # coordinates in the .xyzuvw files are in km


def read_xyzuvw(path):
    """Self-contained parser for the DMODELS ASCII .xyzuvw format."""
    with open(path, 'rt') as fh:
        lines = fh.readlines()

    header, data_lines = lines[0], lines[2:]
    nx, ny, nz = [int(tok) for tok in header.split()[1::2]]

    rows = np.array([[float(v) for v in line.split()] for line in data_lines])
    x, y, z, ux, uy, uz = rows.T
    ux, uy, uz = fill_nan_nearest(x, y, z, ux, uy, uz, source_name=os.path.basename(path))

    shape = (nx, ny, nz)
    X = x.reshape(shape, order='F')
    Y = y.reshape(shape, order='F')
    Z = z.reshape(shape, order='F')
    Ux = ux.reshape(shape, order='F')
    Uy = uy.reshape(shape, order='F')
    Uz = uz.reshape(shape, order='F')
    return X, Y, Z, Ux, Uy, Uz


def fill_nan_nearest(x, y, z, ux, uy, uz, source_name):
    """A handful of grid points sit exactly at/adjacent to Baillard's analytic
    source singularities and come out NaN in the source .xyzuvw file (the
    Yang-spheroid/Okada-dike displacement is undefined there). PyLith's
    SimpleGridDB ASCII reader can't parse literal 'nan' tokens, and dropping
    points would break the required rectilinear grid, so each NaN point is
    replaced with the nearest (3-D Euclidean, in x/y/z) valid neighbor's
    displacement -- a local patch, not a real measurement, at these specific
    singular points only. Operates on the flat (pre-reshape) 1-D arrays."""
    bad = ~np.isfinite(ux) | ~np.isfinite(uy) | ~np.isfinite(uz)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return ux, uy, uz

    pts = np.column_stack([x, y, z])
    good = ~bad
    ux, uy, uz = ux.copy(), uy.copy(), uz.copy()
    for U in (ux, uy, uz):
        interp = NearestNDInterpolator(pts[good], U[good])
        U[bad] = interp(pts[bad])
    print(f'  {source_name}: filled {n_bad} NaN singular-source point(s) with nearest valid neighbor')
    return ux, uy, uz


def write_spatialgriddb(out_path, X, Y, Z, Ux, Uy, Uz, source_name):
    x_pts = np.unique(X)
    y_pts = np.unique(Y)
    z_pts = np.unique(Z)
    n_expected = len(x_pts) * len(y_pts) * len(z_pts)
    if n_expected != X.size:
        raise ValueError(f'{source_name}: grid is not a clean rectilinear product '
                          f'({X.size} points, expected {n_expected} from unique x/y/z)')

    with open(out_path, 'wt') as fh:
        fh.write(f'// Converted from {source_name} (Baillard DMODELS Yang-spheroid + '
                  'Okada-dike surface deformation grid) by convert_deformation_spatialdb.py.\n')
        fh.write('// Coordinates in km (to-meters=1.0e+3, matches cfg/pylithapp.cfg\n')
        fh.write('// reader.coordsys.units = km); displacement values in metres.\n')
        fh.write('#SPATIAL_GRID.ascii 1\n')
        fh.write('SimpleGridDB {\n')
        fh.write(f'  num-x = {len(x_pts)}\n')
        fh.write(f'  num-y = {len(y_pts)}\n')
        fh.write(f'  num-z = {len(z_pts)}\n')
        fh.write('  num-values = 3\n')
        fh.write('  value-names =  displacement_x  displacement_y  displacement_z\n')
        fh.write('  value-units =  m  m  m\n')
        fh.write('  space-dim = 3\n')
        fh.write('  cs-data = cartesian {\n')
        fh.write(f'    to-meters = {TO_METERS:.1e}\n')
        fh.write('    space-dim = 3\n')
        fh.write('  }\n')
        fh.write('}\n')
        fh.write('// x coordinates (km)\n')
        fh.write('  ' + '  '.join(f'{v:.4f}' for v in x_pts) + '\n\n')
        fh.write('// y coordinates (km)\n')
        fh.write('  ' + '  '.join(f'{v:.4f}' for v in y_pts) + '\n\n')
        fh.write('// z coordinates (km)\n')
        fh.write('  ' + '  '.join(f'{v:.4f}' for v in z_pts) + '\n\n')
        fh.write('// x(km)  y(km)  z(km)  displacement_x(m)  displacement_y(m)  displacement_z(m)\n')
        fh.write('// Order of points does not matter.\n')
        for xi, yi, zi, uxi, uyi, uzi in zip(X.ravel(order='F'), Y.ravel(order='F'),
                                              Z.ravel(order='F'), Ux.ravel(order='F'),
                                              Uy.ravel(order='F'), Uz.ravel(order='F')):
            fh.write(f'{xi:.4f}  {yi:.4f}  {zi:.4f}    {uxi:.6e}  {uyi:.6e}  {uzi:.6e}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', default=None,
                        help='Basenames (no extension) to convert, e.g. def_pre_1 def_syn_6. '
                             'Default: convert every def_*.xyzuvw found.')
    args = parser.parse_args()

    if args.files:
        paths = [os.path.join(DEFORM_DIR, f'{name}.xyzuvw') for name in args.files]
    else:
        paths = sorted(glob.glob(os.path.join(DEFORM_DIR, 'def_*.xyzuvw')))

    os.makedirs(OUT_DIR, exist_ok=True)

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f'Reading {name}.xyzuvw...')
        X, Y, Z, Ux, Uy, Uz = read_xyzuvw(path)
        out_path = os.path.join(OUT_DIR, f'{name}.spatialdb')
        write_spatialgriddb(out_path, X, Y, Z, Ux, Uy, Uz, source_name=f'{name}.xyzuvw')
        print(f'  -> {out_path}  '
              f'(grid {len(np.unique(X))}x{len(np.unique(Y))}x{len(np.unique(Z))}, '
              f'N={X.size:,} points)')

    print(f'\nDone. {len(paths)} spatialdb file(s) written to {OUT_DIR}/')


if __name__ == '__main__':
    main()
