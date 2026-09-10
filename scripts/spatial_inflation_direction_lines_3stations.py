#!/usr/bin/env python3
"""
spatial_inflation_direction_lines_3stations.py

New static figure, per explicit user request: for each of the three stations (AXEC2, AXCC1,
AXAS1), take the fitted atan2 vector-sum model's INFLATION vector direction (beta -- see
animate_arctan_stress_vectors.py's fit_atan2_vectorsum/module docstring for the model), in the
EXTENSION convention (fast direction shifted -90 deg, matching that script's own
EXTENSION_SHIFT_DEG -- NOT the raw compressional convention used by
atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.py's own fit), and draw it as a full-length LINE
(not an arrow -- per explicit user request, this is the 180-deg-ambiguous axial direction, not
a directed vector) through that station's real location, on a single whole-caldera grayscale
bathymetry map with all three station triangles.

NOTE ON WHICH BETA: the wrapped y-data fed into the fit is mathematically IDENTICAL whether or
not the -90 extension shift is applied before wrapping (compute_optimal_wrap absorbs any
constant additive shift -- confirmed empirically earlier this session), so "extension vs
compression" does NOT change the fitted beta's value. What DOES differ between separate script
runs is the multi-start optimizer occasionally landing on a different (equally-good, same r)
local optimum -- so this script re-fits fresh, once, for all three stations together, rather
than reusing possibly-stale beta prints from earlier, separate runs of the GIF/PDF scripts.

PURPOSE, per explicit user request: extending each station's inflation-direction line across
the whole map and checking whether the three lines intersect (or what region they bound) is a
simple lateral (map-view) constraint on where a common inflation source could sit, if the
"fixed background + growing inflation from a point source" cartoon has any literal spatial
truth to it.

Produces:
    spatial_inflation_direction_lines_3stations.pdf/.png

Run with:
    python3 spatial_inflation_direction_lines_3stations.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arctan_uplift_vs_phi_axec2_axcc1_axas1_90day import STATIONS
from animate_arctan_stress_vectors import (
    load_station_data, load_station_xy, load_bathy_gray, fit_atan2_vectorsum_joint,
    azimuth_to_math_angle, _azimuth, U_Z0, ALPHA_EXTENSION_RAD,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'spatial_inflation_direction_lines_3stations')

STATION_NAMES = ('AXEC2', 'AXCC1', 'AXAS1')
STATION_COLORS = {'AXEC2': '#D62728', 'AXCC1': '#1F77B4', 'AXAS1': '#2CA02C'}

# Same whole-caldera km-East/North extent convention used elsewhere in this repo (e.g.
# sws_mesh_regional.py's X_START/X_END/Y_START/Y_END) -- confirmed all three stations sit
# comfortably inside it.
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0

# Long enough to run each line off the edge of the map from any station position inside the
# box above (box diagonal is ~14.4 km; 20 km of half-length in each direction is generous).
LINE_HALF_LENGTH_KM = 20.0


def line_through(x0, y0, math_angle_rad, half_length=LINE_HALF_LENGTH_KM):
    """Full-length line segment (NOT an arrow -- axial, 180-deg-ambiguous direction) through
    (x0,y0) at the given math angle, extending half_length in BOTH directions."""
    dx, dy = np.cos(math_angle_rad), np.sin(math_angle_rad)
    return (x0 - half_length * dx, x0 + half_length * dx), (y0 - half_length * dy, y0 + half_length * dy)


def intersect_lines(p1, ang1, p2, ang2):
    """Intersection of two infinite lines, each given as a (point, math-angle-direction) pair.
    Returns None if parallel (or anti-parallel, same axial line)."""
    x1, y1 = p1
    x2, y2 = p2
    d1 = np.array([np.cos(ang1), np.sin(ang1)])
    d2 = np.array([np.cos(ang2), np.sin(ang2)])
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    t = ((x2 - x1) * d2[1] - (y2 - y1) * d2[0]) / denom
    return x1 + t * d1[0], y1 + t * d1[1]


def main():
    stations_by_name = {s['name']: s for s in STATIONS}

    print('Loading data for joint fit (shared A across AXEC2/AXCC1/AXAS1)...')
    datasets = []
    for name in STATION_NAMES:
        x, y, wrap, df, merged = load_station_data(stations_by_name[name])
        datasets.append((name, x, y, U_Z0[name]))

    print('Jointly fitting (alpha = extension-frame 80°, single shared A)...')
    A_shared, per_station = fit_atan2_vectorsum_joint(datasets, ALPHA_EXTENSION_RAD)
    print(f'  Shared background magnitude A = {A_shared:.3f}')

    results = {}
    for name in STATION_NAMES:
        C1, C2, beta, r, model = per_station[name]
        beta_az = _azimuth(beta)
        beta_math = azimuth_to_math_angle(beta_az)
        x0, y0 = load_station_xy(name)
        results[name] = dict(x0=x0, y0=y0, beta_az=beta_az, beta_math=beta_math, r=r)
        print(f'  {name}: inflation vector azimuth (extension convention) = {beta_az:.1f}°, '
             f'r={r:.2f}, station at ({x0:.3f}, {y0:.3f}) km')

    # -- Whole-caldera bathymetry, cropped to the exact conventional box via axis limits after
    # loading a (slightly larger, square) crop -- see load_bathy_gray's docstring.
    cx, cy = (X_START + X_END) / 2.0, (Y_START + Y_END) / 2.0
    lim = max(X_END - X_START, Y_END - Y_START) / 2.0
    bathy_gray, bathy_extent = load_bathy_gray(cx, cy, lim)

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.imshow(bathy_gray, origin='upper', extent=bathy_extent, aspect='auto', cmap='gray',
             alpha=0.5, zorder=0)

    # -- Pairwise intersections, computed BEFORE drawing so we can report/mark them.
    pairs = [('AXEC2', 'AXCC1'), ('AXEC2', 'AXAS1'), ('AXCC1', 'AXAS1')]
    intersections = []
    for a, b in pairs:
        ra, rb = results[a], results[b]
        pt = intersect_lines((ra['x0'], ra['y0']), ra['beta_math'],
                             (rb['x0'], rb['y0']), rb['beta_math'])
        if pt is not None:
            inside = (X_START <= pt[0] <= X_END) and (Y_START <= pt[1] <= Y_END)
            intersections.append((a, b, pt, inside))
            print(f'  Intersection {a}-{b}: ({pt[0]:.3f}, {pt[1]:.3f}) km'
                 f'{"" if inside else " -- OUTSIDE the mapped box"}')
        else:
            intersections.append((a, b, None, False))
            print(f'  Intersection {a}-{b}: lines are parallel -- no intersection')

    for name in STATION_NAMES:
        r_ = results[name]
        color = STATION_COLORS[name]
        (lx0, lx1), (ly0, ly1) = line_through(r_['x0'], r_['y0'], r_['beta_math'])
        ax.plot([lx0, lx1], [ly0, ly1], color=color, lw=1.8, linestyle='-', alpha=0.85,
               zorder=5, label=f'{name}: inflation azimuth β={r_["beta_az"]:.0f}° (r={r_["r"]:.2f})')
        ax.plot(r_['x0'], r_['y0'], marker='^', ms=13, mfc='#FFD700', mec='k', mew=1.2, zorder=12)
        ax.annotate(name, (r_['x0'], r_['y0']), textcoords='offset points', xytext=(8, 8),
                   fontsize=10, fontweight='bold', zorder=13)

    valid_pts = [pt for (_, _, pt, inside) in intersections if pt is not None and inside]
    for a, b, pt, inside in intersections:
        if pt is not None and inside:
            ax.plot(pt[0], pt[1], marker='x', ms=10, mew=2.5, color='black', zorder=14)
    if valid_pts:
        centroid = np.mean(valid_pts, axis=0)
        ax.plot(centroid[0], centroid[1], marker='*', ms=22, mfc='white', mec='black', mew=1.5,
               zorder=15, label=f'Centroid of in-bounds intersections ({centroid[0]:.2f}, '
                                f'{centroid[1]:.2f}) km')
        print(f'  Centroid of {len(valid_pts)} in-bounds intersection(s): '
             f'({centroid[0]:.3f}, {centroid[1]:.3f}) km')
    else:
        print('  No pairwise intersection falls inside the mapped box.')

    ax.set_xlim(X_START, X_END)
    ax.set_ylim(Y_START, Y_END)
    ax.set_aspect('equal')
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_title('Fitted inflation-vector direction lines (extension convention)\n'
                 f'AXEC2 / AXCC1 / AXAS1 -- joint fit, shared A={A_shared:.2f} -- '
                 'full-length, 180° axial ambiguity',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        out_path = f'{OUT_BASE}.{ext}'
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
