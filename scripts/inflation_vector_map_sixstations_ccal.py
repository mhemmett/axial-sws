#!/usr/bin/env python3
"""
inflation_vector_map_sixstations_ccal.py

Single grayscale-bathymetry caldera map, per explicit user request: one page, all 6 stations
as yellow triangles + labels (matching the repo-wide station-marker convention -- see
sws_percent_anisotropy.py / animate_arctan_stress_vectors.py), with a red arrow at each
station pointing in the direction of that station's fitted INFLATION vector azimuth (beta_az)
from atan2_uplift_vs_phi_sixstations_ccal_30day.py's atan2 vector-sum fit (Central Caldera BOTPT
uplift vs. fast direction, 30-day windows both axes) -- reuses compute_all_station_fits() from
that script directly rather than re-deriving/copying the fitted betas by hand, so this map can
never drift out of sync with that fit.

Bathymetry + station-xy + local flat-Earth km projection are reused from
animate_arctan_stress_vectors.py's load_bathy_gray()/load_station_xy() (same GeoTIFF, same
INI_LON/INI_LAT origin, same grayscale luminance conversion) -- the single map window is sized
to cover all 6 stations with a fixed margin, rather than that script's per-station small-window
crop (built for its own single-station GIFs).

Arrow length is a fixed fraction of the map's plotted span (ARROW_LEN_FRAC), not a physical
uplift-rate scale -- beta_az is a fitted DIRECTION (azimuth), not a vector with physical
magnitude in km, so all 6 arrows are drawn the same length purely for visibility.

Caveat (see atan2_uplift_vs_phi_sixstations_ccal_30day.py run output / prior conversation):
AXCC1's fit has a much larger A (~42, vs. 0.4-20 for the other 5 stations), which the earlier
per-station page plots showed producing a near-discontinuous (rather than smooth S-curve)
transition -- treat AXCC1's arrow direction with more caution than the other 5 until that fit is
revisited.

Produces (1 page):
    inflation_vector_map_sixstations_ccal.pdf

Run with:
    python3 inflation_vector_map_sixstations_ccal.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from atan2_uplift_vs_phi_sixstations_ccal_30day import STATION_ORDER, compute_all_station_fits
from animate_arctan_stress_vectors import (
    load_bathy_gray, load_station_xy, azimuth_to_math_angle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'inflation_vector_map_sixstations_ccal.pdf')

MAP_MARGIN_KM = 1.0   # extra half-width beyond the stations' own bounding box
ARROW_LEN_FRAC = 0.09  # arrow length as a fraction of the map's plotted span (fixed length,
                       # since beta_az is a direction only -- no physical magnitude to scale to)
LABEL_OFFSET_KM = (0.12, 0.12)


def main():
    print('Fitting all 6 stations (atan2 vector-sum, Central Caldera BOTPT, 30-day windows)...')
    fits = compute_all_station_fits()

    xy = {name: load_station_xy(name) for name in STATION_ORDER}
    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    x0 = 0.5 * (min(xs) + max(xs))
    y0 = 0.5 * (min(ys) + max(ys))
    half_span = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 + MAP_MARGIN_KM
    arrow_len = 2.0 * half_span * ARROW_LEN_FRAC

    print(f'Map center (x0,y0)=({x0:.2f},{y0:.2f}) km, half-span={half_span:.2f} km')
    gray, extent = load_bathy_gray(x0, y0, half_span)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(gray, origin='upper', extent=extent, aspect='auto', cmap='gray', alpha=0.7,
             zorder=0)

    for name in STATION_ORDER:
        sx, sy = xy[name]
        beta_az = fits[name]['beta_az']
        r_val = fits[name]['r']

        ax.plot(sx, sy, marker='^', markersize=12, mfc='#FFD700', mec='k', mew=1.2, zorder=12)
        ax.annotate(name, (sx, sy), textcoords='offset points', xytext=(8, 8), fontsize=11,
                   fontweight='bold', zorder=13)

        math_angle = azimuth_to_math_angle(beta_az)
        dx, dy = arrow_len * np.cos(math_angle), arrow_len * np.sin(math_angle)
        ax.annotate('', xy=(sx + dx, sy + dy), xytext=(sx, sy),
                   arrowprops=dict(arrowstyle='-|>', color='red', lw=2.2), zorder=14)
        print(f'  {name}: inflation vector azimuth (beta_az)={beta_az:.1f}° (r={r_val:.2f})')

    ax.set_xlim(x0 - half_span, x0 + half_span)
    ax.set_ylim(y0 - half_span, y0 + half_span)
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_aspect('equal')
    ax.set_title('Axial Seamount Caldera: Fitted Inflation-Vector Direction per Station\n'
                 '(atan2 vector-sum fit vs. Central Caldera BOTPT uplift, 30-day rolling window)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
