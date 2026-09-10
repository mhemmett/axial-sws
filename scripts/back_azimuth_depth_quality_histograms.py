#!/usr/bin/env python3
"""
back_azimuth_depth_quality_histograms.py

Per-station 2D histograms of back-azimuth (station -> event, deg from N) vs.
source depth, and back-azimuth vs. splitting quality Q_w (Wustefeld 2010),
for all 6 stations -- the direct follow-up to the Q_w-mid-band voxel/spatial
tests (axec1_qwband_clustering_stats.py, axec1_qwband_ray_voxel_overlap.py,
qwband_voxel_neighborhood_all_stations.py), none of which found AXEC1 to be
spatially distinctive. Per Wustefeld et al. (2010)'s original theoretical
framing: real, well-constrained nulls cluster at a few discrete
backazimuths (the fast/slow axis directions); an isotropic medium (or a
false null) produces nulls scattered across ALL backazimuths. This checks
that signature directly, rather than via Q_w's spatial voxel coverage.

Baseline cleanup only (success==True, dt>0, quality not NaN) -- no error/
Q_w threshold filtering, so the full null population (Q_w<0) is visible,
matching the existing single-station quality_vs_depth_axec1.py /
quality_vs_geometry_axec1.py convention.

Back-azimuth is computed in the local planar (x, y) frame already used
throughout this session's tomography scripts (station -> event bearing via
atan2(dx, dy), same km-Cartesian projection as
lqt_pykonal_tomography_raylength_anisotropy.ll2xy) rather than re-deriving
lat/lon geodetic azimuths -- consistent with, and reusing, that module's
already-loaded per-event (x, y, z) and per-station (x, y) locations
(rla.load_station(), rla.sta_xy), so no new catalog-join code is needed.

Output: lqt_pykonal_combined_results/back_azimuth_depth_quality_histograms.pdf
    (one page per station, 2 panels: baz-vs-depth, baz-vs-quality)

Run with:
    python3 back_azimuth_depth_quality_histograms.py
"""

import warnings

warnings.filterwarnings('ignore')

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

import lqt_pykonal_tomography_raylength_anisotropy as rla

OUT_DIR = rla.OUT_DIR
OUT_PATH = os.path.join(OUT_DIR, 'back_azimuth_depth_quality_histograms.pdf')

N_BAZ_BINS = 36   # 10-degree bins
N_DEPTH_BINS = 40
N_Q_BINS = 40


def back_azimuth_deg(sta_x, sta_y, ex, ey):
    """Station -> event bearing, degrees from North, clockwise, in [0, 360)."""
    dx = ex - sta_x
    dy = ey - sta_y
    baz = np.degrees(np.arctan2(dx, dy))
    return np.mod(baz, 360.0)


def make_station_page(sta, df):
    baz = df['back_azi'].values
    depth = df['z'].values
    q = df['quality'].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel 1: back-azimuth vs. depth ─────────────────────────────────────
    ax = axes[0]
    baz_edges = np.linspace(0, 360, N_BAZ_BINS + 1)
    depth_edges = np.linspace(depth.min(), depth.max(), N_DEPTH_BINS + 1)
    counts, xe, ye = np.histogram2d(baz, depth, bins=[baz_edges, depth_edges])
    im = ax.pcolormesh(xe, ye, counts.T, cmap='viridis',
                       norm=mcolors.LogNorm(vmin=1, vmax=max(counts.max(), 2)))
    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('Back-azimuth (deg from N)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Source depth (km)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_title('Back-azimuth vs. depth', fontsize=10, fontweight='bold')

    # ── Panel 2: back-azimuth vs. quality Q_w ───────────────────────────────
    ax = axes[1]
    q_edges = np.linspace(-1, 1, N_Q_BINS + 1)
    counts, xe, ye = np.histogram2d(baz, q, bins=[baz_edges, q_edges])
    im = ax.pcolormesh(xe, ye, counts.T, cmap='viridis',
                       norm=mcolors.LogNorm(vmin=1, vmax=max(counts.max(), 2)))
    fig.colorbar(im, ax=ax, label='Count')
    ax.axhline(0.7, color='w', lw=0.8, ls='--', alpha=0.7)
    ax.axhline(-0.5, color='w', lw=0.8, ls='--', alpha=0.7)
    ax.axhline(0.5, color='r', lw=0.8, ls=':', alpha=0.7)
    ax.axhline(-0.5, color='r', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlabel('Back-azimuth (deg from N)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Quality Q$_w$ (Wustefeld 2010)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_title('Back-azimuth vs. quality (dashed=strict/loose Q_w cuts, dotted=mid-band)',
                 fontsize=9, fontweight='bold')

    fig.suptitle(f'{sta} — back-azimuth vs. depth and quality (N={len(df):,}, baseline only)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def main():
    rla._setup_environment()

    print(f'Writing {os.path.basename(OUT_PATH)}...')
    with PdfPages(OUT_PATH) as pdf:
        for sta in rla.STATIONS:
            print(f'\n=== {sta} ===')
            df = rla.load_station(sta)
            df = df.dropna(subset=['quality']).copy()
            sta_x, sta_y = rla.sta_xy[sta]
            df['back_azi'] = back_azimuth_deg(sta_x, sta_y, df['x'].values, df['y'].values)
            print(f'  {len(df):,} baseline measurements with recovered quality')

            fig = make_station_page(sta, df)
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)

    print(f'\nSaved {OUT_PATH}')
    print('Done.')


if __name__ == '__main__':
    main()
