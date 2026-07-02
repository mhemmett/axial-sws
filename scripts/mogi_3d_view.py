#!/usr/bin/env python3
"""
mogi_3d_view.py

3D cutaway figure of the Axial Seamount magma system:
  - Seafloor (grayscale bathymetry) with 6 OBS station markers
  - AMC sphere (Arnulf model, Vs=0 km/s) shown as wireframe
  - Two Kidiwela Mogi source spheres (S1 deep, S2 shallow)
  - Labeled axes: E (km), N (km), Depth (km)
  - Velocity labels on each body
"""

import numpy as np
import tifffile
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import os

# ── Paths & constants ─────────────────────────────────────────────────────────

STATION_FILE  = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
BATHY_GEOTIFF = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
                 'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
OUT_PATH = '/Users/mhemmett/Seismology/axial-splitting-ml/results/mogi_3d_view.pdf'

INI_LON        = -130.1
INI_LAT        =  45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

X_LIM = (4.0, 12.0)    # km
Y_LIM = (0.0, 12.0)    # km
Z_LIM = (0.0,  4.5)    # km depth (positive downward)

def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)

# ── Source / AMC parameters ───────────────────────────────────────────────────

SPHERES = [
    dict(x0=7.57, y0=4.55, d=3.33, R=0.43, label='S1 (deep)',
         color='#E53935', alpha=0.85),
    dict(x0=7.53, y0=6.60, d=1.25, R=0.20, label='S2 (shallow)',
         color='#FB8C00', alpha=0.85),
]

AMC = dict(x0=8.0, y0=5.5, d=2.0, r=1.5,
           label='AMC (Arnulf)',
           color='#1565C0', alpha=0.18)

VS_BACKGROUND = 3.5   # km/s
VS_AMC        = 0.0   # km/s (basaltic melt)

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_LABEL = {s: s[2:] for s in STATIONS}   # drop "AX" prefix

# Optimised dike geometry
DIKE = dict(
    x0=10.01, y_start=3.90, y_end=6.82,
    depth_top=0.31, depth_bot=0.41,
    opening=2.62,
    color='#E91E63', alpha=0.70,
)

# Per-station label offsets (x, y, z) to avoid overlap
STA_LABEL_OFFSET = {
    'AXAS1': ( 0.3, -0.5,  0.0),
    'AXAS2': (-0.8, -0.3,  0.0),
    'AXCC1': (-0.8,  0.3,  0.0),
    'AXEC1': ( 0.3,  0.3,  0.0),
    'AXEC2': ( 0.3,  0.0,  0.0),
    'AXEC3': ( 0.3, -0.4,  0.0),
}


def _sphere_surface(x0, y0, d, r, n=40):
    """Meshgrid of a sphere centred at (x0, y0, d) with radius r."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = d  + r * np.outer(np.ones(n), np.cos(v))
    return x, y, z


def _load_bathy_grayscale():
    """Load and crop the MBARI GeoTIFF to the plot extent, return as grayscale."""
    PILImage.MAX_IMAGE_PIXELS = None
    pil = PILImage.open(BATHY_GEOTIFF)
    tags = pil.tag_v2
    orig_lon = tags[33922][3]
    orig_lat = tags[33922][4]
    px_lon   = tags[33550][0]
    px_lat   = tags[33550][1]
    nc, nr   = pil.size
    pil.close()

    lon_min = INI_LON + X_LIM[0] / KM_PER_DEG_LON
    lon_max = INI_LON + X_LIM[1] / KM_PER_DEG_LON
    lat_min = INI_LAT + Y_LIM[0] / KM_PER_DEG_LAT
    lat_max = INI_LAT + Y_LIM[1] / KM_PER_DEG_LAT

    c0 = max(0,  int((lon_min - orig_lon) / px_lon) - 2)
    c1 = min(nc, int((lon_max - orig_lon) / px_lon) + 2)
    r0 = max(0,  int((orig_lat - lat_max) / px_lat) - 2)
    r1 = min(nr, int((orig_lat - lat_min) / px_lat) + 2)

    rgb = tifffile.imread(BATHY_GEOTIFF)[r0:r1, c0:c1]

    # Downsample and convert to grayscale
    ds = max(1, max(rgb.shape[:2]) // 300)
    rgb = rgb[::ds, ::ds]
    gray = np.dot(rgb[..., :3].astype(np.float32),
                  [0.299, 0.587, 0.114]) / 255.0

    x_ext = [(orig_lon + c0*px_lon - INI_LON)*KM_PER_DEG_LON,
              (orig_lon + c1*px_lon - INI_LON)*KM_PER_DEG_LON]
    y_ext = [(orig_lat - r1*px_lat - INI_LAT)*KM_PER_DEG_LAT,
             (orig_lat - r0*px_lat - INI_LAT)*KM_PER_DEG_LAT]
    return gray, x_ext, y_ext


def make_figure():
    print('Loading bathymetry...')
    gray, x_ext, y_ext = _load_bathy_grayscale()

    print('Loading stations...')
    stations = pd.read_csv(
        STATION_FILE, sep=r'\s+',
        names=['lon', 'lat', 'elev_km', 'station'],
        engine='python',
    ).set_index('station')
    stations = stations.loc[[s for s in STATIONS if s in stations.index]]
    stations['x'], stations['y'] = ll2xy(stations['lat'].values,
                                          stations['lon'].values)

    print('Building 3D figure...')
    fig = plt.figure(figsize=(14, 11))
    ax  = fig.add_subplot(111, projection='3d')

    # ── Seafloor bathymetry as a textured flat surface at z=0 ─────────────────
    nx, ny = gray.shape[1], gray.shape[0]
    xg = np.linspace(x_ext[0], x_ext[1], nx)
    yg = np.linspace(y_ext[0], y_ext[1], ny)
    XG, YG = np.meshgrid(xg, yg)
    ZG     = np.zeros_like(XG)

    # map grayscale [0,1] to RGBA
    gray_rgba = plt.cm.gray(gray)
    ax.plot_surface(XG, YG, ZG,
                    facecolors=gray_rgba,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False,
                    shade=False, zorder=1)

    # seafloor label
    ax.text(X_LIM[0]+0.2, Y_LIM[1]-0.5, 0.0,
            f'Seafloor  Vs = {VS_BACKGROUND:.1f} km/s',
            fontsize=8, color='k', zorder=10)

    # ── OBS stations slightly above seafloor to avoid being hidden ────────────
    Z_STA = 0.05   # km above seafloor
    for sta, row in stations.iterrows():
        ax.scatter(row['x'], row['y'], Z_STA,
                   marker='^', s=220,
                   c='#FFD700', edgecolors='k', linewidths=1.2,
                   zorder=15, depthshade=False)
        dx, dy, dz = STA_LABEL_OFFSET.get(sta, (0.2, 0.2, 0.0))
        ax.text(row['x'] + dx, row['y'] + dy, Z_STA + dz,
                STA_LABEL.get(sta, sta),
                fontsize=9, fontweight='bold', color='k', zorder=16)

    # ── AMC sphere (wireframe + transparent surface) ───────────────────────────
    amc = AMC
    xA, yA, zA = _sphere_surface(amc['x0'], amc['y0'], amc['d'], amc['r'], n=30)
    ax.plot_surface(xA, yA, zA,
                    color=amc['color'], alpha=amc['alpha'],
                    linewidth=0, antialiased=True, shade=True, zorder=5)
    ax.plot_wireframe(xA, yA, zA,
                      color='#0D47A1', linewidth=0.3, alpha=0.35, zorder=6)
    # Label
    ax.text(amc['x0'] + amc['r'] + 0.1,
            amc['y0'],
            amc['d'],
            f"{amc['label']}\nVs = {VS_AMC:.0f} km/s",
            fontsize=8, fontweight='bold',
            color='#0D47A1', zorder=20)

    # ── Mogi source spheres ────────────────────────────────────────────────────
    for sph in SPHERES:
        xs, ys, zs = _sphere_surface(sph['x0'], sph['y0'], sph['d'], sph['R'], n=25)
        ax.plot_surface(xs, ys, zs,
                        color=sph['color'], alpha=sph['alpha'],
                        linewidth=0, antialiased=True, shade=True, zorder=8)
        ax.plot_wireframe(xs, ys, zs,
                          color='k', linewidth=0.4, alpha=0.5, zorder=9)
        # Label with parameters
        ax.text(sph['x0'] + sph['R'] + 0.1,
                sph['y0'],
                sph['d'] + sph['R'] * 0.5,
                (f"{sph['label']}\n"
                 f"R = {sph['R']} km\n"
                 f"d = {sph['d']} km\n"
                 f"ΔP = 0.05 GPa"),
                fontsize=7.5, color='k', zorder=20)

    # ── Optimised dike (vertical N-S tensile crack, shown as rectangle) ────────
    dk = DIKE
    # Four corners of the dike rectangle in 3D: (x, y, depth)
    ys, ye = dk['y_start'], dk['y_end']
    zt, zb = dk['depth_top'], dk['depth_bot']
    xd     = dk['x0']
    # Thin rectangle: plot as a filled Poly3DCollection
    verts = [[(xd, ys, zt), (xd, ye, zt), (xd, ye, zb), (xd, ys, zb)]]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(verts, alpha=dk['alpha'], facecolor=dk['color'],
                            edgecolor='k', linewidth=0.8, zorder=7)
    ax.add_collection3d(poly)
    # Label at mid-point
    ax.text(xd + 0.15, 0.5*(ys+ye), 0.5*(zt+zb),
            f"Dike\n{dk['opening']:.1f} m opening",
            fontsize=7.5, color=dk['color'], fontweight='bold', zorder=20)

    # ── Vertical dashed lines from sources to seafloor (for depth reference) ──
    for sph in SPHERES:
        ax.plot([sph['x0'], sph['x0']],
                [sph['y0'], sph['y0']],
                [0, sph['d'] - sph['R']],
                '--', color='gray', linewidth=0.8, alpha=0.6, zorder=3)

    # ── Depth reference lines on side walls ───────────────────────────────────
    for depth in [1.0, 2.0, 3.0, 4.0]:
        ax.plot(X_LIM, [Y_LIM[0], Y_LIM[0]], [depth, depth],
                '-', color='lightgray', linewidth=0.5, alpha=0.5, zorder=2)
        ax.text(X_LIM[0] - 0.1, Y_LIM[0], depth,
                f'{depth:.0f}', fontsize=7, color='gray',
                ha='right', va='center')

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.set_zlim(Z_LIM[1], Z_LIM[0])   # flip so depth increases downward

    ax.set_xlabel('E (km)', fontsize=11, labelpad=8)
    ax.set_ylabel('N (km)', fontsize=11, labelpad=8)
    ax.set_zlabel('Depth (km)', fontsize=11, labelpad=8)
    ax.zaxis.set_inverted(True)

    ax.set_xticks(np.arange(4, 13, 2))
    ax.set_yticks(np.arange(0, 13, 2))
    ax.set_zticks(np.arange(0, 5, 1))

    ax.set_title('Axial Seamount — Magma System Geometry\n'
                 '(Kidiwela two-sphere Mogi model + Arnulf AMC)',
                 fontsize=12, fontweight='bold', pad=15)

    # Viewing angle: slight elevation, looking from SE
    ax.view_init(elev=25, azim=-50)

    ax.set_box_aspect([
        X_LIM[1] - X_LIM[0],
        Y_LIM[1] - Y_LIM[0],
        Z_LIM[1] - Z_LIM[0],
    ])

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=250, bbox_inches='tight')
    print(f'Saved {OUT_PATH}')
    plt.close(fig)


if __name__ == '__main__':
    make_figure()
