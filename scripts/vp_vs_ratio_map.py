#!/usr/bin/env python3
"""
vp_vs_ratio_map.py

Vp/Vs ratio map from Baillard's 3D velocity models (P-wave: baillard_velocity_p.py, S-wave:
baillard_velocity.py -- same NLL grid geometry, 302x302x92 nodes, 50 m spacing), evaluated on
the same (x,y) grid and 4 depth levels used throughout this session's tomography-style figures
(traveltime_anisotropy_7period_6stations_newdata.py's DEPTH_ROWS: 0.375/1.125/1.875/2.625 km,
i.e. four 0.75 km-thick bins spanning 0-3 km depth), overlaid at opacity=0.5 on grayscale
bathymetry (same MBARI GeoTIFF + loading convention as synthetic_hudson_structure_map.py).

Vp/Vs = vp_at(x,y,z) / vs_at(x,y,z), evaluated directly at each depth level's centre depth (not
ray-traced/event-derived -- this is a static velocity-model property map, not an observational
tomography result).

Produces: vp_vs_ratio_map.pdf (1 page, 4 panels)

Run with:
    python3 vp_vs_ratio_map.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import tifffile
from PIL import Image as PILImage

from baillard_velocity import vs_at
from baillard_velocity_p import vp_at

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'vp_vs_ratio_map.pdf')
STATION_FILE = os.path.join(HERE, '..', 'data', 'stations_axial.llz')

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

# ── Grid (identical to traveltime_anisotropy_7period_6stations_newdata.py) ───────────────────
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
GRID_STEP_KM = 0.1
xn = np.arange(X_START, X_END + GRID_STEP_KM * .5, GRID_STEP_KM)
yn = np.arange(Y_START, Y_END + GRID_STEP_KM * .5, GRID_STEP_KM)
Xg, Yg = np.meshgrid(xn, yn, indexing='ij')

DEPTH_LEVELS = [
    ('0.0-0.75 km', 0.375),
    ('0.75-1.5 km', 1.125),
    ('1.5-2.25 km', 1.875),
    ('2.25-3.0 km', 2.625),
]

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
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


def _stations(ax, sta_df):
    for sta, row in sta_df.iterrows():
        ax.plot(row['x'], row['y'], '^', ms=6, mfc='#FFD700', mec='k', mew=0.7, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=5.5,
                fontweight='bold', zorder=13)


def main():
    print('Loading bathymetry...')
    gray, ext = load_bathymetry()

    sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                         engine='python').set_index('s')
    sta_df = sta_df.loc[[s for s in STATIONS if s in sta_df.index]]
    sta_df['x'], sta_df['y'] = ll2xy(sta_df['lat'].values, sta_df['lon'].values)

    print(f'Evaluating Vp/Vs on a {len(xn)}x{len(yn)} grid at {len(DEPTH_LEVELS)} depth levels...')
    ratios = []
    for label, z0 in DEPTH_LEVELS:
        vp = vp_at(Xg.ravel(), Yg.ravel(), np.full(Xg.size, z0)).reshape(Xg.shape)
        vs = vs_at(Xg.ravel(), Yg.ravel(), np.full(Xg.size, z0)).reshape(Xg.shape)
        ratio = vp / vs
        ratios.append(ratio)
        print(f'  {label} (z={z0} km): Vp/Vs range {ratio.min():.2f}-{ratio.max():.2f}, '
              f'mean {ratio.mean():.2f}')

    vmin = min(r.min() for r in ratios)
    vmax = max(r.max() for r in ratios)
    cmap = plt.colormaps['RdYlBu_r']
    norm = Normalize(vmin, vmax)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, (label, z0), ratio in zip(axes.ravel(), DEPTH_LEVELS, ratios):
        ax.imshow(gray, origin='upper', extent=ext, aspect='auto', cmap='gray', zorder=0)
        ax.pcolormesh(Xg, Yg, ratio, cmap=cmap, norm=norm, alpha=0.5, shading='auto', zorder=5)
        _stations(ax, sta_df)
        ax.set_xlim(X_START, X_END)
        ax.set_ylim(Y_START, Y_END)
        ax.set_aspect('equal', 'box')
        ax.set_title(f'{label} (z={z0} km)', fontsize=11, fontweight='bold')
        ax.set_xlabel('East (km)', fontsize=9)
        ax.set_ylabel('North (km)', fontsize=9)
        ax.tick_params(labelsize=8)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Vp/Vs', fontsize=10)

    fig.suptitle("Vp/Vs ratio from Baillard's 3D velocity models (opacity=0.5 over bathymetry)",
                fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
