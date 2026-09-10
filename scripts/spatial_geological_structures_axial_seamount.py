#!/usr/bin/env python3
"""
spatial_geological_structures_axial_seamount.py

Bathymetry-only base map of Axial Seamount, for a different figure than
spatial_seismicity_catalog_all_stations.py (its structural/geological-context companion): same
50%-desaturated color bathymetry, lat/lon axes, and yellow station triangles (50% larger, per
the same sizing as that script), but per explicit user request this version has NO seismicity
dots, NO legend, and NO station text labels -- just the bathymetry, axes, and triangles, titled
"Geological Structures at Axial Seamount".

Produces: spatial_geological_structures_axial_seamount.pdf (1 page)

Run with:
    python3 spatial_geological_structures_axial_seamount.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from PIL import Image as PILImage
from matplotlib.backends.backend_pdf import PdfPages

from sws_forward_model import STATION_FILE, INI_LON, INI_LAT, KM_PER_DEG_LON, KM_PER_DEG_LAT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'spatial_geological_structures_axial_seamount.pdf')

STATION_ORDER = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
LON_START, LON_END = INI_LON + X_START / KM_PER_DEG_LON, INI_LON + X_END / KM_PER_DEG_LON
LAT_START, LAT_END = INI_LAT + Y_START / KM_PER_DEG_LAT, INI_LAT + Y_END / KM_PER_DEG_LAT


def _setup_environment():
    global _sta_df, _rgb, _ext

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                          engine='python').set_index('s')
    _sta_df = _sta_df.loc[[s for s in STATION_ORDER if s in _sta_df.index]]

    bathy = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
             'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
    PILImage.MAX_IMAGE_PIXELS = None
    _p = PILImage.open(bathy)
    _tag = _p.tag_v2
    _olon, _olat = _tag[33922][3], _tag[33922][4]
    _pl, _pb = _tag[33550][0], _tag[33550][1]
    _nc, _nr = _p.size
    _p.close()
    _c0 = max(0, int((LON_START - _olon) / _pl) - 2)
    _c1 = min(_nc, int((LON_END - _olon) / _pl) + 2)
    _r0 = max(0, int((_olat - LAT_END) / _pb) - 2)
    _r1 = min(_nr, int((_olat - LAT_START) / _pb) + 2)
    _rgb = tifffile.imread(bathy)[_r0:_r1, _c0:_c1]
    _ds = max(1, max(_rgb.shape[:2]) // 1024)
    _rgb = _rgb[::_ds, ::_ds]
    # Reduce color intensity by 50% -- desaturate by blending 50% toward the image's own
    # per-pixel grayscale luminance.
    _rgb_f = _rgb[..., :3].astype(np.float32)
    _lum = np.dot(_rgb_f, [0.299, 0.587, 0.114])[..., None]
    _rgb = (0.5 * _rgb_f + 0.5 * _lum).astype(np.uint8)
    _ext = [_olon + _c0 * _pl, _olon + _c1 * _pl,
            _olat - _r1 * _pb, _olat - _r0 * _pb]


def main():
    _setup_environment()

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(_rgb, origin='upper', extent=_ext, aspect='auto', zorder=0)

    for sta, row in _sta_df.iterrows():
        ax.plot(row['lon'], row['lat'], '^', ms=13.5, mfc='#FFD700', mec='k', mew=1.0, zorder=20)

    ax.set_xlim(LON_START, LON_END)
    ax.set_ylim(LAT_START, LAT_END)
    ax.set_aspect(1.0 / np.cos(np.radians(INI_LAT)))
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')

    ax.set_title('Geological Structures at Axial Seamount', fontsize=13, fontweight='bold')
    fig.tight_layout()

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
