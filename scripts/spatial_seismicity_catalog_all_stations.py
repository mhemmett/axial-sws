#!/usr/bin/env python3
"""
spatial_seismicity_catalog_all_stations.py

Map of all UNIQUE processed events (i.e. events with a phi/dt splitting result -- success and
dt>0, from any of the 6 stations: AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3) as black dots over
Axial Seamount bathymetry, per explicit user request. The same physical earthquake is commonly
recorded/processed at more than one station (each with its own event_id, not globally unique
across stations), so events are deduplicated by origin datetime (a given earthquake has exactly
one origin time in the underlying catalog regardless of which station(s) picked it up) before
plotting -- one dot per unique earthquake, not one dot per station-event record.

Station markers (yellow triangles + labels) are drawn on the TOPMOST layer, above the
seismicity dots, per explicit user instruction. Bathymetry shown in its native color (not
grayscale), Kidiwela source markers omitted, and axes in lat/lon (not local x/y km).

Produces: spatial_seismicity_catalog_all_stations.pdf (1 page)

Run with:
    python3 spatial_seismicity_catalog_all_stations.py
"""

import glob
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
from PIL import Image as PILImage
from matplotlib.backends.backend_pdf import PdfPages

from sws_forward_model import STATION_FILE, ll2xy, INI_LON, INI_LAT, KM_PER_DEG_LON, KM_PER_DEG_LAT

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
AXEC2_META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                              'raw_axec2_all_batches_mfast_filters_metadata.csv')
OUT_PDF = os.path.join(HERE, 'spatial_seismicity_catalog_all_stations.pdf')

STATION_ORDER = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

STATION_FILES = {
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv',
              'splitting_results_AXAS2_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
}

X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
LON_START, LON_END = INI_LON + X_START / KM_PER_DEG_LON, INI_LON + X_END / KM_PER_DEG_LON
LAT_START, LAT_END = INI_LAT + Y_START / KM_PER_DEG_LAT, INI_LAT + Y_END / KM_PER_DEG_LAT


def load_station(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = sorted(glob.glob(os.path.join(
            AXEC2_2015_2021_DIR,
            'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_*.csv')))
        batch_dfs = [pd.read_csv(f) for f in batch_files]
        batch_dfs = [d for d in batch_dfs if len(d) > 0]
        meta = pd.read_csv(AXEC2_META_CSV)[['event_id', 'latitude', 'longitude', 'depth']]
        for d in batch_dfs:
            d = d.merge(meta, on='event_id', how='left')
            dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['latitude', 'longitude', 'datetime'])
    return df[['datetime', 'latitude', 'longitude']]


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
    # per-pixel grayscale luminance, rather than fading to white (alpha) or black.
    _rgb_f = _rgb[..., :3].astype(np.float32)
    _lum = np.dot(_rgb_f, [0.299, 0.587, 0.114])[..., None]
    _rgb = (0.5 * _rgb_f + 0.5 * _lum).astype(np.uint8)
    _ext = [_olon + _c0 * _pl, _olon + _c1 * _pl,
            _olat - _r1 * _pb, _olat - _r0 * _pb]


def main():
    _setup_environment()

    print('Loading processed events (success & dt>0) from all 6 stations...')
    dfs = []
    for sta in STATION_ORDER:
        d = load_station(sta)
        print(f'  {sta}: {len(d):,} processed events')
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f'  TOTAL (all stations, incl. duplicates): {len(all_df):,}')

    # Dedupe by origin datetime -- a given earthquake has exactly one origin time in the
    # underlying catalog, regardless of how many stations picked it up.
    unique_df = all_df.drop_duplicates(subset=['datetime']).copy()
    print(f'  UNIQUE events (deduped by datetime): {len(unique_df):,}')

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(_rgb, origin='upper', extent=_ext, aspect='auto', zorder=0)

    ax.scatter(unique_df['longitude'], unique_df['latitude'], s=0.8, c='black', alpha=0.3,
               linewidths=0, zorder=5, rasterized=True)

    # Station markers drawn LAST (topmost layer), above the seismicity dots. No text labels.
    for sta, row in _sta_df.iterrows():
        ax.plot(row['lon'], row['lat'], '^', ms=13.5, mfc='#FFD700', mec='k', mew=1.0, zorder=20)

    ax.set_xlim(LON_START, LON_END)
    ax.set_ylim(LAT_START, LAT_END)
    ax.set_aspect(1.0 / np.cos(np.radians(INI_LAT)))
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')

    ax.set_title('Seismicity at Axial Seamount, 2015-2026', fontsize=13, fontweight='bold')
    fig.tight_layout()

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
