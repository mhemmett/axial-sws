#!/usr/bin/env python3
"""
map_sws_spatial.py

Spatial map of shear-wave splitting results for all Axial Seamount stations,
adapted from Baillard (2018–2019) plotting scripts.

Two figures are produced:
  Fig 1 — All epicenters colored by station, with a rose diagram of fast
           directions plotted AT each station location.
  Fig 2 — Subsample of events shown as fast-direction sticks (phi orientation,
           length ∝ dt), one panel per station.

Background: GeoTIFF bathymetry if BATHY_GEOTIFF path is set and the file
exists; otherwise falls back to cartopy stock_img() (low-res GEBCO/NE
composite).  GeoTIFF can be any single- or 3-band file covering the extent
(e.g. Axial_42m_bis exported to GeoTIFF, or a GEBCO download).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# ── Optional GeoTIFF bathymetry ───────────────────────────────────────────────
# Set this path to a GeoTIFF covering the map extent and it will be used as
# the background instead of stock_img().  Leave as None to use stock_img().

BATHY_GEOTIFF = '/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif'


def _add_background(ax, geotiff_path=None):
    """Add bathymetry background to a cartopy axes.

    Reads geotransform from GeoTIFF tags (no rasterio required), crops to the
    map extent, downsamples to ≤1024 px on the long edge, and displays with
    imshow.  Falls back to cartopy stock_img() if the file is missing or any
    step fails.
    """
    if geotiff_path and os.path.exists(geotiff_path):
        try:
            import tifffile
            from PIL import Image as PILImage
            PILImage.MAX_IMAGE_PIXELS = None

            # Read geotransform tags without decoding all pixels
            pil_img = PILImage.open(geotiff_path)
            tags = pil_img.tag_v2
            # ModelTiepointTag=33922: (col,row,z, lon,lat,z)
            # ModelPixelScaleTag=33550: (d_lon, d_lat, dz)
            tiepoint = tags[33922]
            pixscale = tags[33550]
            origin_lon = tiepoint[3]
            origin_lat = tiepoint[4]   # top-left (north) edge
            px_lon     = pixscale[0]
            px_lat     = pixscale[1]
            ncols, nrows = pil_img.size
            pil_img.close()

            # Pixel crop indices for our map extent (add 1-px margin)
            col0 = max(0,     int((LON_MIN - origin_lon) / px_lon) - 1)
            col1 = min(ncols, int((LON_MAX - origin_lon) / px_lon) + 2)
            row0 = max(0,     int((origin_lat - LAT_MAX) / px_lat) - 1)
            row1 = min(nrows, int((origin_lat - LAT_MIN) / px_lat) + 2)

            # Read only the cropped region; tifffile uses mmap so this is fast
            data = tifffile.imread(geotiff_path)[row0:row1, col0:col1]

            # Downsample to ≤1024 px on the long edge for fast rendering
            max_px  = 1024
            step    = max(1, max(data.shape[0], data.shape[1]) // max_px)
            data    = data[::step, ::step]

            # Geographic extent of the cropped+downsampled region
            img_extent = [
                origin_lon + col0 * px_lon,
                origin_lon + col1 * px_lon,
                origin_lat - row1 * px_lat,
                origin_lat - row0 * px_lat,
            ]

            ax.imshow(
                data,
                origin='upper',
                extent=img_extent,
                transform=ccrs.PlateCarree(),
                zorder=0,
            )
            return ax

        except Exception as e:
            print(f'GeoTIFF load failed ({e}), falling back to stock_img()')

    ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), color='#cce5ff', zorder=1)
    return ax

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT_DIR      = BASE  # figures saved here

# ── Map extent ────────────────────────────────────────────────────────────────

LAT_MIN, LAT_MAX = 45.90, 46.00
LON_MIN, LON_MAX = -130.04, -129.96

# ── Flat-Earth projection (matches Baillard's ll2xy, ref = caldera center) ────

REF_LON = -130.00
REF_LAT =  45.94
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(REF_LAT))


def ll2xy(lat, lon):
    x = (np.asarray(lon) - REF_LON) * KM_PER_DEG_LON
    y = (np.asarray(lat) - REF_LAT) * KM_PER_DEG_LAT
    return x, y

# ── Station colors ────────────────────────────────────────────────────────────

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_COLOR = {
    'AXAS1': '#e41a1c',
    'AXAS2': '#ff7f00',
    'AXCC1': '#4daf4a',
    'AXEC1': '#377eb8',
    'AXEC2': '#984ea3',
    'AXEC3': '#a65628',
}

# ── Load splitting results ────────────────────────────────────────────────────

def _load(path, trim=True):
    df = pd.read_csv(BASE + path)
    if trim:
        df = df.loc[:, :'dt_error'].dropna()
    return df


station_data_raw = {
    'AXEC3': (_load('splitting_results_mldd_2015_2021_axec3_all_batches.csv'),
              _load('splitting_results_mldd_2022_2026_axec3_all_batches.csv')),
    'AXEC1': (_load('splitting_results_mldd_2015_2021_axec1_all_batches.csv'),
              _load('splitting_results_mldd_2022_2026_axec1_all_batches.csv')),
    'AXEC2': (_load('axial-mldd-2015-2021-axec2.csv'),
              _load('splitting_results_mldd_2022_2026_axec2_all_batches.csv')),
    'AXAS1': (_load('splitting_results_mldd_2015_2021_axas1.csv'),
              _load('splitting_results_mldd_2022_2026_axas1_all_batches.csv')),
    'AXAS2': (_load('splitting_results_mldd_2015_2021_axas2.csv'),
              _load('splitting_results_mldd_2022_2026_axas2_all_batches.csv')),
    'AXCC1': (_load('splitting_results_mldd_2015_2021_axcc1_all_batches.csv'),
              _load('splitting_results_mldd_2022_2026_axcc1_all_batches.csv')),
}

# Merge time periods and clip to map extent
dfs = {}
for sta, (df1, df2) in station_data_raw.items():
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[(df['event_lat'] >= LAT_MIN) & (df['event_lat'] <= LAT_MAX) &
            (df['event_lon'] >= LON_MIN) & (df['event_lon'] <= LON_MAX)]
    dfs[sta] = df

print('Events within map extent:')
for sta, df in dfs.items():
    print(f'  {sta}: {len(df):,}')

# ── Station locations ─────────────────────────────────────────────────────────

stations = pd.read_csv(
    STATION_FILE, sep=r'\s+',
    names=['lon', 'lat', 'elev_km', 'station'],
    engine='python'
).set_index('station')
stations = stations.loc[[s for s in STATIONS if s in stations.index]]

# ── Helper: rose diagram ──────────────────────────────────────────────────────

def plot_rose_inset(ax_main, lon0, lat0, phi_deg, r_deg=0.008,
                   n_bins=18, color='k', alpha=0.7):
    """Draw a rose diagram of fast directions centred at (lon0, lat0).

    r_deg controls the radius of the rose in degrees (map units).
    phi_deg values in [-90, 90] or [0, 180] — symmetric directions.
    """
    edges = np.linspace(-90, 90, n_bins + 1)
    counts, _ = np.histogram(phi_deg % 180 - 90, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.max() == 0:
        return
    r_norm = r_deg * counts / counts.max()

    # Plot bars as symmetric lines (sticks pointing both ways)
    for angle_deg, r in zip(centers, r_norm):
        theta = np.radians(angle_deg)  # azimuth → math convention
        # phi is from North CW → dx = sin(phi), dy = cos(phi)
        dx = r * np.sin(theta) / np.cos(np.radians(REF_LAT))
        dy = r * np.cos(theta)
        ax_main.plot(
            [lon0 - dx, lon0 + dx],
            [lat0 - dy, lat0 + dy],
            color=color, lw=max(0.8, 3 * r / r_deg), alpha=alpha,
            transform=ccrs.PlateCarree(), solid_capstyle='round',
        )

# ── Figure 1 — epicenters + station roses ─────────────────────────────────────

proj = ccrs.PlateCarree()

fig1, ax1 = plt.subplots(
    figsize=(8, 8),
    subplot_kw=dict(projection=proj)
)
ax1.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
_add_background(ax1, BATHY_GEOTIFF)

# Epicenters — rasterized for speed
for sta in STATIONS:
    df = dfs[sta]
    if df.empty:
        continue
    ax1.scatter(
        df['event_lon'], df['event_lat'],
        s=1.5, color=STA_COLOR[sta], alpha=0.25,
        transform=proj, rasterized=True, zorder=2,
    )

# Rose at each station
for sta in STATIONS:
    df = dfs[sta]
    if df.empty or sta not in stations.index:
        continue
    srow = stations.loc[sta]
    plot_rose_inset(
        ax1,
        lon0=srow['lon'], lat0=srow['lat'],
        phi_deg=df['phi'].values,
        r_deg=0.007,
        color=STA_COLOR[sta],
        alpha=0.85,
    )
    # Station marker
    ax1.scatter(
        srow['lon'], srow['lat'],
        marker='^', s=120,
        color=STA_COLOR[sta], edgecolors='k', linewidths=0.8,
        transform=proj, zorder=10,
    )
    ax1.annotate(
        sta, xy=(srow['lon'], srow['lat']),
        xytext=(4, 4), textcoords='offset points',
        fontsize=7, fontweight='bold',
        transform=proj,
    )

# Gridlines with lat/lon labels
gl = ax1.gridlines(
    crs=proj, draw_labels=True,
    linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = __import__('cartopy.mpl.gridliner', fromlist=['LONGITUDE_FORMATTER']).LONGITUDE_FORMATTER
gl.yformatter = __import__('cartopy.mpl.gridliner', fromlist=['LATITUDE_FORMATTER']).LATITUDE_FORMATTER

# Legend
patches = [mpatches.Patch(color=STA_COLOR[s], label=s) for s in STATIONS]
ax1.legend(handles=patches, loc='lower right', fontsize=8,
           title='Station', title_fontsize=8)

ax1.set_title(
    'Axial Seamount — SWS fast directions (roses) & epicenters\n'
    '2015–2026, all stations',
    fontsize=11,
)

fig1.savefig(os.path.join(OUT_DIR, 'map_sws_roses.png'),
             dpi=200, bbox_inches='tight')
print('Saved map_sws_roses.png')

# ── Figure 2 — per-station stick maps (subsampled) ────────────────────────────
# Each stick: centred on the epicenter, oriented by phi (N=0 CW), length ∝ dt.

N_SUBSAMPLE = 3000   # events per station; reduce if too cluttered
STICK_SCALE = 0.003  # degrees per second of dt

fig2 = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig2, hspace=0.35, wspace=0.3)

for idx, sta in enumerate(STATIONS):
    ax = fig2.add_subplot(gs[idx // 3, idx % 3], projection=proj)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
    _add_background(ax, BATHY_GEOTIFF)

    df = dfs[sta]
    if df.empty:
        ax.set_title(sta, fontsize=9)
        continue

    # Subsample
    rng = np.random.default_rng(42)
    sub = df.sample(n=min(N_SUBSAMPLE, len(df)), random_state=42)

    phi_rad = np.radians(sub['phi'].values)
    dt      = sub['dt'].values
    lons    = sub['event_lon'].values
    lats    = sub['event_lat'].values

    half_lon = STICK_SCALE * dt * np.sin(phi_rad) / 2 / np.cos(np.radians(REF_LAT))
    half_lat = STICK_SCALE * dt * np.cos(phi_rad) / 2

    segs = np.stack(
        [
            np.column_stack([lons - half_lon, lats - half_lat]),
            np.column_stack([lons + half_lon, lats + half_lat]),
        ],
        axis=1,
    )  # (N, 2, 2)  — (seg, endpoint, xy)

    lc = LineCollection(
        segs, colors=STA_COLOR[sta], linewidths=0.5, alpha=0.5,
        transform=proj, rasterized=True, zorder=3,
    )
    ax.add_collection(lc)

    # Station marker
    if sta in stations.index:
        srow = stations.loc[sta]
        ax.scatter(
            srow['lon'], srow['lat'],
            marker='^', s=80,
            color=STA_COLOR[sta], edgecolors='k', linewidths=0.7,
            transform=proj, zorder=10,
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(
        f'{sta}  (n={len(sub):,} / {len(df):,})',
        fontsize=9,
    )

# Reference stick annotation on the last panel
ax.annotate(
    '— dt = 0.1 s', xy=(0.05, 0.05), xycoords='axes fraction',
    fontsize=7, color=STA_COLOR[STATIONS[-1]],
)

fig2.suptitle(
    'Axial Seamount — SWS fast-direction sticks per station\n'
    f'(subsampled ≤{N_SUBSAMPLE} events, stick length ∝ dt)',
    fontsize=12,
)

fig2.savefig(os.path.join(OUT_DIR, 'map_sws_sticks.png'),
             dpi=200, bbox_inches='tight')
print('Saved map_sws_sticks.png')

plt.show()
