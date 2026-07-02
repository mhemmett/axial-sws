#!/usr/bin/env python3
"""
seismicity_map.py

Large plan-view map of Axial Seamount with:
  1. MBARI GeoTIFF bathymetry (color, slightly desaturated)
  2. Earthquake density overlay (Reds colormap, Gaussian-smoothed)
  3. OBS stations in gold (#FFD700) with black edge and labels
  4. Axes in geographic lat/lon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.ma as ma
import tifffile
import os
from PIL import Image as PILImage, ImageEnhance
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE          = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE  = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
BATHY_GEOTIFF = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
                 'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
OUT_DIR  = BASE
OUT_PATH = os.path.join(BASE, 'seismicity_map.pdf')

# ── Map extent (lat/lon) ──────────────────────────────────────────────────────

LAT_MIN, LAT_MAX = 45.90, 46.00
LON_MIN, LON_MAX = -130.04, -129.96

# ── Coordinate projection (km, for density binning) ──────────────────────────

INI_LON        = -130.1
INI_LAT        =  45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


def xy2ll(x, y):
    return (np.asarray(y) / KM_PER_DEG_LAT + INI_LAT,
            np.asarray(x) / KM_PER_DEG_LON + INI_LON)


# ── Density grid parameters ───────────────────────────────────────────────────

STEP    = 0.03   # km grid spacing
DIS_LIM = 0.12   # km search radius per bin
NUM_LIM = 500    # no cap — density needs all events
GAUSS_SIGMA = 1.5  # grid cells to smooth

# Map limits in km
X_START = (LON_MIN - INI_LON) * KM_PER_DEG_LON
X_END   = (LON_MAX - INI_LON) * KM_PER_DEG_LON
Y_START = (LAT_MIN - INI_LAT) * KM_PER_DEG_LAT
Y_END   = (LAT_MAX - INI_LAT) * KM_PER_DEG_LAT

# ── Stations ──────────────────────────────────────────────────────────────────

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

LABEL_OFFSET_DEG = {
    'AXAS1': ( 0.002,  -0.0060),   # below triangle, shifted right to clear AXAS2
    'AXAS2': (-0.010,  -0.0060),   # below triangle, shifted further left
    'AXCC1': (-0.010,   0.0030),   # above-left
    'AXEC1': ( 0.0018,  0.0035),   # above-right
    'AXEC2': ( 0.0018,  0.0008),   # right
    'AXEC3': ( 0.0035, -0.0030),   # below-right, clear of AXEC2
}

stations = pd.read_csv(
    STATION_FILE, sep=r'\s+',
    names=['lon', 'lat', 'elev_km', 'station'],
    engine='python',
).set_index('station')
stations = stations.loc[[s for s in STATIONS if s in stations.index]]

# ── Load all splitting data (combined, no filters) ────────────────────────────

def _load(path):
    df = pd.read_csv(BASE + path)
    df = df.loc[:, :'dt_error'].dropna()
    df = df[df['dt'] > 0]   # remove null measurements (dt=0 = no splitting detected)
    return df

print('Loading catalogs...')
all_dfs = []
for df1, df2 in [
    (_load('splitting_results_mldd_2015_2021_axec3_all_batches.csv'),
     _load('splitting_results_mldd_2022_2026_axec3_all_batches.csv')),
    (_load('splitting_results_mldd_2015_2021_axec1_all_batches.csv'),
     _load('splitting_results_mldd_2022_2026_axec1_all_batches.csv')),
    (_load('axial-mldd-2015-2021-axec2.csv'),
     _load('splitting_results_mldd_2022_2026_axec2_all_batches.csv')),
    (_load('splitting_results_mldd_2015_2021_axas1.csv'),
     _load('splitting_results_mldd_2022_2026_axas1_all_batches.csv')),
    (_load('splitting_results_mldd_2015_2021_axas2.csv'),
     _load('splitting_results_mldd_2022_2026_axas2_all_batches.csv')),
    (_load('splitting_results_mldd_2015_2021_axcc1_all_batches.csv'),
     _load('splitting_results_mldd_2022_2026_axcc1_all_batches.csv')),
]:
    all_dfs.append(pd.concat([df1, df2], ignore_index=True))

all_df = pd.concat(all_dfs, ignore_index=True)

# Unique earthquakes only (avoid counting same event multiple times)
unique_eq = all_df.drop_duplicates(subset=['event_datetime', 'event_lat', 'event_lon'])
eq_x, eq_y = ll2xy(unique_eq['event_lat'].values, unique_eq['event_lon'].values)
print(f'  {len(unique_eq):,} unique earthquakes')

# ── Compute 2D density ────────────────────────────────────────────────────────

print('Computing earthquake density...')
x_nodes = np.arange(X_START, X_END + STEP * 0.5, STEP)
y_nodes = np.arange(Y_START, Y_END + STEP * 0.5, STEP)
X_grid, Y_grid = np.meshgrid(x_nodes, y_nodes)
COUNT = np.zeros(X_grid.shape, dtype=float)

tree      = cKDTree(np.column_stack([eq_x, eq_y]))
grid_pts  = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
neighbors = tree.query_ball_point(grid_pts, DIS_LIM)

for k, nbrs in enumerate(neighbors):
    COUNT.ravel()[k] = len(nbrs)

COUNT_smooth = gaussian_filter(COUNT, sigma=GAUSS_SIGMA)
# Fade out low-density bins by building an RGBA array where alpha scales
# with density (transparent below a threshold, opaque at peak density)
COUNT_masked = ma.array(COUNT_smooth, mask=COUNT_smooth < 1)

# Convert grid to lat/lon for plotting
LAT_grid, LON_grid = xy2ll(X_grid, Y_grid)
print(f'  Density range: 0 – {COUNT_smooth.max():.0f} events/bin')

# ── Load and tone down bathymetry ─────────────────────────────────────────────

print('Loading bathymetry...')
PILImage.MAX_IMAGE_PIXELS = None
_pil      = PILImage.open(BATHY_GEOTIFF)
_tags     = _pil.tag_v2
_orig_lon = _tags[33922][3]
_orig_lat = _tags[33922][4]
_px_lon   = _tags[33550][0]
_px_lat   = _tags[33550][1]
_nc, _nr  = _pil.size
_pil.close()

# Crop to map extent
_c0 = max(0,   int((LON_MIN - _orig_lon) / _px_lon) - 2)
_c1 = min(_nc, int((LON_MAX - _orig_lon) / _px_lon) + 2)
_r0 = max(0,   int((_orig_lat - LAT_MAX) / _px_lat) - 2)
_r1 = min(_nr, int((_orig_lat - LAT_MIN) / _px_lat) + 2)

bathy_rgb = tifffile.imread(BATHY_GEOTIFF)[_r0:_r1, _c0:_c1]

# Tone down saturation (0 = grayscale, 1 = original)
bathy_pil  = PILImage.fromarray(bathy_rgb)
bathy_pil  = ImageEnhance.Color(bathy_pil).enhance(0.50)
bathy_toned = np.array(bathy_pil)

# Downsample to ≤2048 for fast rendering at large figure size
_ds = max(1, max(bathy_toned.shape[:2]) // 2048)
bathy_toned = bathy_toned[::_ds, ::_ds]

bathy_extent_ll = [
    _orig_lon + _c0 * _px_lon,
    _orig_lon + _c1 * _px_lon,
    _orig_lat - _r1 * _px_lat,
    _orig_lat - _r0 * _px_lat,
]
print(f'  Bathymetry: {bathy_toned.shape[1]}×{bathy_toned.shape[0]} px')

DENSITY_THRESHOLD = 100

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.ticker as mticker


def make_map(eq_lats, eq_lons, title, out_path, highlight_sta=None):
    """Build and save one seismicity density map."""
    # ── Density ───────────────────────────────────────────────────────────────
    eq_x_m, eq_y_m = ll2xy(eq_lats, eq_lons)
    x_nodes = np.arange(X_START, X_END + STEP * 0.5, STEP)
    y_nodes = np.arange(Y_START, Y_END + STEP * 0.5, STEP)
    Xg, Yg  = np.meshgrid(x_nodes, y_nodes)
    C = np.zeros(Xg.shape, dtype=float)
    tree_m   = cKDTree(np.column_stack([eq_x_m, eq_y_m]))
    gpts     = np.column_stack([Xg.ravel(), Yg.ravel()])
    for k, nbrs in enumerate(tree_m.query_ball_point(gpts, DIS_LIM)):
        C.ravel()[k] = len(nbrs)
    C_smooth = gaussian_filter(C, sigma=GAUSS_SIGMA)
    LAT_g, LON_g = xy2ll(Xg, Yg)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(bathy_toned, origin='upper',
              extent=bathy_extent_ll, aspect='auto', zorder=0)

    cmap_dens = plt.colormaps['Reds']
    dens_max  = float(np.percentile(C_smooth[C_smooth > 0], 99)) if C_smooth.max() > 0 else 1
    dens_norm = mcolors.Normalize(vmin=DENSITY_THRESHOLD, vmax=dens_max)
    rgba      = cmap_dens(dens_norm(C_smooth))
    rgba[..., 3] = np.where(C_smooth >= DENSITY_THRESHOLD, 0.85, 0.0)
    ax.imshow(rgba, origin='lower',
              extent=[LON_g.min(), LON_g.max(), LAT_g.min(), LAT_g.max()],
              aspect='auto', zorder=2, interpolation='bilinear')

    sm = cm.ScalarMappable(cmap=cmap_dens, norm=dens_norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax  = divider.append_axes('right', size='3%', pad=0.12)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Earthquake Density (Events / Bin)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # ── Stations ──────────────────────────────────────────────────────────────
    for sta, row in stations.iterrows():
        mfc = '#FFD700' if (highlight_sta is None or sta == highlight_sta) else 'white'
        ax.scatter(row['lon'], row['lat'], marker='^', s=700,
                   facecolor=mfc, edgecolors='k', linewidths=2.0, zorder=10)
        dx, dy = LABEL_OFFSET_DEG.get(sta, (0.001, 0.002))
        ax.text(row['lon'] + dx, row['lat'] + dy, sta,
                fontsize=24, fontweight='bold', zorder=11)

    # ── Cosmetics ─────────────────────────────────────────────────────────────
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude (°E)', fontsize=17, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=17, fontweight='bold')
    ax.tick_params(labelsize=15)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f°'))
    ax.set_title(title, fontsize=22, fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# ── All-stations map ──────────────────────────────────────────────────────────

print('Plotting all-stations map...')
make_map(
    unique_eq['event_lat'].values, unique_eq['event_lon'].values,
    title='Axial Seamount — Earthquake Density 2015–2026',
    out_path=OUT_PATH,
    highlight_sta=None,
)

# ── Per-station maps ──────────────────────────────────────────────────────────

STA_FILES = {
    'AXAS1': ('splitting_results_mldd_2015_2021_axas1.csv',
              'splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2': ('splitting_results_mldd_2015_2021_axas2.csv',
              'splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1': ('splitting_results_mldd_2015_2021_axcc1_all_batches.csv',
              'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1': ('splitting_results_mldd_2015_2021_axec1_all_batches.csv',
              'splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2': ('axial-mldd-2015-2021-axec2.csv',
              'splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3': ('splitting_results_mldd_2015_2021_axec3_all_batches.csv',
              'splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

print('Plotting per-station maps...')
for sta, (f1, f2) in STA_FILES.items():
    df1 = _load(f1)
    df2 = _load(f2)
    sta_df  = pd.concat([df1, df2], ignore_index=True)
    sta_uniq = sta_df.drop_duplicates(subset=['event_datetime', 'event_lat', 'event_lon'])
    print(f'  {sta}: {len(sta_uniq):,} unique earthquakes')
    make_map(
        sta_uniq['event_lat'].values, sta_uniq['event_lon'].values,
        title=f'{sta} — Earthquake Density 2015–2026',
        out_path=os.path.join(OUT_DIR, f'seismicity_map_{sta}.pdf'),
        highlight_sta=sta,
    )
