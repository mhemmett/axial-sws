#!/usr/bin/env python3
"""
sws_mesh_plot_mldd.py

Replication of Baillard's sws_mesh_plot.py for the MLDD splitting catalog.

For each station, produces one figure:
  rows  = depth slices (0.5, 1.0, 1.5 km below seafloor)
  cols  = fast direction phi [°]  |  delay time dt [s]

Spatial binning matches Baillard's xyzd2mesh:
  For each 2-D grid node (x, y) at a given depth band, gather all events
  within dis_lim km (3-D distance), cap at num_lim closest, take the median.
  Bins with < COUNT_MIN events are masked white.

scipy cKDTree is used instead of Baillard's triple-nested loop for speed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.ma as ma
import os
import tifffile
from PIL import Image as PILImage
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE          = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE  = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
BATHY_GEOTIFF = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
                 'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
OUT_DIR = BASE

# ── Coordinate projection — Baillard's reference ──────────────────────────────

INI_LON         = -130.1
INI_LAT         =  45.9
KM_PER_DEG_LAT  = 111.32
KM_PER_DEG_LON  = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    x = (np.asarray(lon) - INI_LON) * KM_PER_DEG_LON
    y = (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT
    return x, y


# ── Mesh / binning parameters ─────────────────────────────────────────────────
# (matches Baillard's test_maps_lags.py / sws_mesh_compute.py)

X_START, X_END = 4.0, 12.0   # km
Y_START, Y_END = 0.0, 12.0   # km
STEP           = 0.1          # km grid spacing (Baillard used 0.05)
DIS_LIM        = 0.3          # km search radius
NUM_LIM        = 100          # max events per bin
COUNT_MIN      = 15           # mask bins with fewer events
GAUSSIAN_SIGMA = 0.2 / STEP   # smooth kernel in grid cells (≈ Baillard's gaussian_val=0.2)

DEPTH_SLICES   = [0.5, 1.0, 1.5]   # km below seafloor
Z_HALF_WIDTH   = 0.3                # ±km depth band around each slice

# ── Plot settings matching Baillard's dic_plots ───────────────────────────────

DIC_PLOT = {
    'phi': dict(vmin=0, vmax=180, cmap='hsv', cbar_label='φ [° from N]', cbar_step=30, fmt='%.0f'),
    'dt':  dict(vmin=0, vmax=0.15, cmap='Blues', cbar_label='dt [s]',    cbar_step=0.025, fmt='%.3f'),
}
PARAMETERS = ['phi', 'dt']

# ── Station list and colors ───────────────────────────────────────────────────

STATIONS  = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_COLOR = {
    'AXAS1': '#e41a1c', 'AXAS2': '#ff7f00', 'AXCC1': '#4daf4a',
    'AXEC1': '#377eb8', 'AXEC2': '#984ea3', 'AXEC3': '#a65628',
}

# ── Load splitting results ────────────────────────────────────────────────────

DT_ERROR_MAX = None  # None = all data (no error filter); 0.1 = filtered

# Display labels — drop the "AX" prefix for cleaner plot labels
STA_DISPLAY = {
    'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
    'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3',
}


def _load(path, trim=True):
    df = pd.read_csv(BASE + path)
    if trim:
        df = df.loc[:, :'dt_error'].dropna()
    if DT_ERROR_MAX is not None:
        df = df[df['dt_error'] < DT_ERROR_MAX]
    df = df[df['dt'] > 0]   # remove null measurements (dt=0 = no splitting detected)
    return df


FILE_SUFFIX = ''   # '' = all data, '_filtered' = dt_error < 0.1 s


def _load_all_dfs():
    """Reload all station DataFrames using current DT_ERROR_MAX and FILE_SUFFIX."""
    raw = {
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
    result = {}
    for sta, (df1, df2) in raw.items():
        df = pd.concat([df1, df2], ignore_index=True)
        df['x'], df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
        df['z'] = df['event_depth'].values
        df['t'] = pd.to_datetime(df['event_datetime'], utc=True)
        df['phi_az'] = df['phi'] + 90.0
        result[sta] = df
        print(f'  {sta}: {len(df):,} events')
    return result


# ── AMC geometry and ray-path functions ──────────────────────────────────────
# (Arnulf model sphere — Vs=0, Vs=0 barrier)

_AMC_C = np.array([8.0, 5.5, 2.0])   # km (x, y, depth below seafloor)
_AMC_R = 1.5                           # km


def _tangent_path_km(p1, p2):
    """Minimum path (km) from p1 to p2 routing around the AMC sphere."""
    straight = float(np.linalg.norm(p2 - p1))
    a2 = p1[:2]; b2 = p2[:2]; c2 = _AMC_C[:2]
    da = np.linalg.norm(a2 - c2); db = np.linalg.norm(b2 - c2)
    if da <= _AMC_R or db <= _AMC_R:
        return straight
    tang_a = np.sqrt(max(da**2 - _AMC_R**2, 0))
    tang_b = np.sqrt(max(db**2 - _AMC_R**2, 0))
    arc_ang = abs(np.arctan2(float(np.cross(a2 - c2, b2 - c2)),
                              float(np.dot(a2 - c2, b2 - c2))))
    arc = _AMC_R * max(arc_ang - np.arcsin(min(_AMC_R/da, 1))
                                - np.arcsin(min(_AMC_R/db, 1)), 0)
    path_2d = tang_a + arc + tang_b
    s2d = np.linalg.norm(b2 - a2)
    return path_2d * (straight / s2d) if s2d > 0 else straight


def compute_ray_lengths_batch(eq_x, eq_y, eq_z, sta_x, sta_y, sta_z=0.0):
    """Vectorized ray path lengths (km) routing around the AMC sphere."""
    p1 = np.column_stack([eq_x, eq_y, eq_z])   # (N, 3)
    p2 = np.array([sta_x, sta_y, sta_z])
    d = p2 - p1
    f = p1 - _AMC_C
    a = np.einsum('ij,ij->i', d, d)
    b = 2 * np.einsum('ij,ij->i', f, d)
    c_ = np.einsum('ij,ij->i', f, f) - _AMC_R**2
    disc = np.clip(b**2 - 4*a*c_, 0, None)
    t1 = (-b - np.sqrt(disc)) / (2*a + 1e-30)
    t2 = (-b + np.sqrt(disc)) / (2*a + 1e-30)
    hits = (disc > 0) & ((t1.clip(0) <= 1) | (t2.clip(0) <= 1))
    straight = np.linalg.norm(d, axis=1)
    lengths = straight.copy()
    for i in np.where(hits)[0]:
        lengths[i] = _tangent_path_km(p1[i], p2)
    return lengths


print('Loading and projecting catalogs...')
dfs = _load_all_dfs()

# ── Station coordinates ───────────────────────────────────────────────────────

stations = pd.read_csv(
    STATION_FILE, sep=r'\s+',
    names=['lon', 'lat', 'elev_km', 'station'],
    engine='python',
).set_index('station')
stations = stations.loc[[s for s in STATIONS if s in stations.index]]
stations['x'], stations['y'] = ll2xy(stations['lat'].values, stations['lon'].values)

# ── Compute ray path lengths and dt_norm = dt / ray_L ─────────────────────────
print('Computing ray path lengths...')
for sta, df in dfs.items():
    if sta not in stations.index:
        continue
    srow = stations.loc[sta]
    ray_L = compute_ray_lengths_batch(
        df['x'].values, df['y'].values, df['z'].values,
        float(srow['x']), float(srow['y']), sta_z=0.0,
    )
    df['ray_L_km'] = ray_L
    df['dt_norm']  = df['dt'].values / np.maximum(ray_L, 0.1)   # s/km
    mean_L = ray_L.mean()
    print(f'  {sta}: mean ray length = {mean_L:.2f} km')

# ── Pre-load and crop bathymetry ──────────────────────────────────────────────

print('Loading bathymetry...')
PILImage.MAX_IMAGE_PIXELS = None
_pil     = PILImage.open(BATHY_GEOTIFF)
_tags    = _pil.tag_v2
_orig_lon = _tags[33922][3]
_orig_lat = _tags[33922][4]
_px_lon   = _tags[33550][0]
_px_lat   = _tags[33550][1]
_nc, _nr  = _pil.size
_pil.close()

# Map extent in lon/lat
_lon_min = INI_LON + X_START / KM_PER_DEG_LON
_lon_max = INI_LON + X_END   / KM_PER_DEG_LON
_lat_min = INI_LAT + Y_START / KM_PER_DEG_LAT
_lat_max = INI_LAT + Y_END   / KM_PER_DEG_LAT

_c0 = max(0,   int((_lon_min - _orig_lon) / _px_lon) - 2)
_c1 = min(_nc, int((_lon_max - _orig_lon) / _px_lon) + 2)
_r0 = max(0,   int((_orig_lat - _lat_max) / _px_lat) - 2)
_r1 = min(_nr, int((_orig_lat - _lat_min) / _px_lat) + 2)

_bathy_rgb = tifffile.imread(BATHY_GEOTIFF)[_r0:_r1, _c0:_c1]
_ds = max(1, max(_bathy_rgb.shape[:2]) // 1024)
_bathy_rgb = _bathy_rgb[::_ds, ::_ds]

# Extent in km (for plain-axes imshow)
_bathy_km_extent = [
    (_orig_lon + _c0 * _px_lon - INI_LON) * KM_PER_DEG_LON,   # left
    (_orig_lon + _c1 * _px_lon - INI_LON) * KM_PER_DEG_LON,   # right
    (_orig_lat - _r1 * _px_lat - INI_LAT) * KM_PER_DEG_LAT,   # bottom
    (_orig_lat - _r0 * _px_lat - INI_LAT) * KM_PER_DEG_LAT,   # top
]
print(f'  Bathymetry loaded: {_bathy_rgb.shape[1]}×{_bathy_rgb.shape[0]} px  '
      f'  km extent x=[{_bathy_km_extent[0]:.1f},{_bathy_km_extent[1]:.1f}]  '
      f'  y=[{_bathy_km_extent[2]:.1f},{_bathy_km_extent[3]:.1f}]')


# ── Spatial binning ───────────────────────────────────────────────────────────

def circular_median_phi(phi_deg):
    """Circular mean of axially-symmetric fast directions in [0, 180] degrees
    (geographic azimuth from North, CW).

    Uses the doubled-angle trick to handle the 180° wrap correctly.
    Returns a value in [0, 180).
    """
    if len(phi_deg) == 0:
        return np.nan
    angles = 2.0 * np.radians(phi_deg)
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    result = float(np.degrees(np.arctan2(s, c)) / 2.0)
    return result % 180.0   # maps (-90, 90] → [0, 180)


def xyzd2mesh_2d(x, y, z, d_phi, d_dt,
                 x_start, x_end, y_start, y_end,
                 z0, z_hw, step, dis_lim, num_lim):
    """2-D spatial mesh for one depth slice.

    Selects events within |z - z0| <= z_hw, then for each (x,y) grid node
    gathers up to num_lim events within dis_lim km (2-D) and computes the
    median of d_phi (circular) and d_dt (linear).

    Returns
    -------
    X, Y         : 2-D coordinate meshes
    PHI_med      : 2-D median fast direction
    DT_med       : 2-D median delay time
    COUNT        : 2-D event count per bin
    """
    # Depth selection
    mask_z = np.abs(z - z0) <= z_hw
    xs, ys = x[mask_z], y[mask_z]
    phi_s  = d_phi[mask_z]
    dt_s   = d_dt[mask_z]

    # Build grid
    x_nodes = np.arange(x_start, x_end  + step * 0.5, step)
    y_nodes = np.arange(y_start, y_end  + step * 0.5, step)
    X, Y    = np.meshgrid(x_nodes, y_nodes)   # shape (ny, nx)
    ny, nx  = X.shape

    PHI_med = np.zeros((ny, nx))
    DT_med  = np.zeros((ny, nx))
    COUNT   = np.zeros((ny, nx), dtype=int)

    if len(xs) < 3:
        return X, Y, PHI_med, DT_med, COUNT

    # KD-tree on event positions
    tree = cKDTree(np.column_stack([xs, ys]))

    grid_pts = np.column_stack([X.ravel(), Y.ravel()])
    neighbors = tree.query_ball_point(grid_pts, dis_lim)

    for k, nbrs in enumerate(neighbors):
        if len(nbrs) == 0:
            continue
        # Sort by distance, cap at num_lim
        pts = np.column_stack([xs[nbrs], ys[nbrs]])
        gp  = grid_pts[k]
        dist = np.hypot(pts[:, 0] - gp[0], pts[:, 1] - gp[1])
        order = np.argsort(dist)[:num_lim]
        sel   = np.array(nbrs)[order]

        COUNT.ravel()[k]   = len(sel)
        PHI_med.ravel()[k] = circular_median_phi(phi_s[sel])
        DT_med.ravel()[k]  = np.median(dt_s[sel])

    return X, Y, PHI_med, DT_med, COUNT


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _add_bathy(ax):
    ax.imshow(_bathy_rgb, origin='upper', extent=_bathy_km_extent,
              aspect='auto', zorder=0)


# Precompute grayscale version for temporal figures
_bathy_gray = np.dot(_bathy_rgb[..., :3].astype(np.float32),
                     [0.299, 0.587, 0.114]).astype(np.uint8)


def _add_bathy_gray(ax):
    ax.imshow(_bathy_gray, origin='upper', extent=_bathy_km_extent,
              aspect='auto', cmap='gray', zorder=0)


# Per-station label offsets (km) — positions requested by user
LABEL_OFFSET_KM = {
    'AXAS1': (-0.20, -0.55),   # below triangle
    'AXAS2': (-0.45, -0.55),   # below-left of triangle
    'AXCC1': (-0.20,  0.25),   # above triangle
    'AXEC1': ( 0.15,  0.30),   # up-right
    'AXEC2': ( 0.15,  0.05),   # right
    'AXEC3': ( 0.30, -0.28),   # farther right, below
}


def _add_stations(ax, highlight=None):
    for sta, row in stations.iterrows():
        # Gold when: this station is highlighted, or no highlight (ALL plots)
        mfc = '#FFD700' if (highlight is None or sta == highlight) else 'white'
        ax.plot(row['x'], row['y'], marker='^', ms=6,
                mfc=mfc, mec='k', mew=0.7, zorder=12)
        dx, dy = LABEL_OFFSET_KM.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta),
                fontsize=5.5, zorder=13)


# ── Output pipeline (called once per pass) ───────────────────────────────────

AXIS_MAP = [X_START, X_END, Y_START, Y_END]
N_ROWS   = len(DEPTH_SLICES)
N_COLS   = len(PARAMETERS)


def _run_outputs():
    """Run all output-generating sections using current global dfs / FILE_SUFFIX."""
    global dfs, FILE_SUFFIX, DT_ERROR_MAX, STA_DT_VMAX, _dt_all_max, _dt_all_step

    # Ensure ray lengths and dt_norm are computed for current dfs
    if 'dt_norm' not in next(iter(dfs.values())).columns:
        print('  Computing ray path lengths...')
        for sta, df in dfs.items():
            if sta not in stations.index:
                continue
            srow = stations.loc[sta]
            ray_L = compute_ray_lengths_batch(
                df['x'].values, df['y'].values, df['z'].values,
                float(srow['x']), float(srow['y']), sta_z=0.0,
            )
            df['ray_L_km'] = ray_L
            df['dt_norm']  = df['dt'].values / np.maximum(ray_L, 0.1)

    # Per-station dt vmax — maximum median-bin dt across all depth slices
    # (reflects what's actually plotted rather than raw outliers)
    print('Computing per-station dt median-bin maxima...')
    STA_DT_VMAX = {}
    for sta, df in dfs.items():
        bin_maxes = []
        for z0 in DEPTH_SLICES:
            _, _, _, DT_med, COUNT = xyzd2mesh_2d(
                df['x'].values, df['y'].values, df['z'].values,
                df['phi_az'].values, df['dt'].values,
                X_START, X_END, Y_START, Y_END,
                z0=z0, z_hw=Z_HALF_WIDTH,
                step=STEP, dis_lim=DIS_LIM, num_lim=NUM_LIM,
            )
            filled = DT_med[COUNT >= COUNT_MIN]
            if len(filled):
                bin_maxes.append(float(filled.max()))
        dt_max = float(max(bin_maxes)) if bin_maxes else 0.15
        step   = round(dt_max / 6, 3)
        STA_DT_VMAX[sta] = (dt_max, step)
        print(f'  {sta} dt max median-bin: {dt_max:.3f} s  (step {step:.3f} s)')

    # Compute _dt_all_max from the combined-dataset mesh, not from per-station maxima.
    # Pooling all stations changes the binning medians, so we need to run the mesh
    # on the combined dataset directly.
    _all_combined = pd.concat(dfs.values(), ignore_index=True)
    _all_bin_maxes = []
    for _z0 in DEPTH_SLICES:
        _, _, _, _DT_med, _COUNT = xyzd2mesh_2d(
            _all_combined['x'].values, _all_combined['y'].values, _all_combined['z'].values,
            _all_combined['phi_az'].values, _all_combined['dt'].values,
            X_START, X_END, Y_START, Y_END,
            z0=_z0, z_hw=Z_HALF_WIDTH,
            step=STEP, dis_lim=DIS_LIM, num_lim=NUM_LIM,
        )
        _filled = _DT_med[_COUNT >= COUNT_MIN]
        if len(_filled):
            _all_bin_maxes.append(float(_filled.max()))
    _dt_all_max  = float(max(_all_bin_maxes)) if _all_bin_maxes else 0.15
    _dt_all_step = round(_dt_all_max / 6, 3)
    print(f'  All-stations combined dt max median-bin: {_dt_all_max:.3f} s  (step {_dt_all_step:.3f} s)')

    for sta in STATIONS:
        df  = dfs[sta]
        x   = df['x'].values
        y   = df['y'].values
        z   = df['z'].values
        phi = df['phi_az'].values
        dt  = df['dt'].values

        print(f'\n{sta}: computing meshes...')

        fig = plt.figure(figsize=(N_COLS * 3.8, N_ROWS * 3.5))
        # Two data columns, each followed by a narrow colorbar column
        width_ratios = []
        for _ in range(N_COLS):
            width_ratios += [1, 0.07]
        gs = GridSpec(N_ROWS, N_COLS * 2,
                      width_ratios=width_ratios,
                      hspace=0.08, wspace=0.05)

        for row_idx, z0 in enumerate(DEPTH_SLICES):
            print(f'  z = {z0} km...', end=' ', flush=True)

            X, Y, PHI_med, DT_med, COUNT = xyzd2mesh_2d(
                x, y, z, phi, dt,
                X_START, X_END, Y_START, Y_END,
                z0=z0, z_hw=Z_HALF_WIDTH,
                step=STEP, dis_lim=DIS_LIM, num_lim=NUM_LIM,
            )
            print(f'{int((COUNT >= COUNT_MIN).sum())} filled bins')

            for col_idx, param in enumerate(PARAMETERS):
                ax = fig.add_subplot(gs[row_idx, col_idx * 2])

                # Bathymetry background
                _add_bathy_gray(ax)

                dp  = DIC_PLOT[param]
                cmap = plt.cm.get_cmap(dp['cmap'])
                vmin, vmax = dp['vmin'], dp['vmax']

                # Data to plot
                D_raw = PHI_med if param == 'phi' else DT_med
                D_smooth = gaussian_filter(D_raw, sigma=GAUSSIAN_SIGMA)

                # Mask low-count bins
                D_masked = ma.array(D_smooth, mask=COUNT < COUNT_MIN)

                # Color contourf (matches Baillard's type_plot='contourf')
                levels = np.linspace(vmin, vmax, 15)
                h = ax.contourf(X, Y, D_masked, levels=levels,
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                extend='neither', zorder=2, alpha=0.85)

                # Count contour — shows coverage boundary (Baillard's mask contour)
                ax.contour(X, Y, COUNT, levels=[COUNT_MIN],
                           colors='k', linewidths=0.4, linestyles='--', zorder=3)

                # Station markers
                _add_stations(ax, highlight=sta)

                # Axes cosmetics
                ax.set_xlim(X_START, X_END)
                ax.set_ylim(Y_START, Y_END)
                ax.set_aspect('equal', 'box')

                if row_idx == 0:
                    ax.set_title(dp['cbar_label'], fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f'z = {z0} km\nN (km)', fontsize=7)
                else:
                    ax.set_yticklabels([])
                if row_idx == N_ROWS - 1:
                    ax.set_xlabel('E (km)', fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.tick_params(labelsize=6)

                # Colorbar — one per parameter column, drawn on last row
                if row_idx == N_ROWS - 1:
                    cax = fig.add_subplot(gs[:, col_idx * 2 + 1])
                    ticks = np.arange(vmin, vmax + dp['cbar_step'] * 0.5, dp['cbar_step'])
                    cb = plt.colorbar(h, cax=cax, ticks=ticks,
                                      format=FormatStrFormatter(dp['fmt']))
                    cb.set_label(dp['cbar_label'], fontsize=8)
                    cb.ax.tick_params(labelsize=7)

        fig.suptitle(sta, fontsize=13, fontweight='bold', y=0.995)
        fig_path = os.path.join(OUT_DIR, f'sws_mesh_{sta}{FILE_SUFFIX}.pdf')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'  Saved {fig_path}')
        plt.close(fig)

    # ── Combined all-stations figures ────────────────────────────────────────────
    # Pre-compute meshes for every station × depth so we only bin once.

    print('\nPre-computing meshes for combined figures...')
    meshes = {}   # meshes[sta][z0] = (X, Y, PHI_med, DT_med, COUNT)
    for sta in STATIONS:
        df  = dfs[sta]
        meshes[sta] = {}
        for z0 in DEPTH_SLICES:
            X, Y, PHI_med, DT_med, COUNT = xyzd2mesh_2d(
                df['x'].values, df['y'].values, df['z'].values,
                df['phi_az'].values, df['dt'].values,
                X_START, X_END, Y_START, Y_END,
                z0=z0, z_hw=Z_HALF_WIDTH,
                step=STEP, dis_lim=DIS_LIM, num_lim=NUM_LIM,
            )
            meshes[sta][z0] = (X, Y, PHI_med, DT_med, COUNT)
            print(f'  {sta} z={z0}: {int((COUNT >= COUNT_MIN).sum())} bins')

    # One combined figure per parameter
    for param in PARAMETERS:
        dp   = DIC_PLOT[param]
        cmap = plt.colormaps[dp['cmap']]
        vmin, vmax = dp['vmin'], dp['vmax']
        levels = np.linspace(vmin, vmax, 15)

        n_rows = len(DEPTH_SLICES)
        n_cols = len(STATIONS)

        # Layout: narrow colorbar column on the LEFT, then data columns
        width_ratios = [0.05] + [1] * n_cols
        fig = plt.figure(figsize=(n_cols * 2.8 + 0.6, n_rows * 2.8 + 0.4))
        gs  = GridSpec(n_rows, n_cols + 1,
                       width_ratios=width_ratios,
                       hspace=0.06, wspace=0.06)

        last_h = None
        for row_idx, z0 in enumerate(DEPTH_SLICES):
            for col_idx, sta in enumerate(STATIONS):
                ax = fig.add_subplot(gs[row_idx, col_idx + 1])  # +1 to skip cbar column
                _add_bathy_gray(ax)

                X, Y, PHI_med, DT_med, COUNT = meshes[sta][z0]
                D_raw    = PHI_med if param == 'phi' else DT_med
                D_smooth = gaussian_filter(D_raw, sigma=GAUSSIAN_SIGMA)
                D_masked = ma.array(D_smooth, mask=COUNT < COUNT_MIN)

                last_h = ax.contourf(X, Y, D_masked, levels=levels,
                                     cmap=cmap, vmin=vmin, vmax=vmax,
                                     extend='neither', zorder=2, alpha=0.85)
                ax.contour(X, Y, COUNT, levels=[COUNT_MIN],
                           colors='k', linewidths=0.3, linestyles='--', zorder=3)
                _add_stations(ax, highlight=sta)

                ax.set_xlim(X_START, X_END)
                ax.set_ylim(Y_START, Y_END)
                ax.set_aspect('equal', 'box')
                ax.tick_params(labelsize=5)

                # Column headers (station names)
                if row_idx == 0:
                    ax.set_title(sta, fontsize=8, fontweight='bold',
                                 color=STA_COLOR.get(sta, 'k'))
                # Row labels (depth) — only on first data column
                if col_idx == 0:
                    ax.set_ylabel(f'z = {z0} km', fontsize=7)
                else:
                    ax.set_yticklabels([])
                if row_idx < n_rows - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('E (km)', fontsize=6)

        # Shared colorbar on the LEFT (column 0)
        cax   = fig.add_subplot(gs[:, 0])
        ticks = np.arange(vmin, vmax + dp['cbar_step'] * 0.5, dp['cbar_step'])
        cb    = plt.colorbar(last_h, cax=cax, ticks=ticks,
                             format=FormatStrFormatter(dp['fmt']))
        cb.set_label(dp['cbar_label'], fontsize=9)
        cb.ax.tick_params(labelsize=7)
        # Flip so the colorbar reads top-to-bottom naturally on the left side
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')

        fig.suptitle(f'All stations — median {dp["cbar_label"]} (2015–2026)',
                     fontsize=11, fontweight='bold', y=1.002)

        fig_path = os.path.join(OUT_DIR, f'sws_mesh_combined_{param}{FILE_SUFFIX}.pdf')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved {fig_path}')
        plt.close(fig)

    # ── All-stations combined single map ─────────────────────────────────────────
    # Pool every measurement from every station into one dataset, compute the
    # spatial median per bin, and plot phi and dt side by side at each depth.

    print('\nBuilding all-stations combined map...')

    all_df = pd.concat(dfs.values(), ignore_index=True)
    all_x   = all_df['x'].values
    all_y   = all_df['y'].values
    all_z   = all_df['z'].values
    all_phi = all_df['phi_az'].values
    all_dt  = all_df['dt'].values
    print(f'  Total observations: {len(all_df):,}')

    n_rows = len(DEPTH_SLICES)
    n_cols = len(PARAMETERS)

    # Layout: phi panel | phi cbar | dt panel | dt cbar
    width_ratios = [1, 0.05, 1, 0.05]
    fig = plt.figure(figsize=(n_cols * 4.5 + 0.8, n_rows * 4.0 + 0.5))
    gs  = GridSpec(n_rows, 4,
                   width_ratios=width_ratios,
                   hspace=0.08, wspace=0.06)

    # Column indices: phi data=0, phi cbar=1, dt data=2, dt cbar=3
    DATA_COLS = {'phi': 0, 'dt': 2}
    CBAR_COLS = {'phi': 1, 'dt': 3}

    handles = {}   # param -> last contourf handle for colorbar

    for row_idx, z0 in enumerate(DEPTH_SLICES):
        print(f'  z = {z0} km...', end=' ', flush=True)

        X, Y, PHI_med, DT_med, COUNT = xyzd2mesh_2d(
            all_x, all_y, all_z, all_phi, all_dt,
            X_START, X_END, Y_START, Y_END,
            z0=z0, z_hw=Z_HALF_WIDTH,
            step=STEP, dis_lim=DIS_LIM, num_lim=NUM_LIM,
        )
        print(f'{int((COUNT >= COUNT_MIN).sum())} filled bins')

        for col_idx, param in enumerate(PARAMETERS):
            ax = fig.add_subplot(gs[row_idx, DATA_COLS[param]])
            _add_bathy_gray(ax)

            dp   = DIC_PLOT[param]
            cmap = plt.colormaps[dp['cmap']]
            vmin, vmax = dp['vmin'], dp['vmax']
            levels = np.linspace(vmin, vmax, 15)

            D_raw    = PHI_med if param == 'phi' else DT_med
            D_smooth = gaussian_filter(D_raw, sigma=GAUSSIAN_SIGMA)
            D_masked = ma.array(D_smooth, mask=COUNT < COUNT_MIN)

            h = ax.contourf(X, Y, D_masked, levels=levels,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            extend='neither', zorder=2, alpha=0.85)
            ax.contour(X, Y, COUNT, levels=[COUNT_MIN],
                       colors='k', linewidths=0.4, linestyles='--', zorder=3)

            _add_stations(ax)

            ax.set_xlim(X_START, X_END)
            ax.set_ylim(Y_START, Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=7)

            if row_idx == 0:
                ax.set_title(dp['cbar_label'], fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'z = {z0} km\nN (km)', fontsize=9)
            else:
                ax.set_yticklabels([])
            if row_idx < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('E (km)', fontsize=9)

            handles[param] = h

    # Colorbars — phi in middle (col 1), dt on right (col 3)
    for param in PARAMETERS:
        dp         = DIC_PLOT[param]
        vmin, vmax = dp['vmin'], dp['vmax']
        cax   = fig.add_subplot(gs[:, CBAR_COLS[param]])
        ticks = np.arange(vmin, vmax + dp['cbar_step'] * 0.5, dp['cbar_step'])
        cb    = plt.colorbar(handles[param], cax=cax, ticks=ticks,
                             format=FormatStrFormatter(dp['fmt']))
        cb.set_label(dp['cbar_label'], fontsize=10)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle('All stations combined — median φ and dt (2015–2026)',
                 fontsize=13, fontweight='bold', y=1.002)

    fig_path = os.path.join(OUT_DIR, f'sws_mesh_all_stations{FILE_SUFFIX}.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved {fig_path}')
    plt.close(fig)

    # ── Temporal evolution figures ────────────────────────────────────────────────
    # For each station (+ all-stations combined):
    #   Figure A — rows=time periods, cols=depth slices, color=phi
    #   Figure B — rows=time periods, cols=depth slices, color=dt
    #
    # Time periods:
    #   Pre-eruption  : Jan 2015 – Apr 24 2015
    #   Syn-eruption  : Apr 24 – May 19 2015
    #   Post-eruption : May 19 – Dec 31 2015
    #   Annual        : 2016 … 2026

    ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
    ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

    N_DEPTH = len(DEPTH_SLICES)

    # Temporal colormap overrides: grayscale bathy + Blues for dt
    # dt_norm vmax: 99th percentile of dt/ray_L across all stations
    _dt_norm_vmax = float(np.percentile(
        np.concatenate([df['dt_norm'].values for df in dfs.values()]), 99
    ))
    _dt_norm_step = round(_dt_norm_vmax / 6, 4)

    DIC_PLOT_TEMPORAL = {
        'phi':     dict(vmin=0, vmax=180,          cmap='hsv',      cbar_label='φ [° from N]',  cbar_step=30,           fmt='%.0f'),
        'dt':      dict(vmin=0, vmax=0.15,         cmap='Blues',    cbar_label='dt [s]',         cbar_step=0.025,        fmt='%.3f'),
        'dt_norm': dict(vmin=0, vmax=_dt_norm_vmax, cmap='Purples', cbar_label='dt/L [s km⁻¹]', cbar_step=_dt_norm_step, fmt='%.4f'),
    }


    def _subset(df_dict_or_df, t_start, t_end):
        """Return rows within [t_start, t_end)."""
        if isinstance(df_dict_or_df, dict):
            frames = []
            for df in df_dict_or_df.values():
                m = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
                if t_end is not None:
                    m = m & (df['t'] < t_end)
                frames.append(df[m])
            return pd.concat(frames, ignore_index=True)
        else:
            df = df_dict_or_df
            m  = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
            if t_end is not None:
                m = m & (df['t'] < t_end)
            return df[m]


    def _build_time_periods(df_dict_or_df):
        """7 periods: pre-eruption, syn-eruption, 5 equal-count post-eruption bins."""
        # Collect all post-eruption event times
        post = _subset(df_dict_or_df, ERUPTION_END, None)
        post_times = post['t'].sort_values().reset_index(drop=True)
        n = len(post_times)

        # Quintile boundaries by observation count
        boundaries = [ERUPTION_END]
        for i in range(1, 5):
            idx = int(round(i * n / 5))
            idx = min(idx, n - 1)
            boundaries.append(post_times.iloc[idx])
        boundaries.append(None)   # open end

        def _fmt(ts):
            return ts.strftime('%b %Y') if ts is not None else 'present'

        periods = [
            ('Pre-eruption', None, ERUPTION_START),
            ('Syn-eruption', ERUPTION_START, ERUPTION_END),
        ]
        for i in range(5):
            t0, t1 = boundaries[i], boundaries[i + 1]
            label = f'{_fmt(t0)}\n– {_fmt(t1)}'
            periods.append((label, t0, t1))

        return periods


    def _temporal_figure(param, df_dict_or_df, title, out_path,
                         highlight=None, dt_vmax=None, dt_step=None,
                         normalize_by_density=False):
        """Temporal figure: rows = depth slices, cols = 7 time periods.

        Bathymetry is grayscale. dt uses Blues colormap (darker = greater dt).
        Column headers show period label + N = unique earthquake count.
        Post-eruption periods are split into 5 equal-observation bins.
        dt_vmax / dt_step override the default dt colorbar range when provided.
        """
        dp   = dict(DIC_PLOT_TEMPORAL[param])   # copy so we can mutate safely
        if param == 'dt' and dt_vmax is not None and not normalize_by_density:
            dp['vmax']      = dt_vmax
            dp['cbar_step'] = dt_step if dt_step is not None else round(dt_vmax / 6, 3)
        cmap = plt.colormaps[dp['cmap']]

        # Build data-driven time periods
        time_periods = _build_time_periods(df_dict_or_df)
        n_periods    = len(time_periods)   # always 7

        # Pre-compute subsets and unique-event counts
        subsets = []
        n_events = []
        for _, t_start, t_end in time_periods:
            sub = _subset(df_dict_or_df, t_start, t_end)
            subsets.append(sub)
            n_events.append(sub['event_datetime'].nunique())

        # For normalised dt: pre-pass to find the global COUNT max and the
        # actual max normalised value — both needed so every period is normalised
        # by the same denominator and the colorscale matches the data range
        global_count_max = 1.0
        if normalize_by_density and param == 'dt':
            _all_counts = []
            _norm_maxes = []
            for sub in subsets:
                if len(sub) < COUNT_MIN:
                    continue
                for z0 in DEPTH_SLICES:
                    _, _, _, DT_m, C_m = xyzd2mesh_2d(
                        sub['x'].values, sub['y'].values, sub['z'].values,
                        sub['phi_az'].values, sub['dt_norm'].values if param == 'dt_norm' else sub['dt'].values,
                        X_START, X_END, Y_START, Y_END,
                        z0=z0, z_hw=Z_HALF_WIDTH,
                        step=STEP, dis_lim=DIS_LIM, num_lim=NUM_LIM,
                    )
                    _all_counts.append(float(C_m.max()))

            global_count_max = float(max(_all_counts)) if _all_counts else 1.0

            # Now compute normalised values using the global max
            for sub in subsets:
                if len(sub) < COUNT_MIN:
                    continue
                for z0 in DEPTH_SLICES:
                    _, _, _, DT_m, C_m = xyzd2mesh_2d(
                        sub['x'].values, sub['y'].values, sub['z'].values,
                        sub['phi_az'].values, sub['dt_norm'].values if param == 'dt_norm' else sub['dt'].values,
                        X_START, X_END, Y_START, Y_END,
                        z0=z0, z_hw=Z_HALF_WIDTH,
                        step=STEP, dis_lim=DIS_LIM, num_lim=NUM_LIM,
                    )
                    norm_vals = DT_m * (C_m / global_count_max)
                    filled = norm_vals[C_m >= COUNT_MIN]
                    if len(filled):
                        _norm_maxes.append(float(np.percentile(filled, 99)))

            _norm_vmax = float(max(_norm_maxes)) if _norm_maxes else dp['vmax']
            dp['vmax']      = _norm_vmax
            dp['cbar_step'] = round(_norm_vmax / 6, 4)
            dp['cbar_label'] = 'density-weighted dt [s]'
            dp['fmt']        = '%.3f'

        vmin, vmax = dp['vmin'], dp['vmax']
        levels = np.linspace(vmin, vmax, 15)

        panel_w = 2.2
        panel_h = 3.0
        fig = plt.figure(figsize=(n_periods * panel_w + 0.6, N_DEPTH * panel_h + 0.3))
        gs  = GridSpec(N_DEPTH, n_periods + 1,
                       width_ratios=[1] * n_periods + [0.04],
                       hspace=0.04, wspace=0.04)

        last_h = None

        for row_idx, z0 in enumerate(DEPTH_SLICES):
            for col_idx, ((label, _, __), sub, n_eq) in enumerate(
                    zip(time_periods, subsets, n_events)):

                ax = fig.add_subplot(gs[row_idx, col_idx])
                _add_bathy_gray(ax)

                if len(sub) >= COUNT_MIN:
                    X, Y, PHI_med, DT_med, COUNT = xyzd2mesh_2d(
                        sub['x'].values, sub['y'].values, sub['z'].values,
                        sub['phi_az'].values, sub['dt_norm'].values if param == 'dt_norm' else sub['dt'].values,
                        X_START, X_END, Y_START, Y_END,
                        z0=z0, z_hw=Z_HALF_WIDTH,
                        step=STEP, dis_lim=DIS_LIM, num_lim=NUM_LIM,
                    )
                    if param == 'phi':
                        D_raw = PHI_med
                    elif normalize_by_density:
                        # Weight dt by count relative to global max — all periods
                        # share the same denominator so sparse periods are correctly
                        # downweighted relative to dense ones
                        D_raw = DT_med * (COUNT / global_count_max)
                    else:
                        D_raw = DT_med
                    D_smooth = gaussian_filter(D_raw, sigma=GAUSSIAN_SIGMA)
                    D_masked = ma.array(D_smooth, mask=COUNT < COUNT_MIN)

                    last_h = ax.contourf(X, Y, D_masked, levels=levels,
                                         cmap=cmap, vmin=vmin, vmax=vmax,
                                         extend='neither', zorder=2, alpha=0.85)
                    ax.contour(X, Y, COUNT, levels=[COUNT_MIN],
                               colors='k', linewidths=0.3, linestyles='--', zorder=3)
                else:
                    ax.text(0.5, 0.5, 'insufficient\ndata',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=6, color='gray')

                _add_stations(ax, highlight=highlight)
                ax.set_xlim(X_START, X_END)
                ax.set_ylim(Y_START, Y_END)
                ax.set_aspect('equal', 'box')
                ax.tick_params(labelsize=4)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                # Column header: period label + event count (first row only)
                if row_idx == 0:
                    ax.set_title(f'{label}\nN={n_eq:,}', fontsize=7.5,
                                 fontweight='bold', pad=1)
                # Depth label on leftmost column
                if col_idx == 0:
                    ax.set_ylabel(f'z = {z0} km', fontsize=8)

        # Colorbar on far right
        if last_h is not None:
            cax   = fig.add_subplot(gs[:, n_periods])
            ticks = np.arange(vmin, vmax + dp['cbar_step'] * 0.5, dp['cbar_step'])
            cb    = plt.colorbar(last_h, cax=cax, ticks=ticks,
                                 format=FormatStrFormatter(dp['fmt']))
            cb.set_label(dp['cbar_label'], fontsize=9)
            cb.ax.tick_params(labelsize=7)

        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.97)
        if out_path is not None:
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {out_path}')
        else:
            return fig


    print('\nGenerating temporal figures...')

    for sta in STATIONS:
        print(f'\n{sta}:')
        for param in PARAMETERS:
            dp    = DIC_PLOT_TEMPORAL[param]
            fname = os.path.join(OUT_DIR, f'sws_temporal_{sta}_{param}{FILE_SUFFIX}.pdf')
            _temporal_figure(
                param,
                dfs[sta],
                f'{sta} — median {dp["cbar_label"]} by time period',
                fname,
                highlight=sta,
                dt_vmax=STA_DT_VMAX[sta][0],
                dt_step=STA_DT_VMAX[sta][1],
            )

    print('\nAll stations combined temporal figures:')
    for param in PARAMETERS:
        dp    = DIC_PLOT_TEMPORAL[param]
        fname = os.path.join(OUT_DIR, f'sws_temporal_ALL_{param}{FILE_SUFFIX}.pdf')
        _temporal_figure(
            param,
            dfs,
            f'All stations — median {dp["cbar_label"]} by time period',
            fname,
            dt_vmax=_dt_all_max,
            dt_step=_dt_all_step,
        )

    # ── Combined multi-page PDFs for temporal figures ────────────────────────────
    # One PDF per parameter with every station + all-stations on separate pages.

    from matplotlib.backends.backend_pdf import PdfPages

    print('\nBuilding combined temporal PDFs...')

    for param in PARAMETERS:
        dp        = DIC_PLOT_TEMPORAL[param]
        combo_path = os.path.join(OUT_DIR, f'sws_temporal_combined_{param}{FILE_SUFFIX}.pdf')

        with PdfPages(combo_path) as pdf:
            # Per-station pages
            for sta in STATIONS:
                fig = _temporal_figure(
                    param, dfs[sta],
                    f'{sta} — median {dp["cbar_label"]} by time period',
                    out_path=None,          # don't save individually this time
                    highlight=sta,
                    dt_vmax=STA_DT_VMAX[sta][0],
                    dt_step=STA_DT_VMAX[sta][1],
                )
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f'  Added {sta}')

            # All-stations page
            fig = _temporal_figure(
                param, dfs,
                f'All stations — median {dp["cbar_label"]} by time period',
                out_path=None,
                dt_vmax=_dt_all_max,
                dt_step=_dt_all_step,
            )
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'  Added ALL stations')

        print(f'Saved {combo_path}')

    # ── Density-normalised dt temporal PDF ───────────────────────────────────────
    # For each station (and all-stations combined): dt × (COUNT / COUNT_max)
    # This weights each bin by how many observations contribute, so sparse
    # bins are downweighted and only high-dt AND high-density areas appear bright.

    print('\nBuilding density-normalised dt temporal PDFs...')

    for param in ['dt']:   # only dt; phi normalisation is not meaningful here
        dp     = DIC_PLOT_TEMPORAL[param]
        norm_label = dp['cbar_label'].replace('dt [s]', 'density-weighted dt [s]')

        norm_path = os.path.join(OUT_DIR, f'sws_temporal_combined_dt_normalised{FILE_SUFFIX}.pdf')
        with PdfPages(norm_path) as pdf:
            for sta in STATIONS:
                vmax_sta, step_sta = STA_DT_VMAX[sta]
                fig = _temporal_figure(
                    param, dfs[sta],
                    f'{sta} — density-normalised dt by time period',
                    out_path=None,
                    highlight=sta,
                    dt_vmax=vmax_sta,
                    dt_step=step_sta,
                    normalize_by_density=True,
                )
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f'  Added {sta} (normalised)')

            fig = _temporal_figure(
                param, dfs,
                'All stations — density-normalised dt by time period',
                out_path=None,
                dt_vmax=_dt_all_max,
                dt_step=_dt_all_step,
                normalize_by_density=True,
            )
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'  Added ALL stations (normalised)')

        print(f'Saved {norm_path}')

# ── Two-pass execution: all data then filtered ────────────────────────────────
# dfs already loaded above with DT_ERROR_MAX=None (all data), FILE_SUFFIX=''

# Pass 1: all data
print('\n=== Pass 1: all data ===')
_run_outputs()

# Pass 2: filtered (dt_error < 0.1 s)
DT_ERROR_MAX = 0.1
FILE_SUFFIX  = '_filtered'
print(f'\n=== Pass 2: filtered (dt_error < {DT_ERROR_MAX} s) ===')
dfs = _load_all_dfs()
_run_outputs()

# ── Ray-path-normalised dt temporal PDF (dt/L shows true anisotropy) ─────────
# dt_norm = dt / ray_path_length  (s/km)
# High dt_norm = genuine high anisotropy; low dt_norm + high dt = long-path effect

print('\nBuilding ray-path-normalised dt temporal PDFs (dt/L)...')

for param in ['dt_norm']:
    dp    = DIC_PLOT_TEMPORAL[param]
    combo = os.path.join(OUT_DIR, f'sws_temporal_combined_dt_norm{FILE_SUFFIX}.pdf')
    with PdfPages(combo) as pdf:
        for sta in STATIONS:
            vmax_sta = float(np.percentile(dfs[sta]['dt_norm'].values, 99))
            step_sta = round(vmax_sta / 6, 4)
            fig = _temporal_figure(
                param, dfs[sta],
                f'{sta} — median dt/L (s km⁻¹) by time period',
                out_path=None,
                highlight=sta,
                dt_vmax=vmax_sta,
                dt_step=step_sta,
            )
            if fig is not None:
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f'  Added {sta} (dt/L vmax={vmax_sta:.4f} s/km)')

        # All-stations combined
        all_dtnorm = np.concatenate([df['dt_norm'].values for df in dfs.values()])
        vmax_all   = float(np.percentile(all_dtnorm, 99))
        step_all   = round(vmax_all / 6, 4)
        fig = _temporal_figure(
            param, dfs,
            'All stations — median dt/L (s km⁻¹) by time period',
            out_path=None,
            dt_vmax=vmax_all,
            dt_step=step_all,
        )
        if fig is not None:
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'  Added ALL stations (dt/L vmax={vmax_all:.4f} s/km)')

    print(f'Saved {combo}')

print('\nDone.')
