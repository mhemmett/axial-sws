#!/usr/bin/env python3
"""
sws_gif_dt_equalN_bins.py

For each page (station + all-stations) of the
sws_temporal_combined_dt_annual_kidiwela_source_filtered.pdf layout,
divide the data into N_BINS=100 bins of equal earthquake count (sorted by
time), then produce one animated GIF per depth-slice row.

Output:  results/gifs/sws_gif_dt_equalN_{STATION}_z{depth}.gif
         (one GIF per station × depth combination, 21 total)

Each frame shows:
  - bathymetry background
  - dt contourf for that bin's data at that depth slice
  - Kidiwela source locations
  - station triangles
  - title: "STATION — dt [s]  |  date range  N=n_eq"

Frame speed: 300 ms/frame (≈3 fps), infinite loop.
"""

import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma
import tifffile
from PIL import Image as PILImage
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
GIF_DIR      = os.path.join(BASE, 'gifs')
os.makedirs(GIF_DIR, exist_ok=True)

# ── Coordinate origin ─────────────────────────────────────────────────────────
INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)

# ── Data files ────────────────────────────────────────────────────────────────
FILES = {
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
STATIONS    = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET_KM = {
    'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55), 'AXCC1': (-0.75, 0.25),
    'AXEC1': ( 0.15,  0.30), 'AXEC2': ( 0.15,  0.05), 'AXEC3': ( 0.30, -0.28),
}

# ── Kidiwela Mogi source locations ────────────────────────────────────────────
KIDIWELA_SOURCES = [
    dict(x=7.57, y=4.55, label='S1'),
    dict(x=7.53, y=6.60, label='S2'),
]

# ── Grid / binning parameters (same as original) ─────────────────────────────
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
STEP           = 0.1
DIS_LIM        = 0.3
NUM_LIM        = 100
COUNT_MIN      = 15
GAUSS_SIGMA    = 0.2 / STEP
DEPTH_SLICES   = [0.5, 1.0, 1.5]
Z_HALF_WIDTH   = 0.3

DT_VMIN = 0.0
DT_VMAX = 0.15
N_BINS  = 100
GIF_MS  = 300   # ms per frame

# ── Bathymetry (load once) ────────────────────────────────────────────────────
BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
         'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS = None
_p = PILImage.open(BATHY); _t = _p.tag_v2
_olon, _olat = _t[33922][3], _t[33922][4]
_pl,   _pb   = _t[33550][0], _t[33550][1]
_nc,   _nr   = _p.size;  _p.close()
_c0 = max(0,   int(((INI_LON + X_START / KM_PER_DEG_LON) - _olon) / _pl) - 2)
_c1 = min(_nc, int(((INI_LON + X_END   / KM_PER_DEG_LON) - _olon) / _pl) + 2)
_r0 = max(0,   int((_olat - (INI_LAT + Y_END   / KM_PER_DEG_LAT)) / _pb) - 2)
_r1 = min(_nr, int((_olat - (INI_LAT + Y_START / KM_PER_DEG_LAT)) / _pb) + 2)
_rgb  = tifffile.imread(BATHY)[_r0:_r1, _c0:_c1]
_ds   = max(1, max(_rgb.shape[:2]) // 1024)
_rgb  = _rgb[::_ds, ::_ds]
_gray = np.dot(_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
_ext  = [(_olon + _c0 * _pl - INI_LON) * KM_PER_DEG_LON,
         (_olon + _c1 * _pl - INI_LON) * KM_PER_DEG_LON,
         (_olat - _r1 * _pb - INI_LAT) * KM_PER_DEG_LAT,
         (_olat - _r0 * _pb - INI_LAT) * KM_PER_DEG_LAT]

def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto',
              cmap='gray', zorder=0)

# ── Station table ─────────────────────────────────────────────────────────────
_sta = pd.read_csv(STATION_FILE, sep=r'\s+',
                   names=['lon', 'lat', 'elev_km', 'station'],
                   engine='python').set_index('station')
_sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'], _sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)

def _add_stations(ax, highlight=None):
    for sta, row in _sta.iterrows():
        mfc = '#FFD700' if (highlight is None or sta == highlight) else 'white'
        ax.plot(row['x'], row['y'], '^', ms=6, mfc=mfc, mec='k', mew=0.7, zorder=12)
        dx, dy = LABEL_OFFSET_KM.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta),
                fontsize=5.5, zorder=13)

def _add_kidiwela(ax):
    for src in KIDIWELA_SOURCES:
        ax.plot(src['x'], src['y'], 'o',
                ms=4, mfc='red', mec='k', mew=0.5, zorder=14)
        ax.text(src['x'] + 0.12, src['y'] + 0.12, src['label'],
                fontsize=5, color='red', fontweight='bold', zorder=15)

# ── Mesh builder (same as original) ──────────────────────────────────────────
def xyzd2mesh(x, y, z, d_dt, z0):
    mz = np.abs(z - z0) <= Z_HALF_WIDTH
    xs, ys, dt_s = x[mz], y[mz], d_dt[mz]
    xn = np.arange(X_START, X_END + STEP * .5, STEP)
    yn = np.arange(Y_START, Y_END + STEP * .5, STEP)
    Xg, Yg = np.meshgrid(xn, yn); ny, nx = Xg.shape
    DT  = np.zeros((ny, nx))
    CNT = np.zeros((ny, nx), dtype=int)
    if len(xs) < 3:
        return Xg, Yg, DT, CNT
    tree = cKDTree(np.column_stack([xs, ys]))
    gp   = np.column_stack([Xg.ravel(), Yg.ravel()])
    nbrs = tree.query_ball_point(gp, DIS_LIM)
    for k, nb in enumerate(nbrs):
        if not nb:
            continue
        pts  = np.column_stack([xs[nb], ys[nb]]); gpi = gp[k]
        dist = np.hypot(pts[:, 0] - gpi[0], pts[:, 1] - gpi[1])
        sel  = np.array(nb)[np.argsort(dist)[:NUM_LIM]]
        CNT.ravel()[k] = len(sel)
        DT.ravel()[k]  = float(np.median(dt_s[sel]))
    return Xg, Yg, DT, CNT

# ── Equal-N binning ───────────────────────────────────────────────────────────
def make_equal_n_bins(df, n_bins):
    """Return list of (t_start, t_end, n_eq, sub_df) for n_bins equal-count bins."""
    events = df['event_datetime'].drop_duplicates().sort_values().reset_index(drop=True)
    n_ev   = len(events)
    if n_ev < n_bins:
        # fewer events than bins — use one bin per event
        splits = [[ev] for ev in events]
    else:
        splits = np.array_split(events.values, n_bins)
    bins = []
    for chunk in splits:
        if len(chunk) == 0:
            continue
        t_start = pd.Timestamp(chunk[0])
        t_end   = pd.Timestamp(chunk[-1])
        mask    = df['event_datetime'].isin(chunk)
        sub     = df[mask]
        n_eq    = sub['event_datetime'].nunique()
        bins.append((t_start, t_end, n_eq, sub))
    return bins

# ── Single-frame renderer ─────────────────────────────────────────────────────
def render_frame(sub, z0, dt_vmax, title, highlight, fig_size=(4.5, 4.5), dpi=100):
    """Render one map frame; return PIL Image (RGB)."""
    levels = np.linspace(DT_VMIN, dt_vmax, 15)
    cmap   = plt.colormaps['Blues']

    fig, ax = plt.subplots(figsize=fig_size)
    _bathy(ax)
    if len(sub) >= COUNT_MIN:
        Xg, Yg, DT, CNT = xyzd2mesh(
            sub['x'].values, sub['y'].values, sub['z'].values,
            sub['dt'].values, z0)
        D = gaussian_filter(DT, sigma=GAUSS_SIGMA)
        D = ma.array(D, mask=CNT < COUNT_MIN)
        cf = ax.contourf(Xg, Yg, D, levels=levels, cmap=cmap,
                         vmin=DT_VMIN, vmax=dt_vmax,
                         extend='neither', zorder=2, alpha=0.85)
        cbar = plt.colorbar(cf, ax=ax, shrink=0.75, pad=0.02)
        step = round(dt_vmax / 6, 3)
        ticks = np.arange(DT_VMIN, dt_vmax + step * .5, step)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        cbar.set_label('dt [s]', fontsize=8)
        cbar.ax.tick_params(labelsize=6)
    else:
        ax.text(0.5, 0.5, '–', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')

    _add_stations(ax, highlight)
    _add_kidiwela(ax)
    ax.set_xlim(X_START, X_END)
    ax.set_ylim(Y_START, Y_END)
    ax.set_aspect('equal', 'box')
    ax.tick_params(labelsize=6)
    ax.set_xlabel('x [km]', fontsize=7)
    ax.set_ylabel('y [km]', fontsize=7)
    ax.set_title(title, fontsize=7.5, fontweight='bold', pad=3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).convert('RGB')

# ── GIF builder ───────────────────────────────────────────────────────────────
def build_gif(frames_pil, out_path):
    if not frames_pil:
        print(f'  [skip] no frames for {out_path}')
        return
    w, h = frames_pil[0].size
    # Build a global palette from all frames combined so colours are
    # consistent across every frame in the GIF.
    composite = PILImage.new('RGB', (w, h * len(frames_pil)))
    for i, f in enumerate(frames_pil):
        composite.paste(f, (0, i * h))
    global_palette = composite.quantize(colors=256, method=PILImage.Quantize.MEDIANCUT)
    frames_gif = [f.quantize(colors=256, palette=global_palette, dither=0)
                  for f in frames_pil]
    frames_gif[0].save(
        out_path,
        format='GIF',
        save_all=True,
        append_images=frames_gif[1:],
        duration=GIF_MS,
        loop=0,
    )
    print(f'  Saved {out_path}  ({len(frames_gif)} frames)')

# ── Main ──────────────────────────────────────────────────────────────────────
DT_ERR_MAX = 0.1

print('Loading data (dt_error < 0.1 s filter)...')

def _load(fname):
    d = pd.read_csv(BASE + fname).loc[:, :'dt_error'].dropna()
    d = d[d['dt'] > 0]
    d = d[d['dt_error'] < DT_ERR_MAX]
    return d

dfs = {}
for sta, (f1, f2) in FILES.items():
    df = pd.concat([_load(f1), _load(f2)], ignore_index=True)
    df['x'], df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
    df['z'] = df['event_depth'].values
    df['t'] = pd.to_datetime(df['event_datetime'], utc=True)
    dfs[sta] = df
    print(f'  {sta}: {len(df):,} rows')

# Build "all stations" combined dataframe
all_df = pd.concat(dfs.values(), ignore_index=True)

# Per-station dt_vmax (99th percentile of that station's dt)
dt_vmax_per_sta = {sta: float(np.percentile(df['dt'].values, 99))
                   for sta, df in dfs.items()}
dt_vmax_all = float(np.percentile(all_df['dt'].values, 99))

# Targets: each station individually, then the combined
targets = [(sta, dfs[sta], dt_vmax_per_sta[sta]) for sta in STATIONS]
targets.append(('ALL', all_df, dt_vmax_all))

for sta_name, df, dt_vmax in targets:
    highlight = sta_name if sta_name in STATIONS else None
    print(f'\n{sta_name}: {df["event_datetime"].nunique():,} unique events')

    bins = make_equal_n_bins(df, N_BINS)
    print(f'  → {len(bins)} bins of ~{bins[0][2] if bins else 0} events each')

    for z0 in DEPTH_SLICES:
        frames = []
        for i, (t_start, t_end, n_eq, sub) in enumerate(bins):
            date_str = (f'{t_start.strftime("%Y-%m-%d")} – '
                        f'{t_end.strftime("%Y-%m-%d")}')
            title = (f'{sta_name}  dt [s]  z={z0} km\n'
                     f'{date_str}   N={n_eq:,}')
            img = render_frame(sub, z0, dt_vmax, title, highlight)
            frames.append(img)
            if (i + 1) % 25 == 0:
                print(f'    z={z0}  frame {i+1}/{len(bins)}')

        out = os.path.join(GIF_DIR,
                           f'sws_gif_dt_equalN_{sta_name}_z{z0}.gif')
        build_gif(frames, out)

print('\nDone — GIFs written to', GIF_DIR)
