"""
Delay time (dt) mapped spatially by EARTHQUAKE SOURCE location (not by ray-
covered voxel like the travel-time anisotropy tomography). For each station,
bin its own events into the 0.25 km voxel grid by the event's own (x, y,
depth) hypocenter, and take the plain mean dt of events whose source falls in
each voxel (no ray tracing, no travel-time weighting - this is a direct
source-side statistic).

AXEC2 uses the new mfast try_filters results (all batches completed so far);
the other 5 stations (AXAS1, AXAS2, AXCC1, AXEC1, AXEC3) use their existing
old results (2015-2021 enriched files, which already carry event_lat/
event_lon/event_depth).

One page per station (6 pages total), each page a row of 3 depth-range panels
(0.0-0.75, 0.75-1.5, 1.5-2.25 km), sharing one color scale across all pages.
Bathymetry/fault-trace styling reused from the tomography scripts. All 6
station triangles are drawn on every page; only the page's own station is
highlighted gold, the rest are plain white/black outline markers.

QC gate: success==True, 0<dt<=0.30s, dt_error<0.04s, phi_error<20deg (no Q_w
cutoff). Voxels require >=5 contributing events to be shown.

Produces: dt_by_source_voxel_per_station.pdf

Run with:
    python3 plot_dt_by_source_voxel_per_station.py
"""

import glob
import math
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
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import matplotlib.lines as mlines

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
AXEC2_MFAST_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_lqt_pykonal_results')
AXEC2_MFAST_META = os.path.join(
    HERE, 'raw_axec2_all_batches_mfast_filters_data', 'raw_axec2_all_batches_mfast_filters_metadata.csv')

STATION_FILE = '../data/stations_axial.llz'
STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}

OLD_STATION_FILES = {
    'AXAS1': 'splitting_results_AXAS1_2015_2021_all_batches_enriched.csv',
    'AXAS2': 'splitting_results_AXAS2_2015_2021_all_batches_enriched.csv',
    'AXCC1': 'splitting_results_AXCC1_2015_2021_all_batches_enriched.csv',
    'AXEC1': 'splitting_results_AXEC1_2015_2021_all_batches_enriched.csv',
    'AXEC2': 'splitting_results_AXEC2_2015_2021_all_batches_enriched.csv',
    'AXEC3': 'splitting_results_AXEC3_2015_2021_all_batches_enriched.csv',
}

# Pages to render: (page_key, station_for_markers, dataset_label)
PAGE_LIST = [
    ('AXAS1', 'AXAS1', 'old'),
    ('AXAS2', 'AXAS2', 'old'),
    ('AXCC1', 'AXCC1', 'old'),
    ('AXEC1', 'AXEC1', 'old'),
    ('AXEC2_old', 'AXEC2', 'old'),
    ('AXEC2_new', 'AXEC2', 'mfast'),
    ('AXEC3', 'AXEC3', 'old'),
]

PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04
MAX_DT = 0.30
COUNT_MIN = 5

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VXY = 0.25
VZ = 0.25
xn = np.arange(X_START, X_END + VXY * .5, VXY)
yn = np.arange(Y_START, Y_END + VXY * .5, VXY)
zn = np.arange(0., Z_MAX + VZ * .5, VZ)
NX, NY, NZ = len(xn), len(yn), len(zn)

Xg, Yg = np.meshgrid(xn + VXY / 2., yn + VXY / 2., indexing='ij')

DEPTH_ROWS = [
    ((0, 3), '0.0-0.75 km', 0.375),
    ((3, 6), '0.75-1.5 km', 1.125),
    ((6, 9), '1.5-2.25 km', 1.875),
]

# ── Load per-station events (source lat/lon/depth + dt), QC-filtered ─────────


def _qc(df):
    df = df[df['success'] == True].copy()
    df = df[(df['dt'] > 0) & (df['dt'] <= MAX_DT)]
    df = df.dropna(subset=['phi_error', 'dt_error'])
    df = df[(df['dt_error'] < DT_ERR_MAX) & (df['phi_error'] < PHI_ERR_MAX)]
    return df


def load_old_station(sta):
    df = pd.read_csv(os.path.join(DATA_DIR, OLD_STATION_FILES[sta]))
    df = _qc(df)
    df = df.dropna(subset=['event_lat', 'event_lon', 'event_depth'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    return df[['dt', 't', 'event_lat', 'event_lon', 'event_depth']].rename(
        columns={'event_lat': 'lat', 'event_lon': 'lon', 'event_depth': 'depth'})


def load_axec2_mfast():
    batch_files = glob.glob(os.path.join(
        AXEC2_MFAST_DIR, 'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_*.csv'))
    dfs = [pd.read_csv(f) for f in batch_files]
    dfs = [d for d in dfs if len(d) > 0]
    df = pd.concat(dfs, ignore_index=True)
    df = _qc(df)
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    meta = pd.read_csv(AXEC2_MFAST_META)[['event_id', 'latitude', 'longitude', 'depth']]
    df = df.merge(meta, on='event_id', how='left').dropna(subset=['latitude', 'longitude', 'depth'])
    return df[['dt', 't', 'latitude', 'longitude', 'depth']].rename(
        columns={'latitude': 'lat', 'longitude': 'lon'})


def assign_period(t_series):
    """0 = pre-eruption, 1 = syn-eruption, 2 = post-eruption (to latest in that df)."""
    idx = np.full(len(t_series), 2, dtype=np.int64)
    idx[t_series.values < np.datetime64(ERUPTION_START)] = 0
    m_syn = (t_series.values >= np.datetime64(ERUPTION_START)) & (t_series.values < np.datetime64(ERUPTION_END))
    idx[m_syn] = 1
    return idx


print('Loading per-page events...')
station_events = {}
for page_key, sta, dataset in PAGE_LIST:
    if dataset == 'mfast':
        df = load_axec2_mfast()
        print(f'  {page_key} ({sta}, mfast try_filters): {len(df):,} events')
    else:
        df = load_old_station(sta)
        print(f'  {page_key} ({sta}, old): {len(df):,} events')
    x, y = ll2xy(df['lat'].values, df['lon'].values)
    df['x'] = x
    df['y'] = y
    station_events[page_key] = df

# ── Station coordinates + bathymetry (shared map context) ───────────────────

_sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                      engine='python').set_index('s')
_sta_df = _sta_df.loc[[s for s in STATIONS if s in _sta_df.index]]
_sta_df['x'], _sta_df['y'] = ll2xy(_sta_df['lat'].values, _sta_df['lon'].values)

BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
          'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS = None
_p = PILImage.open(BATHY)
_tag = _p.tag_v2
_olon, _olat = _tag[33922][3], _tag[33922][4]
_pl, _pb = _tag[33550][0], _tag[33550][1]
_nc, _nr = _p.size
_p.close()
_c0 = max(0, int(((INI_LON + X_START / KM_PER_DEG_LON) - _olon) / _pl) - 2)
_c1 = min(_nc, int(((INI_LON + X_END / KM_PER_DEG_LON) - _olon) / _pl) + 2)
_r0 = max(0, int((_olat - (INI_LAT + Y_END / KM_PER_DEG_LAT)) / _pb) - 2)
_r1 = min(_nr, int((_olat - (INI_LAT + Y_START / KM_PER_DEG_LAT)) / _pb) + 2)
_rgb = tifffile.imread(BATHY)[_r0:_r1, _c0:_c1]
_ds = max(1, max(_rgb.shape[:2]) // 1024)
_rgb = _rgb[::_ds, ::_ds]
_gray = np.dot(_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
_ext = [(_olon + _c0 * _pl - INI_LON) * KM_PER_DEG_LON, (_olon + _c1 * _pl - INI_LON) * KM_PER_DEG_LON,
        (_olat - _r1 * _pb - INI_LAT) * KM_PER_DEG_LAT, (_olat - _r0 * _pb - INI_LAT) * KM_PER_DEG_LAT]


def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)


def _sta_markers(ax, active_sta):
    for sta, row in _sta_df.iterrows():
        if sta == active_sta:
            ax.plot(row['x'], row['y'], '^', ms=7, mfc='#FFD700', mec='k', mew=0.8, zorder=13)
        else:
            ax.plot(row['x'], row['y'], '^', ms=6, mfc='white', mec='k', mew=0.6, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


COT66 = math.cos(math.radians(66)) / math.sin(math.radians(66))
DEPTH_TOP = 0.1
strike_rad = math.radians(330.)
L = 8.
s_hat = np.array([math.sin(strike_rad), math.cos(strike_rad)])


def _faults(ax, z0):
    for cx, cy, dip_az in [(8.5, 4.0, 60.), (7.0, 4.0, 240.)]:
        horiz = (z0 - DEPTH_TOP) * COT66
        da = math.radians(dip_az)
        cx_z = cx + horiz * math.sin(da)
        cy_z = cy + horiz * math.cos(da)
        x1, y1 = cx_z - L / 2 * s_hat[0], cy_z - L / 2 * s_hat[1]
        x2, y2 = cx_z + L / 2 * s_hat[0], cy_z + L / 2 * s_hat[1]
        ax.plot([x1, x2], [y1, y2], '-', color='#cc0000', lw=0.9, alpha=0.75, zorder=11)


# ── Bin each station's events into the voxel grid by SOURCE location ────────

PERIOD_LABELS = ['Pre-eruption', 'Syn-eruption', 'Post-eruption']


def voxelize(df):
    ix = np.floor((df['x'].values - X_START) / VXY).astype(int)
    iy = np.floor((df['y'].values - Y_START) / VXY).astype(int)
    iz = np.floor(df['depth'].values / VZ).astype(int)
    valid = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < NZ)
    ix, iy, iz, dt = ix[valid], iy[valid], iz[valid], df['dt'].values[valid]
    flat = ix * (NY * NZ) + iy * NZ + iz

    sum_dt = np.zeros(NX * NY * NZ)
    cnt = np.zeros(NX * NY * NZ)
    np.add.at(sum_dt, flat, dt)
    np.add.at(cnt, flat, 1)
    mean_dt = np.divide(sum_dt, cnt, out=np.full_like(sum_dt, np.nan), where=cnt > 0)
    gate = cnt >= COUNT_MIN
    a3d = np.where(gate, mean_dt, np.nan).reshape(NX, NY, NZ)
    return a3d, int(valid.sum())


def voxelize_by_period(df):
    """Returns a list of dict(label, n, a) - one per pre/syn/post-eruption period."""
    period_idx = assign_period(df['t'])
    results = []
    for pi, label in enumerate(PERIOD_LABELS):
        sub = df[period_idx == pi]
        a3d, n = voxelize(sub)
        results.append(dict(label=label, n=n, a=a3d))
    return results


station_grids = {}
for page_key, sta, dataset in PAGE_LIST:
    results = voxelize_by_period(station_events[page_key])
    station_grids[page_key] = results
    for r in results:
        print(f'  {page_key} [{r["label"]}]: {r["n"]:,} events voxelized, '
              f'covered voxels={np.isfinite(r["a"]).sum():,}')

# ── Shared color scale across all pages ──────────────────────────────────────

all_vals = []
for page_key, sta, dataset in PAGE_LIST:
    for r in station_grids[page_key]:
        v = r['a'][np.isfinite(r['a'])]
        if v.size:
            all_vals.append(v)
vmax = float(np.nanpercentile(np.concatenate(all_vals), 99)) if all_vals else 0.1
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 0.1
print(f'\nShared color scale vmax (99th pct, dt in seconds): {vmax:.3f}s')

cmap = plt.colormaps['Blues']
levels = np.linspace(0., vmax, 15)


def make_page(sta, dataset, results):
    n_dep = len(DEPTH_ROWS)
    n_per = len(results)
    fig = plt.figure(figsize=(3.0 * n_per + 0.6, 3.2 * n_dep + 0.5))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.05], hspace=0.06, wspace=0.06)
    last_h = None
    for ri, ((iz0, iz1), zlbl, z0) in enumerate(DEPTH_ROWS):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            _bathy(ax)
            sl = np.nanmean(res['a'][:, :, iz0:iz1], axis=2)
            msk = np.isfinite(sl)
            if msk.sum() >= 3:
                sm = gaussian_filter(np.nan_to_num(sl), sigma=1.0)
                mp = np.ma.masked_where(~msk, sm)
                if not mp.mask.all():
                    ax.contourf(Xg, Yg, mp, levels=levels, cmap=cmap, vmin=0, vmax=vmax, extend='neither')
                    last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
                    last_h.set_array([])
            _sta_markers(ax, sta)
            _faults(ax, z0)
            ax.set_xlim(X_START, X_END)
            ax.set_ylim(Y_START, Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f'{res["label"]}\nN={res["n"]:,}', fontsize=9, fontweight='bold', pad=3)
            if ci == 0:
                ax.set_ylabel(zlbl, fontsize=8)
    if last_h:
        cax = fig.add_subplot(gs[:, n_per])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, vmax, 5), format=FormatStrFormatter('%.3f'))
        cb.set_label('Mean delay time per source voxel (s)', fontsize=8)
        cb.ax.tick_params(labelsize=6)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=7, label=f'{sta} (this page)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='white', mec='k', ms=6, label='Other stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.45, 0.0))
    label = f'{sta} (mfast try_filters)' if dataset == 'mfast' else f'{sta} (old)'
    n_total = sum(r['n'] for r in results)
    fig.suptitle(f'Delay time by earthquake SOURCE voxel - {label}\n'
                 f'N={n_total:,} events, dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}deg, '
                 f'>={COUNT_MIN} events/voxel required, pre/syn/post-eruption periods',
                 fontsize=10, fontweight='bold', y=1.05)
    fig.subplots_adjust(bottom=0.12, top=0.85)
    return fig


OUT_PATH = 'dt_by_source_voxel_per_station.pdf'
with PdfPages(OUT_PATH) as pdf:
    for page_key, sta, dataset in PAGE_LIST:
        results = station_grids[page_key]
        fig = make_page(sta, dataset, results)
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
        plt.close(fig)

print(f'\nSaved {OUT_PATH} ({len(PAGE_LIST)} pages)')
