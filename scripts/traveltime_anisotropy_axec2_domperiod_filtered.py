"""
Travel-time-weighted percent shear-wave anisotropy tomography (ray-traced, same formula as
lqt_pykonal_tomography_traveltime_anisotropy.py / plot_traveltime_anisotropy_old_vs_new_pre_eruption.py),
for AXEC2 mfast try_filters data ONLY, with the dt<=T_dom/2 cycle-skip-risk cut applied on top
of the standard QC gate (Q_w>=0.5, phi_error<20 deg, dt_error<0.04 s, success, 0<dt<=0.30s) --
same filtered event set as dt_by_source_voxel_axec2_domperiod_filtered.py and
rose_axec2_mfast_dt_vs_half_dom_period_filter.png.

Formula:
    T_travel = sum_k seg_len_km[k] / Vs_voxel[k]   (per ray, k = crossed voxels)
    A_ray    = dt * 100 / T_travel                 (percent)
    V_voxel  = (sum_rays A_ray * tt_k) / (sum_rays tt_k),  tt_k = seg_len_km/Vs_voxel

3 depth ranges (0.0-0.75, 0.75-1.5, 1.5-2.25 km) x pre/syn/post-eruption periods.

Produces: traveltime_anisotropy_axec2_domperiod_filtered.pdf

Run with:
    python3 traveltime_anisotropy_axec2_domperiod_filtered.py
"""

import glob
import math
import time
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

from pykonal_raytracer import BaillardRayTracer
from baillard_velocity import vs_at

STATION_FILE = '../data/stations_axial.llz'
STATION = 'AXEC2'
STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')
PERIOD_LABELS = ['Pre-eruption', 'Syn-eruption', 'Post-eruption']

MAX_DT = 0.30
QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04
N_RAY = 200

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

COUNT_MIN = 5
MIN_PATH_KM = 0.3

# ── Load AXEC2 mfast events, standard QC + dt<=T_dom/2 cut ───────────────────

result_files = glob.glob('production_axec2_mfast_filters_lqt_pykonal_results/'
                          'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_*.csv')
dfs = [pd.read_csv(f) for f in result_files]
dfs = [d for d in dfs if len(d) > 0]
events = pd.concat(dfs, ignore_index=True)
events = events[events['success'] == True].copy()
events = events[(events['dt'] > 0) & (events['dt'] <= MAX_DT)]
events = events.dropna(subset=['quality', 'phi_error', 'dt_error', 'chosen_filter_dom_period_samples'])
events = events[(events['quality'] >= QW_MIN) & (events['phi_error'] < PHI_ERR_MAX) & (events['dt_error'] < DT_ERR_MAX)]

n_before = len(events)
events['dom_period_s'] = events['chosen_filter_dom_period_samples'] / 200.0
events = events[events['dt'] <= events['dom_period_s'] / 2.0].copy()
n_after = len(events)
print(f'Standard QC-passing: {n_before}, after dt<=T_dom/2 cut: {n_after} '
      f'({n_before - n_after} removed, {(n_before-n_after)/n_before*100:.1f}%)')

events['t'] = pd.to_datetime(events['datetime'], utc=True, format='ISO8601')

meta = pd.read_csv('raw_axec2_all_batches_mfast_filters_data/raw_axec2_all_batches_mfast_filters_metadata.csv')
meta_ll = meta[['event_id', 'latitude', 'longitude', 'depth']]
events = events.merge(meta_ll, on='event_id', how='left').dropna(subset=['latitude', 'longitude', 'depth'])
ex, ey = ll2xy(events['latitude'].values, events['longitude'].values)
events['x'] = ex
events['y'] = ey
events['z'] = events['depth'].values
print(f'Events with location: {len(events)}')

# ── Vs per voxel ──────────────────────────────────────────────────────────────

print('Querying Baillard Vs at each voxel centre...')
X3, Y3, Z3 = np.meshgrid(xn + VXY / 2., yn + VXY / 2., zn + VZ / 2., indexing='ij')
vs_voxels = vs_at(X3.ravel(), Y3.ravel(), Z3.ravel()).reshape(NX, NY, NZ)
vs_voxels = np.clip(vs_voxels, 0.3, 5.0)
vs_flat = vs_voxels.ravel()

# ── Station table + bathymetry ────────────────────────────────────────────────

_sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                      engine='python').set_index('s')
_sta_df = _sta_df.loc[[s for s in STATIONS if s in _sta_df.index]]
_sta_df['x'], _sta_df['y'] = ll2xy(_sta_df['lat'].values, _sta_df['lon'].values)
sta_x, sta_y = float(_sta_df.loc[STATION, 'x']), float(_sta_df.loc[STATION, 'y'])

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


def _sta_markers(ax):
    for sta, row in _sta_df.iterrows():
        ax.plot(row['x'], row['y'], '^', ms=6, mfc='#FFD700', mec='k', mew=0.7, zorder=12)
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


# ── Ray tracing: once per unique event ────────────────────────────────────────

print(f'\nTracing {len(events):,} events to AXEC2 (n_pts={N_RAY})...')
tracer = BaillardRayTracer()
tracer.precompute_station(STATION, sta_x, sta_y)

t0 = time.time()
event_T_travel = {}
event_voxel_hits = {}

for i, row in enumerate(events.itertuples()):
    ex_, ey_, ez_ = float(row.x), float(row.y), float(row.z)
    if ez_ < 0 or ez_ > Z_MAX:
        continue
    try:
        ray = tracer.trace(STATION, ex_, ey_, ez_, n_pts=N_RAY)
        vcols, seg_len_km = tracer.ray_to_voxels(ray, xn, yn, zn)
        if len(vcols) == 0:
            continue
        vs_seg = vs_flat[vcols]
        tt_seg = seg_len_km / np.maximum(vs_seg, 1e-9)
        T_travel = float(tt_seg.sum())
        if T_travel <= 0:
            continue
        event_T_travel[row.event_id] = T_travel
        ix = vcols // (NY * NZ)
        rem = vcols % (NY * NZ)
        iy = rem // NZ
        iz = rem % NZ
        event_voxel_hits[row.event_id] = (ix, iy, iz, tt_seg)
    except RuntimeError:
        continue
    if (i + 1) % 2000 == 0:
        print(f'  {i+1:,}/{len(events):,}  {time.time()-t0:.0f}s', end='\r', flush=True)

print(f'\nTraced {len(event_T_travel):,}/{len(events):,} events successfully in {time.time()-t0:.0f}s')


def assign_period(t_series):
    idx = np.full(len(t_series), 2, dtype=np.int64)
    idx[t_series.values < np.datetime64(ERUPTION_START)] = 0
    m_syn = (t_series.values >= np.datetime64(ERUPTION_START)) & (t_series.values < np.datetime64(ERUPTION_END))
    idx[m_syn] = 1
    return idx


events_traced = events[events['event_id'].isin(event_T_travel.keys())].copy()
events_traced['period'] = assign_period(events_traced['t'])

results = []
for pi, label in enumerate(PERIOD_LABELS):
    sub = events_traced[events_traced['period'] == pi]
    sum_wa = np.zeros(NX * NY * NZ)
    sum_w = np.zeros(NX * NY * NZ)
    sum_p = np.zeros(NX * NY * NZ)
    cnt = np.zeros(NX * NY * NZ)
    for r in sub.itertuples():
        T_travel = event_T_travel[r.event_id]
        a_ray = r.dt * 100.0 / T_travel
        ix, iy, iz, tt_seg = event_voxel_hits[r.event_id]
        flat = ix * (NY * NZ) + iy * NZ + iz
        np.add.at(sum_wa, flat, a_ray * tt_seg)
        np.add.at(sum_w, flat, tt_seg)
        np.add.at(sum_p, flat, tt_seg * vs_flat[flat])
        np.add.at(cnt, flat, 1)
    wmean = np.divide(sum_wa, sum_w, out=np.full_like(sum_wa, np.nan), where=sum_w > 0)
    gate = (cnt >= COUNT_MIN) & (sum_p >= MIN_PATH_KM)
    a3d = np.where(gate, wmean, np.nan).reshape(NX, NY, NZ)
    results.append(dict(label=label, n=len(sub), a=a3d))
    print(f'  {label}: N={len(sub):,}, covered voxels={np.isfinite(a3d).sum():,}')

all_vals = [r['a'][np.isfinite(r['a'])] for r in results if np.isfinite(r['a']).any()]
a_vmax = float(np.nanpercentile(np.concatenate(all_vals), 99)) if all_vals else 30.
if not np.isfinite(a_vmax) or a_vmax <= 0:
    a_vmax = 30.
print(f'Color scale vmax (99th pct): {a_vmax:.2f}%')

cmap = plt.colormaps['Blues']
levels = np.linspace(0., a_vmax, 15)

fig = plt.figure(figsize=(3.2 * 3 + 0.6, 3.4 * 3 + 0.5))
gs = GridSpec(3, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.08, wspace=0.06)
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
                ax.contourf(Xg, Yg, mp, levels=levels, cmap=cmap, vmin=0, vmax=a_vmax, extend='neither')
                last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, a_vmax))
                last_h.set_array([])
        _sta_markers(ax)
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
    cax = fig.add_subplot(gs[:, 3])
    cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, a_vmax, 5), format=FormatStrFormatter('%.1f'))
    cb.set_label('Travel-time-weighted mean % shear anisotropy per voxel', fontsize=9)
    cb.ax.tick_params(labelsize=7)
legend_handles = [
    mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='OBS station'),
    mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=8, framealpha=0.9,
           bbox_to_anchor=(0.45, 0.0))
n_total = sum(r['n'] for r in results)
fig.suptitle(f'AXEC2 (mfast, dt<=T_dom/2 filtered): travel-time-weighted % anisotropy\n'
             f'N={n_total:,}, Q_w>={QW_MIN}, phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s, dt<=T_dom/2',
             fontsize=11, fontweight='bold', y=0.99)
fig.subplots_adjust(bottom=0.10)

OUT_PATH = 'traveltime_anisotropy_axec2_domperiod_filtered.pdf'
with PdfPages(OUT_PATH) as pdf:
    pdf.savefig(fig, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {OUT_PATH}')
