"""
Travel-time-weighted percent shear-wave anisotropy, old (fixed 5-40 Hz bandpass)
vs. new (mfast try_filters) AXEC2 pipeline, restricted to pre-eruption events
(datetime < 2015-04-24 06:00 UTC) in the batches the mfast run has completed so
far, split into 3 equally-spaced-by-earthquake-COUNT time periods and 3 depth
ranges (0.0-0.75, 0.75-1.5, 1.5-2.25 km).

Method: same travel-time-weighted fractional-anisotropy formula as
lqt_pykonal_tomography_traveltime_anisotropy.py:
    T_travel = sum_k seg_len_km[k] / Vs_voxel[k]     (per ray, k = crossed voxels)
    A_ray    = dt * 100 / T_travel                   (percent)
    V_voxel  = (sum_rays A_ray * tt_k) / (sum_rays tt_k),  tt_k = seg_len_km/Vs_voxel

Ray geometry (voxel crossings) depends only on event (x,y,z) and station
location, not on which bandpass filter produced the dt - so each unique event
is traced ONCE and its cached geometry is reused to compute A_ray separately
from the old dt and the new dt (huge speedup vs. tracing both datasets).

Bathymetry/station/fault-trace plotting helpers ported verbatim from
lqt_pykonal_tomography_traveltime_anisotropy.py / sws_tomography_inversion.py.

Produces one 2-page PDF (page 1 = old, page 2 = new), same color scale on both
pages so the reduction in spurious high-dt spatial anomalies is directly
comparable: traveltime_anisotropy_old_vs_new_pre_eruption.pdf

Run with:
    python3 plot_traveltime_anisotropy_old_vs_new_pre_eruption.py
"""

import glob
import math
import re
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
STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']  # for map context only
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

MAX_DT = 0.30
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

# 3 depth ranges, 0.75 km each (0-2.25 km) - same 0.25 km voxel convention as
# the sibling tomography scripts, just the first 3 of their 4 rows.
DEPTH_ROWS = [
    ((0, 3), '0.0-0.75 km', 0.375),
    ((3, 6), '0.75-1.5 km', 1.125),
    ((6, 9), '1.5-2.25 km', 1.875),
]

COUNT_MIN = 5
MIN_PATH_KM = 0.3

# ── Load old + new AXEC2 mfast-batch-matched, pre-eruption results ────────────

new_result_files = glob.glob('production_axec2_mfast_filters_lqt_pykonal_results/'
                              'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_*.csv')
batch_nums = sorted(int(re.search(r'batch_(\d+)\.csv', f).group(1)) for f in new_result_files)
print(f'Batches available: {len(batch_nums)}, range {batch_nums[0]}-{batch_nums[-1]}')

old_files = [f'production_axec2_lqt_pykonal_results/'
             f'splitting_results_mldd_2015_2021_axec2_batch_{n}.csv' for n in batch_nums]
new_files = [f'production_axec2_mfast_filters_lqt_pykonal_results/'
             f'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_{n}.csv' for n in batch_nums]

old_meta = pd.read_csv('raw_axec2_all_batches_data/raw_axec2_all_batches_metadata.csv')
old_meta = old_meta[old_meta['batch'].isin(batch_nums)][['event_id', 'latitude', 'longitude', 'depth']]
new_meta = pd.read_csv('raw_axec2_all_batches_mfast_filters_data/raw_axec2_all_batches_mfast_filters_metadata.csv')
new_meta = new_meta[new_meta['batch'].isin(batch_nums)][['event_id', 'latitude', 'longitude', 'depth']]


def load(files, meta):
    dfs = [pd.read_csv(f) for f in files]
    dfs = [d for d in dfs if len(d) > 0]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[(df['dt'] > 0) & (df['dt'] <= MAX_DT)]
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    df = df[df['t'] < ERUPTION_START].copy()
    df = df.merge(meta, on='event_id', how='left').dropna(subset=['latitude', 'longitude', 'depth'])
    return df


old_df = load(old_files, old_meta)
new_df = load(new_files, new_meta)
print(f'Old, pre-eruption, baseline QC: {len(old_df):,} events')
print(f'New, pre-eruption, baseline QC: {len(new_df):,} events')

# ── Unique event geometry (union of old+new event_ids) - trace ONCE each ─────

union = pd.concat([old_df[['event_id', 'latitude', 'longitude', 'depth']],
                    new_df[['event_id', 'latitude', 'longitude', 'depth']]], ignore_index=True)
union = union.drop_duplicates(subset='event_id').reset_index(drop=True)
ux, uy = ll2xy(union['latitude'].values, union['longitude'].values)
union['x'] = ux
union['y'] = uy
union['z'] = union['depth'].values
print(f'Unique events to trace (union of old+new): {len(union):,}')

# ── Vs per voxel (Baillard model, true voxel centres, clipped) ───────────────

print('Querying Baillard Vs at each voxel centre...')
X3, Y3, Z3 = np.meshgrid(xn + VXY / 2., yn + VXY / 2., zn + VZ / 2., indexing='ij')
vs_voxels = vs_at(X3.ravel(), Y3.ravel(), Z3.ravel()).reshape(NX, NY, NZ)
vs_voxels = np.clip(vs_voxels, 0.3, 5.0)
vs_flat = vs_voxels.ravel()

# ── Station table + bathymetry (for map context / plotting) ──────────────────

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

print(f'\nTracing {len(union):,} unique events to AXEC2 (n_pts={N_RAY})...')
tracer = BaillardRayTracer()
tracer.precompute_station(STATION, sta_x, sta_y)

t0 = time.time()
event_T_travel = {}   # event_id -> T_travel [s]
event_voxel_hits = {}  # event_id -> list of (ix,iy,iz, tt_k) for aggregation

for i, row in enumerate(union.itertuples()):
    ex, ey, ez = float(row.x), float(row.y), float(row.z)
    if ez < 0 or ez > Z_MAX:
        continue
    try:
        ray = tracer.trace(STATION, ex, ey, ez, n_pts=N_RAY)
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
        print(f'  {i+1:,}/{len(union):,}  {time.time()-t0:.0f}s', end='\r', flush=True)

print(f'\nTraced {len(event_T_travel):,}/{len(union):,} events successfully in {time.time()-t0:.0f}s')

# ── 3 equal-earthquake-count pre-eruption periods (from the union event set) ─

union_traced = union[union['event_id'].isin(event_T_travel.keys())].copy()
union_traced = union_traced.merge(
    pd.concat([old_df[['event_id', 't']], new_df[['event_id', 't']]]).drop_duplicates('event_id'),
    on='event_id', how='left')
union_traced = union_traced.sort_values('t').reset_index(drop=True)
n = len(union_traced)
bounds = [None]
for i in range(1, 3):
    bounds.append(union_traced['t'].iloc[min(int(round(i * n / 3)), n - 1)])
bounds.append(None)


def fmt(ts):
    return 'start' if ts is None else ts.strftime('%b %d')


periods = []
for i in range(3):
    t0_, t1_ = bounds[i], bounds[i + 1]
    lbl = f'{fmt(t0_)}-{fmt(t1_) if t1_ is not None else "eruption"}'
    periods.append((lbl, t0_, t1_))

print('\nPre-eruption equal-count periods:')
for lbl, t0_, t1_ in periods:
    print(f'  {lbl}: [{t0_}, {t1_})')


def assign_period(t_series):
    idx = np.full(len(t_series), -1, dtype=np.int64)
    tvals = t_series.values
    for pi, (_lbl, t0_, t1_) in enumerate(periods):
        m = np.ones(len(t_series), dtype=bool)
        if t0_ is not None:
            m &= tvals >= np.datetime64(t0_)
        if t1_ is not None:
            m &= tvals < np.datetime64(t1_)
        idx[m & (idx < 0)] = pi
    return idx


# ── Aggregate per dataset (old / new) ─────────────────────────────────────────

def aggregate_dataset(df):
    """df: old_df or new_df (event_id, dt, t, ...). Returns list of dict(label,n,a)
    per period, using cached per-event geometry (event_T_travel/event_voxel_hits)."""
    df = df[df['event_id'].isin(event_T_travel.keys())].copy()
    df['period'] = assign_period(df['t'])
    df = df[df['period'] >= 0]

    results = []
    for pi, (lbl, _t0, _t1) in enumerate(periods):
        sub = df[df['period'] == pi]
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
            np.add.at(sum_p, flat, tt_seg * vs_flat[flat])  # back out seg_len_km = tt*vs
            np.add.at(cnt, flat, 1)
        wmean = np.divide(sum_wa, sum_w, out=np.full_like(sum_wa, np.nan), where=sum_w > 0)
        gate = (cnt >= COUNT_MIN) & (sum_p >= MIN_PATH_KM)
        a3d = np.where(gate, wmean, np.nan).reshape(NX, NY, NZ)
        results.append(dict(label=lbl, n=len(sub), a=a3d))
    return results


DT_ERR_MAX = 0.04
PHI_ERR_MAX = 20.0
DT_CUTOFF = MAX_DT * 0.8   # 0.24 s


def _clean(df):
    return df.dropna(subset=['dt_error', 'phi_error', 'quality'])


FILTERS = [
    ('baseline (success & 0<dt<=0.30s)', lambda df: df),
    (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}deg',
     lambda df: _clean(df)[(_clean(df)['dt_error'] < DT_ERR_MAX) & (_clean(df)['phi_error'] < PHI_ERR_MAX)]),
    (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}deg & Q_w>=0.7',
     lambda df: (lambda d: d[(d['dt_error'] < DT_ERR_MAX) & (d['phi_error'] < PHI_ERR_MAX) & (d['quality'] >= 0.7)])(_clean(df))),
    (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}deg & Q_w>=0.7 & dt<={DT_CUTOFF:.2f}s',
     lambda df: (lambda d: d[(d['dt_error'] < DT_ERR_MAX) & (d['phi_error'] < PHI_ERR_MAX) & (d['quality'] >= 0.7) & (d['dt'] <= DT_CUTOFF)])(_clean(df))),
    (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}deg & Q_w>=-0.5',
     lambda df: (lambda d: d[(d['dt_error'] < DT_ERR_MAX) & (d['phi_error'] < PHI_ERR_MAX) & (d['quality'] >= -0.5)])(_clean(df))),
    (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}deg & Q_w>=-0.5 & dt<={DT_CUTOFF:.2f}s',
     lambda df: (lambda d: d[(d['dt_error'] < DT_ERR_MAX) & (d['phi_error'] < PHI_ERR_MAX) & (d['quality'] >= -0.5) & (d['dt'] <= DT_CUTOFF)])(_clean(df))),
]

cmap = plt.colormaps['Blues']


def make_page(results, title, levels, a_vmax):
    n_per = len(results)
    n_dep = len(DEPTH_ROWS)
    fig = plt.figure(figsize=(n_per * 3.2 + 0.6, n_dep * 3.4 + 0.5))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.05], hspace=0.08, wspace=0.06)
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
                ax.set_ylabel(f'{zlbl}', fontsize=8)
    if last_h:
        cax = fig.add_subplot(gs[:, n_per])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, a_vmax, 5), format=FormatStrFormatter('%.1f'))
        cb.set_label('Travel-time-weighted mean % shear anisotropy per voxel', fontsize=9)
        cb.ax.tick_params(labelsize=7)
    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='OBS station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.45, 0.0))
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.10)
    return fig


OUT_PATH = 'traveltime_anisotropy_old_vs_new_pre_eruption.pdf'
with PdfPages(OUT_PATH) as pdf:
    for filt_label, filt_fn in FILTERS:
        print(f'\n=== Filter: {filt_label} ===')
        old_f = filt_fn(old_df)
        new_f = filt_fn(new_df)

        print('Aggregating old (5-40 Hz fixed)...')
        old_results = aggregate_dataset(old_f)
        for r in old_results:
            print(f'  {r["label"]}: N={r["n"]:,}, covered voxels={np.isfinite(r["a"]).sum():,}')

        print('Aggregating new (mfast try_filters)...')
        new_results = aggregate_dataset(new_f)
        for r in new_results:
            print(f'  {r["label"]}: N={r["n"]:,}, covered voxels={np.isfinite(r["a"]).sum():,}')

        # Shared color scale across BOTH pages of this filter level, for direct
        # old-vs-new comparison at that QC level (scale varies across filter
        # levels since stricter QC removes more of the high-dt tail).
        all_vals = []
        for res_list in (old_results, new_results):
            for r in res_list:
                v = r['a'][np.isfinite(r['a'])]
                if v.size:
                    all_vals.append(v)
        a_vmax = float(np.nanpercentile(np.concatenate(all_vals), 99)) if all_vals else 30.
        if not np.isfinite(a_vmax) or a_vmax <= 0:
            a_vmax = 30.
        levels = np.linspace(0., a_vmax, 15)
        print(f'  Shared color scale vmax (99th pct, this filter): {a_vmax:.2f}%')

        fig1 = make_page(old_results, f'AXEC2 pre-eruption, batches {batch_nums[0]}-{batch_nums[-1]}: '
                                       f'Old (5-40 Hz fixed) travel-time-weighted % anisotropy\n'
                                       f'3 equal-earthquake-count periods, filter: {filt_label}',
                          levels, a_vmax)
        pdf.savefig(fig1, dpi=200, bbox_inches='tight')
        plt.close(fig1)

        fig2 = make_page(new_results, f'AXEC2 pre-eruption, batches {batch_nums[0]}-{batch_nums[-1]}: '
                                       f'New (mfast try_filters) travel-time-weighted % anisotropy\n'
                                       f'3 equal-earthquake-count periods, filter: {filt_label}',
                          levels, a_vmax)
        pdf.savefig(fig2, dpi=200, bbox_inches='tight')
        plt.close(fig2)

print(f'\nSaved {OUT_PATH} ({len(FILTERS) * 2} pages)')
