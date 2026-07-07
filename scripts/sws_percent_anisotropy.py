#!/usr/bin/env python3
"""
sws_percent_anisotropy.py

New version of the spatial dt-plot family (sws_raytraced_dt.py) with no tomographic
inversion - just the per-voxel MEDIAN of a directly-computed percent-anisotropy proxy:

    A = V_s,mean * dt * 100 / r

where, per ray (one ray per event-station splitting measurement):
    V_s,mean = mean S-wave velocity along the ray path (Baillard 3D model)
    dt       = that event's measured delay time (s)
    r        = the ray's total path length (km)

Unlike sws_raytraced_dt.py (which uses bent_ray pseudo-bending and a coverage-weighted
MEAN), this uses PyKonal FMM ray tracing (the more rigorous method established during
today's incidence-angle work) and a per-voxel MEDIAN across all rays intersecting that
voxel (each ray contributes its single scalar A once per voxel it passes through).

QC: phi_error < 20 deg, dt_error < 0.04 s, dt > 0 (null-measurement removal).
"Sufficient ray coverage" = at least COUNT_MIN=5 distinct rays per voxel (matching
sws_raytraced_dt.py's existing convention).

Produces two PDFs (7-period and annual), one page per station plus one all-stations-combined
page, following the exact depth-slice/bathymetry/station-marker/Blues-colormap style of
sws_raytraced_dt.py's output.

Run with:
    python3 sws_percent_anisotropy.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from baillard_velocity import vs_at
from pykonal_raytracer import BaillardRayTracer

BASE = os.path.join(HERE, '..', 'results') + os.sep
STATION_FILE = os.path.join(HERE, '..', 'data', 'stations_axial.llz')
BATHY_FILE = os.path.join(HERE, '..', 'data', 'AXIAL_MODEL_3P_VELOCITY.S.mod.buf')
BATHY_TIF = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
             'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')

OUT_7PANEL = os.path.join(HERE, 'sws_temporal_combined_percent_anisotropy_7period.pdf')
OUT_ANNUAL = os.path.join(HERE, 'sws_temporal_combined_percent_anisotropy_annual.pdf')

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
             (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')
STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET_KM = {
    'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55), 'AXCC1': (-0.75, 0.25),
    'AXEC1': (0.15, 0.30), 'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28),
}

# New QC thresholds (replacing sws_raytraced_dt.py's PHI_ERR_MAX=15-only filter)
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04

# Voxel grid - identical to sws_raytraced_dt.py, for direct spatial comparability
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VOXEL_XY = 0.30
VOXEL_Z = 0.25
xn = np.arange(X_START, X_END + VOXEL_XY * .5, VOXEL_XY)
yn = np.arange(Y_START, Y_END + VOXEL_XY * .5, VOXEL_XY)
zn = np.arange(0., Z_MAX + VOXEL_Z * .5, VOXEL_Z)
NX, NY, NZ = len(xn), len(yn), len(zn)

DEPTH_SLICES = [0.3125, 0.9375, 1.5625, 2.1875]
Z_HALF = 0.3125
DEPTH_LABELS = ['0.0–0.625 km', '0.625–1.25 km', '1.25–1.875 km', '1.875–2.5 km']
COUNT_MIN = 5  # min distinct rays per voxel to display, matching sws_raytraced_dt.py

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


def ray_length_and_vs_mean(ray_xyz):
    """r [km] and mean Vs [km/s] along a PyKonal-traced ray (N,3) array."""
    seg = np.diff(ray_xyz, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    r = float(seg_len.sum())
    mid = 0.5 * (ray_xyz[:-1] + ray_xyz[1:])
    vs_vals = vs_at(mid[:, 0], mid[:, 1], mid[:, 2])
    vs_mean = float(np.average(vs_vals, weights=np.maximum(seg_len, 1e-9)))
    return r, vs_mean


def voxelize_ray(ray_xyz):
    """Distinct voxel (ix,iy,iz) indices the ray passes through (segment midpoints)."""
    mid = 0.5 * (ray_xyz[:-1] + ray_xyz[1:])
    ix = ((mid[:, 0] - X_START) / VOXEL_XY).astype(int)
    iy = ((mid[:, 1] - Y_START) / VOXEL_XY).astype(int)
    iz = (mid[:, 2] / VOXEL_Z).astype(int)
    valid = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < NZ)
    return set(zip(ix[valid].tolist(), iy[valid].tolist(), iz[valid].tolist()))


def build_7_periods(all_df):
    post = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post)
    bounds = [ERUPTION_END]
    for i in range(1, 5):
        idx = min(int(round(i * n / 5)), n - 1)
        bounds.append(post['t'].iloc[idx])
    bounds.append(None)

    def fmt(ts):
        return ts.strftime('%b %Y') if ts else 'present'

    periods = [('Pre-eruption', None, ERUPTION_START),
               ('Syn-eruption', ERUPTION_START, ERUPTION_END)]
    for i in range(5):
        periods.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i + 1])}', bounds[i], bounds[i + 1]))
    return periods


def build_annual_periods():
    periods = [
        ('Pre-eruption\n2015', None, ERUPTION_START),
        ('Syn-eruption\n2015', ERUPTION_START, ERUPTION_END),
        ('Post-eruption\n2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for yr in range(2016, 2027):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr + 1}-01-01', tz='UTC') if yr < 2026 else None
        periods.append((str(yr), t0, t1))
    return periods


def load_data():
    print(f'Loading events (phi_error < {PHI_ERR_MAX}°, dt_error < {DT_ERR_MAX}s)...')
    dfs = {}
    for sta, (f1, f2) in FILES.items():
        def _load(f):
            path = BASE + f
            if not os.path.exists(path):
                return pd.DataFrame()
            d = pd.read_csv(path).loc[:, :'dt_error'].dropna()
            return d[(d['dt'] > 0) & (d['phi_error'] < PHI_ERR_MAX) & (d['dt_error'] < DT_ERR_MAX)]
        df = pd.concat([_load(f1), _load(f2)], ignore_index=True)
        if len(df) == 0:
            print(f'  {sta}: 0 (no files found)')
            continue
        df['x'], df['y'] = ll2xy(df['event_lat'].values, df['event_lon'].values)
        df['z'] = df['event_depth'].values
        df['t'] = pd.to_datetime(df['event_datetime'], utc=True)
        dfs[sta] = df
        print(f'  {sta}: {len(df):,}')
    return dfs


def compute_voxel_A_lists(dfs, sta_xy, tracer, periods):
    """Returns A_lists[sta][period_idx][(ix,iy,iz)] = [A values]."""
    n_per = len(periods)
    STA_LIST = [sta for sta in STATIONS if sta in sta_xy and sta in dfs]
    A_lists = {sta: [dict() for _ in range(n_per)] for sta in STA_LIST}
    N_rays = {sta: [0] * n_per for sta in STA_LIST}

    t_start = time.time()
    total = sum(len(dfs[sta]) for sta in STA_LIST)
    done = 0
    n_skipped_trace = 0

    for sta in STA_LIST:
        sx, sy = sta_xy[sta]
        df_s = dfs[sta]

        for _, row in df_s.iterrows():
            eq_x, eq_y, eq_z = float(row['x']), float(row['y']), float(row['z'])
            dt_val = float(row['dt'])
            t_evt = row['t']
            done += 1

            if eq_z < 0 or eq_z > Z_MAX:
                continue

            per_idx = None
            for j, (lbl, t0, t1) in enumerate(periods):
                m = (t_evt >= t0) if t0 is not None else True
                if t1 is not None:
                    m = m and (t_evt < t1)
                if m:
                    per_idx = j
                    break
            if per_idx is None:
                continue

            try:
                ray = tracer.trace(sta, eq_x, eq_y, eq_z)
            except Exception:
                n_skipped_trace += 1
                continue

            r, vs_mean = ray_length_and_vs_mean(ray)
            if r <= 0:
                continue
            A = vs_mean * dt_val * 100.0 / r

            voxels = voxelize_ray(ray)
            for vox in voxels:
                A_lists[sta][per_idx].setdefault(vox, []).append(A)
            N_rays[sta][per_idx] += 1

            if done % 5000 == 0:
                el = time.time() - t_start
                print(f'  {done:,}/{total:,}  {el:.0f}s elapsed  '
                      f'~{el/done*(total-done):.0f}s remaining', end='\r', flush=True)

    print(f'\nDone in {time.time()-t_start:.0f}s ({n_skipped_trace} rays failed to trace)')
    for sta in STA_LIST:
        print(f'  {sta}: {sum(N_rays[sta]):,} rays total')
    return A_lists, N_rays, STA_LIST


def median_voxel_grid(A_lists_period):
    """dict (ix,iy,iz)->[A] -> (NX,NY,NZ) median grid + (NX,NY,NZ) count grid."""
    med = np.full((NX, NY, NZ), np.nan)
    cnt = np.zeros((NX, NY, NZ), dtype=int)
    for (ix, iy, iz), vals in A_lists_period.items():
        med[ix, iy, iz] = np.median(vals)
        cnt[ix, iy, iz] = len(vals)
    return med, cnt


def main():
    dfs = load_data()
    if not dfs:
        print("No data loaded - aborting.")
        return

    _sta = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'elev_km', 'station'],
                        engine='python').set_index('station')
    _sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
    _sta['x'], _sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)
    sta_xy = {sta: (float(_sta.loc[sta, 'x']), float(_sta.loc[sta, 'y'])) for sta in STATIONS if sta in _sta.index}

    print("Precomputing PyKonal FMM travel-time fields for all stations...")
    tracer = BaillardRayTracer(bathy_file=BATHY_FILE)
    for sta, (sx, sy) in sta_xy.items():
        if sta in dfs:
            tracer.precompute_station(sta, sx, sy)

    all_df = pd.concat(dfs.values(), ignore_index=True)

    # Bathymetry (optional - skip gracefully if unavailable)
    have_bathy = os.path.exists(BATHY_TIF)
    if have_bathy:
        import tifffile
        from PIL import Image as PILImage
        PILImage.MAX_IMAGE_PIXELS = None
        _p = PILImage.open(BATHY_TIF)
        _t = _p.tag_v2
        _olon, _olat = _t[33922][3], _t[33922][4]
        _pl, _pb = _t[33550][0], _t[33550][1]
        _nc, _nr = _p.size
        _p.close()
        _c0 = max(0, int(((INI_LON + X_START / KM_PER_DEG_LON) - _olon) / _pl) - 2)
        _c1 = min(_nc, int(((INI_LON + X_END / KM_PER_DEG_LON) - _olon) / _pl) + 2)
        _r0 = max(0, int((_olat - (INI_LAT + Y_END / KM_PER_DEG_LAT)) / _pb) - 2)
        _r1 = min(_nr, int((_olat - (INI_LAT + Y_START / KM_PER_DEG_LAT)) / _pb) + 2)
        _rgb = tifffile.imread(BATHY_TIF)[_r0:_r1, _c0:_c1]
        _ds = max(1, max(_rgb.shape[:2]) // 1024)
        _rgb = _rgb[::_ds, ::_ds]
        _gray = np.dot(_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
        _ext = [(_olon + _c0 * _pl - INI_LON) * KM_PER_DEG_LON, (_olon + _c1 * _pl - INI_LON) * KM_PER_DEG_LON,
                (_olat - _r1 * _pb - INI_LAT) * KM_PER_DEG_LAT, (_olat - _r0 * _pb - INI_LAT) * KM_PER_DEG_LAT]

        def _bathy(ax):
            ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)
    else:
        print("Bathymetry file not found - plotting without background.")

        def _bathy(ax):
            pass

    def _add_stations(ax, highlight=None):
        for sta, row in _sta.iterrows():
            mfc = '#FFD700' if (highlight is None or sta == highlight) else 'white'
            ax.plot(row['x'], row['y'], '^', ms=6, mfc=mfc, mec='k', mew=0.7, zorder=12)
            dx, dy = LABEL_OFFSET_KM.get(sta, (0.12, 0.12))
            ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)

    Xg, Yg = np.meshgrid(xn, yn, indexing='ij')
    cmap = plt.colormaps['Blues']

    def make_pdf(periods, out_path, title_suffix):
        print(f"\nRay-tracing + computing percent anisotropy for {title_suffix}...")
        A_lists, N_rays, STA_LIST = compute_voxel_A_lists(dfs, sta_xy, tracer, periods)
        n_per = len(periods)

        # Precompute median/count grids per station+period
        grids = {sta: [median_voxel_grid(A_lists[sta][j]) for j in range(n_per)] for sta in STA_LIST}

        # Global A vmax (99th pct across all medians)
        all_vals = []
        for sta in STA_LIST:
            for j in range(n_per):
                med, cnt = grids[sta][j]
                m = cnt >= COUNT_MIN
                if m.any():
                    all_vals.extend(med[m].ravel())
        a_vmax = float(np.percentile(all_vals, 99)) if all_vals else 1.0
        a_vmin = 0.
        levels = np.linspace(a_vmin, a_vmax, 15)
        print(f'Global percent-anisotropy range: 0 – {a_vmax:.3f} % (99th pct)')

        def combined_grid(period_idx):
            med_sum = np.zeros((NX, NY, NZ))
            cnt_sum = np.zeros((NX, NY, NZ), dtype=int)
            all_vals_vox = {}
            for sta in STA_LIST:
                for vox, vals in A_lists[sta][period_idx].items():
                    all_vals_vox.setdefault(vox, []).extend(vals)
            return median_voxel_grid(all_vals_vox)

        def make_page(sta_key, highlight, title):
            n_dep = len(DEPTH_SLICES)
            panel_w = 2.2
            fig = plt.figure(figsize=(n_per * panel_w + 0.6, n_dep * 3.0 + 0.3))
            gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.04], hspace=0.04, wspace=0.04)
            last_h = None

            for ri, (z0, zlbl) in enumerate(zip(DEPTH_SLICES, DEPTH_LABELS)):
                iz0 = max(0, int((z0 - Z_HALF) / VOXEL_Z))
                iz1 = min(NZ, int((z0 + Z_HALF) / VOXEL_Z) + 1)

                for ci, (lbl, t0, t1) in enumerate(periods):
                    ax = fig.add_subplot(gs[ri, ci])
                    _bathy(ax)

                    if sta_key is not None:
                        med_3d, cnt_3d = grids[sta_key][ci]
                        n_rays_ci = N_rays[sta_key][ci]
                    else:
                        med_3d, cnt_3d = combined_grid(ci)
                        n_rays_ci = sum(N_rays[s][ci] for s in STA_LIST)

                    cnt_sl = cnt_3d[:, :, iz0:iz1].sum(axis=2)
                    med_sl_stack = med_3d[:, :, iz0:iz1]
                    with np.errstate(invalid='ignore'):
                        med_sl = np.nanmedian(np.where(np.isnan(med_sl_stack), np.nan, med_sl_stack), axis=2)
                    n_slice = int((cnt_sl >= COUNT_MIN).sum())

                    if n_slice >= 1:
                        a_sm = gaussian_filter(np.nan_to_num(med_sl), sigma=1.0)
                        mask = cnt_sl >= COUNT_MIN
                        a_pl = np.where(mask, a_sm, np.nan)
                        ma = np.ma.masked_invalid(a_pl)
                        if not ma.mask.all():
                            ax.contourf(Xg, Yg, ma, levels=levels, cmap=cmap, vmin=a_vmin, vmax=a_vmax, extend='neither')
                            last_h = ScalarMappable(cmap=cmap, norm=Normalize(a_vmin, a_vmax))
                            last_h.set_array([])

                    _add_stations(ax, highlight)
                    ax.set_xlim(X_START, X_END)
                    ax.set_ylim(Y_START, Y_END)
                    ax.set_aspect('equal', 'box')
                    ax.tick_params(labelsize=4)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    if ri == 0:
                        ax.set_title(f'{lbl}\nN={n_rays_ci:,}', fontsize=7.5, fontweight='bold', pad=1)
                    if ci == 0:
                        ax.set_ylabel(f'{zlbl}\n(n_vox={n_slice:,})', fontsize=6.5)

            if last_h:
                cax = fig.add_subplot(gs[:, n_per])
                ticks = np.linspace(a_vmin, a_vmax, 5)
                cb = plt.colorbar(last_h, cax=cax, ticks=ticks, format=FormatStrFormatter('%.2f'))
                cb.set_label('Median percent anisotropy A [%]', fontsize=9)
                cb.ax.tick_params(labelsize=7)
            fig.suptitle(title, fontsize=11, fontweight='bold', y=0.97)
            return fig

        print(f'Writing {os.path.basename(out_path)}...')
        with PdfPages(out_path) as pdf:
            for sta in STA_LIST:
                fig = make_page(sta_key=sta, highlight=sta,
                                 title=f'{sta} — median percent anisotropy A (no inversion)  '
                                       f'(φ_err<{PHI_ERR_MAX}°, δt_err<{DT_ERR_MAX}s, rays from {sta} only)')
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f'  {sta}')
            fig = make_page(sta_key=None, highlight=None,
                             title=f'All stations — median percent anisotropy A (no inversion)  '
                                   f'(φ_err<{PHI_ERR_MAX}°, δt_err<{DT_ERR_MAX}s, all rays combined)')
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print('  All stations')
        print(f'Saved {out_path}')

    periods_7 = build_7_periods(all_df)
    make_pdf(periods_7, OUT_7PANEL, '7-period')

    periods_annual = build_annual_periods()
    make_pdf(periods_annual, OUT_ANNUAL, 'annual')


if __name__ == '__main__':
    main()
