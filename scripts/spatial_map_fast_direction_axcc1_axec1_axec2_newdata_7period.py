#!/usr/bin/env python3
"""
spatial_map_fast_direction_axcc1_axec1_axec2_newdata_7period.py

Spatial map of fast-polarization direction (phi), 7-period columns, using the new mfast
max_dt=0.2s data for AXCC1/AXEC1/AXEC2 -- same data sources, QC, voxel grid (X:4-12, Y:0-12
km, 0.25 km cells), bathymetry/fault/Kidiwela overlays, and yellow/white station-marker
convention as traveltime_anisotropy_7period_axcc1_axec1_axec2_newdata.py, but:

  - DEPTH-RESOLVED, same 4 depth rows as the sibling (0.0-0.75, 0.75-1.5, 1.5-2.25,
    2.25-3.0 km) -- but each event is binned by its OWN EPICENTER depth directly (its
    hypocentral z), not by which depth band a traced ray happens to pass through.
  - NOT ray-traced: each event is binned into the single 3D voxel (x,y,z) containing its
    OWN EPICENTER location, not into every voxel a traced ray between event and station
    would cross (that ray-based binning is what the sibling traveltime-anisotropy script
    does; this script has no ray tracer at all).
  - Plots fast direction (phi), not percent anisotropy: each occupied voxel (>=COUNT_MIN
    events) gets a double-headed tick mark (undirected axial symmetry, phi is mod 180) at
    the CIRCULAR MEDIAN phi of its events -- same circular_median_phi() convention and
    quiver tick-mark style already used in this repo's deformation_geometry_stress_hemmett.py
    for its "observed fast direction" overlay.

QC (same as the sibling traveltime-anisotropy script): success, dt>0, quality>=0.5,
dt<T_dom/2 (per-event dominant period), phi_error<20 deg, dt_error<0.05s.

Produces: spatial_map_fast_direction_axcc1_axec1_axec2_newdata_7period.pdf (1 page, 4 depth rows x 7 period columns)

Run with:
    python3 spatial_map_fast_direction_axcc1_axec1_axec2_newdata_7period.py
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.lines as mlines

from sws_forward_model import STATION_FILE, ll2xy

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
AXEC2_META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                              'raw_axec2_all_batches_mfast_filters_metadata.csv')
OUT_PDF = os.path.join(HERE, 'spatial_map_fast_direction_axcc1_axec1_axec2_newdata_7period.pdf')

QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05
COUNT_MIN = 5   # minimum events per voxel to plot a tick mark (matches deformation_geometry_stress_hemmett.py)

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

ALL_STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
INCLUDED_STATIONS = ['AXCC1', 'AXEC1', 'AXEC2']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}
KIDIWELA = [dict(x=7.57, y=4.55, label='S1'), dict(x=7.53, y=6.60, label='S2')]

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

# ── Voxel grid (same X/Y/Z extent + cell size as the ray-traced sibling) ─────
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VXY = 0.25
VZ = 0.25
xn = np.arange(X_START, X_END + VXY * .5, VXY)   # left edges
yn = np.arange(Y_START, Y_END + VXY * .5, VXY)
zn = np.arange(0., Z_MAX + VZ * .5, VZ)
NX, NY, NZ = len(xn), len(yn), len(zn)

# Same 4 depth rows as traveltime_anisotropy_7period_axcc1_axec1_axec2_newdata.py (indices
# into the VZ=0.25 zn grid -- 3 cells = 0.75 km per row).
DEPTH_ROWS = [
    ((0, 3), '0.0–0.75 km', 0.375),
    ((3, 6), '0.75–1.5 km', 1.125),
    ((6, 9), '1.5–2.25 km', 1.875),
    ((9, 12), '2.25–3.0 km', 2.625),
]

STATION_FILES = {
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
}


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
            d['dominant_period'] = d['chosen_filter_dom_period_samples'] / 200.0
            d = d.merge(meta, on='event_id', how='left')
            dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'phi', 'quality', 'dominant_period',
                            'phi_error', 'dt_error'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')

    df = df[(df['quality'] >= QW_MIN) &
            (df['dt'] < df['dominant_period'] / 2.0) &
            (df['phi_error'] < PHI_ERR_MAX) &
            (df['dt_error'] < DT_ERR_MAX)]

    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    df['phi_az'] = df['phi'] % 180.0
    return df[['t', 'phi_az', 'x', 'y', 'z']]


def build_7_periods(all_df):
    post = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post)
    bounds = [ERUPTION_END]
    for i in range(1, 5):
        bounds.append(post['t'].iloc[min(int(round(i * n / 5)), n - 1)])
    bounds.append(None)

    def fmt(ts):
        return ts.strftime('%b %Y') if ts else 'present'

    pds = [('Pre-eruption', None, ERUPTION_START), ('Syn-eruption', ERUPTION_START, ERUPTION_END)]
    for i in range(5):
        pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i + 1])}', bounds[i], bounds[i + 1]))
    return pds


def _subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None:
        m = m & (df['t'] < t1)
    return df[m]


def circular_median_phi(phi_deg):
    """Circular median of axially-symmetric (mod 180) fast directions (doubled-angle trick)
    -- same convention as deformation_geometry_stress_hemmett.py."""
    if len(phi_deg) == 0:
        return np.nan
    angles = 2.0 * np.radians(phi_deg)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return float(np.degrees(np.arctan2(s, c)) / 2.0) % 180.0


def bin_by_epicenter(df, z_lo=None, z_hi=None):
    """Bin events into the 2D voxel containing their OWN epicenter (x,y) -- no ray tracing.
    If z_lo/z_hi are given, first restrict to events whose OWN epicenter depth falls in
    [z_lo, z_hi) (also not ray-traced -- each event contributes to exactly one depth band,
    its own hypocentral depth, not every band a ray to some station might pass through).
    Returns (PHI, CNT), both (NX,NY): circular-median phi (NaN if <COUNT_MIN) and raw count."""
    if z_lo is not None:
        df = df[(df['z'] >= z_lo) & (df['z'] < z_hi)]
    ix = np.floor((df['x'].values - X_START) / VXY).astype(int)
    iy = np.floor((df['y'].values - Y_START) / VXY).astype(int)
    valid = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    ix, iy, phi = ix[valid], iy[valid], df['phi_az'].values[valid]

    PHI = np.full((NX, NY), np.nan)
    CNT = np.zeros((NX, NY), dtype=int)
    if len(ix) == 0:
        return PHI, CNT

    flat = ix * NY + iy
    order = np.argsort(flat)
    flat_sorted, phi_sorted = flat[order], phi[order]
    uniq, start_idx, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
    for u, s0, c in zip(uniq, start_idx, counts):
        vals = phi_sorted[s0:s0 + c]
        vi, vj = u // NY, u % NY
        CNT[vi, vj] = c
        if c >= COUNT_MIN:
            PHI[vi, vj] = circular_median_phi(vals)
    return PHI, CNT


def _setup_environment():
    global _sta_df, _gray, _ext

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                          engine='python').set_index('s')
    _sta_df = _sta_df.loc[[s for s in ALL_STATIONS if s in _sta_df.index]]
    _sta_df['x'], _sta_df['y'] = ((np.asarray(_sta_df['lon']) - INI_LON) * KM_PER_DEG_LON,
                                  (np.asarray(_sta_df['lat']) - INI_LAT) * KM_PER_DEG_LAT)

    bathy = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
             'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
    PILImage.MAX_IMAGE_PIXELS = None
    _p = PILImage.open(bathy)
    _tag = _p.tag_v2
    _olon, _olat = _tag[33922][3], _tag[33922][4]
    _pl, _pb = _tag[33550][0], _tag[33550][1]
    _nc, _nr = _p.size
    _p.close()
    _c0 = max(0, int(((INI_LON + X_START / KM_PER_DEG_LON) - _olon) / _pl) - 2)
    _c1 = min(_nc, int(((INI_LON + X_END / KM_PER_DEG_LON) - _olon) / _pl) + 2)
    _r0 = max(0, int((_olat - (INI_LAT + Y_END / KM_PER_DEG_LAT)) / _pb) - 2)
    _r1 = min(_nr, int((_olat - (INI_LAT + Y_START / KM_PER_DEG_LAT)) / _pb) + 2)
    _rgb = tifffile.imread(bathy)[_r0:_r1, _c0:_c1]
    _ds = max(1, max(_rgb.shape[:2]) // 1024)
    _rgb = _rgb[::_ds, ::_ds]
    _gray = np.dot(_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
    _ext = [(_olon + _c0 * _pl - INI_LON) * KM_PER_DEG_LON,
            (_olon + _c1 * _pl - INI_LON) * KM_PER_DEG_LON,
            (_olat - _r1 * _pb - INI_LAT) * KM_PER_DEG_LAT,
            (_olat - _r0 * _pb - INI_LAT) * KM_PER_DEG_LAT]


def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)


def _sta(ax):
    for sta, row in _sta_df.iterrows():
        color = '#FFD700' if sta in INCLUDED_STATIONS else '#FFFFFF'
        ax.plot(row['x'], row['y'], '^', ms=6, mfc=color, mec='k', mew=0.7, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=5.5, zorder=13)


def _kid(ax):
    for s in KIDIWELA:
        ax.plot(s['x'], s['y'], 'o', ms=4, mfc='red', mec='k', mew=0.5, zorder=14)
        ax.text(s['x'] + 0.12, s['y'] + 0.12, s['label'], fontsize=5, color='red',
                fontweight='bold', zorder=15)


COT66 = math.cos(math.radians(66)) / math.sin(math.radians(66))
DEPTH_TOP = 0.1
strike_rad = math.radians(330.)
L = 8.
s_hat = np.array([math.sin(strike_rad), math.cos(strike_rad)])


def _faults(ax, z0=0.375):
    for cx, cy, dip_az in [(8.5, 4.0, 60.), (7.0, 4.0, 240.)]:
        horiz = (z0 - DEPTH_TOP) * COT66
        da = math.radians(dip_az)
        cx_z = cx + horiz * math.sin(da)
        cy_z = cy + horiz * math.cos(da)
        x1, y1 = cx_z - L / 2 * s_hat[0], cy_z - L / 2 * s_hat[1]
        x2, y2 = cx_z + L / 2 * s_hat[0], cy_z + L / 2 * s_hat[1]
        ax.plot([x1, x2], [y1, y2], '-', color='#cc0000', lw=0.9, alpha=0.75, zorder=11)


# Azimuth colormap: phi is axial (mod 180 -- 0 deg and 180 deg are the SAME direction),
# so a cyclic colormap is used ('hsv' wraps back to its own starting color at the top of
# its range) with the norm's range set to [0, 180] rather than the usual [0, 360] -- that
# makes phi=0 and phi=180 map to the identical color, consistent with the axial symmetry.
AZ_CMAP = plt.colormaps['hsv']
AZ_NORM = Normalize(vmin=0.0, vmax=180.0)


def make_page(results, title):
    """results: list (one per period) of dicts with 'label', 'n', and 'PHI_rows' --
    a list of (NX,NY) arrays, one per DEPTH_ROWS entry."""
    n_per = len(results)
    n_dep = len(DEPTH_ROWS)
    fig = plt.figure(figsize=(n_per * 2.4 + 0.5, n_dep * 2.6 + 0.5))
    gs = GridSpec(n_dep, n_per + 1, width_ratios=[1] * n_per + [0.05], hspace=0.06, wspace=0.06)

    tick_kw = dict(scale=18, width=0.006, headlength=0, headaxislength=0, headwidth=0,
                  pivot='middle', cmap=AZ_CMAP, norm=AZ_NORM, alpha=0.95, zorder=7)

    for ri, (_iz_range, zlbl, z0) in enumerate(DEPTH_ROWS):
        for ci, res in enumerate(results):
            ax = fig.add_subplot(gs[ri, ci])
            _bathy(ax)

            PHI = res['PHI_rows'][ri]
            iix, iiy = np.where(np.isfinite(PHI))
            if len(iix):
                xo = X_START + (iix + 0.5) * VXY
                yo = Y_START + (iiy + 0.5) * VXY
                phi_vals = PHI[iix, iiy]
                ro = np.radians(phi_vals)
                uo, vo = np.sin(ro), np.cos(ro)
                ax.quiver(xo, yo, uo, vo, phi_vals, **tick_kw)
                ax.quiver(xo, yo, -uo, -vo, phi_vals, **tick_kw)

            _sta(ax)
            _kid(ax)
            _faults(ax, z0)
            ax.set_xlim(X_START, X_END)
            ax.set_ylim(Y_START, Y_END)
            ax.set_aspect('equal', 'box')
            ax.tick_params(labelsize=5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(f'{res["label"]}\nN={res["n"]:,}', fontsize=7.5,
                             fontweight='bold', pad=2)
            if ci == 0:
                ax.set_ylabel(zlbl, fontsize=7)

    cax = fig.add_subplot(gs[:, n_per])
    sm = ScalarMappable(cmap=AZ_CMAP, norm=AZ_NORM)
    sm.set_array([])
    cb = fig.colorbar(cax=cax, mappable=sm, ticks=np.linspace(0, 180, 7))
    cb.set_label('Fast direction $\\phi$ (deg, axial 0-180)', fontsize=8)
    cb.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6,
                      label='Included station (new data)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFFFFF', mec='k', ms=6,
                      label='Excluded station (no new data)'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title + f'\n(tick marks: N$\\geq${COUNT_MIN}/voxel, colored by fast direction)',
                 fontsize=10, fontweight='bold', y=0.995)
    fig.subplots_adjust(bottom=0.06, top=0.90)
    return fig


def main():
    _setup_environment()

    print('Loading new-data QC-filtered events (quality>=0.5, dt<T_dom/2, '
          f'phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s)...')
    dfs = {}
    for sta in INCLUDED_STATIONS:
        dfs[sta] = load_station(sta)
        print(f'  {sta}: {len(dfs[sta]):,}')
    all_df = pd.concat(dfs.values(), ignore_index=True)
    print(f'  TOTAL: {len(all_df):,}')

    periods = build_7_periods(all_df)

    results = []
    for lbl, t0, t1 in periods:
        sub = _subset(all_df, t0, t1)
        phi_rows = []
        n_vox_total = 0
        for (iz0, iz1), _zlbl, _z0 in DEPTH_ROWS:
            z_lo, z_hi = iz0 * VZ, iz1 * VZ
            PHI, _CNT = bin_by_epicenter(sub, z_lo=z_lo, z_hi=z_hi)
            phi_rows.append(PHI)
            n_vox_total += int(np.isfinite(PHI).sum())
        print(f'  {lbl.splitlines()[0]:20s}: N={len(sub):,}  occupied voxels (all depths)={n_vox_total:,}')
        results.append(dict(label=lbl, n=len(sub), PHI_rows=phi_rows))

    title = (f'AXCC1/AXEC1/AXEC2 (new mfast max_dt=0.2s data) — fast direction by epicenter '
             f'location (not ray-traced), 7-period x 4 depth rows\n'
             f'quality>={QW_MIN}, dt<T_dom/2, phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s '
             f'(n={len(all_df):,})')
    fig = make_page(results, title)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
