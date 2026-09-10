#!/usr/bin/env python3
"""
rose_map_by_station_axcc1_axec1_axec2_newdata_7period.py

7-period spatial map where, instead of tick marks or a voxel field, each panel shows the
FULL ROSE DIAGRAM (doubled-angle fast-direction histogram, same drawing convention as
rose_7period_axcc1_axec1_axec2_newdata.py's _draw_rose) for each station, drawn as a polar
inset positioned AT that station's actual (x,y) location on top of the grayscale bathymetry
map -- using sws_methods.get_ax_inset (the same inset-axes helper documented for this exact
purpose in sws_methods.py). Ring faults and the two Kidiwela sources are also drawn, matching
the sibling scripts' style (traveltime_anisotropy_7period_axcc1_axec1_axec2_newdata.py /
spatial_map_fast_direction_axcc1_axec1_axec2_newdata_7period.py).

Data: same new mfast max_dt=0.2s data + QC as the other new-data scripts this session
(AXCC1/AXEC1/AXEC2 only; success, dt>0, quality>=0.5, dt<T_dom/2, phi_error<20 deg,
dt_error<0.05s). 7 eruption-relative periods (verbatim scheme).

Produces: rose_map_by_station_axcc1_axec1_axec2_newdata_7period.pdf (1 page, 7 panels)

Run with:
    python3 rose_map_by_station_axcc1_axec1_axec2_newdata_7period.py
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
import matplotlib.colors as mcolors
import tifffile
from PIL import Image as PILImage
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines

import sws_methods as swm
from sws_forward_model import STATION_FILE, ll2xy

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
AXEC2_META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                              'raw_axec2_all_batches_mfast_filters_metadata.csv')
OUT_PDF = os.path.join(HERE, 'rose_map_by_station_axcc1_axec1_axec2_newdata_7period.pdf')

QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05

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


def _period_colors(n_periods):
    """Same 7-period color scheme as rose_7period_axcc1_axec1_axec2_newdata.py's
    _period_colors: pre-eruption=purple, syn-eruption=red, then a light-blue-to-purple
    gradient across the post-eruption bins. All stations within a period share this one
    color -- color encodes TIME PERIOD, not station."""
    colors = ['#800080', '#CC0000']
    n_post = n_periods - 2
    start = np.array(mcolors.to_rgb('#ADD8E6'))
    end = np.array(mcolors.to_rgb('#800080'))
    for i in range(n_post):
        t = i / max(n_post - 1, 1)
        colors.append(mcolors.to_hex((1 - t) * start + t * end))
    return colors

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
ROSE_WIDTH_KM = 1.3   # inset polar-plot diameter, in map data (km) units

NBINS = 36

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
        for d in batch_dfs:
            d['dominant_period'] = d['chosen_filter_dom_period_samples'] / 200.0
        dfs.extend(batch_dfs)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['phi', 'quality', 'dominant_period', 'phi_error', 'dt_error'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')

    df = df[(df['quality'] >= QW_MIN) &
            (df['dt'] < df['dominant_period'] / 2.0) &
            (df['phi_error'] < PHI_ERR_MAX) &
            (df['dt_error'] < DT_ERR_MAX)]

    df['phi_az'] = df['phi'] % 180.0
    return df[['t', 'phi_az']]


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


def _draw_rose(ax, phi_az_vals, color):
    """Doubled-angle fast-direction histogram, same convention as
    rose_7period_axcc1_axec1_axec2_newdata.py's _draw_rose."""
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('none')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(0.5)
    ax.grid(False)

    if len(phi_az_vals) == 0:
        ax.set_ylim(0, 1)
        return

    doubled_angles = []
    for phi in phi_az_vals:
        p = float(phi) % 360.0
        doubled_angles.extend([np.deg2rad(p), np.deg2rad((p + 180.0) % 360.0)])
    doubled_angles = np.array(doubled_angles)

    bins = np.linspace(0, 2 * np.pi, NBINS + 1)
    counts, edges = np.histogram(doubled_angles, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 2 * np.pi / NBINS

    ax.bar(centers, counts, width=width, bottom=0, color=color, edgecolor='black',
          linewidth=0.3, alpha=0.9)
    ax.set_ylim(0, counts.max() * 1.15 if counts.max() > 0 else 1)


def _setup_environment():
    global _sta_df, _gray, _ext

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                          engine='python').set_index('s')
    _sta_df = _sta_df.loc[[s for s in ALL_STATIONS if s in _sta_df.index]]
    _sta_df['x'], _sta_df['y'] = ll2xy(_sta_df['lat'].values, _sta_df['lon'].values)

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


def _kid(ax):
    for s in KIDIWELA:
        ax.plot(s['x'], s['y'], 'o', ms=4, mfc='red', mec='k', mew=0.5, zorder=4)
        ax.text(s['x'] + 0.12, s['y'] + 0.12, s['label'], fontsize=5, color='red',
                fontweight='bold', zorder=5)


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
        ax.plot([x1, x2], [y1, y2], '-', color='#cc0000', lw=0.9, alpha=0.75, zorder=1)


def make_page(period_data, title):
    n_per = len(period_data)
    colors = _period_colors(n_per)
    fig = plt.figure(figsize=(n_per * 2.4, 3.2))
    gs = GridSpec(1, n_per, hspace=0.04, wspace=0.08)

    for ci, (lbl, sta_phi) in enumerate(period_data):
        ax = fig.add_subplot(gs[0, ci])
        _bathy(ax)
        _faults(ax)
        _kid(ax)
        ax.set_xlim(X_START, X_END)
        ax.set_ylim(Y_START, Y_END)
        ax.set_aspect('equal', 'box')
        ax.tick_params(labelsize=5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        n_total = sum(len(v) for v in sta_phi.values())
        ax.set_title(f'{lbl}\nN={n_total:,}', fontsize=7.5, fontweight='bold', pad=2)

        # get_ax_inset() needs an up-to-date renderer/position, per its own docstring.
        fig.canvas.draw()

        for sta in INCLUDED_STATIONS:
            phi_vals = sta_phi[sta]
            sx, sy = float(_sta_df.loc[sta, 'x']), float(_sta_df.loc[sta, 'y'])
            ax_inset = swm.get_ax_inset(ax, sx, sy, width=ROSE_WIDTH_KM,
                                       projection='polar', alpha=0.75, visible_axis=False)
            _draw_rose(ax_inset, phi_vals, colors[ci])

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.05)
    fig.subplots_adjust(bottom=0.02)
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

    periods = build_7_periods(all_df)

    period_data = []
    for lbl, t0, t1 in periods:
        sta_phi = {}
        for sta in INCLUDED_STATIONS:
            sub = _subset(dfs[sta], t0, t1)
            sta_phi[sta] = sub['phi_az'].values
        counts = {sta: len(v) for sta, v in sta_phi.items()}
        print(f'  {lbl.splitlines()[0]:20s}: {counts}')
        period_data.append((lbl, sta_phi))

    title = ('AXCC1/AXEC1/AXEC2 (new mfast max_dt=0.2s data) — per-station rose diagrams '
             'on bathymetry, 7-period\n'
             f'quality>={QW_MIN}, dt<T_dom/2, phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s')
    fig = make_page(period_data, title)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
