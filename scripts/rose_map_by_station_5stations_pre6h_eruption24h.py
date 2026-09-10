#!/usr/bin/env python3
"""
rose_map_by_station_5stations_pre6h_eruption24h.py

Zoomed-in variant of rose_map_by_station_5stations_newdata_7period.py: same bathymetry +
ring-fault + Kidiwela background and same per-station rose-diagram-as-polar-inset drawing
(sws_methods.get_ax_inset, same convention), for the 5 new-data stations (AXAS1, AXCC1,
AXEC1, AXEC2, AXEC3 -- AXAS2 excluded, no new-data rerun yet), but:

  - STATION TRIANGLE MARKERS ARE NOT DRAWN (only the rose insets themselves mark each
    station's location) -- per explicit user request, unlike the sibling script.
  - Time periods are NOT the usual 7 eruption-relative periods. Instead: one 6-HOUR
    pre-eruption bin (eruption onset - 6h to onset), followed by 24 HOURLY bins covering the
    first 24 hours of the eruption (onset to onset+24h) -- 25 panels total.
  - The panel grid (5 rows x 5 cols) exists ONLY to fit all 25 panels on one page -- rows do
    NOT correspond to depth or any other physical binning, unlike every other multi-row map
    in this repo.

QC: same tier-4 standard as this session's other 5-station scripts (quality>=0.5,
dt<T_dom/2, phi_error<20 deg, dt_error<0.05s), imported directly from
rose_7period_5stations_newdata.py.

Produces: rose_map_by_station_5stations_pre6h_eruption24h.pdf (1 page, 25 panels)

Run with:
    python3 rose_map_by_station_5stations_pre6h_eruption24h.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import tifffile
from PIL import Image as PILImage

import sws_methods as swm
from sws_forward_model import STATION_FILE, ll2xy

from rose_7period_5stations_newdata import STATION_ORDER, load_station_raw, apply_tier

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_map_by_station_5stations_pre6h_eruption24h.pdf')

TIER_IDX = 4   # quality>=0.5, dt<T_dom/2, phi_error<20deg, dt_error<0.05s

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
PRE_WINDOW_HOURS = 6
N_HOURS_ZOOM = 24

KIDIWELA = [dict(x=7.57, y=4.55, label='S1'), dict(x=7.53, y=6.60, label='S2')]

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
ROSE_WIDTH_KM = 0.65   # shrunk from the 7-period sibling's original convention since
                       # AXEC1/AXEC2/AXEC3 sit only ~0.5-1.2 km apart

NBINS = 36
N_ROWS, N_COLS = 5, 5   # purely a page-fitting grid for the 25 panels -- NOT depth rows


def build_periods():
    """One 6h pre-eruption bin, then 24 hourly bins covering the first 24h of the eruption."""
    pre_lbl = f'Pre-eruption\n(-{PRE_WINDOW_HOURS}h to onset)'
    periods = [(pre_lbl, ERUPTION_START - pd.Timedelta(hours=PRE_WINDOW_HOURS), ERUPTION_START)]
    for h in range(N_HOURS_ZOOM):
        h0 = ERUPTION_START + pd.Timedelta(hours=h)
        h1 = ERUPTION_START + pd.Timedelta(hours=h + 1)
        periods.append((f'Hour {h}-{h+1}\n{h0.strftime("%m-%d %H:%M")}', h0, h1))
    return periods


def _period_colors(n_periods):
    """First color (pre-eruption) is purple; remaining n_periods-1 (the 24 hourly bins)
    get a light-blue-to-red gradient showing chronological progression through the onset."""
    colors = ['#800080']
    n_rest = n_periods - 1
    start = np.array(mcolors.to_rgb('#ADD8E6'))
    end = np.array(mcolors.to_rgb('#CC0000'))
    for i in range(n_rest):
        t = i / max(n_rest - 1, 1)
        colors.append(mcolors.to_hex((1 - t) * start + t * end))
    return colors


def _subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None:
        m = m & (df['t'] < t1)
    return df[m]


def _draw_rose(ax, phi_az_vals, color):
    """Doubled-angle fast-direction histogram, same convention as
    rose_7period_5stations_newdata.py's _draw_rose."""
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
    _sta_df = _sta_df.loc[[s for s in STATION_ORDER if s in _sta_df.index]]
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


import math
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
    fig = plt.figure(figsize=(N_COLS * 2.4, N_ROWS * 3.0))
    gs = GridSpec(N_ROWS, N_COLS, hspace=0.25, wspace=0.08)

    for pi, (lbl, sta_phi) in enumerate(period_data):
        ri, ci = divmod(pi, N_COLS)
        ax = fig.add_subplot(gs[ri, ci])
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
        ax.set_title(f'{lbl}\nN={n_total:,}', fontsize=6.5, fontweight='bold', pad=2)

        # get_ax_inset() needs an up-to-date renderer/position, per its own docstring.
        fig.canvas.draw()

        for sta in STATION_ORDER:
            phi_vals = sta_phi[sta]
            sx, sy = float(_sta_df.loc[sta, 'x']), float(_sta_df.loc[sta, 'y'])
            ax_inset = swm.get_ax_inset(ax, sx, sy, width=ROSE_WIDTH_KM,
                                       projection='polar', alpha=0.75, visible_axis=False)
            _draw_rose(ax_inset, phi_vals, colors[pi])

    legend_handles = [
        mlines.Line2D([], [], marker='o', color='w', mfc='red', mec='k', ms=5, label='Kidiwela'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    return fig


def main():
    _setup_environment()

    print('Loading new mfast max_dt=0.2s splitting results (AXCC1, AXEC1, AXEC3, AXAS1: '
          'complete; AXEC2 2015-2021: partial, in-progress re-run)...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
    dfs = {sta: apply_tier(raw[sta], TIER_IDX) for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        print(f'  {sta}: {len(dfs[sta]):,} QC-passing events (tier {TIER_IDX})')

    periods = build_periods()

    period_data = []
    for lbl, t0, t1 in periods:
        sta_phi = {}
        for sta in STATION_ORDER:
            sub = _subset(dfs[sta], t0, t1)
            sta_phi[sta] = sub['phi_az'].values
        counts = {sta: len(v) for sta, v in sta_phi.items()}
        print(f'  {lbl.splitlines()[0]:24s}: {counts}')
        period_data.append((lbl, sta_phi))

    title = ('AXAS1/AXCC1/AXEC1/AXEC2/AXEC3 (new mfast max_dt=0.2s data) — per-station rose '
             'diagrams on bathymetry\n'
             f'6h pre-eruption + first {N_HOURS_ZOOM}h of eruption (hourly), '
             'no station markers\n'
             'quality>=0.5, dt<T_dom/2, phi_err<20deg, dt_err<0.05s')
    fig = make_page(period_data, title)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
