#!/usr/bin/env python3
"""
rose_baz_map_all_stations_7period_6stations_newdata_snr2_q75.py

Combines rose_by_depth_per_station_7period_6stations_newdata_snr2_q75_baz_ring.py's
depth-row x time-period rose+baz-ring grid with the bathymetry/fault map background used by
the dt_by_source_voxel_per_station_* / phi_quiver_by_source_voxel_per_station_* scripts, but
instead of one page per station (each highlighting its own triangle marker), this produces
ONE page total: 3 depth rows (0.0-0.75 / 0.75-1.5 / 1.5-2.25 km) x 7 eruption-relative
time-period columns, and within EACH cell, all 6 stations' rose+baz-ring diagrams (using
that station's own events for that depth bin x period, same grade-3 filter) are drawn
directly on the bathymetry at their true (x, y) station locations, with the depth-corrected
ring-fault overlay -- no station triangle markers/labels (the roses themselves mark each
station's location), and no polar-axis spine circle around any rose (clean overlay onto the
bathymetry). Same period color scheme (pre-eruption purple, syn-eruption red, blue->purple
post-eruption gradient) as the other rose scripts this session.

Single filter (same "Page 1 @ quality>=0.75" grade used throughout this session):
SNR >= 2.0, quality (Q_w) >= 0.75, dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg.

Produces:
    rose_baz_map_all_stations_7period_6stations_newdata_snr2_q75.pdf
(1 page: 3 depth rows x 7 time-period columns, all 6 stations' rose+baz-ring diagrams
overlaid on bathymetry+faults in every cell) -- a NEW file, does not touch any existing
output.

Run with:
    python3 rose_baz_map_all_stations_7period_6stations_newdata_snr2_q75.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile
from PIL import Image as PILImage
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_baz_map_all_stations_7period_6stations_newdata_snr2_q75.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg
assert GRADE['key'] == 'page1_q0.75', GRADE
DT_ERR_MAX = T.DT_ERR_MAX

DEPTH_ROWS = [(0.0, 0.75, '0.0–0.75 km', 0.375),
              (0.75, 1.5, '0.75–1.5 km', 1.125),
              (1.5, 2.25, '1.5–2.25 km', 1.875)]

NBINS = 36
BAZ_BINS = np.arange(0, 361, 60)
BAZ_CTRS = (BAZ_BINS[:-1] + BAZ_BINS[1:]) / 2

# Rose-glyph geometry, in km (data units of the bathymetry map) -- kept small relative to
# typical inter-station spacing (~1-4 km) so neighboring stations' roses don't overlap.
MAIN_R_KM = 0.336
RING_R_KM = 0.55
SMALL_R_KM = 0.192


def load_station_events(sta):
    dfs = [pd.read_csv(os.path.join(T.TRANSFER_DIR, f)) for f in T.STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['depth', 'back_azimuth', 'quality', 'dominant_period',
                            'snr_horizontal', 'phi_error', 'dt_error', 'phi'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')

    mask = ((df['quality'] >= GRADE['qw_min']) &
            (df['snr_horizontal'] >= GRADE['snr_min']) &
            (df['dt_error'] <= DT_ERR_MAX) &
            (df['dt'] <= df['dominant_period'] / 2.0) &
            (df['phi_error'] <= GRADE['phi_err_max']))
    df = df[mask].copy()
    df['phi_az'] = df['phi'] % 180.0
    df['back_azimuth'] = df['back_azimuth'] % 360.0
    return df[['t', 'phi_az', 'back_azimuth', 'depth']].reset_index(drop=True)


def _subset_period(df, t0, t1):
    m = np.ones(len(df), dtype=bool)
    if t0 is not None:
        m &= (df['t'].values >= np.datetime64(t0))
    if t1 is not None:
        m &= (df['t'].values < np.datetime64(t1))
    return df[m]


def _subset_depth(df, lo, hi):
    return df[(df['depth'].values >= lo) & (df['depth'].values < hi)]


def _period_colors(n_periods):
    colors = ['#800080', '#CC0000']
    n_post = n_periods - 2
    c0 = np.array(mcolors.to_rgb('#ADD8E6'))
    c1 = np.array(mcolors.to_rgb('#800080'))
    for i in range(n_post):
        f = i / max(n_post - 1, 1)
        colors.append(mcolors.to_hex(c0 + f * (c1 - c0)))
    return colors


def _draw_rose(ax, phi_az_vals, color):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('none')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(False)
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

    ax.bar(centers, counts, width=width, bottom=0,
           color='white', edgecolor='black', linewidth=0.4, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)


_rgb_color = None
_ext_color = None


def _setup_color_bathy():
    """Same crop/downsample as T._setup_environment, but keeping the RGB image (not
    converted to grayscale) so the bathymetry can be shown in color."""
    global _rgb_color, _ext_color

    bathy = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
             'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
    PILImage.MAX_IMAGE_PIXELS = None
    _p = PILImage.open(bathy)
    _tag = _p.tag_v2
    _olon, _olat = _tag[33922][3], _tag[33922][4]
    _pl, _pb = _tag[33550][0], _tag[33550][1]
    _nc, _nr = _p.size
    _p.close()
    _c0 = max(0, int(((T.INI_LON + T.X_START / T.KM_PER_DEG_LON) - _olon) / _pl) - 2)
    _c1 = min(_nc, int(((T.INI_LON + T.X_END / T.KM_PER_DEG_LON) - _olon) / _pl) + 2)
    _r0 = max(0, int((_olat - (T.INI_LAT + T.Y_END / T.KM_PER_DEG_LAT)) / _pb) - 2)
    _r1 = min(_nr, int((_olat - (T.INI_LAT + T.Y_START / T.KM_PER_DEG_LAT)) / _pb) + 2)
    _rgb = tifffile.imread(bathy)[_r0:_r1, _c0:_c1]
    _ds = max(1, max(_rgb.shape[:2]) // 1024)
    _rgb = _rgb[::_ds, ::_ds]
    _rgb_color = _rgb[..., :3]
    _ext_color = [(_olon + _c0 * _pl - T.INI_LON) * T.KM_PER_DEG_LON,
                  (_olon + _c1 * _pl - T.INI_LON) * T.KM_PER_DEG_LON,
                  (_olat - _r1 * _pb - T.INI_LAT) * T.KM_PER_DEG_LAT,
                  (_olat - _r0 * _pb - T.INI_LAT) * T.KM_PER_DEG_LAT]


def _bathy_faded(ax, alpha=0.4):
    ax.imshow(_rgb_color, origin='upper', extent=_ext_color, aspect='auto',
              alpha=alpha, zorder=0)


def _km_to_fig_frac(ax, fig, r_km):
    """Physical radius (km, in data units) -> (width_frac, height_frac) in figure
    coordinates, using this ax's own data<->display scale (aspect is already 'equal')."""
    bbox = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_w_in = bbox.width * fig_w_in
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    scale_in_per_km = ax_w_in / xr
    diameter_in = 2 * r_km * scale_in_per_km
    return diameter_in / fig_w_in, diameter_in / fig_h_in


def _place_polar(fig, ax, x_km, y_km, r_km):
    disp = ax.transData.transform((x_km, y_km))
    fx, fy = fig.transFigure.inverted().transform(disp)
    w, h = _km_to_fig_frac(ax, fig, r_km)
    return fig.add_axes([fx - w / 2, fy - h / 2, w, h], projection='polar')


def make_page(station_dfs, periods, title):
    n_rows = len(DEPTH_ROWS)
    n_cols = len(periods)
    colors = _period_colors(n_cols)

    panel_w_in = 3.0
    panel_h_in = panel_w_in * (T.Y_END - T.Y_START) / (T.X_END - T.X_START)
    fig = plt.figure(figsize=(n_cols * panel_w_in + 0.3, n_rows * panel_h_in + 0.5))
    gs = GridSpec(n_rows, n_cols, hspace=0.10, wspace=0.06, top=0.93, bottom=0.03,
                  left=0.02, right=0.99)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.985)

    for ri, (lo, hi, zlbl, z0) in enumerate(DEPTH_ROWS):
        for ci, (label, t0, t1) in enumerate(periods):
            ax = fig.add_subplot(gs[ri, ci])
            _bathy_faded(ax, alpha=0.4)
            T._faults(ax, z0)
            ax.set_xlim(T.X_START, T.X_END)
            ax.set_ylim(T.Y_START, T.Y_END)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            color = colors[ci]
            n_cell = 0
            for sta in T.STATIONS:
                if sta not in T.sta_xy:
                    continue
                sub = _subset_period(_subset_depth(station_dfs[sta], lo, hi), t0, t1)
                n_cell += len(sub)
                sx, sy = T.sta_xy[sta]

                ax_main = _place_polar(fig, ax, sx, sy, MAIN_R_KM)
                _draw_rose(ax_main, sub['phi_az'].values, color)

                for baz_lo, baz_hi, baz_c in zip(BAZ_BINS[:-1], BAZ_BINS[1:], BAZ_CTRS):
                    baz_rad = np.deg2rad(baz_c)
                    ox = sx + RING_R_KM * np.sin(baz_rad)
                    oy = sy + RING_R_KM * np.cos(baz_rad)
                    baz_mask = (sub['back_azimuth'].values >= baz_lo) & (sub['back_azimuth'].values < baz_hi)
                    sub_baz = sub[baz_mask]
                    ax_s = _place_polar(fig, ax, ox, oy, SMALL_R_KM)
                    _draw_rose(ax_s, sub_baz['phi_az'].values, color)

            if ri == 0:
                ax.set_title(f'{label}\nN={n_cell:,}', fontsize=7.5, fontweight='bold', pad=4)
            else:
                ax.set_title(f'N={n_cell:,}', fontsize=6.5, fontweight='bold', pad=2)
            if ci == 0:
                ax.set_ylabel(zlbl, fontsize=8, fontweight='bold')

    return fig


def main():
    T._setup_environment(need_tracer=False)
    _setup_color_bathy()

    print(f'Loading per-station events ({GRADE["label"]})...')
    station_dfs = {}
    for sta in T.STATIONS:
        df = load_station_events(sta)
        station_dfs[sta] = df
        print(f'  {sta}: {len(df):,} events pass the filter')

    all_t = pd.concat([station_dfs[sta]['t'] for sta in T.STATIONS], ignore_index=True)
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))

    n_total = sum(len(station_dfs[sta]) for sta in T.STATIONS)
    title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (newdata) — fast-direction rose + '
             f'back-azimuth ring, on bathymetry, by depth x time period (N={n_total:,})\n'
             f'{GRADE["label"]}')
    fig = make_page(station_dfs, periods, title)

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
