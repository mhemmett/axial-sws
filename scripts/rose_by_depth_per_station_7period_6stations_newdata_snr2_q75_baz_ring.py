#!/usr/bin/env python3
"""
rose_by_depth_per_station_7period_6stations_newdata_snr2_q75_baz_ring.py

Updated version of rose_by_depth_per_station_7period_6stations_newdata_snr2_q75.py's
per-station, depth-binned rose grid (3 depth rows: 0.0-0.75 / 0.75-1.5 / 1.5-2.25 km, x 7
eruption-relative time-period columns, one page per station), with each cell's central rose
now surrounded by 6 back-azimuth sub-roses (60 deg bins) arranged in a ring, following
axec1_rose_by_depth_lqt_pykonal.py's make_depth_period_baz_ring_page() / this session's
rose_7period_6stations_newdata_snr2_q75_baz_ring.py ring layout.

Same single filter (no grade sweep): SNR >= 2.0, quality (Q_w) >= 0.75, dt_err <= 0.05s,
dt <= T_dom/2, phi_err <= 20 deg. Back-azimuth (station -> event, deg from N) is taken
directly from the catalog's own 'back_azimuth' column.

Produces:
    rose_by_depth_per_station_7period_6stations_newdata_snr2_q75_baz_ring.pdf
(6 pages, one station per page: 3 depth rows x 7 time-period columns, central rose + 6 baz
sub-roses per cell) -- a NEW file, does not touch any existing output.

Run with:
    python3 rose_by_depth_per_station_7period_6stations_newdata_snr2_q75_baz_ring.py
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
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
OUT_PDF = os.path.join(
    HERE, 'rose_by_depth_per_station_7period_6stations_newdata_snr2_q75_baz_ring.pdf')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STATION_FILES = {
    sta: [f'splitting_results_{sta}_2015_2021_all_batches.csv',
          f'splitting_results_{sta}_2022_2026_all_batches.csv']
    for sta in STATIONS
}

GRADE = dict(label='SNR >= 2.0, quality >= 0.75, dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg',
             snr_min=2.0, phi_err_max=20.0, qw_min=0.75)
DT_ERR_MAX = 0.05

DEPTH_ROWS = [(0.0, 0.75, '0.0–0.75 km'), (0.75, 1.5, '0.75–1.5 km'), (1.5, 2.25, '1.5–2.25 km')]

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

NBINS = 36

# Back-azimuth ring bins: 6 x 60 deg bins (same convention as this session's other baz-ring scripts)
BAZ_BINS = np.arange(0, 361, 60)
BAZ_CTRS = (BAZ_BINS[:-1] + BAZ_BINS[1:]) / 2


def load_station_events(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
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


def build_7_periods(all_t):
    post = all_t[all_t >= ERUPTION_END].sort_values().reset_index(drop=True)
    n = len(post)
    bounds = [ERUPTION_END]
    for i in range(1, 5):
        bounds.append(post.iloc[min(int(round(i * n / 5)), n - 1)])
    bounds.append(None)

    def fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else 'present'

    periods = [('Pre-eruption', None, ERUPTION_START), ('Syn-eruption', ERUPTION_START, ERUPTION_END)]
    for i in range(5):
        periods.append((f'{fmt(bounds[i])}\n– {fmt(bounds[i + 1])}', bounds[i], bounds[i + 1]))
    return periods


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
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(0.8)
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
           color=color, edgecolor='black', linewidth=0.5, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)


def make_page(sta, df_sta, periods, title):
    n_rows = len(DEPTH_ROWS)
    n_cols = len(periods)
    colors = _period_colors(n_cols)

    panel_w = 2.6
    panel_h = 2.6
    row_gap = 0.15
    left_margin = 1.0
    top_margin = 0.75

    panel_scale = min(panel_w, panel_h)
    main_r = panel_scale * 0.18
    ring_r_h = panel_scale * 0.35
    ring_r_v = panel_scale * 0.42
    small_r = panel_scale * 0.12

    fig_h = n_rows * (panel_h + row_gap) + top_margin
    fig_w = n_cols * panel_w + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.005)

    for ri, (lo, hi, zlbl) in enumerate(DEPTH_ROWS):
        d_sub = _subset_depth(df_sta, lo, hi)
        cell_top = fig_h - top_margin - ri * (panel_h + row_gap)
        cy_in = cell_top - panel_h / 2

        for ci, (label, t0, t1) in enumerate(periods):
            sub = _subset_period(d_sub, t0, t1)
            color = colors[ci]
            cx_in = left_margin + (ci + 0.5) * panel_w

            ax_main = fig.add_axes(
                [(cx_in - main_r) / fig_w, (cy_in - main_r) / fig_h,
                 2 * main_r / fig_w, 2 * main_r / fig_h],
                projection='polar')
            _draw_rose(ax_main, sub['phi_az'].values, color)
            if ri == 0:
                ax_main.set_title(f'{label}\nN={len(sub):,}', fontsize=6.5, fontweight='bold', pad=2)
            else:
                ax_main.set_title(f'N={len(sub):,}', fontsize=5.5, fontweight='bold', pad=1)

            for baz_lo, baz_hi, baz_c in zip(BAZ_BINS[:-1], BAZ_BINS[1:], BAZ_CTRS):
                baz_rad = np.deg2rad(baz_c)
                sc_cx = cx_in + ring_r_h * np.sin(baz_rad)
                sc_cy = cy_in + ring_r_v * np.cos(baz_rad)

                ax_s = fig.add_axes(
                    [(sc_cx - small_r) / fig_w, (sc_cy - small_r) / fig_h,
                     2 * small_r / fig_w, 2 * small_r / fig_h],
                    projection='polar')

                baz_mask = (sub['back_azimuth'].values >= baz_lo) & (sub['back_azimuth'].values < baz_hi)
                sub_baz = sub[baz_mask]
                _draw_rose(ax_s, sub_baz['phi_az'].values, color)
                ax_s.set_title(f'{int(baz_lo)}–{int(baz_hi)}°\nN={len(sub_baz):,}',
                               fontsize=3.8, pad=1)

        y_center = (cell_top - panel_h / 2) / fig_h
        fig.text(0.0, y_center, zlbl, fontsize=10, fontweight='bold',
                  ha='left', va='center', transform=fig.transFigure)

    return fig


def main():
    print(f'Loading per-station events ({GRADE["label"]})...')
    station_dfs = {}
    for sta in STATIONS:
        df = load_station_events(sta)
        station_dfs[sta] = df
        print(f'  {sta}: {len(df):,} events pass the filter')

    all_t = pd.concat([station_dfs[sta]['t'] for sta in STATIONS], ignore_index=True)
    periods = build_7_periods(all_t)

    with PdfPages(OUT_PDF) as pdf:
        for sta in STATIONS:
            df = station_dfs[sta]
            title = (f'Fast-Direction Rose by Depth, with Back-Azimuth Ring — {sta}\n'
                     f'{GRADE["label"]}')
            fig = make_page(sta, df, periods, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
