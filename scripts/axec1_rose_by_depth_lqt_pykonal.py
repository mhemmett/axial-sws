#!/usr/bin/env python3
"""
axec1_rose_by_depth_lqt_pykonal.py

AXEC1-only fast-direction rose plot, all time periods pooled, split into 4
source-depth bins: 0-0.5, 0.5-1.25, 1.25-1.75, >1.75 km. Uses the new LQT +
PyKonal-FMM production results (lqt_pykonal_combined_results/), ALL data
(baseline cleanup only -- success==True, dt>0 -- no Q_w/phi_error/dt_error
quality threshold, per explicit request not to filter by quality here).

Event depths are recovered by joining to the MLdd catalogs
(data/mldd_catalog_2015_2021.csv, data/mldd_catalog_2022_2026.csv) on
(station, event_datetime), same join used in quality_vs_depth_axec1.py /
lqt_pykonal_tomography_backprojection.py.

Data: lqt_pykonal_combined_results/splitting_results_AXEC1_2015_2021_all_batches.csv
      + splitting_results_AXEC1_2022_2026_all_batches.csv

Both pages are expanded into a 4-row (depth bin) x 7-column (time period)
grid, using the same 7-period eruption-relative model used throughout this
session (rose_plots_lqt_pykonal_unweighted.py's _build_time_periods: pre-
eruption, syn-eruption, 5 equal-count post-eruption bins, computed from
AXEC1's own timestamps).

Page 2 adds 6 back-azimuth sub-roses (one per 60 deg BAZ bin) arranged in a
ring around each cell's central rose, following rose_plots_baz.py's
"6 back-azimuth sub-roses arranged in a ring around each central rose"
layout (there: station x period grid; here: depth-bin x period grid).
Back-azimuth (station -> event, deg from N) is computed via obspy
gps2dist_azimuth from the same catalog-joined event_lat/lon used for depth.

Output: lqt_pykonal_combined_results/axec1_rose_by_depth_lqt_pykonal.pdf
    page 1 — 4 depth rows x 7 time-period columns, central roses only
    page 2 — 4 depth rows x 7 time-period columns, each with 6 surrounding
             back-azimuth sub-roses

Run with:
    python3 axec1_rose_by_depth_lqt_pykonal.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from obspy.geodetics import gps2dist_azimuth

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
CATALOG_DIR = os.path.join(os.path.dirname(HERE), 'data')
OUT_DIR = DATA_DIR

STATION = 'AXEC1'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
STATION_FILES = [
    'splitting_results_AXEC1_2015_2021_all_batches.csv',
    'splitting_results_AXEC1_2022_2026_all_batches.csv',
]

DEPTH_BINS = [(0, 0.5), (0.5, 1.25), (1.25, 1.75), (1.75, None)]
DEPTH_LABELS = ['0–0.5 km', '0.5–1.25 km', '1.25–1.75 km', '>1.75 km']
NBINS = 36

# Back-azimuth bins: 6 x 60 deg bins (same convention as rose_plots_baz.py)
BAZ_BINS = np.arange(0, 361, 60)
BAZ_CTRS = (BAZ_BINS[:-1] + BAZ_BINS[1:]) / 2

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')


def _build_time_periods(df):
    """7 periods: pre, syn, 5 equal-count post-eruption bins (verbatim from
    rose_plots_lqt_pykonal_unweighted.py)."""
    post = df[df['t'] >= ERUPTION_END].copy()
    post_times = post['t'].sort_values().reset_index(drop=True)
    n = len(post_times)

    boundaries = [ERUPTION_END]
    for i in range(1, 5):
        idx = int(round(i * n / 5))
        idx = min(idx, n - 1)
        boundaries.append(post_times.iloc[idx])
    boundaries.append(None)

    def _fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else 'present'

    periods = [
        ('Pre-eruption', None, ERUPTION_START),
        ('Syn-eruption', ERUPTION_START, ERUPTION_END),
    ]
    for i in range(5):
        t0, t1 = boundaries[i], boundaries[i + 1]
        periods.append((f'{_fmt(t0)}\n– {_fmt(t1)}', t0, t1))

    return periods


def _subset_period(df, t_start, t_end):
    m = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
    if t_end is not None:
        m = m & (df['t'] < t_end)
    return df[m]


def _subset_depth(df, lo, hi):
    if hi is None:
        return df[df['event_depth'] >= lo]
    return df[(df['event_depth'] >= lo) & (df['event_depth'] < hi)]


def load_axec1():
    """ALL AXEC1 measurements, baseline cleanup only (success==True, dt>0) --
    no quality/error thresholding."""
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True)
    df['phi_az'] = df['phi'] % 180.0
    return df


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


def _period_colors(n_periods):
    colors = ['#800080', '#CC0000']  # pre-eruption (purple), syn (red)
    n_post = n_periods - 2
    start = np.array(mcolors.to_rgb('#ADD8E6'))
    end = np.array(mcolors.to_rgb('#800080'))
    for i in range(n_post):
        t = i / max(n_post - 1, 1)
        colors.append(mcolors.to_hex((1 - t) * start + t * end))
    return colors


def make_depth_period_page(df, time_periods, title):
    """Page 1: 4 depth rows x N time-period columns, central roses only."""
    n_rows = len(DEPTH_BINS)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_size = 1.8
    row_gap = 0.10
    left_margin = 0.9
    top_margin = 0.65

    fig_h = n_rows * (panel_size + row_gap) + top_margin
    fig_w = n_cols * panel_size + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.01)

    for row_idx, (lo, hi) in enumerate(DEPTH_BINS):
        df_depth = _subset_depth(df, lo, hi)
        cell_top = fig_h - top_margin - row_idx * (panel_size + row_gap)

        for col_idx, (label, t0, t1) in enumerate(time_periods):
            sub = _subset_period(df_depth, t0, t1)

            x0 = (left_margin + col_idx * panel_size) / fig_w
            y0 = (cell_top - panel_size) / fig_h
            w = panel_size / fig_w
            h = panel_size / fig_h

            ax = fig.add_axes([x0, y0, w, h], projection='polar')
            _draw_rose(ax, sub['phi_az'].values, colors[col_idx])

            if row_idx == 0:
                ax.set_title(f'{label}\nN={len(sub):,}', fontsize=6.5, fontweight='bold', pad=2)
            else:
                ax.set_title(f'N={len(sub):,}', fontsize=6.5, fontweight='bold', pad=2)

        y_center = (cell_top - panel_size / 2) / fig_h
        fig.text(0.0, y_center, DEPTH_LABELS[row_idx], fontsize=10, fontweight='bold',
                  ha='left', va='center', transform=fig.transFigure)

    return fig


def make_depth_period_baz_ring_page(df, time_periods, title):
    """Page 2: 4 depth rows x N time-period columns, each cell a central rose
    + 6 back-azimuth sub-roses (60 deg bins) arranged in a ring around it --
    layout adapted from rose_plots_baz.py's make_baz_page() (there: station x
    period grid; here: depth-bin x period grid)."""
    n_rows = len(DEPTH_BINS)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_w = 2.6
    panel_h = 2.6
    row_gap = 0.15
    left_margin = 1.0
    top_margin = 0.65

    panel_scale = min(panel_w, panel_h)
    main_r = panel_scale * 0.18
    ring_r_h = panel_scale * 0.35
    ring_r_v = panel_scale * 0.42
    small_r = panel_scale * 0.12

    fig_h = n_rows * (panel_h + row_gap) + top_margin
    fig_w = n_cols * panel_w + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.005)

    for row_idx, (lo, hi) in enumerate(DEPTH_BINS):
        df_depth = _subset_depth(df, lo, hi)
        cell_top = fig_h - top_margin - row_idx * (panel_h + row_gap)
        cy_in = cell_top - panel_h / 2

        for col_idx, (label, t0, t1) in enumerate(time_periods):
            sub = _subset_period(df_depth, t0, t1)
            color = colors[col_idx]
            cx_in = left_margin + (col_idx + 0.5) * panel_w

            ax_main = fig.add_axes(
                [(cx_in - main_r) / fig_w, (cy_in - main_r) / fig_h,
                 2 * main_r / fig_w, 2 * main_r / fig_h],
                projection='polar')
            _draw_rose(ax_main, sub['phi_az'].values, color)
            if row_idx == 0:
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

                baz_mask = (sub['back_azimuth'] >= baz_lo) & (sub['back_azimuth'] < baz_hi)
                sub_baz = sub[baz_mask]
                _draw_rose(ax_s, sub_baz['phi_az'].values, color)
                ax_s.set_title(f'{int(baz_lo)}–{int(baz_hi)}°\nN={len(sub_baz):,}',
                               fontsize=3.8, pad=1)

        y_center = (cell_top - panel_h / 2) / fig_h
        fig.text(0.0, y_center, DEPTH_LABELS[row_idx], fontsize=10, fontweight='bold',
                  ha='left', va='center', transform=fig.transFigure)

    return fig


def main():
    print(f'Loading {STATION} LQT + PyKonal-FMM combined results (all data, baseline cleanup only)...')
    df = load_axec1()
    print(f'  {len(df):,} measurements')

    print('Loading MLdd catalogs for event depths + locations...')
    cat1 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'))
    cat2 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'))
    cat1['event_datetime'] = pd.to_datetime(cat1['event_datetime'], utc=True, format='mixed')
    cat2['event_datetime'] = pd.to_datetime(cat2['event_datetime'], utc=True, format='mixed')
    catalog = pd.concat([cat1, cat2], ignore_index=True)
    sta_cat = catalog[catalog['station'] == STATION].drop_duplicates(subset='event_datetime')

    df = df.merge(sta_cat[['event_datetime', 'event_depth', 'event_lat', 'event_lon']],
                  left_on='t', right_on='event_datetime', how='left')
    n_unmatched = df['event_depth'].isna().sum()
    if n_unmatched:
        print(f'  Dropping {n_unmatched:,} measurements with no catalog location match')
    df = df.dropna(subset=['event_depth', 'event_lat', 'event_lon'])
    print(f'  {len(df):,} measurements with recovered event depth + location')

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                           engine='python').set_index('s')
    sta_lon, sta_lat = float(_sta_df.loc[STATION, 'lon']), float(_sta_df.loc[STATION, 'lat'])

    print('Computing back-azimuth (station -> event)...')
    df['back_azimuth'] = [
        gps2dist_azimuth(sta_lat, sta_lon, elat, elon)[1]
        for elat, elon in zip(df['event_lat'], df['event_lon'])
    ]

    time_periods = _build_time_periods(df)

    out_path = os.path.join(OUT_DIR, 'axec1_rose_by_depth_lqt_pykonal.pdf')
    with PdfPages(out_path) as pdf:
        print('Page 1: depth-bin x time-period roses...')
        fig1 = make_depth_period_page(
            df, time_periods,
            f'{STATION} — LQT + PyKonal-FMM fast direction rose by source depth x time period (all data, N={len(df):,})',
        )
        pdf.savefig(fig1, dpi=300, bbox_inches='tight')
        plt.close(fig1)

        print('Page 2: depth-bin x time-period roses with 6 back-azimuth sub-roses...')
        fig2 = make_depth_period_baz_ring_page(
            df, time_periods,
            f'{STATION} — fast direction rose by depth x time period, with 6 back-azimuth sub-roses (all data, N={len(df):,})',
        )
        pdf.savefig(fig2, dpi=300, bbox_inches='tight')
        plt.close(fig2)

    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
