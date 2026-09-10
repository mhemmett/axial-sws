#!/usr/bin/env python3
"""
rose_by_depth_per_station_7period_6stations_newdata_snr2_q75.py

Per-station, depth-binned rose plot for the six-station "newdata" dataset
(AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3 from
mfast_maxdt_pipeline_transfer/splitting_results_{STA}_{2015_2021,2022_2026}_all_batches.csv),
combining:
  - the 7-period rose-drawing/grid layout from rose_7period_6stations_newdata_snr_grades.py
    (_draw_rose, _period_colors, fig.add_axes(..., projection='polar') grid placement), and
  - the 3-row source-depth binning (0.0-0.75 / 0.75-1.5 / 1.5-2.25 km) used by the
    dt_by_source_voxel_per_station_* voxel scripts,
into ONE page per station (6 pages total): rows = depth bin, columns = the same 7
eruption-relative time periods.

Single filter (no grade sweep): SNR >= 2.0, quality (Q_w) >= 0.75, dt_err <= 0.05s,
dt <= T_dom/2, phi_err <= 20 deg -- i.e. the same "Page 1 @ quality>=0.75" grade used by
dt_by_source_voxel_per_station_7period_6stations_newdata_snr2_q75_countmin10_truemax.py.

Fast direction phi is folded to 0-180 deg (axial/undirected quantity) and doubled at
phi/phi+180 when histogrammed, per the existing rose-plot convention, so each rose is
symmetric across the full circle.

Produces:
    rose_by_depth_per_station_7period_6stations_newdata_snr2_q75.pdf
(6 pages, one station per page) -- a NEW file, does not touch any existing output.

Run with:
    python3 rose_by_depth_per_station_7period_6stations_newdata_snr2_q75.py
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
OUT_PDF = os.path.join(HERE, 'rose_by_depth_per_station_7period_6stations_newdata_snr2_q75.pdf')

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


def load_station_events(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['depth', 'quality', 'dominant_period', 'snr_horizontal',
                            'phi_error', 'dt_error', 'phi'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')

    mask = ((df['quality'] >= GRADE['qw_min']) &
            (df['snr_horizontal'] >= GRADE['snr_min']) &
            (df['dt_error'] <= DT_ERR_MAX) &
            (df['dt'] <= df['dominant_period'] / 2.0) &
            (df['phi_error'] <= GRADE['phi_err_max']))
    df = df[mask].copy()
    df['phi_az'] = df['phi'] % 180.0
    return df[['t', 'phi_az', 'depth']].reset_index(drop=True)


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


def _subset(df, t0, t1):
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

    panel_size = 1.8
    row_gap = 0.10
    left_margin = 0.95
    top_margin = 0.75

    fig_h = n_rows * (panel_size + row_gap) + top_margin
    fig_w = n_cols * panel_size + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.998)

    row_tops = []
    y_cursor = fig_h - top_margin
    for _ in range(n_rows):
        row_tops.append(y_cursor)
        y_cursor -= (panel_size + row_gap)

    col_lefts = [left_margin + ci * panel_size for ci in range(n_cols)]

    for ri, (lo, hi, zlbl) in enumerate(DEPTH_ROWS):
        cell_top = row_tops[ri]
        d_sub = _subset_depth(df_sta, lo, hi)

        for ci, (label, t0, t1) in enumerate(periods):
            sub = _subset(d_sub, t0, t1)

            x0 = col_lefts[ci] / fig_w
            y0 = (cell_top - panel_size) / fig_h
            w = panel_size / fig_w
            h = panel_size / fig_h

            ax = fig.add_axes([x0, y0, w, h], projection='polar')
            _draw_rose(ax, sub['phi_az'].values, colors[ci])

            n_events = len(sub)
            if ri == 0:
                ax.set_title(f'{label}\nN={n_events:,}', fontsize=6.5, fontweight='bold', pad=2)
            else:
                ax.set_title(f'N={n_events:,}', fontsize=6.5, fontweight='bold', pad=2)

        y_center = (cell_top - panel_size / 2) / fig_h
        fig.text(0.0, y_center, zlbl, fontsize=9, fontweight='bold',
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
            title = f'Fast-Direction Rose by Depth — {sta}\n{GRADE["label"]}'
            fig = make_page(sta, df, periods, title)
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
