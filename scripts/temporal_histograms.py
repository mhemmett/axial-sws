#!/usr/bin/env python3
"""
temporal_histograms.py

Two-page combined PDF of 2D moving-window histograms, 2015-2026:
  Page 1 — delay time (dt)  for all 6 stations
  Page 2 — fast direction (phi) for all 6 stations

Layout mirrors rose_plots_temporal.py:
  Rows (top→bottom): AXAS2, AXAS1, AXCC1, AXEC1, AXEC2, AXEC3
  Section headers: "Western Caldera", "Central Caldera", "Eastern Caldera"
  Station names on the left side

Also produces filtered versions (dt_error < 0.1 s).
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from obspy import UTCDateTime

matplotlib.rcParams['font.family'] = 'Arial'

# ── Import Baillard's sws_methods ─────────────────────────────────────────────

sys.path.insert(0, '/Users/mhemmett/Seismology/Axial_Splitting/scripts')
import sws_methods as swm

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE    = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
OUT_DIR = BASE

ERUPTION_START = UTCDateTime(2015, 4, 24, 6)
ERUPTION_END   = UTCDateTime(2015, 5, 19, 0)
SAMPLING_RATE  = 200.0

STATION_ORDER = ['AXAS2', 'AXAS1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

SECTION_LABELS = {
    'AXAS2': 'Western Caldera',
    'AXCC1': 'Central Caldera',
    'AXEC1': 'Eastern Caldera',
}

FILES = {
    'AXAS2': ('splitting_results_mldd_2015_2021_axas2.csv',
              'splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXAS1': ('splitting_results_mldd_2015_2021_axas1.csv',
              'splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXCC1': ('splitting_results_mldd_2015_2021_axcc1_all_batches.csv',
              'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1': ('splitting_results_mldd_2015_2021_axec1_all_batches.csv',
              'splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2': ('axial-mldd-2015-2021-axec2.csv',
              'splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3': ('splitting_results_mldd_2015_2021_axec3_all_batches.csv',
              'splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

# ── Data loading ──────────────────────────────────────────────────────────────

def _load(path):
    df = pd.read_csv(BASE + path)
    df = df.loc[:, :'dt_error'].dropna()
    df = df[df['dt'] > 0]
    return df


def load_station(sta, dt_error_max=None):
    f1, f2 = FILES[sta]
    df = pd.concat([_load(f1), _load(f2)], ignore_index=True)
    if dt_error_max is not None:
        df = df[df['dt_error'] < dt_error_max]
    return df


# ── plot_movehisto2d ──────────────────────────────────────────────────────────

def plot_movehisto2d(x, y, x_label='X', y_label='Y', ax=None, vmax=None,
                     cmap='Blues', **kwargs):
    time_flag = isinstance(x[0], UTCDateTime)
    x_start = kwargs.get('x_start', None)
    x_end   = kwargs.get('x_end',   None)

    if time_flag:
        x       = np.array([v.timestamp for v in x])
        x_start = x_start.timestamp if x_start is not None else None
        x_end   = x_end.timestamp   if x_end   is not None else None
        kwargs['x_start'] = x_start
        kwargs['x_end']   = x_end

    X, Y, Z, x_bins, x_diffs = swm.movehisto2d_bin(x, y, **kwargs)

    if ax is None:
        _, ax = plt.subplots()

    if time_flag:
        X      = np.array(swm.timestamp2matplotlib(X.ravel())).reshape(X.shape)
        x_bins = swm.timestamp2matplotlib(x_bins)
        ax.set_xlim(swm.timestamp2matplotlib([x_start, x_end]))
    else:
        ax.set_xlim([x_start, x_end])

    Xm, Ym = swm.XY2XY_pcolormesh(X, Y)
    im = ax.pcolormesh(Xm, Ym, Z, cmap=cmap, rasterized=True, vmax=vmax)
    ax.set_ylim([kwargs.get('y_start', Y.min()), kwargs.get('y_end', Y.max())])
    ax.set_aspect('auto')
    if time_flag:
        ax.xaxis_date()
    ax.set_ylabel(y_label)
    return ax, im, X, Y, Z, x_bins, x_diffs


# ── Single-row histogram ──────────────────────────────────────────────────────

X_WIDTH   = 3 * 24 * 3600   # 3-day window
X_OVERLAP = 0.50             # 50% overlap
X_FILT    = 0.5              # 0.5% time smoothing
Y_FILT    = 3


def _draw_dt(ax, df):
    times   = pd.to_datetime(df['event_datetime'], utc=True)
    x       = np.array([UTCDateTime(t.isoformat()) for t in times])
    x_start = UTCDateTime(times.min().isoformat())
    x_end   = UTCDateTime(times.max().isoformat())
    y       = df['dt'].values * SAMPLING_RATE

    ax, im, *_ = plot_movehisto2d(
        x, y, ax=ax, cmap='Blues',
        x_width=X_WIDTH, x_start=x_start, x_end=x_end, x_over=X_OVERLAP,
        y_start=0, y_end=30, y_width=1, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=X_FILT, y_filter_per=Y_FILT, filter_mode='nearest',
    )
    for ex in [ERUPTION_START.matplotlib_date, ERUPTION_END.matplotlib_date]:
        ax.axvline(ex, color='black', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.plot(ex, 15, marker='*', color='black', markersize=12, zorder=5)
    ax.set_ylabel('Delay Time δt (samples at 200 Hz)', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=8)
    return im


def _draw_phi(ax, df):
    times   = pd.to_datetime(df['event_datetime'], utc=True)
    x       = np.array([UTCDateTime(t.isoformat()) for t in times])
    x_start = UTCDateTime(times.min().isoformat())
    x_end   = UTCDateTime(times.max().isoformat())
    # Convert swspy CW-from-N → CCW-from-E (Baillard convention)
    phi_raw  = df['phi'].values
    phi_plot = 90.0 - phi_raw
    phi_plot = np.where(phi_plot >  90, phi_plot - 180, phi_plot)
    phi_plot = np.where(phi_plot < -90, phi_plot + 180, phi_plot)

    ax, im, *_ = plot_movehisto2d(
        x, phi_plot, ax=ax, cmap='Reds',
        x_width=X_WIDTH, x_start=x_start, x_end=x_end, x_over=X_OVERLAP,
        y_start=-90, y_end=90, y_width=180/40, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=X_FILT, y_filter_per=Y_FILT,
        filter_mode=['wrap', 'nearest'],
    )
    for ex in [ERUPTION_START.matplotlib_date, ERUPTION_END.matplotlib_date]:
        ax.axvline(ex, color='black', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.plot(ex, 0, marker='*', color='black', markersize=12, zorder=5)
    ax.set_ylabel('Fast Direction φ (° CCW from E)', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=8)
    return im


# ── Page builder ──────────────────────────────────────────────────────────────

def make_page(param, dfs, title):
    """One page: 6 rows (stations), 1 histogram column, section labels."""
    n_rows     = len(STATION_ORDER)
    panel_h    = 1.6   # inches per row
    row_gap    = 0.10
    left_margin = 1.0
    top_margin  = 0.55
    right_margin = 0.85  # for colorbar
    section_gap = 0.45
    section_rows = {s: i for i, s in enumerate(STATION_ORDER) if s in SECTION_LABELS}

    fig_h = (n_rows * (panel_h + row_gap) + top_margin
             + len(section_rows) * section_gap)
    fig_w = 14.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.995)

    # Row y-positions (top of each cell, inches from bottom)
    row_tops = []
    y_cursor = fig_h - top_margin
    for sta in STATION_ORDER:
        if sta in SECTION_LABELS:
            y_cursor -= section_gap
        row_tops.append(y_cursor)
        y_cursor -= (panel_h + row_gap)

    draw_fn = _draw_dt if param == 'dt' else _draw_phi
    last_im = None

    for row_idx, sta in enumerate(STATION_ORDER):
        df   = dfs[sta]
        cell_top = row_tops[row_idx]

        x0 = left_margin / fig_w
        y0 = (cell_top - panel_h) / fig_h
        w  = (fig_w - left_margin - right_margin) / fig_w
        h  = panel_h / fig_h

        ax = fig.add_axes([x0, y0, w, h])
        last_im = draw_fn(ax, df)

        # x-axis: only show on last row
        if row_idx < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

        # N= count
        dt_mean = df['dt'].mean()
        ax.set_title(
            f'N = {len(df):,}  |  mean δt = {dt_mean:.3f} s  ({dt_mean*SAMPLING_RATE:.1f} samples)',
            fontsize=7.5, fontweight='bold', loc='right', pad=2,
        )

        # Station name on left
        y_center = (cell_top - panel_h / 2) / fig_h
        fig.text(0.0, y_center, sta, fontsize=11, fontweight='bold',
                 ha='left', va='center', transform=fig.transFigure)

        # Section label above row
        if sta in SECTION_LABELS:
            y_label = (cell_top + section_gap * 0.55) / fig_h
            fig.text(left_margin / fig_w, y_label, SECTION_LABELS[sta],
                     fontsize=13, fontweight='bold', color='black',
                     ha='left', va='bottom', transform=fig.transFigure,
                     fontfamily='Arial')

    # Shared colorbar on far right
    if last_im is not None:
        cax = fig.add_axes([1 - right_margin / fig_w + 0.01, 0.05, 0.02, 0.90])
        cb  = plt.colorbar(last_im, cax=cax)
        cb.set_label('Normalised Density', fontsize=9)
        cb.ax.tick_params(labelsize=8)

    return fig


# ── Run ───────────────────────────────────────────────────────────────────────

for suffix, dt_err_max in [('', None), ('_filtered', 0.1)]:
    label = f'(dt_error < {dt_err_max} s)' if dt_err_max else '(all data)'
    print(f'\nLoading data {label}...')
    dfs = {}
    for sta in STATION_ORDER:
        dfs[sta] = load_station(sta, dt_error_max=dt_err_max)
        print(f'  {sta}: {len(dfs[sta]):,} events')

    out_path = os.path.join(OUT_DIR, f'temporal_histograms{suffix}.pdf')
    with PdfPages(out_path) as pdf:
        for param, param_label in [('dt', 'Delay Time δt'),
                                    ('phi', 'Fast Direction φ')]:
            print(f'  Building {param_label} page...')
            fig = make_page(
                param, dfs,
                f'Axial Seamount — {param_label} Temporal Evolution 2015–2026 {label}',
            )
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'  Added {param_label} page')

    print(f'Saved {out_path}')

print('\nDone.')
