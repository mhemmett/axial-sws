#!/usr/bin/env python3
"""
temporal_histograms_lqt_pykonal.py

Per-station 2D moving-window density histograms (delay time / fast direction
vs. time) for the LQT + PyKonal-FMM production run, generalizing
axec2_temporal_histogram_lqt_pykonal.py (which only covers AXEC2's
2015-2021 half) to all 6 stations using the full combined dataset in
lqt_pykonal_combined_results/ (2022-2026 for every station, plus 2015-2021
for AXCC1/AXEC1 there and AXEC2 from production_axec2_lqt_pykonal_results/,
same as rose_plots_lqt_pykonal_unweighted.py).

Style is identical to axec2_temporal_histogram_lqt_pykonal.py / the legacy
temporal_histograms.py (plot_movehisto2d, 3-day moving window, 50% overlap,
Gaussian-smoothed, eruption start/end markers), and to rose_plots_lqt_pykonal
_unweighted.py's phi convention (0-180 deg mod-180 azimuth, CW from N -- not
Baillard's legacy 90-phi CCW-from-E convention). dt is plotted in seconds
directly. No quality/error filtering is applied here (matches
axec2_temporal_histogram_lqt_pykonal.py's precedent: success==True, dt>0
only) -- these are raw density plots, not the QC'd rose/tomography figures.

Produces one 2-panel (dt, phi) PDF per station in lqt_pykonal_combined_results/:
    axas1_temporal_histogram_lqt_pykonal.pdf
    axas2_temporal_histogram_lqt_pykonal.pdf
    axcc1_temporal_histogram_lqt_pykonal.pdf
    axec1_temporal_histogram_lqt_pykonal.pdf
    axec2_temporal_histogram_lqt_pykonal.pdf
    axec3_temporal_histogram_lqt_pykonal.pdf

Run with:
    python3 temporal_histograms_lqt_pykonal.py
    python3 temporal_histograms_lqt_pykonal.py AXAS1 AXAS2   # only regenerate given stations
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime

import sws_methods as swm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_lqt_pykonal_results')
OUT_DIR = DATA_DIR

STATION_ORDER = ['AXAS2', 'AXAS1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

ERUPTION_START = UTCDateTime(2015, 4, 24, 6)
ERUPTION_END = UTCDateTime(2015, 5, 19, 0)

X_WIDTH = 3 * 24 * 3600   # 3-day moving window
X_OVERLAP = 0.50
X_FILT = 0.5
Y_FILT = 3

DT_Y_END = 0.30  # s

STATION_FILES = {
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv',
              'splitting_results_AXAS2_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
}


def load_station(sta):
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = glob.glob(os.path.join(
            AXEC2_2015_2021_DIR, 'splitting_results_mldd_2015_2021_axec2_batch_*.csv'))
        dfs.extend(pd.read_csv(f) for f in batch_files)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df['phi_az'] = df['phi'] % 180.0
    return df


def plot_movehisto2d(x, y, ax, cmap, **kwargs):
    time_flag = isinstance(x[0], UTCDateTime)
    x_start = kwargs.get('x_start', None)
    x_end = kwargs.get('x_end', None)

    if time_flag:
        x = np.array([v.timestamp for v in x])
        x_start = x_start.timestamp if x_start is not None else None
        x_end = x_end.timestamp if x_end is not None else None
        kwargs['x_start'] = x_start
        kwargs['x_end'] = x_end

    X, Y, Z, x_bins, x_diffs = swm.movehisto2d_bin(x, y, **kwargs)

    if time_flag:
        X = np.array(swm.timestamp2matplotlib(X.ravel())).reshape(X.shape)
        ax.set_xlim(swm.timestamp2matplotlib([x_start, x_end]))
    else:
        ax.set_xlim([x_start, x_end])

    Xm, Ym = swm.XY2XY_pcolormesh(X, Y)
    im = ax.pcolormesh(Xm, Ym, Z, cmap=cmap, rasterized=True)
    ax.set_ylim([kwargs.get('y_start', Y.min()), kwargs.get('y_end', Y.max())])
    ax.set_aspect('auto')
    if time_flag:
        ax.xaxis_date()
    return ax, im


def draw_dt(ax, df):
    times = pd.to_datetime(df['datetime'], utc=True)
    x = np.array([UTCDateTime(t.isoformat()) for t in times])
    x_start = UTCDateTime(times.min().isoformat())
    x_end = UTCDateTime(times.max().isoformat())
    y = df['dt'].values

    ax, im = plot_movehisto2d(
        x, y, ax=ax, cmap='Blues',
        x_width=X_WIDTH, x_start=x_start, x_end=x_end, x_over=X_OVERLAP,
        y_start=0, y_end=DT_Y_END, y_width=DT_Y_END / 30, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=X_FILT, y_filter_per=Y_FILT, filter_mode='nearest',
    )
    for ex in [ERUPTION_START.matplotlib_date, ERUPTION_END.matplotlib_date]:
        ax.axvline(ex, color='black', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.plot(ex, DT_Y_END * 0.5, marker='*', color='black', markersize=12, zorder=5)
    ax.set_ylabel(r'Delay time $\delta t$ [s]', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.set_xticklabels([])
    return im


def draw_phi(ax, df):
    times = pd.to_datetime(df['datetime'], utc=True)
    x = np.array([UTCDateTime(t.isoformat()) for t in times])
    x_start = UTCDateTime(times.min().isoformat())
    x_end = UTCDateTime(times.max().isoformat())
    y = df['phi_az'].values  # 0-180 deg, CW from N

    ax, im = plot_movehisto2d(
        x, y, ax=ax, cmap='Reds',
        x_width=X_WIDTH, x_start=x_start, x_end=x_end, x_over=X_OVERLAP,
        y_start=0, y_end=180, y_width=180 / 40, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=X_FILT, y_filter_per=Y_FILT,
        filter_mode=['wrap', 'nearest'],
    )
    for ex in [ERUPTION_START.matplotlib_date, ERUPTION_END.matplotlib_date]:
        ax.axvline(ex, color='black', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.plot(ex, 90, marker='*', color='black', markersize=12, zorder=5)
    ax.set_ylabel(r'Fast direction $\phi$ [° from N]', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    return im


def make_station_figure(sta, df):
    fig, (ax_dt, ax_phi) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    fig.suptitle(f'{sta} — LQT + PyKonal-FMM incidence (35° cut) — '
                 f'Temporal Evolution  (N={len(df):,})',
                 fontsize=12, fontweight='bold')

    im_dt = draw_dt(ax_dt, df)
    im_phi = draw_phi(ax_phi, df)

    ax_dt.set_title(f'mean $\\delta t$ = {df["dt"].mean():.3f} s', fontsize=8.5, loc='right')
    ax_phi.set_title('circular data, mod 180°', fontsize=8.5, loc='right')

    cax_dt = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cb_dt = plt.colorbar(im_dt, cax=cax_dt)
    cb_dt.set_label('Normalised density', fontsize=8)
    cb_dt.ax.tick_params(labelsize=7)

    cax_phi = fig.add_axes([0.92, 0.10, 0.015, 0.35])
    cb_phi = plt.colorbar(im_phi, cax=cax_phi)
    cb_phi.set_label('Normalised density', fontsize=8)
    cb_phi.ax.tick_params(labelsize=7)

    fig.subplots_adjust(right=0.90, hspace=0.08)
    return fig


def main():
    requested = [s.upper() for s in sys.argv[1:]]
    stations = [s for s in STATION_ORDER if s in requested] if requested else STATION_ORDER
    for sta in stations:
        print(f'Loading combined LQT + PyKonal-FMM results for {sta}...')
        df = load_station(sta)
        print(f'  {sta}: {len(df):,} successful, non-null splitting measurements')

        fig = make_station_figure(sta, df)
        out_path = os.path.join(OUT_DIR, f'{sta.lower()}_temporal_histogram_lqt_pykonal.pdf')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
