#!/usr/bin/env python3
"""
baz_incidence_depth_temporal_histograms.py

Per-station 2D moving-window density histograms (back-azimuth, incidence
angle, and source depth vs. time) for all 6 stations, 2015-2026 -- checks
whether the volume of rock being sampled by the raypaths drifts or varies
significantly over the 10-year record, which could confound the observed
temporal changes in the splitting observables (dt, phi) themselves. This is
the same check Illsley-Kemp et al. (2018) ran at Afar ("we examine temporal
variations in the raypath properties; back azimuth, source depth, and
incidence angle... to ensure that any temporal variations we observe are not
originating from a change in the volume being sampled").

Modeled directly on temporal_histograms_lqt_pykonal.py (plot_movehisto2d,
3-day moving window, 50% overlap, Gaussian-smoothed, eruption start/end
markers) but with 3 panels instead of 2, and reading the geometry-enriched
CSVs produced by enrich_lqt_pykonal_results_geometry.py (back_azimuth,
incidence, event_depth columns) rather than the bare production CSVs.

Population: success==True only (no dt>0 cut) -- unlike the dt/phi temporal
histograms, null measurements still sample a real raypath/volume, so
excluding them here would bias the geometry check itself.

Produces one 3-panel PDF per station in lqt_pykonal_combined_results/:
    axas1_baz_incidence_depth_temporal_histogram.pdf
    ... (one per station)

Run with:
    python3 baz_incidence_depth_temporal_histograms.py
    python3 baz_incidence_depth_temporal_histograms.py AXAS1 AXAS2   # subset
"""

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
OUT_DIR = DATA_DIR

STATION_ORDER = ['AXAS2', 'AXAS1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

ERUPTION_START = UTCDateTime(2015, 4, 24, 6)
ERUPTION_END = UTCDateTime(2015, 5, 19, 0)

X_WIDTH = 3 * 24 * 3600   # 3-day moving window
X_OVERLAP = 0.50
X_FILT = 0.5
Y_FILT = 3

STATION_FILES = {
    sta: [f'splitting_results_{sta}_2015_2021_all_batches_enriched.csv',
          f'splitting_results_{sta}_2022_2026_all_batches_enriched.csv']
    for sta in STATION_ORDER
}


def load_station(sta):
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df.dropna(subset=['back_azimuth', 'incidence', 'event_depth'])
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


def _time_x(df):
    times = pd.to_datetime(df['datetime'], utc=True)
    x = np.array([UTCDateTime(t.isoformat()) for t in times])
    x_start = UTCDateTime(times.min().isoformat())
    x_end = UTCDateTime(times.max().isoformat())
    return x, x_start, x_end


def draw_baz(ax, df):
    x, x_start, x_end = _time_x(df)
    y = df['back_azimuth'].values

    ax, im = plot_movehisto2d(
        x, y, ax=ax, cmap='Purples',
        x_width=X_WIDTH, x_start=x_start, x_end=x_end, x_over=X_OVERLAP,
        y_start=0, y_end=360, y_width=360 / 40, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=X_FILT, y_filter_per=Y_FILT,
        filter_mode=['wrap', 'nearest'],
    )
    for ex in [ERUPTION_START.matplotlib_date, ERUPTION_END.matplotlib_date]:
        ax.axvline(ex, color='black', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.plot(ex, 180, marker='*', color='black', markersize=12, zorder=5)
    ax.set_ylabel('Back-azimuth\n[° from N]', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.set_xticklabels([])
    return im


def draw_incidence(ax, df):
    x, x_start, x_end = _time_x(df)
    y = df['incidence'].values

    ax, im = plot_movehisto2d(
        x, y, ax=ax, cmap='Greens',
        x_width=X_WIDTH, x_start=x_start, x_end=x_end, x_over=X_OVERLAP,
        y_start=0, y_end=90, y_width=90 / 40, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=X_FILT, y_filter_per=Y_FILT, filter_mode='nearest',
    )
    for ex in [ERUPTION_START.matplotlib_date, ERUPTION_END.matplotlib_date]:
        ax.axvline(ex, color='black', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.plot(ex, 45, marker='*', color='black', markersize=12, zorder=5)
    ax.axhline(35, color='red', lw=0.8, ls=':', alpha=0.7)
    ax.set_ylabel('Incidence angle\n[° from vertical]', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.set_xticklabels([])
    return im


def draw_depth(ax, df):
    x, x_start, x_end = _time_x(df)
    y = df['event_depth'].values
    y_end = float(np.nanpercentile(y, 99.5))

    ax, im = plot_movehisto2d(
        x, y, ax=ax, cmap='Oranges',
        x_width=X_WIDTH, x_start=x_start, x_end=x_end, x_over=X_OVERLAP,
        y_start=0, y_end=y_end, y_width=y_end / 40, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=X_FILT, y_filter_per=Y_FILT, filter_mode='nearest',
    )
    for ex in [ERUPTION_START.matplotlib_date, ERUPTION_END.matplotlib_date]:
        ax.axvline(ex, color='black', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.plot(ex, y_end * 0.5, marker='*', color='black', markersize=12, zorder=5)
    ax.set_ylabel('Source depth\n[km]', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    return im


def make_station_figure(sta, df):
    fig, (ax_baz, ax_inc, ax_dep) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'{sta} — raypath sampling geometry vs. time  (N={len(df):,})',
                 fontsize=12, fontweight='bold')

    im_baz = draw_baz(ax_baz, df)
    im_inc = draw_incidence(ax_inc, df)
    im_dep = draw_depth(ax_dep, df)

    ax_inc.set_title('dotted red = 35° QC cutoff', fontsize=8.5, loc='right')

    cax_baz = fig.add_axes([0.92, 0.68, 0.015, 0.22])
    cb_baz = plt.colorbar(im_baz, cax=cax_baz)
    cb_baz.set_label('Normalised density', fontsize=8)
    cb_baz.ax.tick_params(labelsize=7)

    cax_inc = fig.add_axes([0.92, 0.38, 0.015, 0.22])
    cb_inc = plt.colorbar(im_inc, cax=cax_inc)
    cb_inc.set_label('Normalised density', fontsize=8)
    cb_inc.ax.tick_params(labelsize=7)

    cax_dep = fig.add_axes([0.92, 0.08, 0.015, 0.22])
    cb_dep = plt.colorbar(im_dep, cax=cax_dep)
    cb_dep.set_label('Normalised density', fontsize=8)
    cb_dep.ax.tick_params(labelsize=7)

    fig.subplots_adjust(right=0.90, hspace=0.08)
    return fig


def main():
    requested = [s.upper() for s in sys.argv[1:]]
    stations = [s for s in STATION_ORDER if s in requested] if requested else STATION_ORDER
    for sta in stations:
        print(f'Loading geometry-enriched results for {sta}...')
        df = load_station(sta)
        print(f'  {sta}: {len(df):,} successful measurements with geometry')

        fig = make_station_figure(sta, df)
        out_path = os.path.join(OUT_DIR, f'{sta.lower()}_baz_incidence_depth_temporal_histogram.pdf')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
