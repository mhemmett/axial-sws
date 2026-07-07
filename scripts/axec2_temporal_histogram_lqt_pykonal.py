#!/usr/bin/env python3
"""
AXEC2 temporal histogram (2D moving-window density of delay time / fast direction
vs. time), 2015-2021, using the new LQT + PyKonal-FMM production dataset
(production_axec2_lqt_pykonal_results/, 497/497 batches, 88,649 measurements) -
NOT the old P-Jurkevics-incidence results used by the legacy temporal_histograms.py
/ sws_mesh_plot_mldd.py (which are also hardcoded to a different repo's results/).

Style follows temporal_histograms.py's per-station rows (plot_movehisto2d, 3-day
moving window, 50% overlap, Gaussian-smoothed, eruption start/end markers), but:
  - phi uses this project's own established mod-180 azimuth convention (0-180 deg,
    CW from N) rather than Baillard's legacy 90-phi CCW-from-E convention.
  - dt is plotted in seconds directly (our pipeline already reports dt in seconds -
    no assumed sampling rate needed to convert from samples).

Output: axec2_temporal_histogram_lqt_pykonal_2015_2021.pdf (new name - does not
overwrite temporal_histograms.pdf / axec2_temporal_test.pdf / sws_temporal_AXEC2_*.pdf)
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime

import sws_methods as swm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'production_axec2_lqt_pykonal_results')
OUT_PATH = os.path.join(HERE, 'axec2_temporal_histogram_lqt_pykonal_2015_2021.pdf')

STATION = 'AXEC2'
ERUPTION_START = UTCDateTime(2015, 4, 24, 6)
ERUPTION_END = UTCDateTime(2015, 5, 19, 0)

X_WIDTH = 3 * 24 * 3600   # 3-day moving window
X_OVERLAP = 0.50
X_FILT = 0.5
Y_FILT = 3

DT_Y_END = 0.30  # s - covers the production run's observed dt range (max ~0.25s)


def load_combined_results():
    result_files = glob.glob(os.path.join(RESULTS_DIR, 'splitting_results_mldd_2015_2021_axec2_batch_*.csv'))
    dfs = [pd.read_csv(f) for f in result_files]
    dfs = [d for d in dfs if len(d) > 0]
    results = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    results = results[results['success'] == True].copy()
    results = results[results['dt'] > 0]
    results['phi_az'] = results['phi'] % 180.0
    return results


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
    y = df['phi_az'].values  # 0-180 deg, CW from N (this project's convention)

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


def main():
    print("Loading combined AXEC2 LQT + PyKonal-FMM production results...")
    df = load_combined_results()
    print(f"Total successful, non-null splitting measurements: {len(df):,}")

    fig, (ax_dt, ax_phi) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    fig.suptitle(f'{STATION} — LQT + PyKonal-FMM incidence (35° cut) — '
                 f'Temporal Evolution 2015–2021  (N={len(df):,})',
                 fontsize=12, fontweight='bold')

    im_dt = draw_dt(ax_dt, df)
    im_phi = draw_phi(ax_phi, df)

    ax_dt.set_title(f'mean $\\delta t$ = {df["dt"].mean():.3f} s', fontsize=8.5, loc='right')
    ax_phi.set_title(f'circular data, mod 180°', fontsize=8.5, loc='right')

    cax_dt = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cb_dt = plt.colorbar(im_dt, cax=cax_dt)
    cb_dt.set_label('Normalised density', fontsize=8)
    cb_dt.ax.tick_params(labelsize=7)

    cax_phi = fig.add_axes([0.92, 0.10, 0.015, 0.35])
    cb_phi = plt.colorbar(im_phi, cax=cax_phi)
    cb_phi.set_label('Normalised density', fontsize=8)
    cb_phi.ax.tick_params(labelsize=7)

    fig.subplots_adjust(right=0.90, hspace=0.08)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f"Saved {OUT_PATH}")


if __name__ == '__main__':
    main()
