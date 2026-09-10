#!/usr/bin/env python3
"""
dt_vs_depth_binned_5stations_newdata.py

Delay time (dt) vs. event depth for the 5 stations with new mfast try_filters +
max_t_shift_s=0.2s splitting results (AXAS1, AXCC1, AXEC1, AXEC2, AXEC3 -- same dataset/QC as
dt_and_pct_aniso_vs_depth_5stations_newdata.py; AXAS2 excluded, no new-data rerun exists for it
yet). Reuses that script's load_station_qc() loader (same QC: quality>=0.5, dt<T_dom/2,
phi_error<20 deg, dt_error<0.05s).

Per the user's request, replaces the hexbin density plot with:
  - Individual events plotted as black dots.
  - Depth binned into 0.25 km bins; per bin, mean dt +/- standard error of the mean
    (std/sqrt(n)) as an error bar.
  - A linear (dt vs. depth) least-squares fit overlaid per station.
  - A second, combined figure pooling all 5 stations' events into one panel with the same
    treatment.

Produces:
    dt_vs_depth_binned_5stations_newdata.pdf   (1 row x 5 station panels)
    dt_vs_depth_binned_combined_newdata.pdf    (1 panel, all stations pooled)

Run with:
    python3 dt_vs_depth_binned_5stations_newdata.py
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt

from dt_and_pct_aniso_vs_depth_5stations_newdata import (
    load_station_qc, STATION_ORDER, QW_MIN, PHI_ERR_MAX, DT_ERR_MAX,
)

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PER_STATION_PDF = os.path.join(HERE, 'dt_vs_depth_binned_5stations_newdata.pdf')
OUT_COMBINED_PDF = os.path.join(HERE, 'dt_vs_depth_binned_combined_newdata.pdf')

DEPTH_BIN_KM = 0.25


def bin_and_fit(depth, dt):
    """Return (bin_centers, bin_means, bin_sems, bin_counts, slope, intercept)."""
    depth = np.asarray(depth, dtype=float)
    dt = np.asarray(dt, dtype=float)

    d_min = np.floor(depth.min() / DEPTH_BIN_KM) * DEPTH_BIN_KM
    d_max = np.ceil(depth.max() / DEPTH_BIN_KM) * DEPTH_BIN_KM
    edges = np.arange(d_min, d_max + DEPTH_BIN_KM, DEPTH_BIN_KM)
    bin_idx = np.digitize(depth, edges) - 1

    centers, means, sems, counts = [], [], [], []
    for b in range(len(edges) - 1):
        m = bin_idx == b
        n = m.sum()
        if n == 0:
            continue
        vals = dt[m]
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        counts.append(n)

    centers = np.array(centers)
    means = np.array(means)
    sems = np.array(sems)
    counts = np.array(counts)

    slope, intercept = np.polyfit(depth, dt, 1)

    return centers, means, sems, counts, slope, intercept


def draw_panel(ax, depth, dt, title):
    ax.scatter(depth, dt, s=3, color='black', alpha=0.15, linewidths=0, rasterized=True)

    centers, means, sems, counts, slope, intercept = bin_and_fit(depth, dt)
    ax.errorbar(centers, means, yerr=sems, fmt='o', color='#CC0000', ecolor='#CC0000',
                markersize=4, capsize=2, linewidth=1.0, zorder=5,
                label=f'Mean dt / {DEPTH_BIN_KM} km bin ± SEM')

    x_line = np.array([depth.min(), depth.max()])
    ax.plot(x_line, slope * x_line + intercept, color='#1f77b4', linewidth=1.5, zorder=6,
            label=f'Linear fit: dt = {slope:.4f}×depth + {intercept:.4f}')

    ax.set_xlabel('Event depth [km]', fontsize=9)
    ax.set_title(f'{title}\nN={len(depth):,}', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, loc='upper right')
    return slope, intercept


def main():
    print('Loading + QC-ing new mfast max_dt=0.2s splitting results '
          '(AXCC1, AXEC1, AXEC3, AXAS1: complete; AXEC2 2015-2021: partial, in-progress re-run)...')
    station_events = {}
    for sta in STATION_ORDER:
        df = load_station_qc(sta)
        station_events[sta] = df
        print(f'  {sta}: {len(df):,} QC-passing events')

    # ── per-station panels ───────────────────────────────────────────────────
    n_sta = len(STATION_ORDER)
    fig1, axes1 = plt.subplots(1, n_sta, figsize=(4.2 * n_sta, 4.2), sharey=True)
    for ax, sta in zip(axes1, STATION_ORDER):
        df = station_events[sta]
        slope, intercept = draw_panel(ax, df['depth'].values, df['dt'].values, sta)
        print(f'  {sta}: linear fit slope={slope:.4f} s/km, intercept={intercept:.4f} s')
    axes1[0].set_ylabel(r'$\delta t$ [s]', fontsize=10, fontweight='bold')
    fig1.suptitle(f'Delay time vs. event depth, {DEPTH_BIN_KM} km bins (new mfast max_dt=0.2s data)\n'
                  f'quality>={QW_MIN}, dt<T_dom/2, phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s',
                  fontsize=12, fontweight='bold')
    fig1.tight_layout(rect=[0, 0, 1, 0.88])
    fig1.savefig(OUT_PER_STATION_PDF, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'\nSaved {OUT_PER_STATION_PDF}')

    # ── combined (all stations pooled) ──────────────────────────────────────
    combined = pd.concat(station_events.values(), ignore_index=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    slope, intercept = draw_panel(ax2, combined['depth'].values, combined['dt'].values,
                                   'All 5 stations combined')
    ax2.set_ylabel(r'$\delta t$ [s]', fontsize=10, fontweight='bold')
    print(f'  Combined: linear fit slope={slope:.4f} s/km, intercept={intercept:.4f} s')
    fig2.suptitle(f'Delay time vs. event depth, {DEPTH_BIN_KM} km bins, all stations combined\n'
                  f'(new mfast max_dt=0.2s data) quality>={QW_MIN}, dt<T_dom/2, '
                  f'phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s',
                  fontsize=11, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.88])
    fig2.savefig(OUT_COMBINED_PDF, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'\nSaved {OUT_COMBINED_PDF}')


if __name__ == '__main__':
    main()
