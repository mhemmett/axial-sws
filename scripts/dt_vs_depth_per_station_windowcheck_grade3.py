#!/usr/bin/env python3
"""
dt_vs_depth_per_station_windowcheck_grade3.py

Current-chain replacement for dt_vs_depth_per_station_newdata_snr2_q75.py: per-station 2D
histogram of delay time vs. source depth, same pcolormesh / LogNorm style, but built from the
production WINDOW-CHECK run under the Grade-3 cut (load_station_raw + apply_grade, as every
current figure is):

    SNR >= 2.0, Q_w >= 0.75, dt_err <= 0.05 s, dt <= T_dom/2, phi_err <= 20 deg

The previous version read the pre-windowcheck "newdata" transfer folder. AXEC2 2015-2021
hypocentres: the windowcheck CSV for that half carries none, so load_station_raw now fills them
from the raw-batch metadata (event_id + origin times verified identical); this script checks and
prints AXEC2's 2015-2021 vs 2022-2026 counts so the inclusion is visible.

dt is reported on the splitting grid search's 10 ms step, so the dt bins are centred on those
steps (DT_STEP_S) -- one row per value, no empty stripes. A white line marks the MEAN dt (+-1 SE)
in each depth bin with >= MIN_BIN_EVENTS events; with 10 ms quantisation the median can only jump
between adjacent steps, while the mean resolves sub-step shifts.

Produces: dt_vs_depth_per_station_windowcheck_grade3.pdf (6 pages, one station per page)
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import GRADE, load_station_raw, apply_grade
from rose_7period_6stations_newdata_snr_grades import STATION_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'dt_vs_depth_per_station_windowcheck_grade3.pdf')

N_DEPTH_BINS = 40
DT_STEP_S = 0.01   # splitting grid-search step; dt values are exact multiples
MIN_BIN_EVENTS = 20


def load_station_events(sta):
    df = apply_grade(load_station_raw(sta), GRADE)
    n_all = len(df)
    df = df.dropna(subset=['depth'])
    early = df['t'].dt.year < 2022
    print(f'  {sta}: {len(df):,} of {n_all:,} Grade-3 events have a depth '
          f'(2015-2021: {int(early.sum()):,}, 2022-2026: {int((~early).sum()):,})')
    return df[['depth', 'dt']].reset_index(drop=True)


def make_station_page(sta, df):
    depth = df['depth'].values
    dt = df['dt'].values

    fig, ax = plt.subplots(figsize=(7.5, 6))
    depth_edges = np.linspace(depth.min(), depth.max(), N_DEPTH_BINS + 1)
    dt_edges = np.arange(DT_STEP_S / 2.0, dt.max() + DT_STEP_S, DT_STEP_S)
    counts, xe, ye = np.histogram2d(depth, dt, bins=[depth_edges, dt_edges])
    im = ax.pcolormesh(xe, ye, counts.T, cmap='viridis',
                       norm=mcolors.LogNorm(vmin=1, vmax=max(counts.max(), 2)))
    fig.colorbar(im, ax=ax, label='Count')

    idx = np.digitize(depth, depth_edges) - 1
    mids, means, ses = [], [], []
    for b in range(N_DEPTH_BINS):
        sel = idx == b
        if sel.sum() >= MIN_BIN_EVENTS:
            mids.append(0.5 * (depth_edges[b] + depth_edges[b + 1]))
            means.append(dt[sel].mean())
            ses.append(dt[sel].std() / np.sqrt(sel.sum()))
    mids, means, ses = map(np.asarray, (mids, means, ses))
    ax.fill_between(mids, means - ses, means + ses, color='white', alpha=0.35, lw=0)
    ax.plot(mids, means, color='white', lw=2, marker='o', ms=3,
            path_effects=[pe.withStroke(linewidth=4, foreground='black')],
            label=f'Mean δt ± 1 SE per depth bin (≥ {MIN_BIN_EVENTS} events)')
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    ax.set_xlabel('Source depth below seafloor (km)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Delay time δt (s)', fontsize=10, fontweight='bold')
    fig.suptitle(f'{sta} — delay time vs. depth (N={len(df):,}, windowcheck run)\n'
                 f'{GRADE["label"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    print(f'Loading per-station events ({GRADE["label"]})...')
    with PdfPages(OUT_PDF) as pdf:
        for sta in STATION_ORDER:
            fig = make_station_page(sta, load_station_events(sta))
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
