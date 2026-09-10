#!/usr/bin/env python3
"""
lqt_pykonal_dt_depth_histograms_by_station.py

Per-station 2D histograms of delay time (x) vs. event depth (y), for the
"passing data" at each of the two quality thresholds already established for
per-station comparisons in lqt_pykonal_tomography_source_voxel_per_station.py:
  - "strict": dt_err<0.04s & phi_err<20 deg & Q_w>=0.7 & dt<=0.24s
  - "loose":  dt_err<0.04s & phi_err<20 deg & Q_w>=-0.5 & dt<=0.24s
(the same two filter definitions from the shared
lqt_pykonal_tomography_raylength_anisotropy.build_page_filters() -- indices 3
and 5 -- so thresholds can never drift from the anisotropy scripts).

Not a spatial/voxel plot -- this is a raw per-event distribution check (no
ray tracing, no grid, no bathymetry): does the depth/delay-time distribution
of "passing" events look reasonable/consistent per station, and how does it
change between the strict and loose quality cuts?

One PDF page, 6 stations (rows) x 2 quality thresholds (columns), each panel
a log-count 2D histogram (x = delay time [s], y = depth [km], depth axis
inverted so shallower is up).

Produces one PDF (distinct name, does not overwrite any existing output):
    lqt_pykonal_dt_depth_histograms_by_station.pdf

Run with:
    python3 lqt_pykonal_dt_depth_histograms_by_station.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

import lqt_pykonal_tomography_raylength_anisotropy as rla
from lqt_pykonal_tomography_source_voxel_anisotropy import prepare_source_voxel_data

OUT_DIR = rla.OUT_DIR

DT_BINS = np.linspace(0., rla.DT_CUTOFF, 41)     # 0-0.24 s (both filters already cut at dt<=0.24s)
DEPTH_BINS = np.linspace(0., 5., 41)             # 0-5 km, raw event depths (not clipped to the 0-4km voxel grid)

STRICT_FILTER_IDX = 3   # dt_err<0.04 & phi_err<20 & Q_w>=0.7 & dt<=0.24
LOOSE_FILTER_IDX = 5    # dt_err<0.04 & phi_err<20 & Q_w>=-0.5 & dt<=0.24


def main():
    all_df, summary, _a_ray_pct, _in_grid, page_filters, _event_ns = prepare_source_voxel_data()

    strict_label, strict_mask = page_filters[STRICT_FILTER_IDX]
    loose_label, loose_mask = page_filters[LOOSE_FILTER_IDX]
    assert 'Q_w≥0.7' in strict_label, f'unexpected filter at index {STRICT_FILTER_IDX}: {strict_label!r}'
    assert 'Q_w≥-0.5' in loose_label, f'unexpected filter at index {LOOSE_FILTER_IDX}: {loose_label!r}'

    dt = summary['dt'].values.astype(np.float64)
    depth = all_df['z'].values.astype(np.float64)
    station = summary['station'].values

    filters = [('Q_w≥0.7 (strict)', strict_label, strict_mask),
              ('Q_w≥-0.5 (loose)', loose_label, loose_mask)]

    n_sta = len(rla.STATIONS)
    n_filt = len(filters)
    fig, axes = plt.subplots(n_sta, n_filt, figsize=(5.5 * n_filt, 3.2 * n_sta), squeeze=False)

    for ri, sta in enumerate(rla.STATIONS):
        sta_mask = (station == sta)
        for ci, (short_label, full_label, base_mask) in enumerate(filters):
            mask = base_mask & sta_mask
            ax = axes[ri][ci]
            n = int(mask.sum())
            print(f'  {sta} — {short_label}: {n:,} events')
            if n > 0:
                counts, xedges, yedges, im = ax.hist2d(
                    dt[mask], depth[mask], bins=[DT_BINS, DEPTH_BINS],
                    cmap='Blues', norm=LogNorm() if counts_gt1(dt[mask], depth[mask]) else None)
                cb = plt.colorbar(im, ax=ax)
                cb.set_label('count', fontsize=7)
                cb.ax.tick_params(labelsize=6)
            ax.invert_yaxis()
            ax.set_xlabel('Delay time δt [s]', fontsize=8)
            ax.set_ylabel('Depth [km]', fontsize=8)
            ax.set_title(f'{sta} — {short_label}\nN={n:,}', fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=7)

    fig.suptitle('LQT + PyKonal-FMM — depth vs. delay time per station\n'
                'dt_err<0.04s & phi_err<20° & dt≤0.24s, split by Q_w threshold',
                fontsize=12, fontweight='bold', y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(OUT_DIR, 'lqt_pykonal_dt_depth_histograms_by_station.pdf')
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


def counts_gt1(x, y):
    """True if a plain 2D histogram of (x, y) would have any bin with >1
    count -- i.e. LogNorm is meaningful (avoids an all-single-count histogram
    silently rendering blank under log scaling)."""
    if len(x) == 0:
        return False
    h, _, _ = np.histogram2d(x, y, bins=[DT_BINS, DEPTH_BINS])
    return bool((h > 1).any())


if __name__ == '__main__':
    main()
