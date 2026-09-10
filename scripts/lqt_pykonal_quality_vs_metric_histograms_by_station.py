#!/usr/bin/env python3
"""
lqt_pykonal_quality_vs_metric_histograms_by_station.py

Per-station 2D histograms of quality (Q_w, Wustefeld 2010) against each of
three companion metrics -- depth, delay time, and fast direction (phi) -- to
see how the Q_w distribution itself relates to depth/dt/phi per station.

Filter: dt_err<0.04s & phi_err<20 deg & dt<=0.24s (the same three non-quality
thresholds used throughout this family of plots) -- NO Q_w cut, since quality
is the plotted x-axis here, not a filter criterion.

Not a spatial/voxel plot -- raw per-event distribution check, reusing
lqt_pykonal_tomography_source_voxel_anisotropy.prepare_source_voxel_data()
for the aligned (dt, phi_error, dt_error, quality, station, depth, phi) per
event (phi comes from load_station()'s 'phi' column, added there for this
script -- see lqt_pykonal_tomography_raylength_anisotropy.py's load_station()).

Three pages (one per companion metric), each a 2-row x 3-col grid of
per-station log-count 2D histograms: x = Q_w, y = the companion metric.

Produces one PDF (distinct name, does not overwrite any existing output):
    lqt_pykonal_quality_vs_metric_histograms_by_station.pdf   (3 pages)

Run with:
    python3 lqt_pykonal_quality_vs_metric_histograms_by_station.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

import lqt_pykonal_tomography_raylength_anisotropy as rla
from lqt_pykonal_tomography_source_voxel_anisotropy import prepare_source_voxel_data

OUT_DIR = rla.OUT_DIR

QW_BINS = np.linspace(-1., 1., 41)
DEPTH_BINS = np.linspace(0., 5., 41)
DT_BINS = np.linspace(0., rla.DT_CUTOFF, 41)
PHI_BINS = np.linspace(0., 180., 41)

N_ROWS, N_COLS = 2, 3   # 6 stations


def counts_gt1(x, y, xbins, ybins):
    """True if a plain 2D histogram of (x, y) has any bin with >1 count --
    i.e. LogNorm is meaningful (avoids an all-singleton histogram rendering
    blank under log scaling)."""
    if len(x) == 0:
        return False
    h, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
    return bool((h > 1).any())


def plot_metric_page(pdf, fig_title, qw, metric, metric_bins, metric_label,
                     station, mask_base, invert_y=False):
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(5.5 * N_COLS, 4.0 * N_ROWS), squeeze=False)
    for i, sta in enumerate(rla.STATIONS):
        ri, ci = divmod(i, N_COLS)
        ax = axes[ri][ci]
        mask = mask_base & (station == sta)
        n = int(mask.sum())
        print(f'  {sta}: {n:,} events')
        if n > 0:
            x, y = qw[mask], metric[mask]
            norm = LogNorm() if counts_gt1(x, y, QW_BINS, metric_bins) else None
            counts, xedges, yedges, im = ax.hist2d(
                x, y, bins=[QW_BINS, metric_bins], cmap='Blues', norm=norm)
            cb = plt.colorbar(im, ax=ax)
            cb.set_label('count', fontsize=7)
            cb.ax.tick_params(labelsize=6)
        if invert_y:
            ax.invert_yaxis()
        ax.set_xlabel('Quality Q$_w$', fontsize=8)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.set_title(f'{sta}\nN={n:,}', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
    fig.suptitle(fig_title, fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    all_df, summary, _a_ray_pct, _in_grid, _page_filters, _event_ns = prepare_source_voxel_data()

    dt = summary['dt'].values.astype(np.float64)
    phe = summary['phi_error'].values.astype(np.float64)
    dte = summary['dt_error'].values.astype(np.float64)
    qw = summary['quality'].values.astype(np.float64)
    ok = summary['trace_ok'].values.astype(bool)
    station = summary['station'].values

    mask_base = ok & (dte < rla.DT_ERR_MAX) & (phe < rla.PHI_ERR_MAX) & (dt <= rla.DT_CUTOFF)
    filt_label = f'dt_err<{rla.DT_ERR_MAX}s & phi_err<{rla.PHI_ERR_MAX:.0f}° & dt≤{rla.DT_CUTOFF:.2f}s'
    print(f'Filter: {filt_label}')
    print(f'  total passing (all stations): {int(mask_base.sum()):,}')

    depth = all_df['z'].values.astype(np.float64)
    phi_az = all_df['phi'].values.astype(np.float64) % 180.0

    out_path = os.path.join(OUT_DIR, 'lqt_pykonal_quality_vs_metric_histograms_by_station.pdf')
    with PdfPages(out_path) as pdf:
        print('\n=== Quality vs depth ===')
        plot_metric_page(
            pdf, f'LQT + PyKonal-FMM — quality (Q$_w$) vs. depth, per station\n{filt_label}',
            qw, depth, DEPTH_BINS, 'Depth [km]', station, mask_base, invert_y=True)

        print('\n=== Quality vs delay time ===')
        plot_metric_page(
            pdf, f'LQT + PyKonal-FMM — quality (Q$_w$) vs. delay time, per station\n{filt_label}',
            qw, dt, DT_BINS, 'Delay time δt [s]', station, mask_base)

        print('\n=== Quality vs phi ===')
        plot_metric_page(
            pdf, f'LQT + PyKonal-FMM — quality (Q$_w$) vs. fast direction φ, per station\n{filt_label}',
            qw, phi_az, PHI_BINS, 'Fast direction φ [° from N, mod 180°]', station, mask_base)

    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
