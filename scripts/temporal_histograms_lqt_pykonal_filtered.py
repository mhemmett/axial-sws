#!/usr/bin/env python3
"""
temporal_histograms_lqt_pykonal_filtered.py

QC-filtered version of temporal_histograms_lqt_pykonal.py's per-station
dt/phi moving-window density plots. The original script applies ONLY the
baseline cleanup (success==True, dt>0) -- no phi_error/dt_error/dt-cutoff/
quality filtering at all ("these are raw density plots, not the QC'd rose/
tomography figures", per its own docstring) -- so it does NOT carry the
same thresholds used throughout the rest of this session's rose/tomography
plots.

This script applies the "three set filters" used throughout this session
(identical constants to lqt_pykonal_tomography_raylength_anisotropy.py):
    dt_err < 0.04 s,  phi_err < 20 deg,  dt <= 0.24 s (0.8 * max_dt)
further split into the same two quality-threshold variants used throughout
this session's per-station comparisons:
    strict: Q_w >= 0.7
    loose:  Q_w >= -0.5

Reuses temporal_histograms_lqt_pykonal.py's load_station()/draw_dt()/
draw_phi() verbatim (unmodified) -- only the loaded dataframe is filtered
before plotting, and the page title is rewritten to explicitly state the
filter in use (the original script's title only ever says "incidence<35deg",
with no phi_error/dt_error/quality/dt-cutoff wording since it doesn't apply
any).

Produces one PDF (distinct name, does not overwrite any of the six existing
per-station outputs), 12 pages (6 stations x 2 quality variants):
    lqt_pykonal_temporal_histograms_filtered.pdf

Run with:
    python3 temporal_histograms_lqt_pykonal_filtered.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import temporal_histograms_lqt_pykonal as tha

OUT_DIR = tha.OUT_DIR

# Same constants/values as lqt_pykonal_tomography_raylength_anisotropy.py.
DT_ERR_MAX = 0.04
PHI_ERR_MAX = 20.0
MAX_DT = 0.30
DT_CUTOFF = MAX_DT * 0.8   # 0.24 s

QW_STRICT = 0.7
QW_LOOSE = -0.5


def make_station_figure_filtered(sta, df, filt_label):
    fig, (ax_dt, ax_phi) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    fig.suptitle(f'{sta} — LQT + PyKonal-FMM incidence (35° cut) — Temporal Evolution\n'
                f'{filt_label}  (N={len(df):,})', fontsize=11, fontweight='bold')
    im_dt = tha.draw_dt(ax_dt, df)
    im_phi = tha.draw_phi(ax_phi, df)
    ax_dt.set_title(f'mean δt = {df["dt"].mean():.3f} s' if len(df) else 'mean δt = n/a',
                   fontsize=8.5, loc='right')
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
    filters = [
        (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}° & Q_w≥{QW_STRICT} & dt≤{DT_CUTOFF:.2f}s', QW_STRICT),
        (f'dt_err<{DT_ERR_MAX}s & phi_err<{PHI_ERR_MAX:.0f}° & Q_w≥{QW_LOOSE} & dt≤{DT_CUTOFF:.2f}s', QW_LOOSE),
    ]

    out_path = os.path.join(OUT_DIR, 'lqt_pykonal_temporal_histograms_filtered.pdf')
    with PdfPages(out_path) as pdf:
        for sta in tha.STATION_ORDER:
            print(f'\nLoading {sta}...')
            df_raw = tha.load_station(sta)
            print(f'  {sta}: {len(df_raw):,} baseline (success & dt>0)')
            for filt_label, qw_min in filters:
                mask = ((df_raw['dt_error'] < DT_ERR_MAX) &
                       (df_raw['phi_error'] < PHI_ERR_MAX) &
                       (df_raw['dt'] <= DT_CUTOFF) &
                       (df_raw['quality'] >= qw_min))
                df = df_raw[mask]
                print(f'    {filt_label}: {len(df):,}')
                fig = make_station_figure_filtered(sta, df, filt_label)
                pdf.savefig(fig, dpi=200, bbox_inches='tight')
                plt.close(fig)

    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
