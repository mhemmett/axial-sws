#!/usr/bin/env python3
"""
phi_vs_back_azimuth_per_station_newdata_grade3.py

Per-station 2D histogram of fast-polarization direction (phi, axial mod 180 deg) vs.
back-azimuth (station -> event, deg from N), for the six-station "newdata" dataset
(AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3 from
mfast_maxdt_pipeline_transfer/splitting_results_{STA}_{2015_2021,2022_2026}_all_batches.csv)
-- the same dataset used by sws_tomography_johnson2011_newdata_grade3_7period.py (the most
recent run with data for all six stations) -- following the same 2D-histogram/pcolormesh/
LogNorm style as back_azimuth_vs_dt_per_station_newdata_snr2_q75.py.

Grade-3 filter (T.GRADES[3], same as the Johnson-2011 newdata re-run and this session's other
grade-3 figures): SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg.

Back-azimuth is taken directly from the catalog's own 'back_azimuth' column (station -> event
bearing, deg from N), not recomputed from station/event geometry. phi is folded to axial
0-180 deg (mod 180) per this project's standard convention for fast-polarization direction.

Produces:
    phi_vs_back_azimuth_per_station_newdata_grade3.pdf
(6 pages, one station per page) -- a NEW file, does not touch any existing output.

Run with:
    python3 phi_vs_back_azimuth_per_station_newdata_grade3.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
OUT_PDF = os.path.join(HERE, 'phi_vs_back_azimuth_per_station_newdata_grade3.pdf')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STATION_FILES = {
    sta: [f'splitting_results_{sta}_2015_2021_all_batches.csv',
          f'splitting_results_{sta}_2022_2026_all_batches.csv']
    for sta in STATIONS
}

GRADE = T.GRADES[3]
assert GRADE['key'] == 'page1_q0.75', GRADE
DT_ERR_MAX = 0.05

N_BAZ_BINS = 36   # 10-degree bins
N_PHI_BINS = 36   # 5-degree bins (axial 0-180)


def load_station_events(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['back_azimuth', 'phi', 'quality', 'dominant_period',
                            'snr_horizontal', 'phi_error', 'dt_error'])

    mask = ((df['quality'] >= GRADE['qw_min']) &
            (df['snr_horizontal'] >= GRADE['snr_min']) &
            (df['dt_error'] <= DT_ERR_MAX) &
            (df['dt'] <= df['dominant_period'] / 2.0) &
            (df['phi_error'] <= GRADE['phi_err_max']))
    df = df[mask].copy()
    df['baz'] = df['back_azimuth'].values % 360.0
    df['phi_az'] = df['phi'].values % 180.0
    return df[['baz', 'phi_az']].reset_index(drop=True)


def make_station_page(sta, df):
    baz = df['baz'].values
    phi = df['phi_az'].values

    fig, ax = plt.subplots(figsize=(7.5, 6))

    baz_edges = np.linspace(0, 360, N_BAZ_BINS + 1)
    phi_edges = np.linspace(0, 180, N_PHI_BINS + 1)
    counts, xe, ye = np.histogram2d(baz, phi, bins=[baz_edges, phi_edges])
    im = ax.pcolormesh(xe, ye, counts.T, cmap='viridis',
                       norm=mcolors.LogNorm(vmin=1, vmax=max(counts.max(), 2)))
    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('Back-azimuth (deg from N)', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'Fast direction $\phi$ (deg, axial 0-180)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 180)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_yticks(np.arange(0, 181, 30))

    fig.suptitle(f'{sta} — fast direction vs. back-azimuth (N={len(df):,})\n{GRADE["label"]}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    print(f'Loading per-station events ({GRADE["label"]})...')
    with PdfPages(OUT_PDF) as pdf:
        for sta in STATIONS:
            df = load_station_events(sta)
            print(f'  {sta}: {len(df):,} events pass the filter')
            fig = make_station_page(sta, df)
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
