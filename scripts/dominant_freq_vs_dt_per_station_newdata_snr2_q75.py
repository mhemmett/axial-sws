#!/usr/bin/env python3
"""
dominant_freq_vs_dt_per_station_newdata_snr2_q75.py

Per-station 2D histogram of dominant frequency (1 / dominant_period, Hz) vs. delay time
(dt, seconds), for the six-station "newdata" dataset (AXAS1, AXAS2, AXCC1, AXEC1, AXEC2,
AXEC3 from mfast_maxdt_pipeline_transfer/splitting_results_{STA}_{2015_2021,2022_2026}_
all_batches.csv), following the same 2D-histogram/pcolormesh/LogNorm style as
dt_vs_depth_per_station_newdata_snr2_q75.py / back_azimuth_vs_dt_per_station_newdata_snr2_q75.py.

Single filter (same "Page 1 @ quality>=0.75" grade used throughout this session):
SNR >= 2.0, quality (Q_w) >= 0.75, dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg.

Produces:
    dominant_freq_vs_dt_per_station_newdata_snr2_q75.pdf
(6 pages, one station per page) -- a NEW file, does not touch any existing output.

Run with:
    python3 dominant_freq_vs_dt_per_station_newdata_snr2_q75.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
OUT_PDF = os.path.join(HERE, 'dominant_freq_vs_dt_per_station_newdata_snr2_q75.pdf')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STATION_FILES = {
    sta: [f'splitting_results_{sta}_2015_2021_all_batches.csv',
          f'splitting_results_{sta}_2022_2026_all_batches.csv']
    for sta in STATIONS
}

GRADE = dict(label='SNR >= 2.0, quality >= 0.75, dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg',
             snr_min=2.0, phi_err_max=20.0, qw_min=0.75)
DT_ERR_MAX = 0.05

N_FREQ_BINS = 40
N_DT_BINS = 40


def load_station_events(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['dominant_period', 'quality', 'snr_horizontal', 'phi_error', 'dt_error'])
    df = df[df['dominant_period'] > 0]

    mask = ((df['quality'] >= GRADE['qw_min']) &
            (df['snr_horizontal'] >= GRADE['snr_min']) &
            (df['dt_error'] <= DT_ERR_MAX) &
            (df['dt'] <= df['dominant_period'] / 2.0) &
            (df['phi_error'] <= GRADE['phi_err_max']))
    df = df[mask].copy()
    df['freq'] = 1.0 / df['dominant_period'].values
    return df[['freq', 'dt']].reset_index(drop=True)


def make_station_page(sta, df):
    freq = df['freq'].values
    dt = df['dt'].values

    fig, ax = plt.subplots(figsize=(7.5, 6))

    freq_edges = np.linspace(freq.min(), freq.max(), N_FREQ_BINS + 1)
    dt_edges = np.linspace(0, dt.max(), N_DT_BINS + 1)
    counts, xe, ye = np.histogram2d(freq, dt, bins=[freq_edges, dt_edges])
    im = ax.pcolormesh(xe, ye, counts.T, cmap='viridis',
                       norm=mcolors.LogNorm(vmin=1, vmax=max(counts.max(), 2)))
    fig.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel('Dominant frequency (Hz)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Delay time δt (s)', fontsize=10, fontweight='bold')

    fig.suptitle(f'{sta} — dominant frequency vs. delay time (N={len(df):,})\n{GRADE["label"]}',
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
