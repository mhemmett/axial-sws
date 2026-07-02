#!/usr/bin/env python3
"""
sws_histograms.py

Multi-page PDF of φ, φ_error, δt, δt_error distributions
for all data combined and per calendar year.

Pages:
  1. φ and φ_error  — filtered (dt_error < 0.1 s)
  2. φ and φ_error  — unfiltered
  3. δt and δt_error — filtered
  4. δt and δt_error — unfiltered
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import os

BASE = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
DATA = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'

FILES = {
    'AXAS1': ('splitting_results_mldd_2015_2021_axas1.csv',
              'splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2': ('splitting_results_mldd_2015_2021_axas2.csv',
              'splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1': ('splitting_results_mldd_2015_2021_axcc1_all_batches.csv',
              'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1': ('splitting_results_mldd_2015_2021_axec1_all_batches.csv',
              'splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2': ('axial-mldd-2015-2021-axec2.csv',
              'splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3': ('splitting_results_mldd_2015_2021_axec3_all_batches.csv',
              'splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

YEARS = list(range(2015, 2027))
ERUPTION_START = pd.Timestamp('2015-04-24', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19', tz='UTC')

# Histogram parameters per variable
VAR_PARAMS = {
    'phi_az':    dict(label='φ (° from N)', bins=36, range=(0, 180),
                      color='steelblue'),
    'phi_error': dict(label='φ error (°)',  bins=40, range=(0, 45),
                      color='darkorange'),
    'dt':        dict(label='δt (s)',        bins=40, range=(0, 0.6),
                      color='forestgreen'),
    'dt_error':  dict(label='δt error (s)', bins=40, range=(0, 0.15),
                      color='firebrick'),
}

# ── Load all data ─────────────────────────────────────────────────────────────
print('Loading data...')
frames = []
for sta, (f1, f2) in FILES.items():
    for f in [f1, f2]:
        df = pd.read_csv(DATA + f).loc[:, :'dt_error'].dropna()
        df = df[df['dt'] > 0]
        df['station'] = sta
        df['t'] = pd.to_datetime(df['event_datetime'], utc=True)
        df['year'] = df['t'].dt.year
        df['phi_az'] = df['phi'] + 90.0   # geographic azimuth
        frames.append(df)
all_df = pd.concat(frames, ignore_index=True)
filt_df = all_df[all_df['dt_error'] < 0.1].copy()
print(f'  Total: {len(all_df):,}   Filtered: {len(filt_df):,}')

# ── Page builder ──────────────────────────────────────────────────────────────
N_COLS = 3
N_YEAR_ROWS = int(np.ceil(len(YEARS) / N_COLS))

def make_page(df, var1, var2, page_title):
    """
    One page: top row = all-data histograms for var1 and var2,
    then N_YEAR_ROWS × N_COLS yearly histograms for each variable
    (var1 left half, var2 right half).
    """
    p1 = VAR_PARAMS[var1]; p2 = VAR_PARAMS[var2]

    # Layout: 1 all-data row + N_YEAR_ROWS year rows,
    #         left N_COLS cols = var1, right N_COLS cols = var2
    total_rows = 1 + N_YEAR_ROWS
    total_cols = N_COLS * 2 + 1   # gap column in middle

    fig = plt.figure(figsize=(18, 3 + N_YEAR_ROWS * 2.8))
    fig.suptitle(page_title, fontsize=13, fontweight='bold', y=0.98)

    gs = GridSpec(total_rows, total_cols, figure=fig,
                  hspace=0.45, wspace=0.4,
                  left=0.05, right=0.97, top=0.93, bottom=0.04)

    def hist(ax, data, vp, title, n):
        if len(data) > 0:
            ax.hist(data, bins=vp['bins'], range=vp['range'],
                    color=vp['color'], alpha=0.80, edgecolor='none')
            med = float(np.median(data))
            ax.axvline(med, color='k', lw=1.2, ls='--')
        ax.set_title(f'{title}  N={n:,}', fontsize=6.5, pad=2)
        ax.set_xlabel(vp['label'], fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_xlim(vp['range'])

    # ── All-data row ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :N_COLS])
    hist(ax1, df[var1].dropna().values, p1,
         f'All years — {p1["label"]}', len(df))

    ax2 = fig.add_subplot(gs[0, N_COLS+1:])
    hist(ax2, df[var2].dropna().values, p2,
         f'All years — {p2["label"]}', len(df))

    ax1.set_ylabel('Count', fontsize=6)
    ax2.set_ylabel('Count', fontsize=6)

    # ── Yearly rows ───────────────────────────────────────────────────────────
    for k, yr in enumerate(YEARS):
        row = 1 + k // N_COLS
        col_v1 = k % N_COLS
        col_v2 = N_COLS + 1 + k % N_COLS

        sub = df[df['year'] == yr]
        n = len(sub)
        label = f'{yr}  N={n:,}'

        ax1 = fig.add_subplot(gs[row, col_v1])
        hist(ax1, sub[var1].dropna().values, p1, label, n)
        if col_v1 == 0:
            ax1.set_ylabel('Count', fontsize=5.5)

        ax2 = fig.add_subplot(gs[row, col_v2])
        hist(ax2, sub[var2].dropna().values, p2, label, n)

    return fig

# ── Generate PDF ──────────────────────────────────────────────────────────────
out = os.path.join(BASE, 'sws_histograms.pdf')
print(f'Writing {out}...')

with PdfPages(out) as pdf:
    pages = [
        (filt_df,  'phi_az',    'phi_error',
         'φ and φ error — filtered (dt_error < 0.1 s)'),
        (all_df,   'phi_az',    'phi_error',
         'φ and φ error — unfiltered (all data)'),
        (filt_df,  'dt',        'dt_error',
         'δt and δt error — filtered (dt_error < 0.1 s)'),
        (all_df,   'dt',        'dt_error',
         'δt and δt error — unfiltered (all data)'),
    ]
    for df, v1, v2, title in pages:
        fig = make_page(df, v1, v2, title)
        pdf.savefig(fig, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'  Page done: {title[:50]}')

print(f'Saved {out}')
