#!/usr/bin/env python3
"""
sws_histograms_strict_filter.py

Compares two filter levels for phi and dt distributions:
  Blue:  dt_error < 0.1 s  (standard filter)
  Navy:  dt_error < 0.1 s  AND phi_error < 33.8° (mean phi_error of standard filter)

Two-page PDF: phi, then dt. Each page: all-years + yearly panels.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import os

DATA  = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
OUT   = DATA + 'sws_histograms_strict_filter.pdf'
YEARS = list(range(2015, 2027))
N_COLS = 3
N_YEAR_ROWS = int(np.ceil(len(YEARS) / N_COLS))

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

# ── Load ──────────────────────────────────────────────────────────────────────
print('Loading...')
frames = []
for sta, (f1, f2) in FILES.items():
    for f in [f1, f2]:
        df = pd.read_csv(DATA + f).loc[:, :'dt_error'].dropna()
        df = df[df['dt'] > 0]
        df['t']      = pd.to_datetime(df['event_datetime'], utc=True)
        df['year']   = df['t'].dt.year
        df['phi_az'] = df['phi'] + 90.
        frames.append(df)
all_df = pd.concat(frames, ignore_index=True)

filt_std    = all_df[all_df['dt_error'] < 0.1].copy()
filt_strict = all_df[(all_df['dt_error'] < 0.1) &
                      (all_df['phi_error'] <= 33.8)].copy()

print(f'  Standard (dt_err<0.1):            N={len(filt_std):,}')
print(f'  Strict   (+phi_err≤33.8°):        N={len(filt_strict):,}  '
      f'({100*len(filt_strict)/len(filt_std):.1f}% of standard)')

# ── Figure builder ────────────────────────────────────────────────────────────
def make_page(var, xlabel, vrange, bins, c_std, c_strict, title):
    fig = plt.figure(figsize=(18, 3.5 + N_YEAR_ROWS * 2.8))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.99)

    gs = GridSpec(1 + N_YEAR_ROWS, N_COLS, figure=fig,
                  hspace=0.50, wspace=0.35,
                  left=0.06, right=0.97, top=0.94, bottom=0.04)

    def dual(ax, d_std, d_strict, panel_title):
        if len(d_std):
            ax.hist(d_std, bins=bins, range=vrange, color=c_std,
                    alpha=0.55, edgecolor='none',
                    label=f'dt_err<0.1  (N={len(d_std):,})')
        if len(d_strict):
            ax.hist(d_strict, bins=bins, range=vrange, color=c_strict,
                    alpha=0.80, edgecolor='none',
                    label=f'+φ_err≤33.8°  (N={len(d_strict):,})')
        for d, c in [(d_std, c_std), (d_strict, c_strict)]:
            if len(d):
                ax.axvline(np.median(d), color=c, lw=1.4, ls='--')
        ax.set_title(panel_title, fontsize=7, pad=3)
        ax.set_xlabel(xlabel, fontsize=6.5)
        ax.set_xlim(vrange)
        ax.tick_params(labelsize=5.5)

    # All-data panel spanning all columns
    ax_all = fig.add_subplot(gs[0, :])
    dual(ax_all,
         filt_std[var].dropna().values,
         filt_strict[var].dropna().values,
         'All years')
    ax_all.set_ylabel('Count', fontsize=8)
    ax_all.legend(fontsize=8, loc='upper right')

    # Yearly panels
    for k, yr in enumerate(YEARS):
        row = 1 + k // N_COLS
        col = k % N_COLS
        s_std    = filt_std[filt_std['year'] == yr][var].dropna().values
        s_strict = filt_strict[filt_strict['year'] == yr][var].dropna().values
        ax = fig.add_subplot(gs[row, col])
        dual(ax, s_std, s_strict, str(yr))
        if col == 0:
            ax.set_ylabel('Count', fontsize=6)

    return fig

# ── Write PDF ─────────────────────────────────────────────────────────────────
with PdfPages(OUT) as pdf:
    fig = make_page('phi_az', 'φ (° from N)', (0, 180), 36,
                    'steelblue', 'navy',
                    'φ distribution — standard filter vs. strict filter (also φ_error ≤ 33.8°)')
    pdf.savefig(fig, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('  Page 1: φ done')

    fig = make_page('dt', 'δt (s)', (0, 0.5), 40,
                    'forestgreen', 'darkgreen',
                    'δt distribution — standard filter vs. strict filter (also φ_error ≤ 33.8°)')
    pdf.savefig(fig, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('  Page 2: δt done')

print(f'Saved {OUT}')
