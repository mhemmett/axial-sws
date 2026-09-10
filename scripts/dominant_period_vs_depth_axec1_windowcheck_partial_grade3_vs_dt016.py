#!/usr/bin/env python3
"""
dominant_period_vs_depth_axec1_windowcheck_partial_grade3_vs_dt016.py

Per-event dominant period (T_dom, seconds) vs. earthquake source depth (km), 2D histogram
(pcolormesh, LogNorm count scale) -- same style/structure as
dt_vs_depth_axec1_windowcheck_partial_grade3_vs_dt016.py and
dt_vs_dominant_period_axec1_windowcheck_partial_grade3_vs_dt016.py, just T_dom vs. depth
directly -- for the NEW partial AXEC1 window-check rerun only
(mfast_maxdt_windowcheck_pipeline_transfer/splitting_results_AXEC1_2015_2021_combined_partial.csv).

Two filters, SIDE BY SIDE in one page (same axis ranges on both panels for direct comparison):
  - Grade 3: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg
  - Grade 3 (dt<=0.16s): same, but dt<=0.16s (0.8*max_dt=0.2s) instead of the T_dom/2 cut

(No second depth-restricted page here, unlike the two companion scripts -- depth IS one of the
two axes in this figure, so a "depth<=1.5km" subset page would just be a zoomed-in crop of the
same panel rather than a new comparison.)

Prints a linear regression of T_dom on depth (Pearson r, R^2, slope, intercept) for each filter,
to check directly whether the strong dt-vs-T_dom relationship found in the companion script is
itself just tracking a T_dom-vs-depth relationship (i.e. whether dominant period is itself
depth-dependent), or whether T_dom and depth are largely independent.

Produces:
    dominant_period_vs_depth_axec1_windowcheck_partial_grade3_vs_dt016.pdf
(1 page, 2 panels) -- a NEW file, does not touch any existing output.

Run with:
    python3 dominant_period_vs_depth_axec1_windowcheck_partial_grade3_vs_dt016.py
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
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_CSV = os.path.join(HERE, '..', 'mfast_maxdt_windowcheck_pipeline_transfer',
                       'splitting_results_AXEC1_2015_2021_combined_partial.csv')
OUT_PDF = os.path.join(HERE, 'dominant_period_vs_depth_axec1_windowcheck_partial_grade3_vs_dt016.pdf')

MAX_DT = 0.2                  # this session's production max_t_shift_s override
DT_ALT_MAX = 0.8 * MAX_DT      # 0.16s
DT_ERR_MAX = 0.05

FILTERS = [
    dict(key='grade3', short='Grade 3 (dt<=T_dom/2)',
         label='Grade 3: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg'),
    dict(key='grade3_dt016', short=f'Grade 3 (dt<={DT_ALT_MAX:.2f}s)',
         label=f'Grade 3 (dt<={DT_ALT_MAX:.2f}s instead of T_dom/2): SNR>=2.0, quality>=0.75, '
               f'dt_err<=0.05s, dt<={DT_ALT_MAX:.2f}s, phi_err<=20deg'),
]

N_DEPTH_BINS = 40
N_TDOM_BINS = 40


def load_events():
    df = pd.read_csv(NEW_CSV)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['depth', 'quality', 'dominant_period', 'snr_horizontal',
                            'phi_error', 'dt_error'])
    return df.reset_index(drop=True)


def apply_filter(df, key):
    base = ((df['snr_horizontal'] >= 2.0) & (df['quality'] >= 0.75) &
            (df['dt_error'] <= DT_ERR_MAX) & (df['phi_error'] <= 20.0))
    if key == 'grade3':
        return df[base & (df['dt'] <= df['dominant_period'] / 2.0)].copy()
    elif key == 'grade3_dt016':
        return df[base & (df['dt'] <= DT_ALT_MAX)].copy()
    raise ValueError(key)


def print_regression(subs, page_label):
    print(f'\n  Linear regression of T_dom on depth -- {page_label}:')
    for sub, filt in zip(subs, FILTERS):
        if len(sub) < 3:
            print(f'    {filt["key"]}: N={len(sub)} too few points for regression')
            continue
        r, p = stats.pearsonr(sub['depth'], sub['dominant_period'])
        slope, intercept, r_lr, p_lr, se = stats.linregress(sub['depth'], sub['dominant_period'])
        print(f'    {filt["key"]}: N={len(sub):,}  Pearson r={r:.4f} (p={p:.2e})  '
              f'slope={slope:.5f} s/km  intercept={intercept:.4f}s  R^2={r_lr**2:.4f}')


def make_figure(subs):
    depth_min = min(s['depth'].min() for s in subs)
    depth_max = max(s['depth'].max() for s in subs)
    tdom_max = max(s['dominant_period'].max() for s in subs)
    depth_edges = np.linspace(depth_min, depth_max, N_DEPTH_BINS + 1)
    tdom_edges = np.linspace(0, tdom_max, N_TDOM_BINS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.35))
    for ax, sub, filt in zip(axes, subs, FILTERS):
        depth = sub['depth'].values
        tdom = sub['dominant_period'].values
        counts, xe, ye = np.histogram2d(depth, tdom, bins=[depth_edges, tdom_edges])
        im = ax.pcolormesh(xe, ye, counts.T, cmap='viridis',
                           norm=mcolors.LogNorm(vmin=1, vmax=max(counts.max(), 2)))
        fig.colorbar(im, ax=ax, label='Count')
        ax.set_xlabel('Source depth (km)', fontsize=10, fontweight='bold')
        ax.set_title(f'{filt["short"]}\nN={len(sub):,}', fontsize=10.5, fontweight='bold', pad=8)
    axes[0].set_ylabel('Dominant period $T_{dom}$ (s)', fontsize=10, fontweight='bold')

    filter_lines = '\n'.join(f'{f["short"]}: {f["label"]}' for f in FILTERS)
    title = 'AXEC1 (window-check partial rerun): dominant period vs. source depth'
    fig.suptitle(f'{title}\n{filter_lines}', fontsize=11, fontweight='bold', y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def main():
    print('Loading AXEC1 window-check partial data...')
    df = load_events()
    print(f'  {len(df):,} success rows loaded')

    subs = []
    for filt in FILTERS:
        sub = apply_filter(df, filt['key'])
        print(f"  {filt['key']}: N={len(sub):,}")
        subs.append(sub)
    print_regression(subs, 'all depths')

    fig = make_figure(subs)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
