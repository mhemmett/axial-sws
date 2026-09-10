#!/usr/bin/env python3
"""
rose_batch1_production_vs_windowcheck.py

One-batch (AXEC2 batch 1, 2015-2021 mfast max_dt=0.2s data) comparison of production SWSPy
(swspy/swspy/splitting/split.py) vs the window-check fix
(swspy/swspy/splitting/split_windowcheck.py -- undersized grid-search windows widened to
exactly max_t_shift_s at window-setup time, rather than left short enough to wrap around on
themselves in the cyclic-roll lag search), per explicit user request, while the remaining
batches 2-10 continue running in the background.

Two QC tiers (batch 1 alone is too small for the usual tier-5 count to be meaningful -- 11/11
events survive tier 5 for BOTH methods -- so quality>=0.5 is used as the primary tier here):
  Tier A: success & quality>=0.5
  Tier B (tier 5): + dt<T_dom/2, phi_error<10deg, dt_error<0.05s

Also reports a PAIRED per-event comparison (same event_id, success in both) of the fast
direction (circular, mod-180) and delay-time differences directly attributable to the window
layout change.

Produces: rose_batch1_production_vs_windowcheck.pdf (1 page: 2 tiers x 2 methods rose grid,
plus a text summary of the paired comparison)

Run with:
    python3 rose_batch1_production_vs_windowcheck.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
PROD_CSV = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results',
                        'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_1.csv')
WC_CSV = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_windowcheck_lqt_pykonal_results',
                      'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_batch_1.csv')
OUT_PDF = os.path.join(HERE, 'rose_batch1_production_vs_windowcheck.pdf')

QW_MIN = 0.5
PHI_ERR_MAX = 10.0
DT_ERR_MAX = 0.05

NBINS = 36


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df['dominant_period'] = df['chosen_filter_dom_period_samples'] / 200.0
    df['phi_az'] = df['phi'] % 180.0
    return df


def apply_tier(df, tier):
    d = df.dropna(subset=['quality'])
    d = d[d['quality'] >= QW_MIN]
    if tier == 'A':
        return d
    d = d.dropna(subset=['dominant_period'])
    d = d[d['dt'] < d['dominant_period'] / 2.0]
    d = d.dropna(subset=['phi_error'])
    d = d[d['phi_error'] < PHI_ERR_MAX]
    d = d.dropna(subset=['dt_error'])
    d = d[d['dt_error'] < DT_ERR_MAX]
    return d


def _draw_rose(ax, phi_az_vals, color):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('none')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(0.8)
    ax.grid(False)

    if len(phi_az_vals) == 0:
        ax.set_ylim(0, 1)
        return

    doubled_angles = []
    for phi in phi_az_vals:
        p = float(phi) % 360.0
        doubled_angles.extend([np.deg2rad(p), np.deg2rad((p + 180.0) % 360.0)])
    doubled_angles = np.array(doubled_angles)

    bins = np.linspace(0, 2 * np.pi, NBINS + 1)
    counts, edges = np.histogram(doubled_angles, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 2 * np.pi / NBINS

    ax.bar(centers, counts, width=width, bottom=0,
           color=color, edgecolor='black', linewidth=0.5, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)


def circular_mean_diff_deg(phi1, phi2):
    """Smallest signed difference between two mod-180 (axial) angles, in degrees, in
    [-90, 90]."""
    d = (phi1 - phi2 + 90.0) % 180.0 - 90.0
    return d


def make_figure(prod_df, wc_df, paired):
    tiers = [('A: quality>=0.5', 'A'), ('B: tier 5 (+dt<T_dom/2, phi_err<10, dt_err<0.05)', 'B')]
    methods = [('Production (split.py)', prod_df, '#CC0000'), ('Window-check fix (split_windowcheck.py)', wc_df, '#0072B2')]

    fig = plt.figure(figsize=(9, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.5, wspace=0.15)

    for ri, (tier_label, tier_key) in enumerate(tiers):
        for ci, (method_label, df, color) in enumerate(methods):
            sub = apply_tier(df, tier_key)
            ax = fig.add_subplot(gs[ri, ci], projection='polar')
            _draw_rose(ax, sub['phi_az'].values, color)
            ax.set_title(f'{method_label}\n{tier_label}\nN={len(sub)}', fontsize=8.5, fontweight='bold', pad=10)

    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')
    ax_text.text(0.0, 1.0, paired, fontsize=9, family='monospace', va='top', ha='left',
                 transform=ax_text.transAxes)

    fig.suptitle('AXEC2 batch 1 (216 events): production vs window-check fix', fontsize=12,
                 fontweight='bold', y=0.985)
    return fig


def build_paired_summary(prod_df, wc_df):
    p = prod_df[['event_id', 'phi_az', 'dt', 'quality']].rename(
        columns={'phi_az': 'phi_az_prod', 'dt': 'dt_prod', 'quality': 'quality_prod'})
    w = wc_df[['event_id', 'phi_az', 'dt', 'quality']].rename(
        columns={'phi_az': 'phi_az_wc', 'dt': 'dt_wc', 'quality': 'quality_wc'})
    m = p.merge(w, on='event_id', how='inner')

    m['dphi'] = circular_mean_diff_deg(m['phi_az_wc'].values, m['phi_az_prod'].values)
    m['ddt'] = m['dt_wc'] - m['dt_prod']

    n_both = len(m)
    n_phi_changed = int((np.abs(m['dphi']) > 10.0).sum())
    n_dt_changed = int((np.abs(m['ddt']) > 0.01).sum())
    n_identical = int(((np.abs(m['dphi']) < 1e-6) & (np.abs(m['ddt']) < 1e-6)).sum())

    lines = [
        f"Paired per-event comparison (both methods succeeded): N={n_both}",
        f"  Identical result (both phi and dt unchanged): {n_identical} ({100*n_identical/max(n_both,1):.1f}%)",
        f"  |d(phi)| > 10 deg (mod-180, circular): {n_phi_changed} ({100*n_phi_changed/max(n_both,1):.1f}%)",
        f"  |d(dt)| > 0.01 s: {n_dt_changed} ({100*n_dt_changed/max(n_both,1):.1f}%)",
        f"  Mean |d(phi)| = {np.abs(m['dphi']).mean():.1f} deg, median = {np.abs(m['dphi']).median():.1f} deg, max = {np.abs(m['dphi']).max():.1f} deg",
        f"  Mean |d(dt)|  = {np.abs(m['ddt']).mean():.4f} s, median = {np.abs(m['ddt']).median():.4f} s, max = {np.abs(m['ddt']).max():.4f} s",
        f"  Mean quality: production = {m['quality_prod'].mean():.3f}, window-check = {m['quality_wc'].mean():.3f}",
    ]
    return '\n'.join(lines), m


def main():
    prod_df = load(PROD_CSV)
    wc_df = load(WC_CSV)
    print(f'Production: {len(prod_df)} successful events')
    print(f'Window-check: {len(wc_df)} successful events')

    paired_text, m = build_paired_summary(prod_df, wc_df)
    print('\n' + paired_text)

    fig = make_figure(prod_df, wc_df, paired_text)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
