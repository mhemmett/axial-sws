#!/usr/bin/env python3
"""
rose_axec2_windowcheck_vs_newdata_grade3_matched_batches.py

Rose-diagram comparison of the AXEC2 2015-2021 full windowcheck rerun (whatever batches are
DONE SO FAR -- run_axec2_2015_2021_windowcheck_full_batches.py, still in progress) against the
"newdata" (most recent COMPLETE six-station run, i.e. the non-windowcheck production dataset
used everywhere else this session as AXEC2's 2015-2021 half) restricted to the EXACT SAME BATCH
NUMBERS, per explicit user request -- so the comparison isn't confounded by the windowcheck run
simply covering less of the catalog so far.

Extends rose_batches1_10_production_vs_windowcheck.py's approach (same two directories, same
rose-drawing/paired-comparison machinery) to whatever batch range is currently complete, and
swaps that script's two QC tiers for this session's single grade-3 filter (T.GRADES[3], per
explicit user follow-up): SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg.

Batch numbers are read from disk at run time (glob of the windowcheck output directory), not
hardcoded, so re-running this script later (as more batches finish) picks up more data
automatically.

Produces: rose_axec2_windowcheck_vs_newdata_grade3_matched_batches.pdf (1 page: 2 rose panels
side by side, plus a paired per-event comparison text summary) -- a NEW file, does not touch
any existing output.

Run with:
    python3 rose_axec2_windowcheck_vs_newdata_grade3_matched_batches.py
"""

import glob
import os
import re
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
NEWDATA_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
WC_DIR = os.path.join(HERE, 'production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results')
OUT_PDF = os.path.join(HERE, 'rose_axec2_windowcheck_vs_newdata_grade3_matched_batches.pdf')

NEWDATA_PATTERN = 'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_{}.csv'
WC_PATTERN = 'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_batch_{}.csv'
WC_GLOB_RE = re.compile(r'_batch_(\d+)\.csv$')

GRADE = dict(label='Grade 3: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg',
             snr_min=2.0, qw_min=0.75, phi_err_max=20.0, dt_err_max=0.05)

NBINS = 36


def get_completed_batch_nums():
    """Batch numbers currently present in the windowcheck output directory (whatever's done
    so far), read from disk -- not hardcoded."""
    paths = glob.glob(os.path.join(WC_DIR, WC_PATTERN.format('*')))
    nums = sorted(int(WC_GLOB_RE.search(os.path.basename(p)).group(1)) for p in paths)
    return nums


def load_batches(directory, filename_pattern, batch_nums):
    dfs = []
    for b in batch_nums:
        path = os.path.join(directory, filename_pattern.format(b))
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        if len(d) > 0:
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df['dominant_period'] = df['chosen_filter_dom_period_samples'] / 200.0
    df['phi_az'] = df['phi'] % 180.0
    return df


def apply_grade3(df):
    d = df.dropna(subset=['quality', 'snr_horizontal', 'dt_error', 'dominant_period', 'phi_error'])
    d = d[(d['snr_horizontal'] >= GRADE['snr_min']) &
         (d['quality'] >= GRADE['qw_min']) &
         (d['dt_error'] <= GRADE['dt_err_max']) &
         (d['dt'] <= d['dominant_period'] / 2.0) &
         (d['phi_error'] <= GRADE['phi_err_max'])]
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


def build_paired_summary(new_df, wc_df):
    p = new_df[['event_id', 'phi_az', 'dt', 'quality']].rename(
        columns={'phi_az': 'phi_az_new', 'dt': 'dt_new', 'quality': 'quality_new'})
    w = wc_df[['event_id', 'phi_az', 'dt', 'quality']].rename(
        columns={'phi_az': 'phi_az_wc', 'dt': 'dt_wc', 'quality': 'quality_wc'})
    m = p.merge(w, on='event_id', how='inner')

    m['dphi'] = circular_mean_diff_deg(m['phi_az_wc'].values, m['phi_az_new'].values)
    m['ddt'] = m['dt_wc'] - m['dt_new']

    n_both = len(m)
    n_phi_changed = int((np.abs(m['dphi']) > 10.0).sum())
    n_dt_changed = int((np.abs(m['ddt']) > 0.01).sum())
    n_identical = int(((np.abs(m['dphi']) < 1e-6) & (np.abs(m['ddt']) < 1e-6)).sum())

    lines = [
        f"Paired per-event comparison (both pass grade-3 in both datasets): N={n_both}",
        f"  Identical result (both phi and dt unchanged): {n_identical} ({100*n_identical/max(n_both,1):.1f}%)",
        f"  |d(phi)| > 10 deg (mod-180, circular): {n_phi_changed} ({100*n_phi_changed/max(n_both,1):.1f}%)",
        f"  |d(dt)| > 0.01 s: {n_dt_changed} ({100*n_dt_changed/max(n_both,1):.1f}%)",
        f"  Mean |d(phi)| = {np.abs(m['dphi']).mean():.1f} deg, median = {np.abs(m['dphi']).median():.1f} deg, max = {np.abs(m['dphi']).max():.1f} deg",
        f"  Mean |d(dt)|  = {np.abs(m['ddt']).mean():.4f} s, median = {np.abs(m['ddt']).median():.4f} s, max = {np.abs(m['ddt']).max():.4f} s",
        f"  Mean quality: newdata = {m['quality_new'].mean():.3f}, window-check = {m['quality_wc'].mean():.3f}",
    ]
    return '\n'.join(lines)


def make_figure(new_df, wc_df, paired, batch_nums):
    methods = [('Newdata (production, non-windowcheck)', new_df, '#CC0000'),
              ('Window-check (in progress)', wc_df, '#0072B2')]

    fig = plt.figure(figsize=(9, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.55], hspace=0.45, wspace=0.15)

    for ci, (method_label, df, color) in enumerate(methods):
        sub = apply_grade3(df)
        ax = fig.add_subplot(gs[0, ci], projection='polar')
        _draw_rose(ax, sub['phi_az'].values, color)
        ax.set_title(f'{method_label}\nN={len(sub):,}', fontsize=9.5, fontweight='bold', pad=10)

    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')
    ax_text.text(0.0, 1.0, paired, fontsize=9, family='monospace', va='top', ha='left',
                transform=ax_text.transAxes)

    n_batches = len(batch_nums)
    b_lo, b_hi = min(batch_nums), max(batch_nums)
    fig.suptitle(
        f'AXEC2 batches {b_lo}-{b_hi} ({n_batches} batches done so far): '
        f'newdata vs. windowcheck (in progress)\n{GRADE["label"]}',
        fontsize=11, fontweight='bold', y=0.99)
    return fig


def main():
    batch_nums = get_completed_batch_nums()
    print(f'{len(batch_nums)} windowcheck batches complete so far '
         f'(range {min(batch_nums)}-{max(batch_nums)})')

    new_all = load_batches(NEWDATA_DIR, NEWDATA_PATTERN, batch_nums)
    wc_all = load_batches(WC_DIR, WC_PATTERN, batch_nums)
    print(f'Newdata (matched batches): {len(new_all):,} successful events')
    print(f'Window-check (matched batches): {len(wc_all):,} successful events')

    new_g3 = apply_grade3(new_all)
    wc_g3 = apply_grade3(wc_all)
    print(f'Newdata grade-3: {len(new_g3):,}')
    print(f'Window-check grade-3: {len(wc_g3):,}')

    paired_text = build_paired_summary(new_g3, wc_g3)
    print('\n' + paired_text)

    fig = make_figure(new_all, wc_all, paired_text, batch_nums)
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
