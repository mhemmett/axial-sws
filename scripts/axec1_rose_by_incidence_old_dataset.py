#!/usr/bin/env python3
"""
axec1_rose_by_incidence_old_dataset.py

AXEC1-only fast-direction rose plot, all time periods pooled, split into 3
incidence-angle bins: 0-10, 10-20, 20-30 deg. Uses the OLD (pre-LQT/PyKonal)
MLdd production results, which already carry a precomputed `incidence` column
(the old P-wave Jurkevics incidence -- NOT the true PyKonal-FMM incidence used
in the new LQT + PyKonal run). That old dataset is hard-capped at
incidence < 30 deg (the old, since-abandoned incidence QC pre-filter -- see
build_raw_axec1_batch1.py-style comments in build_raw_axec2_batch1.py), which
is why the bins here are 0-10/10-20/20-30 rather than reaching 35 deg.

Data: results/splitting_results_mldd_2015_2021_axec1_all_batches.csv
      + results/splitting_results_mldd_2022_2026_axec1_all_batches.csv

Output: results/axec1_rose_by_incidence_old_dataset.pdf

Run with:
    python3 axec1_rose_by_incidence_old_dataset.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(HERE), 'results')

FILES = [
    'splitting_results_mldd_2015_2021_axec1_all_batches.csv',
    'splitting_results_mldd_2022_2026_axec1_all_batches.csv',
]

INCIDENCE_BINS = [(0, 10), (10, 20), (20, 30)]
NBINS = 36


def load_axec1():
    dfs = [pd.read_csv(os.path.join(RESULTS_DIR, f)) for f in FILES]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['phi', 'incidence', 'dt'])
    df = df[df['dt'] > 0]  # remove null measurements
    df['phi_az'] = df['phi'] % 180.0
    return df


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


def main():
    print('Loading old MLdd AXEC1 results (2015-2021 + 2022-2026)...')
    df = load_axec1()
    print(f'  {len(df):,} measurements (dt>0, non-null)')

    colors = ['#ADD8E6', '#6699CC', '#800080']  # light blue -> purple, low -> high incidence

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), subplot_kw={'projection': 'polar'})
    for ax, (lo, hi), color in zip(axes, INCIDENCE_BINS, colors):
        sub = df[(df['incidence'] >= lo) & (df['incidence'] < hi)]
        _draw_rose(ax, sub['phi_az'].values, color)
        ax.set_title(f'{lo}–{hi}°\nN={len(sub):,}', fontsize=9, fontweight='bold', pad=6)
        print(f'  incidence {lo}-{hi} deg: {len(sub):,}')

    fig.suptitle(f'AXEC1 — fast direction rose by incidence angle (old MLdd dataset, all periods, N={len(df):,})',
                 fontsize=11, fontweight='bold', y=1.05)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, 'axec1_rose_by_incidence_old_dataset.pdf')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
