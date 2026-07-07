"""
Rebuild the rose-plot PDF from the already-computed
lqt_zne_incidence_comparison_axec2_batch1_results.csv, without rerunning the splitting
analysis - use this when only QUALITY_PANELS/VERSIONS labels/plotting cosmetics change.

Run with:
    python3 replot_rose_from_cached_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_lqt_zne_incidence_comparison_axec2_batch1 as mod

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, 'lqt_zne_incidence_comparison_axec2_batch1_results.csv')


def main():
    df = pd.read_csv(RESULTS_CSV)

    fig, axes = plt.subplots(len(mod.VERSIONS), len(mod.QUALITY_PANELS),
                              figsize=(4 * len(mod.QUALITY_PANELS), 16),
                              subplot_kw={'projection': 'polar'})

    for row_i, version in enumerate(mod.VERSIONS):
        label = version['label']
        sub = df[(df['version'] == label) & (df['success'] == True)]

        for col_i, (col_label, quality_test) in enumerate(mod.QUALITY_PANELS):
            ax = axes[row_i, col_i]
            phis = [row['phi'] for _, row in sub.iterrows() if quality_test(row['quality'])]
            mod.draw_rose(ax, phis)

            if row_i == 0:
                ax.text(0.5, 1.35, col_label, transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
            if col_i == 0:
                ax.text(-0.35, 0.5, label, transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, fontweight='bold', rotation=90)

    fig.suptitle(f"{mod.STATION} batch 1 — ZNE vs LQT, Eigenvalue-S vs PyKonal-FMM incidence (35° cut)",
                 fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.03, 0, 1, 0.98])
    fig.savefig(mod.OUT_PDF, dpi=200)
    print(f"Saved {mod.OUT_PDF}")


if __name__ == '__main__':
    main()
