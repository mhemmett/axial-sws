"""
Build the 3-page rose-plot PDF (extends build_two_page_rose_pdf.py with page 3, the
clustering-parameter sweep) for a given station's results CSVs.

Run with:
    python3 build_three_page_rose_pdf.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import run_lqt_zne_incidence_comparison_axec2_batch1 as mod
from build_two_page_rose_pdf import draw_page, PAGE2_ROWS, RESULTS_CSV
from run_page3_cluster_sweep import EPS_VALUES, MIN_SAMPLES_VALUES, OUT_CSV as PAGE3_CSV


def draw_page3(fig, axes, df):
    for row_i, eps in enumerate(EPS_VALUES):
        for col_i, min_samples in enumerate(MIN_SAMPLES_VALUES):
            ax = axes[row_i, col_i]
            sub = df[(df['eps'] == eps) & (df['min_samples'] == min_samples) & (df['success'] == True)]
            phis = sub[sub['quality'] > 0.5]['phi'].tolist()
            mod.draw_rose(ax, phis)

            if row_i == 0:
                ax.text(0.5, 1.35, f"min_samples={min_samples}", transform=ax.transAxes,
                        ha='center', fontsize=10, fontweight='bold')
            if col_i == 0:
                ax.text(-0.35, 0.5, f"eps={eps}", transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, fontweight='bold', rotation=90)


def main():
    df = pd.read_csv(RESULTS_CSV)
    df3 = pd.read_csv(PAGE3_CSV)

    with PdfPages(mod.OUT_PDF) as pdf:
        # Page 1
        fig1, axes1 = plt.subplots(len(mod.VERSIONS), len(mod.QUALITY_PANELS),
                                    figsize=(4 * len(mod.QUALITY_PANELS), 16),
                                    subplot_kw={'projection': 'polar'})
        page1_rows = [{'label': v['label'], 'version': v['label'], 'error_filter': False} for v in mod.VERSIONS]
        draw_page(fig1, axes1, page1_rows, df)
        fig1.suptitle(f"{mod.STATION} batch 1 (page 1) — ZNE vs LQT, Eigenvalue-S vs PyKonal-FMM incidence (35° cut)",
                      fontsize=13, fontweight='bold', y=0.995)
        fig1.tight_layout(rect=[0.03, 0, 1, 0.98])
        pdf.savefig(fig1, dpi=200)
        plt.close(fig1)

        # Page 2
        fig2, axes2 = plt.subplots(len(PAGE2_ROWS), len(mod.QUALITY_PANELS),
                                    figsize=(4 * len(mod.QUALITY_PANELS), 12),
                                    subplot_kw={'projection': 'polar'})
        draw_page(fig2, axes2, PAGE2_ROWS, df)
        fig2.suptitle(f"{mod.STATION} batch 1 (page 2) — old P-Jurkevics incidence, and φ/δt-error-filtered comparisons",
                      fontsize=13, fontweight='bold', y=0.995)
        fig2.tight_layout(rect=[0.03, 0, 1, 0.97])
        pdf.savefig(fig2, dpi=200)
        plt.close(fig2)

        # Page 3
        fig3, axes3 = plt.subplots(len(EPS_VALUES), len(MIN_SAMPLES_VALUES),
                                    figsize=(4 * len(MIN_SAMPLES_VALUES), 4 * len(EPS_VALUES)),
                                    subplot_kw={'projection': 'polar'})
        draw_page3(fig3, axes3, df3)
        fig3.suptitle(f"{mod.STATION} batch 1 (page 3) — DBSCAN clustering-parameter sweep, LQT/PyKonal-FMM, Q_w > 0.5 only",
                      fontsize=13, fontweight='bold', y=0.995)
        fig3.tight_layout(rect=[0.03, 0, 1, 0.97])
        pdf.savefig(fig3, dpi=200)
        plt.close(fig3)

    print(f"Saved 3-page PDF to {mod.OUT_PDF}")


if __name__ == '__main__':
    main()
