"""
Build the 2-page rose-plot PDF from lqt_zne_incidence_comparison_axec2_batch1_results.csv:

Page 1 (unchanged): the 4 original versions (ZNE/Eigenvalue-S, ZNE/PyKonal-FMM,
LQT/Eigenvalue-S, LQT/PyKonal-FMM), columns = all / null (Q_w<=0) / good (Q_w>0) / good (Q_w>0.5).

Page 2 (new): 3 rows, same 4 quality columns -
  1. ZNE / P-Jurkevics cut (old P-wave incidence angle) - unfiltered
  2. Same, but pre-filtered to phi_error < 20 deg and dt_error < 0.04 s
  3. LQT / PyKonal-FMM (page 1's row 4), pre-filtered to phi_error < 20 deg and dt_error < 0.04 s

Run with:
    python3 build_two_page_rose_pdf.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import run_lqt_zne_incidence_comparison_axec2_batch1 as mod

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, 'lqt_zne_incidence_comparison_axec2_batch1_results.csv')

PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04

PAGE2_ROWS = [
    {'label': 'ZNE / P-Jurkevics cut',            'version': 'ZNE / P-Jurkevics cut', 'error_filter': False},
    {'label': 'ZNE / P-Jurkevics cut\n(φ_err<20°, δt_err<0.04s)', 'version': 'ZNE / P-Jurkevics cut', 'error_filter': True},
    {'label': 'LQT / PyKonal-FMM\n(φ_err<20°, δt_err<0.04s)',     'version': 'LQT / PyKonal-FMM',     'error_filter': True},
]


def draw_page(fig, axes, rows, df):
    for row_i, row_spec in enumerate(rows):
        sub = df[(df['version'] == row_spec['version']) & (df['success'] == True)]
        if row_spec.get('error_filter'):
            sub = sub[(sub['phi_error'] < PHI_ERR_MAX) & (sub['dt_error'] < DT_ERR_MAX)]

        for col_i, (col_label, quality_test) in enumerate(mod.QUALITY_PANELS):
            ax = axes[row_i, col_i] if len(rows) > 1 else axes[col_i]
            phis = [r['phi'] for _, r in sub.iterrows() if quality_test(r['quality'])]
            mod.draw_rose(ax, phis)

            if row_i == 0:
                ax.text(0.5, 1.35, col_label, transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
            if col_i == 0:
                ax.text(-0.35, 0.5, row_spec['label'], transform=ax.transAxes, ha='center', va='center',
                        fontsize=9, fontweight='bold', rotation=90)


def main():
    df = pd.read_csv(RESULTS_CSV)

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

    print(f"Saved 2-page PDF to {mod.OUT_PDF}")


if __name__ == '__main__':
    main()
