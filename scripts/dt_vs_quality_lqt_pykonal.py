#!/usr/bin/env python3
"""
dt_vs_quality_lqt_pykonal.py

Delay time (dt) vs. quality (Q_w, Wustefeld 2010) density plot for the
LQT + PyKonal-FMM production run, pooled across all 6 stations. Uses the same
baseline cleanup as rose_plots_lqt_pykonal_unweighted.py's load_station_raw()
(success==True, dt>0) -- no quality/error thresholding, so the full Q_w range
(including nulls, Q_w<0) is visible.

Also reports mean dt overall vs. mean dt restricted to Q_w >= 0.7.

Output: lqt_pykonal_combined_results/dt_vs_quality_lqt_pykonal.pdf

Run with:
    python3 dt_vs_quality_lqt_pykonal.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rose_plots_lqt_pykonal_unweighted import load_station_raw, STATION_ORDER

OUT_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')

QW_STRICT = 0.7


def main():
    print('Loading LQT + PyKonal-FMM combined results (all stations, baseline cleanup only)...')
    dfs = []
    for sta in STATION_ORDER:
        df = load_station_raw(sta)
        dfs.append(df)
        print(f'  {sta}: {len(df):,}')
    all_df = pd.concat(dfs, ignore_index=True)
    print(f'  TOTAL: {len(all_df):,}')

    mean_dt_all = all_df['dt'].mean()
    strict = all_df[all_df['quality'] >= QW_STRICT]
    mean_dt_strict = strict['dt'].mean()

    print(f'\nMean dt overall (N={len(all_df):,}): {mean_dt_all:.4f} s')
    print(f'Mean dt for Q_w >= {QW_STRICT} (N={len(strict):,}): {mean_dt_strict:.4f} s')

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(all_df['quality'], all_df['dt'], gridsize=80, cmap='viridis',
                    bins='log', mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log$_{10}$(count + 1)', fontsize=9)

    ax.axvline(QW_STRICT, color='red', linestyle='--', linewidth=1.2,
               label=f'Q$_w$={QW_STRICT}')
    ax.axhline(mean_dt_all, color='black', linestyle=':', linewidth=1.2,
               label=f'mean $\\delta$t (all) = {mean_dt_all:.3f} s')
    ax.axhline(mean_dt_strict, color='crimson', linestyle=':', linewidth=1.2,
               label=f'mean $\\delta$t (Q$_w\\geq${QW_STRICT}) = {mean_dt_strict:.3f} s')

    ax.set_xlabel('Quality Q$_w$ (Wustefeld 2010)', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Delay time $\delta t$ [s]', fontsize=11, fontweight='bold')
    ax.set_title(f'LQT + PyKonal-FMM — $\\delta$t vs. quality (N={len(all_df):,}, all 6 stations)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

    out_path = os.path.join(OUT_DIR, 'dt_vs_quality_lqt_pykonal.pdf')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
