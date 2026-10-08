#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chart of deformation_scenario_ranking_grade3.py's two CSVs (no recomputation): how well each
Baillard DMODELS scenario's modeled principal-compression azimuth matches the Grade 3 observed
fast directions.

    Left : per-grid Phi RMS misfit (pre grids vs pre-eruption obs, syn grids vs syn-eruption obs)
    Right: pre_i -> syn_j CHANGE misfit, RMS of dPhi_cal - dPhi_obs about the 1:1 line

Both use the circular-mean observed fast direction. Dotted reference: 90/sqrt(3) = 52 deg, the
RMS of a uniformly random axial misfit on [-90, 90] -- bars near it carry no information.
Byte-identical DMODELS grids (def_syn_2 == def_syn_8, def_syn_4 == def_syn_10) are labeled.

Colours: IBM colour-blind-safe blue / orange (pre / syn), passes the dataviz palette validator.

Output: deformation_figures/scenario_ranking_grade3.pdf / .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'deformation_figures')
OUT_BASE = os.path.join(FIG_DIR, 'scenario_ranking_grade3')

RANDOM_RMS = 90.0 / np.sqrt(3.0)
PERIOD_COLORS = {'pre': '#648FFF', 'syn': '#FE6100'}
DUPLICATES = {'syn_2': 'syn_2 (= syn_8)', 'syn_8': 'syn_8 (= syn_2)',
              'syn_4': 'syn_4 (= syn_10)', 'syn_10': 'syn_10 (= syn_4)'}


def _dup_pair_label(pair):
    pre, syn = pair.split('->')
    return f'{pre} → {DUPLICATES.get(syn, syn)}'


def _style(ax, n):
    ax.axvline(RANDOM_RMS, color='0.35', lw=1, ls=':', zorder=0)
    ax.set_xlim(0, 70)
    ax.set_ylim(n - 0.3, -0.7)
    ax.grid(axis='x', alpha=0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def main():
    grids = pd.read_csv(os.path.join(FIG_DIR, 'scenario_ranking_grade3_grids.csv'))
    pairs = pd.read_csv(os.path.join(FIG_DIR, 'scenario_ranking_grade3_pairs.csv'))
    grids = grids.sort_values('rms_fast').reset_index(drop=True)
    pairs = pairs.sort_values('rms_dphi_1to1_fast').reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.2),
                                   gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.45))

    y = np.arange(len(grids))
    ax1.barh(y, grids['rms_fast'], height=0.7,
             color=[PERIOD_COLORS[p] for p in grids['period']], edgecolor='white', lw=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels([DUPLICATES.get(g, g) for g in grids['grid']], fontsize=9)
    for yi, v in zip(y, grids['rms_fast']):
        ax1.text(v + 0.8, yi, f'{v:.0f}°', va='center', fontsize=8, color='0.25')
    _style(ax1, len(grids))
    ax1.set_xlabel('Φ RMS misfit, model vs observed (°)')
    ax1.set_title('Each scenario vs its own period', fontsize=12, fontweight='bold', loc='left')
    ref = plt.Line2D([], [], color='0.35', lw=1, ls=':')
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PERIOD_COLORS[p]) for p in ('pre', 'syn')]
               + [ref],
               labels=['Pre-eruption grid vs pre obs', 'Syn-eruption grid vs syn obs',
                       'Random misfit (52°)'],
               loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8, frameon=False)

    y = np.arange(len(pairs))
    is_ref = pairs['pair'] == 'pre_1->syn_6'
    ax2.barh(y, pairs['rms_dphi_1to1_fast'], height=0.7,
             color=np.where(is_ref, '#000000', '#999999'), edgecolor='white', lw=2)
    ax2.set_yticks(y)
    ax2.set_yticklabels([_dup_pair_label(p) for p in pairs['pair']], fontsize=8.5)
    for yi, v in zip(y, pairs['rms_dphi_1to1_fast']):
        ax2.text(v + 0.8, yi, f'{v:.0f}°', va='center', fontsize=8, color='0.25')
    _style(ax2, len(pairs))
    ax2.set_xlabel('ΔΦ RMS misfit about 1:1, pre → syn change (°)')
    ax2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#000000'),
                        plt.Rectangle((0, 0), 1, 1, color='#999999'), ref],
               labels=['Pre-1 → Syn-6', 'Other pairs', 'Random misfit (52°)'],
               loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8, frameon=False)
    ax2.set_title('Pre → syn change (black: Pre-1 → Syn-6)', fontsize=12, fontweight='bold',
                  loc='left')

    fig.suptitle('Baillard DMODELS scenarios vs Grade 3 fast directions (6 stations)',
                 fontsize=15, fontweight='bold', y=0.99)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
