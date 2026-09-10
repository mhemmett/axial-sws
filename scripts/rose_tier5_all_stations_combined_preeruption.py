#!/usr/bin/env python3
"""
rose_tier5_all_stations_combined_preeruption.py

PRE-ERUPTION-ONLY variant of rose_tier5_all_stations_combined.py: identical pooling (all 6
stations, tier-5 QC -- quality>=0.5, dt<T_dom/2, phi_error<10 deg, dt_error<0.05s), but
restricted to events strictly before the 2015 eruption START (t < ERUPTION_START,
2015-04-24 06:00 UTC). Syn-eruption and post-eruption events are excluded.

Reuses rose_7period_6stations_newdata.py's exact station loading (load_station_raw), tier QC
(apply_tier, tier_idx=5), and ERUPTION_START constant.

Produces: rose_tier5_all_stations_combined_preeruption.pdf (1 page, 1 rose)

Run with:
    python3 rose_tier5_all_stations_combined_preeruption.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rose_7period_6stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, _draw_rose, ERUPTION_START,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_tier5_all_stations_combined_preeruption.pdf')

TIER5_IDX = 5
ROSE_COLOR = '#800080'


def main():
    print(f'Loading tier-5 events for all 6 stations, restricted to t < {ERUPTION_START} '
          f'(pre-eruption)...')
    dfs = []
    for sta in STATION_ORDER:
        raw = load_station_raw(sta)
        tier5 = apply_tier(raw, TIER5_IDX)
        post = tier5[tier5['t'] < ERUPTION_START]
        print(f'  {sta}: {len(post):,} (of {len(tier5):,} tier-5 total)')
        dfs.append(post)

    all_df = pd.concat(dfs, ignore_index=True)
    n_total = len(all_df)
    print(f'  TOTAL (all 6 stations, pooled, pre-eruption): {n_total:,}')

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
    _draw_rose(ax, all_df['phi_az'].values, np.ones(n_total), ROSE_COLOR)
    ax.set_title(f'Fast direction, tier 5 (quality>=0.5, dt<T_dom/2, phi_err<10deg, '
                f'dt_err<0.05s)\nAll 6 stations combined, PRE-ERUPTION ONLY '
                f'(< {ERUPTION_START.strftime("%Y-%m-%d")}), N={n_total:,}',
                fontsize=12, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
