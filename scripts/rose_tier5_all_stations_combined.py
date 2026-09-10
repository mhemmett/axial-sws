#!/usr/bin/env python3
"""
rose_tier5_all_stations_combined.py

One-off figure (not part of the pipeline): a SINGLE combined rose plot of fast direction (phi)
pooling tier-5 QC events (this session's tightest filter -- quality>=0.5, dt<T_dom/2,
phi_error<10 deg, dt_error<0.05s) from ALL 6 stations together (AXAS1, AXAS2, AXCC1, AXEC1,
AXEC2, AXEC3), across the full 2015-2026 record (no time-period split, no per-station split --
every tier-5 survivor from every station goes into one rose).

Reuses rose_7period_6stations_newdata.py's exact station loading (load_station_raw) and tier
QC (apply_tier, tier_idx=5) so this pooled population matches that script's own tier-5 page.

Produces: rose_tier5_all_stations_combined.pdf (1 page, 1 rose)

Run with:
    python3 rose_tier5_all_stations_combined.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rose_7period_6stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, _draw_rose,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_tier5_all_stations_combined.pdf')

TIER5_IDX = 5
ROSE_COLOR = '#800080'


def main():
    print('Loading tier-5 events for all 6 stations...')
    dfs = []
    for sta in STATION_ORDER:
        raw = load_station_raw(sta)
        tier5 = apply_tier(raw, TIER5_IDX)
        print(f'  {sta}: {len(tier5):,}')
        dfs.append(tier5)

    import pandas as pd
    all_df = pd.concat(dfs, ignore_index=True)
    n_total = len(all_df)
    print(f'  TOTAL (all 6 stations, pooled): {n_total:,}')

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
    _draw_rose(ax, all_df['phi_az'].values, np.ones(n_total), ROSE_COLOR)
    ax.set_title(f'Fast direction, tier 5 (quality>=0.5, dt<T_dom/2, phi_err<10deg, '
                f'dt_err<0.05s)\nAll 6 stations combined, 2015-2026, N={n_total:,}',
                fontsize=12, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
