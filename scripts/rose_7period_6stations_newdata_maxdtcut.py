#!/usr/bin/env python3
"""
rose_7period_6stations_newdata_maxdtcut.py

One-off variant (not part of the pipeline) of rose_7period_6stations_newdata.py's tier-5 page,
per explicit user request to test an alternative cycle-skip-risk cut: instead of the usual
`dt < T_dom/2` (per-event dominant period), this uses a FIXED cut `dt < max_t_shift_s * 0.8`
(0.2 * 0.8 = 0.16s), motivated by the grid-search wraparound investigation this session (the
production pipeline's max_t_shift_s=0.2s override, and the observation that the winning-window
solution's own analysis window can be close to max_t_shift_s in length for high-frequency
events -- so a cut anchored to max_t_shift_s itself, rather than to the event's own T_dom, is
being explored as an alternative/complementary QC criterion).

Filter (otherwise identical to tier 5 in rose_7period_6stations_newdata.py): quality>=0.5,
dt<MAX_DT_CUT (0.16s, fixed), phi_error<10deg (stricter), dt_error<0.05s.

Reports total event counts (per station and combined) for this alternative cut, for direct
comparison against the standard tier-5 (dt<T_dom/2) counts already reported by
rose_7period_6stations_newdata.py.

Produces: rose_7period_6stations_newdata_maxdtcut.pdf (1 page, 7-period x 6-station rose grid)

Run with:
    python3 rose_7period_6stations_newdata_maxdtcut.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_6stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, _build_time_periods, make_rose_figure,
    QW_MIN, PHI_ERR_MAX_STRICT, DT_ERR_MAX,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_7period_6stations_newdata_maxdtcut.pdf')

MAX_T_SHIFT_S = 0.2   # production override (see run_axec2_mfast_filters_maxdt02_batches.py)
MAX_DT_CUT = MAX_T_SHIFT_S * 0.8   # 0.16s


def apply_maxdtcut_tier(df):
    """Same as rose_7period_6stations_newdata.py's tier 5, except the dt<T_dom/2 cut is
    replaced by a fixed dt<MAX_DT_CUT=0.16s cut."""
    d = df.dropna(subset=['quality'])
    d = d[d['quality'] >= QW_MIN]
    d = d[d['dt'] < MAX_DT_CUT]
    d = d.dropna(subset=['phi_error'])
    d = d[d['phi_error'] < PHI_ERR_MAX_STRICT]
    d = d.dropna(subset=['dt_error'])
    d = d[d['dt_error'] < DT_ERR_MAX]
    return d


def main():
    print('Loading new mfast max_dt=0.2s splitting results for all 6 stations...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
        print(f'  {sta}: {len(raw[sta]):,} baseline events (success & dt>0)')

    # Standard tier-5 (dt<T_dom/2) for direct comparison:
    print(f'\n=== Standard tier 5 (quality>={QW_MIN}, dt<T_dom/2, phi_err<{PHI_ERR_MAX_STRICT}deg, '
          f'dt_err<{DT_ERR_MAX}s) ===')
    tier5_std = {sta: apply_tier(raw[sta], 5) for sta in STATION_ORDER}
    total_std = 0
    for sta in STATION_ORDER:
        n = len(tier5_std[sta])
        total_std += n
        print(f'  {sta}: {n:,}')
    print(f'  TOTAL: {total_std:,}')

    # New dt<MAX_DT_CUT=0.16s variant:
    print(f'\n=== Alternative tier 5 (quality>={QW_MIN}, dt<{MAX_DT_CUT}s [=max_t_shift_s*0.8], '
          f'phi_err<{PHI_ERR_MAX_STRICT}deg, dt_err<{DT_ERR_MAX}s) ===')
    dfs = {sta: apply_maxdtcut_tier(raw[sta]) for sta in STATION_ORDER}
    total_new = 0
    for sta in STATION_ORDER:
        n = len(dfs[sta])
        total_new += n
        print(f'  {sta}: {n:,}')
    print(f'  TOTAL: {total_new:,}')

    print(f'\nDifference: {total_new - total_std:+,} events '
          f'({100*(total_new-total_std)/total_std:+.1f}% relative to standard tier 5)')

    all_combined = pd.concat(raw.values(), ignore_index=True)
    time_periods = _build_time_periods(all_combined)

    with PdfPages(OUT_PDF) as pdf:
        fig = make_rose_figure(
            dfs, time_periods,
            title=(f'AXAS1 / AXAS2 / AXCC1 / AXEC1 / AXEC2 / AXEC3 fast direction rose '
                   f'(new mfast max_dt=0.2s data) — quality>={QW_MIN}, dt<{MAX_DT_CUT}s '
                   f'(=max_t_shift_s*0.8), phi_err<{PHI_ERR_MAX_STRICT}deg, dt_err<{DT_ERR_MAX}s '
                   f'(N={total_new:,}, vs. N={total_std:,} for standard tier 5)'))
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
