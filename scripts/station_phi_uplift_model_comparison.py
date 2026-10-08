#!/usr/bin/env python3
"""
station_phi_uplift_model_comparison.py

Per-STATION companion to region_phi_uplift_model_comparison.py.

The regional comparison found that no functional form is well supported by the pooled
western or eastern means, and that the production atan2 vector-sum model ranks 12/15 (east)
and 15/15 (west) once the series is decimated to ~independent samples. That test used
regional POOLED means, which are a noisier construct than the per-station series the atan2
model was actually built and published on. This script runs the identical fifteen-family
battery on each station individually, which is the fair test of the incumbent.

Everything -- the families, the multi-start fitting, the AICc/BIC scoring, the raw-vs-thinned
autocorrelation guard, the figure layout -- is IMPORTED from the regional script. Only the
grouping changes, so the two comparisons are directly commensurable.

AXEC2's UNLEVELLED WINDOW IS DROPPED
    Handled by load_region_pool (called with a single-station list), which removes AXEC2's
    2021-07 to 2022-09 window. That is the right call for a model comparison -- fitting
    functional forms through a known ~50 deg instrumental drift would score the drift, not
    the physics.

    NOTE THE CONSEQUENCE: the published per-station atan2 figure INCLUDES those points (it
    flags them green rather than dropping them), and reports r=0.74 for AXEC2. The AXEC2
    numbers here are therefore not directly comparable to that figure. Every other station is
    unaffected.

Produces:
    station_phi_uplift_model_comparison.pdf   (one page per station)
    station_phi_uplift_model_comparison.csv   (full ranking, raw and thinned, all stations)

Run with:
    python3 station_phi_uplift_model_comparison.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bpr_inflation_periods_ccal import POST_ERUPTION_START
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_regions_windowcheck_grade3 import (
    STATION_ORDER, _circular_mean_and_se_deg,
)
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
)
from animate_arctan_stress_vectors import compute_optimal_wrap
from atan2_uplift_vs_phi_sixstations_ccal_30day import ROLL_WINDOW_DAYS, ROLL_MIN_DAYS
from atan2_uplift_vs_phi_regions_ccal_30day import load_region_pool
from region_phi_uplift_model_comparison import (
    rank_on, print_table, make_figure, THIN_DAYS,
)
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'station_phi_uplift_model_comparison.pdf')
OUT_CSV = os.path.join(HERE, 'station_phi_uplift_model_comparison.csv')


def main():
    _dd, _raw, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']

    figs, tables, summary = [], [], []

    for sta in STATION_ORDER:
        print(f'\n{"="*74}\n{sta}\n{"="*74}')
        pool = load_region_pool([sta])

        pre = pool[pool['t'] < ERUPTION_START]
        baseline_phi, _ = _circular_mean_and_se_deg(pre['phi_az'].values)
        roll = rolling_phi_stats_daily_then_roll(pool, baseline_phi,
                                                 window_days=ROLL_WINDOW_DAYS,
                                                 min_days=ROLL_MIN_DAYS)

        valid = roll.dropna(subset=['mean_phi']).copy()
        valid = valid[valid['t'] >= POST_ERUPTION_START]
        merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                               direction='nearest', tolerance=pd.Timedelta('20D'))
        merged = merged.dropna(subset=['inflation_m']).reset_index(drop=True)

        if len(merged) < 60:
            print(f'  only {len(merged)} usable points -- skipping')
            continue

        x = merged['inflation_m'].values
        wrap = compute_optimal_wrap(merged['mean_phi'].values)
        y = (merged['mean_phi'].values - wrap) % 180.0

        raw = rank_on(x, y, 'raw (daily)')
        print_table(raw, f'RAW series, N={len(x)} daily points (autocorrelated)')

        t0 = merged['t'].iloc[0]
        bucket = ((merged['t'] - t0).dt.days // THIN_DAYS).values
        first = np.concatenate(([True], bucket[1:] != bucket[:-1]))
        xt, yt = x[first], y[first]
        thin = rank_on(xt, yt, f'thinned (1 per {THIN_DAYS}d)')
        print_table(thin, f'THINNED series, N={len(xt)} points (~independent)')

        inc_raw = raw[raw['model'] == 'atan2 vector-sum'].iloc[0]
        inc_thin = thin[thin['model'] == 'atan2 vector-sum'].iloc[0]
        agree = raw.iloc[0]['model'] == thin.iloc[0]['model']
        print(f'\n  Raw best:     {raw.iloc[0]["model"]}')
        print(f'  Thinned best: {thin.iloc[0]["model"]}')
        print(f'  -> {"AGREE" if agree else "DISAGREE -- raw ranking unreliable"}')
        print(f'  atan2 vector-sum: rank {int(inc_raw["rank"])}/{len(raw)} raw, '
              f'{int(inc_thin["rank"])}/{len(thin)} thinned (dAICc={inc_thin["d_aicc"]:.1f})')

        summary.append(dict(
            station=sta, n_raw=len(x), n_thin=len(xt),
            raw_best=raw.iloc[0]['model'], thin_best=thin.iloc[0]['model'],
            agree=agree,
            atan2_rank_raw=int(inc_raw['rank']), atan2_rank_thin=int(inc_thin['rank']),
            atan2_daicc_thin=float(inc_thin['d_aicc']),
            best_r2_thin=float(thin.iloc[0]['r2']),
            atan2_r2_thin=float(inc_thin['r2']),
            median_n_eff=float(raw['n_eff'].median())))

        for df, tag in ((raw, 'raw (daily)'), (thin, f'thinned (1 per {THIN_DAYS}d)')):
            out = df.drop(columns=['_func', '_popt']).copy()
            out.insert(0, 'station', sta)
            out['series'] = tag
            tables.append(out)

        figs.append(make_figure(sta, x, y, raw, thin, wrap))

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)

    pd.concat(tables, ignore_index=True).to_csv(OUT_CSV, index=False)

    s = pd.DataFrame(summary)
    print(f'\n\n{"="*94}\nPER-STATION SUMMARY\n{"="*94}')
    print(f'{"station":<9}{"raw best":<18}{"thinned best":<18}{"agree":<7}'
          f'{"atan2 rank":<12}{"dAICc":>8}{"best R2":>9}{"atan2 R2":>10}')
    print('-' * 94)
    for _, r in s.iterrows():
        print(f'{r["station"]:<9}{r["raw_best"]:<18}{r["thin_best"]:<18}'
              f'{"yes" if r["agree"] else "NO":<7}'
              f'{str(int(r["atan2_rank_thin"]))+"/15":<12}{r["atan2_daicc_thin"]:>8.1f}'
              f'{r["best_r2_thin"]:>9.3f}{r["atan2_r2_thin"]:>10.3f}')
    print(f'\n(atan2 rank / dAICc / R2 columns are on the THINNED series)')

    s.to_csv(OUT_CSV.replace('.csv', '_summary.csv'), index=False)
    print(f'\nSaved {OUT_PDF} ({len(figs)} pages)')
    print(f'Saved {OUT_CSV} and _summary.csv')


if __name__ == '__main__':
    main()
