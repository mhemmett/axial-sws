"""
histograms_all_columns_newdata_combined.py

Histograms of every histogram-able (numeric) column, for ALL of our new mfast max_dt=0.2s
data combined into one pool (AXCC1 + AXEC1 + AXEC2, no per-station/per-period split -- see
rose_7period_axcc1_axec1_axec2_newdata.py for the per-station/per-period breakdown):

  AXCC1: mfast_maxdt_pipeline_transfer/splitting_results_AXCC1_{2015_2021,2022_2026}_all_batches.csv
  AXEC1: mfast_maxdt_pipeline_transfer/splitting_results_AXEC1_{2015_2021,2022_2026}_all_batches.csv
  AXEC2: mfast_maxdt_pipeline_transfer/splitting_results_AXEC2_2022_2026_all_batches.csv
         + our own in-progress maxdt02 re-run's 2015-2021 batches (partial as of this run --
           production_axec2_mfast_filters_maxdt02_lqt_pykonal_results/*.csv)

Column handling:
  - event_id, datetime, success: not histogram-able (identifier / timestamp / all-True flag)
    and excluded.
  - dominant_period: present directly in the mfast_maxdt_pipeline_transfer files; computed as
    chosen_filter_dom_period_samples/200.0 (200 Hz sampling) for the in-progress AXEC2
    2015-2021 batches, which don't carry it directly (same convention used elsewhere in this
    repo, e.g. rose_axec2_batches1_153_original_vs_mfast_vs_mfast_maxdt02.py).
  - incidence_p_jurkevics vs incidence_eigenvalue_jurkevics: same underlying quantity (P-wave
    Jurkevics-eigenvalue incidence angle) under two different column names between the
    mfast_maxdt_pipeline_transfer files and the in-progress AXEC2 batches -- unified into a
    single incidence_p_jurkevics column before histogramming.
  - latitude/longitude/depth: absent from the in-progress AXEC2 2015-2021 batches (event
    metadata isn't joined in that pipeline stage) -- NaN for those rows, dropped per-column
    when histogramming (each column's histogram uses only its own non-NaN values, not a
    single global dropna, so this doesn't reduce sample size for other columns).

Excluded from histogramming (in addition to event_id/datetime/success): latitude, longitude,
incidence_p_jurkevics.

5 cumulative QC tiers, same scheme as rose_7period_axcc1_axec1_axec2_newdata.py -- each tier's
page(s) apply one more filter on top of the previous:
    Tier 0: raw (success & dt>0)
    Tier 1: + quality >= 0.5
    Tier 2: + dt < T_dom/2 (cycle-skip-risk cut, per-event dominant period)
    Tier 3: + phi_error < 20 deg
    Tier 4: + dt_error < 0.05 s

Produces: histograms_all_columns_newdata_combined.pdf (one page per column-group per tier,
4x4 grid of columns per page; tiers appear in order, each with its own page(s))

Run with:
    python3 histograms_all_columns_newdata_combined.py
"""

import glob
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
OUT_PDF = os.path.join(HERE, 'histograms_all_columns_newdata_combined.pdf')

STATION_FILES = {
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
}

EXCLUDE_COLS = {'event_id', 'datetime', 'success', 'latitude', 'longitude', 'incidence_p_jurkevics'}

QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05

TIERS = [
    'Raw (success & dt>0)',
    f'+ quality >= {QW_MIN}',
    '+ dt < T_dom/2 (cycle-skip-risk cut)',
    f'+ phi_err < {PHI_ERR_MAX:.0f} deg',
    f'+ dt_err < {DT_ERR_MAX}s',
]

NCOLS_PER_PAGE = 4
NROWS_PER_PAGE = 4
NBINS = 50


def apply_tier(df, tier_idx):
    if tier_idx == 0:
        return df
    d = df.dropna(subset=['quality'])
    d1 = d[d['quality'] >= QW_MIN]
    if tier_idx == 1:
        return d1
    d2 = d1.dropna(subset=['dominant_period'])
    d2 = d2[d2['dt'] < d2['dominant_period'] / 2.0]
    if tier_idx == 2:
        return d2
    d3 = d2.dropna(subset=['phi_error'])
    d3 = d3[d3['phi_error'] < PHI_ERR_MAX]
    if tier_idx == 3:
        return d3
    d4 = d3.dropna(subset=['dt_error'])
    d4 = d4[d4['dt_error'] < DT_ERR_MAX]
    return d4


def load_all():
    dfs = []
    for sta, files in STATION_FILES.items():
        for f in files:
            dfs.append(pd.read_csv(os.path.join(TRANSFER_DIR, f)))
        print(f'  {sta}: {sum(len(pd.read_csv(os.path.join(TRANSFER_DIR, f))) for f in files):,} '
              f'rows from mfast_maxdt_pipeline_transfer/')

    batch_files = sorted(glob.glob(os.path.join(
        AXEC2_2015_2021_DIR, 'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_*.csv')))
    batch_dfs = [pd.read_csv(f) for f in batch_files]
    batch_dfs = [d for d in batch_dfs if len(d) > 0]
    n_axec2_new = sum(len(d) for d in batch_dfs)
    print(f'  AXEC2 (2015-2021, in-progress re-run, {len(batch_files)} batches): {n_axec2_new:,} rows')
    for d in batch_dfs:
        d['dominant_period'] = d['chosen_filter_dom_period_samples'] / 200.0
        d['incidence_p_jurkevics'] = d['incidence_eigenvalue_jurkevics']
        d.drop(columns=['incidence_eigenvalue_jurkevics'], inplace=True)
    dfs.extend(batch_dfs)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    print(f'\n  Combined: {len(df):,} total events (success==True & dt>0)')
    return df


def main():
    print('Loading all new mfast max_dt=0.2s data...')
    raw = load_all()

    hist_cols = [c for c in raw.columns if c not in EXCLUDE_COLS
                and pd.api.types.is_numeric_dtype(raw[c])]
    print(f'\nHistogramming {len(hist_cols)} numeric columns: {hist_cols}')

    per_page = NCOLS_PER_PAGE * NROWS_PER_PAGE
    with PdfPages(OUT_PDF) as pdf:
        for tier_idx, tier_label in enumerate(TIERS):
            df = apply_tier(raw, tier_idx)
            print(f'  tier {tier_idx} ({tier_label}): {len(df):,} events')
            for page_start in range(0, len(hist_cols), per_page):
                page_cols = hist_cols[page_start:page_start + per_page]
                fig, axes = plt.subplots(NROWS_PER_PAGE, NCOLS_PER_PAGE, figsize=(16, 14))
                axes = np.array(axes).reshape(-1)
                for ax, col in zip(axes, page_cols):
                    vals = df[col].dropna().values
                    vals = vals[np.isfinite(vals)]
                    ax.hist(vals, bins=NBINS, color='#3B6FA0', edgecolor='none')
                    ax.set_title(f'{col}\nN={len(vals):,}', fontsize=9, fontweight='bold')
                    ax.tick_params(labelsize=7)
                for ax in axes[len(page_cols):]:
                    ax.axis('off')
                fig.suptitle(f'All new mfast max_dt=0.2s data (AXCC1+AXEC1+AXEC2) — {tier_label}',
                             fontsize=12, fontweight='bold')
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig, dpi=200)
                plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
