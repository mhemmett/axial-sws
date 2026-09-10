#!/usr/bin/env python3
"""
add_real_vs_synth_comparison_pages.py

Appends 2 new pages to EACH of the existing synthetic Hudson rose-plot PDFs
(synthetic_hudson_rose_plots.pdf, synthetic_hudson_rose_plots_two_sills.pdf), comparing the
synthetic pre-/syn-eruption columns side by side with the REAL data's pre-/syn-eruption
columns, per explicit user request:

    Page A: real data at tier 4 (quality>=0.5, dt<T_dom/2, phi_error<20 deg, dt_error<0.05s)
            vs. synthetic, side by side.
    Page B: real data at tier 5 (tier 4 + phi_error<10 deg, stricter) vs. synthetic, side by
            side.

Each page: 6 station rows x 4 columns (Real Pre-eruption, Synthetic Pre-eruption,
Real Syn-eruption, Synthetic Syn-eruption), same purple/red pre/syn color convention used
throughout this session. Real data is loaded fresh (same tier-4/5 QC as rose_7period_
6stations_newdata.py); synthetic data is read from the already-saved CSVs (NOT re-run -- no
ray tracing needed here).

Uses PyPDF2 to merge these 2 new pages onto the END of each existing PDF (the original
per-scenario page(s) are preserved, not regenerated), so this doesn't touch the driver
scripts' own ray-tracing output.

Run with:
    python3 add_real_vs_synth_comparison_pages.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfReader, PdfWriter

from rose_7period_6stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, _draw_rose, ERUPTION_START, ERUPTION_END,
)

HERE = os.path.dirname(os.path.abspath(__file__))

TIER4_IDX = 4
TIER5_IDX = 5
TIER_LABELS = {
    TIER4_IDX: 'tier 4: quality>=0.5, dt<T_dom/2, phi_err<20deg, dt_err<0.05s',
    TIER5_IDX: 'tier 5: tier 4 + phi_err<10deg (stricter)',
}

PRE_COLOR = '#800080'
SYN_COLOR = '#CC0000'


def _lighten(hex_color, frac=0.45):
    """Blend hex_color toward white by `frac` (0=no change, 1=white) -- used so the
    synthetic columns are visibly distinguishable from the real-data columns at a glance,
    per explicit user request, while keeping the same pre/syn hue."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex((1 - frac) * rgb + frac * np.array([1., 1., 1.]))


PRE_COLOR_SYNTH = _lighten(PRE_COLOR)
SYN_COLOR_SYNTH = _lighten(SYN_COLOR)

# (synthetic CSV, existing rose-plot PDF to append to)
# NOTE: the earlier "_v2" iteration chain was retired and fully baked into
# synthetic_hudson_crack_scenarios.py's base SILL/DIKE_WAYPOINTS (so "single-sill"/"two-sills"
# below use that consolidated geometry). A NEW, distinct "_v2" variant has since been created
# (sill1 moved 0.25 km west of AXEC1, only the south-going dike segment redrawn) -- these are
# genuinely different models now, both listed below.
MODEL_FILES = [
    ('synthetic_hudson_pre_syn_phidt.csv', 'synthetic_hudson_rose_plots.pdf', 'Single-sill model'),
    ('synthetic_hudson_pre_syn_phidt_two_sills.csv', 'synthetic_hudson_rose_plots_two_sills.pdf',
     'Two-sills model'),
    # rest below are unchanged -- no cosine-similarity data exists for these (v2 / sill2deep),
    # so COSINE_MODEL_KEY (below) deliberately has no entry for them.
    ('synthetic_hudson_pre_syn_phidt_v2.csv', 'synthetic_hudson_rose_plots_v2.pdf',
     'Single-sill model v2 (sill1 moved 0.25km west of AXEC1)'),
    ('synthetic_hudson_pre_syn_phidt_two_sills_v2.csv', 'synthetic_hudson_rose_plots_two_sills_v2.pdf',
     'Two-sills model v2 (sill1 moved 0.25km west of AXEC1)'),
    ('synthetic_hudson_pre_syn_phidt_two_sills_sill2deep.csv',
     'synthetic_hudson_rose_plots_two_sills_sill2deep.pdf',
     'Two-sills model, sill2 deepened (3.33km, |dP|=200MPa, calibrated to -2.4m eruption-onset subsidence)'),
]


# Maps a MODEL_FILES synth_csv filename to the 'model' label used in
# compute_synth_vs_real_cosine_similarity.py's output CSV -- only the single-sill and
# two-sills (shallow sill2) models have cosine-similarity data (per explicit user request,
# the v2 / sill2deep variants were excluded from that computation).
COSINE_MODEL_KEY = {
    'synthetic_hudson_pre_syn_phidt.csv': 'Single-sill model',
    'synthetic_hudson_pre_syn_phidt_two_sills.csv': 'Two-sills model (shallow sill2)',
}
COSINE_CSV = os.path.join(HERE, 'synth_vs_real_cosine_similarity.csv')


def load_cosine_lookup():
    """Returns {(model_label, tier, station, period): cosine_similarity}, or {} if the CSV
    doesn't exist yet (compute_synth_vs_real_cosine_similarity.py hasn't been run)."""
    if not os.path.exists(COSINE_CSV):
        return {}
    df = pd.read_csv(COSINE_CSV)
    return {(row.model, row.tier, row.station, row.period): row.cosine_similarity
            for row in df.itertuples()}


def load_real_tier(tier_idx):
    """Returns dfs[station] -> tier-filtered dataframe with 't'/'phi_az' columns."""
    raw = {sta: load_station_raw(sta) for sta in STATION_ORDER}
    return {sta: apply_tier(raw[sta], tier_idx) for sta in STATION_ORDER}


def _subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None:
        m = m & (df['t'] < t1)
    return df[m]


def make_comparison_page(real_dfs, synth_df, tier_idx, model_label, cos_lookup=None,
                         cos_model_key=None):
    n_rows = len(STATION_ORDER)
    cols = [
        ('Real: Pre-eruption', PRE_COLOR),
        ('Synthetic: Pre-eruption', PRE_COLOR_SYNTH),
        ('Real: Syn-eruption', SYN_COLOR),
        ('Synthetic: Syn-eruption', SYN_COLOR_SYNTH),
    ]
    # cosine similarity is only meaningful on the SYNTHETIC columns (it measures how well that
    # column's rose shape matches the real column right next to it)
    cos_period_for_col = {1: 'Pre-eruption', 3: 'Syn-eruption'}

    fig, axes = plt.subplots(n_rows, len(cols), figsize=(3.2 * len(cols), 3.2 * n_rows),
                             subplot_kw=dict(projection='polar'))

    for ri, sta in enumerate(STATION_ORDER):
        real_df = real_dfs[sta]
        real_pre = _subset(real_df, None, ERUPTION_START)
        real_syn = _subset(real_df, ERUPTION_START, ERUPTION_END)
        synth_pre = synth_df[(synth_df['station'] == sta) & (synth_df['period'] == 'Pre-eruption')]
        synth_syn = synth_df[(synth_df['station'] == sta) & (synth_df['period'] == 'Syn-eruption')]

        panels = [real_pre['phi_az'].values if len(real_pre) else np.array([]),
                 synth_pre['phi_synth'].values % 180.0 if len(synth_pre) else np.array([]),
                 real_syn['phi_az'].values if len(real_syn) else np.array([]),
                 synth_syn['phi_synth'].values % 180.0 if len(synth_syn) else np.array([])]
        counts = [len(real_pre), len(synth_pre), len(real_syn), len(synth_syn)]

        for ci, (phi_vals, n) in enumerate(zip(panels, counts)):
            ax = axes[ri, ci]
            color = cols[ci][1]
            _draw_rose(ax, phi_vals, np.ones(len(phi_vals)), color)
            title = f'{cols[ci][0]}\nN={n:,}' if ri == 0 else f'N={n:,}'
            if cos_lookup is not None and ci in cos_period_for_col:
                cos_val = cos_lookup.get((cos_model_key, tier_idx, sta, cos_period_for_col[ci]))
                if cos_val is not None and np.isfinite(cos_val):
                    title += f'\ncos_sim={cos_val:.2f}'
            ax.set_title(title, fontsize=8, fontweight='bold')
            if ci == 0:
                ax.text(-0.28, 0.5, sta, fontsize=10, fontweight='bold',
                        ha='center', va='center', transform=ax.transAxes)

    fig.suptitle(f'{model_label}: real (new mfast max_dt=0.2s data, {TIER_LABELS[tier_idx]}) '
                 f'vs. synthetic, side by side', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


ORIGINAL_N_PAGES = 1   # every driver script's rose-plot PDF is a single page before any
                       # comparison pages get appended


def append_pages(pdf_path, new_figs):
    """Replace any previously-appended comparison pages with new_figs -- keeps only the
    FIRST ORIGINAL_N_PAGES page(s) (the driver script's own rose-plot output, never touched)
    and appends new_figs after them, via a temp PDF + PyPDF2 merge. Safe to rerun repeatedly
    (e.g. after a color-scheme tweak) without accumulating duplicate comparison pages."""
    tmp_path = pdf_path + '.new_pages_tmp.pdf'
    with PdfPages(tmp_path) as pdf:
        for fig in new_figs:
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    reader_orig = PdfReader(pdf_path)
    reader_new = PdfReader(tmp_path)
    writer = PdfWriter()
    for page in reader_orig.pages[:ORIGINAL_N_PAGES]:
        writer.add_page(page)
    for page in reader_new.pages:
        writer.add_page(page)
    with open(pdf_path, 'wb') as f:
        writer.write(f)
    os.remove(tmp_path)


def main():
    print('Loading real data (tier 4 and tier 5)...')
    real_tier4 = load_real_tier(TIER4_IDX)
    real_tier5 = load_real_tier(TIER5_IDX)
    for sta in STATION_ORDER:
        print(f'  {sta}: tier4 N={len(real_tier4[sta]):,}, tier5 N={len(real_tier5[sta]):,}')

    cos_lookup = load_cosine_lookup()
    print(f'Loaded {len(cos_lookup)} cosine-similarity entries' if cos_lookup else
          'No cosine-similarity CSV found -- run compute_synth_vs_real_cosine_similarity.py first '
          'for cos_sim annotations')

    for synth_csv, pdf_name, model_label in MODEL_FILES:
        synth_path = os.path.join(HERE, synth_csv)
        pdf_path = os.path.join(HERE, pdf_name)
        if not os.path.exists(synth_path) or not os.path.exists(pdf_path):
            print(f'  Skipping {model_label}: {synth_csv} or {pdf_name} not found')
            continue
        cos_model_key = COSINE_MODEL_KEY.get(synth_csv)
        force_rebuild = cos_model_key is not None and cos_lookup
        n_pages = len(PdfReader(pdf_path).pages)
        if n_pages > 1 and not force_rebuild:
            print(f'  Skipping {model_label}: {pdf_name} already has {n_pages} pages '
                  f'(comparison pages likely already appended -- rerun the driver script '
                  f'first if you want fresh comparison pages)')
            continue
        print(f'\n{model_label}: building comparison pages'
              f'{" (with cosine-similarity annotations)" if force_rebuild else ""}...')
        synth_df = pd.read_csv(synth_path)

        fig_t4 = make_comparison_page(real_tier4, synth_df, TIER4_IDX, model_label,
                                      cos_lookup=cos_lookup, cos_model_key=cos_model_key)
        fig_t5 = make_comparison_page(real_tier5, synth_df, TIER5_IDX, model_label,
                                      cos_lookup=cos_lookup, cos_model_key=cos_model_key)

        append_pages(pdf_path, [fig_t4, fig_t5])
        print(f'  Appended 2 comparison pages to {pdf_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
