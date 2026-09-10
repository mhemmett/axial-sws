"""
Batches-1-153 comparison: original (fixed 5-40 Hz) AXEC2 vs. mfast try_filters
(max_t_shift_s default 0.30s) vs. mfast try_filters with the grid search
constrained to max_t_shift_s=0.2s (run_axec2_mfast_filters_maxdt02_batches.py), restricted
to the same event window (batches 1-153, 2015-01-22 to 2015-03-22, all pre-eruption) --
i.e. as many batches as the max_dt=0.2 re-run has completed so far -- so the three columns
are as close to an apples-to-apples comparison as the three pipelines allow.

Supersedes the earlier 10-batch smoke-test version of this comparison (same script, same
filenames) now that the mfast max_dt=0.2 re-run has progressed to 153/497 batches. Batch
range/window here should be bumped again (and BATCH_END_NUM's docstring/filenames updated)
as the re-run progresses further -- check the highest completed batch in
production_axec2_mfast_filters_maxdt02_lqt_pykonal_results/ before reusing this script.

5 pages, one cumulative QC tier added per page:
    Page 1: raw (success & dt>0)
    Page 2: + Q_w >= 0.5
    Page 3: + dt <= T_dom/2 (cycle-skip-risk cut, per-event dominant period)
    Page 4: + phi_error < 20 deg
    Page 5: + dt_error < 0.04s

The original (fixed 5-40 Hz) dataset does not track a per-event chosen-filter dominant
period (chosen_filter_dom_period_samples), so the T_dom/2 cut (page 3 onward) cannot be
applied to it - its column on pages 3-5 carries the page-2 (Q_w>=0.5) subset forward
unfiltered by dt<=T_dom/2, labeled "N/A: T_dom not tracked".

Produces two separate PDFs, each with one page per filter tier:
    rose_axec2_batches1_153_original_vs_mfast_vs_mfast_maxdt02.pdf
        1 row x 3 columns: unweighted rose plot (doubled-angle 180 deg symmetry) per dataset
        (original / mfast max_dt=0.3 / mfast max_dt=0.2), reusing the drawing convention from
        build_production_rose_plots_axec2_qw05.py.
    rose_axec2_batches1_153_original_vs_mfast_vs_mfast_maxdt02_histograms.pdf
        1 row x 3 columns: overlaid histograms (quality Q_w, fast direction, delay time), one
        panel per quantity, each panel overlaying all 3 datasets (same colors as the rose PDF)
        at alpha=0.5.

Run with:
    python3 rose_axec2_batches1_153_original_vs_mfast_vs_mfast_maxdt02.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
NBINS = 36

PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04
QW_MIN = 0.5

BATCH_START_NUM = 1
BATCH_END_NUM = 153
WINDOW_START = pd.Timestamp('2015-01-22 00:00', tz='UTC')
WINDOW_END = pd.Timestamp('2015-03-23 00:00', tz='UTC')
WINDOW_LABEL = '2015-01-22 to 2015-03-22'

DATASETS = [
    ('Original (fixed 5-40 Hz,\nmax_dt=0.3s)',
     os.path.join(HERE, 'production_axec2_lqt_pykonal_results',
                  'splitting_results_mldd_2015_2021_axec2_batch_{n}.csv')),
    ('mfast try_filters\n(max_dt=0.3s)',
     os.path.join(HERE, 'production_axec2_mfast_filters_lqt_pykonal_results',
                  'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_{n}.csv')),
    ('mfast try_filters\n(max_dt=0.2s)',
     os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results',
                  'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_{n}.csv')),
]


def load(path_pattern):
    files = [path_pattern.format(n=n) for n in range(BATCH_START_NUM, BATCH_END_NUM + 1)]
    files = [f for f in files if os.path.exists(f)]
    dfs = [pd.read_csv(f) for f in files]
    dfs = [d for d in dfs if len(d) > 0]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    df = df[(df['t'] >= WINDOW_START) & (df['t'] < WINDOW_END)]
    df['phi_az'] = df['phi'] % 180.0
    if 'chosen_filter_dom_period_samples' in df.columns:
        df['dom_period_s'] = df['chosen_filter_dom_period_samples'] / 200.0
    else:
        df['dom_period_s'] = np.nan
    return df


def _draw_rose(ax, phi_az_vals, weights, color):
    """Verbatim drawing convention from build_production_rose_plots_axec2_qw05.py."""
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('none')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(0.8)
    ax.grid(False)

    if len(phi_az_vals) == 0:
        ax.set_ylim(0, 1)
        return

    doubled_angles = []
    doubled_weights = []
    for phi, w in zip(phi_az_vals, weights):
        p = float(phi) % 360.0
        doubled_angles.extend([np.deg2rad(p), np.deg2rad((p + 180.0) % 360.0)])
        doubled_weights.extend([w, w])

    doubled_angles = np.array(doubled_angles)
    doubled_weights = np.array(doubled_weights)

    bins = np.linspace(0, 2 * np.pi, NBINS + 1)
    counts, edges = np.histogram(doubled_angles, bins=bins, weights=doubled_weights)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 2 * np.pi / NBINS

    ax.bar(centers, counts, width=width, bottom=0, color=color, edgecolor='black', linewidth=0.5, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)


def apply_tier(df, tier_idx):
    """Returns (subset, na_note) - na_note is set when this dataset can't support this tier's
    filter (missing T_dom), in which case subset carries forward the prior tier's result."""
    d = df.dropna(subset=['quality'])
    if tier_idx == 0:
        return df, None

    m = d['quality'] >= QW_MIN
    d1 = d[m]
    if tier_idx == 1:
        return d1, None

    has_dom_period = d1['dom_period_s'].notna().any()
    if not has_dom_period:
        d2 = d1
        na_note = 'N/A: T_dom not tracked'
    else:
        d2 = d1.dropna(subset=['dom_period_s'])
        d2 = d2[d2['dt'] <= d2['dom_period_s'] / 2.0]
        na_note = None
    if tier_idx == 2:
        return d2, na_note

    d3 = d2.dropna(subset=['phi_error'])
    d3 = d3[d3['phi_error'] < PHI_ERR_MAX]
    if tier_idx == 3:
        return d3, na_note

    d4 = d3.dropna(subset=['dt_error'])
    d4 = d4[d4['dt_error'] < DT_ERR_MAX]
    return d4, na_note


TIERS = [
    'Raw (success & dt>0)',
    f'+ Q_w >= {QW_MIN}',
    '+ dt <= T_dom/2 (cycle-skip-risk cut)',
    f'+ phi_err < {PHI_ERR_MAX:.0f} deg',
    f'+ dt_err < {DT_ERR_MAX}s',
]

COLORS = ['#888888', '#CC0000', '#0055CC']

HIST_ALPHA = 0.5
QUALITY_BINS = np.linspace(-1.0, 1.0, 21)
PHI_BINS = np.linspace(0.0, 180.0, 19)
DT_BINS = np.linspace(0.0, 0.30, 31)


def compute_subsets(dfs, tier_idx):
    subs, na_notes = [], []
    for df in dfs:
        sub, na_note = apply_tier(df, tier_idx)
        subs.append(sub)
        na_notes.append(na_note)
    return subs, na_notes


def make_rose_page(dfs, tier_idx):
    subs, na_notes = compute_subsets(dfs, tier_idx)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), subplot_kw={'projection': 'polar'})
    for col, ((label, _path), sub, na_note) in enumerate(zip(DATASETS, subs, na_notes)):
        ax = axes[col]
        weights = np.ones(len(sub))
        _draw_rose(ax, sub['phi_az'].values, weights, COLORS[col])
        max_dt = sub['dt'].max() if len(sub) else float('nan')
        title = f'{label}\nN={len(sub):,}, max dt={max_dt:.2f}s' if len(sub) else f'{label}\nN=0'
        if na_note:
            title += f'\n({na_note})'
        ax.set_title(title, fontsize=8.5, fontweight='bold', pad=10)
    fig.suptitle(f'AXEC2 batches {BATCH_START_NUM}-{BATCH_END_NUM} ({WINDOW_LABEL}, pre-eruption): {TIERS[tier_idx]}',
                 fontsize=11, fontweight='bold', y=1.05)
    fig.tight_layout()
    return fig


def make_histogram_page(dfs, tier_idx):
    subs, _na_notes = compute_subsets(dfs, tier_idx)
    fig, hist_axes = plt.subplots(1, 3, figsize=(9, 3.2))

    hist_specs = [
        (hist_axes[0], 'quality', QUALITY_BINS, 'Quality (Q$_w$)'),
        (hist_axes[1], 'phi_az', PHI_BINS, 'Fast direction (deg)'),
        (hist_axes[2], 'dt', DT_BINS, 'Delay time (s)'),
    ]
    for ax, col_name, bins, xlabel in hist_specs:
        for col, ((label, _path), sub) in enumerate(zip(DATASETS, subs)):
            if len(sub) == 0:
                continue
            ax.hist(sub[col_name].values, bins=bins, color=COLORS[col], alpha=HIST_ALPHA,
                    edgecolor=COLORS[col], linewidth=0.8, label=label.replace('\n', ' '))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=7)
    hist_axes[0].legend(fontsize=6, loc='upper right')

    fig.suptitle(f'AXEC2 batches {BATCH_START_NUM}-{BATCH_END_NUM} ({WINDOW_LABEL}, pre-eruption): {TIERS[tier_idx]}',
                 fontsize=11, fontweight='bold', y=1.03)
    fig.tight_layout()
    return fig


def main():
    dfs = [load(path) for _, path in DATASETS]
    for (label, _), df in zip(DATASETS, dfs):
        print(f'{label.splitlines()[0]}: {len(df)} baseline events (success & dt>0)')

    rose_path = os.path.join(HERE, 'rose_axec2_batches1_153_original_vs_mfast_vs_mfast_maxdt02.pdf')
    with PdfPages(rose_path) as pdf:
        for tier_idx in range(len(TIERS)):
            fig = make_rose_page(dfs, tier_idx)
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {rose_path}')

    hist_path = os.path.join(HERE, 'rose_axec2_batches1_153_original_vs_mfast_vs_mfast_maxdt02_histograms.pdf')
    with PdfPages(hist_path) as pdf:
        for tier_idx in range(len(TIERS)):
            fig = make_histogram_page(dfs, tier_idx)
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {hist_path}')


if __name__ == '__main__':
    main()
