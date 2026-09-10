#!/usr/bin/env python3
"""
rose_plots_lqt_pykonal_unweighted.py

Count-weighted ("unweighted") fast-direction rose plots for the LQT + PyKonal-FMM
production run, built from the combined per-station CSVs in
lqt_pykonal_combined_results/ (new naming vs. rose_plots_temporal.py's
splitting_unweighted.pdf, so neither script overwrites the other's output).
AXEC2's 2015-2021 half is pulled in separately from the per-batch CSVs in
production_axec2_lqt_pykonal_results/, since lqt_pykonal_combined_results/
only has AXEC2's 2022-2026 half combined.

Restricted to good-quality measurements only:
    quality (Q_w, Wustefeld 2010) > 0.5
    phi_error < 20 deg
    dt_error  < 0.04 s

The production pipeline that writes lqt_pykonal_combined_results/ already
applies the incidence < 35 deg QC cut itself (true PyKonal-FMM ray tracing
through the Baillard 3D Vs model, "LQT + PyKonal-FMM incidence (35 deg cut)"),
so no incidence filter is re-applied here -- re-tracing every event's ray from
scratch here as well was redundant and, at the default (non-stride) tracer
resolution across all 6 stations' full combined datasets, prohibitively slow.

Layout matches rose_plots_temporal.py: 6-row (station) x N-col (time period)
grid, doubled-angle rose histograms, count-weighted (each measurement counts
once regardless of dt).

Each output PDF is nine pages:
    page 1 — quality filter only (as above)
    page 2 — quality filter + dt <= 0.8 * max_dt (max_dt = 0.30 s), dropping
             measurements pinned near the grid-search's upper dt bound
    page 3 — same as page 1 but with a higher quality threshold, Q_w >= 0.7
    page 4 — "null" page: Q_w < 0 (Wustefeld 2010 null identification), no
             phi_error/dt_error cut
    page 5 — "true splits": Q_w >= 0.7, no phi_error/dt_error cut
    page 6 — "true nulls": Q_w <= -0.7, no phi_error/dt_error cut
    page 7 — "gray zone": -0.7 < Q_w < 0.7, no phi_error/dt_error cut
    page 8 — "indeterminate quality": -0.2 < Q_w < 0.2, no phi_error/dt_error cut
    page 9 — Q_w > -0.5, no phi_error/dt_error cut

Produces two PDFs in lqt_pykonal_combined_results/:
    lqt_pykonal_splitting_unweighted.pdf        — eruption-relative bins
    lqt_pykonal_splitting_unweighted_annual.pdf — annual bins

Run with:
    python3 rose_plots_lqt_pykonal_unweighted.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

import glob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_lqt_pykonal_results')
OUT_DIR = DATA_DIR

QW_MIN = 0.5
QW_MIN_STRICT = 0.7
QW_NULL_MAX = 0.0   # Q_w < 0 -> null identification (Wustefeld 2010)
QW_TRUE_SPLIT_MIN = 0.7    # Q_w >= 0.7 -> "true splits"
QW_TRUE_NULL_MAX = -0.7    # Q_w <= -0.7 -> "true nulls"; in between -> "gray zone"
QW_INDETERMINATE_ABS = 0.2  # -0.2 < Q_w < 0.2 -> "indeterminate quality"
QW_ABOVE_NEG_HALF = -0.5    # Q_w > -0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04

MAX_DT = 0.30
DT_CUTOFF = MAX_DT * 0.8

MAX_INCIDENCE = 35.0  # deg from vertical -- already applied upstream, in the production pipeline

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

# ── Station order, labels, and available combined files ─────────────────────

STATION_ORDER = ['AXAS2', 'AXAS1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

SECTION_LABELS = {
    'AXAS2': 'Western Caldera',
    'AXCC1': 'Central Caldera',
    'AXEC1': 'Eastern Caldera',
}

STATION_FILES = {
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv',
              'splitting_results_AXAS2_2022_2026_all_batches.csv'],
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
              'splitting_results_AXAS1_2022_2026_all_batches.csv'],
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
              'splitting_results_AXEC3_2022_2026_all_batches.csv'],
}


def load_station_raw(sta):
    """Load the combined LQT + PyKonal results for one station, with only the
    baseline (non-quality-threshold) cleanup applied."""
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = glob.glob(os.path.join(
            AXEC2_2015_2021_DIR, 'splitting_results_mldd_2015_2021_axec2_batch_*.csv'))
        dfs.extend(pd.read_csv(f) for f in batch_files)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]  # remove null measurements
    df = df.dropna(subset=['quality', 'phi_error', 'dt_error'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True)
    df['phi_az'] = df['phi'] % 180.0
    return df


def load_station(sta):
    """Baseline good-quality filter: quality > QW_MIN, phi_error < PHI_ERR_MAX,
    dt_error < DT_ERR_MAX."""
    df = load_station_raw(sta)
    return df[(df['quality'] > QW_MIN) &
              (df['phi_error'] < PHI_ERR_MAX) &
              (df['dt_error'] < DT_ERR_MAX)]


def load_station_strict(sta):
    """Same as load_station() but with a higher quality threshold, Q_w >= QW_MIN_STRICT."""
    df = load_station_raw(sta)
    return df[(df['quality'] >= QW_MIN_STRICT) &
              (df['phi_error'] < PHI_ERR_MAX) &
              (df['dt_error'] < DT_ERR_MAX)]


def load_station_null(sta):
    """'Null' measurements: Q_w < QW_NULL_MAX (Wustefeld 2010 null identification),
    no phi_error/dt_error cut."""
    df = load_station_raw(sta)
    return df[df['quality'] < QW_NULL_MAX]


def load_station_true_splits(sta):
    """'True splits': Q_w >= QW_TRUE_SPLIT_MIN, no phi_error/dt_error cut
    (Wustefeld 2010 quality classification)."""
    df = load_station_raw(sta)
    return df[df['quality'] >= QW_TRUE_SPLIT_MIN]


def load_station_true_nulls(sta):
    """'True nulls': Q_w <= QW_TRUE_NULL_MAX, no phi_error/dt_error cut
    (Wustefeld 2010 quality classification)."""
    df = load_station_raw(sta)
    return df[df['quality'] <= QW_TRUE_NULL_MAX]


def load_station_gray_zone(sta):
    """'Gray zone': QW_TRUE_NULL_MAX < Q_w < QW_TRUE_SPLIT_MIN (neither a
    confident split nor a confident null), no phi_error/dt_error cut."""
    df = load_station_raw(sta)
    return df[(df['quality'] > QW_TRUE_NULL_MAX) & (df['quality'] < QW_TRUE_SPLIT_MIN)]


def load_station_indeterminate(sta):
    """'Indeterminate quality': -QW_INDETERMINATE_ABS < Q_w < QW_INDETERMINATE_ABS
    (a narrower band around Q_w=0 than 'gray zone'), no phi_error/dt_error cut."""
    df = load_station_raw(sta)
    return df[(df['quality'] > -QW_INDETERMINATE_ABS) & (df['quality'] < QW_INDETERMINATE_ABS)]


def load_station_above_neg_half(sta):
    """Q_w > QW_ABOVE_NEG_HALF (-0.5), no phi_error/dt_error cut."""
    df = load_station_raw(sta)
    return df[df['quality'] > QW_ABOVE_NEG_HALF]


# ── Period definitions (verbatim from rose_plots_temporal.py) ───────────────

def _build_time_periods(all_df):
    """7 periods: pre, syn, 5 equal-count post-eruption bins."""
    post = all_df[all_df['t'] >= ERUPTION_END].copy()
    post_times = post['t'].sort_values().reset_index(drop=True)
    n = len(post_times)

    boundaries = [ERUPTION_END]
    for i in range(1, 5):
        idx = int(round(i * n / 5))
        idx = min(idx, n - 1)
        boundaries.append(post_times.iloc[idx])
    boundaries.append(None)

    def _fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else 'present'

    periods = [
        ('Pre-eruption', None, ERUPTION_START),
        ('Syn-eruption', ERUPTION_START, ERUPTION_END),
    ]
    for i in range(5):
        t0, t1 = boundaries[i], boundaries[i + 1]
        periods.append((f'{_fmt(t0)}\n– {_fmt(t1)}', t0, t1))

    return periods


def _build_annual_periods():
    """Annual bins: Pre-eruption | Syn-eruption | Post-2015 | 2016 ... 2026."""
    periods = [
        ('Pre-eruption\n2015', None, ERUPTION_START),
        ('Syn-eruption\n2015', ERUPTION_START, ERUPTION_END),
        ('Post-eruption\n2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for yr in range(2016, 2027):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr+1}-01-01', tz='UTC') if yr < 2026 else None
        periods.append((str(yr), t0, t1))
    return periods


def _subset(df, t_start, t_end):
    m = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
    if t_end is not None:
        m = m & (df['t'] < t_end)
    return df[m]


# ── Rose drawing (verbatim from rose_plots_temporal.py) ─────────────────────

NBINS = 36


def _draw_rose(ax, phi_az_vals, weights, color):
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

    ax.bar(centers, counts, width=width, bottom=0,
           color=color, edgecolor='black', linewidth=0.5, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)


def _period_colors(n_periods):
    colors = ['#800080', '#CC0000']  # pre-eruption (purple), syn (red)
    n_post = n_periods - 2
    start = np.array(mcolors.to_rgb('#ADD8E6'))
    end = np.array(mcolors.to_rgb('#800080'))
    for i in range(n_post):
        t = i / max(n_post - 1, 1)
        colors.append(mcolors.to_hex((1 - t) * start + t * end))
    return colors


# ── Main figure builder (verbatim layout from rose_plots_temporal.py) ───────

def make_rose_figure(dfs, time_periods, title):
    """6-row (station) x N-col (time period) grid, count-weighted."""
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_size = 1.8
    row_gap = 0.10
    left_margin = 0.75
    top_margin = 0.65
    section_gap = 0.45

    section_rows = {sta: i for i, sta in enumerate(STATION_ORDER) if sta in SECTION_LABELS}

    fig_h = (n_rows * (panel_size + row_gap) + top_margin
             + len(section_rows) * section_gap)
    fig_w = n_cols * panel_size + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold', y=1.002)

    row_tops = []
    y_cursor = fig_h - top_margin
    for row_idx, sta in enumerate(STATION_ORDER):
        if sta in SECTION_LABELS:
            y_cursor -= section_gap
        row_tops.append(y_cursor)
        y_cursor -= (panel_size + row_gap)

    col_lefts = [left_margin + col_idx * panel_size for col_idx in range(n_cols)]

    for row_idx, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        cell_top = row_tops[row_idx]

        for col_idx, (label, t_start, t_end) in enumerate(time_periods):
            sub = _subset(df, t_start, t_end)

            x0 = col_lefts[col_idx] / fig_w
            y0 = (cell_top - panel_size) / fig_h
            w = panel_size / fig_w
            h = panel_size / fig_h

            ax = fig.add_axes([x0, y0, w, h], projection='polar')

            phi_vals = sub['phi_az'].values
            wts = np.ones(len(sub))  # unweighted = count-weighted

            _draw_rose(ax, phi_vals, wts, colors[col_idx])

            n_events = len(sub)

            if row_idx == 0:
                ax.set_title(f'{label}\nN={n_events:,}', fontsize=6.5,
                             fontweight='bold', pad=2)
            else:
                ax.set_title(f'N={n_events:,}', fontsize=6.5,
                             fontweight='bold', pad=2)

        y_center = (cell_top - panel_size / 2) / fig_h
        fig.text(0.0, y_center, sta, fontsize=11, fontweight='bold',
                  ha='left', va='center', rotation=0,
                  transform=fig.transFigure)

    for sta, row_idx in section_rows.items():
        cell_top = row_tops[row_idx]
        y_label = (cell_top + section_gap * 0.55) / fig_h
        fig.text(left_margin / fig_w, y_label,
                  SECTION_LABELS[sta],
                  fontsize=13, fontweight='bold', color='black',
                  ha='left', va='bottom',
                  transform=fig.transFigure,
                  fontfamily='Arial')

    return fig


def main():
    inc_note = f'incidence<{MAX_INCIDENCE:.0f}$\\degree$ (applied upstream, true PyKonal-FMM)'
    filter_note = (f'Q$_w$>{QW_MIN}, $\\phi$ err<{PHI_ERR_MAX}$\\degree$, '
                    f'$\\delta$t err<{DT_ERR_MAX}s, {inc_note}')
    dtcut_note = filter_note + f', $\\delta$t$\\leq${DT_CUTOFF:.2f}s (0.8$\\times$max_dt={MAX_DT}s)'
    strict_note = (f'Q$_w\\geq${QW_MIN_STRICT}, $\\phi$ err<{PHI_ERR_MAX}$\\degree$, '
                   f'$\\delta$t err<{DT_ERR_MAX}s, {inc_note}')
    null_note = f'Q$_w$<{QW_NULL_MAX} (null), {inc_note}'
    true_split_note = f'Q$_w\\geq${QW_TRUE_SPLIT_MIN} ("true splits"), {inc_note}'
    true_null_note = f'Q$_w\\leq${QW_TRUE_NULL_MAX} ("true nulls"), {inc_note}'
    gray_zone_note = (f'{QW_TRUE_NULL_MAX}<Q$_w$<{QW_TRUE_SPLIT_MIN} '
                      f'("gray zone"), {inc_note}')
    indeterminate_note = (f'-{QW_INDETERMINATE_ABS}<Q$_w$<{QW_INDETERMINATE_ABS} '
                          f'("indeterminate quality"), {inc_note}')
    above_neg_half_note = f'Q$_w$>{QW_ABOVE_NEG_HALF}, {inc_note}'

    print(f'Loading LQT + PyKonal-FMM combined results (incidence < {MAX_INCIDENCE} deg '
          f'already applied upstream, quality > {QW_MIN}, phi_error < {PHI_ERR_MAX} deg, '
          f'dt_error < {DT_ERR_MAX} s)...')

    dfs = {}
    dfs_dtcut = {}
    dfs_strict = {}
    dfs_null = {}
    dfs_true_split = {}
    dfs_true_null = {}
    dfs_gray_zone = {}
    dfs_indeterminate = {}
    dfs_above_neg_half = {}
    for sta in STATION_ORDER:
        print(f'  Loading {sta}...')
        raw = load_station_raw(sta)
        df = raw[(raw['quality'] > QW_MIN) &
                 (raw['phi_error'] < PHI_ERR_MAX) &
                 (raw['dt_error'] < DT_ERR_MAX)]
        dfs[sta] = df
        dfs_dtcut[sta] = df[df['dt'] <= DT_CUTOFF]
        dfs_strict[sta] = raw[(raw['quality'] >= QW_MIN_STRICT) &
                               (raw['phi_error'] < PHI_ERR_MAX) &
                               (raw['dt_error'] < DT_ERR_MAX)]
        dfs_null[sta] = raw[raw['quality'] < QW_NULL_MAX]
        dfs_true_split[sta] = raw[raw['quality'] >= QW_TRUE_SPLIT_MIN]
        dfs_true_null[sta] = raw[raw['quality'] <= QW_TRUE_NULL_MAX]
        dfs_gray_zone[sta] = raw[(raw['quality'] > QW_TRUE_NULL_MAX) &
                                  (raw['quality'] < QW_TRUE_SPLIT_MIN)]
        dfs_indeterminate[sta] = raw[(raw['quality'] > -QW_INDETERMINATE_ABS) &
                                      (raw['quality'] < QW_INDETERMINATE_ABS)]
        dfs_above_neg_half[sta] = raw[raw['quality'] > QW_ABOVE_NEG_HALF]
        print(f'  {sta}: {len(raw):,} incidence<{MAX_INCIDENCE:.0f}deg measurements -> '
              f'{len(df):,} good-quality '
              f'({len(dfs_dtcut[sta]):,} after dt<={DT_CUTOFF:.2f}s cut, '
              f'{len(dfs_strict[sta]):,} at Q_w>={QW_MIN_STRICT}, '
              f'{len(dfs_null[sta]):,} null, '
              f'{len(dfs_true_split[sta]):,} true splits, '
              f'{len(dfs_true_null[sta]):,} true nulls, '
              f'{len(dfs_gray_zone[sta]):,} gray zone, '
              f'{len(dfs_indeterminate[sta]):,} indeterminate, '
              f'{len(dfs_above_neg_half[sta]):,} Q_w>{QW_ABOVE_NEG_HALF})')

    all_combined = pd.concat(dfs.values(), ignore_index=True)

    # ── Eruption-relative bins ───────────────────────────────────────────────
    time_periods = _build_time_periods(all_combined)

    fig = make_rose_figure(
        dfs, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — {filter_note}',
    )
    fig_dtcut = make_rose_figure(
        dfs_dtcut, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — {dtcut_note}',
    )
    fig_strict = make_rose_figure(
        dfs_strict, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — {strict_note}',
    )
    fig_null = make_rose_figure(
        dfs_null, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — {null_note}',
    )
    fig_true_split = make_rose_figure(
        dfs_true_split, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — true splits — {true_split_note}',
    )
    fig_true_null = make_rose_figure(
        dfs_true_null, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — true nulls — {true_null_note}',
    )
    fig_gray_zone = make_rose_figure(
        dfs_gray_zone, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — gray zone — {gray_zone_note}',
    )
    fig_indeterminate = make_rose_figure(
        dfs_indeterminate, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — indeterminate quality — {indeterminate_note}',
    )
    fig_above_neg_half = make_rose_figure(
        dfs_above_neg_half, time_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted — {above_neg_half_note}',
    )
    out_path = os.path.join(OUT_DIR, 'lqt_pykonal_splitting_unweighted.pdf')
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_dtcut, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_strict, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_null, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_true_split, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_true_null, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_gray_zone, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_indeterminate, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_above_neg_half, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig_dtcut)
    plt.close(fig_strict)
    plt.close(fig_null)
    plt.close(fig_true_split)
    plt.close(fig_true_null)
    plt.close(fig_gray_zone)
    plt.close(fig_indeterminate)
    plt.close(fig_above_neg_half)
    print(f'Saved {out_path}')

    # ── Annual bins ───────────────────────────────────────────────────────────
    annual_periods = _build_annual_periods()

    fig_annual = make_rose_figure(
        dfs, annual_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted annual — {filter_note}',
    )
    fig_annual_dtcut = make_rose_figure(
        dfs_dtcut, annual_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted annual — {dtcut_note}',
    )
    fig_annual_strict = make_rose_figure(
        dfs_strict, annual_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted annual — {strict_note}',
    )
    fig_annual_null = make_rose_figure(
        dfs_null, annual_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted annual — {null_note}',
    )
    fig_annual_true_split = make_rose_figure(
        dfs_true_split, annual_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted annual — true splits — {true_split_note}',
    )
    fig_annual_true_null = make_rose_figure(
        dfs_true_null, annual_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted annual — true nulls — {true_null_note}',
    )
    fig_annual_gray_zone = make_rose_figure(
        dfs_gray_zone, annual_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted annual — gray zone — {gray_zone_note}',
    )
    fig_annual_indeterminate = make_rose_figure(
        dfs_indeterminate, annual_periods,
        title=(f'LQT + PyKonal-FMM fast direction rose — unweighted annual — '
               f'indeterminate quality — {indeterminate_note}'),
    )
    fig_annual_above_neg_half = make_rose_figure(
        dfs_above_neg_half, annual_periods,
        title=f'LQT + PyKonal-FMM fast direction rose — unweighted annual — {above_neg_half_note}',
    )
    out_annual = os.path.join(OUT_DIR, 'lqt_pykonal_splitting_unweighted_annual.pdf')
    with PdfPages(out_annual) as pdf:
        pdf.savefig(fig_annual, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_annual_dtcut, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_annual_strict, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_annual_null, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_annual_true_split, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_annual_true_null, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_annual_gray_zone, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_annual_indeterminate, dpi=300, bbox_inches='tight')
        pdf.savefig(fig_annual_above_neg_half, dpi=300, bbox_inches='tight')
    plt.close(fig_annual)
    plt.close(fig_annual_dtcut)
    plt.close(fig_annual_strict)
    plt.close(fig_annual_null)
    plt.close(fig_annual_true_split)
    plt.close(fig_annual_true_null)
    plt.close(fig_annual_gray_zone)
    plt.close(fig_annual_indeterminate)
    plt.close(fig_annual_above_neg_half)
    print(f'Saved {out_annual}')

    print('\nDone.')


if __name__ == '__main__':
    main()
