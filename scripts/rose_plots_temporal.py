#!/usr/bin/env python3
"""
rose_plots_temporal.py

Fast-direction rose plots for all 6 stations across 7 time periods,
laid out as a 6-row × 7-column figure.

Rows (top to bottom): AXAS2, AXAS1, AXCC1, AXEC1, AXEC2, AXEC3
Cols (left to right):  Pre-eruption | Syn-eruption | 5 equal-count post-eruption bins

Section labels:
  "Western Caldera"  bold, above AXAS2 row
  "Central Caldera"  bold, above AXCC1 row
  "Eastern Caldera"  bold, above AXEC1 row

Produces four PDFs and one combined PDF:
  splitting_weighted.pdf          — all data, weighted by |dt|
  splitting_unweighted.pdf        — all data, count-weighted
  splitting_weighted_filtered.pdf — dt_error < 0.1 s, weighted by |dt|
  splitting_unweighted_filtered.pdf — dt_error < 0.1 s, count-weighted
  splitting_weighted_annual.pdf          — annual bins, weighted by |dt|
  splitting_weighted_annual_filtered.pdf — annual bins, dt_error < 0.1 s, weighted by |dt|
  splitting_rose_all.pdf          — all four weighted figures combined
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['font.family'] = 'Arial'

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE    = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
OUT_DIR = BASE

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

# ── Station order and labels ──────────────────────────────────────────────────

STATION_ORDER = ['AXAS2', 'AXAS1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

SECTION_LABELS = {
    'AXAS2': 'Western Caldera',
    'AXCC1': 'Central Caldera',
    'AXEC1': 'Eastern Caldera',
}

# ── Load data ─────────────────────────────────────────────────────────────────

def _load_station(sta, dt_error_max=None):
    files = {
        'AXAS2': ('splitting_results_mldd_2015_2021_axas2.csv',
                  'splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
        'AXAS1': ('splitting_results_mldd_2015_2021_axas1.csv',
                  'splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
        'AXCC1': ('splitting_results_mldd_2015_2021_axcc1_all_batches.csv',
                  'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
        'AXEC1': ('splitting_results_mldd_2015_2021_axec1_all_batches.csv',
                  'splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
        'AXEC2': ('axial-mldd-2015-2021-axec2.csv',
                  'splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
        'AXEC3': ('splitting_results_mldd_2015_2021_axec3_all_batches.csv',
                  'splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
    }
    f1, f2 = files[sta]
    dfs = []
    for f in (f1, f2):
        df = pd.read_csv(BASE + f)
        df = df.loc[:, :'dt_error'].dropna()
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['dt'] > 0]   # remove null measurements (dt=0 = no splitting detected)
    if dt_error_max is not None:
        df = df[df['dt_error'] < dt_error_max]
    df['t']      = pd.to_datetime(df['event_datetime'], utc=True)
    df['phi_az'] = df['phi'] % 180.0    # geographic azimuth [0, 180]
    return df


# ── Time period computation (matching sws_temporal_ALL) ───────────────────────

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
    """Annual bins: Pre-eruption | Syn-eruption | Post-2015 | 2016 … 2026"""
    periods = [
        ('Pre-eruption\n2015',  None,            ERUPTION_START),
        ('Syn-eruption\n2015',  ERUPTION_START,  ERUPTION_END),
        ('Post-eruption\n2015', ERUPTION_END,    pd.Timestamp('2016-01-01', tz='UTC')),
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


# ── Rose drawing ──────────────────────────────────────────────────────────────

NBINS = 36

def _draw_rose(ax, phi_az_vals, weights, color):
    """Draw one rose on a polar axes.

    phi_az_vals : azimuth in [0, 180] degrees (geographic, CW from N)
    weights     : per-event weight (|dt| for weighted, 1 for unweighted)
    """
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

    # Doubled-angle trick for 180° symmetry
    doubled_angles  = []
    doubled_weights = []
    for phi, w in zip(phi_az_vals, weights):
        p = float(phi) % 360.0
        doubled_angles.extend([np.deg2rad(p), np.deg2rad((p + 180.0) % 360.0)])
        doubled_weights.extend([w, w])

    doubled_angles  = np.array(doubled_angles)
    doubled_weights = np.array(doubled_weights)

    bins   = np.linspace(0, 2 * np.pi, NBINS + 1)
    counts, edges = np.histogram(doubled_angles, bins=bins, weights=doubled_weights)
    centers = (edges[:-1] + edges[1:]) / 2
    width   = 2 * np.pi / NBINS

    ax.bar(centers, counts, width=width, bottom=0,
           color=color, edgecolor='black', linewidth=0.5, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)


# ── Period color scheme ───────────────────────────────────────────────────────

import matplotlib.colors as mcolors

def _period_colors(n_periods=7):
    """Colors for n_periods time periods.

    Pre-eruption: purple, syn-eruption: red,
    remaining post-eruption bins: light blue → purple gradient.
    """
    colors = ['#800080', '#CC0000']   # pre-eruption (purple), syn (red)
    n_post = n_periods - 2
    start  = np.array(mcolors.to_rgb('#ADD8E6'))   # light blue
    end    = np.array(mcolors.to_rgb('#800080'))   # purple
    for i in range(n_post):
        t = i / max(n_post - 1, 1)
        colors.append(mcolors.to_hex((1 - t) * start + t * end))
    return colors


# ── Main figure builder ───────────────────────────────────────────────────────

def make_rose_figure(dfs, time_periods, dt_weighted=True, title=None):
    """Build the 6-row × 7-col rose figure.

    dfs          : dict sta → DataFrame (already filtered/projected)
    time_periods : list of (label, t_start, t_end)
    dt_weighted  : True → weight by |dt|, False → count-weighted
    """
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_size  = 1.8   # inches per cell
    row_gap     = 0.10  # extra vertical gap between every row
    left_margin = 0.75  # for station labels
    top_margin  = 0.65  # for column headers
    section_gap = 0.45  # extra vertical space for section labels

    # Rows that need extra space above them for section labels
    section_rows = {sta: i for i, sta in enumerate(STATION_ORDER) if sta in SECTION_LABELS}

    # Total figure height: rows × (panel + gap) + margins + section gaps
    fig_h = (n_rows * (panel_size + row_gap) + top_margin
             + len(section_rows) * section_gap)
    fig_w = n_cols * panel_size + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=1.002)

    # Precompute row y-positions (top of each row, in inches from bottom)
    # Rows are drawn top-to-bottom
    row_tops = []   # y coordinate of the top of each cell row (inches from bottom)
    y_cursor = fig_h - top_margin
    for row_idx, sta in enumerate(STATION_ORDER):
        if sta in SECTION_LABELS:
            y_cursor -= section_gap
        row_tops.append(y_cursor)
        y_cursor -= (panel_size + row_gap)

    # Column x-positions (left edge of each cell, inches from left)
    col_lefts = [left_margin + col_idx * panel_size for col_idx in range(n_cols)]

    axes_grid = {}

    for row_idx, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        cell_top = row_tops[row_idx]

        for col_idx, (label, t_start, t_end) in enumerate(time_periods):
            sub = _subset(df, t_start, t_end)

            # Convert to figure coordinates [0,1]
            x0 = col_lefts[col_idx] / fig_w
            y0 = (cell_top - panel_size) / fig_h
            w  = panel_size / fig_w
            h  = panel_size / fig_h

            ax = fig.add_axes([x0, y0, w, h], projection='polar')

            phi_vals = sub['phi_az'].values
            if dt_weighted:
                wts = np.abs(sub['dt'].values)
            else:
                wts = np.ones(len(sub))

            _draw_rose(ax, phi_vals, wts, colors[col_idx])

            n_eq = sub['event_datetime'].nunique()

            # Column headers on first row; N= on every row, always bold, same size
            if row_idx == 0:
                ax.set_title(f'{label}\nN={n_eq:,}', fontsize=6.5,
                             fontweight='bold', pad=2)
            else:
                ax.set_title(f'N={n_eq:,}', fontsize=6.5,
                             fontweight='bold', pad=2)

            axes_grid[(row_idx, col_idx)] = ax

        # Station name label on the left
        y_center = (cell_top - panel_size / 2) / fig_h
        fig.text(0.0, y_center, sta, fontsize=11, fontweight='bold',
                 ha='left', va='center', rotation=0,
                 transform=fig.transFigure)

    # Section labels (bold, black, left-aligned above row)
    for sta, row_idx in section_rows.items():
        cell_top = row_tops[row_idx]
        y_label  = (cell_top + section_gap * 0.55) / fig_h
        fig.text(left_margin / fig_w, y_label,
                 SECTION_LABELS[sta],
                 fontsize=13, fontweight='bold', color='black',
                 ha='left', va='bottom',
                 transform=fig.transFigure,
                 fontfamily='Arial')

    return fig


# ── Run ───────────────────────────────────────────────────────────────────────

combined_weighted   = []   # (fig, label)
combined_unweighted = []

for dt_error_filter, suffix in [(None, ''), (0.1, '_filtered')]:
    label_filter = f' (dt_error < {dt_error_filter} s)' if dt_error_filter else ''
    print(f'\nLoading data{label_filter}...')

    dfs = {}
    for sta in STATION_ORDER:
        df = _load_station(sta, dt_error_max=dt_error_filter)
        dfs[sta] = df
        print(f'  {sta}: {len(df):,} events')

    # ── Eruption-based periods ────────────────────────────────────────────────
    all_combined = pd.concat(dfs.values(), ignore_index=True)
    time_periods = _build_time_periods(all_combined)
    print(f'  Time periods (eruption bins):')
    for label, t0, t1 in time_periods:
        t0_str = t0.strftime('%Y-%m-%d') if t0 is not None else 'start'
        t1_str = t1.strftime('%Y-%m-%d') if t1 is not None else 'present'
        print(f'    {label.replace(chr(10)," ")} : {t0_str} → {t1_str}')

    for dt_weighted, kind in [(True, 'weighted'), (False, 'unweighted')]:
        print(f'  Building {kind} figure...')
        fig = make_rose_figure(
            dfs, time_periods, dt_weighted=dt_weighted,
            title=f'Fast direction rose — {kind} {"(dt_error < 0.1 s)" if dt_error_filter else "(all data)"}',
        )
        out = os.path.join(OUT_DIR, f'splitting_{kind}{suffix}.pdf')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        if kind == 'weighted':
            combined_weighted.append((fig, f'Weighted{suffix}'))
        else:
            combined_unweighted.append((fig, f'Unweighted{suffix}'))
        print(f'  Saved {out}')

    # ── Annual periods ────────────────────────────────────────────────────────
    annual_periods = _build_annual_periods()
    print(f'  Time periods (annual bins): {len(annual_periods)} periods')

    for dt_weighted, kind in [(True, 'weighted'), (False, 'unweighted')]:
        print(f'  Building {kind} annual figure...')
        fig_annual = make_rose_figure(
            dfs, annual_periods, dt_weighted=dt_weighted,
            title=f'Fast direction rose — {kind} annual {"(dt_error < 0.1 s)" if dt_error_filter else "(all data)"}',
        )
        out_annual = os.path.join(OUT_DIR, f'splitting_{kind}_annual{suffix}.pdf')
        fig_annual.savefig(out_annual, dpi=300, bbox_inches='tight')
        if kind == 'weighted':
            combined_weighted.append((fig_annual, f'Weighted annual{suffix}'))
        else:
            combined_unweighted.append((fig_annual, f'Unweighted annual{suffix}'))
        print(f'  Saved {out_annual}')

# ── Combined PDFs ─────────────────────────────────────────────────────────────
for combined, fname in [(combined_weighted,   'splitting_rose_all.pdf'),
                        (combined_unweighted, 'splitting_rose_all_unweighted.pdf')]:
    out = os.path.join(OUT_DIR, fname)
    with PdfPages(out) as pdf:
        for fig, label in combined:
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'  Added to {fname}: {label}')
    print(f'Saved {out}')

print('\nDone.')
