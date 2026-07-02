#!/usr/bin/env python3
"""
rose_plots_baz.py

New version of our rose plots with 6 back-azimuth sub-roses arranged in a
ring around each central rose — one sub-rose per 60° BAZ bin.

Produces one PDF (splitting_baz.pdf) with 4 pages:
  Page 1: weighted (all data)
  Page 2: unweighted (all data)
  Page 3: weighted (dt_error < 0.1 s)
  Page 4: unweighted (dt_error < 0.1 s)

Layout: rows = stations (AXAS2→AXEC3), cols = 7 time periods.
Adapted from Baillard's plot_fast_direction_rose_annual_baz_dt_weighted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import os

matplotlib.rcParams['font.family'] = 'Arial'

BASE    = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
OUT_PDF = os.path.join(BASE, 'splitting_baz.pdf')

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

STATION_ORDER = ['AXAS2', 'AXAS1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
SECTION_LABELS = {
    'AXAS2': 'Western Caldera',
    'AXCC1': 'Central Caldera',
    'AXEC1': 'Eastern Caldera',
}

FILES = {
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv',
             'splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv',
             'splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv',
             'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv',
             'splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv',
             'splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv',
             'splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

# Back-azimuth bins: 6 × 60° bins
BAZ_BINS = np.arange(0, 361, 60)
BAZ_CTRS = (BAZ_BINS[:-1] + BAZ_BINS[1:]) / 2   # [30, 90, 150, 210, 270, 330]

# Colors
_POST_START = np.array(mcolors.to_rgb('#ADD8E6'))
_POST_END   = np.array(mcolors.to_rgb('#800080'))
def _period_colors(n=7):
    """Generate n colors: purple, red, then a gradient back to purple."""
    cols = ['#800080', '#CC0000']
    n_grad = max(n - 2, 1)
    for i in range(n_grad):
        t = i / max(n_grad - 1, 1)
        cols.append(mcolors.to_hex((1-t)*_POST_START + t*_POST_END))
    return cols[:n]


def _build_time_periods(all_df):
    """7 data-driven time periods: pre, syn, 5 equal-count post."""
    post = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post)
    bounds = [ERUPTION_END]
    for i in range(1, 5):
        idx = min(int(round(i*n/5)), n-1)
        bounds.append(post['t'].iloc[idx])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds = [('Pre-eruption', None, ERUPTION_START),
           ('Syn-eruption', ERUPTION_START, ERUPTION_END)]
    for i in range(5):
        t0,t1 = bounds[i],bounds[i+1]
        pds.append((f'{fmt(t0)}\n–{fmt(t1)}', t0, t1))
    return pds


def _subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None: m = m & (df['t'] < t1)
    return df[m]


def _fill_rose(ax, phi_vals, weights, nbins, color, lw=0.5):
    """Draw a polar rose histogram (doubled angles, optional weighting)."""
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('none')
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(lw * 1.5)
    ax.grid(False)
    if len(phi_vals) == 0:
        ax.set_ylim(0, 1); return

    # Vectorised doubled-angle trick for 180° symmetry
    phi_arr = np.asarray(phi_vals, dtype=float) % 360.0
    wts_arr = np.asarray(weights,  dtype=float)
    dbl_ang = np.deg2rad(np.concatenate([phi_arr, (phi_arr + 180) % 360]))
    dbl_wt  = np.concatenate([wts_arr, wts_arr])
    bins = np.linspace(0, 2*np.pi, nbins+1)
    counts, edges = np.histogram(dbl_ang, bins=bins, weights=dbl_wt)
    centers = (edges[:-1]+edges[1:]) / 2
    width = 2*np.pi / nbins
    ax.bar(centers, counts, width=width, bottom=0,
           color=color, edgecolor='black', linewidth=lw, alpha=1.0)
    ax.set_ylim(0, counts.max()*1.25 if counts.max() > 0 else 1)


def make_baz_page(dfs, time_periods, dt_weighted, title):
    """
    Build one full figure page with central rose + 6 BAZ sub-roses per cell.
    dfs       : dict sta → DataFrame
    dt_weighted: True = weight by |dt|, False = count weight
    """
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    # Layout geometry (all in inches)
    panel_w   = 2.5     # inches per column
    panel_h   = 2.5     # inches per row
    row_gap   = 0.15    # gap between rows
    left_margin = 1.0
    top_margin  = 0.65
    section_gap = 0.45
    section_rows = {s: i for i, s in enumerate(STATION_ORDER) if s in SECTION_LABELS}

    # Sub-rose ring geometry (relative to panel_scale)
    panel_scale = min(panel_w, panel_h)
    main_r   = panel_scale * 0.18    # central rose radius — smaller
    ring_r_h = panel_scale * 0.35    # horizontal ring radius
    ring_r_v = panel_scale * 0.42    # vertical ring radius (top/bottom sub-roses)

    small_r  = panel_scale * 0.12    # sub-rose radius
    # Figure height
    fig_h = (n_rows*(panel_h+row_gap) + top_margin
             + len(section_rows)*section_gap)
    fig_w = n_cols*panel_w + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.995)

    # Row top y-positions (inches from bottom)
    row_tops = []
    y_cursor = fig_h - top_margin
    for sta in STATION_ORDER:
        if sta in SECTION_LABELS:
            y_cursor -= section_gap
        row_tops.append(y_cursor)
        y_cursor -= (panel_h + row_gap)

    # Pre-compute subsets
    subsets = []
    n_events = []
    for _, t0, t1 in time_periods:
        frames = [_subset(df, t0, t1) for df in dfs.values()]
        # We need per-station subsets; store as dict
        sub_dict = {sta: _subset(dfs[sta], t0, t1) for sta in STATION_ORDER}
        subsets.append(sub_dict)
        n_events.append(sum(len(v) for v in sub_dict.values()))

    for row_idx, sta in enumerate(STATION_ORDER):
        df_sta  = dfs[sta]
        cell_top = row_tops[row_idx]
        cy_in    = cell_top - panel_h / 2    # cell centre y (inches from bottom)

        for col_idx, ((label, t0, t1), sub_dict) in enumerate(zip(time_periods, subsets)):
            sub = sub_dict[sta]
            cx_in = left_margin + (col_idx + 0.5) * panel_w   # cell centre x

            color = colors[col_idx]
            phi_vals = sub['phi_az'].values if len(sub) > 0 else np.array([])
            wts = np.abs(sub['dt'].values) if dt_weighted else np.ones(len(sub))
            neq = sub['event_datetime'].nunique() if len(sub) > 0 else 0

            # ── Central rose ──────────────────────────────────────────────────
            ax_main = fig.add_axes(
                [(cx_in - main_r)/fig_w, (cy_in - main_r)/fig_h,
                 2*main_r/fig_w,        2*main_r/fig_h],
                projection='polar')
            _fill_rose(ax_main, phi_vals, wts, nbins=36, color=color, lw=0.8)

            # Column header on first station row; N= on all rows
            if row_idx == 0:
                ax_main.set_title(f'{label}\nN={neq:,}', fontsize=6.5,
                                  fontweight='bold', pad=2)
            else:
                ax_main.set_title(f'N={neq:,}', fontsize=5.5,
                                  fontweight='bold', pad=1)

            # ── 6 BAZ sub-roses ───────────────────────────────────────────────
            for baz_lo, baz_hi, baz_c in zip(BAZ_BINS[:-1], BAZ_BINS[1:], BAZ_CTRS):
                baz_rad = np.deg2rad(baz_c)
                sc_cx = cx_in + ring_r_h * np.sin(baz_rad)
                sc_cy = cy_in + ring_r_v * np.cos(baz_rad)

                ax_s = fig.add_axes(
                    [(sc_cx - small_r)/fig_w, (sc_cy - small_r)/fig_h,
                     2*small_r/fig_w,         2*small_r/fig_h],
                    projection='polar')

                baz_mask = ((sub['back_azimuth'] >= baz_lo) &
                            (sub['back_azimuth'] <  baz_hi))
                sub_baz  = sub[baz_mask]
                phi_b    = sub_baz['phi_az'].values
                wts_b    = (np.abs(sub_baz['dt'].values) if dt_weighted
                            else np.ones(len(sub_baz)))
                _fill_rose(ax_s, phi_b, wts_b, nbins=18, color=color, lw=0.4)
                # small BAZ label
                n_baz = len(sub_baz)
                ax_s.set_title(f'{int(baz_lo)}–{int(baz_hi)}°\nN={n_baz:,}',
                               fontsize=4.0, pad=1)

        # Station name on left
        y_center = (cell_top - panel_h/2) / fig_h
        fig.text(0.0, y_center, sta, fontsize=11, fontweight='bold',
                 ha='left', va='center', transform=fig.transFigure)

        # Section label above row
        if sta in SECTION_LABELS:
            y_label = (cell_top + section_gap*0.55) / fig_h
            fig.text(left_margin/fig_w, y_label, SECTION_LABELS[sta],
                     fontsize=13, fontweight='bold', color='black',
                     ha='left', va='bottom', transform=fig.transFigure,
                     fontfamily='Arial')

    return fig


# ── Run ───────────────────────────────────────────────────────────────────────

VARIANTS = [
    ('weighted',   True,  None,  '(dt-weighted, all data)'),
    ('unweighted', False, None,  '(count-weighted, all data)'),
    ('weighted',   True,  0.1,   '(dt-weighted, dt_error < 0.1 s)'),
    ('unweighted', False, 0.1,   '(count-weighted, dt_error < 0.1 s)'),
]

with PdfPages(OUT_PDF) as pdf:
    for kind, dt_weighted, dt_err_max, desc in VARIANTS:
        label = f'(dt_error < {dt_err_max} s)' if dt_err_max else '(all data)'
        print(f'\nLoading {desc}...')

        def _load(f):
            d = pd.read_csv(BASE+f).loc[:, :'dt_error'].dropna()
            d = d[d['dt'] > 0]
            if dt_err_max: d = d[d['dt_error'] < dt_err_max]
            return d

        dfs = {}
        for sta, (f1, f2) in FILES.items():
            df = pd.concat([_load(f1), _load(f2)], ignore_index=True)
            df['t']      = pd.to_datetime(df['event_datetime'], utc=True)
            df['phi_az'] = df['phi'] % 180.0
            dfs[sta]     = df
            print(f'  {sta}: {len(df):,} events')

        all_df = pd.concat(dfs.values(), ignore_index=True)
        periods = _build_time_periods(all_df)

        title = (f'Axial Seamount — Fast direction roses with BAZ sub-roses\n'
                 f'{desc}')
        print(f'Building figure...')
        fig = make_baz_page(dfs, periods, dt_weighted, title)
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Added page: {kind} {label}')

print(f'\nSaved {OUT_PDF}')


# ── Annual version ─────────────────────────────────────────────────────────────

def build_annual_periods():
    pds = [
        ('Pre-eruption\n2015',  None,            ERUPTION_START),
        ('Syn-eruption\n2015',  ERUPTION_START,  ERUPTION_END),
        ('Post-eruption\n2015', ERUPTION_END,    pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for yr in range(2016, 2027):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr+1}-01-01', tz='UTC') if yr < 2026 else None
        pds.append((str(yr), t0, t1))
    return pds

OUT_PDF_ANNUAL = os.path.join(BASE, 'splitting_baz_annual.pdf')

with PdfPages(OUT_PDF_ANNUAL) as pdf:
    for kind, dt_weighted, dt_err_max, desc in VARIANTS:
        label = f'(dt_error < {dt_err_max} s)' if dt_err_max else '(all data)'
        print(f'Annual: Loading {desc}...')

        def _load_a(f):
            d = pd.read_csv(BASE+f).loc[:, :'dt_error'].dropna()
            d = d[d['dt'] > 0]
            if dt_err_max: d = d[d['dt_error'] < dt_err_max]
            return d

        dfs_a = {}
        for sta, (f1, f2) in FILES.items():
            df = pd.concat([_load_a(f1), _load_a(f2)], ignore_index=True)
            df['t']      = pd.to_datetime(df['event_datetime'], utc=True)
            df['phi_az'] = df['phi'] % 180.0
            dfs_a[sta]   = df
            print(f'  {sta}: {len(df):,} events')

        periods_a = build_annual_periods()
        title_a = f'Axial Seamount — BAZ sub-roses (annual) {desc}'
        fig = make_baz_page(dfs_a, periods_a, dt_weighted, title_a)
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Added page: {kind} {label}')

print(f'Saved {OUT_PDF_ANNUAL}')
print('Done.')
