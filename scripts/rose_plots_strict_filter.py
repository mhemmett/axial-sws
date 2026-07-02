#!/usr/bin/env python3
"""
rose_plots_strict_filter.py

Rose plots with strict phi_error < 15° filter (no dt_error cut).
Produces only this filter variant — no unfiltered versions.

Outputs (all with _strict_filter suffix):
  splitting_weighted_strict_filter.pdf
  splitting_unweighted_strict_filter.pdf
  splitting_weighted_annual_strict_filter.pdf
  splitting_unweighted_annual_strict_filter.pdf
  splitting_baz_strict_filter.pdf
  splitting_baz_annual_strict_filter.pdf
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
OUT_DIR = BASE

PHI_ERR_MAX = 15.0   # strict phi_error filter

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

BAZ_BINS = np.arange(0, 361, 60)
BAZ_CTRS = (BAZ_BINS[:-1] + BAZ_BINS[1:]) / 2

# ── Data loading ───────────────────────────────────────────────────────────────

def _load_all():
    dfs = {}
    for sta, (f1, f2) in FILES.items():
        def _load(f):
            d = pd.read_csv(BASE+f).loc[:, :'dt_error'].dropna()
            d = d[(d['dt'] > 0) & (d['phi_error'] <= PHI_ERR_MAX)]
            return d
        df = pd.concat([_load(f1), _load(f2)], ignore_index=True)
        df['t']      = pd.to_datetime(df['event_datetime'], utc=True)
        df['phi_az'] = df['phi'] % 180.0
        dfs[sta]     = df
        print(f'  {sta}: {len(df):,}')
    return dfs

# ── Time periods ───────────────────────────────────────────────────────────────

def _build_7_periods(all_df):
    post = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post); bounds = [ERUPTION_END]
    for i in range(1, 5):
        bounds.append(post['t'].iloc[min(int(round(i*n/5)), n-1)])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds = [('Pre-eruption', None, ERUPTION_START),
           ('Syn-eruption', ERUPTION_START, ERUPTION_END)]
    for i in range(5):
        pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}', bounds[i], bounds[i+1]))
    return pds

def _build_annual_periods():
    pds = [('Pre-eruption\n2015',  None,            ERUPTION_START),
           ('Syn-eruption\n2015',  ERUPTION_START,  ERUPTION_END),
           ('Post-eruption\n2015', ERUPTION_END,    pd.Timestamp('2016-01-01', tz='UTC'))]
    for yr in range(2016, 2027):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr+1}-01-01', tz='UTC') if yr < 2026 else None
        pds.append((str(yr), t0, t1))
    return pds

def _subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None: m = m & (df['t'] < t1)
    return df[m]

# ── Colors ─────────────────────────────────────────────────────────────────────

def _period_colors(n):
    cols = ['#800080', '#CC0000']
    n_post = n - 2
    start = np.array(mcolors.to_rgb('#ADD8E6'))
    end   = np.array(mcolors.to_rgb('#800080'))
    for i in range(n_post):
        t = i / max(n_post - 1, 1)
        cols.append(mcolors.to_hex((1-t)*start + t*end))
    return cols[:n]

# ── Rose drawing ───────────────────────────────────────────────────────────────

def _fill_rose(ax, phi_vals, weights, nbins, color, lw=0.5):
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_facecolor('none'); ax.set_yticklabels([]); ax.set_xticklabels([])
    ax.set_xticks([]); ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(lw*1.5)
    ax.grid(False)
    if len(phi_vals) == 0:
        ax.set_ylim(0, 1); return
    phi_arr = np.asarray(phi_vals, dtype=float) % 360.
    wts_arr = np.asarray(weights,  dtype=float)
    dbl_ang = np.deg2rad(np.concatenate([phi_arr, (phi_arr+180)%360]))
    dbl_wt  = np.concatenate([wts_arr, wts_arr])
    bins = np.linspace(0, 2*np.pi, nbins+1)
    counts, edges = np.histogram(dbl_ang, bins=bins, weights=dbl_wt)
    centers = (edges[:-1]+edges[1:])/2
    ax.bar(centers, counts, width=2*np.pi/nbins, bottom=0,
           color=color, edgecolor='black', linewidth=lw, alpha=1.0)
    ax.set_ylim(0, counts.max()*1.25 if counts.max() > 0 else 1)

# ── Layout helpers ─────────────────────────────────────────────────────────────

def _layout(n_cols, panel_w, panel_h, row_gap=0.10,
             left_margin=0.75, top_margin=0.65, section_gap=0.45):
    section_rows = {s: i for i, s in enumerate(STATION_ORDER) if s in SECTION_LABELS}
    n_rows = len(STATION_ORDER)
    fig_h = n_rows*(panel_h+row_gap) + top_margin + len(section_rows)*section_gap
    fig_w = n_cols*panel_w + left_margin
    row_tops = []
    y = fig_h - top_margin
    for sta in STATION_ORDER:
        if sta in SECTION_LABELS: y -= section_gap
        row_tops.append(y); y -= (panel_h+row_gap)
    col_lefts = [left_margin + c*panel_w for c in range(n_cols)]
    return fig_h, fig_w, row_tops, col_lefts, section_rows

# ── No-BAZ figure ──────────────────────────────────────────────────────────────

def make_rose_figure(dfs, time_periods, dt_weighted, title):
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)
    panel_size = 1.8
    fig_h, fig_w, row_tops, col_lefts, section_rows = _layout(
        n_cols, panel_size, panel_size)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    if title: fig.suptitle(title, fontsize=11, fontweight='bold', y=1.002)

    for row_idx, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        cell_top = row_tops[row_idx]
        for col_idx, (label, t0, t1) in enumerate(time_periods):
            sub = _subset(df, t0, t1)
            x0 = col_lefts[col_idx]/fig_w
            y0 = (cell_top-panel_size)/fig_h
            ax = fig.add_axes([x0, y0, panel_size/fig_w, panel_size/fig_h],
                               projection='polar')
            phi_vals = sub['phi_az'].values
            wts = np.abs(sub['dt'].values) if dt_weighted else np.ones(len(sub))
            _fill_rose(ax, phi_vals, wts, 36, colors[col_idx])
            neq = sub['event_datetime'].nunique()
            if row_idx == 0:
                ax.set_title(f'{label}\nN={neq:,}', fontsize=6.5,
                             fontweight='bold', pad=2)
            else:
                ax.set_title(f'N={neq:,}', fontsize=6.5, fontweight='bold', pad=2)

        y_center = (cell_top - panel_size/2)/fig_h
        fig.text(0., y_center, sta, fontsize=11, fontweight='bold',
                 ha='left', va='center', transform=fig.transFigure)

    for sta, row_idx in section_rows.items():
        fig.text(0.75/fig_w, (row_tops[row_idx]+0.45*0.55)/fig_h,
                 SECTION_LABELS[sta], fontsize=13, fontweight='bold',
                 ha='left', va='bottom', transform=fig.transFigure,
                 fontfamily='Arial')
    return fig

# ── BAZ figure ─────────────────────────────────────────────────────────────────

def make_baz_figure(dfs, time_periods, dt_weighted, title):
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)
    panel_w = 2.5; panel_h = 2.5
    panel_scale = min(panel_w, panel_h)
    main_r   = panel_scale * 0.18
    ring_r_h = panel_scale * 0.35
    ring_r_v = panel_scale * 0.42
    small_r  = panel_scale * 0.12

    fig_h, fig_w, row_tops, _, section_rows = _layout(
        n_cols, panel_w, panel_h, left_margin=1.0)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.995)

    subsets = [{sta: _subset(dfs[sta], t0, t1) for sta in STATION_ORDER}
               for _, t0, t1 in time_periods]

    for row_idx, sta in enumerate(STATION_ORDER):
        cell_top = row_tops[row_idx]
        cy_in    = cell_top - panel_h/2

        for col_idx, (label, t0, t1) in enumerate(time_periods):
            sub   = subsets[col_idx][sta]
            cx_in = 1.0 + (col_idx+0.5)*panel_w
            color = colors[col_idx]
            phi_vals = sub['phi_az'].values if len(sub) > 0 else np.array([])
            wts = np.abs(sub['dt'].values) if dt_weighted else np.ones(len(sub))
            neq = sub['event_datetime'].nunique() if len(sub) > 0 else 0

            ax_main = fig.add_axes(
                [(cx_in-main_r)/fig_w, (cy_in-main_r)/fig_h,
                 2*main_r/fig_w,       2*main_r/fig_h],
                projection='polar')
            _fill_rose(ax_main, phi_vals, wts, 36, color, lw=0.8)
            if row_idx == 0:
                ax_main.set_title(f'{label}\nN={neq:,}', fontsize=6.5,
                                  fontweight='bold', pad=2)
            else:
                ax_main.set_title(f'N={neq:,}', fontsize=5.5,
                                  fontweight='bold', pad=1)

            for baz_lo, baz_hi, baz_c in zip(BAZ_BINS[:-1], BAZ_BINS[1:], BAZ_CTRS):
                baz_rad = np.deg2rad(baz_c)
                sc_cx = cx_in + ring_r_h*np.sin(baz_rad)
                sc_cy = cy_in + ring_r_v*np.cos(baz_rad)
                ax_s = fig.add_axes(
                    [(sc_cx-small_r)/fig_w, (sc_cy-small_r)/fig_h,
                     2*small_r/fig_w,        2*small_r/fig_h],
                    projection='polar')
                if 'back_azimuth' in sub.columns:
                    bm = (sub['back_azimuth'] >= baz_lo) & (sub['back_azimuth'] < baz_hi)
                    sub_b = sub[bm]
                else:
                    sub_b = sub.iloc[0:0]
                phi_b = sub_b['phi_az'].values
                wts_b = np.abs(sub_b['dt'].values) if dt_weighted else np.ones(len(sub_b))
                _fill_rose(ax_s, phi_b, wts_b, 18, color, lw=0.4)
                ax_s.set_title(f'{int(baz_lo)}–{int(baz_hi)}°\nN={len(sub_b):,}',
                               fontsize=4.0, pad=1)

        y_center = (cell_top - panel_h/2)/fig_h
        fig.text(0., y_center, sta, fontsize=11, fontweight='bold',
                 ha='left', va='center', transform=fig.transFigure)
        if sta in SECTION_LABELS:
            fig.text(1.0/fig_w, (cell_top+0.45*0.55)/fig_h,
                     SECTION_LABELS[sta], fontsize=13, fontweight='bold',
                     ha='left', va='bottom', transform=fig.transFigure,
                     fontfamily='Arial')

    return fig

# ── Run ────────────────────────────────────────────────────────────────────────

print(f'Loading data (phi_error ≤ {PHI_ERR_MAX}°)...')
dfs = _load_all()
all_df = pd.concat(dfs.values(), ignore_index=True)

periods_7      = _build_7_periods(all_df)
periods_annual = _build_annual_periods()

desc = f'phi_error ≤ {PHI_ERR_MAX}°'
SFX  = '_strict_filter'

# ── 7-period weighted & unweighted ─────────────────────────────────────────────
for dt_weighted, kind in [(True, 'weighted'), (False, 'unweighted')]:
    print(f'Building {kind} 7-period...')
    fig = make_rose_figure(dfs, periods_7, dt_weighted,
                           f'Fast direction rose — {kind} ({desc})')
    out = os.path.join(OUT_DIR, f'splitting_{kind}{SFX}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {out}')

# ── Annual weighted & unweighted ──────────────────────────────────────────────
for dt_weighted, kind in [(True, 'weighted'), (False, 'unweighted')]:
    print(f'Building {kind} annual...')
    fig = make_rose_figure(dfs, periods_annual, dt_weighted,
                           f'Fast direction rose — {kind} annual ({desc})')
    out = os.path.join(OUT_DIR, f'splitting_{kind}_annual{SFX}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {out}')

# ── BAZ 7-period ──────────────────────────────────────────────────────────────
print('Building BAZ 7-period...')
with PdfPages(os.path.join(OUT_DIR, f'splitting_baz{SFX}.pdf')) as pdf:
    for dt_weighted, kind in [(True, 'weighted'), (False, 'unweighted')]:
        fig = make_baz_figure(dfs, periods_7, dt_weighted,
                              f'Fast direction BAZ roses — {kind} ({desc})')
        pdf.savefig(fig, dpi=200, bbox_inches='tight'); plt.close(fig)
        print(f'  Added: {kind}')
print(f'  Saved splitting_baz{SFX}.pdf')

# ── BAZ annual ────────────────────────────────────────────────────────────────
print('Building BAZ annual...')
with PdfPages(os.path.join(OUT_DIR, f'splitting_baz_annual{SFX}.pdf')) as pdf:
    for dt_weighted, kind in [(True, 'weighted'), (False, 'unweighted')]:
        fig = make_baz_figure(dfs, periods_annual, dt_weighted,
                              f'Fast direction BAZ roses — {kind} annual ({desc})')
        pdf.savefig(fig, dpi=200, bbox_inches='tight'); plt.close(fig)
        print(f'  Added: {kind}')
print(f'  Saved splitting_baz_annual{SFX}.pdf')

print('\nDone.')
