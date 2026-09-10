#!/usr/bin/env python3
"""
rose_plots_lqt_pykonal_unweighted_baz.py

Back-azimuth ring version of rose_plots_lqt_pykonal_unweighted.py: same 6-row
(station) x N-col (time period) grid, same eight quality-filter pages, same
data/filters/QC (imports load_station_raw() and every filter threshold
directly from rose_plots_lqt_pykonal_unweighted so the two scripts can never
drift apart) -- but every cell's central rose is now surrounded by 6
back-azimuth (station -> event, deg from N) sub-roses, one per 60 deg BAZ
bin, following axec1_rose_by_depth_lqt_pykonal.py's page-2 ring layout
("6 back-azimuth sub-roses arranged in a ring around each central rose";
there: depth-bin x period grid, here: station x period grid) and
rose_plots_baz.py's original station x period ring convention.

Back-azimuth is computed once per station on the RAW (baseline-only)
dataset -- via obspy gps2dist_azimuth from each station's own coordinates to
event_lat/lon (joined from the MLdd catalogs on (station, event_datetime),
same join used throughout this session) -- then every quality-filtered
subset below is re-derived from that same augmented raw dataframe (same
boolean conditions as the imported load_station_*() functions), so every
subset inherits the back_azimuth column.

Each output PDF is nine pages (identical filters to
rose_plots_lqt_pykonal_unweighted.py, in the same order):
    page 1 — quality filter only (Q_w>QW_MIN, phi_err<PHI_ERR_MAX, dt_err<DT_ERR_MAX)
    page 2 — quality filter + dt <= 0.8 * max_dt
    page 3 — same as page 1 but Q_w >= QW_MIN_STRICT
    page 4 — "null": Q_w < QW_NULL_MAX, no phi_error/dt_error cut
    page 5 — "true splits": Q_w >= QW_TRUE_SPLIT_MIN, no phi_error/dt_error cut
    page 6 — "true nulls": Q_w <= QW_TRUE_NULL_MAX, no phi_error/dt_error cut
    page 7 — "gray zone": QW_TRUE_NULL_MAX < Q_w < QW_TRUE_SPLIT_MIN
    page 8 — "indeterminate quality": -QW_INDETERMINATE_ABS < Q_w < QW_INDETERMINATE_ABS
    page 9 — Q_w > QW_ABOVE_NEG_HALF, no phi_error/dt_error cut

Produces two PDFs in lqt_pykonal_combined_results/:
    lqt_pykonal_splitting_unweighted_baz.pdf        — eruption-relative bins
    lqt_pykonal_splitting_unweighted_annual_baz.pdf — annual bins

Run with:
    python3 rose_plots_lqt_pykonal_unweighted_baz.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from obspy.geodetics import gps2dist_azimuth

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rose_plots_lqt_pykonal_unweighted import (
    STATION_ORDER, SECTION_LABELS, OUT_DIR,
    QW_MIN, QW_MIN_STRICT, QW_NULL_MAX, QW_TRUE_SPLIT_MIN, QW_TRUE_NULL_MAX,
    QW_INDETERMINATE_ABS, QW_ABOVE_NEG_HALF, PHI_ERR_MAX, DT_ERR_MAX, MAX_DT, DT_CUTOFF,
    load_station_raw, _build_time_periods, _build_annual_periods,
    _subset, _draw_rose, _period_colors,
)

CATALOG_DIR = os.path.join(os.path.dirname(HERE), 'data')
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'

# Back-azimuth bins: 6 x 60 deg bins (same convention as rose_plots_baz.py /
# axec1_rose_by_depth_lqt_pykonal.py)
BAZ_BINS = np.arange(0, 361, 60)
BAZ_CTRS = (BAZ_BINS[:-1] + BAZ_BINS[1:]) / 2


# ── Catalog join + back-azimuth ──────────────────────────────────────────────

def _load_catalog():
    cat1 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2015_2021.csv'))
    cat2 = pd.read_csv(os.path.join(CATALOG_DIR, 'mldd_catalog_2022_2026.csv'))
    cat1['event_datetime'] = pd.to_datetime(cat1['event_datetime'], utc=True, format='mixed')
    cat2['event_datetime'] = pd.to_datetime(cat2['event_datetime'], utc=True, format='mixed')
    return pd.concat([cat1, cat2], ignore_index=True)


def load_station_raw_with_baz(sta, catalog, sta_lonlat):
    """load_station_raw(sta) + event location join (MLdd catalogs) + back
    azimuth (station -> event, deg from N, via obspy gps2dist_azimuth)."""
    df = load_station_raw(sta)
    sta_cat = catalog[catalog['station'] == sta].drop_duplicates(subset='event_datetime')
    df = df.merge(sta_cat[['event_datetime', 'event_lat', 'event_lon']],
                  left_on='t', right_on='event_datetime', how='left')
    n_unmatched = df['event_lat'].isna().sum()
    if n_unmatched:
        print(f'    {sta}: dropping {n_unmatched:,} measurements with no catalog location match')
    df = df.dropna(subset=['event_lat', 'event_lon'])

    sta_lon, sta_lat = sta_lonlat[sta]
    df = df.assign(back_azimuth=[
        gps2dist_azimuth(sta_lat, sta_lon, elat, elon)[1]
        for elat, elon in zip(df['event_lat'], df['event_lon'])
    ])
    return df


# ── The same 8 quality-filter subsets as rose_plots_lqt_pykonal_unweighted.py,
# re-derived from the back-azimuth-augmented raw dataframe (identical boolean
# conditions to load_station()/load_station_strict()/etc there). ────────────

def _filtered_variants(raw):
    return {
        'filter': raw[(raw['quality'] > QW_MIN) &
                      (raw['phi_error'] < PHI_ERR_MAX) &
                      (raw['dt_error'] < DT_ERR_MAX)],
        'strict': raw[(raw['quality'] >= QW_MIN_STRICT) &
                      (raw['phi_error'] < PHI_ERR_MAX) &
                      (raw['dt_error'] < DT_ERR_MAX)],
        'null': raw[raw['quality'] < QW_NULL_MAX],
        'true_split': raw[raw['quality'] >= QW_TRUE_SPLIT_MIN],
        'true_null': raw[raw['quality'] <= QW_TRUE_NULL_MAX],
        'gray_zone': raw[(raw['quality'] > QW_TRUE_NULL_MAX) & (raw['quality'] < QW_TRUE_SPLIT_MIN)],
        'indeterminate': raw[(raw['quality'] > -QW_INDETERMINATE_ABS) & (raw['quality'] < QW_INDETERMINATE_ABS)],
        'above_neg_half': raw[raw['quality'] > QW_ABOVE_NEG_HALF],
    }


# ── Ring figure builder: 6-row (station) x N-col (time period) grid, each
# cell a central rose + 6 back-azimuth sub-roses arranged in a ring around it
# (layout from axec1_rose_by_depth_lqt_pykonal.py's page-2 ring, generalized
# here to station rows with SECTION_LABELS, as in rose_plots_lqt_pykonal_
# unweighted.py's make_rose_figure). ─────────────────────────────────────────

def make_ring_figure(dfs, time_periods, title):
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_w = 2.6
    panel_h = 2.6
    row_gap = 0.15
    left_margin = 1.0
    top_margin = 0.65
    section_gap = 0.55

    panel_scale = min(panel_w, panel_h)
    main_r = panel_scale * 0.18
    ring_r_h = panel_scale * 0.35
    ring_r_v = panel_scale * 0.42
    small_r = panel_scale * 0.12

    section_rows = {sta: i for i, sta in enumerate(STATION_ORDER) if sta in SECTION_LABELS}

    fig_h = (n_rows * (panel_h + row_gap) + top_margin
             + len(section_rows) * section_gap)
    fig_w = n_cols * panel_w + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold', y=1.003)

    row_tops = []
    y_cursor = fig_h - top_margin
    for row_idx, sta in enumerate(STATION_ORDER):
        if sta in SECTION_LABELS:
            y_cursor -= section_gap
        row_tops.append(y_cursor)
        y_cursor -= (panel_h + row_gap)

    for row_idx, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        cell_top = row_tops[row_idx]
        cy_in = cell_top - panel_h / 2

        for col_idx, (label, t0, t1) in enumerate(time_periods):
            sub = _subset(df, t0, t1)
            color = colors[col_idx]
            cx_in = left_margin + (col_idx + 0.5) * panel_w

            ax_main = fig.add_axes(
                [(cx_in - main_r) / fig_w, (cy_in - main_r) / fig_h,
                 2 * main_r / fig_w, 2 * main_r / fig_h],
                projection='polar')
            _draw_rose(ax_main, sub['phi_az'].values, np.ones(len(sub)), color)
            if row_idx == 0:
                ax_main.set_title(f'{label}\nN={len(sub):,}', fontsize=6.5, fontweight='bold', pad=2)
            else:
                ax_main.set_title(f'N={len(sub):,}', fontsize=5.5, fontweight='bold', pad=1)

            for baz_lo, baz_hi, baz_c in zip(BAZ_BINS[:-1], BAZ_BINS[1:], BAZ_CTRS):
                baz_rad = np.deg2rad(baz_c)
                sc_cx = cx_in + ring_r_h * np.sin(baz_rad)
                sc_cy = cy_in + ring_r_v * np.cos(baz_rad)

                ax_s = fig.add_axes(
                    [(sc_cx - small_r) / fig_w, (sc_cy - small_r) / fig_h,
                     2 * small_r / fig_w, 2 * small_r / fig_h],
                    projection='polar')

                baz_mask = (sub['back_azimuth'] >= baz_lo) & (sub['back_azimuth'] < baz_hi)
                sub_baz = sub[baz_mask]
                _draw_rose(ax_s, sub_baz['phi_az'].values, np.ones(len(sub_baz)), color)
                ax_s.set_title(f'{int(baz_lo)}–{int(baz_hi)}°\nN={len(sub_baz):,}',
                               fontsize=3.8, pad=1)

        y_center = (cell_top - panel_h / 2) / fig_h
        fig.text(0.0, y_center, sta, fontsize=11, fontweight='bold',
                  ha='left', va='center', transform=fig.transFigure)

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
    inc_note = 'incidence<35$\\degree$ (applied upstream, true PyKonal-FMM)'
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

    print('Loading MLdd catalogs for event locations...')
    catalog = _load_catalog()

    _sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                           engine='python').set_index('s')
    sta_lonlat = {sta: (float(_sta_df.loc[sta, 'lon']), float(_sta_df.loc[sta, 'lat']))
                  for sta in STATION_ORDER if sta in _sta_df.index}

    variants = {key: {} for key in ['filter', 'dtcut', 'strict', 'null', 'true_split',
                                     'true_null', 'gray_zone', 'indeterminate', 'above_neg_half']}
    for sta in STATION_ORDER:
        print(f'  Loading {sta} (+ back-azimuth)...')
        raw = load_station_raw_with_baz(sta, catalog, sta_lonlat)
        v = _filtered_variants(raw)
        for key in variants:
            if key == 'dtcut':
                continue
            variants[key][sta] = v[key]
        variants['dtcut'][sta] = v['filter'][v['filter']['dt'] <= DT_CUTOFF]
        print(f'    {sta}: {len(raw):,} baseline -> '
              f'{len(v["filter"]):,} filter, {len(variants["dtcut"][sta]):,} dtcut, '
              f'{len(v["strict"]):,} strict, {len(v["null"]):,} null, '
              f'{len(v["true_split"]):,} true_split, {len(v["true_null"]):,} true_null, '
              f'{len(v["gray_zone"]):,} gray_zone, {len(v["indeterminate"]):,} indeterminate, '
              f'{len(v["above_neg_half"]):,} above_neg_half')

    all_combined = pd.concat(variants['filter'].values(), ignore_index=True)

    pages = [
        ('filter', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — {filter_note}'),
        ('dtcut', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — {dtcut_note}'),
        ('strict', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — {strict_note}'),
        ('null', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — {null_note}'),
        ('true_split', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — true splits — {true_split_note}'),
        ('true_null', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — true nulls — {true_null_note}'),
        ('gray_zone', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — gray zone — {gray_zone_note}'),
        ('indeterminate', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — indeterminate quality — {indeterminate_note}'),
        ('above_neg_half', f'LQT + PyKonal-FMM fast direction rose w/ back-azimuth — {above_neg_half_note}'),
    ]

    # ── Eruption-relative bins ────────────────────────────────────────────────
    time_periods = _build_time_periods(all_combined)
    out_path = os.path.join(OUT_DIR, 'lqt_pykonal_splitting_unweighted_baz.pdf')
    with PdfPages(out_path) as pdf:
        for key, title in pages:
            print(f'  Page: {key} (eruption-relative)...')
            fig = make_ring_figure(variants[key], time_periods, title)
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {out_path}')

    # ── Annual bins ───────────────────────────────────────────────────────────
    annual_periods = _build_annual_periods()
    out_annual = os.path.join(OUT_DIR, 'lqt_pykonal_splitting_unweighted_annual_baz.pdf')
    with PdfPages(out_annual) as pdf:
        for key, title in pages:
            print(f'  Page: {key} (annual)...')
            fig = make_ring_figure(variants[key], annual_periods, title.replace('rose w/', 'rose annual w/'))
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved {out_annual}')

    print('\nDone.')


if __name__ == '__main__':
    main()
