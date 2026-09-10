#!/usr/bin/env python3
"""
rose_7period_windowcheck_grade3_baz_ring.py

Back-azimuth-ring variant of rose_7period_regions_windowcheck_grade3.py: same windowcheck
data source (mfast_maxdt_windowcheck_pipeline_transfer/ for 11 station/period combos, plus the
separate full AXEC2 2015-2021 windowcheck rerun -- see that script's docstring for the full
data-provenance rationale, not repeated here) and same grade-3 filter (SNR>=2.0, quality>=0.75,
dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg), 7 eruption-relative time periods, but each cell's
central rose is now surrounded by 6 back-azimuth (60 deg bin) sub-roses arranged in a ring,
following rose_7period_6stations_newdata_snr2_q75_baz_ring.py's make_baz_ring_page() layout
(itself following axec1_rose_by_depth_lqt_pykonal.py's ring layout) -- station x period grid
instead of that script's newdata (unfixed) source.

Back-azimuth (station -> event, deg from N) is taken directly from the catalog's own
'back_azimuth' column, not recomputed from station/event geometry.

Each cell's MAIN rose title also carries the circular mean fast direction and its standard
error (Avg=X±Y deg) and the axial cosine similarity to that row's "Before Eruption" panel
(cos=, via the same doubled-angle trick as rose_7period_regions_windowcheck_grade3.py's
_circular_mean_and_se_deg / _axial_cosine_similarity_deg, reused here directly) -- the small
baz sub-roses keep just N= to avoid clutter at that scale.

Produces: rose_7period_windowcheck_grade3_baz_ring.pdf (1 page: 6 station rows x 7 time-period
columns, central rose + 6 baz sub-roses per cell) -- a NEW file, does not touch any existing
output.

Run with:
    python3 rose_7period_windowcheck_grade3_baz_ring.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rose_7period_6stations_newdata_snr_grades import _draw_rose, _period_colors
from rose_7period_regions_windowcheck_grade3 import (
    STATION_ORDER, GRADE, load_station_raw, apply_grade, _build_time_periods_single_line,
    _subset, _circular_mean_and_se_deg, _axial_cosine_similarity_deg,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'rose_7period_windowcheck_grade3_baz_ring.pdf')

# Back-azimuth ring bins: 6 x 60 deg bins (same convention as the newdata baz_ring script)
BAZ_BINS = np.arange(0, 361, 60)
BAZ_CTRS = (BAZ_BINS[:-1] + BAZ_BINS[1:]) / 2


def load_station_events(sta):
    raw = load_station_raw(sta)
    df = apply_grade(raw, GRADE)
    df = df.dropna(subset=['back_azimuth']).copy()
    df['back_azimuth'] = df['back_azimuth'] % 360.0
    return df[['t', 'phi_az', 'back_azimuth']].reset_index(drop=True)


def make_baz_ring_page(dfs, periods, title):
    n_rows = len(STATION_ORDER)
    n_cols = len(periods)
    colors = _period_colors(n_cols)

    panel_w = 2.6
    panel_h = 2.6
    row_gap = 0.15
    left_margin = 1.0
    top_margin = 0.75

    panel_scale = min(panel_w, panel_h)
    main_r = panel_scale * 0.18
    ring_r_h = panel_scale * 0.35
    ring_r_v = panel_scale * 0.42
    small_r = panel_scale * 0.12

    fig_h = n_rows * (panel_h + row_gap) + top_margin
    fig_w = n_cols * panel_w + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.005)

    for row_idx, sta in enumerate(STATION_ORDER):
        df_sta = dfs[sta]
        cell_top = fig_h - top_margin - row_idx * (panel_h + row_gap)
        cy_in = cell_top - panel_h / 2

        ref_phi = np.nan
        for col_idx, (label, t0, t1) in enumerate(periods):
            sub = _subset(df_sta, t0, t1)
            color = colors[col_idx]
            cx_in = left_margin + (col_idx + 0.5) * panel_w

            ax_main = fig.add_axes(
                [(cx_in - main_r) / fig_w, (cy_in - main_r) / fig_h,
                 2 * main_r / fig_w, 2 * main_r / fig_h],
                projection='polar')
            _draw_rose(ax_main, sub['phi_az'].values, np.ones(len(sub)), color)

            n_events = len(sub)
            mean_phi, se_phi = _circular_mean_and_se_deg(sub['phi_az'].values)
            if col_idx == 0:
                ref_phi = mean_phi
            cos_sim = _axial_cosine_similarity_deg(mean_phi, ref_phi)

            if n_events == 0 or np.isnan(mean_phi):
                stats_txt = f'N={n_events:,}'
            else:
                avg_txt = f'Avg={mean_phi:.0f}°' if np.isnan(se_phi) else f'Avg={mean_phi:.0f}±{se_phi:.0f}°'
                cos_txt = f', cos={cos_sim:.4f}' if not np.isnan(cos_sim) else ''
                stats_txt = f'N={n_events:,}, {avg_txt}{cos_txt}'

            if row_idx == 0:
                ax_main.set_title(f'{label}\n{stats_txt}', fontsize=6.5, fontweight='bold', pad=2)
            else:
                ax_main.set_title(stats_txt, fontsize=5.5, fontweight='bold', pad=1)

            for baz_lo, baz_hi, baz_c in zip(BAZ_BINS[:-1], BAZ_BINS[1:], BAZ_CTRS):
                baz_rad = np.deg2rad(baz_c)
                sc_cx = cx_in + ring_r_h * np.sin(baz_rad)
                sc_cy = cy_in + ring_r_v * np.cos(baz_rad)

                ax_s = fig.add_axes(
                    [(sc_cx - small_r) / fig_w, (sc_cy - small_r) / fig_h,
                     2 * small_r / fig_w, 2 * small_r / fig_h],
                    projection='polar')

                baz_mask = (sub['back_azimuth'].values >= baz_lo) & (sub['back_azimuth'].values < baz_hi)
                sub_baz = sub[baz_mask]
                _draw_rose(ax_s, sub_baz['phi_az'].values, np.ones(len(sub_baz)), color)
                ax_s.set_title(f'{int(baz_lo)}–{int(baz_hi)}°\nN={len(sub_baz):,}',
                               fontsize=3.8, pad=1)

        y_center = (cell_top - panel_h / 2) / fig_h
        fig.text(0.0, y_center, sta, fontsize=11, fontweight='bold',
                  ha='left', va='center', transform=fig.transFigure)

    return fig


def main():
    print(f'Loading windowcheck per-station events ({GRADE["label"]})...')
    station_dfs = {}
    for sta in STATION_ORDER:
        df = load_station_events(sta)
        station_dfs[sta] = df
        print(f'  {sta}: {len(df):,} events pass the filter')

    all_t = pd.concat([station_dfs[sta]['t'] for sta in STATION_ORDER], ignore_index=True)
    periods = _build_time_periods_single_line(pd.DataFrame({'t': all_t}))

    n_total = sum(len(station_dfs[sta]) for sta in STATION_ORDER)
    title = (f'AXAS1/AXAS2/AXCC1/AXEC1/AXEC2/AXEC3 (windowcheck) — fast direction rose by '
             f'station x time period, with 6 back-azimuth sub-roses (N={n_total:,})\n'
             f'{GRADE["label"]}')
    fig = make_baz_ring_page(station_dfs, periods, title)

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
