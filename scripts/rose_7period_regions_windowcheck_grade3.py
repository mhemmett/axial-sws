#!/usr/bin/env python3
"""
rose_7period_regions_windowcheck_grade3.py

Same large-font/region-labeled figure layout as rose_tier5_7period_regions_newdata_snr2_q75.py
(font-size hierarchy, region titles, suptitle, single-line period labels -- see that script's
docstring for the full typography rationale, not repeated here), but built from the most recent
WINDOW-CHECK SWSPy run (mfast try_filters + max_t_shift_s=0.2s, through the
`split_windowcheck.py` lag-shift wraparound fix -- see
mfast_maxdt_windowcheck_pipeline_transfer/README.md) instead of the older "newdata" (unfixed)
run:

    mfast_maxdt_windowcheck_pipeline_transfer/splitting_results_{STA}_{2015_2021,2022_2026}_all_batches.csv

for the 11 station/period combos present there (AXAS1, AXAS2, AXCC1, AXEC1, AXEC3 x both
periods, plus AXEC2 2022-2026 only). AXEC2's 2015-2021 half is NOT in that folder -- it's the
full 497-batch windowcheck rerun combined separately at:

    scripts/production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results/
        splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_combined.csv

which lacks a `dominant_period` column directly (unlike the transfer-folder CSVs) -- it's
derived here the same way rose_axec2_windowcheck_vs_newdata_grade3_matched_batches.py does:
`chosen_filter_dom_period_samples / 200.0` (200 Hz sample rate).

Filter is this session's "grade 3" (per rose_axec2_windowcheck_vs_newdata_grade3_matched_batches.py's
GRADE / rose_7period_6stations_newdata_snr_grades.py's GRADES[3]):

    SNR >= 2.0, quality (Q_w) >= 0.75, dt_err <= 0.05 s, dt <= T_dom/2, phi_err <= 20 deg

Title, fonts, region groupings, and period-label formatting are all UNCHANGED from
rose_tier5_7period_regions_newdata_snr2_q75.py; only the underlying data (windowcheck fix,
including the full AXEC2 2015-2021 rerun) changes. This is a NEW script producing a NEW output
file -- it does not touch rose_tier5_7period_regions_newdata_snr2_q75.pdf or any other existing
figure.

Produces: rose_7period_regions_windowcheck_grade3.pdf (1 page)

Run with:
    python3 rose_7period_regions_windowcheck_grade3.py
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

from rose_7period_6stations_newdata_snr_grades import (
    STATION_ORDER, _draw_rose, _period_colors, ERUPTION_START, ERUPTION_END,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WC_TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_windowcheck_pipeline_transfer')
AXEC2_2015_2021_WC_CSV = os.path.join(
    HERE, 'production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results',
    'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_combined.csv')

OUT_PDF = os.path.join(HERE, 'rose_7period_regions_windowcheck_grade3.pdf')

GRADE = dict(label='Grade 3: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg',
             snr_min=2.0, qw_min=0.75, phi_err_max=20.0, dt_err_max=0.05)

STATION_FILES = {
    sta: [f'splitting_results_{sta}_2015_2021_all_batches.csv',
          f'splitting_results_{sta}_2022_2026_all_batches.csv']
    for sta in STATION_ORDER if sta != 'AXEC2'
}
STATION_FILES['AXEC2'] = ['splitting_results_AXEC2_2022_2026_all_batches.csv']

# ── Font-size hierarchy (verbatim from rose_tier5_7period_regions_newdata_snr2_q75.py) ──────
FS_N = 12.1875  # 19.5 halved (overlap fix), then +25% per user follow-up
FS_PERIOD = 19.5
FS_STATION = 22
FS_REGION = 26
FS_SUPTITLE = 32

REGION_TITLES = {
    'AXAS1': 'Western Caldera',
    'AXCC1': 'Central Caldera',
    'AXEC1': 'Eastern Caldera',
}


def load_station_raw(sta):
    dfs = [pd.read_csv(os.path.join(WC_TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
    if sta == 'AXEC2':
        wc = pd.read_csv(AXEC2_2015_2021_WC_CSV)
        wc['dominant_period'] = wc['chosen_filter_dom_period_samples'] / 200.0
        dfs.append(wc)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    df['phi_az'] = df['phi'] % 180.0
    return df


def apply_grade(df, grade):
    d = df.dropna(subset=['snr_horizontal', 'dt_error', 'dominant_period', 'phi_error', 'quality'])
    d = d[d['snr_horizontal'] >= grade['snr_min']]
    d = d[d['quality'] >= grade['qw_min']]
    d = d[d['dt_error'] <= grade['dt_err_max']]
    d = d[d['dt'] <= d['dominant_period'] / 2.0]
    d = d[d['phi_error'] <= grade['phi_err_max']]
    return d


def _build_time_periods_single_line(all_df):
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
        ('Before Eruption', None, ERUPTION_START),
        ('During Eruption', ERUPTION_START, ERUPTION_END),
    ]
    for i in range(5):
        t0, t1 = boundaries[i], boundaries[i + 1]
        periods.append((f'{_fmt(t0)} - {_fmt(t1)}', t0, t1))

    return periods


def _circular_mean_and_se_deg(phi_az_vals):
    """Circular (vector) mean of axially-symmetric (mod 180) fast directions, via the
    doubled-angle trick, plus the standard error of that mean (circular std / sqrt(N)).
    Returns (mean_phi_deg, se_deg), both NaN if N==0 or the resultant length is ~0
    (directions too dispersed for a meaningful mean)."""
    n = len(phi_az_vals)
    if n == 0:
        return np.nan, np.nan
    angles = 2.0 * np.radians(np.asarray(phi_az_vals) % 180.0)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    r = np.hypot(s, c)
    mean_phi = (np.degrees(np.arctan2(s, c)) / 2.0) % 180.0
    if r <= 1e-12:
        return mean_phi, np.nan
    circ_std_deg = np.degrees(np.sqrt(-2.0 * np.log(r))) / 2.0
    se_deg = circ_std_deg / np.sqrt(n)
    return mean_phi, se_deg


def _axial_cosine_similarity_deg(phi_a_deg, phi_b_deg):
    """Cosine similarity between two axially-symmetric (mod 180) mean directions, via the
    doubled-angle trick: cos(2*(phi_a - phi_b)) in degrees. +1 = identical orientation,
    -1 = perpendicular (90 deg apart, the maximally dissimilar case for axial data). NaN if
    either input is NaN."""
    if np.isnan(phi_a_deg) or np.isnan(phi_b_deg):
        return np.nan
    return float(np.cos(2.0 * np.radians(phi_a_deg - phi_b_deg)))


def _subset(df, t_start, t_end):
    m = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
    if t_end is not None:
        m = m & (df['t'] < t_end)
    return df[m]


def make_rose_figure(dfs, time_periods, title):
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_size = 3.5
    row_gap = 0.55
    region_gap_extra = 0.9
    left_margin = 1.7
    top_margin = 2.1

    n_region_starts = sum(1 for s in STATION_ORDER if s in REGION_TITLES)
    fig_h = (n_rows * (panel_size + row_gap) + top_margin
             + n_region_starts * region_gap_extra)
    fig_w = n_cols * panel_size + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(title, fontsize=FS_SUPTITLE, fontweight='bold', y=0.997)

    row_tops = []
    y_cursor = fig_h - top_margin
    for row_idx, sta in enumerate(STATION_ORDER):
        if sta in REGION_TITLES:
            y_cursor -= region_gap_extra
        row_tops.append(y_cursor)
        y_cursor -= (panel_size + row_gap)

    col_lefts = [left_margin + col_idx * panel_size for col_idx in range(n_cols)]

    for row_idx, sta in enumerate(STATION_ORDER):
        df = dfs[sta]
        cell_top = row_tops[row_idx]

        if sta in REGION_TITLES:
            region_text_height_frac = (FS_REGION / 72.0) / fig_h
            if sta == 'AXAS1':
                region_y = cell_top / fig_h + 0.075 - region_text_height_frac
            else:
                region_y = (cell_top + region_gap_extra * 0.3) / fig_h + region_text_height_frac / 2.0
            fig.text(0.01, region_y, REGION_TITLES[sta], fontsize=FS_REGION, fontweight='bold',
                     ha='left', va='bottom', transform=fig.transFigure)

        ref_phi = np.nan
        for col_idx, (label, t_start, t_end) in enumerate(time_periods):
            sub = _subset(df, t_start, t_end)

            x0 = col_lefts[col_idx] / fig_w
            y0 = (cell_top - panel_size) / fig_h
            w = panel_size / fig_w
            h = panel_size / fig_h

            ax = fig.add_axes([x0, y0, w, h], projection='polar')

            phi_vals = sub['phi_az'].values
            wts = np.ones(len(sub))
            _draw_rose(ax, phi_vals, wts, colors[col_idx])

            n_events = len(sub)
            mean_phi, se_phi = _circular_mean_and_se_deg(phi_vals)
            if col_idx == 0:
                ref_phi = mean_phi
            cos_sim = _axial_cosine_similarity_deg(mean_phi, ref_phi)

            if n_events == 0 or np.isnan(mean_phi):
                title_txt = f'N={n_events:,}'
            else:
                avg_txt = f'Avg={mean_phi:.0f}°' if np.isnan(se_phi) else f'Avg={mean_phi:.0f}±{se_phi:.0f}°'
                cos_txt = f', cos={cos_sim:.4f}' if not np.isnan(cos_sim) else ''
                title_txt = f'N={n_events:,}, {avg_txt}{cos_txt}'

            if row_idx == 0:
                ax.set_title(title_txt, fontsize=FS_N, fontweight='bold', pad=34)
                ax_pos = ax.get_position()
                fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + 0.026, label,
                         fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
            else:
                ax.set_title(title_txt, fontsize=FS_N, fontweight='bold', pad=8)

        y_center = (cell_top - panel_size / 2) / fig_h
        fig.text(0.01, y_center, sta, fontsize=FS_STATION, fontweight='bold',
                 ha='left', va='center', transform=fig.transFigure)

    return fig


def main():
    print(f'Loading windowcheck splitting results, filter: {GRADE["label"]} ...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
    dfs = {sta: apply_grade(raw[sta], GRADE) for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        print(f'  {sta}: {len(dfs[sta]):,}')

    all_combined = pd.concat(raw.values(), ignore_index=True)
    time_periods = _build_time_periods_single_line(all_combined)

    fig = make_rose_figure(
        dfs, time_periods,
        title='Shear-wave Splitting Fast Direction Across an Eruption Cycle',
    )
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
