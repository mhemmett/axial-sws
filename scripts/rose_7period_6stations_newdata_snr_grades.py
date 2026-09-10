#!/usr/bin/env python3
"""
rose_7period_6stations_newdata_snr_grades.py

7-period fast-direction rose plots (6 rows: AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3 x 7 cols:
eruption-relative periods) built from the most recent "_newdata" mfast try_filters +
max_t_shift_s=0.2s splitting results for which ALL SIX stations have a complete dataset:

    mfast_maxdt_pipeline_transfer/splitting_results_{STA}_{2015_2021,2022_2026}_all_batches.csv

for STA in AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3. This deliberately does NOT use the more
recent in-progress AXEC2-only maxdt-wraparound-fix re-run
(production_axec2_mfast_filters_maxdt02_lqt_pykonal_results/*.csv) -- that fix only exists for
AXEC2 so far and would make AXEC2 inconsistent with the other five stations' processing; this
script sticks to the one dataset where all six stations were processed identically.

6 pages, cumulative-style SNR/quality "grades" (each page independent, not a QC ladder building
tier-on-tier like rose_7period_6stations_newdata.py) -- first 3 pages at quality >= 0.5, next 3
pages repeat the same three SNR/phi_err grades at the stricter quality >= 0.75:
    Page 1: SNR >= 2.0, quality >= 0.5,  dt_err <= 0.05 s, dt <= T_dom/2, phi_err <= 20 deg
    Page 2 ("Grade B"): same, but SNR >= 3.0
    Page 3 ("Grade A"): same, but SNR >= 4.0 and phi_err <= 10 deg
    Page 4: SNR >= 2.0, quality >= 0.75, dt_err <= 0.05 s, dt <= T_dom/2, phi_err <= 20 deg
    Page 5 ("Grade B"): same, but SNR >= 3.0
    Page 6 ("Grade A"): same, but SNR >= 4.0 and phi_err <= 10 deg

All filters use snr_horizontal, quality, dt_error, dominant_period (per-event dominant period,
i.e. the dt<=T_dom/2 cycle-skip-risk cut), and phi_error, all present directly in these CSVs.

7 eruption-relative periods (verbatim scheme from rose_plots_temporal.py /
rose_7period_6stations_newdata.py): Pre-eruption, Syn-eruption, then 5 equal-count bins spanning
post-eruption to present.

Produces: rose_7period_6stations_newdata_snr_grades.pdf (6 pages) -- a NEW pdf, does not
overwrite rose_7period_6stations_newdata.pdf or any other existing output.

Run with:
    python3 rose_7period_6stations_newdata_snr_grades.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
OUT_PDF = os.path.join(HERE, 'rose_7period_6stations_newdata_snr_grades.pdf')

DT_ERR_MAX = 0.05
PHI_ERR_MAX = 20.0
PHI_ERR_MAX_STRICT = 10.0
QW_MIN = 0.5

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

STATION_ORDER = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']

STATION_FILES = {
    sta: [f'splitting_results_{sta}_2015_2021_all_batches.csv',
          f'splitting_results_{sta}_2022_2026_all_batches.csv']
    for sta in STATION_ORDER
}

def _make_grades(qw_min):
    return [
        dict(label=f'SNR >= 2.0, quality >= {qw_min}, dt_err <= {DT_ERR_MAX}s, dt <= T_dom/2, phi_err <= {PHI_ERR_MAX:.0f} deg',
             snr_min=2.0, phi_err_max=PHI_ERR_MAX, qw_min=qw_min),
        dict(label=f'"Grade B": SNR >= 3.0, quality >= {qw_min}, dt_err <= {DT_ERR_MAX}s, dt <= T_dom/2, phi_err <= {PHI_ERR_MAX:.0f} deg',
             snr_min=3.0, phi_err_max=PHI_ERR_MAX, qw_min=qw_min),
        dict(label=f'"Grade A": SNR >= 4.0, quality >= {qw_min}, dt_err <= {DT_ERR_MAX}s, dt <= T_dom/2, phi_err <= {PHI_ERR_MAX_STRICT:.0f} deg',
             snr_min=4.0, phi_err_max=PHI_ERR_MAX_STRICT, qw_min=qw_min),
    ]


GRADES = _make_grades(QW_MIN) + _make_grades(0.75)


def load_station_raw(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in STATION_FILES[sta]]
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
    d = d[d['dt_error'] <= DT_ERR_MAX]
    d = d[d['dt'] <= d['dominant_period'] / 2.0]
    d = d[d['phi_error'] <= grade['phi_err_max']]
    return d


# ── Period definitions (verbatim from rose_7period_6stations_newdata.py) ────

def _build_time_periods(all_df):
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


def _subset(df, t_start, t_end):
    m = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
    if t_end is not None:
        m = m & (df['t'] < t_end)
    return df[m]


# ── Rose drawing (verbatim from rose_7period_6stations_newdata.py) ──────────

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


def make_rose_figure(dfs, time_periods, title):
    """6-row (station) x 7-col (time period) grid, count-weighted."""
    n_rows = len(STATION_ORDER)
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)

    panel_size = 1.8
    row_gap = 0.10
    left_margin = 0.75
    top_margin = 0.65

    fig_h = n_rows * (panel_size + row_gap) + top_margin
    fig_w = n_cols * panel_size + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold', y=1.002)

    row_tops = []
    y_cursor = fig_h - top_margin
    for row_idx in range(n_rows):
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
            wts = np.ones(len(sub))

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

    return fig


def main():
    print('Loading newdata mfast max_dt=0.2s splitting results (all 6 stations, no AXEC2 '
          'maxdt-wraparound-fix re-run)...')
    raw = {}
    for sta in STATION_ORDER:
        raw[sta] = load_station_raw(sta)
        print(f'  {sta}: {len(raw[sta]):,} baseline events (success & dt>0)')

    all_combined = pd.concat(raw.values(), ignore_index=True)
    time_periods = _build_time_periods(all_combined)

    with PdfPages(OUT_PDF) as pdf:
        for grade in GRADES:
            dfs = {sta: apply_grade(raw[sta], grade) for sta in STATION_ORDER}
            for sta in STATION_ORDER:
                print(f'  {grade["label"]} -- {sta}: {len(dfs[sta]):,}')
            fig = make_rose_figure(
                dfs, time_periods,
                title=(f'AXAS1 / AXAS2 / AXCC1 / AXEC1 / AXEC2 / AXEC3 fast direction rose '
                       f'(newdata) — {grade["label"]}'))
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
