"""
Multipanel summary figure: AXAS1 shear-wave splitting measurements over time
(2015-2026), for building a manuscript figure. One-off script, not part of the
pipeline. Exact replica of figure_ec2_multipanel_temporal.py's layout/formatting
(panel sizing, fonts, labels, legends), with AXAS1 in place of AXEC2. Reuses the
exact filtering/colorscale/period conventions already established for the
production AXEC rose and temporal-histogram plots:

  - Data: lqt_pykonal_combined_results/splitting_results_AXAS1_{2015_2021,
    2022_2026}_all_batches_enriched.csv (LQT + PyKonal-FMM incidence<35 deg
    already applied upstream; "enriched" versions carry back_azimuth,
    event_lat/lon, etc.)
  - Filter (same for every panel): success==True, dt>0, quality>0.5,
    phi_error<20 deg, dt_error<0.04 s (rose_plots_lqt_pykonal_unweighted.py's
    QW_MIN/PHI_ERR_MAX/DT_ERR_MAX).
  - Period colorscale: rose_plots_lqt_pykonal_unweighted._period_colors
    (purple/red/blue-to-purple gradient for pre/syn/5 post-eruption bins).

Panels:
  A. Single-row, 7-period (pre/syn/5 post-eruption bins) fast-direction rose
     plot with a ring of 6 back-azimuth (60 deg bin) sub-roses around each
     main rose - same convention as rose_plots_lqt_pykonal_unweighted_baz.py,
     restricted to AXAS1 only (one row instead of 6).
  B. 2015-2026 moving-window density histogram, fast direction (phi) vs time.
  C. Same as B, zoomed to +/- 48 hours around the 2015 eruption onset.
  D. 2015-2026 moving-window density histogram, delay time (dt) vs time.
  E. Same as D, zoomed to +/- 48 hours around the 2015 eruption onset.

Run with: python3 figure_as1_multipanel_temporal.py
"""

import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy import UTCDateTime

FONT_SCALE = 1.3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sws_methods as swm  # noqa: E402
from rose_plots_lqt_pykonal_unweighted import (  # noqa: E402
    QW_MIN, PHI_ERR_MAX, DT_ERR_MAX, ERUPTION_START, ERUPTION_END,
    _build_time_periods, _draw_rose, _period_colors, _subset,
)
from rose_plots_lqt_pykonal_unweighted_baz import BAZ_BINS, BAZ_CTRS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
OUT_PDF = os.path.join(HERE, 'figure_as1_multipanel_temporal.pdf')

STATION = 'AXAS1'
DT_Y_END = 0.30

ZOOM_HALF_WIDTH = pd.Timedelta(hours=48)


def load_axas1_enriched():
    d1 = pd.read_csv(os.path.join(DATA_DIR, 'splitting_results_AXAS1_2015_2021_all_batches_enriched.csv'))
    d2 = pd.read_csv(os.path.join(DATA_DIR, 'splitting_results_AXAS1_2022_2026_all_batches_enriched.csv'))
    df = pd.concat([d1, d2], ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality', 'phi_error', 'dt_error'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True)
    df['phi_az'] = df['phi'] % 180.0
    return df


def apply_quality_filter(df):
    return df[(df['quality'] > QW_MIN) &
              (df['phi_error'] < PHI_ERR_MAX) &
              (df['dt_error'] < DT_ERR_MAX)].copy()


# ── Panel A: single-row BAZ-ring rose figure, rendered standalone then embedded
#    as an image (avoids re-deriving figure-fraction math for ~50 nested polar
#    axes inside the larger composite gridspec). ─────────────────────────────

def build_panel_a_image(df):
    time_periods = _build_time_periods(df)
    colors = _period_colors(len(time_periods))
    n_cols = len(time_periods)

    panel_w = 2.6
    panel_h = 2.6
    left_margin = 0.6
    top_margin = 1.1

    panel_scale = min(panel_w, panel_h)
    main_r = panel_scale * 0.18
    ring_r_h = panel_scale * 0.35
    ring_r_v = panel_scale * 0.42
    small_r = panel_scale * 0.12

    label_gap = 0.15
    fig_h = panel_h + top_margin
    fig_w = n_cols * panel_w + left_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    cy_in = fig_h - top_margin - panel_h / 2

    for col_idx, (label, t0, t1) in enumerate(time_periods):
        sub = _subset(df, t0, t1)
        color = colors[col_idx]
        cx_in = left_margin + (col_idx + 0.5) * panel_w

        ax_main = fig.add_axes(
            [(cx_in - main_r) / fig_w, (cy_in - main_r) / fig_h,
             2 * main_r / fig_w, 2 * main_r / fig_h], projection='polar')
        _draw_rose(ax_main, sub['phi_az'].values, np.ones(len(sub)), color)

        # Period label sits above the WHOLE ring (not squeezed between the main
        # rose and the top back-azimuth sub-roses, where it was overlapping).
        label_y = cy_in + ring_r_v + small_r + label_gap
        fig.text(cx_in / fig_w, label_y / fig_h, f'{label}\nN={len(sub):,}',
                 ha='center', va='bottom', fontsize=8 * FONT_SCALE, fontweight='bold')

        for baz_lo, baz_hi, baz_c in zip(BAZ_BINS[:-1], BAZ_BINS[1:], BAZ_CTRS):
            baz_rad = np.deg2rad(baz_c)
            sc_cx = cx_in + ring_r_h * np.sin(baz_rad)
            sc_cy = cy_in + ring_r_v * np.cos(baz_rad)

            ax_s = fig.add_axes(
                [(sc_cx - small_r) / fig_w, (sc_cy - small_r) / fig_h,
                 2 * small_r / fig_w, 2 * small_r / fig_h], projection='polar')
            baz_mask = (sub['back_azimuth'] >= baz_lo) & (sub['back_azimuth'] < baz_hi)
            sub_baz = sub[baz_mask]
            _draw_rose(ax_s, sub_baz['phi_az'].values, np.ones(len(sub_baz)), color)
            ax_s.set_title(f'{int(baz_lo)}–{int(baz_hi)}°\nN={len(sub_baz):,}',
                            fontsize=5 * FONT_SCALE, pad=1)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    img = mpimg.imread(buf)
    aspect = fig_w / fig_h
    return img, aspect


# ── Panels B-E: moving-window density histograms (phi, dt) vs time ─────────

def plot_movehisto2d(x, y, ax, cmap, **kwargs):
    x_start = kwargs.get('x_start', None)
    x_end = kwargs.get('x_end', None)
    x = np.array([v.timestamp for v in x])
    x_start = x_start.timestamp if x_start is not None else None
    x_end = x_end.timestamp if x_end is not None else None
    kwargs['x_start'] = x_start
    kwargs['x_end'] = x_end

    X, Y, Z, x_bins, x_diffs = swm.movehisto2d_bin(x, y, **kwargs)

    X = np.array(swm.timestamp2matplotlib(X.ravel())).reshape(X.shape)
    ax.set_xlim(swm.timestamp2matplotlib([x_start, x_end]))

    Xm, Ym = swm.XY2XY_pcolormesh(X, Y)
    im = ax.pcolormesh(Xm, Ym, Z, cmap=cmap, rasterized=True)
    ax.set_ylim([kwargs.get('y_start', Y.min()), kwargs.get('y_end', Y.max())])
    ax.set_aspect('auto')
    ax.xaxis_date()
    return im


def _mark_eruption(ax, x_start, x_end, y_mid, show_legend=False, legend_label='Eruption onset or end'):
    eruption_marks = [ERUPTION_START, ERUPTION_END] if (x_start < ERUPTION_END and x_end > ERUPTION_START) else [ERUPTION_START]
    for ex in eruption_marks:
        ex_mpl = mdates.date2num(ex.to_pydatetime())
        if ax.get_xlim()[0] <= ex_mpl <= ax.get_xlim()[1]:
            ax.axvline(ex_mpl, color='black', linestyle='--', linewidth=1.5, alpha=0.9,
                       label=legend_label)
            ax.plot(ex_mpl, y_mid, marker='*', color='black', markersize=10, zorder=5)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right',
                      fontsize=6 * FONT_SCALE, framealpha=0.8)


def _add_matched_colorbar(fig, ax, im, label_txt):
    """Colorbar sized to exactly match ax's own height (not fig.colorbar's
    fraction/shrink approximation)."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label_txt, fontsize=7 * FONT_SCALE)
    cb.ax.tick_params(labelsize=6 * FONT_SCALE)


def draw_phi_panel(ax, df, x_start, x_end, x_width, x_over, title, show_legend=False,
                   legend_label='Eruption onset or end'):
    x = np.array([UTCDateTime(t.isoformat()) for t in pd.to_datetime(df['t'])])
    y = df['phi_az'].values
    im = plot_movehisto2d(
        x, y, ax=ax, cmap='Reds',
        x_width=x_width, x_start=UTCDateTime(x_start.isoformat()),
        x_end=UTCDateTime(x_end.isoformat()), x_over=x_over,
        y_start=0, y_end=180, y_width=180 / 40, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=0.5, y_filter_per=3, filter_mode=['wrap', 'nearest'],
    )
    _mark_eruption(ax, x_start, x_end, y_mid=90, show_legend=show_legend, legend_label=legend_label)
    ax.set_ylabel(r'$\phi$ [$^o$ from N]', fontsize=9 * FONT_SCALE, fontweight='bold')
    ax.tick_params(labelsize=7 * FONT_SCALE)
    ax.set_title(title, fontsize=10 * FONT_SCALE)
    return im


def draw_dt_panel(ax, df, x_start, x_end, x_width, x_over, title, show_legend=False):
    x = np.array([UTCDateTime(t.isoformat()) for t in pd.to_datetime(df['t'])])
    y = df['dt'].values
    im = plot_movehisto2d(
        x, y, ax=ax, cmap='Blues',
        x_width=x_width, x_start=UTCDateTime(x_start.isoformat()),
        x_end=UTCDateTime(x_end.isoformat()), x_over=x_over,
        y_start=0, y_end=DT_Y_END, y_width=DT_Y_END / 30, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=0.5, y_filter_per=3, filter_mode='nearest',
    )
    _mark_eruption(ax, x_start, x_end, y_mid=DT_Y_END * 0.5, show_legend=show_legend)
    ax.set_ylabel(r'$\delta t$ [s]', fontsize=9 * FONT_SCALE, fontweight='bold')
    ax.tick_params(labelsize=7 * FONT_SCALE)
    ax.set_title(title, fontsize=10 * FONT_SCALE)
    return im


def main():
    print('Loading AXAS1 enriched LQT + PyKonal-FMM results (2015-2026)...')
    raw = load_axas1_enriched()
    df = apply_quality_filter(raw)
    print(f'  {len(raw):,} baseline (success, dt>0) -> {len(df):,} after '
          f'quality>{QW_MIN}, phi_error<{PHI_ERR_MAX}, dt_error<{DT_ERR_MAX} filter')

    print('Building panel A (BAZ-ring rose, single row)...')
    img_a, aspect_a = build_panel_a_image(df)

    t_min, t_max = df['t'].min(), df['t'].max()
    zoom_start = ERUPTION_START - ZOOM_HALF_WIDTH
    zoom_end = ERUPTION_START + ZOOM_HALF_WIDTH

    fig = plt.figure(figsize=(20, 15))
    fig_w_in, fig_h_in = fig.get_size_inches()

    # ALL rows (A and the bcde grid) share the exact same left/right span, so
    # every row is the same total width - this was the actual bug: bcde_gs had
    # no explicit left/right, so it fell back to matplotlib's default margins
    # (0.125-0.9) while A used a different, independently-computed span.
    LEFT, RIGHT = 0.05, 0.98
    width_frac = RIGHT - LEFT

    # Panel A's height is DERIVED from that fixed, shared width plus the image's
    # own aspect ratio - this is what keeps the rose circles round. (Forcing A's
    # height to equal B's row height, with width also fixed to the full page
    # width, over-determines the system given aspect_a and is what produced the
    # squished ellipses before.)
    a_top = 0.94
    h_a = (width_frac * fig_w_in / aspect_a) / fig_h_in
    a_bottom = a_top - h_a

    gap_a_to_bcde = 0.035
    bcde_top = a_bottom - gap_a_to_bcde
    bcde_bottom = 0.06
    hspace_bcde = 0.30

    ax_a = fig.add_axes([LEFT, a_bottom, width_frac, h_a])
    ax_a.imshow(img_a, aspect='auto')
    ax_a.axis('off')
    ax_a.set_title('Fast Direction by Back Azimuth', fontsize=10 * FONT_SCALE)

    bcde_gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.7, 0.375],
                               hspace=hspace_bcde, wspace=0.08,
                               left=LEFT, right=RIGHT, top=bcde_top, bottom=bcde_bottom)

    ax_b = fig.add_subplot(bcde_gs[0, 0])
    ax_c = fig.add_subplot(bcde_gs[0, 1])
    ax_d = fig.add_subplot(bcde_gs[1, 0], sharex=ax_b)
    ax_e = fig.add_subplot(bcde_gs[1, 1], sharex=ax_c)

    print('Building panel B (fast direction, 2015-2026)...')
    im_b = draw_phi_panel(ax_b, df, t_min, t_max, x_width=3 * 24 * 3600, x_over=0.5,
                          title='Fast Direction Over an Eruption Cycle', show_legend=True,
                          legend_label='Eruption onset or end')
    print('Building panel C (fast direction, ±48 hr around eruption onset)...')
    im_c = draw_phi_panel(ax_c, df, zoom_start, zoom_end, x_width=6 * 3600, x_over=0.85,
                          title='Fast Direction: ±48 hr Around Eruption Onset', show_legend=True,
                          legend_label='Eruption onset')

    print('Building panel D (delay time, 2015-2026)...')
    im_d = draw_dt_panel(ax_d, df, t_min, t_max, x_width=3 * 24 * 3600, x_over=0.5,
                        title='Delay Time Over an Eruption Cycle')
    print('Building panel E (delay time, ±48 hr around eruption onset)...')
    im_e = draw_dt_panel(ax_e, df, zoom_start, zoom_end, x_width=6 * 3600, x_over=0.85,
                        title='Delay Time: ±48 hr Around Eruption Onset')

    ax_b.xaxis.set_major_locator(mdates.YearLocator())
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_d.xaxis.set_major_locator(mdates.YearLocator())
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_b.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=30, ha='right')

    ax_c.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax_e.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax_c.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.setp(ax_e.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # C/E: no y-axis (redundant with B/D's y-axis right next to them)
    ax_c.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_e.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_c.set_ylabel('')
    ax_e.set_ylabel('')

    for ax, im in [(ax_c, im_c), (ax_e, im_e)]:
        _add_matched_colorbar(fig, ax, im, 'Normalised density')

    # ── A-E labels, top-left of each panel ──────────────────────────────────
    # C/E are narrow, so their (centered) titles sit close to the left edge
    # where the label goes - they need more vertical clearance than A/B/D,
    # whose titles are far enough away horizontally to never collide even at
    # a small gap.
    label_gaps = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.01}
    label_x_offsets = {'A': -0.02, 'B': -0.02, 'C': -0.035, 'D': -0.02, 'E': -0.035}
    for ax, label in [(ax_a, 'A'), (ax_b, 'B'), (ax_c, 'C'), (ax_d, 'D'), (ax_e, 'E')]:
        pos = ax.get_position()
        fig.text(pos.x0 + label_x_offsets[label], pos.y1 + label_gaps[label], label,
                  fontsize=16 * FONT_SCALE, fontweight='bold', va='bottom', ha='left')

    fig.suptitle('AS1: Shear-wave Splitting Measurements Over Time', fontsize=17 * FONT_SCALE,
                fontweight='bold', y=0.995)
    fig.savefig(OUT_PDF, dpi=600, bbox_inches='tight')
    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
