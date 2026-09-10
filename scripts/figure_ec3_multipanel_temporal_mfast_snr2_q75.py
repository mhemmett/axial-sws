"""
Multipanel summary figure: AXEC3 shear-wave splitting measurements over time (2015-2026).

Variant of figure_ec3_multipanel_temporal_mfast_tier5.py (same panel layout A-D, same
movehisto2d moving-window binning, same 21-day/12-hour window widths, same two-page
percent-anisotropy/fractional-anisotropy structure -- see that script's docstring for the
full panel description, not repeated here), with ONE change: the QC filter, per explicit
user request, is now the session's newer "Page 1 @ quality>=0.75" grade (same filter used in
rose_tier5_7period_regions_newdata_snr2_q75.py /
traveltime_anisotropy_7period_6stations_newdata_snr2_q75.py) instead of the old tier-5 filter:

    OLD (tier5_tier5_tier5.py): quality>=0.5, dt<T_dom/2, phi_error<10 deg, dt_error<0.05s
    NEW (this script):          SNR>=2.0, quality>=0.75, dt<=T_dom/2, phi_error<=20 deg,
                                 dt_error<=0.05s

T_travel (S-wave travel time, for both the percent- and fractional-anisotropy panels) is
REUSED from traveltime_anisotropy_7period_6stations_newdata_snr_grades.py's ray-tracing
cache (AXEC3's rows only) rather than re-traced -- every AXEC3 event passing the new filter
was already traced as part of that script's broader quality>=0.5 cache, so this script only
re-filters + merges, it does not call the ray tracer at all.

Produces: figure_ec3_multipanel_temporal_mfast_snr2_q75.pdf (2 pages, same structure as the
tier-5 original -- does not overwrite it)

Run with: python3 figure_ec3_multipanel_temporal_mfast_snr2_q75.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy import UTCDateTime

FONT_SCALE = 1.3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sws_methods as swm  # noqa: E402
from rose_plots_lqt_pykonal_unweighted import ERUPTION_START, ERUPTION_END  # noqa: E402
import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T  # noqa: E402

TRANSFER_DIR = os.path.join(os.path.dirname(HERE), 'mfast_maxdt_pipeline_transfer')
OUT_PDF = os.path.join(HERE, 'figure_ec3_multipanel_temporal_mfast_snr2_q75.pdf')

STATION = 'AXEC3'

# ── New filter: "Page 1 @ quality>=0.75" (see module docstring) ──────────────────────────
SNR_MIN = 2.0
QW_MIN = 0.75
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05

# ── Anisotropy y-axes (unchanged from the tier-5 original) ───────────────────────────────
PCT_ANISO_Y_END = 10.0
FRAC_ANISO_Y_END = 10.0

# ── Moving-window time binning (unchanged from the tier-5 original) ──────────────────────
X_WIDTH_FULL = 21 * 24 * 3600   # 3 weeks
X_OVER_FULL = 0.5
X_WIDTH_ZOOM = 12 * 3600        # 12 hours
X_OVER_ZOOM = 0.85

ZOOM_HALF_WIDTH = pd.Timedelta(hours=24)


def load_axec3_newdata():
    files = ['splitting_results_AXEC3_2015_2021_all_batches.csv',
             'splitting_results_AXEC3_2022_2026_all_batches.csv']
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality', 'snr_horizontal', 'phi_error', 'dt_error',
                            'dominant_period', 'latitude', 'longitude', 'depth'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    df['phi_az'] = df['phi'] % 180.0
    return df


def apply_new_filter(df):
    return df[(df['quality'] >= QW_MIN) &
              (df['snr_horizontal'] >= SNR_MIN) &
              (df['dt'] <= df['dominant_period'] / 2.0) &
              (df['phi_error'] <= PHI_ERR_MAX) &
              (df['dt_error'] <= DT_ERR_MAX)].copy()


def add_T_travel_from_cache(df):
    """Attach T_travel for AXEC3 events by reusing the already-built ray-tracing cache
    (traveltime_anisotropy_7period_6stations_newdata_snr_grades.py) instead of re-tracing --
    every row surviving apply_new_filter is a strict subset of that cache's broadest
    (quality>=0.5, SNR>=2.0, phi_err<=20, dt_err<=0.05, dt<=T_dom/2) filter."""
    cache_exists = os.path.exists(T.RAY_SUMMARY_CSV) and os.path.exists(T.RAY_VOXEL_CSV)
    if not cache_exists:
        raise SystemExit(
            f'Ray-tracing cache not found ({T.RAY_SUMMARY_CSV} / {T.RAY_VOXEL_CSV}). '
            'Run traveltime_anisotropy_7period_6stations_newdata_snr_grades.py first.')

    T._setup_environment(need_tracer=False)
    summary, vox = T.load_cache()
    vray = vox['ray_id'].values
    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)

    N = len(summary)
    T_travel = np.zeros(N, dtype=np.float64)
    np.add.at(T_travel, vray, tt_seg)
    summary = summary.copy()
    summary['T_travel'] = T_travel
    summary['event_datetime_parsed'] = pd.to_datetime(
        summary['event_datetime'], utc=True, format='ISO8601')

    sta_summary = summary[summary['station'] == STATION][
        ['event_datetime_parsed', 'dt', 'T_travel']].drop_duplicates(
        subset=['event_datetime_parsed', 'dt'])

    merged = df.merge(sta_summary, left_on=['t', 'dt'], right_on=['event_datetime_parsed', 'dt'],
                      how='inner')
    n_lost = len(df) - len(merged)
    if n_lost:
        print(f'  Warning: {n_lost:,}/{len(df):,} AXEC3 rows had no matching cached ray '
              f'(should be 0 -- these events were somehow not in the quality>=0.5 broadest cache)')
    return merged.drop(columns=['event_datetime_parsed'])


def add_pct_aniso(df):
    df = df.copy()
    df['pct_aniso'] = df['dt'] / df['T_travel'] * 100.0
    df = df.dropna(subset=['pct_aniso'])
    return df[np.isfinite(df['pct_aniso'])]


def add_frac_aniso(df):
    """Fractional shear-wave anisotropy 'a', solved from a^2 + 2x*a - 4 = 0 with
    x = 2*T_travel/dt; kept only if 0<=a<=1, else the event is dropped."""
    df = df.copy()
    x = 2.0 * df['T_travel'] / df['dt']
    a = -x + np.sqrt(x**2 + 4.0)
    df['frac_aniso'] = a * 100.0
    mask = (a >= 0.0) & (a <= 1.0)
    return df[mask].copy()


# ── Panels A-D: moving-window density histograms (phi, anisotropy) vs time ───────────────

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


LEGEND_LABEL_FONTSIZE = 8 * FONT_SCALE


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
            return ax.legend(by_label.values(), by_label.keys(), loc='lower right',
                             fontsize=LEGEND_LABEL_FONTSIZE, framealpha=0.8)
    return None


def _legend_left_edge_axfrac(ax, legend):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer)
    frac = ax.transAxes.inverted().transform(bbox)
    return frac[0][0]


def _add_matched_colorbar(fig, ax, im, label_txt):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label_txt, fontsize=7 * FONT_SCALE)
    cb.ax.tick_params(labelsize=6 * FONT_SCALE)


def _format_window_length(x_width_s):
    if x_width_s >= 86400 and x_width_s % 86400 == 0:
        return f'{x_width_s/86400:.0f}-day window'
    return f'{x_width_s/3600:.0f}-hour window'


def draw_phi_panel(ax, df, x_start, x_end, x_width, x_over, title, show_legend=False,
                   legend_label='Eruption onset or end', annotate_window=False):
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
    legend = _mark_eruption(ax, x_start, x_end, y_mid=90, show_legend=show_legend, legend_label=legend_label)
    ax.set_ylabel(r'$\phi$ [$^o$ from N]', fontsize=9 * FONT_SCALE, fontweight='bold')
    ax.tick_params(labelsize=7 * FONT_SCALE)
    ax.set_title(title, fontsize=10 * FONT_SCALE)
    if annotate_window:
        left_x = _legend_left_edge_axfrac(ax, legend) if legend is not None else 0.7
        ax.text(left_x, 0.10, _format_window_length(x_width), transform=ax.transAxes,
                ha='left', va='bottom', fontsize=LEGEND_LABEL_FONTSIZE,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.8', alpha=0.85),
                zorder=6)
    return im


def draw_aniso_panel(ax, df, y_col, y_label, y_end, x_start, x_end, x_width, x_over, title):
    x = np.array([UTCDateTime(t.isoformat()) for t in pd.to_datetime(df['t'])])
    y = df[y_col].values
    im = plot_movehisto2d(
        x, y, ax=ax, cmap='Blues',
        x_width=x_width, x_start=UTCDateTime(x_start.isoformat()),
        x_end=UTCDateTime(x_end.isoformat()), x_over=x_over,
        y_start=0, y_end=y_end, y_width=y_end / 30, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=0.5, y_filter_per=3, filter_mode='nearest',
    )
    _mark_eruption(ax, x_start, x_end, y_mid=y_end * 0.5, show_legend=False)
    ax.set_ylabel(y_label, fontsize=9 * FONT_SCALE, fontweight='bold')
    ax.tick_params(labelsize=7 * FONT_SCALE)
    ax.set_title(title, fontsize=10 * FONT_SCALE)
    return im


def build_page(fig, df_phi, df_aniso, aniso_col, aniso_label, aniso_y_end,
               d_title, e_title, page_suptitle, t_min, t_max, zoom_start, zoom_end):
    LEFT, RIGHT = 0.05, 0.98

    abcd_top = 0.94
    abcd_bottom = 0.06
    hspace_abcd = 0.15 * 1.15

    abcd_gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.7, 0.375],
                               hspace=hspace_abcd, wspace=0.08,
                               left=LEFT, right=RIGHT, top=abcd_top, bottom=abcd_bottom)

    ax_a = fig.add_subplot(abcd_gs[0, 0])
    ax_b = fig.add_subplot(abcd_gs[0, 1])
    ax_c = fig.add_subplot(abcd_gs[1, 0], sharex=ax_a)
    ax_d = fig.add_subplot(abcd_gs[1, 1], sharex=ax_b)

    print(f'  Panel A (fast direction, full record, {X_WIDTH_FULL/86400:.0f}-day window)...')
    im_a = draw_phi_panel(ax_a, df_phi, t_min, t_max, x_width=X_WIDTH_FULL, x_over=X_OVER_FULL,
                          title='Fast Direction Over an Eruption Cycle', show_legend=True,
                          legend_label='Eruption onset or end', annotate_window=True)
    print(f'  Panel B (fast direction, ±24hr, {X_WIDTH_ZOOM/3600:.0f}-hour window)...')
    im_b = draw_phi_panel(ax_b, df_phi, zoom_start, zoom_end, x_width=X_WIDTH_ZOOM, x_over=X_OVER_ZOOM,
                          title='Fast Direction: ±24 hr Around Eruption Onset', show_legend=True,
                          legend_label='Eruption onset', annotate_window=True)

    print(f'  Panel C ({aniso_label}, full record, {X_WIDTH_FULL/86400:.0f}-day window)...')
    im_c = draw_aniso_panel(ax_c, df_aniso, aniso_col, aniso_label, aniso_y_end,
                            t_min, t_max, x_width=X_WIDTH_FULL, x_over=X_OVER_FULL, title=d_title)
    print(f'  Panel D ({aniso_label}, ±24hr, {X_WIDTH_ZOOM/3600:.0f}-hour window)...')
    im_d = draw_aniso_panel(ax_d, df_aniso, aniso_col, aniso_label, aniso_y_end,
                            zoom_start, zoom_end, x_width=X_WIDTH_ZOOM, x_over=X_OVER_ZOOM, title=e_title)

    ax_a.xaxis.set_major_locator(mdates.YearLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_c.xaxis.set_major_locator(mdates.YearLocator())
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_a.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.setp(ax_c.xaxis.get_majorticklabels(), rotation=30, ha='right')

    ax_b.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax_d.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax_b.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=30, ha='right')

    ax_b.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_d.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_b.set_ylabel('')
    ax_d.set_ylabel('')

    for ax, im in [(ax_b, im_b), (ax_d, im_d)]:
        _add_matched_colorbar(fig, ax, im, 'Normalised density')

    label_gaps = {'A': 0.01, 'B': 0.02, 'C': 0.01, 'D': 0.02}
    label_x_offsets = {'A': -0.02, 'B': -0.035, 'C': -0.02, 'D': -0.035}
    for ax, label in [(ax_a, 'A'), (ax_b, 'B'), (ax_c, 'C'), (ax_d, 'D')]:
        pos = ax.get_position()
        fig.text(pos.x0 + label_x_offsets[label], pos.y1 + label_gaps[label], label,
                  fontsize=16 * FONT_SCALE, fontweight='bold', va='bottom', ha='left')

    fig.suptitle(page_suptitle, fontsize=17 * FONT_SCALE, fontweight='bold', y=0.995)


def main():
    print('Loading AXEC3 results (newdata mfast max_dt=0.2s pipeline, both periods)...')
    raw = load_axec3_newdata()
    df = apply_new_filter(raw)
    print(f'  {len(raw):,} baseline (success, dt>0) -> {len(df):,} after new filter '
          f'(SNR>={SNR_MIN}, quality>={QW_MIN}, dt<=T_dom/2, phi_error<={PHI_ERR_MAX}, '
          f'dt_error<={DT_ERR_MAX})')

    print('Reusing ray-tracing cache for T_travel (no re-tracing)...')
    df = add_T_travel_from_cache(df)
    print(f'  {len(df):,} events with T_travel from cache')

    df_pct = add_pct_aniso(df)
    df_frac = add_frac_aniso(df)
    print(f'  {len(df_pct):,} events for percent-anisotropy page, '
          f'{len(df_frac):,} events for fractional-anisotropy page (0<=a<=1 kept)')

    t_min, t_max = df['t'].min(), df['t'].max()
    zoom_start = ERUPTION_START - ZOOM_HALF_WIDTH
    zoom_end = ERUPTION_START + ZOOM_HALF_WIDTH

    with PdfPages(OUT_PDF) as pdf:
        print('\n=== Page 1: percent anisotropy ===')
        fig1 = plt.figure(figsize=(20, 12))
        build_page(
            fig1, df, df_pct, 'pct_aniso', r'$\delta t / T$ [%]', PCT_ANISO_Y_END,
            d_title='Percent Anisotropy Over an Eruption Cycle',
            e_title='Percent Anisotropy: ±24 hr Around Eruption Onset',
            page_suptitle='AXEC3: Shear-wave Splitting Measurements Over Time',
            t_min=t_min, t_max=t_max, zoom_start=zoom_start, zoom_end=zoom_end,
        )
        pdf.savefig(fig1, dpi=600, bbox_inches='tight')
        plt.close(fig1)

        print('\n=== Page 2: fractional anisotropy ===')
        fig2 = plt.figure(figsize=(20, 12))
        build_page(
            fig2, df, df_frac, 'frac_aniso', 'Fractional anisotropy [%]',
            FRAC_ANISO_Y_END,
            d_title='Fractional Anisotropy Over an Eruption Cycle',
            e_title='Fractional Anisotropy: ±24 hr Around Eruption Onset',
            page_suptitle='AXEC3: Shear-wave Splitting Measurements Over Time',
            t_min=t_min, t_max=t_max, zoom_start=zoom_start, zoom_end=zoom_end,
        )
        pdf.savefig(fig2, dpi=600, bbox_inches='tight')
        plt.close(fig2)

    print(f'\nWrote {OUT_PDF}')


if __name__ == '__main__':
    main()
