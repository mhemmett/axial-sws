"""
Multipanel summary figure: AXEC3 shear-wave splitting measurements over time (2015-2026), for
building a manuscript figure. One-off script, not part of the pipeline.

Variant of figure_ec2_multipanel_temporal_mfast.py, adapted for AXEC3 with several changes per
explicit user request:

  1. Filter: this session's "tightest" tier-5 QC (rose_7period_6stations_newdata.py /
     traveltime_anisotropy_7period_6stations_newdata_phi10.py convention) -- quality>=0.5,
     dt<T_dom/2 (cycle-skip-risk cut, per-event dominant period), phi_error<10 deg (the
     tightest phi threshold used anywhere this session, vs. the EC2 mfast script's looser
     phi_error<20/dt_error<0.04), dt_error<0.05s.
  2. Panels A-D use the ORIGINAL movehisto2d moving-window/overlap time binning (an equal-N
     time-binning variant was tried and reverted per user request), but with WIDER windows
     than the EC2 template to pull in more events per window and reduce scatter:
     X_WIDTH_FULL=21 days (3 weeks) (vs. EC2's 3 days) for the full 2015-2026 panels (A, C), and
     X_WIDTH_ZOOM=12 hours (vs. EC2's 6 hours) for the +/-24hr eruption-onset zoom panels
     (B, D). Overlap fractions (x_over) are unchanged (0.5 full, 0.85 zoom).
  3. Two pages: page 1 keeps panels C/D as travel-time-weighted PERCENT ANISOTROPY
     (dt/T_travel*100, %); page 2 is identical except C/D instead plot the FRACTIONAL
     shear-wave anisotropy 'a' (%) -- same quadratic-relation definition as
     traveltime_anisotropy_7period_6stations_newdata_fractional.py: solve
     a^2 + 2x*a - 4 = 0 with x = 2*T_travel/dt, keep a=-x+sqrt(x^2+4) only if 0<=a<=1 (events
     outside that range are dropped for the fractional-anisotropy page only -- panels A/B
     are unaffected and identical on both pages). Both use the same ray-traced T_travel
     (pykonal_raytracer.BaillardRayTracer + baillard_velocity.vs_at, same method as
     figure_all_stations_temporal_traveltime_anisotropy_newdata.py). Both anisotropy panels'
     y-axis is clamped at 10% (not the EC2/figure_all_stations convention's 15%).
  4. The original panel A (BAZ-ring rose) is REMOVED per user request; the remaining 4 panels
     are re-lettered A-D (previously B-E).

  - Data: mfast_maxdt_pipeline_transfer/splitting_results_AXEC3_{2015_2021,2022_2026}_
    all_batches.csv (new mfast max_dt=0.2s pipeline, complete for both periods -- unlike
    AXEC2, AXEC3 doesn't need a dual-source old/new merge).

Panels (per page):
  A. Fast direction (phi) vs time, 21-day (3-week) moving window, full 2015-2026 record.
  B. Same as A, restricted to +/-24 hours around the 2015 eruption onset, 12-hour moving window.
  C. Page 1: percent anisotropy (dt/T_travel*100) vs time, 21-day (3-week) moving window, full record.
     Page 2: fractional anisotropy 'a' (%) vs time, 21-day (3-week) moving window, full record.
  D. Same as C, restricted to +/-24 hours around eruption onset, 12-hour moving window.

Run with: python3 figure_ec3_multipanel_temporal_mfast_tier5.py
"""

import os
import sys
import time

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
from pykonal_raytracer import BaillardRayTracer  # noqa: E402
from baillard_velocity import vs_at  # noqa: E402

TRANSFER_DIR = os.path.join(os.path.dirname(HERE), 'mfast_maxdt_pipeline_transfer')
OUT_PDF = os.path.join(HERE, 'figure_ec3_multipanel_temporal_mfast_tier5.pdf')
STATION_FILE = os.path.join(HERE, '..', 'data', 'stations_axial.llz')

STATION = 'AXEC3'

# ── Tier-5 QC (this session's tightest filter) ────────────────────────────────────────────
QW_MIN = 0.5
PHI_ERR_MAX = 10.0
DT_ERR_MAX = 0.05

# ── Anisotropy y-axes + ray tracing (matches figure_all_stations_temporal_traveltime_
#    anisotropy_newdata.py, except the y-clamp which is tightened to 10% per user request) ──
PCT_ANISO_Y_END = 10.0
FRAC_ANISO_Y_END = 10.0
Z_MAX = 4.0
N_RAY = 200

# ── Moving-window time binning (widened vs. the EC2 template to reduce scatter) ─────────────
X_WIDTH_FULL = 21 * 24 * 3600   # 3 weeks (EC2 template used 3 days)
X_OVER_FULL = 0.5
X_WIDTH_ZOOM = 12 * 3600        # 12 hours (EC2 template used 6 hours)
X_OVER_ZOOM = 0.85

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

ZOOM_HALF_WIDTH = pd.Timedelta(hours=24)


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


def load_axec3_newdata():
    files = ['splitting_results_AXEC3_2015_2021_all_batches.csv',
             'splitting_results_AXEC3_2022_2026_all_batches.csv']
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality', 'phi_error', 'dt_error', 'dominant_period',
                            'latitude', 'longitude', 'depth'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    df['phi_az'] = df['phi'] % 180.0
    return df


def apply_tier5_filter(df):
    return df[(df['quality'] >= QW_MIN) &
              (df['dt'] < df['dominant_period'] / 2.0) &
              (df['phi_error'] < PHI_ERR_MAX) &
              (df['dt_error'] < DT_ERR_MAX)].copy()


def add_T_travel(df):
    """Ray-trace each QC-surviving event and attach T_travel (S-wave travel time, s), same
    method as figure_all_stations_temporal_traveltime_anisotropy_newdata.py."""
    sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                         engine='python').set_index('s')
    sx, sy = ll2xy(sta_df.loc[STATION, 'lat'], sta_df.loc[STATION, 'lon'])

    print('  Precomputing FMM travel-time field...')
    tracer = BaillardRayTracer()
    tracer.precompute_station(STATION, float(sx), float(sy))

    x, y = ll2xy(df['latitude'].values, df['longitude'].values)
    depth = df['depth'].values
    T_travel = np.full(len(df), np.nan)
    t0 = time.time()
    print(f'  Tracing {len(df):,} rays + integrating ds/Vs (Baillard model)...')
    for i, (ex, ey, ez) in enumerate(zip(x, y, depth)):
        if ez < 0 or ez > Z_MAX:
            continue
        try:
            ray = tracer.trace(STATION, float(ex), float(ey), float(ez), n_pts=N_RAY)
            mid = 0.5 * (ray[:-1] + ray[1:])
            ds = np.linalg.norm(np.diff(ray, axis=0), axis=1)
            vs_mid = vs_at(mid[:, 0], mid[:, 1], mid[:, 2])
            tt = float(np.sum(ds / np.maximum(vs_mid, 1e-9)))
            if tt > 0:
                T_travel[i] = tt
        except RuntimeError:
            continue
        if (i + 1) % 5000 == 0:
            print(f'    {i+1:,}/{len(df):,}  {time.time()-t0:.0f}s', end='\r', flush=True)
    n_valid = np.isfinite(T_travel).sum()
    print(f'  {n_valid:,}/{len(df):,} events traced successfully ({time.time()-t0:.0f}s)')
    df = df.copy()
    df['T_travel'] = T_travel
    df = df.dropna(subset=['T_travel'])
    return df[np.isfinite(df['T_travel'])]


def add_pct_aniso(df):
    df = df.copy()
    df['pct_aniso'] = df['dt'] / df['T_travel'] * 100.0
    df = df.dropna(subset=['pct_aniso'])
    return df[np.isfinite(df['pct_aniso'])]


def add_frac_aniso(df):
    """Fractional shear-wave anisotropy 'a', solved from a^2 + 2x*a - 4 = 0 with
    x = 2*T_travel/dt (see traveltime_anisotropy_7period_6stations_newdata_fractional.py's
    docstring for the derivation); kept only if 0<=a<=1, else the event is dropped."""
    df = df.copy()
    x = 2.0 * df['T_travel'] / df['dt']
    a = -x + np.sqrt(x**2 + 4.0)
    df['frac_aniso'] = a * 100.0
    mask = (a >= 0.0) & (a <= 1.0)
    return df[mask].copy()


# ── Panels B-E: moving-window density histograms (phi, anisotropy) vs time ───────────────

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


LEGEND_LABEL_FONTSIZE = 8 * FONT_SCALE   # shared by the eruption legend and the window-length label


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
    """The legend box's LEFT edge in axes-fraction x, measured from its actual rendered
    extent (its width depends on the legend_label text, so this must be measured per-panel,
    not assumed)."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer)
    frac = ax.transAxes.inverted().transform(bbox)
    return frac[0][0]


def _add_matched_colorbar(fig, ax, im, label_txt):
    """Colorbar sized to exactly match ax's own height (not fig.colorbar's
    fraction/shrink approximation)."""
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
        # Placed just above the bottom-right eruption-onset/end legend (loc='lower right'), with
        # its LEFT edge matching the legend's own LEFT edge -- measured from the legend's actual
        # rendered extent (which varies with legend_label's text length), not a fixed guess.
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
    hspace_abcd = 0.15 * 1.15   # +15% from the previous 0.15 (which was cut 50% from 0.30)

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
    print('Loading AXEC3 results (new mfast max_dt=0.2s pipeline, both periods)...')
    raw = load_axec3_newdata()
    df = apply_tier5_filter(raw)
    print(f'  {len(raw):,} baseline (success, dt>0) -> {len(df):,} after tier-5 filter '
          f'(quality>={QW_MIN}, dt<T_dom/2, phi_error<{PHI_ERR_MAX}, dt_error<{DT_ERR_MAX})')

    print('Ray-tracing for T_travel (shared by both anisotropy definitions)...')
    df = add_T_travel(df)
    print(f'  {len(df):,} events with valid T_travel')

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
