"""
figure_all_stations_temporal_traveltime_anisotropy_newdata.py

Regenerated version of figure_all_stations_temporal_traveltime_anisotropy.py using the NEW
mfast max_dt=0.2s data (and its associated filter convention) for the 4 stations that have it:

  AXCC1/AXEC1: mfast_maxdt_pipeline_transfer/splitting_results_{AXCC1,AXEC1}_
    {2015_2021,2022_2026}_all_batches.csv (complete)
  AXEC2: mfast_maxdt_pipeline_transfer/splitting_results_AXEC2_2022_2026_all_batches.csv
    + our own COMPLETED maxdt02 re-run's 2015-2021 batches (all 497 batches --
    production_axec2_mfast_filters_maxdt02_lqt_pykonal_results/*.csv), joined against
    raw_axec2_all_batches_mfast_filters_data/raw_axec2_all_batches_mfast_filters_metadata.csv
    for event lat/lon/depth.
  AXAS2: mfast_maxdt_pipeline_transfer/splitting_results_AXAS2_{2015_2021,2022_2026}_
    all_batches.csv (2022-2026 rerun completed 2026-07-23, moved here from the OLD/fixed
    5-40 Hz set it used previously).

AXAS1, AXEC3 keep the existing OLD (fixed 5-40 Hz) data -- no new data exists for them yet.

Filters (matching the convention established for this new data elsewhere this session --
rose_7period_axcc1_axec1_axec2_newdata.py, histograms_all_columns_newdata_combined.py,
traveltime_anisotropy_7period_axcc1_axec1_axec2_newdata.py, deformation_util.py's
get_station_obs_hemmett): success, dt>0, quality>=0.5, phi_error<20 deg, dt_error<0.05s for
ALL 6 stations, PLUS dt<T_dom/2 (cycle-skip-risk cut, per-event dominant period) for the 4
new-data stations only -- the 2 old-data stations' pipeline doesn't track a per-event
dominant period so that cut can't be applied to them (same caveat as the original script /
figure_all_stations_temporal_fractional_dt.py).

Method (unchanged from the original): dt/T_travel*100 (%), T_travel = S-wave travel time
from event hypocenter to station via ray-tracing through the Baillard 3D Vs model
(pykonal_raytracer.BaillardRayTracer) + continuous ds/Vs integration along the ray
(baillard_velocity.vs_at). Time-windowed (14-day moving window, full record; 6hr moving
window, +/-48hr eruption zoom), normalized-density (flag_y_norm=True), fixed 0-15% y-axis.

Produces: figure_all_stations_temporal_traveltime_anisotropy_newdata.pdf

Run with:
    python3 figure_all_stations_temporal_traveltime_anisotropy_newdata.py
"""

import glob
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy import UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sws_methods as swm  # noqa: E402
from rose_plots_lqt_pykonal_unweighted import ERUPTION_START, ERUPTION_END  # noqa: E402
from pykonal_raytracer import BaillardRayTracer
from baillard_velocity import vs_at

DATA_DIR = os.path.join(os.path.dirname(HERE), 'lqt_pykonal_combined_results')
TRANSFER_DIR = os.path.join(os.path.dirname(HERE), 'mfast_maxdt_pipeline_transfer')
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
AXEC2_META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                              'raw_axec2_all_batches_mfast_filters_metadata.csv')
OUT_PDF = os.path.join(HERE, 'figure_all_stations_temporal_traveltime_anisotropy_newdata.pdf')

QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
NEWDATA_STATIONS = {'AXCC1', 'AXEC1', 'AXEC2', 'AXAS2'}
STA_DISPLAY = {'AXAS1': 'AXAS1', 'AXAS2': 'AXAS2 (new mfast max_dt=0.2s)',
               'AXCC1': 'AXCC1 (new mfast max_dt=0.2s)',
               'AXEC1': 'AXEC1 (new mfast max_dt=0.2s)', 'AXEC2': 'AXEC2 (new mfast max_dt=0.2s)',
               'AXEC3': 'AXEC3'}

STATION_FILE = '../data/stations_axial.llz'

OLD_STATION_FILES = {
    'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches_enriched.csv',
              'splitting_results_AXAS1_2022_2026_all_batches_enriched.csv'],
    'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches_enriched.csv',
              'splitting_results_AXEC3_2022_2026_all_batches_enriched.csv'],
}
NEWDATA_STATION_FILES = {
    'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
              'splitting_results_AXCC1_2022_2026_all_batches.csv'],
    'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
              'splitting_results_AXEC1_2022_2026_all_batches.csv'],
    'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
    'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv',
              'splitting_results_AXAS2_2022_2026_all_batches.csv'],
}

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


def _qc(df):
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['quality', 'phi_error', 'dt_error'])
    df = df[(df['quality'] >= QW_MIN) & (df['phi_error'] < PHI_ERR_MAX) & (df['dt_error'] < DT_ERR_MAX)]
    return df


def load_old_station(sta):
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in OLD_STATION_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = _qc(df)
    df = df.dropna(subset=['event_lat', 'event_lon', 'event_depth'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    return df[['dt', 't', 'event_lat', 'event_lon', 'event_depth']].rename(
        columns={'event_lat': 'lat', 'event_lon': 'lon', 'event_depth': 'depth'})


def load_newdata_station(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in NEWDATA_STATION_FILES[sta]]
    if sta == 'AXEC2':
        batch_files = sorted(glob.glob(os.path.join(
            AXEC2_2015_2021_DIR,
            'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_*.csv')))
        batch_dfs = [pd.read_csv(f) for f in batch_files]
        batch_dfs = [d for d in batch_dfs if len(d) > 0]
        meta = pd.read_csv(AXEC2_META_CSV)[['event_id', 'latitude', 'longitude', 'depth']]
        for d in batch_dfs:
            d['dominant_period'] = d['chosen_filter_dom_period_samples'] / 200.0
            d = d.merge(meta, on='event_id', how='left')
            dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df = _qc(df)
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'dominant_period'])
    df = df[df['dt'] < df['dominant_period'] / 2.0]
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
    return df[['dt', 't', 'latitude', 'longitude', 'depth']].rename(
        columns={'latitude': 'lat', 'longitude': 'lon'})


print('Loading per-station events...')
station_events = {}
for sta in STATIONS:
    if sta in NEWDATA_STATIONS:
        df = load_newdata_station(sta)
        print(f'  {sta} (new mfast max_dt=0.2s): {len(df):,} events')
    else:
        df = load_old_station(sta)
        print(f'  {sta} (old): {len(df):,} events')
    station_events[sta] = df

# ── Station coordinates + ray tracer ─────────────────────────────────────────

_sta_df = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                      engine='python').set_index('s')
_sta_df = _sta_df.loc[[s for s in STATIONS if s in _sta_df.index]]
_sta_df['x'], _sta_df['y'] = ll2xy(_sta_df['lat'].values, _sta_df['lon'].values)

print('\nPrecomputing FMM travel-time fields for all 6 stations...')
tracer = BaillardRayTracer()
for sta in STATIONS:
    tracer.precompute_station(sta, float(_sta_df.loc[sta, 'x']), float(_sta_df.loc[sta, 'y']))

Z_MAX = 4.0
N_RAY = 200

print('\nTracing rays + integrating ds/Vs (Baillard model) to get S-wave travel time T_travel per event...')
for sta in STATIONS:
    df = station_events[sta]
    x, y = ll2xy(df['lat'].values, df['lon'].values)
    T_travel = np.full(len(df), np.nan)
    t0 = time.time()
    for i, (ex, ey, ez) in enumerate(zip(x, y, df['depth'].values)):
        if ez < 0 or ez > Z_MAX:
            continue
        try:
            ray = tracer.trace(sta, float(ex), float(ey), float(ez), n_pts=N_RAY)
            mid = 0.5 * (ray[:-1] + ray[1:])
            ds = np.linalg.norm(np.diff(ray, axis=0), axis=1)
            vs_mid = vs_at(mid[:, 0], mid[:, 1], mid[:, 2])
            tt = float(np.sum(ds / np.maximum(vs_mid, 1e-9)))
            if tt > 0:
                T_travel[i] = tt
        except RuntimeError:
            continue
        if (i + 1) % 10000 == 0:
            print(f'  {sta}: {i+1:,}/{len(df):,}  {time.time()-t0:.0f}s', end='\r', flush=True)
    df['T_travel'] = T_travel
    n_valid = np.isfinite(T_travel).sum()
    print(f'  {sta}: {n_valid:,}/{len(df):,} events traced successfully ({time.time()-t0:.0f}s)')
    df['pct_aniso'] = df['dt'] / df['T_travel'] * 100.0
    station_events[sta] = df.dropna(subset=['pct_aniso'])
    station_events[sta] = station_events[sta][np.isfinite(station_events[sta]['pct_aniso'])]

# ── Shared y-axis range across all stations ──────────────────────────────────

Y_END = 15.0
print(f'\nShared y-axis range: 0-{Y_END:.2f}% (fixed)')


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


def _mark_eruption(ax, x_start, x_end, y_mid, show_legend=False):
    eruption_marks = [ERUPTION_START, ERUPTION_END] if (x_start < ERUPTION_END and x_end > ERUPTION_START) else [ERUPTION_START]
    for ex in eruption_marks:
        ex_mpl = mdates.date2num(ex.to_pydatetime())
        if ax.get_xlim()[0] <= ex_mpl <= ax.get_xlim()[1]:
            ax.axvline(ex_mpl, color='black', linestyle='--', linewidth=1.3, alpha=0.9,
                       label='Eruption onset or end')
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7, framealpha=0.8)


ZOOM_HALF_WIDTH = pd.Timedelta(hours=48)


def draw_pct_panel(ax, df, x_start, x_end, x_width, x_over, title, show_legend=False):
    x = np.array([UTCDateTime(t.isoformat()) for t in pd.to_datetime(df['t'])])
    y = df['pct_aniso'].values
    im = plot_movehisto2d(
        x, y, ax=ax, cmap='Blues',
        x_width=x_width, x_start=UTCDateTime(x_start.isoformat()),
        x_end=UTCDateTime(x_end.isoformat()), x_over=x_over,
        y_start=0, y_end=Y_END, y_width=Y_END / 30, y_over=0.9,
        flag_y_norm=True, flag_resample=True, flag_filter=True,
        x_filter_per=0.5, y_filter_per=3, filter_mode='nearest',
    )
    _mark_eruption(ax, x_start, x_end, y_mid=Y_END * 0.5, show_legend=show_legend)
    ax.set_ylabel(r'$\delta t / T$ [%]', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    ax.tick_params(labelsize=7)
    return im


def main():
    n_sta = len(STATIONS)
    fig, axes = plt.subplots(n_sta, 2, figsize=(18, 3.0 * n_sta),
                             gridspec_kw={'width_ratios': [1.7, 0.375], 'wspace': 0.12, 'hspace': 0.45})

    for i, sta in enumerate(STATIONS):
        df = station_events[sta]
        t_min, t_max = df['t'].min(), df['t'].max()
        zoom_start = ERUPTION_START - ZOOM_HALF_WIDTH
        zoom_end = ERUPTION_START + ZOOM_HALF_WIDTH
        print(f'Drawing panels for {sta} (N={len(df):,})...')

        ax_full, ax_zoom = axes[i, 0], axes[i, 1]

        im = draw_pct_panel(ax_full, df, t_min, t_max, x_width=14 * 24 * 3600, x_over=0.5,
                            title=f'{STA_DISPLAY[sta]}  N={len(df):,}', show_legend=(i == 0))
        divider = make_axes_locatable(ax_full)
        cax = divider.append_axes('right', size='1.5%', pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Normalised density', fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax_full.xaxis.set_major_locator(mdates.YearLocator())
        ax_full.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax_full.xaxis.get_majorticklabels(), rotation=30, ha='right')

        zoom_df = df[(df['t'] >= zoom_start) & (df['t'] <= zoom_end)]
        if len(zoom_df) == 0:
            ax_zoom.text(0.5, 0.5, 'No events in +/-48hr window', ha='center', va='center',
                         fontsize=8, transform=ax_zoom.transAxes)
            ax_zoom.set_title(f'{STA_DISPLAY[sta]}: +/-48hr around eruption onset  N=0',
                              fontsize=10, fontweight='bold', loc='left')
            ax_zoom.set_xticks([])
            ax_zoom.set_yticks([])
        else:
            im_z = draw_pct_panel(ax_zoom, df, zoom_start, zoom_end, x_width=6 * 3600, x_over=0.85,
                                  title=f'{STA_DISPLAY[sta]}: +/-48hr around eruption onset  N={len(zoom_df):,}',
                                  show_legend=(i == 0))
            divider_z = make_axes_locatable(ax_zoom)
            cax_z = divider_z.append_axes('right', size='4%', pad=0.08)
            cb_z = fig.colorbar(im_z, cax=cax_z)
            cb_z.set_label('Normalised density', fontsize=7)
            cb_z.ax.tick_params(labelsize=6)
            ax_zoom.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.setp(ax_zoom.xaxis.get_majorticklabels(), rotation=30, ha='right')
            ax_zoom.set_ylabel('')
            ax_zoom.tick_params(axis='y', which='both', left=False, labelleft=False)

    fig.suptitle('Percent anisotropy ($\\delta t$ / S-wave travel time, Baillard-Vs ray-traced) over time, all stations\n'
                 '(AXCC1/AXEC1/AXEC2/AXAS2: new mfast max_dt=0.2s data, others: old), '
                 f'quality>={QW_MIN}, phi_err<{PHI_ERR_MAX:.0f}deg, dt_err<{DT_ERR_MAX}s '
                 '(+ dt<T_dom/2 for new-data stations)',
                 fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nWrote {OUT_PDF}')


if __name__ == '__main__':
    main()
