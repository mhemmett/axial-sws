#!/usr/bin/env python3
"""
percent_anisotropy_vs_depth_per_station_newdata_snr2_q75.py

Per-station 2D histogram of per-event travel-time-weighted percent anisotropy
(A = dt*100/T_travel, T_travel = sum(seg_len_km/Vs_voxel) over the PyKonal-traced ray) vs.
earthquake source depth (km), for the six-station "newdata" dataset -- the per-event,
per-station analog of the map-averaged percent-anisotropy tomography in
traveltime_anisotropy_7period_6stations_newdata_snr_grades.py, following the same 2D
pcolormesh/LogNorm style as dt_vs_depth_per_station_newdata_snr2_q75.py.

Reuses the EXISTING ray-tracing cache built by traveltime_anisotropy_7period_6stations_
newdata_snr_grades.py (traveltime_anisotropy_newdata_snr_grades_ray_summary.csv /
_ray_voxel_segments.csv, traced once against the broadest Page-1/quality>=0.5 filter --
already present on disk, so no re-tracing needed here). Since the cached ray summary
doesn't itself store source depth, depth is recovered by joining each cached ray's
(station, event_datetime) back to the broadest per-station "newdata" load (same source
CSVs, same datetime column) to pull its 'z' (depth) value.

Single filter (same "Page 1 @ quality>=0.75" grade used by the other newdata scripts this
session): SNR >= 2.0, quality (Q_w) >= 0.75, dt_err <= 0.05s, dt <= T_dom/2, phi_err <= 20 deg.

Produces:
    percent_anisotropy_vs_depth_per_station_newdata_snr2_q75.pdf
(6 pages, one station per page) -- a NEW file, does not touch any existing output.

Run with:
    python3 percent_anisotropy_vs_depth_per_station_newdata_snr2_q75.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'percent_anisotropy_vs_depth_per_station_newdata_snr2_q75.pdf')

GRADE = T.GRADES[3]   # Page 1 @ quality>=0.75: SNR>=2.0, quality>=0.75, dt_err<=0.05s, dt<=T_dom/2, phi_err<=20 deg
assert GRADE['key'] == 'page1_q0.75', GRADE

N_DEPTH_BINS = 40
N_A_BINS = 40


def build_depth_lookup():
    print('Rebuilding broadest per-station loads to recover source depth per ray...')
    lookup = {}
    for sta in T.STATIONS:
        df = T.load_station_broadest(sta)
        lookup[sta] = pd.Series(df['z'].values, index=df['t'].values)
        print(f'  {sta}: {len(df):,} broadest-filter events loaded for depth lookup')
    return lookup


def main():
    T._setup_environment(need_tracer=False)

    cache_exists = os.path.exists(T.RAY_SUMMARY_CSV) and os.path.exists(T.RAY_VOXEL_CSV)
    if not cache_exists:
        raise RuntimeError(
            f'Ray-tracing cache not found ({T.RAY_SUMMARY_CSV}); run '
            f'traveltime_anisotropy_7period_6stations_newdata_snr_grades.py once first to build it.')

    summary, vox = T.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all(), 'ray_id not contiguous 0..N-1'
    vray = vox['ray_id'].values

    print('\nComputing T_travel per ray and A_ray = dt*100/T_travel from cache...')
    tt_seg, a_ray_tt = T.compute_ttime_anisotropy(summary, vox, vray)
    summary['a_ray'] = a_ray_tt

    summary['t'] = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')

    depth_lookup = build_depth_lookup()
    depths = np.full(N, np.nan)
    for sta in T.STATIONS:
        m = (summary['station'].values == sta)
        depths[m] = depth_lookup[sta].reindex(summary['t'].values[m]).values
    summary['depth'] = depths

    n_matched = int(np.isfinite(depths).sum())
    print(f'  matched depth for {n_matched:,}/{N:,} cached rays')

    mask = ((summary['quality'].values >= GRADE['qw_min']) &
            (summary['snr_horizontal'].values >= GRADE['snr_min']) &
            (summary['phi_error'].values <= GRADE['phi_err_max']) &
            np.isfinite(summary['a_ray'].values) &
            np.isfinite(summary['depth'].values))

    print(f'\nWriting {os.path.basename(OUT_PDF)}...')
    with PdfPages(OUT_PDF) as pdf:
        for sta in T.STATIONS:
            sm = mask & (summary['station'].values == sta)
            depth = summary['depth'].values[sm]
            a = summary['a_ray'].values[sm]
            print(f'  {sta}: {sm.sum():,} events pass the filter (with matched depth)')

            fig, ax = plt.subplots(figsize=(7.5, 6))
            depth_edges = np.linspace(depth.min(), depth.max(), N_DEPTH_BINS + 1)
            a_lo, a_hi = np.nanpercentile(a, [1, 99])
            a_edges = np.linspace(min(0, a_lo), a_hi, N_A_BINS + 1)
            counts, xe, ye = np.histogram2d(depth, a, bins=[depth_edges, a_edges])
            im = ax.pcolormesh(xe, ye, counts.T, cmap='viridis',
                               norm=mcolors.LogNorm(vmin=1, vmax=max(counts.max(), 2)))
            fig.colorbar(im, ax=ax, label='Count')
            ax.set_xlabel('Source depth (km)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Percent anisotropy (%)', fontsize=10, fontweight='bold')

            fig.suptitle(f'{sta} — percent anisotropy vs. depth (N={sm.sum():,})\n'
                         f'{GRADE["label"]}',
                         fontsize=11, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {OUT_PDF}')


if __name__ == '__main__':
    main()
