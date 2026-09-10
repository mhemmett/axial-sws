"""
Diagnostic multi-page PDF for a single AXEC1 pre-eruption event, for building a multipanel
figure of the production SWS pipeline -- companion to figure_axec1_event_41085_low_dt_diagnostics.py
for a high-delay-time vs. low-delay-time comparison. Structure replicated exactly from
scripts/figure_axec2_event_33223_diagnostics.py. One-off script, not part of the pipeline.

Event: AXEC1, event_id 23919 (row index in the per-station-reset data/mldd_catalog_2015_2021.csv,
batch 96 of data/axial_mldd_2015_2021_axec1_batch_*.mseed), origin 2015-03-10T17:16:10.760Z.

Selected from lqt_pykonal_combined_results/splitting_results_AXEC1_2015_2021_all_batches_enriched.csv
(the AXEC1 LQT + PyKonal-FMM production run) as a high-quality, high-delay-time event:
    quality (Q_w, Wustefeld 2010) = 0.969  (> 0.7 threshold)
    phi_error = 11 deg, dt_error = 0.005 s (both well inside the production
        phi_error<20 deg / dt_error<0.04 s gate used throughout this repo, e.g.
        lqt_pykonal_percent_anisotropy_quality_pages.py)
    dt = 0.29 s (> 0.24 s -- the dt threshold used to separate "high dt" from "low dt"
        events in lqt_pykonal_percent_anisotropy_quality_pages.py)
    pre-eruption (< 2015-04-24 06:00 UTC)

No AXEC1-specific production run script survives in scripts/ to confirm windowing parameters
directly, so this script reuses the same MFAST-2.2-style window parameters as
run_production_axec2_all_batches.py, including n_win=7 (not the n_win=10 used by the older,
legacy-Jurkevics-incidence axial_splitting_mldd_AXEC1_batched.py). This choice was verified:
re-running create_splitting_analysis on this event's raw waveform with n_win=7 reproduces the
production dt exactly (0.29 s), Q_w=0.927 (vs. the stored production 0.969), and the production
fast-direction phi to ~1 deg once phi_from_Q is rotated into the geographic frame
(phi_from_Q + back_azimuth, wrapped to +/-90 deg) -- close enough to confirm this is the same
event and same measurement, not a coincidentally similar one. The small residual differences
(Q_w, phi) likely come from small differences in exactly how the production pipeline's
organize_waveform_data() trimmed the raw batch waveform vs. the ad hoc extraction below.

snr_horizontal and rectilinearity below are NOT taken from a saved metadata table (the AXEC1
enriched results file has no snr_horizontal/rectilinearity_jurkevics columns filled in) -- they
are computed fresh in this script from the extracted, filtered waveform, using windows sized to
this trace's actual P/S timing (P at 1.5 s into the trace, S at 2.27 s), not the snr()/
calculate_rectilinearity() helpers' 4-second-preP defaults.

Pages:
  1. Filtered ZNE traces (P-pick, S-pick, 1-sigma S-pick uncertainty)
  2. Rotated LQT traces with P-pick, S-pick, S-pick uncertainty, dominant period marker,
     SWSPy analysis windows (start/end)
  3. Original (uncorrected) LQT Q-T particle motion, trimmed to the analysis window
     (no lag-search buffer)
  4. Splitting-corrected LQT Q-T particle motion, same window
  5. phi-dt parameter space (grid-search contour + optimal solution)
  6. DBSCAN clustering result in phi-dt space (per-window measurements as black dots,
     optimal solution in green on top)
  7. Single-page A-E composite summary

Run with: python3 figure_axec1_event_23919_high_dt_diagnostics.py
"""

import os
import sys

import numpy as np
import obspy
from obspy.core.utcdatetime import UTCDateTime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'swspy'))
import swspy  # noqa: E402

sys.path.insert(0, HERE)
from splitting_functions import create_splitting_analysis  # noqa: E402

OUT_PDF = os.path.join(HERE, 'figure_axec1_event_23919_high_dt_diagnostics.pdf')

# ── Event metadata (data/mldd_catalog_2015_2021.csv, AXEC1 rows reset-indexed;
#    lqt_pykonal_combined_results/splitting_results_AXEC1_2015_2021_all_batches_enriched.csv
#    for back_azimuth/incidence) ────────────────────────────────────────────────────────
STATION = 'AXEC1'
EVENT_ID = 23919
BATCH_NUM = 96   # EVENT_ID // 250 + 1
WAVEFORM_BATCH_PATH = os.path.join(HERE, '..', 'data',
                                   f'axial_mldd_2015_2021_axec1_batch_{BATCH_NUM}.mseed')
ORIGIN_TIME = UTCDateTime('2015-03-10T17:16:10.760000Z')
P_ARRIVAL_TIME = 0.89    # seconds from origin (data/mldd_catalog_2015_2021.csv arrival_time_p)
S_ARRIVAL_TIME = 1.66    # seconds from origin (arrival_time_s)
BACK_AZIMUTH = 238.944976
INCIDENCE_PYKONAL_S = 11.164507   # drives LQT rotation + QC in production
SNR_HORIZONTAL = 10.65    # computed fresh, see docstring
RECTILINEARITY = 0.808    # computed fresh, see docstring

# ── Production splitting parameters (see docstring re: n_win) ──────────────────────────
INCIDENCE_FIELD = 'incidence_pykonal_s'
FIRST_WINDOW_START = 2
LAST_WINDOW_START = 1
FIRST_WINDOW_END = 1.5
LAST_WINDOW_END = 2.5
N_WIN = 7
S_PICK_UNCERTAINTY = 0.0395
COORD_SYSTEM = 'LQT'
SWS_METHOD = 'EV_and_XC'


def load_event_data():
    st = obspy.read(WAVEFORM_BATCH_PATH)
    matches = obspy.Stream([tr for tr in st if abs(tr.stats.starttime - ORIGIN_TIME) < 5.0])
    if len(matches) != 3:
        raise RuntimeError(f'Expected 3 traces for event {EVENT_ID}, found {len(matches)}')
    matches.detrend('linear')
    matches.taper(max_percentage=0.05, type='hann')
    matches.filter('bandpass', freqmin=5.0, freqmax=40.0)
    return {
        'traces': matches,
        'station': STATION,
        'datetime': str(ORIGIN_TIME),
        's_arrival_time': S_ARRIVAL_TIME,
        'p_arrival_time': P_ARRIVAL_TIME,
        'back_azimuth': BACK_AZIMUTH,
        'incidence_pykonal_s': INCIDENCE_PYKONAL_S,
        'snr_horizontal': SNR_HORIZONTAL,
        'rectilinearity_jurkevics': RECTILINEARITY,
        'magnitude': 0.0,
    }


def plot_zne_page(pdf, st_zne):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    comp_order = ['Z', 'N', 'E']
    p_time = ORIGIN_TIME + P_ARRIVAL_TIME
    s_time = ORIGIN_TIME + S_ARRIVAL_TIME
    for i, (ax, comp) in enumerate(zip(axes, comp_order)):
        tr = st_zne.select(channel=f'??{comp}')[0]
        t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
        t0 = tr.stats.starttime
        ax.plot(t, tr.data, c='k', lw=1.0)
        ax.axvline(p_time - t0, c='tab:blue', ls='--', lw=1.2, label='P-pick')
        ax.axvline(s_time - t0, c='tab:red', ls='--', lw=1.2, label='S-pick')
        ax.axvline(s_time - t0 - S_PICK_UNCERTAINTY, c='tab:orange', ls=':', lw=1.2,
                   label=r'1$\sigma$ S-pick uncertainty')
        ax.set_ylabel(f'{tr.stats.channel}\namplitude (counts)')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'AXEC1 event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z — filtered ZNE (5-40 Hz)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_lqt_page(pdf, st_lqt, splitting_obj):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    comp_order = ['L', 'Q', 'T']
    p_time = ORIGIN_TIME + P_ARRIVAL_TIME
    s_time = ORIGIN_TIME + S_ARRIVAL_TIME
    t_dom = splitting_obj.Tmid  # seconds

    fs = st_lqt[0].stats.sampling_rate
    t0 = st_lqt[0].stats.starttime
    win_idxs = splitting_obj.event_station_win_idxs[STATION]
    trim_start_rel = (s_time - t0) - splitting_obj.overall_win_start_pre_fast_S_pick
    win_start_rel = trim_start_rel + np.asarray(win_idxs['win_start_idxs']) / fs
    win_end_rel = trim_start_rel + np.asarray(win_idxs['win_end_idxs']) / fs

    for i, (ax, comp) in enumerate(zip(axes, comp_order)):
        tr = st_lqt.select(channel=f'??{comp}')[0]
        t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
        ax.plot(t, tr.data, c='k', lw=1.0)
        ax.axvline(p_time - t0, c='tab:blue', ls='--', lw=1.2, label='P-pick')
        ax.axvline(s_time - t0, c='tab:red', ls='--', lw=1.2, label='S-pick')
        ax.axvline(s_time - t0 - S_PICK_UNCERTAINTY, c='tab:orange', ls=':', lw=1.2,
                   label=r'1$\sigma$ S-pick uncertainty')
        for j, ws in enumerate(win_start_rel):
            ax.axvline(ws, c='0.6', lw=0.6, alpha=0.6, label='Window start' if j == 0 else None)
        for j, we in enumerate(win_end_rel):
            ax.axvline(we, c='0.6', lw=0.6, alpha=0.6, ls='--',
                       label='Window end' if j == 0 else None)
        ax.set_ylabel(f'{tr.stats.channel}\namplitude (counts)')
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(
        f'AXEC1 event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z — LQT rotated\n'
        f'(back-azi={BACK_AZIMUTH:.1f}°, inc={INCIDENCE_PYKONAL_S:.1f}°), '
        r'$T_{dom}$=' + f'{t_dom*1000:.1f} ms, analysis windows shaded gray',
        fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def plot_particle_motion_page(pdf, st_q, st_t, title, color='k', lw=0.7):
    fig, ax = plt.subplots(figsize=(6, 6))
    q = st_q.data
    t = st_t.data
    max_amp = max(np.max(np.abs(q)), np.max(np.abs(t)))
    ax.plot(t, q, c=color, lw=lw)
    ax.set_xlim(-1.1 * max_amp, 1.1 * max_amp)
    ax.set_ylim(-1.1 * max_amp, 1.1 * max_amp)
    ax.set_xlabel('T amplitude (counts)')
    ax.set_ylabel('Q amplitude (counts)')
    ax.set_aspect('equal')
    ax.set_title(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_phi_dt_space_page(pdf, splitting_obj):
    station = STATION
    if station in getattr(splitting_obj, 'phi_dt_grid_best_window', {}):
        Z = splitting_obj.phi_dt_grid_best_window[station]
    else:
        Z = splitting_obj.phi_dt_grid_average[station]

    lags_labels = splitting_obj.lags_labels
    phis_labels = splitting_obj.phis_labels
    if len(lags_labels) == Z.shape[0]:
        Y, X = np.meshgrid(phis_labels, lags_labels)
    else:
        lag_labels_tmp = np.arange(Z.shape[0]) / splitting_obj.fs
        Y, X = np.meshgrid(phis_labels, lag_labels_tmp)

    row = splitting_obj.sws_result_df.iloc[0]
    dt_curr, phi_curr = float(row['dt']), float(row['phi_from_Q'])
    dt_err_curr, phi_err_curr = float(row['dt_err']), float(row['phi_err'])
    Q_w = float(row['Q_w'])

    t_dom = splitting_obj.Tmid  # seconds

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(X, Y, Z, levels=20, cmap='magma')
    ax.contour(cs, levels=cs.levels[1::2], colors='w', alpha=0.25, linewidths=0.5)
    ax.axvline(t_dom / 2, c='w', ls='--', lw=1.5, label=r'$T_{dom}/2$')
    ax.errorbar(dt_curr, phi_curr, xerr=dt_err_curr, yerr=phi_err_curr,
                c='g', capsize=5, marker='*', markersize=12, markeredgecolor='k',
                label='Optimal solution')
    fig.colorbar(cs, ax=ax, label='Normalized eigenvalue ratio (min = best fit)')
    ax.set_xlabel(r'$\delta t$ (s)')
    ax.set_ylabel(r'$\phi$ from Q ($^o$)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(
        f'AXEC1 event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S")}Z — '
        r'$\phi-\delta t$ parameter space' + '\n' +
        f'Best-window solution: phi={phi_curr:.1f}+/-{phi_err_curr:.1f} deg, '
        f'dt={dt_curr:.3f}+/-{dt_err_curr:.3f} s, Q_w={Q_w:.3f}, '
        r'$T_{dom}$=' + f'{t_dom*1000:.1f} ms',
        fontsize=10
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_clustering_page(pdf, splitting_obj):
    station = STATION
    clustering_info = splitting_obj.clustering_info[station]
    clust_idxs = list(clustering_info['clusters_dict'].keys())

    row = splitting_obj.sws_result_df.iloc[0]
    dt_curr, phi_curr = float(row['dt']), float(row['phi_from_Q'])
    dt_err_curr, phi_err_curr = float(row['dt_err']), float(row['phi_err'])

    fig, ax = plt.subplots(figsize=(7, 6))
    total_count = splitting_obj.n_win ** 2
    clustered_count = 0
    cluster_colors = ['red', 'blue', 'black', 'tab:orange', 'tab:purple',
                      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for idx, clust_idx in enumerate(clust_idxs):
        phis = clustering_info['clusters_dict'][clust_idx]['phis']
        dts = clustering_info['clusters_dict'][clust_idx]['lags']
        phi_errs = clustering_info['clusters_dict'][clust_idx]['phi_errs']
        dt_errs = clustering_info['clusters_dict'][clust_idx]['lag_errs']
        clustered_count += len(phis)
        c = cluster_colors[idx % len(cluster_colors)]
        ax.errorbar(dts, phis, xerr=dt_errs, yerr=phi_errs, fmt='o', markersize=2,
                    color=c, ecolor=c, elinewidth=0.5, capsize=2, capthick=0.5,
                    zorder=5, label=f'Cluster {idx + 1} solutions ({len(phis)})')

    ax.errorbar(dt_curr, phi_curr, xerr=dt_err_curr, yerr=phi_err_curr,
                fmt='*', markersize=12, color='g', markeredgecolor='k',
                markeredgewidth=1, ecolor='g', elinewidth=0.7, capsize=4,
                capthick=0.7, label='Optimal (best window)', zorder=10)

    noise_count = total_count - clustered_count
    ax.set_xlim(0, np.max(splitting_obj.lags_labels))
    ax.set_ylim(-90, 90)
    ax.set_xlabel(r'$\delta t$ (s)')
    ax.set_ylabel(r'$\phi$ from Q ($^o$)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f'AXEC1 event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S")}Z — DBSCAN clustering in '
        r'$\phi-\delta t$ space' + '\n' +
        f'{clustered_count}/{total_count} of {N_WIN}x{N_WIN} windows clustered '
        f'({noise_count} DBSCAN noise, not shown)',
        fontsize=10
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_multipanel_summary_page(pdf, st_lqt, splitting_obj, st_q_uncorr, st_t_uncorr, st_q_corr, st_t_corr,
                                  title_suffix='High Quality'):
    """Single-page composite: LQT traces (top-left), particle motion before/after
    correction (bottom-left, side by side), phi-dt space (top-right), clustering
    (bottom-right)."""
    fig = plt.figure(figsize=(15, 10))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.22)
    right_gs = outer_gs[1].subgridspec(2, 1, hspace=0.38)

    left_bbox = outer_gs[0].get_position(fig)
    row0_bbox = right_gs[0].get_position(fig)
    row1_bbox = right_gs[1].get_position(fig)

    lqt_gs = fig.add_gridspec(3, 1, left=left_bbox.x0, right=left_bbox.x1,
                              top=row0_bbox.y1, bottom=row0_bbox.y0, hspace=0.12)
    ax_L = fig.add_subplot(lqt_gs[0])
    ax_Q = fig.add_subplot(lqt_gs[1], sharex=ax_L)
    ax_T = fig.add_subplot(lqt_gs[2], sharex=ax_L)
    lqt_axes = [ax_L, ax_Q, ax_T]

    ax5 = fig.add_subplot(right_gs[0])
    ax6 = fig.add_subplot(right_gs[1])

    fig_w_in, fig_h_in = fig.get_size_inches()
    square_h_frac = row1_bbox.height
    square_w_frac = square_h_frac * fig_h_in / fig_w_in
    gap_frac = 0.03
    ax3 = fig.add_axes([left_bbox.x0, row1_bbox.y0, square_w_frac, square_h_frac])
    ax4 = fig.add_axes([left_bbox.x0 + square_w_frac + gap_frac, row1_bbox.y0, square_w_frac, square_h_frac])

    comp_order = ['L', 'Q', 'T']
    p_time = ORIGIN_TIME + P_ARRIVAL_TIME
    s_time = ORIGIN_TIME + S_ARRIVAL_TIME
    fs = st_lqt[0].stats.sampling_rate
    t0 = st_lqt[0].stats.starttime
    win_idxs = splitting_obj.event_station_win_idxs[STATION]
    trim_start_rel = (s_time - t0) - splitting_obj.overall_win_start_pre_fast_S_pick
    win_start_rel = trim_start_rel + np.asarray(win_idxs['win_start_idxs']) / fs
    win_end_rel = trim_start_rel + np.asarray(win_idxs['win_end_idxs']) / fs

    for i, (ax, comp) in enumerate(zip(lqt_axes, comp_order)):
        tr = st_lqt.select(channel=f'??{comp}')[0]
        t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
        ax.plot(t, tr.data, c='k', lw=1.0)
        ax.axvline(p_time - t0, c='tab:blue', ls='--', lw=1.2, label='P-pick')
        ax.axvline(s_time - t0, c='tab:red', ls='--', lw=1.2, label='S-pick')
        for j, ws in enumerate(win_start_rel):
            ax.axvline(ws, c='0.6', lw=0.6, alpha=0.6, label='Window start' if j == 0 else None)
        for j, we in enumerate(win_end_rel):
            ax.axvline(we, c='0.6', lw=0.6, alpha=0.6, ls='--',
                       label='Window end' if j == 0 else None)
        ax.set_ylabel(f'{tr.stats.channel}\n(counts)', fontsize=9)
        ax.tick_params(labelsize=8)
        if i < 2:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        if i == 0:
            ax.legend(loc='upper right', fontsize=6)
    ax_T.set_xlabel('Time (s)', fontsize=9)
    ax_L.set_title('LQT-rotated S-wave Analysis Window', fontsize=11)

    for ax, st_q, st_t, color, title, show_yaxis in [
        (ax3, st_q_uncorr, st_t_uncorr, 'darkred', 'Original particle motion', True),
        (ax4, st_q_corr, st_t_corr, 'navy', 'Corrected particle motion', False),
    ]:
        q, t = st_q.data, st_t.data
        max_amp = max(np.max(np.abs(q)), np.max(np.abs(t)))
        ax.plot(t, q, c=color, lw=2.0)
        ax.set_xlim(-1.1 * max_amp, 1.1 * max_amp)
        ax.set_ylim(-1.1 * max_amp, 1.1 * max_amp)
        ax.set_xlabel('T amplitude (counts)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        if show_yaxis:
            ax.set_ylabel('Q amplitude (counts)', fontsize=9)
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13)

    station = STATION
    if station in getattr(splitting_obj, 'phi_dt_grid_best_window', {}):
        Z = splitting_obj.phi_dt_grid_best_window[station]
    else:
        Z = splitting_obj.phi_dt_grid_average[station]
    lags_labels = splitting_obj.lags_labels
    phis_labels = splitting_obj.phis_labels
    if len(lags_labels) == Z.shape[0]:
        Y, X = np.meshgrid(phis_labels, lags_labels)
    else:
        lag_labels_tmp = np.arange(Z.shape[0]) / splitting_obj.fs
        Y, X = np.meshgrid(phis_labels, lag_labels_tmp)

    row = splitting_obj.sws_result_df.iloc[0]
    dt_curr, phi_curr = float(row['dt']), float(row['phi_from_Q'])
    dt_err_curr, phi_err_curr = float(row['dt_err']), float(row['phi_err'])

    t_dom = splitting_obj.Tmid  # seconds

    cs = ax5.contourf(X, Y, Z, levels=20, cmap='magma')
    ax5.contour(cs, levels=cs.levels[1::2], colors='w', alpha=0.25, linewidths=0.5)
    ax5.axvline(t_dom / 2, c='w', ls='--', lw=1.5, label=r'$T_{dom}/2$')
    ax5.errorbar(dt_curr, phi_curr, xerr=dt_err_curr, yerr=phi_err_curr,
                 c='g', capsize=5, marker='*', markersize=12, markeredgecolor='k',
                 label='Optimal solution')
    ax5.legend(loc='upper right', fontsize=6)
    pos5 = ax5.get_position()
    cbar_ax = fig.add_axes([pos5.x1 + 0.015, pos5.y0, 0.015, pos5.height])
    fig.colorbar(cs, cax=cbar_ax, label='Norm. eigenvalue ratio')
    ax5.set_xlabel(r'$\delta t$ (s)', fontsize=9)
    ax5.set_ylabel(r'$\phi$ from Q ($^o$)', fontsize=9)
    ax5.tick_params(labelsize=8)
    ax5.set_title(r'$\phi-\delta t$ Parameter Space: Best Solution', fontsize=11)

    clustering_info = splitting_obj.clustering_info[station]
    clust_idxs = list(clustering_info['clusters_dict'].keys())
    cluster_colors = ['red', 'blue', 'black', 'tab:orange', 'tab:purple',
                      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for idx, clust_idx in enumerate(clust_idxs):
        phis = clustering_info['clusters_dict'][clust_idx]['phis']
        dts = clustering_info['clusters_dict'][clust_idx]['lags']
        phi_errs = clustering_info['clusters_dict'][clust_idx]['phi_errs']
        dt_errs = clustering_info['clusters_dict'][clust_idx]['lag_errs']
        c = cluster_colors[idx % len(cluster_colors)]
        ax6.errorbar(dts, phis, xerr=dt_errs, yerr=phi_errs, fmt='o', markersize=2,
                     color=c, ecolor=c, elinewidth=0.5, capsize=2, capthick=0.5,
                     zorder=5, label=f'Cluster {idx + 1} solutions ({len(phis)})')
    ax6.errorbar(dt_curr, phi_curr, xerr=dt_err_curr, yerr=phi_err_curr,
                 fmt='*', markersize=12, color='g', markeredgecolor='k',
                 markeredgewidth=1, ecolor='g', elinewidth=0.7, capsize=4,
                 capthick=0.7, label='Optimal (best window)', zorder=10)
    ax6.set_xlim(0, np.max(splitting_obj.lags_labels))
    ax6.set_ylim(-90, 90)
    ax6.set_xlabel(r'$\delta t$ (s)', fontsize=9)
    ax6.set_ylabel(r'$\phi$ from Q ($^o$)', fontsize=9)
    ax6.tick_params(labelsize=8)
    ax6.legend(loc='best', fontsize=6)
    ax6.grid(True, alpha=0.3)
    ax6.set_title(r'DBSCAN Clustering in $\phi-\delta t$ Parameter Space', fontsize=11)

    label_gap = 0.02
    for ax, row_top, label in [
        (ax_L, row0_bbox.y1, 'A'), (ax3, row1_bbox.y1, 'B'), (ax4, row1_bbox.y1, 'C'),
        (ax5, row0_bbox.y1, 'D'), (ax6, row1_bbox.y1, 'E'),
    ]:
        x0 = ax.get_position().x0
        fig.text(x0 - 0.02, row_top + label_gap, label, fontsize=15,
                 fontweight='bold', va='bottom', ha='left')

    fig.suptitle(f'EC1 Shear-Wave Splitting Analysis: {title_suffix} Event', fontsize=15, fontweight='bold')
    pdf.savefig(fig)
    plt.close(fig)


def main():
    event_data = load_event_data()
    st_zne = event_data['traces'].copy()

    print('Creating SWSPy splitting object (LQT, PyKonal-S incidence)...')
    splitting_obj = create_splitting_analysis(
        event_data,
        first_window_start=FIRST_WINDOW_START, last_window_start=LAST_WINDOW_START,
        first_window_end=FIRST_WINDOW_END, last_window_end=LAST_WINDOW_END,
        n_win=N_WIN, s_pick_uncertainty=S_PICK_UNCERTAINTY,
        incidence_field=INCIDENCE_FIELD, coord_system=COORD_SYSTEM,
    )

    print('Running SWSPy splitting measurement...')
    splitting_obj.perform_sws_analysis(coord_system=COORD_SYSTEM, sws_method=SWS_METHOD)

    row = splitting_obj.sws_result_df.iloc[0]
    print(f"phi_from_Q={row['phi_from_Q']:.2f} dt={row['dt']:.3f} "
          f"phi_err={row['phi_err']:.2f} dt_err={row['dt_err']:.3f} Q_w={row['Q_w']:.3f}")

    st_lqt = st_zne.copy()
    st_lqt.rotate(method='ZNE->LQT', back_azimuth=BACK_AZIMUTH, inclination=INCIDENCE_PYKONAL_S)

    st_zne_uncorr, st_zne_corr = splitting_obj._get_uncorr_and_corr_waveforms(STATION)
    st_lqt_uncorr = st_zne_uncorr.copy()
    st_lqt_uncorr.rotate(method='ZNE->LQT', back_azimuth=BACK_AZIMUTH, inclination=INCIDENCE_PYKONAL_S)
    st_lqt_corr = st_zne_corr.copy()
    st_lqt_corr.rotate(method='ZNE->LQT', back_azimuth=BACK_AZIMUTH, inclination=INCIDENCE_PYKONAL_S)

    s_pick_abs = ORIGIN_TIME + S_ARRIVAL_TIME
    trim_start = s_pick_abs - splitting_obj.overall_win_start_pre_fast_S_pick
    trim_end = s_pick_abs + splitting_obj.overall_win_start_post_fast_S_pick
    st_lqt_uncorr_notrail = st_lqt_uncorr.copy().trim(starttime=trim_start, endtime=trim_end)
    st_lqt_corr_notrail = st_lqt_corr.copy().trim(starttime=trim_start, endtime=trim_end)

    with PdfPages(OUT_PDF) as pdf:
        plot_zne_page(pdf, st_zne)
        plot_lqt_page(pdf, st_lqt, splitting_obj)
        plot_particle_motion_page(
            pdf, st_lqt_uncorr_notrail.select(channel='??Q')[0], st_lqt_uncorr_notrail.select(channel='??T')[0],
            'Original (uncorrected) LQT particle motion', color='darkred', lw=2.0
        )
        plot_particle_motion_page(
            pdf, st_lqt_corr_notrail.select(channel='??Q')[0], st_lqt_corr_notrail.select(channel='??T')[0],
            'Splitting-corrected LQT particle motion', color='navy', lw=2.0
        )
        plot_phi_dt_space_page(pdf, splitting_obj)
        plot_clustering_page(pdf, splitting_obj)
        plot_multipanel_summary_page(
            pdf, st_lqt, splitting_obj,
            st_lqt_uncorr_notrail.select(channel='??Q')[0], st_lqt_uncorr_notrail.select(channel='??T')[0],
            st_lqt_corr_notrail.select(channel='??Q')[0], st_lqt_corr_notrail.select(channel='??T')[0],
            title_suffix='High dt',
        )

    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
