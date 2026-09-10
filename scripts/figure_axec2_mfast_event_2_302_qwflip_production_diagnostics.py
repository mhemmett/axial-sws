"""
figure_axec2_mfast_event_2_302_qwflip_production_diagnostics.py

Diagnostic multi-page PDF for AXEC2 event_id 2_302 (batch 2), run through UNPATCHED production
swspy.splitting.split (split.py). Companion to
figure_axec2_mfast_event_2_302_qwflip_windowcheck_diagnostics.py, which runs the SAME event
through the 2026-07-24 windowing fix (split_windowcheck.py).

Event selected from the AXEC2 10-batch test (batches 1-10,
production_axec2_mfast_filters_maxdt02_lqt_pykonal_results/ vs.
..._windowcheck_lqt_pykonal_results/) as a "high-quality, affected-by-the-fix" example: quality/
success/tier-5 QC all pass under BOTH production and the window-check fix, and the reported
best-fit phi/dt are IDENTICAL between the two runs (same grid-search optimum found), but the
reported Q_w quality metric is NOT:
    Production   (split.py, THIS script):            phi=-82.36 deg, dt=0.020 s, phi_err=3 deg,
                                                       dt_err=0.005 s, Q_w = -0.953
                                                       (fails a Q_w>=0.5 QC gate)
    Window-check (split_windowcheck.py, companion):   phi=-82.36 deg, dt=0.020 s, phi_err=3 deg,
                                                       dt_err=0.005 s, Q_w = +1.000
                                                       (passes cleanly)
This is exactly the failure mode the fix targets: T_dom=0.0756s (13.2 Hz) puts this event's
shortest candidate window well below max_t_shift_s=0.2s, so under production the grid search's
cyclic np.roll lag-shift (_phi_dt_grid_search) wraps short windows around on themselves. The
*single* best window still happens to land on the correct answer (phi/dt match), but the
corrupted neighboring windows destabilize the DBSCAN clustering variance that Q_w is computed
from -- silently flipping a genuinely well-constrained measurement from "passes QC" to "fails
QC" (or vice versa), with no crash and no visible difference in phi/dt alone. Compare PAGE 5
(phi-dt parameter space, look for a periodic/comb-like wraparound artifact) and the clustering
page directly against the window-check companion script's output for this same event.

Selection: searched production_axec2_mfast_filters_maxdt02_lqt_pykonal_results/ batches 1-10 for
tier-5-passing events in the window-check run (quality>=0.9, dt<T_dom/2, phi_error<10 deg,
dt_error<0.05s) with dominant_period < 0.107s (the max_t_shift_s=0.2s-derived threshold below
which the fix's window-widening branch fires: 1.5*T_dom + 0.0395 < 0.2), then joined to the
paired production result for the same event_id. event_id 2_302 has the largest swing in Q_w
of any candidate with identical phi/dt between the two runs (-0.953 -> +1.000).

Event: AXEC2, event_id 2_302 (batch 2), origin 2015-01-25T16:37:08.063Z (pre-eruption).
Dominant period T_dom = 0.0756s (~13.2 Hz), chosen filter band (try_filters): 10-20 Hz.
snr_horizontal=8.17, rectilinearity_jurkevics=0.977, incidence_pykonal_s=18.37 deg.

Pages: same 9-page structure as figure_axec2_mfast_event_74456_hq_diagnostics.py (ZNE, LQT,
particle motion before/after, phi-dt parameter space [PAGE 5 -- inspect for wraparound/roll-over
signature here], DBSCAN clustering, ZNE composite summary, cluster-stability + Q/T overlay,
fast/slow alignment). Additionally prints the winning window's length in seconds and its ratio
to both T_dom and max_t_shift_s, for direct comparison against the window-check twin script.

Run with: python3 figure_axec2_mfast_event_2_302_qwflip_production_diagnostics.py
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

OUT_PDF = os.path.join(HERE, 'figure_axec2_mfast_event_2_302_qwflip_production_diagnostics.pdf')

# ── Event metadata (raw_axec2_all_batches_mfast_filters_data metadata + splitting results,
#    event_id 2_302) ────────────────────────────────────────────────────────────────────────
STATION = 'AXEC2'
WAVEFORM_PATH = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data', 'waveforms',
                              'batch2_event_2015-01-25T16-37-08.063000Z.mseed')
ORIGIN_TIME = UTCDateTime('2015-01-25T16:37:08.063000Z')
S_ARRIVAL_TIME = 1.037   # seconds from origin
P_ARRIVAL_TIME = 0.527   # seconds from origin
BACK_AZIMUTH = 301.63649
INCIDENCE_PYKONAL_S = 18.374938   # drives LQT rotation + QC in production
SNR_HORIZONTAL = 8.174989
RECTILINEARITY = 0.977328
CHOSEN_FILTER_BAND = (10.0, 20.0)   # try_filters optimal band for this event

# ── Production splitting parameters (run_axec2_mfast_filters_maxdt02_batches.py) ────────────
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
    st = obspy.read(WAVEFORM_PATH)
    return {
        'traces': st,
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
    fig.suptitle(f'AXEC2 (mfast) event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z - '
                 f'filtered ZNE ({CHOSEN_FILTER_BAND[0]:.0f}-{CHOSEN_FILTER_BAND[1]:.0f} Hz, try_filters)')
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
        f'AXEC2 (mfast) event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z - LQT rotated\n'
        f'(back-azi={BACK_AZIMUTH:.1f} deg, inc={INCIDENCE_PYKONAL_S:.1f} deg), '
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
        f'AXEC2 (mfast) event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S")}Z - '
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
        f'AXEC2 (mfast) event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S")}Z - DBSCAN clustering in '
        r'$\phi-\delta t$ space' + '\n' +
        f'{clustered_count}/{total_count} of {N_WIN}x{N_WIN} windows clustered '
        f'({noise_count} DBSCAN noise, not shown)',
        fontsize=10
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_multipanel_summary_page(pdf, st_zne, splitting_obj, st_q_uncorr, st_t_uncorr, st_q_corr, st_t_corr,
                                  title_suffix='High quality'):
    fig = plt.figure(figsize=(15, 10))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.22)
    right_gs = outer_gs[1].subgridspec(2, 1, hspace=0.38)

    left_bbox = outer_gs[0].get_position(fig)
    row0_bbox = right_gs[0].get_position(fig)
    row1_bbox = right_gs[1].get_position(fig)

    zne_gs = fig.add_gridspec(3, 1, left=left_bbox.x0, right=left_bbox.x1,
                              top=row0_bbox.y1, bottom=row0_bbox.y0, hspace=0.12)
    ax_Z = fig.add_subplot(zne_gs[0])
    ax_N = fig.add_subplot(zne_gs[1], sharex=ax_Z)
    ax_E = fig.add_subplot(zne_gs[2], sharex=ax_Z)
    zne_axes = [ax_Z, ax_N, ax_E]

    ax5 = fig.add_subplot(right_gs[0])
    ax6 = fig.add_subplot(right_gs[1])

    fig_w_in, fig_h_in = fig.get_size_inches()
    square_h_frac = row1_bbox.height
    square_w_frac = square_h_frac * fig_h_in / fig_w_in
    gap_frac = 0.03
    ax3 = fig.add_axes([left_bbox.x0, row1_bbox.y0, square_w_frac, square_h_frac])
    ax4 = fig.add_axes([left_bbox.x0 + square_w_frac + gap_frac, row1_bbox.y0, square_w_frac, square_h_frac])

    comp_order = ['Z', 'N', 'E']
    p_time = ORIGIN_TIME + P_ARRIVAL_TIME
    s_time = ORIGIN_TIME + S_ARRIVAL_TIME
    fs = st_zne[0].stats.sampling_rate
    t0 = st_zne[0].stats.starttime
    win_idxs = splitting_obj.event_station_win_idxs[STATION]
    trim_start_rel = (s_time - t0) - splitting_obj.overall_win_start_pre_fast_S_pick
    win_start_rel = trim_start_rel + np.asarray(win_idxs['win_start_idxs']) / fs
    win_end_rel = trim_start_rel + np.asarray(win_idxs['win_end_idxs']) / fs

    for i, (ax, comp) in enumerate(zip(zne_axes, comp_order)):
        tr = st_zne.select(channel=f'??{comp}')[0]
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
    ax_E.set_xlabel('Time (s)', fontsize=9)
    ax_Z.set_title('ZNE S-wave Analysis Window', fontsize=11)

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
        (ax_Z, row0_bbox.y1, 'A'), (ax3, row1_bbox.y1, 'B'), (ax4, row1_bbox.y1, 'C'),
        (ax5, row0_bbox.y1, 'D'), (ax6, row1_bbox.y1, 'E'),
    ]:
        x0 = ax.get_position().x0
        fig.text(x0 - 0.02, row_top + label_gap, label, fontsize=15,
                 fontweight='bold', va='bottom', ha='left')

    fig.suptitle(f'AXEC2 (mfast) Shear-Wave Splitting Analysis: {title_suffix} Event', fontsize=15, fontweight='bold')
    pdf.savefig(fig)
    plt.close(fig)


def plot_stability_and_waveforms_page(pdf, splitting_obj, st_q_uncorr, st_t_uncorr, st_q_corr, st_t_corr):
    """Extra diagnostics page: (top row) stability of the winning DBSCAN cluster's per-window
    solutions -- phi and dt (each with their Silver & Chan 1991 error bar) plotted against
    window number (the flattened start/end-window-pair index used by swspy's clustering) --
    plus (bottom two rows) the Q and T wavelets themselves (not a particle-motion cross-plot),
    initial vs. corrected OVERLAID on the same axes so the correction is visible at a glance
    (with independent per-panel y-scaling, a small dt correction can otherwise look like "no
    change" even though the underlying samples differ substantially)."""
    station = STATION
    min_var_idx = splitting_obj.clustering_info[station]['min_var_idx']
    winning_cluster = splitting_obj.clustering_info[station]['clusters_dict'][str(min_var_idx)]
    order = np.argsort(winning_cluster['sample_idxs'])
    win_num = winning_cluster['sample_idxs'][order]
    phis = winning_cluster['phis'][order]
    phi_errs = winning_cluster['phi_errs'][order]
    lags = winning_cluster['lags'][order]
    lag_errs = winning_cluster['lag_errs'][order]

    row = splitting_obj.sws_result_df.iloc[0]
    dt_curr, phi_curr = float(row['dt']), float(row['phi_from_Q'])

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.45, wspace=0.28)

    ax_phi = fig.add_subplot(gs[0, 0])
    ax_dt = fig.add_subplot(gs[0, 1])

    ax_phi.errorbar(win_num, phis, yerr=phi_errs, fmt='o', ms=4, color='tab:blue',
                    ecolor='tab:blue', elinewidth=0.8, capsize=3, zorder=5)
    ax_phi.axhline(phi_curr, c='g', ls='--', lw=1.2, label='Optimal solution')
    ax_phi.set_xlabel('Window number')
    ax_phi.set_ylabel(r'$\phi$ from Q ($^o$)')
    ax_phi.set_title(f'Cluster stability: $\\phi$ per window ({len(win_num)} windows)', fontsize=10)
    ax_phi.legend(loc='best', fontsize=7)
    ax_phi.grid(alpha=0.3)

    ax_dt.errorbar(win_num, lags, yerr=lag_errs, fmt='o', ms=4, color='tab:green',
                   ecolor='tab:green', elinewidth=0.8, capsize=3, zorder=5)
    ax_dt.axhline(dt_curr, c='g', ls='--', lw=1.2, label='Optimal solution')
    ax_dt.set_xlabel('Window number')
    ax_dt.set_ylabel(r'$\delta t$ (s)')
    ax_dt.set_title(f'Cluster stability: $\\delta t$ per window ({len(win_num)} windows)', fontsize=10)
    ax_dt.legend(loc='best', fontsize=7)
    ax_dt.grid(alpha=0.3)

    # Q and T, initial vs. corrected OVERLAID on the same axes (shared scale) -- rather than 4
    # separately-auto-scaled panels -- so the correction (a small dt=few-sample shift here) is
    # actually visible instead of being hidden by independent y-axis rescaling per panel.
    ax_q = fig.add_subplot(gs[1, :])
    ax_t = fig.add_subplot(gs[2, :], sharex=ax_q, sharey=ax_q)

    # Same y-scale for Q and T (not just initial-vs-corrected within a panel) so relative
    # amplitudes between the two components are directly comparable too.
    y_max = max(
        np.max(np.abs(st_q_uncorr.data)), np.max(np.abs(st_q_corr.data)),
        np.max(np.abs(st_t_uncorr.data)), np.max(np.abs(st_t_corr.data)),
    ) * 1.1

    for ax, tr_uncorr, tr_corr, comp in [(ax_q, st_q_uncorr, st_q_corr, 'Q'), (ax_t, st_t_uncorr, st_t_corr, 'T')]:
        t = np.arange(tr_uncorr.stats.npts) / tr_uncorr.stats.sampling_rate
        ax.plot(t, tr_uncorr.data, c='k', lw=1.3, ls='-', label='Initial')
        ax.plot(t, tr_corr.data, c='red', lw=1.3, ls='--', label='Corrected')
        max_diff = np.max(np.abs(tr_corr.data - tr_uncorr.data))
        ax.set_title(f'{comp} component: initial vs. corrected (max |diff|={max_diff:.2e} counts)',
                     fontsize=10)
        ax.set_ylabel('Amplitude (counts)', fontsize=9)
        ax.set_ylim(-y_max, y_max)
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.legend(loc='upper right', fontsize=8)
    ax_t.set_xlabel('Time (s)', fontsize=9)

    fig.suptitle(
        f'AXEC2 (mfast) event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z - '
        'Cluster stability + Q/T waveforms before/after splitting correction',
        fontsize=13, fontweight='bold'
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_fast_slow_alignment_page(pdf, fast_before, slow_before, fast_after, slow_after, phi_curr, dt_curr):
    """Fast and slow wavelets overlaid on the same seismogram (fast = solid black, slow =
    dashed red), before splitting correction (as recorded, offset by dt) and after (re-aligned
    by the dt/2 forward/backward shift swspy's remove_splitting() applies internally -- see
    that function's docstring: fast is rolled +dt/2, slow is rolled -dt/2, in the phi-rotated
    frame, before rotating back to ZNE)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    t_before = np.arange(fast_before.stats.npts) / fast_before.stats.sampling_rate
    axes[0].plot(t_before, fast_before.data, c='k', lw=1.2, ls='-', label='Fast')
    axes[0].plot(t_before, slow_before.data, c='red', lw=1.2, ls='--', label='Slow')
    axes[0].set_title('Before splitting correction (original fast/slow wavelets)', fontsize=10)
    axes[0].set_ylabel('Amplitude (counts)', fontsize=9)
    axes[0].legend(loc='upper right', fontsize=8)

    t_after = np.arange(fast_after.stats.npts) / fast_after.stats.sampling_rate
    axes[1].plot(t_after, fast_after.data, c='k', lw=1.2, ls='-', label='Fast (+dt/2)')
    axes[1].plot(t_after, slow_after.data, c='red', lw=1.2, ls='--', label='Slow (-dt/2)')
    axes[1].set_title('After splitting correction (fast/slow re-aligned)', fontsize=10)
    axes[1].set_ylabel('Amplitude (counts)', fontsize=9)
    axes[1].set_xlabel('Time (s)', fontsize=9)
    axes[1].legend(loc='upper right', fontsize=8)

    fig.suptitle(
        f'AXEC2 (mfast) event {ORIGIN_TIME.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z - '
        f'Fast/slow wavelet alignment (phi={phi_curr:.1f} deg, dt={dt_curr:.3f} s)',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
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
    # IMPORTANT: explicitly match the production pipeline's max_t_shift_s override (0.2s), not
    # swspy's own default of 0.30s -- this is exactly the parameter under investigation. Note
    # this is set AFTER construction (production's own pattern, see
    # splitting_functions.py's perform_splitting_analysis()) -- unlike the window-check fix,
    # unpatched split.py's window layout is already fixed by the time __init__ returns, so this
    # post-hoc override cannot retroactively widen undersized windows the way the fix does.
    splitting_obj.max_t_shift_s = 0.2

    print('Running SWSPy splitting measurement...')
    splitting_obj.perform_sws_analysis(coord_system=COORD_SYSTEM, sws_method=SWS_METHOD)

    row = splitting_obj.sws_result_df.iloc[0]
    print(f"phi_from_Q={row['phi_from_Q']:.2f} dt={row['dt']:.3f} "
          f"phi_err={row['phi_err']:.2f} dt_err={row['dt_err']:.3f} Q_w={row['Q_w']:.3f}")

    # ── Report the winning window's length vs T_dom and max_t_shift_s ──────────────────────
    min_var_idx = splitting_obj.clustering_info[STATION]['min_var_idx']
    winning_cluster = splitting_obj.clustering_info[STATION]['clusters_dict'][str(min_var_idx)]
    cluster_vars_tmp = winning_cluster['lag_errs']**2 + winning_cluster['phi_errs']**2
    opt_idx_in_cluster = int(np.argmin(cluster_vars_tmp))
    opt_obs_global_idx = int(winning_cluster['sample_idxs'][opt_idx_in_cluster])
    win_idxs = splitting_obj.event_station_win_idxs[STATION]
    a = opt_obs_global_idx // N_WIN
    b = opt_obs_global_idx % N_WIN
    win_start_idx = win_idxs['win_start_idxs'][a]
    win_end_idx = win_idxs['win_end_idxs'][b]
    window_length_s = (win_end_idx - win_start_idx) / splitting_obj.fs
    print(f"\nWinning window length: {window_length_s:.4f}s")
    print(f"T_dom (Tmid): {splitting_obj.Tmid:.4f}s  ->  window/T_dom = {window_length_s/splitting_obj.Tmid:.2f}")
    print(f"max_t_shift_s: {splitting_obj.max_t_shift_s}  ->  window/max_t_shift_s = "
          f"{window_length_s/splitting_obj.max_t_shift_s:.2f}\n")

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

    # Fast/slow wavelets: st_zne_corr carries "??F"/"??S" channels from remove_splitting()
    # (return_FS=True default) -- these are the fast/slow components BEFORE the dt shift is
    # applied. Reconstruct the AFTER-shift (re-aligned) pair ourselves with the identical
    # +dt/2 / -dt/2 sample roll remove_splitting() uses internally (see split.py).
    dt_curr = float(row['dt'])
    phi_curr = float(row['phi_from_Q'])
    fast_before_full = st_zne_corr.select(channel='??F')[0].copy()
    slow_before_full = st_zne_corr.select(channel='??S')[0].copy()
    fs_fs = fast_before_full.stats.sampling_rate
    fast_after_full = fast_before_full.copy()
    fast_after_full.data = np.roll(fast_before_full.data, int(round((dt_curr / 2) * fs_fs)))
    slow_after_full = slow_before_full.copy()
    slow_after_full.data = np.roll(slow_before_full.data, -int(round((dt_curr / 2) * fs_fs)))

    fast_before = fast_before_full.copy().trim(starttime=trim_start, endtime=trim_end)
    slow_before = slow_before_full.copy().trim(starttime=trim_start, endtime=trim_end)
    fast_after = fast_after_full.copy().trim(starttime=trim_start, endtime=trim_end)
    slow_after = slow_after_full.copy().trim(starttime=trim_start, endtime=trim_end)

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
            pdf, st_zne, splitting_obj,
            st_lqt_uncorr_notrail.select(channel='??Q')[0], st_lqt_uncorr_notrail.select(channel='??T')[0],
            st_lqt_corr_notrail.select(channel='??Q')[0], st_lqt_corr_notrail.select(channel='??T')[0],
            title_suffix='Q_w Stability Check (Production)',
        )
        plot_stability_and_waveforms_page(
            pdf, splitting_obj,
            st_lqt_uncorr_notrail.select(channel='??Q')[0], st_lqt_uncorr_notrail.select(channel='??T')[0],
            st_lqt_corr_notrail.select(channel='??Q')[0], st_lqt_corr_notrail.select(channel='??T')[0],
        )
        plot_fast_slow_alignment_page(
            pdf, fast_before, slow_before, fast_after, slow_after, phi_curr, dt_curr
        )

    print(f'Wrote {OUT_PDF}')


if __name__ == '__main__':
    main()
