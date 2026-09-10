"""
Diagnostic multi-page PDF generator for individual AXEC2 pre-eruption events, for building
multipanel figures of the production SWS pipeline. One-off script, not part of the pipeline.
Generalized from figure_axec2_event_33223_diagnostics.py to run over a list of events.

All events below pass the same tight QC thresholds (SNR>3.0, incidence_pykonal_s<20 deg,
rectilinearity>0.9, pre-eruption i.e. < 2015-04-24 06:00 UTC, phi_error<20 deg, dt_error<0.04 s,
dt>0), but span different Q_w (Wustefeld 2010) outcomes:
  - event_id 33223: Q_w=1.0 (best-quality, well-constrained fast/slow split)
  - event_id 35382: Q_w=-0.984 (the discretized floor value = confident/"true" null - no splitting)
  - event_id 38195: Q_w=0.0 (ambiguous / neither confidently split nor confidently null)

Pages per event:
  1. Filtered ZNE traces (P-pick, S-pick, 1-sigma S-pick uncertainty)
  2. Rotated LQT traces with P-pick, S-pick, S-pick uncertainty, SWSPy analysis windows
     (start/end)
  3. Original (uncorrected) LQT Q-T particle motion, trimmed to the analysis window
     (no lag-search buffer)
  4. Splitting-corrected LQT Q-T particle motion, same window
  5. phi-dt parameter space (grid-search contour + optimal solution)
  6. DBSCAN clustering result in phi-dt space (per-window measurements as black dots with
     thin error bars, optimal solution in green on top)

Run with: python3 figure_axec2_event_diagnostics.py
"""

import os
import sys
from types import SimpleNamespace

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

STATION = 'AXEC2'
WAVEFORMS_DIR = os.path.join(HERE, 'raw_axec2_all_batches_data', 'waveforms')

# ── Production splitting parameters (scripts/run_production_axec2_all_batches.py) ──────
INCIDENCE_FIELD = 'incidence_pykonal_s'
FIRST_WINDOW_START = 2
LAST_WINDOW_START = 1
FIRST_WINDOW_END = 1.5
LAST_WINDOW_END = 2.5
N_WIN = 7
S_PICK_UNCERTAINTY = 0.0395
COORD_SYSTEM = 'LQT'
SWS_METHOD = 'EV_and_XC'

# ── Events (from scripts/raw_axec2_all_batches_data/raw_axec2_all_batches_metadata.csv) ─
EVENTS = [
    dict(
        label='q1p0', event_id=33223, waveform_file='batch133_event_2015-03-18T00-44-15.475000Z.mseed',
        origin_time='2015-03-18T00:44:15.475000Z', s_arrival_time=1.265, p_arrival_time=0.635,
        back_azimuth=308.089917, incidence_pykonal_s=16.516842, incidence_p_jurkevics=7.788619,
        snr_horizontal=15.6876, rectilinearity=0.981681,
        note='Q_w=1.0 (best-quality, well-constrained split)',
        title_suffix='High Quality',
    ),
    dict(
        label='qnull', event_id=35382, waveform_file='batch142_event_2015-03-19T23-42-40.420000Z.mseed',
        origin_time='2015-03-19T23:42:40.420000Z', s_arrival_time=0.76, p_arrival_time=0.33,
        back_azimuth=234.081334, incidence_pykonal_s=15.707771, incidence_p_jurkevics=7.789301,
        snr_horizontal=14.522696, rectilinearity=0.993688,
        note='Q_w=-0.984 (discretized floor = confident/"true" null)',
        title_suffix='Null Quality',
    ),
    dict(
        label='q0p0', event_id=38195, waveform_file='batch153_event_2015-03-22T04-16-33.906000Z.mseed',
        origin_time='2015-03-22T04:16:33.906000Z', s_arrival_time=1.294, p_arrival_time=0.674,
        back_azimuth=322.71335, incidence_pykonal_s=15.119402, incidence_p_jurkevics=7.054338,
        snr_horizontal=16.162053, rectilinearity=0.979599,
        note='Q_w=0.0 (ambiguous - neither confidently split nor confidently null)',
        title_suffix='Indeterminate Quality',
    ),
]


def load_event_data(ev):
    st = obspy.read(os.path.join(WAVEFORMS_DIR, ev.waveform_file))
    return {
        'traces': st,
        'station': STATION,
        'datetime': str(ev.origin_time),
        's_arrival_time': ev.s_arrival_time,
        'p_arrival_time': ev.p_arrival_time,
        'back_azimuth': ev.back_azimuth,
        'incidence_pykonal_s': ev.incidence_pykonal_s,
        'incidence_eigenvalue_jurkevics': ev.incidence_p_jurkevics,
        'snr_horizontal': ev.snr_horizontal,
        'rectilinearity_jurkevics': ev.rectilinearity,
        'magnitude': 0.0,
    }


def plot_zne_page(pdf, ev, st_zne):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    comp_order = ['Z', 'N', 'E']
    p_time = ev.origin_time + ev.p_arrival_time
    s_time = ev.origin_time + ev.s_arrival_time
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
    fig.suptitle(f'AXEC2 event {ev.origin_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z — filtered ZNE (5-40 Hz)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_lqt_page(pdf, ev, st_lqt, splitting_obj):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    comp_order = ['L', 'Q', 'T']
    p_time = ev.origin_time + ev.p_arrival_time
    s_time = ev.origin_time + ev.s_arrival_time

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
        f'AXEC2 event {ev.origin_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z — LQT rotated\n'
        f'(back-azi={ev.back_azimuth:.1f}°, inc={ev.incidence_pykonal_s:.1f}°), '
        'analysis windows shaded gray',
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


def plot_phi_dt_space_page(pdf, ev, splitting_obj):
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

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(X, Y, Z, levels=20, cmap='magma')
    ax.contour(cs, levels=cs.levels[1::2], colors='w', alpha=0.25, linewidths=0.5)
    ax.errorbar(dt_curr, phi_curr, xerr=dt_err_curr, yerr=phi_err_curr,
                c='g', capsize=5, marker='*', markersize=12, markeredgecolor='k')
    fig.colorbar(cs, ax=ax, label='Normalized eigenvalue ratio (min = best fit)')
    ax.set_xlabel(r'$\delta t$ (s)')
    ax.set_ylabel(r'$\phi$ from Q ($^o$)')
    ax.set_title(
        f'AXEC2 event {ev.origin_time.strftime("%Y-%m-%dT%H:%M:%S")}Z — '
        r'$\phi-\delta t$ parameter space' + '\n' +
        f'Best-window solution: phi={phi_curr:.1f}+/-{phi_err_curr:.1f} deg, '
        f'dt={dt_curr:.3f}+/-{dt_err_curr:.3f} s, Q_w={Q_w:.3f}',
        fontsize=10
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_clustering_page(pdf, ev, splitting_obj):
    station = STATION
    clustering_info = splitting_obj.clustering_info[station]
    clust_idxs = list(clustering_info['clusters_dict'].keys())

    row = splitting_obj.sws_result_df.iloc[0]
    dt_curr, phi_curr = float(row['dt']), float(row['phi_from_Q'])
    dt_err_curr, phi_err_curr = float(row['dt_err']), float(row['phi_err'])

    fig, ax = plt.subplots(figsize=(7, 6))
    total_count = splitting_obj.n_win ** 2
    clustered_count = 0
    for idx, clust_idx in enumerate(clust_idxs):
        phis = clustering_info['clusters_dict'][clust_idx]['phis']
        dts = clustering_info['clusters_dict'][clust_idx]['lags']
        phi_errs = clustering_info['clusters_dict'][clust_idx]['phi_errs']
        dt_errs = clustering_info['clusters_dict'][clust_idx]['lag_errs']
        clustered_count += len(phis)
        ax.errorbar(dts, phis, xerr=dt_errs, yerr=phi_errs, fmt='o', markersize=2,
                    color='k', ecolor='k', elinewidth=0.5, capsize=2, capthick=0.5,
                    zorder=5, label=f'Cluster {clust_idx} solutions' if idx == 0 else None)

    # Best (optimal) solution on top, in the same green as the phi-dt space plot,
    # with thin error-bar lines so it doesn't overwhelm the cluster dots underneath.
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
        f'AXEC2 event {ev.origin_time.strftime("%Y-%m-%dT%H:%M:%S")}Z — DBSCAN clustering in '
        r'$\phi-\delta t$ space' + '\n' +
        f'{clustered_count}/{total_count} of {N_WIN}x{N_WIN} windows clustered '
        f'({noise_count} DBSCAN noise, not shown)',
        fontsize=10
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_multipanel_summary_page(pdf, ev, st_lqt, splitting_obj, st_q_uncorr, st_t_uncorr, st_q_corr, st_t_corr):
    """Single-page composite: LQT traces (top-left), particle motion before/after
    correction (bottom-left, side by side), phi-dt space (top-right), clustering
    (bottom-right)."""
    fig = plt.figure(figsize=(15, 10))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.22)
    right_gs = outer_gs[1].subgridspec(2, 1, hspace=0.38)

    # Row bounding boxes for D (top) and E (bottom), taken before any axes exist -
    # used as the authoritative top/bottom for A and B/C respectively, so the left
    # and right columns are always equal height regardless of aspect constraints.
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

    # Place B/C manually, sized/positioned so their row spans the exact same
    # top/bottom as E's row (right column) - keeps the two columns visually equal
    # height even though B/C are square (aspect='equal') and E isn't.
    fig_w_in, fig_h_in = fig.get_size_inches()
    square_h_frac = row1_bbox.height
    square_w_frac = square_h_frac * fig_h_in / fig_w_in
    gap_frac = 0.03
    ax3 = fig.add_axes([left_bbox.x0, row1_bbox.y0, square_w_frac, square_h_frac])
    ax4 = fig.add_axes([left_bbox.x0 + square_w_frac + gap_frac, row1_bbox.y0, square_w_frac, square_h_frac])

    # ── Panel A: LQT traces + analysis windows ──────────────────────────────────────
    comp_order = ['L', 'Q', 'T']
    p_time = ev.origin_time + ev.p_arrival_time
    s_time = ev.origin_time + ev.s_arrival_time
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

    # ── Panel B/C: particle motion, before/after correction ─────────────────────────
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

    # ── Panel D: phi-dt parameter space ─────────────────────────────────────────────
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

    cs = ax5.contourf(X, Y, Z, levels=20, cmap='magma')
    ax5.contour(cs, levels=cs.levels[1::2], colors='w', alpha=0.25, linewidths=0.5)
    ax5.errorbar(dt_curr, phi_curr, xerr=dt_err_curr, yerr=phi_err_curr,
                 c='g', capsize=5, marker='*', markersize=12, markeredgecolor='k')
    # Colorbar goes in its own axes, positioned off of ax5's bounding box, so it doesn't
    # shrink ax5's plot area relative to ax6 (E) - it's fine for the colorbar to stick out.
    pos5 = ax5.get_position()
    cbar_ax = fig.add_axes([pos5.x1 + 0.015, pos5.y0, 0.015, pos5.height])
    fig.colorbar(cs, cax=cbar_ax, label='Norm. eigenvalue ratio')
    ax5.set_xlabel(r'$\delta t$ (s)', fontsize=9)
    ax5.set_ylabel(r'$\phi$ from Q ($^o$)', fontsize=9)
    ax5.tick_params(labelsize=8)
    ax5.set_title(r'$\phi-\delta t$ Parameter Space: Best Solution', fontsize=11)

    # ── Panel E: DBSCAN clustering in phi-dt space ──────────────────────────────────
    clustering_info = splitting_obj.clustering_info[station]
    clust_idxs = list(clustering_info['clusters_dict'].keys())
    for idx, clust_idx in enumerate(clust_idxs):
        phis = clustering_info['clusters_dict'][clust_idx]['phis']
        dts = clustering_info['clusters_dict'][clust_idx]['lags']
        phi_errs = clustering_info['clusters_dict'][clust_idx]['phi_errs']
        dt_errs = clustering_info['clusters_dict'][clust_idx]['lag_errs']
        ax6.errorbar(dts, phis, xerr=dt_errs, yerr=phi_errs, fmt='o', markersize=2,
                     color='k', ecolor='k', elinewidth=0.5, capsize=2, capthick=0.5,
                     zorder=5, label=f'Cluster {clust_idx} solutions' if idx == 0 else None)
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

    # ── A-E labels, top-left of each subplot ────────────────────────────────────────
    # Use absolute figure coordinates (not each axes' own transAxes) so A lines up
    # with D (same row0_bbox top) and B/C line up with E (same row1_bbox top),
    # regardless of how tall each individual axes is within its row.
    label_gap = 0.02
    for ax, row_top, label in [
        (ax_L, row0_bbox.y1, 'A'), (ax3, row1_bbox.y1, 'B'), (ax4, row1_bbox.y1, 'C'),
        (ax5, row0_bbox.y1, 'D'), (ax6, row1_bbox.y1, 'E'),
    ]:
        x0 = ax.get_position().x0
        fig.text(x0 - 0.02, row_top + label_gap, label, fontsize=15,
                 fontweight='bold', va='bottom', ha='left')

    fig.suptitle(f'EC2 Shear-Wave Splitting Analysis: {ev.title_suffix} Event', fontsize=15, fontweight='bold')
    pdf.savefig(fig)
    plt.close(fig)


def run_event(ev_dict):
    ev = SimpleNamespace(**ev_dict)
    ev.origin_time = UTCDateTime(ev.origin_time)

    print(f'\n=== Event {ev.event_id} ({ev.label}): {ev.note} ===')
    event_data = load_event_data(ev)
    st_zne = event_data['traces'].copy()

    splitting_obj = create_splitting_analysis(
        event_data,
        first_window_start=FIRST_WINDOW_START, last_window_start=LAST_WINDOW_START,
        first_window_end=FIRST_WINDOW_END, last_window_end=LAST_WINDOW_END,
        n_win=N_WIN, s_pick_uncertainty=S_PICK_UNCERTAINTY,
        incidence_field=INCIDENCE_FIELD, coord_system=COORD_SYSTEM,
    )
    splitting_obj.perform_sws_analysis(coord_system=COORD_SYSTEM, sws_method=SWS_METHOD)

    row = splitting_obj.sws_result_df.iloc[0]
    print(f"phi_from_Q={row['phi_from_Q']:.2f} dt={row['dt']:.3f} "
          f"phi_err={row['phi_err']:.2f} dt_err={row['dt_err']:.3f} Q_w={row['Q_w']:.3f}")

    st_lqt = st_zne.copy()
    st_lqt.rotate(method='ZNE->LQT', back_azimuth=ev.back_azimuth, inclination=ev.incidence_pykonal_s)

    st_zne_uncorr, st_zne_corr = splitting_obj._get_uncorr_and_corr_waveforms(STATION)
    st_lqt_uncorr = st_zne_uncorr.copy()
    st_lqt_uncorr.rotate(method='ZNE->LQT', back_azimuth=ev.back_azimuth, inclination=ev.incidence_pykonal_s)
    st_lqt_corr = st_zne_corr.copy()
    st_lqt_corr.rotate(method='ZNE->LQT', back_azimuth=ev.back_azimuth, inclination=ev.incidence_pykonal_s)

    s_pick_abs = ev.origin_time + ev.s_arrival_time
    trim_start = s_pick_abs - splitting_obj.overall_win_start_pre_fast_S_pick
    trim_end = s_pick_abs + splitting_obj.overall_win_start_post_fast_S_pick
    st_lqt_uncorr_notrail = st_lqt_uncorr.copy().trim(starttime=trim_start, endtime=trim_end)
    st_lqt_corr_notrail = st_lqt_corr.copy().trim(starttime=trim_start, endtime=trim_end)

    out_pdf = os.path.join(HERE, f'figure_axec2_event_{ev.event_id}_{ev.label}_diagnostics.pdf')
    with PdfPages(out_pdf) as pdf:
        plot_zne_page(pdf, ev, st_zne)
        plot_lqt_page(pdf, ev, st_lqt, splitting_obj)
        plot_particle_motion_page(
            pdf, st_lqt_uncorr_notrail.select(channel='??Q')[0], st_lqt_uncorr_notrail.select(channel='??T')[0],
            'Original (uncorrected) LQT particle motion', color='darkred', lw=2.0
        )
        plot_particle_motion_page(
            pdf, st_lqt_corr_notrail.select(channel='??Q')[0], st_lqt_corr_notrail.select(channel='??T')[0],
            'Splitting-corrected LQT particle motion', color='navy', lw=2.0
        )
        plot_phi_dt_space_page(pdf, ev, splitting_obj)
        plot_clustering_page(pdf, ev, splitting_obj)
        plot_multipanel_summary_page(
            pdf, ev, st_lqt, splitting_obj,
            st_lqt_uncorr_notrail.select(channel='??Q')[0], st_lqt_uncorr_notrail.select(channel='??T')[0],
            st_lqt_corr_notrail.select(channel='??Q')[0], st_lqt_corr_notrail.select(channel='??T')[0],
        )

    print(f'Wrote {out_pdf}')


def main():
    for ev_dict in EVENTS:
        run_event(ev_dict)


if __name__ == '__main__':
    main()
