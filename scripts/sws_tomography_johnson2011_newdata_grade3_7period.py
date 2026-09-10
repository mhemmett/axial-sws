#!/usr/bin/env python3
"""
sws_tomography_johnson2011_newdata_grade3_7period.py

Re-run of the Johnson, Savage & Townend (2011) 2-D delay-time SWS tomography
(sws_tomography_johnson2011.py, faithfully reimplemented earlier this project),
on the CURRENT 6-station "newdata" catalog (mfast_maxdt_pipeline_transfer/,
per-event try_filters bandpass + max_t_shift_s=0.2s) -- the dataset used
throughout this session for the traveltime-anisotropy/phi-quadtree figures --
instead of the original script's stale lqt_pykonal_combined_results/ source
(which no longer exists on disk).

Two changes from the original script's own defaults, per explicit user
instruction:
  1. FILTER: replaces apply_filters()'s quality_min/incidence-window gate with
     this session's grade-3 filter (T.GRADES[3]): SNR>=2.0, quality>=0.75,
     dt_err<=0.05s, dt<=T_dom/2, phi_err<=20deg. No incidence-angle cut (not
     part of grade-3). Applied BEFORE ray tracing (grade-3 depends only on
     already-computed per-event columns), which also cuts the tracing cost from
     ~730k raw newdata rows to ~93k grade-3 survivors across all 6 stations.
  2. EPOCHS -> 7 PERIODS: replaces the original script's 3 fixed epochs
     (combined / 2015-2021 / 2022-2026) with the session's standard 7
     eruption-relative periods (Pre-eruption, Syn-eruption, then 5 equal-count
     post-eruption bins) -- SAME period boundaries as
     traveltime_anisotropy_7period_6stations_newdata_snr_grades.py's
     build_7_periods(), run independently per period (not a cumulative ladder).

Everything else reuses the original module's machinery UNCHANGED and imported,
not copied: JOINT six-station ray-id-offset merge (sws_tomography_johnson2011_
joint.py's pattern), quad-tree gridding, G/d system assembly, covariance
whitening, bounded WLS solve, per-block circular phi averaging (inv_d2
weighting), and the checkerboard-recovery resolution test.

Produces TWO figures (1 row x 7 periods each, true-max shared colorbar, same
bathy/station/fault conventions as this session's other 7-period 1-row
figures):
  sws_tomography_johnson2011_newdata_grade3_7period_strength.pdf
      quad-tree blocks coloured by recovered s_b [s/km] (eq. 1); blocks that
      fail the checkerboard-recovery test are greyed/hatched (in the model,
      not shown as trustworthy) -- same convention as plot_dt_strength().
  sws_tomography_johnson2011_newdata_grade3_7period_phi_quiver.pdf
      per-block fast-polarization (principal-stress-proxy) direction, as a
      double-headed quiver tick per KEPT block (sigma_phi/SE_phi gates),
      oriented + coloured by phi_bar on a cyclic ('hsv') colormap normalized to
      [0, 180] -- same quiver convention as
      phi_7period_6stations_newdata_snr2_q75_countmin50_midpoint_quadtree_1row.py,
      tick length scaled to the block's own quad-tree size.

Run with:
    python3 sws_tomography_johnson2011_newdata_grade3_7period.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Patch

import sws_tomography_johnson2011 as J
from pykonal_raytracer import BaillardRayTracer
import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
OUT_DIR = os.path.join(HERE, 'sws_tomography_johnson2011_newdata_grade3_7period_results')
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PDF_STRENGTH = os.path.join(HERE, 'sws_tomography_johnson2011_newdata_grade3_7period_strength.pdf')
OUT_PDF_PHI = os.path.join(HERE, 'sws_tomography_johnson2011_newdata_grade3_7period_phi_quiver.pdf')

STATIONS = J.STATIONS
NEWDATA_FILES = {sta: [f'splitting_results_{sta}_2015_2021_all_batches.csv',
                       f'splitting_results_{sta}_2022_2026_all_batches.csv']
                 for sta in STATIONS}

GRADE = T.GRADES[3]
assert GRADE['key'] == 'page1_q0.75', GRADE

RAY_ID_OFFSET = 10_000_000
PHI_WEIGHT = 'inv_d2'

# Display-only gate (does NOT change the underlying model/inversion -- COUNT_KEEP=8
# still governs which blocks are USED columns in G): a block is only PLOTTED if its
# quad-tree leaf 'count' (distinct rays crossing it) is >= this, per explicit user
# instruction. Applies uniformly to every Johnson-et-al. figure in this session.
DISPLAY_COUNT_MIN = 20

# Resolution gate (per explicit user instruction): a block is only PLOTTED if its
# posterior-variance log10(var) [active-set, see significance_active()] is BELOW
# this level, i.e. its resolution/certainty is high enough -- same -5.5 threshold
# already used by resolution_diagnostics()/significance_active() throughout
# sws_tomography_johnson2011.py. Applied IN ADDITION TO DISPLAY_COUNT_MIN (both
# must pass), on the strength and phi-quiver figures only.
SIGNIFICANCE_LEVEL = -5.5


def _significant_mask(res):
    """Boolean array (aligned with res['system']['used_leaves']) -- True where a
    block's active-set posterior-variance log10(var) < SIGNIFICANCE_LEVEL."""
    log10_var = res['significance']['log10_var']
    return np.isfinite(log10_var) & (log10_var < SIGNIFICANCE_LEVEL)


# ── Data loading: newdata source, grade-3 pre-filter (before tracing) ──────────

def load_station_newdata_grade3(sta):
    dfs = [pd.read_csv(os.path.join(TRANSFER_DIR, f)) for f in NEWDATA_FILES[sta]]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'quality', 'snr_horizontal',
                            'dt_error', 'dominant_period', 'phi_error', 'phi', 'event_id'])
    df['t'] = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')

    mask = ((df['snr_horizontal'] >= GRADE['snr_min']) &
            (df['quality'] >= GRADE['qw_min']) &
            (df['dt_error'] <= 0.05) &
            (df['dt'] <= df['dominant_period'] / 2.0) &
            (df['phi_error'] <= GRADE['phi_err_max']))
    df = df[mask].copy()

    from sws_forward_model import ll2xy
    df['x'], df['y'] = ll2xy(df['latitude'].values, df['longitude'].values)
    df['z'] = df['depth'].values
    return df[['event_id', 't', 'dt', 'dt_error', 'phi', 'phi_error', 'quality',
               'x', 'y', 'z']].reset_index(drop=True)


def get_station_cache(sta, tracer):
    s_csv = os.path.join(OUT_DIR, f'{sta}_ray_summary.csv')
    c_csv = os.path.join(OUT_DIR, f'{sta}_ray_finecells.csv')
    if os.path.exists(s_csv) and os.path.exists(c_csv):
        print(f'  {sta}: loading cached rays')
        summ = pd.read_csv(s_csv, parse_dates=['t'])
        cells = pd.read_csv(c_csv, dtype={'ray_id': np.int64, 'ixf': np.int32,
                                          'iyf': np.int32, 'seg_len_km': np.float64})
        return summ, cells
    df = load_station_newdata_grade3(sta)
    print(f'  {sta}: tracing {len(df):,} grade-3 rays...')
    return J.build_ray_finecell_cache(sta, df, tracer, out_dir=OUT_DIR)


def build_joint_cache():
    sta_xy = J.station_xy()
    tracer = BaillardRayTracer(stride=5)
    for sta in STATIONS:
        tracer.precompute_station(sta, *sta_xy[sta])

    per_summ, per_cells = [], []
    station_index = {}
    for si, sta in enumerate(STATIONS):
        summ_s, cells_s = get_station_cache(sta, tracer)
        max_rid = int(summ_s['ray_id'].max())
        assert max_rid < RAY_ID_OFFSET, f'{sta}: ray_id overflow vs offset'
        base = si * RAY_ID_OFFSET
        summ_s = summ_s.copy()
        cells_s = cells_s.copy()
        summ_s['ray_id'] = summ_s['ray_id'].astype(np.int64) + base
        cells_s['ray_id'] = cells_s['ray_id'].astype(np.int64) + base
        summ_s['station'] = sta
        per_summ.append(summ_s)
        per_cells.append(cells_s)
        station_index[sta] = si

    joint_summ = pd.concat(per_summ, ignore_index=True)
    joint_cells = pd.concat(per_cells, ignore_index=True)
    assert joint_summ['ray_id'].is_unique, 'ray_id collision across stations'
    return joint_summ, joint_cells, station_index, sta_xy


# ── Validity gate (trace_ok / in_grid / sig_dt2) -- grade-3 already applied upstream ─

def apply_validity(summ, dt_error_alpha=J.DT_ERROR_ALPHA):
    counts = {'total': int(len(summ))}
    df = summ[summ['trace_ok'] == 1].copy()
    counts['trace_ok'] = int(len(df))
    arc = df['arc_len_km'].values.astype(float)
    scell = df['sum_finecell_km'].values.astype(float)
    in_grid = np.abs(scell - arc) <= 1e-9 * np.maximum(arc, 1.0)
    df = df[in_grid]
    counts['in_grid'] = int(len(df))

    z = J.dt_error_z(dt_error_alpha)
    e = pd.to_numeric(df['dt_error'], errors='coerce').values.astype(float) / z
    e2 = e ** 2
    good = np.isfinite(e2) & (e2 > 0)
    med = np.median(e2[good]) if good.any() else 1e-4
    e2 = np.where(good, e2, med)
    df['sig_dt2'] = e2
    return df.reset_index(drop=True), counts


def restrict_period(df, t0, t1):
    t = pd.to_datetime(df['t'], utc=True, format='mixed')
    m = pd.Series(True, index=df.index)
    if t0 is not None:
        m &= (t >= t0).values
    if t1 is not None:
        m &= (t < t1).values
    return df[m].reset_index(drop=True)


def run_chain_grade3_period(summ, cells, t0, t1, n=100, L_b=J.DELTA,
                             count_subdivide=J.COUNT_SUBDIVIDE, count_keep=J.COUNT_KEEP,
                             phi_weight=PHI_WEIGHT, checkerboard_lam=1.0,
                             sigma_phi_max=J.SIGMA_PHI_MAX, se_phi_max=J.SE_PHI_MAX,
                             cb_recovery_tol=J.CB_RECOVERY_TOL, verbose=True):
    kept, counts = apply_validity(summ)
    kept = restrict_period(kept, t0, t1)
    counts['period'] = int(len(kept))
    if len(kept) < count_keep:
        raise ValueError(f'STOP: only {len(kept)} rays survive grade-3+period -- '
                         f'too sparse for a {count_keep}-ray/{count_subdivide}-ray grid.')
    ray_meta = kept.set_index('ray_id')[['dt', 'sig_dt2', 'phi', 'sx', 'sy']]

    leaves, r_ray, r_ixf, r_iyf, r_seg = J.build_quadtree(
        cells, kept['ray_id'].values, count_subdivide=count_subdivide)
    system = J.build_system(leaves, r_ray, r_ixf, r_iyf, r_seg, ray_meta, count_keep=count_keep)

    sigma_m2, _ = J.estimate_sigma_m2(system['G'], system['sig_dt2'], return_cond=True)
    alpha, D = J.build_covariance(system['sig_dt2'], system['n_b'], n, L_b, sigma_m2)
    sol = J.solve_tomography(system['G'], system['d'], D, alpha, L_min=L_b,
                             used_leaves=system['used_leaves'])
    m = sol['m']

    phi_stats = {w: J.block_phi_average(system, m, weight=w, sigma_phi_max=sigma_phi_max,
                                        se_phi_max=se_phi_max) for w in J.PHI_WEIGHTS}
    n_phi_kept = int(sum(1 for b in phi_stats[phi_weight] if b['kept']))

    cb = J.run_checkerboard(system, D=D, alpha=alpha, sig_dt2=system['sig_dt2'],
                            lam=checkerboard_lam, noise='per_ray')
    cb_resolvable = J.checkerboard_block_mask(cb, tol=cb_recovery_tol)

    # Resolution / posterior-variance diagnostics (§6): significance_active recomputes
    # diag(C_m) on the ACTIVE (unclipped) block set only, matching the bound-
    # constrained s_b map actually shown (see its docstring) -- this is the per-block
    # "resolution" gate applied downstream in addition to count_min.
    resolution = J.resolution_diagnostics(sol['GtCiG'])
    significance = J.significance_active(sol['G_tilde'], sol['free_mask'],
                                         significance_level=resolution['significance_level'])

    diagnostics = dict(counts=counts, created=system['created'], used=system['used'],
                       n_rays=len(system['ray_ids']), s_b_min=float(m.min()),
                       s_b_max=float(m.max()), s_b_median=float(np.median(m)),
                       cond=sol['cond'], n_phi_kept=n_phi_kept, cb_corr=cb['corr'],
                       cb_recovery_tol=cb_recovery_tol,
                       n_cb_resolvable=int(cb_resolvable.sum()),
                       n_sig_active=significance['n_significant'])
    if verbose:
        print(f'    rays={diagnostics["n_rays"]:,} blocks {system["created"]}->{system["used"]} '
              f's_b[{m.min():.4f},{m.max():.4f}] cond={sol["cond"]:.2e} '
              f'phi_kept={n_phi_kept} cb_resolvable={diagnostics["n_cb_resolvable"]}/{system["used"]} '
              f'sig_active={diagnostics["n_sig_active"]}/{system["used"]}')
    return dict(system=system, m=m, sol=sol, phi_stats=phi_stats,
               cb_resolvable=cb_resolvable, checkerboard=cb, resolution=resolution,
               significance=significance, diagnostics=diagnostics)


# ── Plotting: 1 row x 7 periods, shared true-max colorbar ──────────────────────

FS_N = 7.5
FS_PERIOD = 8.5


def _panel_common(ax, sta_xy_all):
    T._bathy(ax)
    for sta, row in T._sta_df.iterrows():
        ax.plot(row['x'], row['y'], '^', ms=6, mfc='#FFD700', mec='k', mew=0.8, zorder=13)
    T._faults(ax, 1.5)
    ax.set_xlim(T.X_START, T.X_END)
    ax.set_ylim(T.Y_START, T.Y_END)
    ax.set_aspect('equal', 'box')
    ax.tick_params(labelsize=4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def make_strength_figure(results, period_labels):
    n_per = len(results)
    all_m = np.concatenate([r['m'] for r in results if r is not None and len(r['m'])])
    vmax = float(np.percentile(all_m, 99)) if all_m.size else 0.05
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.05
    cmap = plt.colormaps['viridis']

    panel_w = 2.5
    n_cols = n_per + 1
    width_ratios = [1] * n_per + [0.05]
    header_room_in, footer_room_in = 0.85, 0.32
    fig_h_in = 3.6 + footer_room_in + header_room_in
    fig = plt.figure(figsize=(n_per * panel_w + 1.0, fig_h_in))
    gs_top = 1.0 - header_room_in / fig_h_in
    gs_bottom = footer_room_in / fig_h_in
    gs = GridSpec(1, n_cols, width_ratios=width_ratios, wspace=0.04, top=gs_top, bottom=gs_bottom)
    period_label_offset_frac = 0.32 / fig_h_in

    ref_ax_pos = None
    last_h = None
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci])
        _panel_common(ax, None)
        if res is not None:
            leaves = res['system']['used_leaves']
            m = res['m']
            resolvable = np.asarray(res['cb_resolvable'])
            significant = _significant_mask(res)
            for lf, val, ok, sig in zip(leaves, m, resolvable, significant):
                if lf['count'] < DISPLAY_COUNT_MIN or not sig:
                    continue
                xc, yc = lf['center']
                s = lf['side_km']
                color = cmap(min(val / vmax, 1.0))
                ax.add_patch(Rectangle((xc - s / 2, yc - s / 2), s, s,
                                       facecolor=color, edgecolor='0.4', lw=0.2, zorder=6))
                if not ok:
                    ax.add_patch(Rectangle((xc - s / 2, yc - s / 2), s, s,
                                           facecolor='0.7', alpha=0.6, hatch='xxx',
                                           edgecolor='0.35', lw=0.2, zorder=7))
            last_h = ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
            last_h.set_array([])
            n_rays = res['diagnostics']['n_rays']
        else:
            n_rays = 0
        ax.set_title(f'N={n_rays:,}', fontsize=FS_N, fontweight='bold', pad=14)
        ax_pos = ax.get_position()
        fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + period_label_offset_frac,
                 period_labels[ci].replace('\n–', ' – ').replace('\n', ' '),
                 fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
        ref_ax_pos = ax_pos

    if last_h:
        cell = gs[:, n_cols - 1].get_position(fig)
        cax = fig.add_axes([cell.x0, ref_ax_pos.y0, cell.width, ref_ax_pos.height])
        cb = plt.colorbar(last_h, cax=cax, ticks=np.linspace(0, vmax, 5),
                          format=FormatStrFormatter('%.3f'))
        cb.set_label(r'$s_b$ anisotropy strength [s/km]', fontsize=7.5)
        cb.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
        Patch(facecolor='0.7', alpha=0.6, hatch='xxx', edgecolor='0.35',
             label='fails checkerboard recovery'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=7,
              framealpha=0.9, bbox_to_anchor=(0.45, 0.05))
    fig.suptitle('Johnson et al. (2011) 2-D SWS tomography: anisotropy strength $s_b$ '
                 '(newdata, grade-3 filter)', fontsize=10, fontweight='bold', y=0.99)
    return fig


def make_phi_quiver_figure(results, period_labels):
    n_per = len(results)
    AZ_CMAP = plt.colormaps['hsv']
    AZ_NORM = Normalize(0, 180)

    panel_w = 2.5
    n_cols = n_per + 1
    width_ratios = [1] * n_per + [0.05]
    header_room_in, footer_room_in = 0.85, 0.32
    fig_h_in = 3.6 + footer_room_in + header_room_in
    fig = plt.figure(figsize=(n_per * panel_w + 1.0, fig_h_in))
    gs_top = 1.0 - header_room_in / fig_h_in
    gs_bottom = footer_room_in / fig_h_in
    gs = GridSpec(1, n_cols, width_ratios=width_ratios, wspace=0.04, top=gs_top, bottom=gs_bottom)
    period_label_offset_frac = 0.32 / fig_h_in

    ref_ax_pos = None
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs[0, ci])
        _panel_common(ax, None)
        n_kept = 0
        if res is not None:
            stats = res['phi_stats'][PHI_WEIGHT]
            significant = _significant_mask(res)
            kept_blocks = [b for b, sig in zip(stats, significant)
                          if b['kept'] and np.isfinite(b['phi_bar'])
                          and b['count'] >= DISPLAY_COUNT_MIN and sig]
            n_kept = len(kept_blocks)
            if kept_blocks:
                cx = np.array([b['center'][0] for b in kept_blocks])
                cy = np.array([b['center'][1] for b in kept_blocks])
                sz = np.array([b['side_km'] for b in kept_blocks])
                phi_vals = np.array([b['phi_bar'] for b in kept_blocks])
                r = np.radians(phi_vals)
                length = 0.8 * sz
                u, v = length * np.sin(r), length * np.cos(r)
                tick_kw = dict(scale=1, scale_units='xy', width=0.005, headlength=0,
                              headaxislength=0, headwidth=0, pivot='middle',
                              cmap=AZ_CMAP, norm=AZ_NORM, alpha=0.95, zorder=7)
                ax.quiver(cx, cy, u, v, phi_vals, **tick_kw)
                ax.quiver(cx, cy, -u, -v, phi_vals, **tick_kw)
        ax.set_title(f'{n_kept} blocks', fontsize=FS_N, fontweight='bold', pad=14)
        ax_pos = ax.get_position()
        fig.text(ax_pos.x0 + ax_pos.width / 2, ax_pos.y1 + period_label_offset_frac,
                 period_labels[ci].replace('\n–', ' – ').replace('\n', ' '),
                 fontsize=FS_PERIOD, fontweight='bold', ha='center', va='bottom')
        ref_ax_pos = ax_pos

    cell = gs[:, n_cols - 1].get_position(fig)
    cax = fig.add_axes([cell.x0, ref_ax_pos.y0, cell.width, ref_ax_pos.height])
    sm = ScalarMappable(cmap=AZ_CMAP, norm=AZ_NORM)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, ticks=np.linspace(0, 180, 7))
    cb.set_label('Fast direction $\\phi$ (deg, axial 0-180)', fontsize=7.5)
    cb.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=6, label='Stations'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Faults'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=7,
              framealpha=0.9, bbox_to_anchor=(0.45, 0.05))
    fig.suptitle('Johnson et al. (2011) 2-D SWS tomography: per-block fast direction '
                 '(newdata, grade-3 filter)', fontsize=10, fontweight='bold', y=0.99)
    return fig


def main():
    T._setup_environment(need_tracer=False)

    print('Building/loading JOINT six-station grade-3 ray cache (newdata)...')
    joint_summ, joint_cells, station_index, sta_xy = build_joint_cache()
    print(f'\nJoint cache: {len(joint_summ):,} rays total across {len(STATIONS)} stations')

    # Cached 't' columns round-trip through CSV as strings; build_7_periods does a
    # direct datetime comparison, so parse explicitly here (restrict_period() inside
    # run_chain_grade3_period already does this itself, per-call). Only matters on a
    # re-run from cache -- the first run traced fresh Timestamps directly.
    t_parsed = pd.to_datetime(joint_summ['t'], utc=True, format='mixed')
    periods = T.build_7_periods(pd.DataFrame({'t': t_parsed}))
    period_labels = [lbl for lbl, _t0, _t1 in periods]

    print('\n=== Running Johnson et al. (2011) chain independently per period ===')
    results = []
    for lbl, t0, t1 in periods:
        print(f'\n-- {lbl.replace(chr(10), " ")} --')
        try:
            res = run_chain_grade3_period(joint_summ, joint_cells, t0, t1)
        except ValueError as exc:
            print(f'  [SKIP] {exc}')
            res = None
        results.append(res)

    print('\nBuilding figures...')
    fig1 = make_strength_figure(results, period_labels)
    with PdfPages(OUT_PDF_STRENGTH) as pdf:
        pdf.savefig(fig1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved {OUT_PDF_STRENGTH}')

    fig2 = make_phi_quiver_figure(results, period_labels)
    with PdfPages(OUT_PDF_PHI) as pdf:
        pdf.savefig(fig2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {OUT_PDF_PHI}')


if __name__ == '__main__':
    main()
