#!/usr/bin/env python3
"""
sws_tomography_johnson2011_demo.py

Demo driver for sws_tomography_johnson2011: runs the full 2-D delay-time SWS
tomography chain (Johnson, Savage & Townend 2011, §3.2-3.3) on the REAL Axial
Seamount MLdd splitting results.

What it does
------------
1. Reports per-station survivable ray counts (baseline / q>=-0.5 / q>=0.7) for
   every station and epoch WITHOUT tracing, so the best-sampled station(s) are
   obvious and reproducible.
2. Traces (and caches) the best-sampled station(s), then runs the full chain for
   the 3 epochs {combined, 2015_2021, 2022_2026} x 2 quality thresholds
   {-0.5, 0.7} = 6 configurations.
3. For each configuration produces three figures under
   <repo_root>/sws_tomography_johnson2011_results/:
     <sta>_<epoch>_q<thr>_dt_strength.png   quad-tree blocks coloured by s_b
     <sta>_<epoch>_q<thr>_phi_rose.png       per-block fast-polarization bars+rose
     <sta>_<epoch>_q<thr>_checkerboard.png   input vs recovered checkerboard
4. Runs the data-driven §7 benchmarks (arc length, uniform strength,
   checkerboard recovery, resolution) and the data-free asserts, printing the
   actual numbers.

Run:
    python3 sws_tomography_johnson2011_demo.py [--stations AXEC2 AXCC1]
                                               [--retrace] [--n-figs N]

Nothing is git-added; the results dir is a new, gitignored-style local output.
"""

import argparse
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import sws_tomography_johnson2011 as tomo
from pykonal_raytracer import BaillardRayTracer

QUALITY_THRESHOLDS = [-0.5, 0.7]
EPOCHS = ['combined', '2015_2021', '2022_2026']


# ── Pre-trace census (no ray tracing) ────────────────────────────────────────

def census(catalog):
    """Report per-station, per-epoch survivable counts using only the merged
    splitting results + catalog (no tracing; incidence not yet applied)."""
    print('\n=== Pre-trace ray census (baseline / q>=-0.5 / q>=0.7) ===')
    print(f'{"station":>8} {"epoch":>10} {"baseline":>10} {"q>=-0.5":>9} '
          f'{"q>=0.7":>9}')
    totals = {}
    for sta in tomo.STATIONS:
        try:
            df = tomo.load_station_measurements(sta, catalog)
        except Exception as exc:
            print(f'{sta:>8}  load failed: {exc}')
            continue
        for epoch in EPOCHS:
            t0, t1 = tomo.EPOCHS[epoch]
            t = df['t']
            m = np.ones(len(df), dtype=bool)
            if t0 is not None:
                m &= (t >= t0).values
            if t1 is not None:
                m &= (t < t1).values
            sub = df[m]
            base = len(sub)
            q05 = int((sub['quality'] >= -0.5).sum())
            q07 = int((sub['quality'] >= 0.7).sum())
            print(f'{sta:>8} {epoch:>10} {base:>10,} {q05:>9,} {q07:>9,}')
            if epoch == 'combined':
                totals[sta] = q05
    return totals


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _block_patches(leaves):
    """Rectangles for each used leaf (lower-left corner + side)."""
    rects = []
    for lf in leaves:
        xc, yc = lf['center']
        s = lf['side_km']
        rects.append(Rectangle((xc - s / 2, yc - s / 2), s, s))
    return rects


def _base_ax(ax, sta_xy, highlight):
    for s, (x, y) in sta_xy.items():
        c = 'gold' if s == highlight else 'white'
        ax.plot(x, y, '^', ms=8, mfc=c, mec='k', mew=0.8, zorder=12)
    ax.set_xlim(tomo.FINE_X0 + 1, tomo.FINE_X1 - 4)
    ax.set_ylim(tomo.FINE_Y0 + 1, tomo.FINE_Y1 - 4)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('East [km]')
    ax.set_ylabel('North [km]')


def plot_dt_strength(result, sta, sta_xy, out_path):
    """delta_t anisotropy-strength map: used quad-tree blocks coloured by the
    recovered s_b [s/km], RESTRICTED to checkerboard-validated regions.

    Finishing changes:
      #1  Blocks that FAIL the checkerboard-recovery criterion (|m_rec/m_cb - 1|
          > cb_recovery_tol; result['cb_resolvable']) are visually SUPPRESSED --
          greyed + hatched -- not deleted from the model. The map therefore only
          presents checkerboard-validated blocks as trustworthy.
      #4  'Significant' outlines now use the ACTIVE-SET posterior variance
          (result['significance_active']), recomputed on the unclipped block set to
          match the bound-constrained s_b map, NOT the unconstrained normal
          equations.
    """
    from matplotlib.patches import Patch
    system = result['system']
    m = result['m']
    leaves = system['used_leaves']
    resolvable = np.asarray(result['cb_resolvable'])            # #1
    sig_a = result['significance_active']                       # #4
    log10_var = sig_a['log10_var']
    sig_level = sig_a['significance_level']
    fig, ax = plt.subplots(figsize=(6.5, 6))
    rects = _block_patches(leaves)
    pc = PatchCollection(rects, cmap='viridis',
                         edgecolor='0.6', linewidths=0.3)
    pc.set_array(m)
    pc.set_clim(0, max(np.percentile(m, 98), 1e-6))
    ax.add_collection(pc)
    # #1: grey + hatch blocks that FAIL checkerboard recovery so they do not read
    # as trustworthy (kept in the model, presentation only).
    for lf, ok in zip(leaves, resolvable):
        if not ok:
            xc, yc = lf['center']
            s = lf['side_km']
            ax.add_patch(Rectangle((xc - s / 2, yc - s / 2), s, s,
                                   facecolor='0.7', alpha=0.8, hatch='xxx',
                                   edgecolor='0.35', lw=0.3, zorder=5))
    # #4: significant blocks (active-set posterior variance < 10^-5.5) in red.
    for lf, lv in zip(leaves, log10_var):
        if np.isfinite(lv) and lv < sig_level:
            xc, yc = lf['center']
            s = lf['side_km']
            ax.add_patch(Rectangle((xc - s / 2, yc - s / 2), s, s,
                                   fill=False, edgecolor='red', lw=1.0,
                                   zorder=6))
    _base_ax(ax, sta_xy, sta)
    cb = fig.colorbar(pc, ax=ax, shrink=0.8)
    cb.set_label(r'$s_b$  anisotropy strength [s/km]  (eq. 1)')
    d = result['diagnostics']
    n_res = int(resolvable.sum())
    ax.set_title(f'{sta}  {result["epoch"]}  q>={result["quality_min"]:g}\n'
                 f'blocks {d["created"]}->{d["used"]}  rays {d["n_rays"]:,}  '
                 f'cond={d["cond"]:.1e}\n'
                 f'{n_res}/{d["used"]} checkerboard-validated (solid); '
                 f'red = active-set signif. (var<1e-5.5, {d["n_sig_active"]})',
                 fontsize=8)
    # Legend + caption: state that the map is restricted to validated regions.
    handles = [Patch(facecolor='0.7', alpha=0.8, hatch='xxx', edgecolor='0.35',
                     label=(f'fails checkerboard recovery '
                            f'(|m_rec/m_cb-1|>{d["cb_recovery_tol"]:g}) -- '
                            f'not resolvable')),
               Patch(facecolor='none', edgecolor='red',
                     label='significant (active-set var<1e-5.5)')]
    ax.legend(handles=handles, loc='upper left', fontsize=6, framealpha=0.85)
    fig.text(0.5, 0.005,
             'Strength map restricted to checkerboard-validated regions; greyed/'
             'hatched blocks are retained in the model but not shown as reliable.',
             ha='center', fontsize=6.5, style='italic')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phi_rose(result, sta, sta_xy, out_path, weight='inv_d2'):
    """Per-block fast-polarization bar map (kept blocks only) + a rose diagram of
    the kept block phi_bar values (double-angle circular mean, eqs. 6-7)."""
    stats = result['phi_stats'][weight]
    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(1, 2, 1)
    for b in stats:
        if not b['kept']:
            continue
        xc, yc = b['center']
        s = b['side_km']
        ang = np.radians(b['phi_bar'])       # 0 deg = North; bar along phi
        dx = 0.45 * s * np.sin(ang)
        dy = 0.45 * s * np.cos(ang)
        ax.plot([xc - dx, xc + dx], [yc - dy, yc + dy], '-', color='crimson',
                lw=1.6, zorder=8)
    _base_ax(ax, sta_xy, sta)
    n_kept = sum(1 for b in stats if b['kept'])
    d = result['diagnostics']
    # #2: report the ACTUAL (data-driven, relaxed) keep-gates used, not 30/10.
    ax.set_title(f'{sta} {result["epoch"]} q>={result["quality_min"]:g}\n'
                 f'per-block phi_bar ({weight}), {n_kept}/{d["used"]} kept '
                 f'(sigma_phi<{d["sigma_phi_max"]:g}, '
                 f'SE_phi<{d["se_phi_max"]:g})', fontsize=9)

    axr = fig.add_subplot(1, 2, 2, projection='polar')
    phis = [b['phi_bar'] for b in stats if b['kept'] and np.isfinite(b['phi_bar'])]
    if phis:
        # Axial data: plot each phi and its 180-deg mirror. 0 deg = North, CW.
        ang = np.radians(phis)
        both = np.concatenate([ang, ang + np.pi])
        axr.hist(both, bins=36, range=(0, 2 * np.pi), color='crimson',
                 alpha=0.7)
    axr.set_theta_zero_location('N')
    axr.set_theta_direction(-1)
    axr.set_title('kept block phi_bar (axial)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_checkerboard(result, sta, sta_xy, out_path):
    """Checkerboard test (eq. 5): input model m_CB vs recovered m_rec, block
    maps side by side. Recovery uses per-ray N(0, sigma_dt) noise (Auditor #2)."""
    system = result['system']
    cb = result['checkerboard']
    leaves = system['used_leaves']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    vmax = max(cb['m_cb'].max(), np.percentile(cb['m_rec'], 98), 1e-6)
    for ax, vals, ttl in ((axes[0], cb['m_cb'], 'input $m_{CB}$'),
                          (axes[1], cb['m_rec'], 'recovered $m_{rec}$')):
        rects = _block_patches(leaves)
        pc = PatchCollection(rects, cmap='RdBu_r', edgecolor='0.6',
                             linewidths=0.3)
        pc.set_array(np.asarray(vals))
        pc.set_clim(0, vmax)
        ax.add_collection(pc)
        _base_ax(ax, sta_xy, sta)
        fig.colorbar(pc, ax=ax, shrink=0.7).set_label('s [s/km]')
        ax.set_title(ttl, fontsize=10)
    fig.suptitle(f'{sta} {result["epoch"]} q>={result["quality_min"]:g}  '
                 f'checkerboard (lam=1 km)  corr={cb["corr"]:.3f}  '
                 f'well-sampled corr={cb["corr_well_sampled"]:.3f}',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stations', nargs='*', default=None,
                    help='stations to trace + invert (default: best-sampled 1)')
    ap.add_argument('--n-best', type=int, default=1,
                    help='how many best-sampled stations to run if --stations '
                         'not given (default 1)')
    ap.add_argument('--retrace', action='store_true')
    args = ap.parse_args()

    os.makedirs(tomo.RESULTS_DIR, exist_ok=True)
    print('Loading MLdd catalog...')
    catalog = tomo.load_catalog()
    sta_xy = tomo.station_xy()

    totals = census(catalog)

    if args.stations:
        run_stations = args.stations
    else:
        run_stations = [s for s, _ in sorted(totals.items(),
                                             key=lambda kv: -kv[1])][:args.n_best]
    print(f'\nBest-sampled (by combined q>=-0.5): '
          f'{sorted(totals.items(), key=lambda kv: -kv[1])}')
    print(f'Running full chain for: {run_stations}')

    # Build the ray tracer once; precompute FMM fields for the run stations.
    tracer = BaillardRayTracer(stride=5)
    for sta in run_stations:
        tracer.precompute_station(sta, *sta_xy[sta])

    # Data-free asserts (also run in __main__ of the module).
    print('\n=== Data-free verification (Auditor #1, #5) ===')
    print('  phi axial mean :', tomo.verify_phi_axial_mean())
    for c in tomo.verify_whitening():
        print('  whitening      :', c)

    for sta in run_stations:
        print(f'\n########## STATION {sta} ##########')
        summ, cells = tomo.load_or_build_ray_cache(sta, catalog, tracer,
                                                   retrace=args.retrace)
        # §7a arc-length benchmark (hard assert) on the cache.
        ok_a, max_rel, n_a, n_leak = tomo.verify_arclength(summ, cells)
        print(f'  §7a arc-length: max rel err {max_rel:.2e} over {n_a:,} in-grid '
              f'rays (pass={ok_a}); {n_leak:,} far-field leaks excluded upstream')

        for q in QUALITY_THRESHOLDS:
            for epoch in EPOCHS:
                tag = f'{sta}_{epoch}_q{q:g}'
                try:
                    res = tomo.run_chain(summ, cells, quality_min=q,
                                         epoch=epoch)
                except ValueError as exc:
                    print(f'  [SKIP] {tag}: {exc}')
                    continue
                # §7b, §7c, §7d benchmarks.
                ok_b, err_b = tomo.verify_uniform_strength(res['system']['G'])
                ok_c, corr_c = tomo.verify_checkerboard(
                    res['system'], res['D'], res['alpha'])
                ok_d, res_err, cond_d, wc = tomo.verify_resolution(
                    res['sol']['GtCiG'])
                d = res['diagnostics']
                print(f'    [{tag}] §7b uniform err={err_b:.1e}  '
                      f'§7c cb-corr(noise-free)={corr_c:.3f}  '
                      f'§7d ||Res-I||={res_err:.1e} cond={cond_d:.1e} '
                      f'well-cond={wc}')
                # Figures.
                plot_dt_strength(res, sta, sta_xy,
                                 os.path.join(tomo.RESULTS_DIR,
                                              f'{tag}_dt_strength.png'))
                plot_phi_rose(res, sta, sta_xy,
                              os.path.join(tomo.RESULTS_DIR,
                                           f'{tag}_phi_rose.png'))
                plot_checkerboard(res, sta, sta_xy,
                                  os.path.join(tomo.RESULTS_DIR,
                                               f'{tag}_checkerboard.png'))

    print(f'\nDone. Figures + caches in {tomo.RESULTS_DIR}')


if __name__ == '__main__':
    main()
