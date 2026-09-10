#!/usr/bin/env python3
"""
synthetic_prolate_spheroid_shifted_nw_vs_johnson_preeruption.py

Follow-up to synthetic_prolate_spheroid_axas1_axas2_stress_field.py, per explicit user
request: move the vertical prolate spheroid source (rapidly inflating, 3.0-5.0 km depth,
between AXAS1/AXAS2) slightly NORTH and slightly WEST of the AXAS1-AXAS2 midpoint, then
overlay the Johnson et al. (2011) newdata grade-3 tomography's PRE-ERUPTION observed fast
direction on top.

Shift amount: NOT specified by the user ("slightly") -- ASSUMED 0.5 km north and 0.5 km west
of the AXAS1-AXAS2 midpoint (SHIFT_NORTH_KM / SHIFT_WEST_KM below); change and re-run for a
different offset. All other source parameters (depth range 3.0-5.0 km, b=0.3 km assumed,
dP=+200 MPa assumed) are UNCHANGED from the base script.

Reuses the base script's Yang (1988) prolate-spheroid physics via import (not duplicated).
Color/layer convention matches synthetic_hudson_vs_johnson_observed_3panel.py: modeled sigma1
orientation in BLUE, observed Johnson-2011 pre-eruption phi (same DISPLAY_COUNT_MIN=20 +
SIGNIFICANCE_LEVEL=-5.5 gates as that script) in RED on top, structures (spheroid footprint,
ring faults) in grayscale/black, differential-stress-magnitude background in Blues.

Produces:
    synthetic_prolate_spheroid_shifted_nw_vs_johnson_preeruption.pdf
(1 page, 1 panel) -- a NEW file, does not touch any existing output.

Run with:
    python3 synthetic_prolate_spheroid_shifted_nw_vs_johnson_preeruption.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
import sws_tomography_johnson2011_newdata_grade3_7period as S
import synthetic_prolate_spheroid_axas1_axas2_stress_field as BASE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'synthetic_prolate_spheroid_shifted_nw_vs_johnson_preeruption.pdf')

SHIFT_NORTH_KM = 0.5   # ASSUMED ("slightly north") -- not specified by user
SHIFT_WEST_KM = 0.5    # ASSUMED ("slightly west") -- not specified by user

MODEL_COLOR = 'tab:blue'
OBS_COLOR = 'tab:red'


def main():
    T._setup_environment(need_tracer=False)
    x0_mid = float((T._sta_df.loc['AXAS1', 'x'] + T._sta_df.loc['AXAS2', 'x']) / 2.0)
    y0_mid = float((T._sta_df.loc['AXAS1', 'y'] + T._sta_df.loc['AXAS2', 'y']) / 2.0)
    x0 = x0_mid - SHIFT_WEST_KM
    y0 = y0_mid + SHIFT_NORTH_KM
    print(f'AXAS1-AXAS2 midpoint: ({x0_mid:.3f}, {y0_mid:.3f}) km')
    print(f'Shifted source location: ({x0:.3f}, {y0:.3f}) km '
          f'({SHIFT_NORTH_KM} km N, {SHIFT_WEST_KM} km W)')

    xn = np.arange(T.X_START, T.X_END + 1e-9, BASE.GRID_STEP_KM)
    yn = np.arange(T.Y_START, T.Y_END + 1e-9, BASE.GRID_STEP_KM)
    X, Y = np.meshgrid(xn, yn)

    print('Computing Yang (1988) prolate-spheroid surface stress field (shifted source)...')
    sig = BASE.yang_surface_stress(X, Y, x0, y0, BASE.SPH_DEPTH, BASE.SPH_A, BASE.SPH_B,
                                   BASE.SPH_DIP, BASE.SPH_AZIMUTH, BASE.SPH_DP_PA, BASE.MU, BASE.NU)
    sxx, syy, sxy = sig[..., 0], sig[..., 1], sig[..., 2]
    sigma1, sigma2, az1 = BASE.principal_stress_2d(sxx, syy, sxy)
    diff_stress_mpa = (sigma2 - sigma1) / 1e6

    print('Loading JOINT six-station grade-3 ray cache (reusing existing cache, no re-tracing)...')
    joint_summ, joint_cells, station_index, sta_xy = S.build_joint_cache()
    t_parsed = pd.to_datetime(joint_summ['t'], utc=True, format='mixed')
    periods = T.build_7_periods(pd.DataFrame({'t': t_parsed}))
    pre_lbl, pre_t0, pre_t1 = periods[0]   # Pre-eruption
    print(f'\n-- Johnson et al. (2011): {pre_lbl} --')
    johnson_res = S.run_chain_grade3_period(joint_summ, joint_cells, pre_t0, pre_t1)

    fig, ax = plt.subplots(figsize=(8, 9))
    T._bathy(ax)

    vmax = float(np.nanpercentile(diff_stress_mpa, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    cs = ax.contourf(X, Y, diff_stress_mpa, levels=np.linspace(0, vmax, 15),
                     cmap='Blues', alpha=0.6, zorder=4, extend='max')
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, format=FormatStrFormatter('%.2f'))
    cb.set_label('Horizontal differential stress $\\sigma_2-\\sigma_1$ (MPa)', fontsize=8)

    # Modeled sigma1 orientation (blue), subsampled for legible tick density.
    step = max(1, int(round(0.5 / BASE.GRID_STEP_KM)))
    Xs, Ys, azs = X[::step, ::step], Y[::step, ::step], az1[::step, ::step]
    mask = np.isfinite(azs)
    r = np.radians(azs[mask])
    u, v = np.sin(r), np.cos(r)
    tick_kw = dict(scale=22, width=0.0035, headlength=0, headaxislength=0, headwidth=0,
                  pivot='middle', color=MODEL_COLOR, alpha=0.8, zorder=9)
    ax.quiver(Xs[mask], Ys[mask], u, v, **tick_kw)
    ax.quiver(Xs[mask], Ys[mask], -u, -v, **tick_kw)

    # Observed Johnson-2011 pre-eruption phi (red), on top -- same gates as the standard figure.
    stats = johnson_res['phi_stats'][S.PHI_WEIGHT]
    significant = S._significant_mask(johnson_res)
    kept_blocks = [b for b, sig_ in zip(stats, significant)
                  if b['kept'] and np.isfinite(b['phi_bar'])
                  and b['count'] >= S.DISPLAY_COUNT_MIN and sig_]
    print(f'  Observed (Johnson pre-eruption): {len(kept_blocks)} significant blocks '
         f'(count>={S.DISPLAY_COUNT_MIN})')
    if kept_blocks:
        cx = np.array([b['center'][0] for b in kept_blocks])
        cy = np.array([b['center'][1] for b in kept_blocks])
        sz = np.array([b['side_km'] for b in kept_blocks])
        phi_vals = np.array([b['phi_bar'] for b in kept_blocks])
        ro = np.radians(phi_vals)
        length = 0.8 * sz
        uo, vo = length * np.sin(ro), length * np.cos(ro)
        obs_kw = dict(scale=1, scale_units='xy', width=0.006, headlength=0, headaxislength=0,
                     headwidth=0, pivot='middle', color=OBS_COLOR, alpha=0.95, zorder=11)
        ax.quiver(cx, cy, uo, vo, **obs_kw)
        ax.quiver(cx, cy, -uo, -vo, **obs_kw)

    # Structures: spheroid footprint (radius=b) at the SHIFTED location, ring faults.
    circ = plt.Circle((x0, y0), BASE.SPH_B, fill=False, color='0.1', lw=2.0, zorder=10)
    ax.add_patch(circ)
    T._faults(ax, BASE.SPH_DEPTH)

    for sta, row in T._sta_df.iterrows():
        highlight = sta in ('AXAS1', 'AXAS2')
        color = '#FFD700' if highlight else 'white'
        ax.plot(row['x'], row['y'], '^', ms=8 if highlight else 6, mfc=color, mec='k',
               mew=0.9, zorder=13)
        dx, dy = T.LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, T.STA_DISPLAY.get(sta, sta), fontsize=7,
               fontweight='bold', zorder=13)

    ax.set_xlim(T.X_START, T.X_END)
    ax.set_ylim(T.Y_START, T.Y_END)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('East [km]', fontsize=9)
    ax.set_ylabel('North [km]', fontsize=9)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8,
                      label='AXAS1 / AXAS2'),
        mlines.Line2D([], [], marker='^', color='w', mfc='white', mec='k', ms=6,
                      label='Other stations'),
        mlines.Line2D([], [], color='0.1', lw=2.0, label='Spheroid footprint (radius=b), shifted'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults'),
        mlines.Line2D([], [], color=MODEL_COLOR, lw=2.0, label='Modeled $\\sigma_1$ (Hudson/Yang)'),
        mlines.Line2D([], [], color=OBS_COLOR, lw=2.0, label='Observed $\\phi$ (Johnson et al. 2011)'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=7, framealpha=0.9)

    ax.set_title(
        'Rapidly inflating vertical prolate spheroid, shifted '
        f'{SHIFT_NORTH_KM:.1f} km N / {SHIFT_WEST_KM:.1f} km W of AXAS1-AXAS2 midpoint\n'
        f'depth {BASE.SPH_DEPTH-BASE.SPH_A:.1f}-{BASE.SPH_DEPTH+BASE.SPH_A:.1f} km, '
        f'b={BASE.SPH_B:.1f} km (assumed), dP=+{BASE.SPH_DP_PA/1e6:.0f} MPa (assumed) '
        f'vs. Johnson et al. (2011) {pre_lbl}',
        fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
