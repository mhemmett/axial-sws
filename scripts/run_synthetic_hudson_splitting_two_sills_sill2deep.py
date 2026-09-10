#!/usr/bin/env python3
"""
run_synthetic_hudson_splitting_two_sills_sill2deep.py

Variant of run_synthetic_hudson_splitting_two_sills.py, per explicit user request: make SILL2
(Kidiwela S2) deeper and inflate/deflate more, such that its deflation at eruption onset
(Scenario C) alone reproduces the real ~2.4 m of central-caldera subsidence observed at the
April 2015 eruption (Nooner & Chadwick). Per a further explicit follow-up, SILL2's horizontal
location was ALSO moved to the deep Kidiwela-S1 source's own (x0, y0) -- i.e. SILL2 is now
co-located (x, y, and z) with the Kidiwela-S1 deep source, not just deepened at its original S2
map location. SILL1 and the dike are untouched -- "for sill 2 only" per the user's instruction.

Depth choice: 3.33 km, matching SILL1/Kidiwela-S1's depth (the main, deep AMC) -- a physically
motivated "deeper" choice (same reservoir depth as the primary source), rather than an arbitrary
number, and consistent with the "deeper, hydraulically-connected magmatic system" framing from
this week's literature reading (Zhu et al. 2026).

Pressure-change magnitude: FIXED at the same 200 MPa magnitude as SILL1 (not re-derived per
depth), per explicit user follow-up -- "match sill1's dP order" rather than accept whatever huge
value (~2 GPa) the geodetically-calibrated mu_cal=0.72 GPa (mogi_stress_model.py's own Kidiwela
fit) would otherwise require at 3.33 km depth. Since the Mogi displacement formula scales as
dP/mu, holding dP fixed at 200 MPa and solving INSTEAD for the shear modulus that reproduces
-2.4 m of caldera-centre subsidence gives mu ~= 73 MPa (0.073 GPa) -- much softer than mu_cal,
consistent with treating the deeper source region as material close to melt (a partially molten
mush zone), which is physically defensible near a deep AMC. This mu is used ONLY for this
displacement calibration/sanity-check print -- the STRESS actually fed into the Hudson crack
model (mogi_stress_tensor_3d) is mu-independent (mu cancels: sigma ~ dP*R^3, see
mogi_stress_model.py's calibrate_mu() docstring), so the crack-opening physics is unaffected by
which mu value is used here.

Everything else (background stress, both fault-damage-zone crack sets, dike geometry/opening,
SILL1's own inflation/deflation and overprint, event subsample size, QC, period-scenario
pairing) is identical to run_synthetic_hudson_splitting_two_sills.py -- imported directly, not
duplicated.

Produces:
    synthetic_hudson_pre_syn_phidt_two_sills_sill2deep.csv
    synthetic_hudson_rose_plots_two_sills_sill2deep.pdf

Run with:
    python3 run_synthetic_hudson_splitting_two_sills_sill2deep.py
"""

import math
import os
import time

import synthetic_hudson_crack_scenarios as shc
from sws_forward_model import SPHERES
import mogi_stress_model as msm

# ── Reference point + target (real 2015 eruption-onset caldera subsidence) ───────────────────
CALDERA_X, CALDERA_Y = msm.CALDERA_X, msm.CALDERA_Y   # 8.0, 5.5 km -- same ref as msm's own calibration
TARGET_UZ_M = -2.4   # m, observed central-caldera subsidence at the April 2015 eruption onset

# ── SILL2 geometry: moved to the deep Kidiwela-S1 source location (lat/lon + depth) ───────────
_KIDIWELA_S1 = next(s for s in SPHERES if s['label'] == 'S1')
_KIDIWELA_S2 = next(s for s in SPHERES if s['label'] == 'S2')
NEW_SILL2_DEPTH_KM = _KIDIWELA_S1['d']   # 3.33 km -- the deep Kidiwela-S1 AMC depth (NOTE: this
                                         # is deeper than shc.SILL['d']=1.5 km, the shallow crack-
                                         # model sill1 -- picked instead of shc.SILL['d'] because
                                         # 1.5 km is barely deeper than sill2's original 1.25 km
                                         # and wouldn't read as a meaningful depth change; 3.33 km
                                         # is a real, already-used-in-this-repo deep-source depth)
# Per explicit user follow-up: SILL2 also moves horizontally to S1's own (x0, y0) -- the real
# deep-source Kidiwela lat/lon -- so SILL2 is now co-located (in x,y,z) with the Kidiwela-S1
# source, rather than sitting at S2's shallower location.
SILL2 = dict(x0=_KIDIWELA_S1['x0'], y0=_KIDIWELA_S1['y0'], d=NEW_SILL2_DEPTH_KM, R=shc.SILL['R'])

print(f'SILL1 (unchanged): x0={shc.SILL["x0"]:.2f}, y0={shc.SILL["y0"]:.2f}, '
      f'd={shc.SILL["d"]:.2f}, R={shc.SILL["R"]:.2f} km, dP={shc.SILL_DP_PA/1e6:.0f} MPa')
print(f'SILL2 (moved to Kidiwela S1 deep-source location): x0={SILL2["x0"]:.2f} '
      f'(was {_KIDIWELA_S2["x0"]:.2f}), y0={SILL2["y0"]:.2f} (was {_KIDIWELA_S2["y0"]:.2f}), '
      f'd={SILL2["d"]:.2f} km (was {_KIDIWELA_S2["d"]:.2f} km), R={SILL2["R"]:.2f} km')

# ── SILL2 dP magnitude: fixed at SILL1's own 200 MPa (per explicit user follow-up) ────────────
SILL2_DP_MAG_PA = shc.SILL_DP_PA   # 200 MPa, same magnitude as SILL1


def _uz_from_sill2(dP_pa, mu):
    dx = (CALDERA_X - SILL2['x0']) * 1e3
    dy = (CALDERA_Y - SILL2['y0']) * 1e3
    d = SILL2['d'] * 1e3
    R3 = (dx ** 2 + dy ** 2 + d ** 2) ** 1.5
    R_m = SILL2['R'] * 1e3
    dV = math.pi * R_m ** 3 * dP_pa / mu
    C = dV * (1 - shc.NU_FIX) / math.pi
    return C * d / R3


# Solve for the (softer) shear modulus that reproduces -2.4 m at this fixed dP, so the
# calibration check below is self-consistent (uz is linear in 1/mu -> solve from a unit-mu probe)
_uz_per_inv_mu = _uz_from_sill2(-SILL2_DP_MAG_PA, 1.0)
mu_required = _uz_per_inv_mu / TARGET_UZ_M

_uz_check = _uz_from_sill2(-SILL2_DP_MAG_PA, mu_required)
print(f'\nSILL2 |dP| fixed at {SILL2_DP_MAG_PA/1e6:.0f} MPa (matching SILL1). Required shear '
      f'modulus for -2.4 m caldera-centre subsidence: mu={mu_required/1e6:.1f} MPa '
      f'({mu_required/1e9:.4f} GPa) -> checked uz = {_uz_check:.2f} m (target {TARGET_UZ_M:.2f} m)')
print(f'(For reference, mogi_stress_model.py\'s own geodetically-calibrated mu_cal = '
      f'{msm.calibrate_mu()/1e9:.3f} GPa; this softer value reflects treating the deep source '
      f'region as closer to melt. Stress fed into the Hudson crack model is mu-independent, so '
      f'the crack-opening physics is unaffected by this choice.)')

SILL2_INFLATE_DP_PA = SILL2_DP_MAG_PA      # Scenario B: inflating
SILL2_DEFLATE_DP_PA = -SILL2_DP_MAG_PA     # Scenario C: deflating (matches -2.4 m target)

_orig_total_stress_and_crack_sets = shc.total_stress_and_crack_sets


def total_stress_and_crack_sets_two_sills_deep(scenario, x_km, y_km, z_km, dike_interps=None):
    sigma, extra = _orig_total_stress_and_crack_sets(scenario, x_km, y_km, z_km, dike_interps)
    if scenario == 'B':
        sigma = sigma + shc.mogi_stress_tensor_3d(x_km, y_km, z_km, SILL2, SILL2_INFLATE_DP_PA,
                                                  shc.MU_BG, shc.NU_FIX)
    elif scenario == 'C':
        sigma = sigma + shc.mogi_stress_tensor_3d(x_km, y_km, z_km, SILL2, SILL2_DEFLATE_DP_PA,
                                                  shc.MU_BG, shc.NU_FIX)
    if scenario in ('B', 'C'):
        if math.hypot(x_km - SILL2['x0'], y_km - SILL2['y0']) <= SILL2['R'] + shc.SILL_OVERPRINT_MARGIN_KM:
            extra = []
    return sigma, extra


shc.total_stress_and_crack_sets = total_stress_and_crack_sets_two_sills_deep

import run_synthetic_hudson_splitting as base

base.HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    dfs = base.load_real_events()

    print('\nPrecomputing FMM travel-time fields for all stations...')
    tracer = base.BaillardRayTracer()
    for sta, (sx, sy) in base.STATION_XY.items():
        if sta in base.STATION_ORDER:
            tracer.precompute_station(sta, sx, sy)

    print('Building dike stress interpolators (Scenario C)...')
    dike_interps = base.build_dike_stress_interpolators(base.DIKE_XN, base.DIKE_YN)

    n_hats = base.fibonacci_sphere(base.N_CRACKS)

    results = []
    for sta in base.STATION_ORDER:
        for label, _t0, _t1, sc, _color in base.PERIOD_SCENARIOS:
            df = dfs[sta][label]
            if len(df) == 0:
                print(f'  {sta} / {label} (scenario {sc}): 0 events, skipping')
                continue
            x_arr, y_arr = base.ll2xy_local(df['latitude'].values, df['longitude'].values)
            z_arr = df['depth'].values
            eid_arr = df['event_id'].values if 'event_id' in df.columns else base.np.arange(len(df))

            t0 = time.time()
            n_traced = 0
            for i in range(len(df)):
                ex, ey, ez = float(x_arr[i]), float(y_arr[i]), float(z_arr[i])
                if ez < 0 or ez > base.Z_MAX:
                    continue
                try:
                    ray = tracer.trace(sta, ex, ey, ez, n_pts=base.N_RAY_PTS)
                except RuntimeError:
                    continue
                n_traced += 1
                phi_s, dt_s = base.integrate_synthetic_splitting(
                    sc, ray, dike_interps=dike_interps if sc == 'C' else None,
                    n_hats=n_hats)
                results.append(dict(event_id=eid_arr[i], station=sta, period=label,
                                    scenario=sc, phi_synth=phi_s, dt_synth=dt_s))
                if (i + 1) % 50 == 0:
                    print(f'  {sta}/{label} (scenario {sc}): {i+1}/{len(df)}  '
                          f'{time.time()-t0:.0f}s', end='\r', flush=True)
            print(f'  {sta} / {label} (scenario {sc}): {n_traced}/{len(df)} events traced '
                  f'successfully ({time.time()-t0:.0f}s)')

    out_csv = os.path.join(base.HERE, 'synthetic_hudson_pre_syn_phidt_two_sills_sill2deep.csv')
    df_all = base.pd.DataFrame(results)
    df_all.to_csv(out_csv, index=False)
    print(f'Saved {out_csv} ({len(df_all):,} rows)')

    out_pdf = os.path.join(base.HERE, 'synthetic_hudson_rose_plots_two_sills_sill2deep.pdf')
    with base.PdfPages(out_pdf) as pdf:
        fig, axes = base.plt.subplots(
            len(base.STATION_ORDER), len(base.PERIOD_SCENARIOS),
            figsize=(3.2 * len(base.PERIOD_SCENARIOS), 3.2 * len(base.STATION_ORDER)),
            subplot_kw=dict(projection='polar'))
        for ri, sta in enumerate(base.STATION_ORDER):
            for ci, (label, _t0, _t1, sc, color) in enumerate(base.PERIOD_SCENARIOS):
                ax = axes[ri, ci]
                sub = df_all[(df_all['station'] == sta) & (df_all['period'] == label)]
                phi_az = sub['phi_synth'].values % 180.0
                base._draw_rose(ax, phi_az, base.np.ones(len(sub)), color)
                title = (f'{label} (Scenario {sc})\nN={len(sub):,}' if ri == 0
                         else f'N={len(sub):,}')
                ax.set_title(title, fontsize=8, fontweight='bold')
                if ci == 0:
                    ax.text(-0.25, 0.5, sta, fontsize=10, fontweight='bold',
                            ha='center', va='center', transform=ax.transAxes)
        fig.suptitle(f'Synthetic Hudson-crack splitting, TWO SILLS (sill2 moved to Kidiwela-S1 '
                     f'deep-source location, {SILL2["d"]:.2f} km, |dP|={SILL2_DP_MAG_PA/1e6:.0f} '
                     f'MPa, softer mu calibrated to -2.4 m caldera-centre subsidence at eruption '
                     f'onset): Pre-eruption = '
                     f'Scenario B (2 sills), Syn-eruption = Scenario C (dike + both sills '
                     f'deflated)\n(subsample, up to {base.N_EVENTS_PER_STATION_PER_PERIOD}/'
                     f'station/period)',
                     fontsize=9, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        base.plt.close(fig)
    print(f'\nSaved {out_pdf}')
