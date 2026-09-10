#!/usr/bin/env python3
"""
run_synthetic_hudson_splitting_two_sills.py

Variant of run_synthetic_hudson_splitting.py with a SECOND sill added to Scenario B
(pre-eruption), per explicit user request: same size (R) and inflation pressure as the
original sill, located at the Kidiwela S2 source location (sws_forward_model.py's SPHERES:
x0=7.53, y0=6.60, d=1.25 km). Syn-eruption (Scenario C) is unaffected by SILL2 (only the
original sill is present there, deflated, same as the single-sill model).

Everything else (background stress, both fault-damage-zone crack sets, dike geometry/opening,
event subsample size, QC, period-scenario pairing) is identical to
run_synthetic_hudson_splitting.py -- imported directly, not duplicated, except for the
total_stress_and_crack_sets override (adds SILL2's Mogi stress for Scenario B) applied to the
shared synthetic_hudson_crack_scenarios module before running.

Produces:
    synthetic_hudson_pre_syn_phidt_two_sills.csv
    synthetic_hudson_rose_plots_two_sills.pdf

Run with:
    python3 run_synthetic_hudson_splitting_two_sills.py
"""

import os
import time

import synthetic_hudson_crack_scenarios as shc
from sws_forward_model import SPHERES

_KIDIWELA_S2 = next(s for s in SPHERES if s['label'] == 'S2')
SILL2 = dict(x0=_KIDIWELA_S2['x0'], y0=_KIDIWELA_S2['y0'], d=_KIDIWELA_S2['d'], R=shc.SILL['R'])
print(f'SILL2 (Kidiwela S2 location): x0={SILL2["x0"]:.2f}, y0={SILL2["y0"]:.2f}, '
      f'd={SILL2["d"]:.2f}, R={SILL2["R"]:.2f} km')

_orig_total_stress_and_crack_sets = shc.total_stress_and_crack_sets


def total_stress_and_crack_sets_two_sills(scenario, x_km, y_km, z_km, dike_interps=None):
    sigma, extra = _orig_total_stress_and_crack_sets(scenario, x_km, y_km, z_km, dike_interps)
    if scenario == 'B':
        sigma = sigma + shc.mogi_stress_tensor_3d(x_km, y_km, z_km, SILL2, shc.SILL_DP_PA,
                                                  shc.MU_BG, shc.NU_FIX)
    elif scenario == 'C':
        # SILL2's crack population stays present (not removed) but now DEFLATED, per explicit
        # user request -- same treatment sill1 already gets in Scenario C.
        sigma = sigma + shc.mogi_stress_tensor_3d(x_km, y_km, z_km, SILL2, -shc.SILL_DP_PA,
                                                  shc.MU_BG, shc.NU_FIX)
    if scenario in ('B', 'C'):
        # SILL2 overprint: same "sill overprints fault" rule as the base module applies to
        # SILL1 -- suppress the fault's fixed crack set within SILL2's own overprint radius
        # too, even if the base call above already included it (point may be near SILL2 but
        # not near SILL1/the dike). Now applies in Scenario C too, since SILL2 is present
        # (deflated) there as well.
        import math
        if math.hypot(x_km - SILL2['x0'], y_km - SILL2['y0']) <= SILL2['R'] + shc.SILL_OVERPRINT_MARGIN_KM:
            extra = []
    return sigma, extra


shc.total_stress_and_crack_sets = total_stress_and_crack_sets_two_sills

import run_synthetic_hudson_splitting as base

base.HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    dfs = base.load_real_events()

    print('\nPrecomputing FMM travel-time fields for all 5 stations...')
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

    out_csv = os.path.join(base.HERE, 'synthetic_hudson_pre_syn_phidt_two_sills.csv')
    df_all = base.pd.DataFrame(results)
    df_all.to_csv(out_csv, index=False)
    print(f'Saved {out_csv} ({len(df_all):,} rows)')

    out_pdf = os.path.join(base.HERE, 'synthetic_hudson_rose_plots_two_sills.pdf')
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
        fig.suptitle('Synthetic Hudson-crack splitting (TWO SILLS: original + Kidiwela S2 '
                     'location): Pre-eruption = Scenario B (2 sills), Syn-eruption = Scenario '
                     'C (dike + deflated original sill)\n'
                     f'(subsample, up to {base.N_EVENTS_PER_STATION_PER_PERIOD}/station/period)',
                     fontsize=10, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        base.plt.close(fig)
    print(f'\nSaved {out_pdf}')
