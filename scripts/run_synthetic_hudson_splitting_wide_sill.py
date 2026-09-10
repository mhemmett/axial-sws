#!/usr/bin/env python3
"""
run_synthetic_hudson_splitting_wide_sill.py

Variant of run_synthetic_hudson_splitting.py with the sill (Scenario B) parameters modified
per explicit user request:
    - Radius doubled: 0.25 km -> 0.5 km
    - Moved 0.5 km further west: x0 = 9.28 -> 8.78 km

Everything else (background stress, both fault-damage-zone crack sets, dike geometry/opening,
event subsample size, QC, period-scenario pairing) is identical to
run_synthetic_hudson_splitting.py -- imported directly, not duplicated, except for the sill
override applied to the shared synthetic_hudson_crack_scenarios.SILL dict before running.

Produces:
    synthetic_hudson_pre_syn_phidt_wide_sill.csv
    synthetic_hudson_rose_plots_wide_sill.pdf

Run with:
    python3 run_synthetic_hudson_splitting_wide_sill.py
"""

import os

import synthetic_hudson_crack_scenarios as shc

print(f'Original sill: x0={shc.SILL["x0"]:.2f}, y0={shc.SILL["y0"]:.2f}, R={shc.SILL["R"]:.2f} km')
shc.SILL['R'] *= 2.0
shc.SILL['x0'] -= 0.5
print(f'Modified sill: x0={shc.SILL["x0"]:.2f}, y0={shc.SILL["y0"]:.2f}, R={shc.SILL["R"]:.2f} km')

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

            import time
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

    out_csv = os.path.join(base.HERE, 'synthetic_hudson_pre_syn_phidt_wide_sill.csv')
    df_all = base.pd.DataFrame(results)
    df_all.to_csv(out_csv, index=False)
    print(f'Saved {out_csv} ({len(df_all):,} rows)')

    out_pdf = os.path.join(base.HERE, 'synthetic_hudson_rose_plots_wide_sill.pdf')
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
        fig.suptitle('Synthetic Hudson-crack splitting (WIDE SILL: R=0.5km, x0-0.5km): '
                     'Pre-eruption = Scenario B (sill), Syn-eruption = Scenario C (dike)\n'
                     f'(subsample, up to {base.N_EVENTS_PER_STATION_PER_PERIOD}/station/period)',
                     fontsize=10, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        base.plt.close(fig)
    print(f'\nSaved {out_pdf}')
