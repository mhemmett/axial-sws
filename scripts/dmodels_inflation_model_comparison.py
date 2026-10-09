#!/usr/bin/env python3
"""
dmodels_inflation_model_comparison.py

Side-by-side comparison of the inflation sources behind the rebuilt 2015/2026 dike test
(dmodels_dike_comparison.py): the Kidiwela two-sphere Mogi model, Baillard's single Yang
prolate spheroid, and that spheroid with its axis direction reversed (its orientation is
ambiguous in scripts/baillard_simple_model.py).

Run dmodels_axial/run_axial_dmodels('<model>') for each model first. Uses the same observed
change (Grade 3, Jan-Apr 2015 vs equal-inflation period 5) and the same superposition code.

Outputs (dmodels_axial/outputs/model_comparison/):
    station_comparison.csv   per station and extension rate: observed and modeled change and
                             absolute sigma_Hmax azimuths for every model
    summary.csv              per model and extension rate: RMS misfits, sign agreement,
                             best-fitting dike opening
    station_tables.md        the change and both states, observed (± standard error) vs
                             each model, with no extension
    model_comparison.png

Run with:
    python3 dmodels_inflation_model_comparison.py [--p5-start 2021-01-01]
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dmodels_dike_comparison as ddc

# The Grade-3 loader module asks for Arial, which is not installed everywhere
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

MODELS = ['two_sphere', 'yang', 'yang_reversed']
LABELS = {'two_sphere': 'Two Mogi spheres (Kidiwela)',
          'yang': 'Single Yang spheroid (Baillard)',
          'yang_reversed': 'Yang, axis reversed'}
COLORS = {'two_sphere': '#0072B2', 'yang': '#D55E00', 'yang_reversed': '#CC79A7'}
OUT_DIR = os.path.join(ddc.OUTPUTS_DIR, 'model_comparison')
PLOT_RATE = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p5-start', default=ddc.P5_START_DEFAULT)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    obs = ddc.observed_change(args.p5_start)
    d_obs = obs['dphi_obs'].values
    models = {m: ddc.load_model(os.path.join(ddc.OUTPUTS_DIR, m)) for m in MODELS}

    station_rows, summary_rows, scans = [], [], {}
    for m, (E, sc, rates) in models.items():
        for rate in rates:
            az15, az26, _ = ddc.model_azimuths(E, sc, rate, 1.0)
            d_mod = ddc.wrap90(az26 - az15)
            scan = np.array([ddc.rms(d_obs - ddc.wrap90(ddc.model_azimuths(E, sc, rate, k)[1]
                                                        - az15))
                             for k in ddc.OPENING_SCALES])
            if np.isclose(rate, PLOT_RATE):
                scans[m] = scan
            summary_rows.append(dict(
                model=m, rate_per_yr=rate,
                rms_change_2m_dike=ddc.rms(d_obs - d_mod),
                rms_change_null=ddc.rms(d_obs),
                sign_agree_of_6=int(np.sum(np.sign(d_mod) == np.sign(d_obs))),
                best_opening_m=2.0 * ddc.OPENING_SCALES[int(np.argmin(scan))],
                rms_change_best_opening=float(scan.min()),
                rms_abs_2015=ddc.rms(ddc.wrap90(obs.phi_pre.values - az15)),
                rms_abs_2026=ddc.rms(ddc.wrap90(obs.phi_p5.values - az26))))
            for i, sta in enumerate(ddc.STATIONS):
                station_rows.append(dict(
                    model=m, rate_per_yr=rate, station=sta,
                    phi_pre_obs=obs.loc[sta, 'phi_pre'], phi_pre_se=obs.loc[sta, 'se_pre'],
                    phi_p5_obs=obs.loc[sta, 'phi_p5'], phi_p5_se=obs.loc[sta, 'se_p5'],
                    dphi_obs=d_obs[i], dphi_obs_se=obs.loc[sta, 'dphi_obs_se'],
                    az_2015=az15[i], az_2026=az26[i], dphi_model=d_mod[i],
                    misfit_2015=float(ddc.wrap90(az15[i] - obs.loc[sta, 'phi_pre'])),
                    misfit_2026=float(ddc.wrap90(az26[i] - obs.loc[sta, 'phi_p5'])),
                    misfit_dphi=float(ddc.wrap90(d_mod[i] - d_obs[i]))))

    stations = pd.DataFrame(station_rows)
    summary = pd.DataFrame(summary_rows)
    stations.to_csv(os.path.join(OUT_DIR, 'station_comparison.csv'), index=False,
                    float_format='%.6g')
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, float_format='%.6g')

    pd.set_option('display.width', 200)
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    md = station_tables(stations, summary, obs, args.p5_start)
    with open(os.path.join(OUT_DIR, 'station_tables.md'), 'w') as fh:
        fh.write(md)
    print('\n' + md)

    plot(stations, obs, scans, args.p5_start)


def station_tables(stations, summary, obs, p5_start):
    """Markdown tables at PLOT_RATE: the change, then each state, observed (circular mean
    +- its standard error) against every model. Model cells show the modeled sigma_Hmax
    azimuth with its axial misfit (model - observed, wrapped to +-90) in parentheses."""
    sub = stations[np.isclose(stations.rate_per_yr, PLOT_RATE)]
    summ = summary[np.isclose(summary.rate_per_yr, PLOT_RATE)].set_index('model')
    cell = {m: sub[sub.model == m].set_index('station') for m in MODELS}
    head = '| Station | Observed | ' + ' | '.join(MODELS) + ' |\n|---|---|' + '---|' * len(MODELS)

    def table(obs_col, se_col, model_col, misfit_col, rms_col, signed):
        fmt = '{:+.1f}°' if signed else '{:.1f}°'
        rows = [head]
        for sta in ddc.STATIONS:
            o = fmt.format(obs.loc[sta, obs_col]) + f' ± {obs.loc[sta, se_col]:.1f}°'
            ms = [fmt.format(cell[m].loc[sta, model_col])
                  + f' ({cell[m].loc[sta, misfit_col]:+.1f})' for m in MODELS]
            rows.append(f'| {sta} | {o} | ' + ' | '.join(ms) + ' |')
        rows.append('| **RMS misfit** | | '
                    + ' | '.join(f'**{summ.loc[m, rms_col]:.1f}°**' for m in MODELS) + ' |')
        return '\n'.join(rows)

    return (
        f'Grade 3, circular mean ± standard error; pre-eruption = Jan-Apr 2015, present = '
        f'equal-inflation period 5 (from {p5_start}). Models: 2 m dike, no extension. '
        f'Model cells: modeled value (model - observed, wrapped to ±90°).\n\n'
        f'### Change, present - pre-eruption 2015 (Δφ)\n\n'
        + table('dphi_obs', 'dphi_obs_se', 'dphi_model', 'misfit_dphi',
                'rms_change_2m_dike', True)
        + f'\n\n"Nothing changed" scores {ddc.rms(obs.dphi_obs.values):.1f}° RMS.\n\n'
        f'### Pre-eruption 2015: observed φ vs modeled σHmax (pre_2015)\n\n'
        + table('phi_pre', 'se_pre', 'az_2015', 'misfit_2015', 'rms_abs_2015', False)
        + '\n\n### Present (period 5): observed φ vs modeled σHmax (pre_2026)\n\n'
        + table('phi_p5', 'se_p5', 'az_2026', 'misfit_2026', 'rms_abs_2026', False)
        + '\n')


def plot(stations, obs, scans, p5_start):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    sub = stations[np.isclose(stations.rate_per_yr, PLOT_RATE)]
    x = np.arange(len(ddc.STATIONS))
    width = 0.2

    ax = axes[0]
    ax.bar(x - 1.5 * width, obs['dphi_obs'], width, color='0.35', label='observed',
           yerr=obs['dphi_obs_se'], capsize=2)
    for k, m in enumerate(MODELS):
        vals = sub[sub.model == m].set_index('station').loc[ddc.STATIONS, 'dphi_model']
        ax.bar(x + (k - 0.5) * width, vals, width, color=COLORS[m], label=LABELS[m])
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ddc.STATIONS)
    ax.set_ylabel('$\\Delta$ azimuth, 2026 $-$ 2015 (deg)')
    ax.set_title('Change: observed $\\Delta\\phi$ vs modeled (2 m dike, no extension)')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, obs['phi_pre'], 'ks', ms=8, mfc='none', label='observed pre-2015')
    ax.plot(x, obs['phi_p5'], 'k^', ms=8, mfc='none', label='observed period 5')
    for m in MODELS:
        s = sub[sub.model == m].set_index('station').loc[ddc.STATIONS]
        ax.plot(x, s['az_2015'], 's', color=COLORS[m], ms=6)
        ax.plot(x, s['az_2026'], '^', color=COLORS[m], ms=6, label=LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(ddc.STATIONS)
    ax.set_ylim(0, 180)
    ax.set_yticks(range(0, 181, 30))
    ax.set_ylabel('azimuth (deg); squares 2015, triangles 2026')
    ax.set_title('State: fast direction vs modeled $\\sigma_{Hmax}$')
    ax.legend(fontsize=7, loc='lower left')

    ax = axes[2]
    ax.axhline(ddc.rms(obs['dphi_obs'].values), color='k', ls=':', label='null: no change')
    for m in MODELS:
        ax.plot(2.0 * ddc.OPENING_SCALES, scans[m], color=COLORS[m], label=LABELS[m])
    ax.set_xlabel('2015 dike opening (m)')
    ax.set_ylabel('RMS(observed $-$ modeled $\\Delta\\phi$), 6 stations (deg)')
    ax.set_title('Dike opening scan (no extension)')
    ax.legend(fontsize=8)

    fig.suptitle(f'Inflation source comparison: Grade 3 pre-eruption 2015 vs period 5 '
                 f'(from {p5_start})')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'model_comparison.png')
    fig.savefig(path, dpi=150)
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
