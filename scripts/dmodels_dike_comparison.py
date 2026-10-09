#!/usr/bin/env python3
"""
dmodels_dike_comparison.py

Can the presence of the 2015 dike explain how the Grade-3 fast direction changed between the
2015 pre-eruption state and the most recent equal-inflation period?

Observed, per station (Grade 3, circular mean of phi mod 180):
    pre-eruption  t < ERUPTION_START (record starts 2015-01-22, so Jan-Apr 2015)
    period 5      t >= P5_START, the last of the 5 equal-inflation (Central Caldera BOTPT)
                  bins in rose_7period_regions_windowcheck_grade3_ccal_periods.py
    dphi_obs = period 5 - pre, wrapped to [-90, 90)

Modeled, from dmodels_axial/run_axial_dmodels.m for one inflation model (--model:
two_sphere, yang or yang_reversed; station_components.csv holds the strain
of each unit source at each station, so scenarios are rebuilt by superposition):
    pre_2015         inflation (2.4 m) + extension (rate * 4.05 yr)
    pre_2026         inflation (2.6 m) + extension (rate * 11.05 yr) + 2015 dike
    pre_2026_nodike  the same without the dike (control)
    azimuth of the most compressive horizontal stress, compared with the fast direction.

With no extension the inflation sources only change in amplitude between the two states,
which does not rotate stress axes, so the modeled change is then entirely the dike.
The dike opening is also scanned (strain is linear in opening) to find the opening that
best fits, and every fit is compared with the null "nothing changed" (dphi = 0).

P5_START: the BOTPT file that defines the period boundaries
(data/bpr_detided_seafloor_depth_ccal_*.csv) is not in data/, so the start is taken from the
published rose figure label ("Jan 2021 - present") and its sensitivity is reported.

Outputs (dmodels_axial/outputs/<model>/, next to the model tables):
    dmodels_dike_comparison.csv
    dmodels_dike_comparison.png

Run with:
    python3 dmodels_dike_comparison.py [--model two_sphere] [--p5-start 2021-01-01]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rose_7period_regions_windowcheck_grade3 as g3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..')
# Model station tables and this script's outputs share dmodels_axial/outputs/<model>/
OUTPUTS_DIR = os.path.join(REPO, 'dmodels_axial', 'outputs')

# The Grade-3 inputs were delivered under data/ rather than the loader's default locations
DATA_DIR = os.path.join(REPO, 'data')
g3.WC_TRANSFER_DIR = os.path.join(DATA_DIR, 'mfast_maxdt_windowcheck_pipeline_transfer')
g3.AXEC2_2015_2021_WC_CSV = os.path.join(
    DATA_DIR, 'scripts',
    'production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results',
    'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_combined.csv')
g3.AXEC2_2015_2021_META_CSV = os.path.join(
    DATA_DIR, 'scripts', 'raw_axec2_all_batches_mfast_filters_data',
    'raw_axec2_all_batches_mfast_filters_metadata.csv')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
P5_START_DEFAULT = '2021-01-01'
P5_SENSITIVITY = ['2020-10-01', '2021-04-01']
NU = 0.25
OPENING_SCALES = np.linspace(0, 5, 101)   # x the 2 m reference opening


def wrap90(a):
    return (np.asarray(a) + 90.0) % 180.0 - 90.0


def shmax_azimuth(exx, eyy, exy):
    """Most compressive horizontal axis, same convention as principal_compression.m and
    deformation_util.compute_sigma1_2d (the 2-D Hooke prefactors cancel in the angle
    except through exx - eyy, which lambda does not enter)."""
    theta = 0.5 * np.arctan2(2 * exy, exx - eyy)
    return np.degrees(-theta) % 180.0


def observed_change(p5_start):
    t_p5 = pd.Timestamp(p5_start, tz='UTC')
    rows = []
    for sta in STATIONS:
        df = g3.apply_grade(g3.load_station_raw(sta), g3.GRADE)
        pre = df[df['t'] < g3.ERUPTION_START]['phi_az'].values
        p5 = df[df['t'] >= t_p5]['phi_az'].values
        m_pre, se_pre = g3._circular_mean_and_se_deg(pre)
        m_p5, se_p5 = g3._circular_mean_and_se_deg(p5)
        rows.append(dict(station=sta, n_pre=len(pre), phi_pre=m_pre, se_pre=se_pre,
                         n_p5=len(p5), phi_p5=m_p5, se_p5=se_p5,
                         dphi_obs=float(wrap90(m_p5 - m_pre)),
                         dphi_obs_se=float(np.hypot(se_pre, se_p5))))
    return pd.DataFrame(rows).set_index('station')


def load_model(model_dir):
    comp = pd.read_csv(os.path.join(model_dir, 'station_components.csv'))
    man = pd.read_csv(os.path.join(model_dir, 'manifest.csv'))
    E = {c: comp[comp.component == c].set_index('station').loc[
            STATIONS, ['exx_per_unit', 'eyy_per_unit', 'exy_per_unit']].values
         for c in ('infl', 'dike', 'ext')}
    sc = {s: man[man.scenario == s].iloc[0] for s in ('pre_2015', 'pre_2026')}
    rates = np.sort(man['rate_per_yr'].unique())
    return E, sc, rates


def model_azimuths(E, sc, rate, opening_scale):
    s15, s26 = sc['pre_2015'], sc['pre_2026']
    e15 = s15.infl_scale * E['infl'] + rate * s15.years * E['ext']
    e26_nodike = s26.infl_scale * E['infl'] + rate * s26.years * E['ext']
    e26 = e26_nodike + opening_scale * E['dike']
    return shmax_azimuth(*e15.T), shmax_azimuth(*e26.T), shmax_azimuth(*e26_nodike.T)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='two_sphere')
    ap.add_argument('--p5-start', default=P5_START_DEFAULT)
    args = ap.parse_args()
    out_dir = os.path.join(OUTPUTS_DIR, args.model)

    obs = observed_change(args.p5_start)
    E, sc, rates = load_model(out_dir)
    print(f'Inflation model: {args.model}')
    d_obs = obs['dphi_obs'].values

    print(f'Observed (Grade 3), pre-eruption vs period 5 from {args.p5_start}:')
    print(obs.round(2).to_string())
    print(f'\nRMS of observed change (the "nothing changed" null): {rms(d_obs):.1f} deg')

    rows = []
    for rate in rates:
        az15, az26, az26n = model_azimuths(E, sc, rate, 1.0)
        d_dike = wrap90(az26 - az15)
        d_nodike = wrap90(az26n - az15)
        scan = [rms(d_obs - wrap90(model_azimuths(E, sc, rate, k)[1] - az15))
                for k in OPENING_SCALES]
        k_best = OPENING_SCALES[int(np.argmin(scan))]
        for i, sta in enumerate(STATIONS):
            rows.append(dict(rate_per_yr=rate, station=sta,
                             phi_pre_obs=obs.loc[sta, 'phi_pre'], phi_p5_obs=obs.loc[sta, 'phi_p5'],
                             dphi_obs=d_obs[i], dphi_obs_se=obs.loc[sta, 'dphi_obs_se'],
                             az_2015=az15[i], az_2026=az26[i], az_2026_nodike=az26n[i],
                             dphi_model_dike=d_dike[i], dphi_model_nodike=d_nodike[i]))
        print(f'\nextension {rate:.0e}/yr:'
              f'  RMS(obs - dike model) {rms(d_obs - d_dike):5.1f}'
              f' | no dike {rms(d_obs - d_nodike):5.1f}'
              f' | null {rms(d_obs):5.1f}'
              f' | sign agrees {int(np.sum(np.sign(d_dike) == np.sign(d_obs)))}/6'
              f' | best opening {2.0 * k_best:.1f} m (RMS {min(scan):.1f})'
              f' | absolute misfit 2015 {rms(wrap90(obs.phi_pre.values - az15)):5.1f},'
              f' 2026 {rms(wrap90(obs.phi_p5.values - az26)):5.1f}')
        print('   dike model dphi: ' + '  '.join(f'{s} {d:+6.1f}' for s, d in zip(STATIONS, d_dike)))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, 'dmodels_dike_comparison.csv'), index=False)

    for alt in P5_SENSITIVITY:
        d_alt = observed_change(alt)['dphi_obs']
        print(f'\nP5 start {alt}: dphi_obs ' +
              '  '.join(f'{s} {d:+6.1f}' for s, d in d_alt.items()))

    plot(out, obs, E, sc, rates, args.p5_start, args.model, out_dir)


def plot(out, obs, E, sc, rates, p5_start, model, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    show = [0.0, 1e-5, 3e-5]
    colors = ['#0072B2', '#E69F00', '#CC79A7']
    lim = 40
    ax1.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax1.axhline(0, color='0.7', lw=0.6)
    ax1.axvline(0, color='0.7', lw=0.6)
    for rate, col in zip(show, colors):
        sub = out[np.isclose(out.rate_per_yr, rate)]
        ax1.errorbar(sub.dphi_obs, sub.dphi_model_dike, xerr=sub.dphi_obs_se, fmt='o',
                     color=col, ms=6, capsize=2, label=f'extension {rate:.0e}/yr')
        if rate == 0:
            for _, r in sub.iterrows():
                ax1.annotate(r.station, (r.dphi_obs, r.dphi_model_dike), fontsize=8,
                             xytext=(4, 4), textcoords='offset points')
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel('observed $\\Delta\\phi$, period 5 $-$ pre-2015 (deg)')
    ax1.set_ylabel('modeled $\\Delta$ azimuth, pre_2026 $-$ pre_2015 (deg)')
    ax1.set_title('Change per station (2 m dike)')
    ax1.legend(fontsize=8, loc='upper left')

    d_obs = obs['dphi_obs'].values
    ax2.axhline(rms(d_obs), color='k', ls=':', label='null: no change')
    for rate, col in zip(rates, plt.cm.viridis(np.linspace(0, 0.9, len(rates)))):
        az15 = model_azimuths(E, sc, rate, 0)[0]
        scan = [rms(d_obs - wrap90(model_azimuths(E, sc, rate, k)[1] - az15))
                for k in OPENING_SCALES]
        ax2.plot(2.0 * OPENING_SCALES, scan, color=col, label=f'{rate:.0e}/yr')
    ax2.set_xlabel('2015 dike opening (m)')
    ax2.set_ylabel('RMS(observed $-$ modeled $\\Delta\\phi$) over 6 stations (deg)')
    ax2.set_title('How much dike does the change want?')
    ax2.legend(fontsize=8, title='extension rate')
    fig.suptitle(f'Grade 3: pre-eruption 2015 vs period 5 (from {p5_start}), '
                 f'{model.replace("_", " ")} inflation')
    fig.tight_layout()
    path = os.path.join(out_dir, 'dmodels_dike_comparison.png')
    fig.savefig(path, dpi=150)
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
