"""
Reproducible figure-generation script for the AXEC2 agentified-SWS pilot
(Stage 6: sws-plotting-interpretation).

Regenerates the three pilot figures from the Stage-5 final-pass CSV:
    - axec2_pilot_rose.png
    - axec2_pilot_polar_density.png
    - axec2_pilot_temporal.png

Input: sws_agent/axec2_pilot_splitting_final_passed.csv
       (N=46, final_pass=True rows only, produced by Stage 5 / sws-final-filtering)

Conventions mirrored from scripts/build_production_rose_plots_axec2_qw05.py
(production rose-plot script, NOT modified by this script):
    - phi_az = phi % 180.0                (axial wrap; raw 'phi' column is (-90, 90])
    - doubled-angle histogram trick for axial (180 deg symmetric) rose diagrams,
      36 angular bins, theta zero at North, clockwise direction (see _draw_rose,
      build_production_rose_plots_axec2_qw05.py L129-158)
    - axial circular mean/spread computed via 2*phi doubling (Mardia axial statistics),
      NOT a naive linear mean of phi

ROSE WEIGHTING MODE (resolves Stage-6 auditor Minor finding #9 — made explicit here):
    This script uses UNWEIGHTED (raw event count per angular bin) rose weighting,
    i.e. weights = np.ones(N), NOT |dt|-weighted. This matches the
    "..._qw05_unweighted.pdf" branch of build_production_rose_plots_axec2_qw05.py's
    two-variant output (dt_weighted=False), not its |dt|-weighted companion.
    To switch to |dt|-weighted rose bars, set ROSE_DT_WEIGHTED = True below.

No changes made to swspy/, splitting_functions.py, or any file outside sws_agent/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'axec2_pilot_splitting_final_passed.csv')

STATION = 'AXEC2'
NBINS = 36

# --- Rose weighting mode (see module docstring) ---
ROSE_DT_WEIGHTED = False   # False = unweighted (event count per bin); True = |dt|-weighted

# Mandatory pre-eruption-snapshot caveat (this pilot's 46 events span only
# ~3.5 days, 2015-01-22 to 2015-01-25, ~3 months before the 2015-04-24 eruption
# onset used elsewhere in this repo — see rose_plots_temporal.py). No eruption
# crossing, no pre/syn/post comparison, no temporal-anisotropy inference is
# licensed by this dataset.
CAVEAT_TITLE = ("PILOT SNAPSHOT ONLY — NOT AN ERUPTION-CYCLE RESULT\n"
                 "~3.5 days (2015-01-22 to 2015-01-25), ~3 months BEFORE the "
                 "2015-04-24 eruption onset. No eruption crossing; no "
                 "temporal/eruption-cycle inference is licensed by this data.")


def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df[df['final_pass'] == True].copy()
    df['t'] = pd.to_datetime(df['datetime'], utc=True)
    df['phi_az'] = df['phi'] % 180.0
    return df


def axial_stats(phi_raw_deg):
    """Axial (2*phi-doubling) circular mean and spread, per Stage 5/6 convention."""
    phi_rad = np.deg2rad(phi_raw_deg.values)
    c = np.mean(np.cos(2 * phi_rad))
    s = np.mean(np.sin(2 * phi_rad))
    R = np.hypot(c, s)
    mean_2phi = np.arctan2(s, c)
    mean_phi = np.rad2deg(mean_2phi) / 2.0
    # wrap mean_phi into (-90, 90]
    if mean_phi <= -90:
        mean_phi += 180
    elif mean_phi > 90:
        mean_phi -= 180
    spread_deg = np.rad2deg(np.sqrt(-2 * np.log(R))) / 2.0 if R > 0 else np.nan  # /2: std of 2*phi -> std of phi
    return mean_phi, spread_deg, R, c, s


def _draw_rose(ax, phi_az_vals, weights, color):
    """Doubled-angle rose, verbatim convention from
    build_production_rose_plots_axec2_qw05.py::_draw_rose."""
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('none')
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)

    doubled_angles = []
    doubled_weights = []
    for phi, w in zip(phi_az_vals, weights):
        p = float(phi) % 360.0
        doubled_angles.extend([np.deg2rad(p), np.deg2rad((p + 180.0) % 360.0)])
        doubled_weights.extend([w, w])

    doubled_angles = np.array(doubled_angles)
    doubled_weights = np.array(doubled_weights)

    bins = np.linspace(0, 2 * np.pi, NBINS + 1)
    counts, edges = np.histogram(doubled_angles, bins=bins, weights=doubled_weights)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 2 * np.pi / NBINS

    ax.bar(centers, counts, width=width, bottom=0, color=color,
           edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)


def make_rose(df, out_path):
    mean_phi, spread_deg, R, c, s = axial_stats(df['phi'])
    mean_phi_axial = mean_phi % 180.0

    weights = np.abs(df['dt'].values) if ROSE_DT_WEIGHTED else np.ones(len(df))
    wlabel = '|dt|-weighted' if ROSE_DT_WEIGHTED else 'unweighted (event count)'

    fig = plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(111, projection='polar')
    _draw_rose(ax, df['phi_az'].values, weights, color='#4C72B0')

    title = (f"AXEC2 pilot — fast-polarization (phi) rose diagram, {wlabel}\n"
             f"MLdd catalog, N={len(df)}, Q_w>=0.5 & phi_err<20 deg & dt_err<0.04 s guard\n"
             f"2015-01-22 to 2015-01-25 (pre-eruption pilot snapshot)\n"
             f"Axial mean phi = {mean_phi:.1f} deg (raw) = {mean_phi_axial:.1f} deg [0,180); "
             f"spread = {spread_deg:.1f} deg (R={R:.3f})")
    fig.suptitle(title, fontsize=9, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return mean_phi, mean_phi_axial, spread_deg, R


def make_polar_density(df, out_path):
    phi_az = df['phi_az'].values
    dt = df['dt'].values

    doubled_angles = np.concatenate([np.deg2rad(phi_az), np.deg2rad((phi_az + 180.0) % 360.0)])
    doubled_dt = np.concatenate([dt, dt])

    n_ang_bins = 36
    n_rad_bins = 14
    ang_bins = np.linspace(0, 2 * np.pi, n_ang_bins + 1)
    rad_bins = np.linspace(0, dt.max() * 1.05, n_rad_bins + 1)

    H, ang_edges, rad_edges = np.histogram2d(doubled_angles, doubled_dt, bins=[ang_bins, rad_bins])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    Theta, Rad = np.meshgrid(ang_edges, rad_edges, indexing='ij')
    pc = ax.pcolormesh(Theta, Rad, H, cmap='viridis', shading='auto')
    fig.colorbar(pc, ax=ax, pad=0.1, label='count per bin (axially mirrored)')

    # overlay individual points (both mirrored copies for axial symmetry)
    ax.scatter(doubled_angles, doubled_dt, s=10, color='white', edgecolor='black',
               linewidth=0.3, alpha=0.7, zorder=5)

    ax.set_title(f"AXEC2 pilot — phi (axial, deg) vs dt (s) polar density\n"
                 f"N={len(df)} events, 2015-01-22 to 2015-01-25 (pre-eruption pilot snapshot)",
                 fontsize=10, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_temporal(df, out_path, mean_phi):
    df = df.sort_values('t')
    t = df['t']
    phi = df['phi']
    dt = df['dt']
    phi_err = df['phi_error']
    dt_err = df['dt_error']
    qw = df['quality']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax0 = axes[0]
    ax0.errorbar(t, phi, yerr=phi_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
    sc0 = ax0.scatter(t, phi, c=qw, cmap='plasma', s=20 + 60 * (qw - qw.min()) / max(qw.max() - qw.min(), 1e-9),
                       edgecolor='black', linewidth=0.3, zorder=2)
    ax0.axhline(mean_phi, color='red', linestyle='--', linewidth=1,
                label=f'axial mean phi = {mean_phi:.1f} deg')
    ax0.set_ylabel('phi (deg, raw (-90,90])')
    ax0.legend(fontsize=8, loc='best')
    fig.colorbar(sc0, ax=ax0, label='Q_w')

    ax1 = axes[1]
    ax1.errorbar(t, dt, yerr=dt_err, fmt='none', ecolor='gray', alpha=0.5, zorder=1)
    sc1 = ax1.scatter(t, dt, c=qw, cmap='plasma', s=20 + 60 * (qw - qw.min()) / max(qw.max() - qw.min(), 1e-9),
                       edgecolor='black', linewidth=0.3, zorder=2)
    ax1.set_ylabel('dt (s)')
    ax1.set_xlabel('time (UTC)')
    fig.colorbar(sc1, ax=ax1, label='Q_w')

    fig.suptitle(f"AXEC2 pilot — phi & dt vs time, N={len(df)}\n{CAVEAT_TITLE}",
                 fontsize=9, fontweight='bold', color='darkred')
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = load_data()
    print(f"Loaded {len(df)} final-pass rows from {CSV_PATH}")
    print(f"Rose weighting mode: {'|dt|-weighted' if ROSE_DT_WEIGHTED else 'unweighted (event count)'}")

    mean_phi, mean_phi_axial, spread_deg, R = make_rose(df, os.path.join(HERE, 'axec2_pilot_rose.png'))
    print(f"Axial mean phi = {mean_phi:.2f} deg (raw) = {mean_phi_axial:.2f} deg [0,180); "
          f"spread = {spread_deg:.2f} deg; R = {R:.4f}")

    make_polar_density(df, os.path.join(HERE, 'axec2_pilot_polar_density.png'))
    make_temporal(df, os.path.join(HERE, 'axec2_pilot_temporal.png'), mean_phi)

    print("Wrote: axec2_pilot_rose.png, axec2_pilot_polar_density.png, axec2_pilot_temporal.png")


if __name__ == '__main__':
    main()
