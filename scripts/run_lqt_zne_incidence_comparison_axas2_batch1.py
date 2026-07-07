"""
Four-version splitting comparison for AXAS2 raw batch 1 (first 250 catalog events,
rebuilt from raw via build_raw_axas2_batch1.py to avoid the old cache's P-incidence<30
deg pre-filter bias - see that script's docstring):

  1. ZNE,  QC-cut at 35 deg on Eigenvalue-S incidence
  2. ZNE,  QC-cut at 35 deg on PyKonal-FMM incidence
  3. LQT,  QC-cut at 35 deg on Eigenvalue-S incidence, rotation driven by Eigenvalue-S
  4. LQT,  QC-cut at 35 deg on PyKonal-FMM incidence,  rotation driven by PyKonal-FMM

For each version: run perform_splitting_on_organized_waveforms on the QC-filtered event set,
then produce 3 rose plots (all events, Q_w <= 0 "null", Q_w > 0 "good"). All 12 panels go into
one PDF as a 4-row x 3-column grid, rows labeled by (frame, incidence version), columns
labeled by (all, null, good).

Run with:
    python3 build_raw_axas2_batch1.py   # once, to build the metadata CSV + per-event mseed this script loads
    python3 run_lqt_zne_incidence_comparison_axas2_batch1.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import obspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

_SWSPY_LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swspy'))
if _SWSPY_LOCAL_PATH not in sys.path:
    sys.path.insert(0, _SWSPY_LOCAL_PATH)
import swspy  # noqa: F401

from splitting_functions import perform_splitting_on_organized_waveforms

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(HERE, 'raw_axas2_batch1_data')
RAW_METADATA_CSV = os.path.join(RAW_DATA_DIR, 'raw_axas2_batch1_metadata.csv')
OUT_PDF = os.path.join(HERE, 'rose_plots_lqt_zne_incidence_comparison_axas2_batch1.pdf')

STATION = 'AXAS2'
INCIDENCE_CUT_DEG = 35.0

# SWSPy dynamic-window parameters, matching the production template notebook
# (axial_splitting_mldd_AXAS2_batched.ipynb) for MLdd-catalog runs.
FIRST_WINDOW_START = 2
LAST_WINDOW_START = 1
FIRST_WINDOW_END = 1.5
LAST_WINDOW_END = 2.5
N_WIN = 7
S_PICK_UNCERTAINTY = 0.0395  # Kaiwen's MLdd catalog S-pick uncertainty

VERSIONS = [
    {'label': 'ZNE / Eigenvalue-S cut',  'coord_system': 'ZNE', 'incidence_field': 'incidence_eigenvalue_jurkevics_s'},
    {'label': 'ZNE / PyKonal-FMM cut',   'coord_system': 'ZNE', 'incidence_field': 'incidence_pykonal_s'},
    {'label': 'LQT / Eigenvalue-S',      'coord_system': 'LQT', 'incidence_field': 'incidence_eigenvalue_jurkevics_s'},
    {'label': 'LQT / PyKonal-FMM',       'coord_system': 'LQT', 'incidence_field': 'incidence_pykonal_s'},
]

QUALITY_PANELS = [
    ('all',  lambda q: True),
    ('null (Q_w <= 0)', lambda q: (not np.isnan(q)) and q <= 0.0),
    ('good (Q_w > 0)',  lambda q: (not np.isnan(q)) and q > 0.0),
    ('good (Q_w > 0.5)', lambda q: (not np.isnan(q)) and q > 0.5),
]


def build_organized_waveforms(batch_df):
    """Build the organized_waveforms dict perform_splitting_on_organized_waveforms expects,
    from the raw-rebuilt AXAS2 metadata CSV + per-event mseed files (build_raw_axas2_batch1.py
    output), carrying both S-incidence fields."""
    organized = {}
    for _, row in batch_df.iterrows():
        waveform_path = os.path.join(RAW_DATA_DIR, row['waveform_file'])
        try:
            st = obspy.read(waveform_path)
        except Exception:
            continue
        organized[row['event_id']] = {
            'traces': st,
            'station': row['station'],
            'datetime': row['datetime'],
            's_arrival_time': row['s_arrival_time'],
            'p_arrival_time': row['p_arrival_time'],
            'back_azimuth': row['back_azimuth'],
            'snr_horizontal': row['snr_horizontal'],
            'rectilinearity_jurkevics': row['rectilinearity_jurkevics'],
            'incidence_eigenvalue_jurkevics': row['incidence_p_jurkevics'],
            'incidence_eigenvalue_jurkevics_s': row['incidence_eigenvalue_s'],
            'incidence_pykonal_s': row['incidence_pykonal_s'],
            'magnitude': row['magnitude'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'depth': row['depth'],
        }
    return organized


def draw_rose(ax, phis, title=None, nbins=36, color='steelblue'):
    """Polar rose histogram of fast directions with 180-degree symmetry, matching the
    established convention in rose_plots_temporal.py's _draw_rose (phi % 180, doubled-angle
    trick) - NOT plot_fast_direction_rose's +90 shift, which is a different, inconsistent
    convention not used by the production splitting_*.pdf / rose_plot_*.png outputs."""
    doubled_angles = []
    for phi in phis:
        phi_az = float(phi) % 180.0  # geographic azimuth [0, 180), matching _load_station
        p = phi_az % 360.0
        doubled_angles.append(np.deg2rad(p))
        doubled_angles.append(np.deg2rad((p + 180.0) % 360.0))
    doubled_angles = np.array(doubled_angles)

    bins = np.linspace(0, 2 * np.pi, nbins + 1)
    counts, bin_edges = np.histogram(doubled_angles, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 2 * np.pi / nbins

    ax.bar(bin_centers, counts, width=width, bottom=0.0, color=color, edgecolor='black', linewidth=0.4, alpha=0.75)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
    n_events = len(phis)
    if title:
        ax.set_title(f"{title}\nN={n_events}", fontsize=9)
    else:
        ax.set_title(f"N={n_events}", fontsize=9)


def main():
    print(f"Loading raw-rebuilt batch 1 metadata for {STATION} (SNR>=2.0, rectilinearity>=0.7, no incidence pre-filter)...")
    batch1 = pd.read_csv(RAW_METADATA_CSV)
    print(f"Batch 1: {len(batch1)} events")

    print("Reading waveforms and building organized_waveforms...")
    organized_all = build_organized_waveforms(batch1)
    print(f"  {len(organized_all)} events loaded")

    all_results = {}
    for version in VERSIONS:
        label = version['label']
        field = version['incidence_field']

        filtered = {eid: ed for eid, ed in organized_all.items()
                    if not np.isnan(ed[field]) and ed[field] <= INCIDENCE_CUT_DEG}
        print(f"\n=== {label}: {len(filtered)}/{len(organized_all)} events pass <= {INCIDENCE_CUT_DEG} deg cut ===")

        t0 = time.time()
        results = perform_splitting_on_organized_waveforms(
            filtered, FIRST_WINDOW_START, LAST_WINDOW_START, FIRST_WINDOW_END, LAST_WINDOW_END,
            N_WIN, S_PICK_UNCERTAINTY, mode='swspy',
            coord_system=version['coord_system'], sws_method='EV_and_XC', incidence_field=field,
            plot_results=False
        )
        print(f"  Done in {time.time()-t0:.0f}s")
        all_results[label] = results

    # Save raw phi/dt/Q_w per version to CSV for inspection
    rows = []
    for label, results in all_results.items():
        for eid, r in results.items():
            res = r['result']
            rows.append({'version': label, 'event_id': eid, 'phi': res.get('phi'), 'dt': res.get('dt'),
                         'phi_error': res.get('phi_error'), 'dt_error': res.get('dt_error'),
                         'quality': res.get('quality'), 'success': res.get('success')})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'lqt_zne_incidence_comparison_axas2_batch1_results.csv'), index=False)

    print("\nBuilding rose-plot PDF...")
    fig, axes = plt.subplots(len(VERSIONS), len(QUALITY_PANELS), figsize=(4 * len(QUALITY_PANELS), 16),
                              subplot_kw={'projection': 'polar'})

    for row_i, version in enumerate(VERSIONS):
        label = version['label']
        results = all_results[label]
        successful = {eid: r['result'] for eid, r in results.items() if r['result'].get('success')}

        for col_i, (col_label, quality_test) in enumerate(QUALITY_PANELS):
            ax = axes[row_i, col_i]
            phis = [r['phi'] for r in successful.values() if quality_test(r.get('quality', np.nan))]
            draw_rose(ax, phis)

            if row_i == 0:
                ax.text(0.5, 1.35, col_label, transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
            if col_i == 0:
                ax.text(-0.35, 0.5, label, transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, fontweight='bold', rotation=90)

    fig.suptitle(f"{STATION} batch 1 — ZNE vs LQT, Eigenvalue-S vs PyKonal-FMM incidence (35° cut)",
                 fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0.03, 0, 1, 0.98])
    fig.savefig(OUT_PDF, dpi=200)
    print(f"Saved {OUT_PDF}")


if __name__ == '__main__':
    main()
