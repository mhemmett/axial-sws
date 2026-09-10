"""
Annual rose plots for AXEC2, full 2015-2026 record, restricted to "tight"/good
measurements only:
    quality (Q_w, Wustefeld 2010) >= 0.5
    phi_error < 20 deg
    dt_error  < 0.04 s

Combines the enriched 2015-2021 and 2022-2026 result files (which already
carry the QC/geometry enrichment columns) rather than the 2015-2021-only
production batch CSVs used by build_production_rose_plots_axec2_qw05.py, so
this covers the full record including the 2022 instrumental (unleveled
sensor) window.

Produces:
    axec2_annual_rose_qw05_weighted.pdf
    axec2_annual_rose_qw05_unweighted.pdf

Run with:
    python3 axec2_annual_rose_plots_qw05_full.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'lqt_pykonal_combined_results')

STATION = 'AXEC2'
NBINS = 36

QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')


def load_combined_results():
    files = [
        os.path.join(DATA_DIR, 'splitting_results_AXEC2_2015_2021_all_batches_enriched.csv'),
        os.path.join(DATA_DIR, 'splitting_results_AXEC2_2022_2026_all_batches_enriched.csv'),
    ]
    dfs = [pd.read_csv(f) for f in files]
    results = pd.concat(dfs, ignore_index=True)
    results = results[results['success'] == True].copy()
    results = results[results['dt'] > 0]
    results = results.dropna(subset=['quality', 'phi_error', 'dt_error'])
    results = results[(results['quality'] >= QW_MIN) &
                       (results['phi_error'] < PHI_ERR_MAX) &
                       (results['dt_error'] < DT_ERR_MAX)]
    results['t'] = pd.to_datetime(results['datetime'], utc=True)
    results['phi_az'] = results['phi'] % 180.0
    return results


def _build_annual_periods(df):
    """Pre/syn/post-2015, then one bin per calendar year through the last year of data."""
    last_year = df['t'].max().year
    periods = [
        ('Pre-eruption\n2015', None, ERUPTION_START),
        ('Syn-eruption\n2015', ERUPTION_START, ERUPTION_END),
        ('Post-eruption\n2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for yr in range(2016, last_year + 1):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr + 1}-01-01', tz='UTC') if yr < last_year else None
        periods.append((str(yr), t0, t1))
    return periods


def _subset(df, t_start, t_end):
    m = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
    if t_end is not None:
        m = m & (df['t'] < t_end)
    return df[m]


def _period_colors(n_periods):
    colors = ['#800080', '#CC0000']  # pre-eruption (purple), syn (red)
    n_post = n_periods - 2
    start = np.array(mcolors.to_rgb('#ADD8E6'))
    end = np.array(mcolors.to_rgb('#800080'))
    for i in range(n_post):
        t = i / max(n_post - 1, 1)
        colors.append(mcolors.to_hex((1 - t) * start + t * end))
    return colors


def _draw_rose(ax, phi_az_vals, weights, color):
    """Doubled-angle trick for 180 deg (axial) symmetry."""
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('none')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(0.8)
    ax.grid(False)

    if len(phi_az_vals) == 0:
        ax.set_ylim(0, 1)
        return

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

    ax.bar(centers, counts, width=width, bottom=0, color=color, edgecolor='black', linewidth=0.5, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.25 if counts.max() > 0 else 1)


def make_annual_figure(df, time_periods, dt_weighted, title):
    n_cols = len(time_periods)
    colors = _period_colors(n_cols)
    panel_size = 2.0

    fig, axes = plt.subplots(1, n_cols, figsize=(panel_size * n_cols, panel_size + 1.0),
                              subplot_kw={'projection': 'polar'})
    if n_cols == 1:
        axes = [axes]

    for col_idx, (label, t_start, t_end) in enumerate(time_periods):
        sub = _subset(df, t_start, t_end)
        ax = axes[col_idx]

        phi_vals = sub['phi_az'].values
        weights = np.abs(sub['dt'].values) if dt_weighted else np.ones(len(sub))
        _draw_rose(ax, phi_vals, weights, colors[col_idx])

        n_events = len(sub)
        ax.set_title(f'{label}\nN={n_events:,}', fontsize=7.5, fontweight='bold', pad=4)

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.05)
    fig.tight_layout()
    return fig


def main():
    print(f"Loading combined AXEC2 results 2015-2026 (Q_w >= {QW_MIN}, "
          f"phi_error < {PHI_ERR_MAX} deg, dt_error < {DT_ERR_MAX} s)...")
    df = load_combined_results()
    print(f"Total good-quality splitting measurements: {len(df)}")

    annual_periods = _build_annual_periods(df)
    weight_label = {True: 'weighted', False: 'unweighted'}

    for dt_weighted in (True, False):
        wlabel = weight_label[dt_weighted]
        fig = make_annual_figure(
            df, annual_periods, dt_weighted,
            f"{STATION} — LQT + PyKonal-FMM incidence (35° cut), Q$_w\\geq${QW_MIN} — annual 2015-2026, {wlabel}"
        )
        out_path = os.path.join(HERE, f'axec2_annual_rose_qw05_{wlabel}.pdf')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
