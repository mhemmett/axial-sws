"""
7-panel (eruption-relative) and annual rose plots for the AXEC2 LQT + PyKonal-FMM
production run, following the exact style/conventions of rose_plots_temporal.py's
splitting_*.pdf figures (same period definitions, same doubled-angle %180 rose drawing,
same color scheme) - just scoped to a single station instead of the 6-station grid, since
rose_plots_temporal.py's make_rose_figure is tightly coupled to its 6-station STATION_ORDER
and isn't safely importable (it executes its whole pipeline at import time against a stale
hardcoded path).

Produces, matching rose_plots_temporal.py's naming convention:
    axec2_splitting_weighted.pdf            - 7-panel, weighted by |dt|
    axec2_splitting_unweighted.pdf          - 7-panel, count-weighted
    axec2_splitting_weighted_annual.pdf     - annual bins, weighted by |dt|
    axec2_splitting_unweighted_annual.pdf   - annual bins, count-weighted

Run with:
    python3 build_production_rose_plots_axec2.py
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'production_axec2_lqt_pykonal_results')
RAW_METADATA_CSV = os.path.join(HERE, 'raw_axec2_all_batches_data', 'raw_axec2_all_batches_metadata.csv')

STATION = 'AXEC2'
NBINS = 36

# Same eruption timing as rose_plots_temporal.py
ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')


def load_combined_results():
    """Combine all per-batch production results with the raw metadata (for datetime/lat/lon)."""
    result_files = glob.glob(os.path.join(RESULTS_DIR, 'splitting_results_mldd_2015_2021_axec2_batch_*.csv'))
    dfs = [pd.read_csv(f) for f in result_files]
    dfs = [d for d in dfs if len(d) > 0]
    results = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    results = results[results['success'] == True].copy()
    results = results[results['dt'] > 0]  # remove null measurements, matching _load_station convention
    results['t'] = pd.to_datetime(results['datetime'], utc=True)
    results['phi_az'] = results['phi'] % 180.0
    return results


# ── Period definitions (verbatim from rose_plots_temporal.py) ────────────────────

def _build_time_periods(all_df):
    """7 periods: pre, syn, 5 equal-count post-eruption bins."""
    post = all_df[all_df['t'] >= ERUPTION_END].copy()
    post_times = post['t'].sort_values().reset_index(drop=True)
    n = len(post_times)

    boundaries = [ERUPTION_END]
    for i in range(1, 5):
        idx = int(round(i * n / 5))
        idx = min(idx, n - 1)
        boundaries.append(post_times.iloc[idx])
    boundaries.append(None)

    last_date = post_times.iloc[-1] if n > 0 else ERUPTION_END

    def _fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else last_date.strftime('%b %Y')

    periods = [
        ('Pre-eruption', None, ERUPTION_START),
        ('Syn-eruption', ERUPTION_START, ERUPTION_END),
    ]
    for i in range(5):
        t0, t1 = boundaries[i], boundaries[i + 1]
        periods.append((f'{_fmt(t0)}\n– {_fmt(t1)}', t0, t1))

    return periods


def _build_annual_periods():
    """Annual bins: Pre-eruption | Syn-eruption | Post-2015 | 2016 ... 2021 (data ends 2021)."""
    periods = [
        ('Pre-eruption\n2015', None, ERUPTION_START),
        ('Syn-eruption\n2015', ERUPTION_START, ERUPTION_END),
        ('Post-eruption\n2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for yr in range(2016, 2022):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr+1}-01-01', tz='UTC') if yr < 2021 else None
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
    """Verbatim from rose_plots_temporal.py: doubled-angle trick for 180 deg symmetry."""
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


def make_single_station_figure(df, time_periods, dt_weighted, title):
    """Single-row rose figure across time_periods, for one station."""
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
    print("Loading combined production results...")
    df = load_combined_results()
    print(f"Total successful, non-null splitting measurements: {len(df)}")

    weight_label = {True: 'weighted', False: 'unweighted'}

    for dt_weighted in (True, False):
        wlabel = weight_label[dt_weighted]

        time_periods = _build_time_periods(df)
        fig = make_single_station_figure(
            df, time_periods, dt_weighted,
            f"{STATION} — LQT + PyKonal-FMM incidence (35° cut) — {wlabel}"
        )
        out_path = os.path.join(HERE, f'axec2_splitting_lqt_pykonal_final_{wlabel}.pdf')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

        annual_periods = _build_annual_periods()
        fig = make_single_station_figure(
            df, annual_periods, dt_weighted,
            f"{STATION} — LQT + PyKonal-FMM incidence (35° cut) — annual, {wlabel}"
        )
        out_path = os.path.join(HERE, f'axec2_splitting_lqt_pykonal_final_{wlabel}_annual.pdf')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
