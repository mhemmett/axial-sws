#!/usr/bin/env python3
"""
synthetic_hudson_eruption_cycle_summary.py

Combined summary figure, per explicit user request. Panels labeled left-to-right, then
top-to-bottom (A-F); letters are placed in the upper-left corner of each panel's own space (NOT
folded into the title text).

  Top row (panels A, B, C) -- synthetic Hudson-crack-model STRUCTURE MAPS (same tick-mark-
    field method as synthetic_hudson_structure_map.py, ONE shared azimuth colorbar), single-
    sill model only (no second source in the Before/During panels -- the now-retired two-sills
    variant is not used here):
      A. Before Eruption  (Scenario B: background + fault damage zones + sill)
      B. During Eruption  (Scenario C: background + fault damage zones + dike, sill deflated)
      C. Re-Inflation     (Scenario D, NEW: background + fault damage zones + ONE inflating
                            sill relocated to where the retired model's "second sill" (Kidiwela
                            S2 location) used to sit -- no dike, no original sill at all; see
                            synthetic_hudson_reinflation_scenario.py)

  Bottom block (panels D, E, F) -- "Observed vs. Synthetic Shear-wave Splitting": each of D/E/F
    is a 6-station x 2-column (Real, Synthetic) mini rose-comparison grid (12 rose panels each,
    same layout/convention as add_real_vs_synth_comparison_pages_final_filter.py), aligned
    beneath its matching top-row map:
      D. Before Eruption:  real = final-filter Pre-eruption; synthetic = Scenario B
                          (synthetic_hudson_pre_syn_phidt.csv)
      E. During Eruption:  real = final-filter Syn-eruption; synthetic = Scenario C (same CSV)
      F. Re-Inflation:     real = final-filter data in the SAME period-7 (most recent of the
                          standard 7 eruption-cycle periods, rose_7period_6stations_newdata.py)
                          time window used to select the Scenario-D synthetic events;
                          synthetic = Scenario D (synthetic_hudson_reinflation_phidt.csv)

Note the two different QC filters in play, same as every other synthetic-vs-real comparison
this session: tier-4 QC picks which real hypocenters get FED INTO the forward Hudson model (an
arbitrary, generous filter -- the real phi/dt at those events is discarded), while the
final-filter QC governs what's actually DISPLAYED as "Observed" data here.

All 39 subplots (3 maps + 36 rose panels) sit on a single uniform GridSpec (equal row heights,
equal column widths for the 6 data columns) with small hspace/wspace, per explicit user request
that panels be the same size and not far apart.

Produces: synthetic_hudson_eruption_cycle_summary.pdf (1 page)

Run with:
    python3 synthetic_hudson_eruption_cycle_summary.py
"""

import math
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable

import synthetic_hudson_structure_map as sm
# Installs the Scenario-D-aware total_stress_and_crack_sets onto synthetic_hudson_crack_scenarios
# (delegates A/B/C to the original single-sill physics, unchanged).
import synthetic_hudson_reinflation_scenario as reinflation
from rose_7period_6stations_newdata import _draw_rose
from newdata_final_filter import (
    STATION_ORDER, load_station_final_filtered, ERUPTION_START, ERUPTION_END,
)
from run_synthetic_hudson_splitting_reinflation import get_reinflation_window

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'synthetic_hudson_eruption_cycle_summary.pdf')

SILL2 = reinflation.SILL2

PRE_COLOR = '#800080'
SYN_COLOR = '#CC0000'
REINF_COLOR = '#0072B2'   # colorblind-friendly blue, distinct from pre/syn purple/red


def _lighten(hex_color, frac=0.45):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.to_hex((1 - frac) * rgb + frac * np.array([1., 1., 1.]))


PRE_COLOR_SYNTH = _lighten(PRE_COLOR)
SYN_COLOR_SYNTH = _lighten(SYN_COLOR)
REINF_COLOR_SYNTH = _lighten(REINF_COLOR)


# ── Scenario-D-aware overrides for the structure-map module (same pattern as
# synthetic_hudson_structure_map_two_sills.py's monkeypatch of sm.compute_sill_dike_field /
# sm.total_stress_and_crack_sets) ────────────────────────────────────────────────

sm.total_stress_and_crack_sets = reinflation.total_stress_and_crack_sets_reinflation


def compute_sill_dike_field_with_reinflation(scenario, xn, yn, dike_interps=None):
    PHI = np.full((len(xn), len(yn)), np.nan)
    DV = np.zeros((len(xn), len(yn)))
    GATE = np.zeros((len(xn), len(yn)), dtype=bool)
    for ix, x in enumerate(xn):
        for iy, y in enumerate(yn):
            sill_gate = (scenario == 'B' and
                        math.hypot(x - sm.SILL['x0'], y - sm.SILL['y0'])
                        <= sm.SILL['R'] + sm.SILL_GATE_MARGIN_KM)
            dike_gate = (scenario == 'C' and sm._dist_to_dike_km(x, y) <= sm.DIKE_GATE_HALFWIDTH_KM)
            sill2_gate = (scenario == 'D' and
                         math.hypot(x - SILL2['x0'], y - SILL2['y0'])
                         <= SILL2['R'] + sm.SILL_GATE_MARGIN_KM)
            gate = sill_gate or dike_gate or sill2_gate
            GATE[ix, iy] = gate
            if not gate:
                continue
            sigma, _extra_fault_ignored = sm.total_stress_and_crack_sets(
                scenario, x, y, sm.EVAL_DEPTH_KM, dike_interps=dike_interps)
            C_eff, _ = sm.effective_stiffness_with_extra_cracks(sigma, [], n_hats=sm.n_hats)
            phi, dV, _, _ = sm.christoffel_splitting(C_eff, sm.p_vert, sm.RHO)
            PHI[ix, iy] = phi
            DV[ix, iy] = dV
    return PHI, DV, GATE


sm.compute_sill_dike_field = compute_sill_dike_field_with_reinflation


def _sill2_circle(ax, deflated=False):
    circ = plt.Circle((SILL2['x0'], SILL2['y0']), SILL2['R'], fill=False, color='deepskyblue',
                      lw=2.2, zorder=10, linestyle='--' if deflated else '-')
    ax.add_patch(circ)


def _draw_hydrothermal_rosette(ax, xy, radius_km, n_ticks=12, color='#8B4513'):
    """Decorative annotation marking the AXAS2 isotropic hydrothermal-pathway crack-density
    boost (Scenario D only, synthetic_hudson_reinflation_scenario.py's epsilon0 boost) -- NOT
    itself computed from the physics tick-mark field, since a single pointwise Christoffel
    evaluation only ever yields ONE fast azimuth per grid point, not "all directions"
    simultaneously. Drawn as an explicit multi-angle tick rosette instead, to visually convey
    the intended "cracks going all directions" hydrothermal-pathway concept at a glance."""
    x0, y0 = xy
    circ = plt.Circle((x0, y0), radius_km, fill=False, color=color, lw=1.6, ls=':', zorder=10)
    ax.add_patch(circ)
    for k in range(n_ticks):
        ang = np.pi * k / n_ticks   # 0-180 deg, axial ticks (crack faces are 180-deg symmetric)
        dx, dy = 0.55 * radius_km * np.sin(ang), 0.55 * radius_km * np.cos(ang)
        ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], '-', color=color, lw=1.3, alpha=0.9,
               zorder=11)


def _corner_label(ax, letter, fontsize=14):
    ax.text(0.03, 0.96, letter, fontsize=fontsize, fontweight='bold', color='black',
            ha='left', va='top', transform=ax.transAxes, zorder=25,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))


def _set_latlon_axes(ax, show_ylabel=True):
    """Relabels a map panel's km-based x/y ticks as longitude/latitude, per explicit user
    request -- the underlying plotted geometry stays in local (East,North) km (everything else
    in this model is defined in that frame), only the tick labels/axis titles change. Panels B
    and C drop their y-axis (latitude) label/ticks entirely (show_ylabel=False), since they sit
    to the right of panel A in the same row and would otherwise repeat it."""
    xticks = [t for t in ax.get_xticks() if sm.X_START <= t <= sm.X_END]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{sm.INI_LON + t / sm.KM_PER_DEG_LON:.2f}' for t in xticks], fontsize=6)
    ax.set_xlabel('Longitude', fontsize=8)

    if show_ylabel:
        yticks = [t for t in ax.get_yticks() if sm.Y_START <= t <= sm.Y_END]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{sm.INI_LAT + t / sm.KM_PER_DEG_LAT:.2f}' for t in yticks], fontsize=6)
        ax.set_ylabel('Latitude', fontsize=8)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')


# ── Structure-map row (panels A, B, C) ───────────────────────────────────────────

def build_station_df():
    xs, ys, names = [], [], []
    for s in sm.ALL_STATIONS:
        if s in sm.STATION_XY:
            xs.append(sm.STATION_XY[s][0]); ys.append(sm.STATION_XY[s][1])
        else:
            sta_file = pd.read_csv(os.path.join(HERE, '..', 'data', 'stations_axial.llz'),
                                   sep=r'\s+', names=['lon', 'lat', 'e', 's'],
                                   engine='python').set_index('s')
            x, y = sm.ll2xy(sta_file.loc[s, 'lat'], sta_file.loc[s, 'lon'])
            xs.append(float(x)); ys.append(float(y))
        names.append(s)
    return pd.DataFrame({'x': xs, 'y': ys}, index=names)


def draw_maps_row(fig, gs_top, cax_rect, gray, ext, sta_df, dike_interps):
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_top[0, 2])

    print('Computing Before Eruption (Scenario B, sill) crack field...')
    sm.draw_panel(ax_a, gray, ext, sta_df, 'B', 'Before Eruption', dike_interps=None)

    print('Computing During Eruption (Scenario C, dike) crack field...')
    sm.draw_panel(ax_b, gray, ext, sta_df, 'C', 'During Eruption', dike_interps=dike_interps)

    print('Computing Re-Inflation (Scenario D, relocated sill) crack field...')
    sm.draw_panel(ax_c, gray, ext, sta_df, 'D', 'Re-Inflation', dike_interps=None)
    _sill2_circle(ax_c, deflated=False)
    _draw_hydrothermal_rosette(ax_c, reinflation.AXAS2_XY, reinflation.AXAS2_HYDROTHERMAL_RADIUS_KM)

    _set_latlon_axes(ax_a, show_ylabel=True)
    _set_latlon_axes(ax_b, show_ylabel=False)
    _set_latlon_axes(ax_c, show_ylabel=False)

    for ax, letter in zip((ax_a, ax_b, ax_c), 'ABC'):
        _corner_label(ax, letter)

    cax = fig.add_axes(cax_rect)
    sm_map = ScalarMappable(cmap=sm.AZ_CMAP, norm=sm.AZ_NORM)
    sm_map.set_array([])
    cb = fig.colorbar(sm_map, cax=cax)
    cb.set_label('Synthetic fast direction $\\phi$ (deg, axial 0-180)', fontsize=7.5)
    cb.ax.tick_params(labelsize=6)

    legend_handles = [
        mlines.Line2D([], [], color='#00CFFF', lw=2.0, label='Ring faults (E + W)'),
        mlines.Line2D([], [], color='orange', lw=2.0, label='Sill (Before Eruption)'),
        mlines.Line2D([], [], color='orange', lw=2.0, ls='--', label='Sill, deflated (During Eruption)'),
        mlines.Line2D([], [], color='lime', lw=2.0, label='Dike trace (During Eruption)'),
        mlines.Line2D([], [], color='deepskyblue', lw=2.0,
                      label='Relocated sill (Re-Inflation, former "sill 2" location)'),
        mlines.Line2D([], [], color='#8B4513', lw=1.6, ls=':',
                      label='Hydrothermal pathways (Re-Inflation, AXAS2, isotropic)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8,
                      label='Station'),
    ]
    return legend_handles


# ── Real/synthetic comparison grids (panels D, E, F) ─────────────────────────────

def load_real_period_data_per_station():
    """dict[period_label][station] -> final-filter QC phi_az array."""
    real_dfs = {sta: load_station_final_filtered(sta) for sta in STATION_ORDER}
    _raw, reinf_t0, reinf_t1 = get_reinflation_window()

    def _subset(df, t0, t1):
        m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
        if t1 is not None:
            m = m & (df['t'] < t1)
        return df[m]

    windows = {
        'Before Eruption': (None, ERUPTION_START),
        'During Eruption': (ERUPTION_START, ERUPTION_END),
        'Re-Inflation': (reinf_t0, reinf_t1),
    }
    out = {label: {} for label in windows}
    for label, (t0, t1) in windows.items():
        for sta in STATION_ORDER:
            out[label][sta] = _subset(real_dfs[sta], t0, t1)['phi_az'].values
    return out


def load_synth_period_data_per_station():
    """dict[period_label][station] -> synthetic phi_synth (mod 180) array."""
    pre_syn = pd.read_csv(os.path.join(HERE, 'synthetic_hudson_pre_syn_phidt.csv'))
    reinf = pd.read_csv(os.path.join(HERE, 'synthetic_hudson_reinflation_phidt.csv'))

    period_csv = {
        'Before Eruption': (pre_syn, 'Pre-eruption'),
        'During Eruption': (pre_syn, 'Syn-eruption'),
        'Re-Inflation': (reinf, 'Re-Inflation'),
    }
    out = {}
    for label, (df, csv_period) in period_csv.items():
        out[label] = {}
        for sta in STATION_ORDER:
            sub = df[(df['station'] == sta) & (df['period'] == csv_period)]
            out[label][sta] = sub['phi_synth'].values % 180.0
    return out


def draw_comparison_grids(fig, gs_bottom, real_data, synth_data):
    groups = [
        ('D', 'Before Eruption', 0, PRE_COLOR, PRE_COLOR_SYNTH),
        ('E', 'During Eruption', 2, SYN_COLOR, SYN_COLOR_SYNTH),
        ('F', 'Re-Inflation', 4, REINF_COLOR, REINF_COLOR_SYNTH),
    ]
    for panel_letter, label, col0, real_color, synth_color in groups:
        for ri, sta in enumerate(STATION_ORDER):
            row = ri
            real_phi = real_data[label][sta]
            synth_phi = synth_data[label][sta]

            ax_real = fig.add_subplot(gs_bottom[row, col0], projection='polar')
            _draw_rose(ax_real, real_phi, np.ones(len(real_phi)), real_color)

            ax_synth = fig.add_subplot(gs_bottom[row, col0 + 1], projection='polar')
            _draw_rose(ax_synth, synth_phi, np.ones(len(synth_phi)), synth_color)

            if ri == 0:
                ax_real.set_title(f'Real: {label}\nN={len(real_phi):,}', fontsize=7.5,
                                  fontweight='bold')
                ax_synth.set_title(f'Synthetic: {label}\nN={len(synth_phi):,}', fontsize=7.5,
                                   fontweight='bold')
                _corner_label(ax_real, panel_letter)
            else:
                ax_real.set_title(f'N={len(real_phi):,}', fontsize=7.5, fontweight='bold')
                ax_synth.set_title(f'N={len(synth_phi):,}', fontsize=7.5, fontweight='bold')

            if col0 == 0:
                ax_real.text(-0.32, 0.5, sta, fontsize=9, fontweight='bold',
                            ha='center', va='center', transform=ax_real.transAxes)

    legend_handles_bottom = [
        mlines.Line2D([], [], marker='s', color='none', mfc='#800080', mec='black', ms=10,
                     label='Real: Before Eruption'),
        mlines.Line2D([], [], marker='s', color='none', mfc=PRE_COLOR_SYNTH, mec='black', ms=10,
                     label='Synthetic: Before Eruption'),
        mlines.Line2D([], [], marker='s', color='none', mfc='#CC0000', mec='black', ms=10,
                     label='Real: During Eruption'),
        mlines.Line2D([], [], marker='s', color='none', mfc=SYN_COLOR_SYNTH, mec='black', ms=10,
                     label='Synthetic: During Eruption'),
        mlines.Line2D([], [], marker='s', color='none', mfc=REINF_COLOR, mec='black', ms=10,
                     label='Real: Re-Inflation'),
        mlines.Line2D([], [], marker='s', color='none', mfc=REINF_COLOR_SYNTH, mec='black', ms=10,
                     label='Synthetic: Re-Inflation'),
    ]
    return legend_handles_bottom


def main():
    print('Loading bathymetry...')
    gray, ext = sm.load_bathymetry()
    sta_df = build_station_df()

    print('Building dike stress interpolators (Scenario B)...')
    dike_xn = np.arange(4.0, 12.01, 0.1)
    dike_yn = np.arange(0.0, 15.01, 0.1)
    dike_interps = sm.build_dike_stress_interpolators(dike_xn, dike_yn)

    print('\nLoading real (final-filter) per-station period data...')
    real_data = load_real_period_data_per_station()
    for lbl in real_data:
        n = sum(len(v) for v in real_data[lbl].values())
        print(f'  {lbl}: N_real={n:,}')

    print('\nLoading synthetic per-station period data...')
    synth_data = load_synth_period_data_per_station()
    for lbl in synth_data:
        n = sum(len(v) for v in synth_data[lbl].values())
        print(f'  {lbl}: N_synth={n:,}')

    n_data_rows = len(STATION_ORDER)   # 6 station rows
    fig = plt.figure(figsize=(15, 21))

    # Two INDEPENDENT GridSpecs sharing the same left/right span, so each map (1 of 3 equal
    # columns in gs_top) is exactly the same width as one Real+Synthetic column-pair (2 of 6
    # equal columns in gs_bottom) -- per explicit user request that A/B/C be scaled up to match
    # D/E/F's width. gs_top gets its own (small) wspace so the 3 maps sit close together,
    # independent of gs_bottom's own column spacing. The shared colorbar is a manually placed
    # axes (not a GridSpec column) so it doesn't need to match either grid's column widths.
    LEFT, RIGHT = 0.06, 0.90

    gs_top = GridSpec(1, 3, left=LEFT, right=RIGHT, top=0.965, bottom=0.685, wspace=0.05)
    gs_bottom = GridSpec(n_data_rows, 6, left=LEFT, right=RIGHT, top=0.635, bottom=0.045,
                         hspace=0.12, wspace=0.08)
    cax_rect = [RIGHT + 0.015, 0.685, 0.015, 0.965 - 0.685]

    legend_handles_top = draw_maps_row(fig, gs_top, cax_rect, gray, ext, sta_df, dike_interps)

    print('\nDrawing observed-vs-synthetic comparison grids (panels D, E, F)...')
    legend_handles_bottom = draw_comparison_grids(fig, gs_bottom, real_data, synth_data)

    fig.legend(handles=legend_handles_top, loc='center', ncol=4, fontsize=7.5,
              framealpha=0.9, bbox_to_anchor=(0.5, 0.658))
    fig.legend(handles=legend_handles_bottom, loc='center', ncol=6, fontsize=7.5,
              framealpha=0.9, bbox_to_anchor=(0.5, 0.025))

    fig.suptitle('Modeling Shear-wave Splitting Across an Eruption Cycle',
                 fontsize=15, fontweight='bold', y=0.985)

    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
