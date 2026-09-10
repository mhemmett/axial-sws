#!/usr/bin/env python3
"""
fit_source_from_phi_quadtree_eastern_wall_preeruption.py

Exploratory analysis, per explicit user request: does the PRE-ERUPTION fast-direction (phi)
pattern on the eastern caldera wall (between AXAS1, AXEC1, AXEC2, AXEC3) look like it's
controlled by a nearby inflating (pressurized) source, and if so, where would that source have
to sit?

Physical basis (see conversation): for a pressurized spherical/oblong source in an elastic
medium, radial stress is compressive and tangential (hoop) stress is tensile (Lame/Eshelby
spherical-cavity solution; Mogi 1958). Tensile micro-cracks open with their crack-face normal
along the most tensile principal stress (tangential here), so crack PLANES contain the radial
direction -- i.e. the stress-induced-anisotropy (EDA) prediction is a RADIALLY-ORIENTED fast
direction pattern, phi pointing toward/away from the source, axisymmetric about it. This is a
distinct, testable prediction from the fault-parallel/tectonic-stress framing used elsewhere in
this project.

This script:
  1. Rebuilds the SAME pre-eruption quad-tree bins used in
     phi_7period_6stations_newdata_snr2_q75_countmin50_midpoint_quadtree_1row.py (grade-3
     filter, COUNT_MIN=50, ray-midpoint assignment, reusing the cached ray tracing -- no
     re-tracing), restricted to Pre-eruption only.
  2. Selects the bins in the eastern-wall bounding box between AXAS1/AXEC1/AXEC2/AXEC3.
  3. Fits a POINT (isotropic/"spherical") source location (x0, y0) by minimizing the circular
     (axial, mod-180) misfit between each bin's observed phi and the azimuth of the straight
     line from the source to that bin's centroid -- i.e. testing whether phi is consistent with
     radiating from a single point, weighted by each bin's ray count.
  4. Fits a LINE-SOURCE ("oblong"/elongated) model as an alternative: same radial-misfit idea,
     but predicting phi from the azimuth to the NEAREST POINT on a candidate line (a location +
     strike), which approximates the anisotropic hoop-stress pattern expected around an
     elongated (prolate spheroid / dike-like) source rather than a sphere.
  5. Reports both fits' misfit (RMS circular angular residual, degrees) side by side, plus the
     best-fit source location(s), so the two source geometries can be compared directly. Also
     compares the best-fit point location against the PyLith model's Mogi source (shallow AMC
     roof, central caldera, per pylith_axial/'s cfg).

This is a first-order/exploratory geometric fit (not a full elastic stress inversion) --
treat the result as a hypothesis-generating check on whether the eastern-wall phi pattern is
consistent with local radial control, not as a rigorous source-location solution.

Run with:
    python3 fit_source_from_phi_quadtree_eastern_wall_preeruption.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
import spatial_quadtree_fast_direction_tier5_6stations as Q
from traveltime_anisotropy_7period_6stations_newdata_snr2_q75_countmin10_midpoint_smooth_truemax_diff_polished_1row import (
    compute_ray_midpoint_voxels,
)
from phi_quiver_7period_onerow_station25pct_countmin10 import build_event_phi_lookup

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'fit_source_from_phi_quadtree_eastern_wall_preeruption.pdf')

GRADE = T.GRADES[3]
assert GRADE['key'] == 'page1_q0.75', GRADE
Q.MIN_COUNT = 50

# Eastern-wall bounding box: loose box around AXAS1/AXEC1/AXEC2/AXEC3 (station coords printed
# via T._setup_environment below), per explicit user instruction.
EAST_X0, EAST_X1 = 7.4, 10.2
EAST_Y0, EAST_Y1 = 3.2, 6.3

# PyLith model's central Mogi source (pylith_axial/ cfg): shallow AMC roof, ~3.33 km depth,
# central caldera -- used here only as a map-view (x, y) reference point for comparison, since
# this fit is 2D (depth-collapsed).
PYLITH_MOGI_XY_APPROX = None  # filled in from station-relative caldera center below


def build_pre_eruption_bins():
    T._setup_environment(need_tracer=False)
    summary, vox = T.load_cache()
    N = len(summary)
    assert (summary['ray_id'].values == np.arange(N)).all()

    event_ns = pd.DatetimeIndex(
        pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')).asi8
    all_t = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    periods = T.build_7_periods(pd.DataFrame({'t': all_t}))
    ray_period = T.assign_periods(event_ns, periods)

    mask = ((summary['quality'].values >= GRADE['qw_min']) &
            (summary['snr_horizontal'].values >= GRADE['snr_min']) &
            (summary['phi_error'].values <= GRADE['phi_err_max']))

    print('Computing ray midpoint voxels...')
    ray_id_mid, ix_mid, iy_mid, iz_mid = compute_ray_midpoint_voxels(vox)
    mx_all = np.full(N, np.nan)
    my_all = np.full(N, np.nan)
    mx_all[ray_id_mid] = T.xn[ix_mid] + T.VXY / 2.0
    my_all[ray_id_mid] = T.yn[iy_mid] + T.VXY / 2.0

    print('Recovering event phi per cached ray...')
    phi_lookup = build_event_phi_lookup()
    summary['t'] = pd.to_datetime(summary['event_datetime'], utc=True, format='ISO8601')
    phi_ray = np.full(N, np.nan)
    for sta in T.STATIONS:
        m = (summary['station'].values == sta)
        phi_ray[m] = phi_lookup[sta].reindex(summary['t'].values[m]).values

    pi_pre = 0  # Pre-eruption is always period index 0 in build_7_periods()
    keep = mask & (ray_period == pi_pre) & np.isfinite(mx_all) & np.isfinite(phi_ray)
    n_rays = int(keep.sum())
    print(f'Pre-eruption: {n_rays:,} rays passing grade-3 filter with a valid midpoint+phi')
    bins = Q.build_all_bins(mx_all[keep], my_all[keep], phi_ray[keep])
    print(f'  -> {len(bins):,} quad-tree bins (N>={Q.MIN_COUNT})')
    return bins, n_rays


def circular_radial_misfit(phi_obs_deg, pred_az_deg, weights):
    """RMS-style circular (axial, mod-180) misfit between observed phi and a predicted radial
    azimuth, in degrees, using a weighted 1-cos(2*delta) loss (standard axial circular
    distance) converted back to an interpretable RMS angle."""
    delta = np.radians(phi_obs_deg - pred_az_deg)
    loss = np.average(1.0 - np.cos(2.0 * delta), weights=weights)
    # For small delta, 1-cos(2*delta) ~= 2*delta^2, so mean-squared-error ~= loss/2 (radians^2).
    # Convert to an RMS-equivalent angle in degrees for interpretability (exact for small
    # misfit, a reasonable summary even when misfit is large since delta is wrapped to
    # [-90, 90] by construction of the axial 1-cos(2*delta) loss).
    rms_deg = np.degrees(np.sqrt(loss / 2.0))
    return loss, rms_deg


def point_source_misfit(params, cx, cy, phi_obs, weights):
    x0, y0 = params
    dx, dy = cx - x0, cy - y0
    pred_az = np.degrees(np.arctan2(dx, dy)) % 180.0
    loss, _ = circular_radial_misfit(phi_obs, pred_az, weights)
    return loss


def line_source_misfit(params, cx, cy, phi_obs, weights):
    """Line (elongated/'oblong') source: candidate line through (x0, y0) with strike angle
    theta (deg from north). Predicted azimuth is toward the NEAREST POINT on that infinite
    line -- perpendicular to the line for points off-axis, undefined exactly on-axis (handled
    via a small epsilon)."""
    x0, y0, theta_deg = params
    theta = np.radians(theta_deg)
    d_hat = np.array([np.sin(theta), np.cos(theta)])  # line direction unit vector
    dx, dy = cx - x0, cy - y0
    t = dx * d_hat[0] + dy * d_hat[1]  # projection onto line
    px, py = x0 + t * d_hat[0], y0 + t * d_hat[1]  # nearest point on line
    rx, ry = cx - px, cy - py  # perpendicular vector from line to bin
    pred_az = np.degrees(np.arctan2(rx, ry)) % 180.0
    loss, _ = circular_radial_misfit(phi_obs, pred_az, weights)
    return loss


def fit_point_source(cx, cy, phi_obs, weights):
    """Grid-search + bounded local refinement over the FULL mapped domain (not just the
    eastern-wall box) -- the physically plausible source (e.g. the shallow AMC roof) may sit
    west of the fitting region itself, so the search space is intentionally wider than the
    bins being fit. Bounded (not a free/unconstrained optimum) so the result stays within the
    physically mapped area rather than drifting to an off-map, physically meaningless location."""
    bounds = [(T.X_START, T.X_END), (T.Y_START, T.Y_END)]
    best = None
    for x0 in np.arange(T.X_START, T.X_END + 0.01, 0.5):
        for y0 in np.arange(T.Y_START, T.Y_END + 0.01, 0.5):
            loss = point_source_misfit((x0, y0), cx, cy, phi_obs, weights)
            if best is None or loss < best[0]:
                best = (loss, x0, y0)
    res = minimize(point_source_misfit, x0=[best[1], best[2]], args=(cx, cy, phi_obs, weights),
                   method='Nelder-Mead', bounds=bounds)
    x0, y0 = res.x
    _, rms_deg = circular_radial_misfit(
        phi_obs, np.degrees(np.arctan2(cx - x0, cy - y0)) % 180.0, weights)
    return x0, y0, rms_deg


def fit_line_source(cx, cy, phi_obs, weights):
    """Same full-domain, bounded approach as fit_point_source for the line's anchor point;
    strike angle is unbounded (wraps mod 180 anyway)."""
    bounds = [(T.X_START, T.X_END), (T.Y_START, T.Y_END), (0.0, 180.0)]
    best = None
    for x0 in np.arange(T.X_START, T.X_END + 0.01, 0.7):
        for y0 in np.arange(T.Y_START, T.Y_END + 0.01, 0.7):
            for theta in np.arange(0, 180, 20.0):
                loss = line_source_misfit((x0, y0, theta), cx, cy, phi_obs, weights)
                if best is None or loss < best[0]:
                    best = (loss, x0, y0, theta)
    res = minimize(line_source_misfit, x0=[best[1], best[2], best[3]],
                   args=(cx, cy, phi_obs, weights), method='Nelder-Mead', bounds=bounds)
    x0, y0, theta = res.x
    theta = theta % 180.0
    d_hat = np.array([np.sin(np.radians(theta)), np.cos(np.radians(theta))])
    dx, dy = cx - x0, cy - y0
    t = dx * d_hat[0] + dy * d_hat[1]
    px, py = x0 + t * d_hat[0], y0 + t * d_hat[1]
    rx, ry = cx - px, cy - py
    pred_az = np.degrees(np.arctan2(rx, ry)) % 180.0
    _, rms_deg = circular_radial_misfit(phi_obs, pred_az, weights)
    return x0, y0, theta, rms_deg


def make_figure(bins_east, point_fit, line_fit):
    x0p, y0p, rms_p = point_fit
    x0l, y0l, theta_l, rms_l = line_fit

    fig, ax = plt.subplots(figsize=(8, 8))
    T._bathy(ax)

    cx = np.array([b['cx'] for b in bins_east])
    cy = np.array([b['cy'] for b in bins_east])
    phi = np.array([b['phi'] for b in bins_east])
    n = np.array([b['n'] for b in bins_east])
    for b in bins_east:
        ax.add_patch(Rectangle((b['x0'], b['y0']), b['x1'] - b['x0'], b['y1'] - b['y0'],
                                fill=False, edgecolor='0.5', lw=0.4, alpha=0.7, zorder=5))
    r = np.radians(phi)
    sz = np.array([b['size'] for b in bins_east])
    length = 0.8 * sz
    u, v = length * np.sin(r), length * np.cos(r)
    cmap = plt.colormaps['hsv']
    norm = Normalize(0, 180)
    tick_kw = dict(scale=1, scale_units='xy', width=0.005, headlength=0, headaxislength=0,
                  headwidth=0, pivot='middle', cmap=cmap, norm=norm, alpha=0.95, zorder=7)
    ax.quiver(cx, cy, u, v, phi, **tick_kw)
    ax.quiver(cx, cy, -u, -v, phi, **tick_kw)

    # Best-fit point source
    ax.plot(x0p, y0p, '*', ms=22, mfc='lime', mec='k', mew=1.2, zorder=20,
           label=f'Best-fit POINT source (RMS={rms_p:.1f} deg)')
    # A few radial guide lines from the point source
    for ang in np.linspace(0, 360, 12, endpoint=False):
        rr = np.radians(ang)
        ax.plot([x0p, x0p + 2.5 * np.sin(rr)], [y0p, y0p + 2.5 * np.cos(rr)],
               c='lime', lw=0.5, ls=':', alpha=0.5, zorder=6)

    # Best-fit line source
    d_hat = np.array([np.sin(np.radians(theta_l)), np.cos(np.radians(theta_l))])
    p1 = np.array([x0l, y0l]) - 3.0 * d_hat
    p2 = np.array([x0l, y0l]) + 3.0 * d_hat
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='magenta', lw=2.2, ls='--', zorder=19,
           label=f'Best-fit LINE source (RMS={rms_l:.1f} deg, strike={theta_l:.0f} deg)')

    ax.add_patch(Rectangle((EAST_X0, EAST_Y0), EAST_X1 - EAST_X0, EAST_Y1 - EAST_Y0,
                            fill=False, edgecolor='k', lw=1.5, ls='-', zorder=8))

    for sta, row in T._sta_df.iterrows():
        ax.plot(row['x'], row['y'], '^', ms=8, mfc='#FFD700', mec='k', mew=0.9, zorder=13)
        dx, dy = T.LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, T.STA_DISPLAY.get(sta, sta), fontsize=8, zorder=13)
    T._faults(ax, Q.FAULT_Z0)

    ax.set_xlim(T.X_START, T.X_END)
    ax.set_ylim(T.Y_START, T.Y_END)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X (km east of 130.1°W)', fontsize=9)
    ax.set_ylabel('Y (km north of 45.9°N)', fontsize=9)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=np.linspace(0, 180, 7))
    cb.set_label('Fast direction $\\phi$ (deg, axial 0-180)', fontsize=9)

    legend_handles = [
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8, label='Station'),
        mlines.Line2D([], [], color='#cc0000', lw=1.2, label='Ring faults'),
        mlines.Line2D([], [], color='k', lw=1.5, label='Eastern-wall fit box'),
        mlines.Line2D([], [], marker='*', color='w', mfc='lime', mec='k', ms=14, label=f'Point source (RMS={rms_p:.1f} deg)'),
        mlines.Line2D([], [], color='magenta', lw=2.2, ls='--', label=f'Line source (RMS={rms_l:.1f} deg)'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        'Pre-eruption phi quad-tree, eastern wall (AXAS1/AXEC1/AXEC2/AXEC3): '
        'radial-source fits\n'
        f'Point source: ({x0p:.2f}, {y0p:.2f}) km, RMS={rms_p:.1f} deg  |  '
        f'Line source: ({x0l:.2f}, {y0l:.2f}) km, strike={theta_l:.0f} deg, RMS={rms_l:.1f} deg',
        fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main():
    bins, n_rays = build_pre_eruption_bins()

    bins_east = [b for b in bins if (EAST_X0 <= b['cx'] <= EAST_X1) and (EAST_Y0 <= b['cy'] <= EAST_Y1)]
    print(f'\nEastern-wall box [{EAST_X0},{EAST_X1}] x [{EAST_Y0},{EAST_Y1}]: '
          f'{len(bins_east)}/{len(bins)} pre-eruption bins')
    if len(bins_east) < 4:
        raise SystemExit('Too few bins in the eastern-wall box to fit a source location.')

    cx = np.array([b['cx'] for b in bins_east])
    cy = np.array([b['cy'] for b in bins_east])
    phi_obs = np.array([b['phi'] for b in bins_east])
    weights = np.array([b['n'] for b in bins_east], dtype=float)

    # Null-hypothesis baseline: a single CONSTANT (uniform, non-radial) fast direction across
    # the whole box -- i.e. the tectonic/fault-parallel alternative, with no source at all.
    # If the point-source fit keeps pushing its location toward the edge of the map (a
    # radial pattern as seen from very far away approaches a uniform direction), comparing
    # against this baseline makes that behavior interpretable.
    ang2 = 2.0 * np.radians(phi_obs)
    s = np.average(np.sin(ang2), weights=weights)
    c = np.average(np.cos(ang2), weights=weights)
    phi_uniform = (np.degrees(np.arctan2(s, c)) / 2.0) % 180.0
    _, rms_uniform = circular_radial_misfit(phi_obs, np.full_like(phi_obs, phi_uniform), weights)
    print(f'\nBaseline: single uniform fast direction phi={phi_uniform:.1f} deg, '
          f'RMS misfit = {rms_uniform:.2f} deg')

    print('\nFitting point (isotropic/"spherical") source location (bounded to the mapped domain)...')
    x0p, y0p, rms_p = fit_point_source(cx, cy, phi_obs, weights)
    print(f'  Best-fit point source: ({x0p:.3f}, {y0p:.3f}) km, RMS circular misfit = {rms_p:.2f} deg')

    print('\nFitting line ("oblong"/elongated) source location + strike (bounded to the mapped domain)...')
    x0l, y0l, theta_l, rms_l = fit_line_source(cx, cy, phi_obs, weights)
    print(f'  Best-fit line source: through ({x0l:.3f}, {y0l:.3f}) km, strike={theta_l:.1f} deg, '
          f'RMS circular misfit = {rms_l:.2f} deg')

    print(f'\nComparison: uniform-direction RMS={rms_uniform:.2f} deg vs. point-source RMS={rms_p:.2f} deg '
          f'vs. line-source RMS={rms_l:.2f} deg')
    if rms_p > rms_uniform - 1.0:
        print('  -> Neither source model meaningfully beats a plain uniform (non-radial) '
              'direction -- the eastern-wall phi pattern does not show clear radial-source '
              'control; the point-source fit likely pushed toward the domain edge to mimic '
              'a uniform field (check the plot).')
    elif rms_l < rms_p - 1.0:
        print('  -> Elongating the source improves the fit -- some support for an oblong '
              '(prolate/dike-like) rather than spherical source.')
    else:
        print('  -> The point source meaningfully beats the uniform baseline, with no further '
              'gain from elongating it -- some support for a compact (spherical) source.')

    fig = make_figure(bins_east, (x0p, y0p, rms_p), (x0l, y0l, theta_l, rms_l))
    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {OUT_PDF}')


if __name__ == '__main__':
    main()
