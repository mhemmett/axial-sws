#!/usr/bin/env python3
"""
animate_arctan_stress_vectors.py

Animated GIF illustrating the SAME atan2 vector-sum model fit in
atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.py (per explicit user request: "use these
functions to remake our gifs") -- AXEC2 and AXCC1 here; AXAS1 (real record starts partway
through, needs gap reconstruction) is handled by
animate_arctan_stress_vectors_axas1_reconstructed.py instead.

FITTED CURVE -- fit_atan2_vectorsum (below), moved here from the atan2 PDF script (that script
now imports it FROM here instead, to avoid duplicating it and to avoid a circular import,
since it already imports compute_optimal_wrap/compute_full_range_ticks from here):

    infl_mag(u) = (u-u0)/A - A*cos(alpha-beta)
    Y(u) = A*sin(alpha) + infl_mag(u)*sin(beta)
    X(u) = A*cos(alpha) + infl_mag(u)*cos(beta)
    phi(u_z) = C1 + C2*atan2(Y(u_z), X(u_z))

alpha=170 deg (ALPHA_FIXED_AZ_DEG) and u0 (U_Z0, per station: 0.6 AXEC2, 1.0 AXCC1, 0.0 AXAS1)
are FIXED; (C1, C2, A, beta) are fit freely. See fit_atan2_vectorsum's docstring for the full
derivation/history of why u0 is placed via infl_mag's closed form (guarantees the curve's
actual steepest transition lands exactly at u0, for ANY A) rather than a simpler-looking but
wrong "magnitude crossover" version tried earlier.

BIG SIMPLIFICATION vs. the previous (plain-arctan) version of this script: since the fitted
model's Y(u)/X(u) ARE a literal 2D vector sum (background A@alpha + inflation infl_mag(u)@beta,
both already in real compass terms -- no separate "rot" rotation needed, unlike the old
version's local-frame-then-rotate construction), the vector-diagram arrows below are drawn
directly from the fit's own real geometry. No per-frame root-finding (the old
solve_inflation_magnitude) and no running-max monotonicity patch are needed any more:
infl_mag(u) is already an EXACT LINEAR function of u (slope 1/A, always positive since A>0 is
bounded away from zero), so it is automatically, exactly monotonic non-decreasing by
construction -- the earlier flip bug (a fixed-direction vector's angle only spans an open ~180
deg arc, causing the old per-frame solver to jump between the two asymptotic extremes for an
unreachable target) cannot happen here since there is no solving step at all.

LEFT panel vector diagram (illustrative -- see caveat below): a real East-North stress plane
centered on the station's real location, three arrows:
  1. Background (dashed, blue): FIXED magnitude A, FIXED at the real compass azimuth alpha
     (ALPHA_FIXED_AZ_DEG=170, SAME for every station -- matching the atan2 PDF script).
  2. Inflation (dashed, orange): FIXED direction beta (this station's own fitted azimuth),
     magnitude infl_mag(u) -- exactly linear in u_z, hence automatically monotonic.
  3. Combined (solid, red): the literal vector sum of (1) and (2).
CAVEAT, important: the combined arrow's own raw geometric compass angle (atan2 of its (X,Y)
position) is NOT the same number as the displayed phi(u_z) text/the right panel's dot, because
the model applies an affine transform (C1 + C2*raw_angle) on top of that raw angle -- C2 is a
free amplitude, not fixed at a value that would make the two coincide. The arrows show the
REAL fitted background/inflation stress vectors; the bold text and the right-panel dot show
the REAL fitted phi(u_z) value; these are two different (related, but not numerically equal)
outputs of the same fit, and both are now exact/genuine rather than back-solved to match.

RIGHT panel: the real (u_z, phi) scatter (blue) + the fitted curve (black dashed) + a RED DOT
sliding along it at the current frame's u_z, PLUS a horizontal dashed red guide line, PLUS a
vertical line at u0 -- now the curve's actual, exact steepest-transition point (see above), not
just a cosmetic reference.

Produces (2 GIFs):
    arctan_stress_vectors_AXEC2.gif
    arctan_stress_vectors_AXCC1.gif

Run with:
    python3 animate_arctan_stress_vectors.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import tifffile
from PIL import Image as PILImage

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from rose_7period_6stations_newdata_snr_grades import _draw_rose
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
)
from scipy.optimize import curve_fit

from arctan_uplift_vs_phi_axec2_axcc1_axas1_90day import STATIONS, UPLIFT_ROLLING_DAYS

HERE = os.path.dirname(os.path.abspath(__file__))

# Local flat-Earth km-East/North projection, and the bathymetry/station-marker conventions,
# matching sws_mesh_regional.py / sws_forward_model.py elsewhere in this repo -- reused here
# (not reinvented) so the vector panel's real-map framing matches the rest of the codebase.
INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))
STATION_COORDS_FILE = os.path.join(HERE, os.pardir, 'data', 'stations_axial.llz')
BATHY_TIF = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
             'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')

# infl_mag(u) (see fit_atan2_vectorsum) is now an EXACT LINEAR function of u, so unlike the
# earlier per-frame-solved version, there is no risk of a huge/degenerate magnitude at any
# single frame. What DOES vary a lot is the fitted A itself across stations (e.g. ~0.3 for
# AXEC2 vs. ~440 for AXCC1), which would make a single fixed physical-km scale look absurd for
# one station or invisible for another. Instead, TARGET_MAX_EXTENT_KM fixes the DISPLAYED
# zoom's half-width (post-scaling) the same for every station, and make_gif solves the actual
# per-station scale factor from the real (station-specific) geometric extent -- see its
# comments. This replaces the earlier fixed VECTOR_SCALE_KM + MAG_CAP pair (no longer needed
# now that there's no numerically-solved magnitude to cap).
TARGET_MAX_EXTENT_KM = 2.5

ROSE_WINDOW_DAYS = 90
COLOR_ROSE = '#0072B2'

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9

N_FRAMES = 90
FPS = 15

COLOR_BG = '#0072B2'      # background (tectonic) vector -- dashed
COLOR_INFL = '#FF7F0E'    # inflation vector -- dashed
COLOR_COMBINED = '#D62728'  # combined vector, right-panel dot/guide-line -- solid / marker

STATION_NAMES = ('AXEC2', 'AXCC1')

BOUND = 1000.0
A_MIN = 0.01

# Per explicit user request: alpha (the background vector's compass azimuth) is FIXED, the
# SAME value for all three stations -- not fit (matches
# atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.py's ALPHA_FIXED_AZ_DEG exactly -- this script no
# longer uses its own separate AZIMUTH_DEG per-station values from an earlier version).
#
# IMPORTANT, per explicit user correction: 170 deg was established in the atan2 PDF script,
# which fits RAW COMPRESSIONAL phi_az (no shift) -- so 170 is a compression-frame azimuth. The
# wrapped y-data fed into curve_fit is mathematically IDENTICAL whether or not the -90
# extension shift is applied before wrapping (compute_optimal_wrap exactly absorbs any
# constant shift -- verified numerically), so the FIT itself (beta/A/C1/C2) cannot tell
# compression and extension framings apart. But alpha is a separately-FIXED constant, never
# routed through that wrap-absorption -- it is a real, independent map-frame azimuth assumed
# directly. Reusing 170 (a compression-frame azimuth) unchanged for the EXTENSION-framed fits
# in this script and in animate_arctan_stress_vectors_axas1_reconstructed.py /
# spatial_inflation_direction_lines_3stations.py was therefore wrong: compression and
# extension stress axes are always orthogonal, so the correct extension-frame azimuth is
# 170-90=80 (axially equivalent to 260) -- ALPHA_EXTENSION_AZ_DEG below. Any fit result that
# used ALPHA_FIXED_RAD against extension-shifted data (an earlier version of this script) is
# suspect -- likely explains an oddly large fitted A (11-440) compensating for the wrong fixed
# background direction.
ALPHA_FIXED_AZ_DEG = 170.0
ALPHA_FIXED_RAD = np.radians(90.0 - ALPHA_FIXED_AZ_DEG)

ALPHA_EXTENSION_AZ_DEG = ALPHA_FIXED_AZ_DEG - 90.0
ALPHA_EXTENSION_RAD = np.radians(90.0 - ALPHA_EXTENSION_AZ_DEG)

# Per explicit user request/correction: the "turnover" u_z (the curve's actual steepest,
# visually-apparent transition point -- see fit_atan2_vectorsum's docstring for why this is
# NOT simply a magnitude crossover) is pinned EXACTLY to these absolute u_z values for every
# station, including AXAS1's "true zero" (see that docstring's TURNOVER note for the
# multi-iteration history of getting this right).
U_Z0 = {'AXEC2': 0.6, 'AXCC1': 1.0, 'AXAS1': 0.0}


def azimuth_to_math_angle(azimuth_deg):
    """Azimuth (deg, clockwise from north) -> standard math angle (rad, counterclockwise
    from +x/east) for plotting in an East-x/North-y plane."""
    return np.radians(90.0 - azimuth_deg)


def math_angle_to_display(math_angle_rad):
    """Standard math angle (rad, ccw from east) -> a real compass reading for display, wrapped
    to (0,180) -- extension/fast direction is a LINE, not a vector (a direction and its +180
    deg opposite are the physically same state), for every station, not just AXCC1/AXAS1
    (an earlier version only wrapped those two, following this script family's pre-existing
    wrap_axis flag from an unrelated page's display convenience -- that flag was never a
    statement about which stations are physically axial; all of them are)."""
    azimuth = (90.0 - np.degrees(math_angle_rad)) % 360.0
    return azimuth % 180.0


def _azimuth(math_angle_rad):
    """Inverse of azimuth_to_math_angle -- math angle (rad, 0=East, CCW+) -> real compass
    azimuth (deg, clockwise from North), NOT folded mod 180, since alpha/beta are genuine
    directed vectors, not axial "lines" like phi itself (contrast math_angle_to_display,
    above, which IS folded mod 180, for phi/axial quantities)."""
    return (90.0 - np.degrees(math_angle_rad)) % 360.0


# Per explicit user request: subtract 90 deg from EVERY station's fast direction (this is a
# genuine physical DATA transformation -- fast direction -> principal EXTENSION direction,
# not a display convenience) -- applied once here, at the data source, so both panels (the
# right panel's phi values and the left panel's vector angles, which are derived from the
# same fitted phi0/phi1) reflect it consistently.
EXTENSION_SHIFT_DEG = {'AXEC2': -90.0, 'AXCC1': -90.0, 'AXAS1': -90.0}

def compute_optimal_wrap(raw_values):
    """Per explicit user request: 'the data shouldn't wrap about the top or bottom of the
    plot -- if it does, shift the axes.' Finds the wrap offset w such that
    y=(raw_values-w)%180 places the data's LARGEST angular gap at the 0/180 boundary,
    guaranteeing the main cluster stays contiguous (no artificial split at the plot's
    top/bottom edge) -- computed per station from its own data, replacing an earlier fixed
    guess (Y_AXIS_WRAP_DEG=90 for everyone) that worked for AXCC1/AXAS1 but happened to
    still wrap AXEC2's cluster once its data was also extension-shifted."""
    v = np.sort(np.asarray(raw_values) % 180.0)
    if len(v) < 2:
        return 0.0
    circ = np.concatenate([v, [v[0] + 180.0]])
    gaps = np.diff(circ)
    idx = int(np.argmax(gaps))
    return float((v[idx] + gaps[idx] / 2.0) % 180.0)


def break_wrapped_line(y_line, threshold=90.0):
    """For plotting only: y_line is the model curve after %180 wrapping, which introduces a
    genuine vertical jump wherever the raw (unwrapped) model output crosses a multiple of
    180 -- correct data-wise, but a plotted line connecting straight across that jump reads
    as a rendering glitch. Insert NaN at each jump so matplotlib breaks the line there
    instead of drawing a spurious near-vertical connector."""
    y_line = y_line.copy()
    jumps = np.flatnonzero(np.abs(np.diff(y_line)) > threshold)
    y_line[jumps + 1] = np.nan
    return y_line


def compute_full_range_ticks(wrap, tick_step=20.0):
    """Right panel y-axis: always the FULL 0-180 degree span (per explicit user request --
    not a data-centered/cropped window, which on a station whose data happens to cluster
    tightly can end up looking like a "0-140" or "0-90" axis). Ticks are still built by
    choosing round labels directly in TRUE-DEGREE space first, then mapping each back to its
    wrapped-space plot position -- not the reverse (evenly-spaced wrapped positions relabeled
    afterward), which has its own unrelated discontinuity (see the git history of this
    function). Labels 0..160 only (step 20, excluding 180 itself, which is the same axial
    direction as 0 and would otherwise land a second, identically-worded tick at the same
    wrapped position). Showing the full mod-180 domain on a linear axis necessarily has
    exactly one wraparound point somewhere in view (e.g. ...,160,0,20,...) -- an intrinsic,
    expected feature of displaying circular/axial data linearly, not a bug, as long as each
    side of that single transition is itself monotonic (guaranteed here since positions are
    derived directly from sorted round labels)."""
    y_lo, y_hi = 0.0, 180.0
    labels = np.arange(0.0, 180.0, tick_step)
    positions = (labels - wrap) % 180.0
    order = np.argsort(positions)
    tick_positions = positions[order]
    tick_labels = [int(round(labels[i])) for i in order]
    return y_lo, y_hi, tick_positions, tick_labels


def fit_atan2_vectorsum(x, y, u0):
    """phi(u) = C1 + C2*atan2(Y,X), Y=A*sin(alpha)+infl_mag(u)*sin(beta),
    X=A*cos(alpha)+infl_mag(u)*cos(beta), with alpha and u0 FIXED (u0 passed in per station,
    alpha=ALPHA_FIXED_RAD for all stations). C1/C2 (the output offset and atan2-scale
    amplitude) are fit alongside A and beta.

    infl_mag(u) = (u-u0)/A - A*cos(alpha-beta) -- places the curve's actual steepest/visually-
    apparent transition point EXACTLY at u=u0 for ANY nonzero A (derivation: the (X,Y) position
    is A*[cos a,sin a] + infl_mag*[cos b,sin b], a point moving along a line as infl_mag varies;
    the curve's steepest slope occurs where that point is CLOSEST to the origin -- a standard
    result for the angular velocity of a point on a line as seen from an external pivot --
    which calculus gives as infl_mag*=-A*cos(a-b); solving infl_mag(u0)=infl_mag* for the shift
    gives the form above). An earlier version used (u-u0+A^2)/A instead, which places the
    background/inflation MAGNITUDE crossover at u0 -- a different point in general (they only
    coincide when beta=alpha+180), and for AXAS1 confirmed to visibly mislabel the turnover
    (axvline at u=0, curve's real transition near u=0.4). A is fit FREELY for every station,
    bounded away from zero (A_MIN) since it is also the divisor above. Free parameters:
    (C1, C2, A, beta), bounded to [-BOUND,BOUND]/[-BOUND,BOUND]/[A_MIN,BOUND]/[-2*pi,2*pi].

    Multi-start over C1 (which end of the data it anchors to), C2's sign/scale, A's scale, and
    beta spanning the full circle. Keeps the lowest-SSE fit. Returns (C1, C2, A, beta, r,
    model) -- model always takes (u, C1, C2, A, beta)."""
    def model(u, C1, C2, A, beta):
        infl_mag = (u - u0) / A - A * np.cos(ALPHA_FIXED_RAD - beta)
        Y = A * np.sin(ALPHA_FIXED_RAD) + infl_mag * np.sin(beta)
        X = A * np.cos(ALPHA_FIXED_RAD) + infl_mag * np.cos(beta)
        return C1 + C2 * np.arctan2(Y, X)

    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    y_span = float(y_hi - y_lo) or 1.0
    C1_cands = [y_lo, y_hi, float(np.mean(y))]
    C2_cands = [v for s in (1.0, -1.0) for v in (s * y_span / np.pi, s * y_span / (2 * np.pi))]
    A_cands = [0.3, 1.0, 3.0]
    beta_cands = np.radians(np.linspace(0.0, 300.0, 6))

    bounds = ([-BOUND, -BOUND, A_MIN, -2 * np.pi], [BOUND, BOUND, BOUND, 2 * np.pi])

    best = None
    for C1_0 in C1_cands:
        for C2_0 in C2_cands:
            for A0 in A_cands:
                for beta0 in beta_cands:
                    p0 = [C1_0, C2_0, A0, beta0]
                    try:
                        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
                    except RuntimeError:
                        continue
                    y_pred = model(x, *popt)
                    sse = float(np.sum((y_pred - y) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, popt)
    C1, C2, A, beta = best[1]
    y_pred = model(x, C1, C2, A, beta)
    r = np.corrcoef(y_pred, y)[0, 1]
    return C1, C2, A, beta, r, model


def invert_atan2_vectorsum(y_target, C1, C2, A, beta, u0, u_bound=5.0, margin=1e-6):
    """Solve for u such that C1 + C2*atan2(Y(u),X(u)) = y_target (alpha fixed at
    ALPHA_FIXED_RAD, all other params including u0 fixed) -- used to reconstruct AXAS1's
    2015-2017 gap. u_bound caps the search to a PHYSICALLY PLAUSIBLE de-tided-uplift range
    (real data here never exceeds a few meters) -- an unbounded scan can cross the same target
    value again at a numerically-real but physically-nonsensical u_z (confirmed once: a gap
    point inverted to u_z=-63 m). Returns NaN where y_target falls outside the achievable range
    within that bound (no sign change found)."""
    def f(u):
        infl_mag = (u - u0) / A - A * np.cos(ALPHA_FIXED_RAD - beta)
        Y = A * np.sin(ALPHA_FIXED_RAD) + infl_mag * np.sin(beta)
        X = A * np.cos(ALPHA_FIXED_RAD) + infl_mag * np.cos(beta)
        return C1 + C2 * np.arctan2(Y, X) - y_target

    u_candidates = u0 + np.concatenate([-np.logspace(-3, np.log10(u_bound), 300), [0.0],
                                        np.logspace(-3, np.log10(u_bound), 300)])
    vals = np.array([f(u) for u in u_candidates])
    for i in range(len(u_candidates) - 1):
        if abs(vals[i]) < margin:
            return float(u_candidates[i])
        if vals[i] * vals[i + 1] < 0:
            try:
                from scipy.optimize import brentq
                return float(brentq(f, u_candidates[i], u_candidates[i + 1]))
            except Exception:
                continue
    return float('nan')


def fit_atan2_vectorsum_joint(datasets, alpha_rad):
    """Per explicit user request: jointly fit ALL stations in `datasets` with a single SHARED
    background magnitude A (one real background stress magnitude, common to every station),
    while (C1, C2, beta) stay free PER STATION. alpha is fixed (passed in explicitly by the
    caller -- see the ALPHA_EXTENSION_AZ_DEG note above for why this must be the correctly
    framed value, not blindly reused from a different phi convention).

    `datasets` is a list of (name, x, y, u0) tuples (same x/y/u0 as fit_atan2_vectorsum's own
    arguments, one triple per station). Solved via scipy.optimize.least_squares on the
    stacked (concatenated) residuals across all stations, with a shared parameter vector
    [A, C1_0, C2_0, beta_0, C1_1, C2_1, beta_1, ...]. Multi-start over A's scale, a shared
    sign flip for every station's C2 (since C2's sign is otherwise a per-station symmetry
    that would blow up the combinatorics if searched independently), and each station's beta
    spanning a coarse compass grid -- keeps the lowest total-SSE solution across all stations
    combined (not the best per-station SSE individually, which is the whole point of sharing
    A: some per-station fit quality may be sacrificed for a single common magnitude).

    Returns (A, per_station) where per_station is {name: (C1, C2, beta, r, model)} -- model
    always takes (u, C1, C2, A, beta), matching fit_atan2_vectorsum's own model signature, so
    downstream code doesn't need to know whether a joint or per-station fit produced it."""
    from scipy.optimize import least_squares

    n = len(datasets)

    def station_model(u, C1, C2, A, beta):
        infl_mag = (u - datasets_u0[_i]) / A - A * np.cos(alpha_rad - beta)
        Y = A * np.sin(alpha_rad) + infl_mag * np.sin(beta)
        X = A * np.cos(alpha_rad) + infl_mag * np.cos(beta)
        return C1 + C2 * np.arctan2(Y, X)

    def unpack(params):
        A = params[0]
        per = [tuple(params[1 + 3 * i:4 + 3 * i]) for i in range(n)]
        return A, per

    def residuals(params):
        A, per = unpack(params)
        out = []
        for (name, x, y, u0), (C1, C2, beta) in zip(datasets, per):
            infl_mag = (x - u0) / A - A * np.cos(alpha_rad - beta)
            Y = A * np.sin(alpha_rad) + infl_mag * np.sin(beta)
            X = A * np.cos(alpha_rad) + infl_mag * np.cos(beta)
            out.append((C1 + C2 * np.arctan2(Y, X)) - y)
        return np.concatenate(out)

    lower = [A_MIN] + [-BOUND, -BOUND, -2 * np.pi] * n
    upper = [BOUND] + [BOUND, BOUND, 2 * np.pi] * n

    A_cands = [0.3, 1.0, 3.0]
    c2_sign_cands = [1.0, -1.0]
    beta_cands = np.radians([0.0, 90.0, 180.0, 270.0])

    c1_seed = [float(np.mean(y)) for (_, _, y, _) in datasets]
    c2_seed = [float(np.max(y) - np.min(y)) / np.pi or 1.0 for (_, _, y, _) in datasets]

    best = None
    for A0 in A_cands:
        for c2_sign in c2_sign_cands:
            for beta_combo in np.array(np.meshgrid(*([beta_cands] * n))).T.reshape(-1, n):
                p0 = [A0]
                for i in range(n):
                    p0 += [c1_seed[i], c2_sign * c2_seed[i], float(beta_combo[i])]
                try:
                    result = least_squares(residuals, p0, bounds=(lower, upper), max_nfev=2000)
                except Exception:
                    continue
                sse = float(np.sum(result.fun ** 2))
                if best is None or sse < best[0]:
                    best = (sse, result.x)

    A, per = unpack(best[1])
    per_station = {}
    for (name, x, y, u0), (C1, C2, beta) in zip(datasets, per):
        infl_mag = (x - u0) / A - A * np.cos(alpha_rad - beta)
        Y = A * np.sin(alpha_rad) + infl_mag * np.sin(beta)
        X = A * np.cos(alpha_rad) + infl_mag * np.cos(beta)
        y_pred = C1 + C2 * np.arctan2(Y, X)
        r = np.corrcoef(y_pred, y)[0, 1]

        def model(u, C1=C1, C2=C2, A=A, beta=beta, u0=u0):
            infl_mag = (u - u0) / A - A * np.cos(alpha_rad - beta)
            Y = A * np.sin(alpha_rad) + infl_mag * np.sin(beta)
            X = A * np.cos(alpha_rad) + infl_mag * np.cos(beta)
            return C1 + C2 * np.arctan2(Y, X)

        per_station[name] = (C1, C2, beta, r, model)
    return A, per_station


def load_station_data(station):
    name = station['name']
    raw = load_station_raw(name)
    df = apply_grade(raw, GRADE)

    pre = df[df['t'] < ERUPTION_START]
    baseline_phi, _ = _circular_mean_and_se_deg(pre['phi_az'].values)

    roll = rolling_phi_stats_daily_then_roll(df, baseline_phi, window_days=ROLL_WINDOW_DAYS,
                                             min_days=ROLL_MIN_DAYS)
    valid = roll.dropna(subset=['mean_phi']).copy()
    valid = valid[valid['t'] >= ERUPTION_END]

    _dd, infl_raw, _infl_roll_30, _rt, _rd = station['infl_module'].load_daily_series()
    inflation_roll = infl_raw.rolling(f'{UPLIFT_ROLLING_DAYS}D', center=True,
                                      min_periods=UPLIFT_ROLLING_DAYS // 2).mean()
    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']

    merged = pd.merge_asof(valid.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    x = merged['inflation_m'].values
    y_extension = merged['mean_phi'].values + EXTENSION_SHIFT_DEG[name]
    wrap = compute_optimal_wrap(y_extension)
    y = (y_extension - wrap) % 180.0
    # df (event-level, same GRADE-filtered population as the scatter above) and merged (its
    # uz<->date correspondence) are returned too -- the new rose panel needs the raw per-event
    # phi_az/t values within a rolling window, which rolling_phi_stats_daily_then_roll's
    # output (roll/valid, day-then-rolled MEANS only) no longer contains.
    return x, y, wrap, df, merged


def load_station_xy(name):
    """Station's real (East km, North km) position from this repo's own data/stations_axial.llz
    (NOT the axial-splitting-ml sibling-repo path some other scripts hardcode), in the same
    flat-Earth projection used throughout the repo (see INI_LON/INI_LAT above)."""
    with open(STATION_COORDS_FILE) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[3] == name:
                lon, lat = float(parts[0]), float(parts[1])
                return (lon - INI_LON) * KM_PER_DEG_LON, (lat - INI_LAT) * KM_PER_DEG_LAT
    raise ValueError(f'{name} not found in {STATION_COORDS_FILE}')


def load_bathy_gray(x0, y0, lim):
    """Crop + grayscale the local high-res bathymetry GeoTIFF to a square km window centered
    on (x0,y0) [km East/North of INI_LON,INI_LAT], returning (gray_array, extent_km) for
    ax.imshow. Ported from sws_mesh_regional.py's module-level bathymetry setup (same GeoTIFF
    tag-based georeferencing, same downsample-to-<=1024px, same grayscale luminance dot
    product), parameterized per-call instead of computed once for a fixed shared caldera-wide
    box, since each of this script's GIFs needs a small window around its own single station."""
    lon_min, lon_max = INI_LON + (x0 - lim) / KM_PER_DEG_LON, INI_LON + (x0 + lim) / KM_PER_DEG_LON
    lat_min, lat_max = INI_LAT + (y0 - lim) / KM_PER_DEG_LAT, INI_LAT + (y0 + lim) / KM_PER_DEG_LAT
    PILImage.MAX_IMAGE_PIXELS = None
    p = PILImage.open(BATHY_TIF)
    t = p.tag_v2
    olon, olat = t[33922][3], t[33922][4]
    pl, pb = t[33550][0], t[33550][1]
    nc, nr = p.size
    p.close()
    c0 = max(0, int((lon_min - olon) / pl) - 2)
    c1 = min(nc, int((lon_max - olon) / pl) + 2)
    r0 = max(0, int((olat - lat_max) / pb) - 2)
    r1 = min(nr, int((olat - lat_min) / pb) + 2)
    rgb = tifffile.imread(BATHY_TIF)[r0:r1, c0:c1]
    ds = max(1, max(rgb.shape[:2]) // 1024)
    rgb = rgb[::ds, ::ds]
    gray = np.dot(rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
    extent = [(olon + c0 * pl - INI_LON) * KM_PER_DEG_LON, (olon + c1 * pl - INI_LON) * KM_PER_DEG_LON,
             (olat - r1 * pb - INI_LAT) * KM_PER_DEG_LAT, (olat - r0 * pb - INI_LAT) * KM_PER_DEG_LAT]
    return gray, extent


def phi_window(df, center, half_days=ROSE_WINDOW_DAYS / 2.0):
    """Raw (unaveraged) fast-direction phi_az values from df with timestamps within
    +/- half_days of center -- the per-event population underlying the rose panel, matching
    the same ROSE_WINDOW_DAYS width as the right panel's rolling window, but at the event
    level (not day-then-rolled) since a rose plot needs the actual within-window distribution,
    not just its mean."""
    lo = center - pd.Timedelta(days=half_days)
    hi = center + pd.Timedelta(days=half_days)
    sel = df[(df['t'] >= lo) & (df['t'] < hi)]
    return sel['phi_az'].values, lo, hi


def draw_rose_mean_line(ax, phi_az_vals, color='red'):
    """Overlay a diameter line through the rose panel's polar axes at the window's circular
    mean fast direction, shifted -90 deg (per explicit user request) so it reads on the same
    principal-EXTENSION-direction convention as the other two panels, not the raw compressional
    one the rose bars themselves use. Drawn as two opposite radial segments (mean and
    mean+180) since this is an axial/line quantity, not a directed vector -- matching how
    every other angle in this script family is treated."""
    if len(phi_az_vals) == 0:
        return
    mean_phi, _se = _circular_mean_and_se_deg(phi_az_vals)
    mean_ext_rad = np.deg2rad((mean_phi - 90.0) % 360.0)
    r_max = ax.get_ylim()[1]
    for theta in (mean_ext_rad, mean_ext_rad + np.pi):
        ax.plot([theta, theta], [0, r_max], color=color, lw=2, zorder=10)




def make_gif(station):
    name = station['name']
    u0 = U_Z0[name]
    print(f'Loading {name}...')
    x, y, wrap, df, merged = load_station_data(station)
    C1, C2, A, beta, r, model = fit_atan2_vectorsum(x, y, u0)
    beta_az = _azimuth(beta)
    print(f'  {name}: C1={C1:.2f}, C2={C2:.2f}, A={A:.3f}, background vector azimuth (alpha, '
         f'fixed)={ALPHA_FIXED_AZ_DEG:.0f}°, inflation vector azimuth (beta)={beta_az:.1f}°, '
         f'turnover u_z={u0:.3f} (exact steepest point), r={r:.2f}, N={len(x)}')

    u_min, u_max = float(x.min()), float(x.max())

    # CRITICAL: the raw model output C1+C2*atan2(...) is NOT confined to [0,180) -- C2 is a
    # free amplitude on top of atan2's own (-pi,pi] range, so it can run well past 180 or
    # below 0 even at real, in-domain u_z. Every place this model's output is treated as a
    # physical phi value must fold it back with %180, exactly like the real data's own
    # y=(...)%180 (see this script family's history for what goes wrong otherwise: a y-axis
    # spanning >180 deg with repeating tick labels, a left-panel angle that doesn't match).
    def model_wrapped(u):
        return model(u, C1, C2, A, beta) % 180.0

    def true_phi(uz):
        """The right panel's TRUE extension-direction value at uz -- same convention as the
        right-panel axis labels, i.e. model_wrapped's WRAPPED-coordinate output with the
        +wrap correction undone."""
        return float((model_wrapped(np.array([uz]))[0] + wrap) % 180.0)

    # The fitted model's Y/X ARE a literal vector sum already in real compass terms (alpha is
    # a real fixed azimuth, not a "local" angle needing a separate rotation) -- see module
    # docstring. infl_mag(u) is EXACT and LINEAR in u (slope 1/A, always positive), so it is
    # automatically, exactly monotonic non-decreasing -- no per-frame solving, no running-max
    # patch, no flip risk, unlike the previous (plain-arctan) version of this script.
    def infl_mag(u):
        return (u - u0) / A - A * np.cos(ALPHA_FIXED_RAD - beta)

    bg_vec = A * np.array([np.cos(ALPHA_FIXED_RAD), np.sin(ALPHA_FIXED_RAD)])

    def vecs(uz):
        infl_vec = infl_mag(uz) * np.array([np.cos(beta), np.sin(beta)])
        return bg_vec, infl_vec, bg_vec + infl_vec

    u_sweep = np.linspace(u_min, u_max, N_FRAMES)

    # A's fitted scale varies wildly across stations (e.g. ~0.3 for AXEC2 vs. ~440 for AXCC1),
    # with no physical km meaning -- so rather than a fixed physical conversion, solve a
    # per-station DISPLAY scale factor that maps this station's own actual geometric extent
    # (over the sweep) onto the same target zoom (TARGET_MAX_EXTENT_KM) for every station.
    raw_max_extent = np.hypot(*bg_vec)
    for uz in u_sweep:
        _, infl_vec, comb_vec = vecs(uz)
        raw_max_extent = max(raw_max_extent, np.hypot(*infl_vec), np.hypot(*comb_vec))
    scale_factor = TARGET_MAX_EXTENT_KM / raw_max_extent if raw_max_extent > 0 else 1.0
    lim = TARGET_MAX_EXTENT_KM * 1.15

    x_line = np.linspace(u_min, u_max, 300)
    y_line = model_wrapped(x_line)

    # -- Real station location (km East/North) and a bathymetry crop centered on it, sized to
    # the same `lim` used for the vector-panel axis limits, so the map always frames the
    # vectors comfortably (see load_bathy_gray/load_station_xy).
    x0, y0 = load_station_xy(name)
    bathy_gray, bathy_extent = load_bathy_gray(x0, y0, lim)

    # -- uz -> real calendar date lookup (nearest by inflation_m), used to window the rose
    # panel's raw phi_az population per frame -- reuses `merged`'s existing (t, inflation_m)
    # correspondence rather than re-deriving one.
    merged_sorted = merged.sort_values('inflation_m')
    lookup_infl = merged_sorted['inflation_m'].values
    lookup_t = merged_sorted['t'].values

    def nearest_date(uz):
        idx = int(np.argmin(np.abs(lookup_infl - uz)))
        # merged_sorted['t'].values strips the tz (returns naive UTC datetime64) -- re-attach
        # it so this compares cleanly against df['t'], which stays tz-aware.
        return pd.Timestamp(lookup_t[idx]).tz_localize('UTC')

    fig = plt.figure(figsize=(16, 5.5))
    axRose = fig.add_subplot(1, 3, 1, projection='polar')
    axL = fig.add_subplot(1, 3, 2)
    axR = fig.add_subplot(1, 3, 3)

    # -- Right panel: static scatter + fit curve, set up once. y_line_plot breaks the line at
    # %180 wrap discontinuities (see break_wrapped_line) so it doesn't draw a spurious
    # near-vertical connector there.
    y_line_plot = break_wrapped_line(y_line)
    axR.scatter(x, y, s=14, color='#0072B2', alpha=0.5, zorder=2)
    axR.plot(x_line, y_line_plot, color='black', lw=1.5, linestyle='--', zorder=3,
            label=r'$\phi(u_z)=C_1+C_2\mathrm{atan2}(Y,X)$'
                 r'$,\ Y=A\sin\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\sin\beta,\ $'
                 r'$X=A\cos\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\cos\beta$'
                 f'\n$C_1$={C1:.2f}, $C_2$={C2:.2f}, A={A:.3f} (all free)\n'
                 f'background vector azimuth α={ALPHA_FIXED_AZ_DEG:.0f}° (fixed), inflation '
                 f'vector azimuth β={beta_az:.1f}°')
    axR.axvline(u0, color='gray', lw=0.6, linestyle=':', zorder=1,
               label=f'Turnover ($u_z$={u0:.3f}, exact steepest point)')
    dotR, = axR.plot([], [], marker='o', markersize=10, color=COLOR_COMBINED, zorder=5)
    guideR, = axR.plot([], [], color=COLOR_COMBINED, lw=1.2, linestyle='--', zorder=4)
    axR.set_xlabel(f'De-tided uplift $u_z$ (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
    axR.set_ylabel(f'Principal extension direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)')

    y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
    axR.set_ylim(y_lo, y_hi)
    axR.set_yticks(tick_positions)
    axR.set_yticklabels([str(v) for v in tick_labels])
    axR.set_title(f'{name}: uplift vs. fast direction\n(atan2 vector-sum fit, r = {r:.2f})',
                 fontsize=10, fontweight='bold')
    axR.legend(loc='lower left', fontsize=7, framealpha=0.9)
    axR.grid(alpha=0.3)
    x_axis_min, x_axis_max = axR.get_xlim()

    # -- Middle panel: vector diagram on real bathymetry, centered on the station's real
    # location -- static frame + dynamic arrows. Unlike the previous version of this script,
    # these are the REAL fitted background/inflation vectors (see module docstring's CAVEAT
    # about why the combined arrow's own geometric angle isn't the same number as the
    # displayed phi text).
    axL.imshow(bathy_gray, origin='upper', extent=bathy_extent, aspect='auto', cmap='gray',
              alpha=0.5, zorder=0)
    axL.set_xlim(x0 - lim, x0 + lim)
    axL.set_ylim(y0 - lim, y0 + lim)
    axL.set_aspect('equal')
    axL.plot(x0, y0, marker='^', markersize=10, mfc='#FFD700', mec='k', mew=1, zorder=12)
    axL.annotate(name, (x0, y0), textcoords='offset points', xytext=(8, 8),
                fontsize=10, fontweight='bold', zorder=13)
    axL.set_title('Background + inflation stress vectors (real fit geometry)',
                 fontsize=10, fontweight='bold')
    axL.set_xlabel('East (km)')
    axL.set_ylabel('North (km)')

    # Small static compass arrow, purely a reference -- unrelated to the dynamic vectors.
    axL.annotate('', xy=(0.05, 0.92), xytext=(0.05, 0.80), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.2), zorder=10)
    axL.annotate('N', (0.05, 0.94), xycoords='axes fraction', ha='center', fontsize=8,
                color='0.3', zorder=10)

    arrow_bg = axL.annotate('', xy=(x0 + bg_vec[0] * scale_factor, y0 + bg_vec[1] * scale_factor),
                            xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='-|>', color=COLOR_BG, lw=2,
                                            linestyle='--'), zorder=14)
    arrow_infl = axL.annotate('', xy=(x0, y0), xytext=(x0, y0),
                              arrowprops=dict(arrowstyle='-|>', color=COLOR_INFL, lw=2,
                                              linestyle='--'), zorder=14)
    arrow_comb = axL.annotate('', xy=(x0, y0), xytext=(x0, y0),
                              arrowprops=dict(arrowstyle='-|>', color=COLOR_COMBINED, lw=2.5),
                              zorder=15)

    angle_text = axL.text(0.97, 0.03, '', transform=axL.transAxes, ha='right', va='bottom',
                          fontsize=11, fontweight='bold', color=COLOR_COMBINED, zorder=16)

    legend_handles = [
        Line2D([0], [0], color=COLOR_BG, lw=2, linestyle='--',
              label=f'Background stress (azimuth α={ALPHA_FIXED_AZ_DEG:.0f}° fixed, mag A={A:.3f})'),
        Line2D([0], [0], color=COLOR_INFL, lw=2, linestyle='--',
              label=f'Inflation stress (azimuth β={beta_az:.1f}° fixed, mag grows linearly '
                    f'in $u_z$)'),
        Line2D([0], [0], color=COLOR_COMBINED, lw=2.5,
              label='Combined (real vector sum -- its own geometric angle is NOT the '
                    'displayed φ, see title text)'),
    ]
    detail_legend = axL.legend(handles=legend_handles, loc='upper center',
                               bbox_to_anchor=(0.5, -0.18), fontsize=7.5, framealpha=0.9)
    axL.add_artist(detail_legend)

    simple_legend_handles = [
        Line2D([0], [0], color=COLOR_BG, lw=2, linestyle='--',
              label='Regional tectonic stress vector'),
        Line2D([0], [0], color=COLOR_INFL, lw=2, linestyle='--',
              label='Increasing inflation stress vector'),
        Line2D([0], [0], color=COLOR_COMBINED, lw=2.5, label='Effective stress vector'),
    ]
    axL.legend(handles=simple_legend_handles, loc='upper right', fontsize=7.5, framealpha=0.9)

    def draw_rose_frame(center):
        axRose.clear()
        phi_vals, lo, hi = phi_window(df, center)
        _draw_rose(axRose, phi_vals, np.ones(len(phi_vals)), COLOR_ROSE)
        draw_rose_mean_line(axRose, phi_vals)
        axRose.set_title(f'{name}: fast direction rose\n(raw $\\phi$, {ROSE_WINDOW_DAYS}-day '
                         f'window, N={len(phi_vals)})\n{lo.date()} to {hi.date()}',
                         fontsize=9, fontweight='bold')

    def update(frame_idx):
        uz = u_sweep[frame_idx]
        bg, infl, comb = vecs(uz)

        arrow_bg.xy = (x0 + bg[0] * scale_factor, y0 + bg[1] * scale_factor)
        arrow_infl.xy = (x0 + infl[0] * scale_factor, y0 + infl[1] * scale_factor)
        arrow_comb.xy = (x0 + comb[0] * scale_factor, y0 + comb[1] * scale_factor)

        # The displayed angle is the model's OWN true phi(u_z) (matching the right panel's dot
        # exactly), NOT derived from the arrow geometry above -- see module docstring's CAVEAT.
        true_deg = true_phi(uz)
        angle_text.set_text(f'{true_deg:.1f}°')

        phi_now = model_wrapped(np.array([uz]))[0]
        dotR.set_data([uz], [phi_now])
        guideR.set_data([x_axis_min, x_axis_max], [phi_now, phi_now])

        draw_rose_frame(nearest_date(uz))

        return arrow_bg, arrow_infl, arrow_comb, dotR, guideR, angle_text

    # Single-line axis labels still need explicit room reserved at the bottom for the shared
    # footnote, plus tight_layout to give the y-axis label space against the panels' titles --
    # without this the ylabel used to render overlapping the title in every frame.
    fig.text(0.5, 0.01,
             'Fast direction shifted $-90°$ to represent extension, not compression (right '
             'panel); $\\phi$ axis wraps mod 180.',
             ha='center', fontsize=7.5, color='0.35')
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    anim = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
    out_path = os.path.join(HERE, f'arctan_stress_vectors_{name}.gif')
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f'  Saved {out_path}')


def main():
    for station in STATIONS:
        if station['name'] not in STATION_NAMES:
            continue
        make_gif(station)


if __name__ == '__main__':
    main()
