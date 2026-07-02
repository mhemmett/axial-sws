#!/usr/bin/env python3
"""
mogi_stress_model.py

Stress-field model for Axial Seamount shear-wave splitting analysis.

Pipeline (adapted from Baillard 2019):
  1. Two-sphere Mogi model (Kidiwela source) → surface displacement field
  2. Finite-difference strain tensor → Hooke's Law stress tensor
  3. Eigendecompose stress → σ₁ direction (predicted φ) and |σ₁−σ₂| (predicted δt proxy)
  4. Scale each of the 7 time periods by the geodetic elevation record
  5. Compute ray-path lengths from earthquake catalog to stations,
     routing around the Arnulf AMC sphere (Vs=0 barrier)
  6. Compare predicted vs. observed φ and δt for each period × station

Geographic origin: 130.1°W, 45.9°N  (Baillard's reference, used throughout)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh
import os

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT_DIR      = BASE

# ── Coordinate system ─────────────────────────────────────────────────────────

INI_LON        = -130.1
INI_LAT        =  45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


# ── Kidiwela source parameters ────────────────────────────────────────────────
# Two pressurized spheres — geographic origin at (130.1°W, 45.9°N)

MU       = 30e9     # Pa  shear modulus
NU       = 0.25     # Poisson's ratio
LAMBDA   = 2 * NU * MU / (1 - 2 * NU)   # = MU for ν=0.25

SPHERES = [
    dict(x0=7.57, y0=4.55, d=3.33, R=0.43, dP=0.05e9, label='S1'),  # km / GPa
    dict(x0=7.53, y0=6.60, d=1.25, R=0.20, dP=0.05e9, label='S2'),
]

# Caldera centre (km) — used for geodetic calibration
CALDERA_X, CALDERA_Y = 8.0, 5.5   # km

# Kidiwela period for μ calibration
KIDIWELA_START = pd.Timestamp('2022-08-01', tz='UTC')
KIDIWELA_END   = pd.Timestamp('2025-08-01', tz='UTC')

# ── AMC geometry (Arnulf et al. 2014/2018) ────────────────────────────────────
# Simplified sphere; Vs=0 (complete barrier — rays route around it)

AMC = dict(
    x0=8.0,    # km   caldera centre
    y0=5.5,    # km
    z0=2.0,    # km depth (positive downward)
    r=1.5,     # km radius
)

# ── Geodetic anchor points ────────────────────────────────────────────────────
# (date, seafloor elevation at central caldera [m])

GEODETIC = [
    (pd.Timestamp('2015-01-01'), 3.50),   # pre-eruption
    (pd.Timestamp('2015-04-24'), 1.00),   # eruption onset (−2.5 m)
    (pd.Timestamp('2018-01-01'), 2.20),
    (pd.Timestamp('2019-01-01'), 2.60),
    (pd.Timestamp('2021-01-01'), 3.00),
    (pd.Timestamp('2022-01-01'), 3.20),   # Kidiwela start ~ Aug 2022
    (pd.Timestamp('2023-01-01'), 3.30),
    (pd.Timestamp('2024-01-01'), 3.34),
    (pd.Timestamp('2025-01-01'), 3.59),   # +0.25 m
    (pd.Timestamp('2026-01-01'), 3.75),   # +0.16 m
    (pd.Timestamp('2026-05-01'), 3.77),   # +0.02 m
]
GEODETIC_T, GEODETIC_Z = zip(*GEODETIC)
GEODETIC_T = pd.to_datetime(list(GEODETIC_T))
GEODETIC_Z = np.array(GEODETIC_Z)

# Post-eruption baseline (deflated floor) and Kidiwela reference elevation
ELEV_BASELINE   = 1.00   # m  (post-eruption minimum)
ELEV_KIDIWELA   = 3.20   # m  (Aug 2022 start of Kidiwela period)
SCALE_DENOM     = ELEV_KIDIWELA - ELEV_BASELINE   # = 2.2 m


def elevation_at(t):
    """Linearly interpolate geodetic elevation at timestamp t."""
    t_num = pd.Timestamp(t).value
    t_arr = GEODETIC_T.asi8.astype(float)
    return float(np.interp(t_num, t_arr, GEODETIC_Z))


def source_scale(t):
    """Amplitude of the Mogi source at time t, relative to Kidiwela reference."""
    elev = elevation_at(t)
    return (elev - ELEV_BASELINE) / SCALE_DENOM


# ── Mogi displacement (surface, z_obs=0) ─────────────────────────────────────

def _mogi_dV(R_km, dP_Pa, mu=MU, nu=NU):
    """Volume change of a pressurized sphere (Mogi 1958, ν=0.25).

    Note: stress field σ = 2μ·ε is independent of μ because ε ∝ ΔV/μ ∝ 1/μ,
    so μ cancels in σ = 2μ·ε.  Only the displacement amplitude (geodetic fit)
    depends on μ — see calibrate_mu() below.
    """
    R = R_km * 1e3   # → m
    return np.pi * R**3 * dP_Pa / mu


def _uz_at_point(x_km, y_km, mu=MU, nu=NU):
    """Total Uz (m) from both Kidiwela spheres at a single surface point."""
    x_m = x_km * 1e3
    y_m = y_km * 1e3
    uz  = 0.0
    for sph in SPHERES:
        dV = _mogi_dV(sph['R'], sph['dP'], mu=mu, nu=nu)
        C  = dV * (1 - nu) / np.pi
        dx = x_m - sph['x0'] * 1e3
        dy = y_m - sph['y0'] * 1e3
        d  = sph['d'] * 1e3
        R3 = (dx**2 + dy**2 + d**2) ** 1.5
        uz += C * d / R3
    return uz


def calibrate_mu():
    """
    Solve for the shear modulus μ that makes the Mogi model match the
    observed geodetic inflation during the Kidiwela period (Aug 2022 – Aug 2025).

    Because Uz ∝ 1/μ, the calibrated value is:
        μ_cal = MU × Uz(MU) / ΔUz_geodetic

    The stress tensor direction is unaffected by μ (μ cancels in σ = 2μ·ε).
    The stress magnitude |σ₁−σ₂| ∝ ΔP (independent of μ).
    """
    elev_start  = elevation_at(KIDIWELA_START)
    elev_end    = elevation_at(KIDIWELA_END)
    dUz_geodetic = elev_end - elev_start          # observed geodetic change (m)

    uz_model_ref = _uz_at_point(CALDERA_X, CALDERA_Y, mu=MU)   # at MU=30 GPa

    mu_cal = MU * uz_model_ref / dUz_geodetic     # solve for mu

    print(f'\n── Geodetic μ calibration ──────────────────────────────────────')
    print(f'  Kidiwela period:   {KIDIWELA_START.date()} → {KIDIWELA_END.date()}')
    print(f'  Elevation change:  {dUz_geodetic*100:.1f} cm  '
          f'({elev_start:.2f} → {elev_end:.2f} m)')
    print(f'  Model Uz at μ={MU/1e9:.0f} GPa:  {uz_model_ref*1000:.1f} mm')
    print(f'  ⟹ Calibrated μ:  {mu_cal/1e9:.3f} GPa')
    print(f'  (Stress directions are μ-independent; '
          f'only displacement amplitude is affected.)')
    print(f'───────────────────────────────────────────────────────────────\n')

    return mu_cal


def mogi_surface(x_km, y_km, sphere, scale=1.0, mu=MU, nu=NU):
    """
    Surface displacement (Ux, Uy, Uz) [m] from one Mogi sphere.

    x_km, y_km : 2-D meshgrid arrays (km)
    sphere     : dict with keys x0, y0, d (km), R (km), dP (Pa)
    scale      : amplitude scaling factor
    Returns    : Ux, Uy, Uz  (same shape as x_km)
    """
    dV = _mogi_dV(sphere['R'], sphere['dP'] * scale, mu, nu)
    C  = dV * (1 - nu) / np.pi                # m³ / m^? (see below)

    dx = (x_km - sphere['x0']) * 1e3          # m
    dy = (y_km - sphere['y0']) * 1e3          # m
    d  = sphere['d'] * 1e3                    # m (depth, positive down)

    R3 = (dx**2 + dy**2 + d**2) ** 1.5        # m³

    Ux = C * dx / R3
    Uy = C * dy / R3
    Uz = C * d  / R3   # positive = uplift

    return Ux, Uy, Uz


def total_displacement(x_km, y_km, scales=None, scale=1.0):
    """Sum displacement from both Kidiwela spheres.

    scales : list of per-sphere scale factors (overrides global scale).
    scale  : global factor applied to all spheres if scales is None.
    """
    if scales is None:
        scales = [scale] * len(SPHERES)
    Ux = np.zeros_like(x_km, dtype=float)
    Uy = np.zeros_like(x_km, dtype=float)
    Uz = np.zeros_like(x_km, dtype=float)
    for sph, sc in zip(SPHERES, scales):
        ux, uy, uz = mogi_surface(x_km, y_km, sph, scale=sc)
        Ux += ux; Uy += uy; Uz += uz
    return Ux, Uy, Uz


# ── Stress field (Baillard's pipeline) ───────────────────────────────────────

def _strain_stress_sigma1(Ux, Uy, dx_km, dy_km, mu=MU, nu=NU):
    """
    Compute the σ₁ field from horizontal displacements using central differences.

    Returns
    -------
    SIGMA1 : ndarray shape (ny, nx, 2)
        Magnitude-weighted compressive eigenvector at each grid node.
        Direction → predicted fast direction φ.
        Magnitude = |σ₁ − σ₂| → scales with predicted δt.
    """
    dx = dx_km * 1e3   # m
    dy = dy_km * 1e3   # m
    lam = LAMBDA

    ny, nx = Ux.shape
    SIGMA1 = np.zeros((ny, nx, 2))

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            duxdx = (Ux[i, j+1] - Ux[i, j-1]) / (2 * dx)
            duxdy = (Ux[i+1, j] - Ux[i-1, j]) / (2 * dy)
            duydx = (Uy[i, j+1] - Uy[i, j-1]) / (2 * dx)
            duydy = (Uy[i+1, j] - Uy[i-1, j]) / (2 * dy)

            e11, e12 = duxdx, 0.5 * (duxdy + duydx)
            e22 = duydy

            tr_e = e11 + e22
            s11 = lam * tr_e + 2 * mu * e11
            s12 = 2 * mu * e12
            s22 = lam * tr_e + 2 * mu * e22

            S = np.array([[s11, s12], [s12, s22]])
            try:
                eigvals, eigvecs = np.linalg.eigh(S)
                # eigh returns ascending order; index 0 = smallest (compression)
                diff  = abs(eigvals[1] - eigvals[0])
                SIGMA1[i, j] = diff * eigvecs[:, 0]
            except Exception:
                pass

    return SIGMA1


def stress_field(x_km, y_km, scale=1.0):
    """
    Full stress pipeline for a given source amplitude.

    Returns
    -------
    SIGMA1 : ndarray (ny, nx, 2)  — magnitude-weighted compressive axis
    Uz     : ndarray (ny, nx)     — vertical displacement [m]
    """
    Ux, Uy, Uz = total_displacement(x_km, y_km, scale=scale)
    dx_km = x_km[0, 1] - x_km[0, 0]
    dy_km = y_km[1, 0] - y_km[0, 0]
    SIGMA1 = _strain_stress_sigma1(Ux, Uy, dx_km, dy_km)
    return SIGMA1, Uz


def sigma1_to_phi(vec):
    """Convert SIGMA1 2-vector to geographic azimuth (° CW from N, 0–180)."""
    phi_rad = np.arctan2(vec[0], vec[1])   # atan2(east, north) = CW from N
    phi_deg = np.degrees(phi_rad) % 180.0
    return phi_deg


def sigma1_magnitude(vec):
    return float(np.linalg.norm(vec))


def sigma1_for_scale(SIGMA1_ref, scale):
    """
    Return the SIGMA1 field appropriate for a given source amplitude scale.

    For scale > 0 (inflation): SIGMA1 direction = tangential (compression axis
    of the inflating source), stored directly in SIGMA1_ref.

    For scale < 0 (deflation): the sign of the stress tensor reverses, so
    the compression and extension axes SWAP.  The new compression axis is
    perpendicular (90°) to the inflation compression axis.  We implement this
    by rotating each SIGMA1_ref vector by 90° and scaling by |scale|.

    Rotating 2D vector (vx, vy) by +90°: (-vy, vx).
    """
    if scale >= 0:
        return SIGMA1_ref * scale
    else:
        # Rotate each vector 90° → new compression axis for deflation
        rot90 = np.stack([-SIGMA1_ref[..., 1],
                           SIGMA1_ref[..., 0]], axis=-1)
        return rot90 * abs(scale)


# ── Ray paths with AMC routing ────────────────────────────────────────────────

def _sphere_intersects(p1, p2, centre, radius):
    """True if the line segment p1→p2 (3D, km) passes through the sphere."""
    d = p2 - p1
    f = p1 - centre
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius ** 2
    disc = b**2 - 4*a*c
    if disc < 0:
        return False
    disc = np.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def _tangent_path_length(p1, p2, centre, radius):
    """
    Minimum path length (km) from p1 to p2 routing around a sphere.

    Projects to 2D (horizontal plane) for the detour calculation,
    then corrects for the 3D segment lengths.
    """
    # Straight-line for reference
    straight = float(np.linalg.norm(p2 - p1))

    # Work in the plane containing p1, p2, and the sphere centre (approx 2D)
    # Project everything to 2D (x, y only) for the tangent calculation
    c2 = centre[:2]
    a2 = p1[:2]
    b2 = p2[:2]

    # Distance from each endpoint to sphere centre (2D)
    da = np.linalg.norm(a2 - c2)
    db = np.linalg.norm(b2 - c2)

    if da <= radius or db <= radius:
        # Source or receiver inside AMC — return straight line (shouldn't happen)
        return straight

    # Tangent length from a point to a circle: sqrt(d² - r²)
    tang_a = np.sqrt(max(da**2 - radius**2, 0))
    tang_b = np.sqrt(max(db**2 - radius**2, 0))
    # Detour ≈ tangent from p1 to tangent point + arc + tangent point to p2
    # Simplified: tangent from p1 + tangent from p2 (skips arc, slight underestimate)
    arc_angle = abs(np.arctan2(np.cross(a2 - c2, b2 - c2),
                               np.dot(a2 - c2, b2 - c2)))
    arc_len = radius * max(arc_angle - np.arcsin(radius / da)
                           - np.arcsin(radius / db), 0)
    path_2d = tang_a + arc_len + tang_b

    # Scale back to 3D (approximate: scale by ratio of 3D to 2D straight)
    straight_2d = np.linalg.norm(b2 - a2)
    scale_3d = straight / straight_2d if straight_2d > 0 else 1.0
    return path_2d * scale_3d


def ray_path_length_km(eq_xyz, sta_xyz):
    """
    Effective ray path length (km) in solid rock from earthquake to station.

    Routes around the AMC sphere (Vs=0 → complete barrier).
    eq_xyz, sta_xyz : (x, y, z) in km  (z positive downward)
    """
    p1 = np.array(eq_xyz, dtype=float)
    p2 = np.array(sta_xyz, dtype=float)
    c  = np.array([AMC['x0'], AMC['y0'], AMC['z0']], dtype=float)
    r  = AMC['r']

    if _sphere_intersects(p1, p2, c, r):
        return _tangent_path_length(p1, p2, c, r)
    else:
        return float(np.linalg.norm(p2 - p1))


# ── Load data ─────────────────────────────────────────────────────────────────

def _load_station():
    stations = pd.read_csv(
        STATION_FILE, sep=r'\s+',
        names=['lon', 'lat', 'elev_km', 'station'],
        engine='python',
    ).set_index('station')
    stations['x'], stations['y'] = ll2xy(stations['lat'].values, stations['lon'].values)
    stations['z'] = stations['elev_km'].abs()   # depth below sea surface, positive
    return stations


def _load_catalog():
    """Load unique earthquake locations from all station splitting CSV files."""
    files = {
        'AXEC3': ('splitting_results_mldd_2015_2021_axec3_all_batches.csv',
                  'splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
        'AXEC1': ('splitting_results_mldd_2015_2021_axec1_all_batches.csv',
                  'splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
        'AXEC2': ('axial-mldd-2015-2021-axec2.csv',
                  'splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
        'AXAS1': ('splitting_results_mldd_2015_2021_axas1.csv',
                  'splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
        'AXAS2': ('splitting_results_mldd_2015_2021_axas2.csv',
                  'splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
        'AXCC1': ('splitting_results_mldd_2015_2021_axcc1_all_batches.csv',
                  'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    }
    dfs = []
    for sta, (f1, f2) in files.items():
        for f in (f1, f2):
            df = pd.read_csv(BASE + f)
            df = df.loc[:, :'dt_error'].dropna()
            df = df[df['dt'] > 0]
            df['station'] = sta
            df['phi_az'] = df['phi'] + 90.0
            df['t']      = pd.to_datetime(df['event_datetime'], utc=True)
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['x'], all_df['y'] = ll2xy(all_df['event_lat'].values,
                                     all_df['event_lon'].values)
    all_df['z'] = all_df['event_depth'].values   # km, positive downward
    return all_df


# ── Time periods (matching sws_mesh_plot_mldd.py) ─────────────────────────────

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')


def build_time_periods(all_df):
    """7 periods matching sws_mesh_plot_mldd: pre, syn, 5 equal-count post."""
    post = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post)
    bounds = [ERUPTION_END]
    for i in range(1, 5):
        idx = min(int(round(i * n / 5)), n - 1)
        bounds.append(post['t'].iloc[idx])
    bounds.append(None)

    def _fmt(ts):
        return ts.strftime('%b %Y') if ts is not None else 'present'

    periods = [
        ('Pre-eruption',  None,            ERUPTION_START),
        ('Syn-eruption',  ERUPTION_START,  ERUPTION_END),
    ]
    for i in range(5):
        t0, t1 = bounds[i], bounds[i + 1]
        periods.append((f'{_fmt(t0)} – {_fmt(t1)}', t0, t1))
    return periods


def period_midpoint(t_start, t_end):
    """Midpoint timestamp of a time period."""
    if t_start is None:
        t_start = pd.Timestamp('2015-01-01', tz='UTC')
    if t_end is None:
        t_end = pd.Timestamp('2026-05-01', tz='UTC')
    return t_start + (t_end - t_start) / 2


def subset_df(df, t_start, t_end):
    m = (df['t'] >= t_start) if t_start is not None else pd.Series(True, index=df.index)
    if t_end is not None:
        m &= df['t'] < t_end
    return df[m]


# ── Per-station ray-length statistics ─────────────────────────────────────────

def compute_mean_ray_lengths(all_df, stations):
    """
    For each station, compute the mean ray path length (km) over all its
    events, routing around the AMC sphere.
    """
    mean_L = {}
    for sta, srow in stations.iterrows():
        sta_df = all_df[all_df['station'] == sta]
        if sta_df.empty:
            mean_L[sta] = 2.0   # fallback
            continue
        lengths = []
        for _, row in sta_df.iterrows():
            eq_xyz  = (row['x'], row['y'], row['z'])
            sta_xyz = (srow['x'], srow['y'], srow['z'])
            lengths.append(ray_path_length_km(eq_xyz, sta_xyz))
        mean_L[sta] = float(np.mean(lengths))
        print(f'  {sta}: mean ray length = {mean_L[sta]:.2f} km  (N={len(lengths):,})')
    return mean_L


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print('Loading data...')
    stations = _load_station()
    all_df   = _load_catalog()
    STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
    stations = stations.loc[[s for s in STATIONS if s in stations.index]]
    print(f'  {len(all_df):,} splitting measurements across {len(STATIONS)} stations')

    # Calibrate shear modulus from geodetic data
    mu_cal = calibrate_mu()

    # Build computation grid
    x_nodes = np.arange(4.0, 12.01, 0.1)
    y_nodes = np.arange(0.0, 12.01, 0.1)
    X, Y    = np.meshgrid(x_nodes, y_nodes)

    # Compute reference stress field (Kidiwela, scale=1)
    # Note: stress directions are μ-independent; we pass mu_cal only for Uz map
    print('Computing reference (Kidiwela) stress field...')
    SIGMA1_ref, Uz_ref_nominal = stress_field(X, Y, scale=1.0)

    # Uz with calibrated μ (for display only — stress is unaffected)
    Ux_cal, Uy_cal, Uz_ref = total_displacement(X, Y, scale=1.0)
    Uz_ref = Uz_ref * (MU / mu_cal)    # rescale to calibrated μ

    ix = np.argmin(np.abs(x_nodes - CALDERA_X))
    iy = np.argmin(np.abs(y_nodes - CALDERA_Y))
    uz_centre_ref = Uz_ref[iy, ix]
    print(f'  Reference Uz at caldera centre (μ_cal): {uz_centre_ref*100:.1f} cm')

    # Build time periods
    time_periods = build_time_periods(all_df)
    print(f'\n7 time periods:')
    for label, t0, t1 in time_periods:
        mid = period_midpoint(t0, t1)
        sc  = source_scale(mid)
        elev = elevation_at(mid)
        print(f'  {label:<30}  elev={elev:.2f} m  scale={sc:.3f}')

    # Compute stress field for each time period
    print('\nComputing per-period stress fields...')
    period_stress = {}
    for label, t0, t1 in time_periods:
        mid  = period_midpoint(t0, t1)
        sc   = source_scale(mid)
        sig1 = sigma1_for_scale(SIGMA1_ref, sc)
        period_stress[label] = (sig1, sc)

    # Compute mean ray lengths (expensive — sample 500 events per station)
    print('\nComputing mean ray path lengths...')
    sample_df = all_df.groupby('station', group_keys=False).apply(
        lambda g: g.sample(min(500, len(g)), random_state=42)
    )
    mean_L = compute_mean_ray_lengths(sample_df, stations)

    # Build observed SWS summaries per station × period
    print('\nBuilding observed SWS summaries...')
    obs = {}   # obs[(sta, label)] = {'phi_med', 'dt_med', 'N'}
    for label, t0, t1 in time_periods:
        for sta in STATIONS:
            sub = subset_df(all_df[all_df['station'] == sta], t0, t1)
            if len(sub) < 15:
                obs[(sta, label)] = None
                continue
            obs[(sta, label)] = {
                'phi_med': float(sub['phi_az'].median()),
                'dt_med':  float(sub['dt'].median()),
                'N':       len(sub),
            }

    # Extract predicted σ₁ at each station location
    print('\nComparing predictions with observations...')
    results = []
    for label, t0, t1 in time_periods:
        sig1, sc = period_stress[label]
        for sta in STATIONS:
            if sta not in stations.index:
                continue
            srow = stations.loc[sta]
            # Nearest grid node to station
            ix_s = np.argmin(np.abs(x_nodes - srow['x']))
            iy_s = np.argmin(np.abs(y_nodes - srow['y']))
            vec  = sig1[iy_s, ix_s]
            phi_pred = sigma1_to_phi(vec)
            mag_pred = sigma1_magnitude(vec)   # |σ₁−σ₂|

            L = mean_L.get(sta, 2.0)
            dt_proxy = mag_pred * L             # |σ₁−σ₂| × L (unnormalised)

            ob = obs.get((sta, label))
            results.append({
                'period':    label,
                'station':   sta,
                'scale':     sc,
                'phi_pred':  phi_pred,
                'mag_pred':  mag_pred,
                'dt_proxy':  dt_proxy,
                'ray_L_km':  L,
                'phi_obs':   ob['phi_med'] if ob else np.nan,
                'dt_obs':    ob['dt_med']  if ob else np.nan,
                'N':         ob['N']       if ob else 0,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, 'mogi_stress_results.csv'), index=False)
    print(f'  Saved mogi_stress_results.csv')

    _make_plots(results_df, X, Y, x_nodes, y_nodes, SIGMA1_ref, Uz_ref,
                time_periods, period_stress, stations)


def _make_plots(results_df, X, Y, x_nodes, y_nodes, SIGMA1_ref, Uz_ref,
                time_periods, period_stress, stations):

    STATIONS = list(stations.index)

    # ── Plot 1: Reference stress field ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    mag = np.linalg.norm(SIGMA1_ref, axis=2)
    im  = ax.pcolormesh(X, Y, mag, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='|σ₁−σ₂| (normalised)')
    # σ₁ direction quivers (subsampled)
    step = 8
    Xs, Ys = X[::step, ::step], Y[::step, ::step]
    S1x = SIGMA1_ref[::step, ::step, 0]
    S1y = SIGMA1_ref[::step, ::step, 1]
    ax.quiver(Xs, Ys, S1x, S1y, color='w', scale=None, width=0.003, alpha=0.7)
    ax.quiver(Xs, Ys, -S1x, -S1y, color='w', scale=None, width=0.003, alpha=0.7)
    # Source locations
    for sph in SPHERES:
        ax.plot(sph['x0'], sph['y0'], 'r*', ms=12, label=sph['label'])
    for sta, row in stations.iterrows():
        ax.plot(row['x'], row['y'], '^', mfc='#FFD700', mec='k', ms=8)
        ax.text(row['x'] + 0.1, row['y'] + 0.1, sta[-3:], fontsize=7)
    ax.set_xlim(4, 12); ax.set_ylim(0, 12)
    ax.set_aspect('equal'); ax.set_title('Reference σ₁ field (Kidiwela, scale=1)')
    ax.set_xlabel('E (km)'); ax.set_ylabel('N (km)')
    ax.legend(fontsize=8)

    ax = axes[1]
    im = ax.pcolormesh(X, Y, Uz_ref * 100, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='Uz (cm)')
    ax.set_xlim(4, 12); ax.set_ylim(0, 12); ax.set_aspect('equal')
    ax.set_title('Reference vertical displacement (Kidiwela, scale=1)')
    ax.set_xlabel('E (km)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mogi_reference_field.pdf'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved mogi_reference_field.pdf')

    # ── Plot 2: Source scale vs. time ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    t_range = pd.date_range('2015-01-01', '2026-06-01', freq='MS')
    ax.plot(t_range, [source_scale(t) for t in t_range], 'k-', lw=1.5)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.axhline(1, color='steelblue', ls='--', lw=0.8, label='Kidiwela ref.')
    for label, t0, t1 in time_periods:
        mid = pd.Timestamp(
            (pd.Timestamp('2015-01-01', tz='UTC') if t0 is None else t0).value // 2
            + (pd.Timestamp('2026-05-01', tz='UTC') if t1 is None else t1).value // 2,
            unit='ns', tz='UTC'
        )
        sc = source_scale(mid)
        ax.axvspan(
            pd.Timestamp('2015-01-01' if t0 is None else t0).tz_localize(None),
            pd.Timestamp('2026-05-01' if t1 is None else t1).tz_localize(None),
            alpha=0.08,
        )
    ax.set_ylabel('Source scale (rel. Kidiwela)')
    ax.set_title('Mogi source amplitude through time (geodetic scaling)')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mogi_source_scale.pdf'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved mogi_source_scale.pdf')

    # ── Plot 3: φ_pred vs φ_obs per station (7 periods each) ─────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cmap = plt.cm.viridis
    period_labels = [label for label, _, _ in time_periods]
    colors = [cmap(i / 6) for i in range(7)]

    for ax_idx, sta in enumerate(STATIONS):
        ax = axes[ax_idx // 3, ax_idx % 3]
        sub = results_df[results_df['station'] == sta].dropna(subset=['phi_obs', 'phi_pred'])
        for k, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(row['phi_obs'], row['phi_pred'],
                       color=colors[k % 7], s=80, zorder=5,
                       label=row['period'] if ax_idx == 0 else None)
            ax.annotate(str(k + 1), (row['phi_obs'], row['phi_pred']),
                        fontsize=7, ha='center', va='center', color='w', fontweight='bold')
        ax.plot([0, 180], [0, 180], 'r--', lw=1)
        ax.plot([0, 180], [-180, 0], 'r--', lw=1, alpha=0.4)
        ax.set_xlim(0, 180); ax.set_ylim(0, 180)
        ax.set_aspect('equal'); ax.set_title(sta)
        ax.set_xlabel('φ observed (°)'); ax.set_ylabel('φ predicted (°)')
        ax.grid(True, alpha=0.3)
    if len(sub) > 0:
        handles = [plt.scatter([], [], color=colors[k], label=f'P{k+1}') for k in range(7)]
        fig.legend(handles=handles, title='Period', loc='lower right', fontsize=8, ncol=2)
    fig.suptitle('Predicted vs. observed fast direction φ', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mogi_phi_comparison.pdf'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved mogi_phi_comparison.pdf')

    # ── Plot 4: δt_proxy vs δt_obs (all stations, color = period) ────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sub = results_df.dropna(subset=['dt_obs', 'dt_proxy'])
    for k, label in enumerate(period_labels):
        pdata = sub[sub['period'] == label]
        ax.scatter(pdata['dt_obs'], pdata['dt_proxy'],
                   color=colors[k], s=60, label=f'P{k+1}: {label[:20]}', alpha=0.8)
    # Linear fit
    if len(sub) > 2:
        a, b = np.polyfit(sub['dt_obs'], sub['dt_proxy'], 1)
        x_fit = np.linspace(sub['dt_obs'].min(), sub['dt_obs'].max(), 50)
        ax.plot(x_fit, a * x_fit + b, 'k--', lw=1.5,
                label=f'fit: slope={a:.1f}')
    ax.set_xlabel('δt observed (s)'); ax.set_ylabel('|σ₁−σ₂| × L (unnorm.)')
    ax.set_title('Predicted stress proxy vs. observed delay time')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mogi_dt_comparison.pdf'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved mogi_dt_comparison.pdf')

    print('\nDone.')




def sigma1_from_scales(x_km, y_km, sphere_scales, X, Y):
    """Compute SIGMA1 field with independent scale factor per sphere."""
    Ux, Uy, _ = total_displacement(x_km, y_km, scales=sphere_scales)
    dx_km = x_km[0, 1] - x_km[0, 0]
    dy_km = y_km[1, 0] - y_km[0, 0]
    return _strain_stress_sigma1(Ux, Uy, dx_km, dy_km)

def eruption_stress_comparison(results_df, X, Y, x_nodes, y_nodes,
                                SIGMA1_ref, period_stress, stations):
    """
    Reproduce Baillard's core figure: compare observed φ and δt changes
    across the 2015 eruption with the Mogi stress-change prediction.

    Three panels per row (one row per station):
      Left  — stress field map with predicted σ₁ sticks and observed φ
      Centre — φ_pred vs φ_obs (pre and syn as two points per station)
      Right  — |σ₁−σ₂| vs δt_obs (linear calibration)

    The syn-eruption stress = (scale_syn − scale_pre) × σ_Kidiwela
    = large negative value → σ₁ rotates ~90° (deflation reverses compression).
    """

    PRE_LABEL = 'Pre-eruption'
    SYN_LABEL = 'Syn-eruption'
    STATIONS  = list(stations.index)

    pre_sig1, scale_pre = period_stress[PRE_LABEL]
    syn_sig1, scale_syn = period_stress[SYN_LABEL]

    # Stress CHANGE pre→syn drives the eruption-period SWS
    # Eruption: only S2 (shallow source, index 1) fully deflates;
    # S1 (deep) remains at pre-eruption level.
    # scale_pre applied to both spheres pre-eruption.
    scales_pre = [scale_pre, scale_pre]
    scales_syn = [scale_pre, scale_syn]   # S1 stays, S2 fully deflates
    delta_scale = scale_syn - scale_pre   # for labelling (S2 change)
    delta_sig1  = sigma1_from_scales(X, Y, scales_syn, X, Y) - sigma1_from_scales(X, Y, scales_pre, X, Y)
    # Recompute absolute syn-eruption SIGMA1 (S2 deflated, S1 at pre level)
    syn_sig1_abs = sigma1_from_scales(X, Y, scales_syn, X, Y)

    # ── Figure: pre and syn stress maps with station sticks ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    def _draw_map(ax, sig1, scale, title, obs_period):
        mag = np.linalg.norm(sig1, axis=2)
        im  = ax.pcolormesh(X, Y, mag, cmap='viridis', shading='auto',
                            vmin=0)
        plt.colorbar(im, ax=ax, label='|σ₁−σ₂| (Pa, normalised)')

        # σ₁ direction sticks (white, subsampled)
        step = 8
        Xs, Ys = X[::step, ::step], Y[::step, ::step]
        S1x = sig1[::step, ::step, 0]
        S1y = sig1[::step, ::step, 1]
        nrm = np.linalg.norm(np.stack([S1x, S1y], axis=-1), axis=-1, keepdims=True) + 1e-30
        S1x_n, S1y_n = S1x / nrm[..., 0], S1y / nrm[..., 0]
        ax.quiver(Xs, Ys,  S1x_n,  S1y_n, color='w', scale=30,
                  width=0.003, alpha=0.6, headwidth=0, headlength=0)
        ax.quiver(Xs, Ys, -S1x_n, -S1y_n, color='w', scale=30,
                  width=0.003, alpha=0.6, headwidth=0, headlength=0)

        # Observed φ sticks at each station
        sub = results_df[results_df['period'] == obs_period]
        for _, row in sub.iterrows():
            if np.isnan(row.get('phi_obs', np.nan)):
                continue
            srow = stations.loc[row['station']]
            phi_rad = np.radians(row['phi_obs'])
            dx_s =  np.sin(phi_rad) * 0.4
            dy_s =  np.cos(phi_rad) * 0.4
            ax.annotate('', xy=(srow['x'] + dx_s, srow['y'] + dy_s),
                        xytext=(srow['x'] - dx_s, srow['y'] - dy_s),
                        arrowprops=dict(arrowstyle='-', color='red', lw=2))
            ax.plot(srow['x'], srow['y'], '^', mfc='#FFD700', mec='k', ms=9, zorder=10)
            ax.text(srow['x'] + 0.12, srow['y'] + 0.12, row['station'][-3:],
                    fontsize=7, color='w', fontweight='bold')

        # Source locations
        for sph in SPHERES:
            ax.plot(sph['x0'], sph['y0'], 'r*', ms=14)

        ax.set_xlim(4, 12); ax.set_ylim(0, 12); ax.set_aspect('equal')
        ax.set_title(f'{title}  (scale={scale:+.3f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('E (km)'); ax.set_ylabel('N (km)')

    _draw_map(axes[0], pre_sig1,  scale_pre, 'Pre-eruption stress', PRE_LABEL)
    _draw_map(axes[1], syn_sig1_abs, delta_scale,
              'Stress change (pre→syn deflation)', SYN_LABEL)

    fig.suptitle('Mogi stress field: pre-eruption inflation vs. eruption deflation\n'
                 'White sticks = predicted σ₁ (compression axis)  |  '
                 'Red sticks = observed φ',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mogi_eruption_maps.pdf'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved mogi_eruption_maps.pdf')

    # ── Baillard-style scatter: φ_pred vs φ_obs, |Δσ| vs δt ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_phi, ax_dt = axes

    # 1:1 reference lines for φ
    for offset in [0, -180, 180]:
        ax_phi.plot([0, 180], [0+offset, 180+offset], '--r', lw=0.8, alpha=0.5)
    ax_phi.set_xlim(0, 180); ax_phi.set_ylim(-90, 270)
    ax_phi.set_xlabel('φ observed (° from N)')
    ax_phi.set_ylabel('φ predicted by σ₁ (° from N)')
    ax_phi.set_title('Fast direction: Baillard comparison')
    ax_phi.grid(True, alpha=0.3)

    phi_obs_all, phi_pred_all = [], []
    dt_obs_all,  mag_pred_all = [], []

    for period_label, sig1_field, marker, color in [
        (PRE_LABEL,  pre_sig1,   'o', '#2196F3'),
        (SYN_LABEL,  syn_sig1_abs, 's', '#F44336'),
    ]:
        sub = results_df[results_df['period'] == period_label].dropna(
            subset=['phi_obs', 'dt_obs'])
        for _, row in sub.iterrows():
            if row['station'] not in stations.index:
                continue
            srow = stations.loc[row['station']]
            ix_s = np.argmin(np.abs(x_nodes - srow['x']))
            iy_s = np.argmin(np.abs(y_nodes - srow['y']))
            vec  = sig1_field[iy_s, ix_s]

            phi_p = sigma1_to_phi(vec)
            mag_p = sigma1_magnitude(vec)
            L     = row['ray_L_km']

            ax_phi.scatter(row['phi_obs'], phi_p, marker=marker,
                           color=color, s=100, zorder=5,
                           label=f'{period_label} – {row["station"][-3:]}')
            ax_phi.annotate(row['station'][-3:],
                            (row['phi_obs'], phi_p),
                            fontsize=7, ha='left', va='bottom',
                            xytext=(3, 3), textcoords='offset points')

            phi_obs_all.append(row['phi_obs'])
            phi_pred_all.append(phi_p)
            dt_obs_all.append(row['dt_obs'])
            mag_pred_all.append(mag_p * L)

    # Legend with just period
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', mfc='#2196F3', ms=9,
               label='Pre-eruption'),
        Line2D([0], [0], marker='s', color='w', mfc='#F44336', ms=9,
               label='Syn-eruption (deflation Δσ)'),
    ]
    ax_phi.legend(handles=legend_els, fontsize=8)

    # RMS angular misfit
    phi_obs_arr  = np.array(phi_obs_all)
    phi_pred_arr = np.array(phi_pred_all)
    # Handle 180° periodicity
    diff = (phi_pred_arr - phi_obs_arr + 90) % 180 - 90
    rms_phi = np.sqrt(np.mean(diff**2))
    ax_phi.text(0.05, 0.95, f'RMS = {rms_phi:.1f}°',
                transform=ax_phi.transAxes, fontsize=9,
                va='top', bbox=dict(boxstyle='round', fc='w', alpha=0.7))

    # |Δσ|·L vs δt (Baillard's right panel)
    dt_arr  = np.array(dt_obs_all)
    mag_arr = np.array(mag_pred_all)
    if len(dt_arr) > 2:
        a, b = np.polyfit(dt_arr, mag_arr, 1)
        x_fit = np.linspace(dt_arr.min(), dt_arr.max(), 50)
        ax_dt.plot(x_fit, a * x_fit + b, '--r', lw=1.5,
                   label=f'fit a={a:.2e}')
        rms_dt = np.sqrt(np.mean((mag_arr - (a * dt_arr + b))**2))
        ax_dt.text(0.05, 0.95, f'RMS = {rms_dt:.2e}\na = {a:.2e}',
                   transform=ax_dt.transAxes, fontsize=9,
                   va='top', bbox=dict(boxstyle='round', fc='w', alpha=0.7))

    ax_dt.scatter(dt_arr, mag_arr, s=80, alpha=0.8, zorder=5)
    ax_dt.set_xlabel('δt observed (s)')
    ax_dt.set_ylabel('|σ₁−σ₂| × L (proxy)')
    ax_dt.set_title('Delay time vs. stress proxy')
    ax_dt.legend(fontsize=8); ax_dt.grid(True, alpha=0.3)

    fig.suptitle('Baillard-style comparison: 2015 eruption stress change',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mogi_baillard_comparison.pdf'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved mogi_baillard_comparison.pdf')

    # Print table
    print(f'\n  RMS φ misfit (pre+syn combined): {rms_phi:.1f}°')
    print('\n  Per-station φ summary:')
    print(f'  {"Station":<8} {"φ_obs_pre":>10} {"φ_pred_pre":>11} '
          f'{"φ_obs_syn":>10} {"φ_pred_syn":>11} {"Δφ_obs":>8} {"Δφ_pred":>9}')
    for sta in STATIONS:
        pre_row = results_df[(results_df['station']==sta) &
                             (results_df['period']==PRE_LABEL)]
        syn_row = results_df[(results_df['station']==sta) &
                             (results_df['period']==SYN_LABEL)]
        if pre_row.empty or syn_row.empty:
            continue
        phi_op  = pre_row['phi_obs'].values[0]
        phi_os  = syn_row['phi_obs'].values[0]
        srow    = stations.loc[sta]
        ix_s = np.argmin(np.abs(x_nodes - srow['x']))
        iy_s = np.argmin(np.abs(y_nodes - srow['y']))
        phi_pp  = sigma1_to_phi(pre_sig1[iy_s, ix_s])
        phi_ps  = sigma1_to_phi(syn_sig1_abs[iy_s, ix_s])
        d_obs   = ((phi_os - phi_op + 90) % 180 - 90)
        d_pred  = ((phi_ps - phi_pp + 90) % 180 - 90)
        print(f'  {sta:<8} {phi_op:>10.1f} {phi_pp:>11.1f} '
              f'{phi_os:>10.1f} {phi_ps:>11.1f} {d_obs:>8.1f} {d_pred:>9.1f}')


def run():
    print('Loading data...')
    stations = _load_station()
    all_df   = _load_catalog()
    STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
    stations = stations.loc[[s for s in STATIONS if s in stations.index]]
    print(f'  {len(all_df):,} splitting measurements across {len(STATIONS)} stations')

    # Calibrate shear modulus from geodetic data
    mu_cal = calibrate_mu()

    # Build computation grid
    x_nodes = np.arange(4.0, 12.01, 0.1)
    y_nodes = np.arange(0.0, 12.01, 0.1)
    X, Y    = np.meshgrid(x_nodes, y_nodes)

    # Compute reference stress field (Kidiwela, scale=1)
    print('Computing reference (Kidiwela) stress field...')
    SIGMA1_ref, Uz_ref_nominal = stress_field(X, Y, scale=1.0)

    # Uz with calibrated μ (for display only — stress is unaffected)
    Ux_cal, Uy_cal, Uz_ref = total_displacement(X, Y, scale=1.0)
    Uz_ref = Uz_ref * (MU / mu_cal)    # rescale to calibrated μ

    ix = np.argmin(np.abs(x_nodes - CALDERA_X))
    iy = np.argmin(np.abs(y_nodes - CALDERA_Y))
    uz_centre_ref = Uz_ref[iy, ix]
    print(f'  Reference Uz at caldera centre (μ_cal): {uz_centre_ref*100:.1f} cm')

    # Build time periods
    time_periods = build_time_periods(all_df)
    print(f'\n7 time periods:')
    for label, t0, t1 in time_periods:
        mid = period_midpoint(t0, t1)
        sc  = source_scale(mid)
        elev = elevation_at(mid)
        print(f'  {label:<30}  elev={elev:.2f} m  scale={sc:.3f}')

    # Compute stress field for each time period
    print('\nComputing per-period stress fields...')
    period_stress = {}
    for label, t0, t1 in time_periods:
        mid  = period_midpoint(t0, t1)
        sc   = source_scale(mid)
        sig1 = sigma1_for_scale(SIGMA1_ref, sc)
        period_stress[label] = (sig1, sc)

    # Compute mean ray lengths (expensive — sample 500 events per station)
    print('\nComputing mean ray path lengths...')
    sample_df = (all_df.groupby('station', group_keys=False)
                 .apply(lambda g: g.sample(min(500, len(g)), random_state=42))
                 .reset_index(drop=True))
    mean_L = compute_mean_ray_lengths(sample_df, stations)

    # Build observed SWS summaries per station × period
    print('\nBuilding observed SWS summaries...')
    obs = {}
    for label, t0, t1 in time_periods:
        for sta in STATIONS:
            sub = subset_df(all_df[all_df['station'] == sta], t0, t1)
            if len(sub) < 15:
                obs[(sta, label)] = None
                continue
            obs[(sta, label)] = {
                'phi_med': float(sub['phi_az'].median()),
                'dt_med':  float(sub['dt'].median()),
                'N':       len(sub),
            }

    # Extract predicted σ₁ at each station location
    print('\nBuilding results table...')
    results = []
    for label, t0, t1 in time_periods:
        sig1, sc = period_stress[label]
        for sta in STATIONS:
            if sta not in stations.index:
                continue
            srow = stations.loc[sta]
            ix_s = np.argmin(np.abs(x_nodes - srow['x']))
            iy_s = np.argmin(np.abs(y_nodes - srow['y']))
            vec  = sig1[iy_s, ix_s]
            phi_pred = sigma1_to_phi(vec)
            mag_pred = sigma1_magnitude(vec)

            L = mean_L.get(sta, 2.0)
            dt_proxy = mag_pred * L

            ob = obs.get((sta, label))
            results.append({
                'period':    label,
                'station':   sta,
                'scale':     sc,
                'phi_pred':  phi_pred,
                'mag_pred':  mag_pred,
                'dt_proxy':  dt_proxy,
                'ray_L_km':  L,
                'phi_obs':   ob['phi_med'] if ob else np.nan,
                'dt_obs':    ob['dt_med']  if ob else np.nan,
                'N':         ob['N']       if ob else 0,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, 'mogi_stress_results.csv'), index=False)
    print(f'  Saved mogi_stress_results.csv')

    # Standard output plots
    _make_plots(results_df, X, Y, x_nodes, y_nodes, SIGMA1_ref, Uz_ref,
                time_periods, period_stress, stations)

    # Baillard-style eruption comparison
    print('\nGenerating eruption stress comparison (Baillard-style)...')
    eruption_stress_comparison(results_df, X, Y, x_nodes, y_nodes,
                               SIGMA1_ref, period_stress, stations)


if __name__ == '__main__':
    run()
