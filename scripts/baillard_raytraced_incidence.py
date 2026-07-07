"""
S-wave ray-traced incidence angle through the Baillard 3D S-velocity model.

Ports the pseudo-bending ray tracer (Um & Thurber 1987) from sws_raytraced_dt.py's bent_ray,
decoupled from that script's top-level side effects (it loads CSVs from a stale hardcoded path
and generates PDF plots on import, so it isn't safely importable as a library). This module
adds the one piece bent_ray itself doesn't compute: the ray's incidence angle at the receiver,
i.e. the angle from vertical of the final ray segment arriving at the station.

Coordinate convention (matching sws_raytraced_dt.py exactly): local (x, y) km via ll2xy
(origin INI_LON/INI_LAT), z positive-down in km with the station at z=0 and the event at
z=eq_depth (i.e. a flat-datum approximation - all stations treated as sitting at a common
reference level, not their true individual elevations).
"""

import numpy as np

from baillard_velocity import vs_at

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

N_RAY = 25
N_ITER = 5
H_GRAD = 0.05
BEND_ALPHA = 0.25
Z_MAX = 4.0


def ll2xy(lat, lon):
    """Local (x, y) km relative to (INI_LON, INI_LAT), matching sws_raytraced_dt.py."""
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


def bent_ray(eq_x, eq_y, eq_z, sta_x, sta_y, n_ray=N_RAY, n_iter=N_ITER,
             h_grad=H_GRAD, bend_alpha=BEND_ALPHA, z_max=Z_MAX):
    """
    Pseudo-bending ray from (eq_x, eq_y, eq_z) to (sta_x, sta_y, 0) through the Baillard 3D
    Vs model. Returns (rx, ry, rz) arrays of n_ray points along the bent ray [km].

    Verbatim port of sws_raytraced_dt.py's bent_ray (same algorithm, parameters exposed as
    arguments instead of module-level constants).
    """
    t = np.linspace(0., 1., n_ray)
    rx = eq_x + t * (sta_x - eq_x)
    ry = eq_y + t * (sta_y - eq_y)
    rz = eq_z * (1. - t)  # linear depth: eq_z -> 0 at station

    for _ in range(n_iter):
        vs = vs_at(rx, ry, rz)

        dsdx = (1. / np.maximum(vs_at(rx + h_grad, ry, rz), 0.1)
                - 1. / np.maximum(vs_at(rx - h_grad, ry, rz), 0.1)) / (2. * h_grad)
        dsdy = (1. / np.maximum(vs_at(rx, ry + h_grad, rz), 0.1)
                - 1. / np.maximum(vs_at(rx, ry - h_grad, rz), 0.1)) / (2. * h_grad)
        dsdz = (1. / np.maximum(vs_at(rx, ry, rz + h_grad), 0.1)
                - 1. / np.maximum(vs_at(rx, ry, rz - h_grad), 0.1)) / (2. * h_grad)

        drx = np.gradient(rx)
        dry = np.gradient(ry)
        drz = np.gradient(rz)
        ds_len = np.sqrt(drx ** 2 + dry ** 2 + drz ** 2) + 1e-10
        tx, ty, tz = drx / ds_len, dry / ds_len, drz / ds_len

        TdotG = tx * dsdx + ty * dsdy + tz * dsdz
        Fx = dsdx - TdotG * tx
        Fy = dsdy - TdotG * ty
        Fz = dsdz - TdotG * tz

        L = float(np.sqrt(np.sum((np.diff(rx) ** 2 + np.diff(ry) ** 2 + np.diff(rz) ** 2))))
        ds2 = (L / (n_ray - 1)) ** 2

        rx[1:-1] += bend_alpha * ds2 * Fx[1:-1]
        ry[1:-1] += bend_alpha * ds2 * Fy[1:-1]
        rz[1:-1] += bend_alpha * ds2 * Fz[1:-1]

        rx = np.clip(rx, 0., 15.)
        ry = np.clip(ry, 0., 15.)
        rz = np.clip(rz, 0., z_max)

    return rx, ry, rz


def incidence_angle_from_ray(rx, ry, rz):
    """
    Incidence angle (degrees, 0-90, 0=vertical) of the ray's final segment arriving at the
    receiver (last two points of the ray arrays).
    """
    dx = rx[-1] - rx[-2]
    dy = ry[-1] - ry[-2]
    dz = rz[-1] - rz[-2]
    horizontal = np.sqrt(dx ** 2 + dy ** 2)
    return float(np.degrees(np.arctan2(horizontal, np.abs(dz))))


def raytraced_incidence_angle(eq_lat, eq_lon, eq_depth, sta_lat, sta_lon, **bend_ray_kwargs):
    """
    S-wave ray-traced incidence angle at the receiver, through the Baillard 3D S-velocity
    model, given event/station lat-lon-depth (matching the calling convention of
    splitting_functions.calculate_incidence_angle, the TauP-based equivalent).

    Parameters:
    -----------
    eq_lat, eq_lon : float
        Earthquake latitude and longitude in degrees
    eq_depth : float
        Earthquake depth in km, positive down, relative to the same flat datum the station is
        treated as sitting at (z=0) - i.e. pass the catalog's raw depth value, not an elevation.
    sta_lat, sta_lon : float
        Station latitude and longitude in degrees

    Returns:
    --------
    float
        Incidence angle in degrees (0-90°), or NaN if the event is above the surface or
        deeper than the velocity model's range (z_max, default 4.0 km).
    """
    eq_x, eq_y = ll2xy(eq_lat, eq_lon)
    sta_x, sta_y = ll2xy(sta_lat, sta_lon)
    eq_z = float(eq_depth)

    z_max = bend_ray_kwargs.get('z_max', Z_MAX)
    if eq_z < 0 or eq_z > z_max:
        return np.nan

    rx, ry, rz = bent_ray(float(eq_x), float(eq_y), eq_z, float(sta_x), float(sta_y), **bend_ray_kwargs)
    return incidence_angle_from_ray(rx, ry, rz)
