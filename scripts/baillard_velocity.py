"""
baillard_velocity.py

Loader and interpolator for Baillard's 3D S-wave velocity model.

NLL format: AXIAL_MODEL_3P_VELOCITY.S.mod.{hdr,buf}
  302 × 302 × 92 nodes, 50 m spacing, origin (0,0,-0.5 km)
  Coordinate origin: 45.9°N, 130.1°W  (same as sws_forward_model.ll2xy)
  x = East [km], y = North [km], z = depth [km] positive downward
"""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

_DIR = os.path.join(os.path.dirname(__file__),
                    '..', 'data',
                    'AXIAL_MODEL_3P_VELOCITY.S.mod.buf')
_DIR = os.path.normpath(_DIR)

# Grid parameters from header
_NX, _NY, _NZ = 302, 302, 92
_DX, _DY, _DZ = 0.05, 0.05, 0.05   # km
_OX, _OY, _OZ = 0.0, 0.0, -0.5    # km origin

_interp = None   # lazy-loaded singleton


def _load():
    global _interp
    if _interp is not None:
        return _interp

    buf = np.fromfile(_DIR, dtype=np.float32).reshape(_NX, _NY, _NZ)

    x = _OX + np.arange(_NX) * _DX   # km East
    y = _OY + np.arange(_NY) * _DY   # km North
    z = _OZ + np.arange(_NZ) * _DZ   # km depth (positive down, starts -0.5)

    # Clamp to physically reasonable range (remove any numerical artefacts)
    buf = np.clip(buf, 0.3, 5.0)

    _interp = RegularGridInterpolator(
        (x, y, z), buf,
        method='linear',
        bounds_error=False,
        fill_value=None,   # extrapolate at edges
    )
    return _interp


def vs_at(x_km, y_km, z_km):
    """
    S-wave velocity [km/s] at one or many points.

    Parameters
    ----------
    x_km : float or array  East km from model origin (45.9°N, 130.1°W)
    y_km : float or array  North km from model origin
    z_km : float or array  Depth km positive downward (0 = seafloor)

    Returns
    -------
    vs : float or ndarray  [km/s]
    """
    interp = _load()
    pts = np.column_stack([
        np.atleast_1d(x_km),
        np.atleast_1d(y_km),
        np.atleast_1d(z_km),
    ])
    vs = interp(pts)
    scalar = (np.ndim(x_km) == 0)
    return float(vs[0]) if scalar else vs


def ray_vs(x0, y0, z0, x1, y1, z1, n=30):
    """
    Sample Vs at n points along a straight-line ray from (x0,y0,z0) to (x1,y1,z1).

    Returns
    -------
    vs_arr : (n,) ndarray  [km/s]
    ds     : float  segment length [km]
    """
    t = np.linspace(0., 1., n)
    rx = x0 + t * (x1 - x0)
    ry = y0 + t * (y1 - y0)
    rz = z0 + t * (z1 - z0)
    vs_arr = vs_at(rx, ry, rz)
    seg_len = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2) / n  # km per segment
    return vs_arr, seg_len
