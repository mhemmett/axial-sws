"""
pykonal_raytracer.py

True eikonal (shortest-path) ray tracing through the Baillard 3D S-wave
velocity model using PyKonal's Fast Marching Method.

Strategy
--------
For each OBS station the FMM eikonal equation is solved ONCE with the
station as the source (T=0).  Every subsequent call to trace() just
follows the gradient of the pre-computed travel-time field from the
earthquake location back to the station — much faster than re-solving FMM
per event.

Coordinate system
-----------------
  x = East  [km from 130.1°W, 45.9°N origin]
  y = North  [km]
  z = depth below seafloor [km, positive downward]

The Baillard NLL model header gives z_min = −0.5 km (500 m above seafloor).
"""

import numpy as np
import pykonal
import warnings
from typing import Dict, Tuple

# Baillard model constants (from .hdr)
_NX, _NY, _NZ = 302, 302, 92
_DX, _DY, _DZ = 0.05, 0.05, 0.05   # km
_OX, _OY, _OZ = 0.0, 0.0, -0.5    # km origin (z=-0.5 = 500m above seafloor)

# Subsample factor for FMM grid (3 → 0.15 km node spacing)
_STRIDE = 3
_BATHY_FILE = ('/Users/mhemmett/Seismology/axial-splitting-ml/data/'
               'AXIAL_MODEL_3P_VELOCITY.S.mod.buf')

if not hasattr(np, 'infty'):
    np.infty = float('inf')


class BaillardRayTracer:
    """
    FMM ray tracer using the Baillard 3D S-wave velocity model.

    Usage
    -----
    tracer = BaillardRayTracer()
    tracer.precompute_station('AXEC2', sx_km, sy_km)
    ray_xyz = tracer.trace('AXEC2', eq_x, eq_y, eq_z)   # (N,3) array
    """

    def __init__(self, stride: int = _STRIDE, bathy_file: str = _BATHY_FILE):
        self.stride = stride

        # Load and subsample Baillard Vs model
        raw = np.fromfile(bathy_file, dtype=np.float32).reshape(_NX, _NY, _NZ)
        s = stride
        self._vs = np.clip(raw[::s, ::s, ::s], 0.30, 5.0).astype(np.float64)
        self._nx, self._ny, self._nz = self._vs.shape
        self._dx = _DX * s; self._dy = _DY * s; self._dz = _DZ * s
        self._ox = _OX;     self._oy = _OY;     self._oz = _OZ

        # Pre-computed traveltime fields per station {sta_name: Field3D}
        self._tt: Dict[str, object] = {}

        print(f'BaillardRayTracer: grid {self._nx}×{self._ny}×{self._nz}  '
              f'spacing {self._dx:.3f}×{self._dy:.3f}×{self._dz:.3f} km  '
              f'(stride={stride})')

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _xyz_to_idx(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Physical coords [km] → nearest grid index, clamped to bounds."""
        ix = int(round((x - self._ox) / self._dx))
        iy = int(round((y - self._oy) / self._dy))
        iz = int(round((z - self._oz) / self._dz))
        ix = max(0, min(ix, self._nx - 1))
        iy = max(0, min(iy, self._ny - 1))
        iz = max(0, min(iz, self._nz - 1))
        return ix, iy, iz

    def _fresh_solver(self) -> pykonal.EikonalSolver:
        """Create and configure a fresh PyKonal solver."""
        solver = pykonal.EikonalSolver(coord_sys='cartesian')
        solver.velocity.min_coords    = (self._ox, self._oy, self._oz)
        solver.velocity.node_intervals = (self._dx, self._dy, self._dz)
        solver.velocity.npts           = (self._nx, self._ny, self._nz)
        solver.velocity.values         = self._vs.copy()
        return solver

    # ── Public API ────────────────────────────────────────────────────────────

    def precompute_station(self, sta_name: str, sx: float, sy: float,
                            sz: float = 0.0) -> None:
        """
        Solve the eikonal equation from station (sx, sy, sz) and cache the
        resulting travel-time field for subsequent ray traces.

        Parameters
        ----------
        sta_name : str   Station identifier
        sx, sy   : float  Station position [km East, km North from origin]
        sz       : float  Station depth [km]; 0 = seafloor (default)
        """
        import time
        solver = self._fresh_solver()
        solver.traveltime.values[:] = np.inf
        solver.unknown[:] = True

        ix, iy, iz = self._xyz_to_idx(sx, sy, sz)
        solver.traveltime.values[ix, iy, iz] = 0.0
        solver.unknown[ix, iy, iz] = False
        solver.trial.push(ix, iy, iz)

        t0 = time.time()
        solver.solve()
        dt = time.time() - t0
        print(f'  FMM {sta_name} ({sx:.2f},{sy:.2f}) → {dt:.1f}s')

        self._tt[sta_name] = solver.traveltime   # store the field

    def trace(self, sta_name: str,
              eq_x: float, eq_y: float, eq_z: float,
              n_pts: int = 50) -> np.ndarray:
        """
        Trace the minimum-time ray from earthquake to station.

        The FMM field was solved from the station, so trace_ray follows
        the gradient from the earthquake location back to the station
        (receiver→source in PyKonal convention).  We reverse the result
        so the returned array runs source→receiver.

        Parameters
        ----------
        sta_name : str   Must have been precomputed via precompute_station()
        eq_x/y/z : float  Earthquake position [km]
        n_pts    : int   Points in the resampled output ray (default 50)

        Returns
        -------
        ray : ndarray (n_pts, 3)  [[x0,y0,z0], …, [xN,yN,zN]] source→receiver
        """
        if sta_name not in self._tt:
            raise KeyError(f'Station {sta_name} not pre-computed. '
                           f'Call precompute_station() first.')

        tt_field = self._tt[sta_name]
        eq_coords = np.array([eq_x, eq_y, eq_z], dtype=np.float64)

        # Clamp earthquake to model bounds
        eq_coords[0] = np.clip(eq_coords[0], self._ox,
                               self._ox + (self._nx-1)*self._dx)
        eq_coords[1] = np.clip(eq_coords[1], self._oy,
                               self._oy + (self._ny-1)*self._dy)
        eq_coords[2] = np.clip(eq_coords[2], self._oz,
                               self._oz + (self._nz-1)*self._dz)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                raw_ray = tt_field.trace_ray(eq_coords)   # receiver→source

            if len(raw_ray) < 2 or not np.all(np.isfinite(raw_ray)):
                raise ValueError('invalid ray')

            ray = raw_ray[::-1]   # now source→receiver

        except Exception as exc:
            # FIX 4: the previous fallback fabricated a vertical straight line to a
            # PLACEHOLDER station at [eq_x, eq_y, 0.], which is physically wrong and
            # would silently inject a bogus ray into the inversion.  Rather than
            # returning a wrong ray we now raise a clear error identifying the
            # station/event.  The calling loops guard with
            # `if len(vcols) == 0: continue`, so a tracing failure should be
            # handled there (skip + count), never papered over with a fake ray.
            raise RuntimeError(
                f'trace_ray failed for station {sta_name!r} at event '
                f'(x={eq_x:.3f}, y={eq_y:.3f}, z={eq_z:.3f}) km: {exc}'
            ) from exc

        # Resample to n_pts with uniform arc-length spacing
        d = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(ray, axis=0), axis=1))])
        if d[-1] == 0:
            return ray
        t_new = np.linspace(0, d[-1], n_pts)
        return np.column_stack([np.interp(t_new, d, ray[:, k]) for k in range(3)])

    def ray_to_voxels(self, ray: np.ndarray,
                       xn: np.ndarray, yn: np.ndarray, zn: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a ray path to (voxel_indices, path_lengths) for the
        inversion grid defined by (xn, yn, zn).

        Parameters
        ----------
        ray  : (N,3) array    ray coordinates [km]
        xn,yn,zn : 1-D arrays  inversion voxel centres / edges [km]

        Returns
        -------
        cols : int array   flat voxel indices into NX×NY×NZ grid
        vals : float array  path lengths [km] through each voxel
        """
        # TODO (human decision — Vs double-counting / kernel-vs-conversion convention):
        # This kernel is purely GEOMETRIC — A_ij is the ray path length [km] through
        # voxel j, so the recovered u_j,v_j are anisotropy *per km* (m_j in s/km).
        # The downstream ΔVs/Vs conversion (m_to_fractional / m_to_dvs_over_vs) then
        # re-divides by the local Vs.  Whether δt should instead be written as a
        # *travel-time* kernel (path length / Vs) so that the model parameter is the
        # dimensionless anisotropy directly — avoiding a possible double use of Vs —
        # is a modelling decision flagged for the human.  Do NOT change this kernel
        # without coordinating that convention with the conversion step.
        NX, NY, NZ = len(xn), len(yn), len(zn)
        VXY = xn[1]-xn[0] if len(xn)>1 else 0.3
        VZ  = zn[1]-zn[0] if len(zn)>1 else 0.25
        X0, Y0, Z0 = xn[0], yn[0], zn[0]

        # Segment midpoints
        mid = 0.5*(ray[:-1] + ray[1:])
        ds  = np.linalg.norm(np.diff(ray, axis=0), axis=1)

        # NOTE (registration convention, see FIX 3 in the plotting scripts):
        # floor((mid-X0)/VXY) treats xn[j] as the LEFT EDGE of voxel j, so voxel j
        # spans [xn[j], xn[j]+VXY).  Its physical centre is therefore xn[j]+VXY/2.
        # Plotting code must use cell centres (xn+VXY/2, yn+VXY/2, zn+VZ/2) to avoid a
        # half-voxel mis-registration of the displayed field.
        ix = np.floor((mid[:,0]-X0)/VXY).astype(int)
        iy = np.floor((mid[:,1]-Y0)/VXY).astype(int)
        iz = np.floor((mid[:,2]-Z0)/VZ ).astype(int)

        valid = (ix>=0)&(ix<NX)&(iy>=0)&(iy<NY)&(iz>=0)&(iz<NZ)
        flat_idx = (ix*NY*NZ + iy*NZ + iz)[valid]
        seg_len  = ds[valid]

        # Merge duplicate voxels
        if len(flat_idx) == 0:
            return np.array([], dtype=int), np.array([])
        u, inv = np.unique(flat_idx, return_inverse=True)
        merged = np.zeros(len(u))
        np.add.at(merged, inv, seg_len)
        return u, merged
