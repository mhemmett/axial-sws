"""
Earthquake Location Algorithms for Axial Seamount

This module provides earthquake location functionality using travel time inversion
with PyKonal for 3D velocity models and ray tracing for real OBS data analysis.

Author: Michael Hemmett
Date: December 2025
"""

import numpy as np
import pykonal
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

if not hasattr(np, "infty"):
    np.infty = float("inf")


class EarthquakeLocator:
    """
    Earthquake location using travel time inversion with PyKonal.
    
    Parameters
    ----------
    velocity_model : AxialVelocityModel
        3D velocity model object
    obs_coords : ndarray
        Array of shape (n_stations, 3) with station coordinates
    
    Attributes
    ----------
    velocity_model : AxialVelocityModel
        3D velocity model for travel time computation
    obs_coords : ndarray
        Station coordinates array
    
    Notes
    -----
    Creates fresh solver instances for each computation to avoid
    state contamination issues with PyKonal.
    """
    
    def __init__(self, velocity_model, obs_coords):
        """
        Initialize earthquake locator.
        
        Creates fresh solver instances for each computation to avoid
        state contamination issues.
        
        Parameters
        ----------
        velocity_model : AxialVelocityModel
            Velocity model object with created model
        obs_coords : ndarray
            Station coordinates array (n_stations, 3)
        """
        self.velocity_model = velocity_model
        self.obs_coords = obs_coords
        
        # Validate velocity model
        if velocity_model.velocity is None:
            raise ValueError("Velocity model not created. Call velocity_model.create_model() first.")
    
    def compute_travel_times(self, eq_x, eq_y, eq_z):
        """
        Compute travel times from earthquake to all stations.
        
        Parameters
        ----------
        eq_x, eq_y, eq_z : float
            Earthquake coordinates in km
        
        Returns
        -------
        travel_times : ndarray
            Travel times to each station in seconds
        """
        vm = self.velocity_model
        
        # Convert earthquake location to grid indices
        src_idx = (
            int((eq_x - vm.x_min) / (vm.x_max - vm.x_min) * (vm.nx - 1)),
            int((eq_y - vm.y_min) / (vm.y_max - vm.y_min) * (vm.ny - 1)),
            int((eq_z - vm.z_min) / (vm.z_max - vm.z_min) * (vm.nz - 1))
        )
        
        # Ensure indices are within bounds
        src_idx = (
            max(0, min(src_idx[0], vm.nx - 1)),
            max(0, min(src_idx[1], vm.ny - 1)),
            max(0, min(src_idx[2], vm.nz - 1))
        )
        
        # Create fresh solver to avoid state contamination
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = vm.x_min, vm.y_min, vm.z_min
        solver.velocity.node_intervals = (
            (vm.x_max - vm.x_min) / (vm.nx - 1),
            (vm.y_max - vm.y_min) / (vm.ny - 1),
            (vm.z_max - vm.z_min) / (vm.nz - 1)
        )
        solver.velocity.npts = vm.nx, vm.ny, vm.nz
        solver.velocity.values = vm.velocity
        
        # Initialize solver state
        solver.traveltime.values[:] = np.inf
        solver.unknown[:] = True
        
        # Set source location
        solver.traveltime.values[src_idx] = 0.0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)
        
        # Solve eikonal equation
        solver.solve()
        
        # Extract travel times at station locations
        interpolator = RegularGridInterpolator(
            (vm.x, vm.y, vm.z),
            solver.traveltime.values,
            method='linear',
            bounds_error=False
        )
        
        travel_times = interpolator(self.obs_coords)
        
        return travel_times
    
    def misfit_function(self, location, observed_times):
        """
        Calculate RMS misfit between observed and predicted travel times.
        
        Parameters
        ----------
        location : array-like
            Candidate earthquake location [x, y, z] in km
        observed_times : ndarray
            Observed travel times at stations
        
        Returns
        -------
        misfit : float
            RMS travel time residual in seconds
        """
        eq_x, eq_y, eq_z = location
        
        # Check if location is within model bounds
        vm = self.velocity_model
        if not (vm.x_min <= eq_x <= vm.x_max and
                vm.y_min <= eq_y <= vm.y_max and
                vm.z_min <= eq_z <= vm.z_max):
            return 1e10  # Large penalty for out-of-bounds
        
        # Compute predicted travel times
        predicted_times = self.compute_travel_times(eq_x, eq_y, eq_z)
        
        # Calculate RMS misfit
        residuals = observed_times - predicted_times
        rms_misfit = np.sqrt(np.mean(residuals**2))
        
        return rms_misfit
    
    def grid_search(self, observed_times, search_bounds,
                   n_points=(7, 7, 6), verbose=True):
        """
        Locate earthquake using grid search.
        
        Parameters
        ----------
        observed_times : ndarray
            Observed travel times at stations
        search_bounds : tuple
            ((x_min, x_max), (y_min, y_max), (z_min, z_max)) for search
        n_points : tuple
            Number of grid points in each direction (nx, ny, nz)
        verbose : bool
            Print progress information
        
        Returns
        -------
        best_location : ndarray
            Best-fit earthquake location [x, y, z]
        min_misfit : float
            Minimum RMS misfit achieved
        misfit_grid : ndarray
            3D array of misfit values
        grid_coords : tuple
            Coordinate arrays for grid (x_grid, y_grid, z_grid)
        """
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = search_bounds
        nx, ny, nz = n_points
        
        # Create search grid
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        z_grid = np.linspace(z_min, z_max, nz)
        
        # Initialize misfit grid
        misfit_grid = np.zeros((nx, ny, nz))
        
        # Evaluate misfit at each grid point
        total_points = nx * ny * nz
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                for k, z in enumerate(z_grid):
                    misfit_grid[i, j, k] = self.misfit_function([x, y, z], observed_times)
                    
                    if verbose and ((i*ny*nz + j*nz + k + 1) % max(1, total_points//10) == 0):
                        progress = (i*ny*nz + j*nz + k + 1) / total_points * 100
                        print(f"  Grid search progress: {progress:.0f}%")
        
        # Find minimum misfit location
        min_idx = np.unravel_index(np.argmin(misfit_grid), misfit_grid.shape)
        best_location = np.array([x_grid[min_idx[0]], y_grid[min_idx[1]], z_grid[min_idx[2]]])
        min_misfit = misfit_grid[min_idx]
        
        if verbose:
            print(f"\nGrid search complete:")
            print(f"  Best location: ({best_location[0]:.3f}, {best_location[1]:.3f}, {best_location[2]:.3f}) km")
            print(f"  Minimum RMS misfit: {min_misfit:.4f} s")
        
        return best_location, min_misfit, misfit_grid, (x_grid, y_grid, z_grid)
    
    def refine_location(self, initial_location, observed_times,
                       method='Nelder-Mead', verbose=True):
        """
        Refine earthquake location using optimization.
        
        Parameters
        ----------
        initial_location : array-like
            Initial guess for location [x, y, z]
        observed_times : ndarray
            Observed travel times at stations
        method : str
            Optimization method (see scipy.optimize.minimize)
        verbose : bool
            Print progress information
        
        Returns
        -------
        result : OptimizeResult
            Optimization result with refined location
        """
        if verbose:
            print(f"\nRefining location using {method} optimization...")
        
        result = minimize(
            fun=lambda loc: self.misfit_function(loc, observed_times),
            x0=initial_location,
            method=method,
            options={'disp': verbose}
        )
        
        if verbose:
            print(f"\nOptimization complete:")
            print(f"  Refined location: ({result.x[0]:.3f}, {result.x[1]:.3f}, {result.x[2]:.3f}) km")
            print(f"  Final RMS misfit: {result.fun:.4f} s")
            print(f"  Iterations: {result.nit}")
        
        return result
    
    def generate_synthetic_data(self, true_location, noise_level=0.05, seed=None):
        """
        Generate synthetic travel time observations with noise.
        
        Parameters
        ----------
        true_location : array-like
            True earthquake location [x, y, z]
        noise_level : float
            Standard deviation of Gaussian noise in seconds
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        observed_times : ndarray
            Synthetic observed travel times
        true_times : ndarray
            Noise-free travel times
        noise : ndarray
            Noise added to each observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Compute true travel times
        true_times = self.compute_travel_times(*true_location)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, len(true_times))
        observed_times = true_times + noise
        
        return observed_times, true_times, noise


class RayTracer:
    """
    Ray tracing for computing ray paths and angles of incidence.
    
    This class computes seismic ray paths through the velocity model and
    calculates angles of incidence at receivers. Important for shear-wave
    splitting analysis where steep angles (< 30° from vertical) are needed
    to avoid S-to-P conversions and reflections.
    
    Parameters
    ----------
    velocity_model : AxialVelocityModel
        3D velocity model
    """
    
    def __init__(self, velocity_model):
        self.velocity_model = velocity_model
        # No persistent solver - create fresh instance for each ray trace
    
    def _create_fresh_solver(self):
        """Create a fresh PyKonal solver instance for ray tracing.
        
        This avoids state contamination from previous solve() calls
        that can cause NaN values in subsequent ray traces.
        
        Returns
        -------
        solver : pykonal.EikonalSolver
            Fresh solver instance configured with velocity model
        """
        vm = self.velocity_model
        
        if vm.velocity is None:
            raise ValueError("Velocity model not created.")
        
        # Create fresh PyKonal solver
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        
        # Define grid
        solver.velocity.min_coords = vm.x_min, vm.y_min, vm.z_min
        solver.velocity.node_intervals = (
            (vm.x_max - vm.x_min) / (vm.nx - 1),
            (vm.y_max - vm.y_min) / (vm.ny - 1),
            (vm.z_max - vm.z_min) / (vm.nz - 1)
        )
        solver.velocity.npts = vm.nx, vm.ny, vm.nz
        solver.velocity.values = vm.velocity
        
        return solver
    
    def compute_ray_path(self, source_loc, receiver_loc, npts=100):
        """
        Compute ray path from source to receiver using PyKonal's trace_ray.
        
        Traces the ray by following the gradient of the traveltime field
        from receiver back to source, then resamples for smooth visualization.
        
        IMPORTANT: PyKonal's methods require different coordinate systems:
        - trial.push() expects grid indices (integers)
        - trace_ray() expects physical coordinates (floats in km)
        
        Parameters
        ----------
        source_loc : array-like
            Source location [x, y, z] in km (physical coordinates)
        receiver_loc : array-like
            Receiver location [x, y, z] in km (physical coordinates)
        npts : int
            Number of points for final resampled ray path (default: 100)
        
        Returns
        -------
        ray_path : ndarray
            Array of shape (npts, 3) with [x, y, z] along ray from source to receiver
        """
        vm = self.velocity_model
        
        # Convert source to grid indices for solver initialization
        src_idx = (
            int((source_loc[0] - vm.x_min) / (vm.x_max - vm.x_min) * (vm.nx - 1)),
            int((source_loc[1] - vm.y_min) / (vm.y_max - vm.y_min) * (vm.ny - 1)),
            int((source_loc[2] - vm.z_min) / (vm.z_max - vm.z_min) * (vm.nz - 1))
        )
        
        # Ensure source indices within bounds
        src_idx = (
            max(0, min(src_idx[0], vm.nx - 1)),
            max(0, min(src_idx[1], vm.ny - 1)),
            max(0, min(src_idx[2], vm.nz - 1))
        )
        
        # Convert receiver to numpy array with physical coordinates for trace_ray
        # NOTE: trace_ray expects physical coordinates (km), not grid indices!
        rec_coords = np.array(receiver_loc, dtype=np.float64)
        
        # Create fresh solver to avoid state contamination
        solver = self._create_fresh_solver()
        
        # Initialize solver state
        solver.traveltime.values[:] = np.inf
        solver.unknown[:] = True
        
        # Set source location (uses grid indices)
        solver.traveltime.values[src_idx] = 0.0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)
        solver.solve()
        
        try:
            # Use PyKonal's native trace_ray method
            # IMPORTANT: Pass physical coordinates (km), not grid indices!
            ray = solver.traveltime.trace_ray(rec_coords)
            
            # Validate ray
            if len(ray) < 2:
                raise ValueError(f"Ray tracing returned too few points: {len(ray)}")
            
            if not np.all(np.isfinite(ray)):
                raise ValueError("Ray path contains NaN or infinite values")
            
            # PyKonal traces from receiver to source, so reverse it
            ray = ray[::-1]
            
            # Resample to exactly npts points with uniform spacing along the ray
            distances = np.zeros(len(ray))
            for i in range(1, len(ray)):
                distances[i] = distances[i-1] + np.linalg.norm(ray[i] - ray[i-1])
            
            total_distance = distances[-1]
            
            if total_distance > 0:
                # Create uniform spacing
                t_new = np.linspace(0, total_distance, npts)
                
                # Interpolate ray coordinates
                ray_resampled = np.column_stack([
                    np.interp(t_new, distances, ray[:, 0]),
                    np.interp(t_new, distances, ray[:, 1]),
                    np.interp(t_new, distances, ray[:, 2])
                ])
                
                return ray_resampled
            else:
                return ray
                
        except (ValueError, RuntimeError, IndexError, AttributeError) as e:
            # Fallback to straight line if ray tracing fails
            print(f"Warning: Ray tracing failed ({str(e)}), using straight line")
            t = np.linspace(0, 1, npts)
            ray = np.outer(1-t, source_loc) + np.outer(t, receiver_loc)
            return ray
    
    def _compute_gradient(self, pos, solver):
        """Compute gradient of travel time field at position.
        
        Parameters
        ----------
        pos : array-like
            Position [x, y, z] in km
        solver : pykonal.EikonalSolver
            Solver instance with computed traveltime field
        
        Returns
        -------
        grad : ndarray
            Gradient vector at position
        """
        vm = self.velocity_model
        
        # Interpolate travel time and compute gradient
        interpolator = RegularGridInterpolator(
            (vm.x, vm.y, vm.z),
            solver.traveltime.values,
            method='linear',
            bounds_error=False
        )
        
        # Numerical gradient
        h = 0.01  # km
        grad = np.zeros(3)
        
        for i in range(3):
            pos_plus = pos.copy()
            pos_minus = pos.copy()
            pos_plus[i] += h
            pos_minus[i] -= h
            
            t_plus = interpolator(pos_plus.reshape(1, -1))[0]
            t_minus = interpolator(pos_minus.reshape(1, -1))[0]
            
            grad[i] = (t_plus - t_minus) / (2 * h)
        
        return grad
    
    def compute_incidence_angle(self, source_loc, receiver_loc):
        """
        Compute angle of incidence at receiver from vertical.
        
        Critical for shear-wave splitting: angles < 30° from vertical avoid
        S-to-P conversions and free-surface reflections that complicate splitting.
        
        Uses fresh solver internally (via compute_ray_path) to avoid state
        contamination from previous calls.
        
        Parameters
        ----------
        source_loc : array-like
            Source location [x, y, z] in km
        receiver_loc : array-like
            Receiver location [x, y, z] in km
        
        Returns
        -------
        angle : float
            Angle of incidence from vertical in degrees
        takeoff_angle : float
            Ray takeoff angle at source from vertical in degrees
        """
        try:
            # Compute ray path (creates fresh solver internally)
            ray_path = self.compute_ray_path(source_loc, receiver_loc, npts=50)
            
            # Validate ray path
            if len(ray_path) < 2:
                raise ValueError("Ray path too short")
            
            if not np.all(np.isfinite(ray_path)):
                raise ValueError("Ray path contains NaN or infinite values")
            
            # Angle of incidence at receiver (last segment)
            last_segment = ray_path[-1] - ray_path[-2]
            last_norm = np.linalg.norm(last_segment)
            
            if last_norm < 1e-10:
                raise ValueError("Last segment too small")
            
            # Angle from vertical (positive z is down)
            cos_inc = abs(last_segment[2]) / last_norm
            cos_inc = np.clip(cos_inc, 0.0, 1.0)  # Ensure valid range
            incidence_angle = np.degrees(np.arccos(cos_inc))
            
            # Takeoff angle at source (first segment)
            first_segment = ray_path[1] - ray_path[0]
            first_norm = np.linalg.norm(first_segment)
            
            if first_norm < 1e-10:
                raise ValueError("First segment too small")
            
            cos_takeoff = abs(first_segment[2]) / first_norm
            cos_takeoff = np.clip(cos_takeoff, 0.0, 1.0)
            takeoff_angle = np.degrees(np.arccos(cos_takeoff))
            
        except (ValueError, RuntimeError, IndexError) as e:
            # Fallback to straight-line geometry
            direction = np.array(receiver_loc) - np.array(source_loc)
            distance = np.linalg.norm(direction)
            
            if distance < 1e-10:
                return 90.0, 90.0  # Undefined, return horizontal
            
            direction = direction / distance
            cos_angle = abs(direction[2])
            cos_angle = np.clip(cos_angle, 0.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            # Same angle for incidence and takeoff in straight-line case
            incidence_angle = angle
            takeoff_angle = angle
        
        return incidence_angle, takeoff_angle
    
    def filter_events_by_incidence(self, event_locs, station_locs,
                                   max_angle=30.0, verbose=False):
        """
        Filter earthquake-station pairs by incidence angle.
        
        For splitting analysis, we typically require incidence angles < 30°
        from vertical to avoid complications from conversions and reflections.
        
        Parameters
        ----------
        event_locs : ndarray
            Array of shape (n_events, 3) with event locations
        station_locs : ndarray
            Array of shape (n_stations, 3) with station locations
        max_angle : float
            Maximum allowed incidence angle from vertical in degrees
        verbose : bool
            Print filtering statistics
        
        Returns
        -------
        filtered_pairs : list
            List of (event_idx, station_idx, angle) tuples that pass filter
        """
        filtered_pairs = []
        
        n_events = len(event_locs)
        n_stations = len(station_locs)
        total_pairs = n_events * n_stations
        
        for i, event_loc in enumerate(event_locs):
            for j, station_loc in enumerate(station_locs):
                angle, _ = self.compute_incidence_angle(event_loc, station_loc)
                
                if angle is not None and angle <= max_angle:
                    filtered_pairs.append((i, j, angle))
        
        if verbose:
            print(f"Incidence angle filtering:")
            print(f"  Total pairs: {total_pairs}")
            print(f"  Passed filter (<{max_angle}°): {len(filtered_pairs)}")
            print(f"  Pass rate: {100*len(filtered_pairs)/total_pairs:.1f}%")
        
        return filtered_pairs
    
    def _trace_ray_with_velocity(self, source_loc, receiver_loc, velocity_model, npts=100):
        """
        Trace a single ray through a specific velocity model.
        
        Helper method for tracing fast and slow rays separately.
        
        Parameters
        ----------
        source_loc : array-like
            Source location [x, y, z] in km
        receiver_loc : array-like
            Receiver location [x, y, z] in km
        velocity_model : ndarray
            3D velocity model to use for ray tracing
        npts : int
            Number of points for resampled ray path
        
        Returns
        -------
        ray_path : ndarray
            Array of shape (npts, 3) with ray path coordinates
        """
        vm = self.velocity_model
        
        # Convert to numpy arrays
        src = np.array(source_loc)
        rec = np.array(receiver_loc)
        
        # Bounds checking
        MARGIN = 0.1
        if not (vm.x_min + MARGIN <= src[0] <= vm.x_max - MARGIN and
                vm.y_min + MARGIN <= src[1] <= vm.y_max - MARGIN and
                vm.z_min + MARGIN <= src[2] <= vm.z_max - MARGIN):
            # Return straight line if out of bounds
            t = np.linspace(0, 1, npts)
            return np.outer(1-t, src) + np.outer(t, rec)
        
        # Convert to grid indices
        src_idx = (
            int((src[0] - vm.x_min) / (vm.x_max - vm.x_min) * (vm.nx - 1)),
            int((src[1] - vm.y_min) / (vm.y_max - vm.y_min) * (vm.ny - 1)),
            int((src[2] - vm.z_min) / (vm.z_max - vm.z_min) * (vm.nz - 1))
        )
        
        # Clamp indices with margin
        src_idx = (
            max(1, min(src_idx[0], vm.nx - 2)),
            max(1, min(src_idx[1], vm.ny - 2)),
            max(1, min(src_idx[2], vm.nz - 2))
        )
        
        # Create solver with specified velocity model
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = vm.x_min, vm.y_min, vm.z_min
        solver.velocity.node_intervals = (
            (vm.x_max - vm.x_min) / (vm.nx - 1),
            (vm.y_max - vm.y_min) / (vm.ny - 1),
            (vm.z_max - vm.z_min) / (vm.nz - 1)
        )
        solver.velocity.npts = vm.nx, vm.ny, vm.nz
        solver.velocity.values = velocity_model
        
        # Initialize solver state
        solver.traveltime.values[:] = np.inf
        solver.unknown[:] = True
        
        # Set source
        solver.traveltime.values[src_idx] = 0.0
        solver.unknown[src_idx] = False
        solver.trial.push(*src_idx)
        
        try:
            # Solve eikonal equation
            solver.solve()
            
            # Trace ray from source to receiver
            rec_coords = np.array(rec)
            ray = solver.traveltime.trace_ray(rec_coords)
            
            # Resample to uniform spacing
            if len(ray) < 2:
                raise ValueError("Ray path too short")
            
            # Compute cumulative distance along ray
            distances = np.zeros(len(ray))
            for i in range(1, len(ray)):
                distances[i] = distances[i-1] + np.linalg.norm(ray[i] - ray[i-1])
            
            total_distance = distances[-1]
            if total_distance < 1e-10:
                raise ValueError("Total ray distance too small")
            
            # Interpolate to uniform spacing
            uniform_distances = np.linspace(0, total_distance, npts)
            ray_resampled = np.zeros((npts, 3))
            
            for dim in range(3):
                ray_resampled[:, dim] = np.interp(uniform_distances, distances, ray[:, dim])
            
            return ray_resampled
            
        except (ValueError, RuntimeError, IndexError, AttributeError) as e:
            # Fallback to straight line
            t = np.linspace(0, 1, npts)
            return np.outer(1-t, src) + np.outer(t, rec)
    
    def compute_anisotropic_ray_paths(self, source_loc, receiver_loc, npts=100):
        """
        Compute separate ray paths for fast and slow S-wave components.
        
        This demonstrates how in situ anisotropy causes physical splitting:
        fast and slow shear waves travel along slightly different paths
        through the anisotropic medium because they experience different
        velocities.
        
        Parameters
        ----------
        source_loc : array-like
            Source location [x, y, z] in km
        receiver_loc : array-like
            Receiver location [x, y, z] in km
        npts : int
            Number of points for resampled ray paths
        
        Returns
        -------
        ray_fast : ndarray
            Ray path for fast S-wave component, shape (npts, 3)
        ray_slow : ndarray
            Ray path for slow S-wave component, shape (npts, 3)
        path_difference : dict
            Dictionary with path statistics:
            - 'max_separation': Maximum 3D distance between rays (km)
            - 'mean_separation': Mean 3D distance between rays (km)
            - 'travel_time_fast': Travel time for fast component (s)
            - 'travel_time_slow': Travel time for slow component (s)
            - 'predicted_dt': Predicted delay time (s)
        
        Notes
        -----
        Requires that the velocity model has anisotropic components created
        via velocity_model.create_anisotropic_model().
        """
        vm = self.velocity_model
        
        if vm.velocity_fast is None or vm.velocity_slow is None:
            raise ValueError("Anisotropic velocity models not created. "
                           "Call velocity_model.create_anisotropic_model() first.")
        
        # Trace ray through fast velocity model
        ray_fast = self._trace_ray_with_velocity(source_loc, receiver_loc, 
                                                 vm.velocity_fast, npts)
        
        # Trace ray through slow velocity model
        ray_slow = self._trace_ray_with_velocity(source_loc, receiver_loc,
                                                 vm.velocity_slow, npts)
        
        # Compute path statistics
        separations = np.linalg.norm(ray_fast - ray_slow, axis=1)
        max_separation = np.max(separations)
        mean_separation = np.mean(separations)
        
        # Compute travel times along each ray path
        travel_time_fast = 0.0
        travel_time_slow = 0.0
        
        interpolator_fast = RegularGridInterpolator(
            (vm.x, vm.y, vm.z), vm.velocity_fast,
            method='linear', bounds_error=False, fill_value=None
        )
        interpolator_slow = RegularGridInterpolator(
            (vm.x, vm.y, vm.z), vm.velocity_slow,
            method='linear', bounds_error=False, fill_value=None
        )
        
        # Integrate travel time along fast ray
        for i in range(1, len(ray_fast)):
            segment_length = np.linalg.norm(ray_fast[i] - ray_fast[i-1])
            midpoint = 0.5 * (ray_fast[i] + ray_fast[i-1])
            velocity = interpolator_fast(midpoint.reshape(1, -1))[0]
            if velocity > 0:
                travel_time_fast += segment_length / velocity
        
        # Integrate travel time along slow ray
        for i in range(1, len(ray_slow)):
            segment_length = np.linalg.norm(ray_slow[i] - ray_slow[i-1])
            midpoint = 0.5 * (ray_slow[i] + ray_slow[i-1])
            velocity = interpolator_slow(midpoint.reshape(1, -1))[0]
            if velocity > 0:
                travel_time_slow += segment_length / velocity
        
        predicted_dt = travel_time_slow - travel_time_fast
        
        path_difference = {
            'max_separation': max_separation,
            'mean_separation': mean_separation,
            'travel_time_fast': travel_time_fast,
            'travel_time_slow': travel_time_slow,
            'predicted_dt': predicted_dt
        }
        
        return ray_fast, ray_slow, path_difference
