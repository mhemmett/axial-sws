"""
Visualization Tools for Axial Seamount Project

This module provides plotting functions for velocity models, station arrays,
earthquake locations, and uncertainty analysis.

Author: Michael Hemmett
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_velocity_slices(velocity_model, figsize=(14, 12), save_path=None):
    """
    Plot 4-panel visualization of velocity model slices.
    
    Parameters
    ----------
    velocity_model : AxialVelocityModel
        Velocity model object with created model
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    vm = velocity_model
    
    if vm.velocity is None:
        raise ValueError("Velocity model not created.")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Surface slice (z=0)
    ax = axes[0, 0]
    im = ax.pcolormesh(vm.X[:, :, 0], vm.Y[:, :, 0], vm.velocity[:, :, 0],
                      shading='auto', cmap='viridis', vmin=1.5, vmax=4.5)
    plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    ax.set_xlabel('X (km) - East')
    ax.set_ylabel('Y (km) - North')
    ax.set_title('Velocity at Surface (z=0 km)', fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Mid-depth slice
    mid_z_idx = vm.nz // 3
    ax = axes[0, 1]
    im = ax.pcolormesh(vm.X[:, :, mid_z_idx], vm.Y[:, :, mid_z_idx],
                      vm.velocity[:, :, mid_z_idx],
                      shading='auto', cmap='viridis', vmin=1.5, vmax=4.5)
    plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    ax.set_xlabel('X (km) - East')
    ax.set_ylabel('Y (km) - North')
    ax.set_title(f'Velocity at Depth z={vm.z[mid_z_idx]:.1f} km', fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Vertical slice at y=0 (E-W cross-section)
    y_mid_idx = vm.ny // 2
    ax = axes[1, 0]
    im = ax.pcolormesh(vm.X[:, y_mid_idx, :], vm.Z[:, y_mid_idx, :],
                      vm.velocity[:, y_mid_idx, :],
                      shading='auto', cmap='viridis', vmin=1.5, vmax=4.5)
    plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    ax.set_xlabel('X (km) - East')
    ax.set_ylabel('Depth (km)')
    ax.set_title('E-W Cross-Section (y=0 km)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Vertical slice at x=0 (N-S cross-section)
    x_mid_idx = vm.nx // 2
    ax = axes[1, 1]
    im = ax.pcolormesh(vm.Y[x_mid_idx, :, :], vm.Z[x_mid_idx, :, :],
                      vm.velocity[x_mid_idx, :, :],
                      shading='auto', cmap='viridis', vmin=1.5, vmax=4.5)
    plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    ax.set_xlabel('Y (km) - North')
    ax.set_ylabel('Depth (km)')
    ax.set_title('N-S Cross-Section (x=0 km)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Axial Seamount 3D Velocity Model', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Velocity slices saved to: {save_path}")
    
    return fig, axes


def plot_obs_array_2d(obs_array, velocity_model=None, earthquake_loc=None,
                      figsize=(12, 10), save_path=None):
    """
    Plot 2D map view of OBS array with optional velocity background and earthquake.
    
    Parameters
    ----------
    obs_array : OBSArray
        OBS array object
    velocity_model : AxialVelocityModel, optional
        Velocity model for background
    earthquake_loc : array-like, optional
        Earthquake location [x, y, z]
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot velocity background if provided
    if velocity_model is not None:
        im = ax.pcolormesh(velocity_model.X[:, :, 0], velocity_model.Y[:, :, 0],
                          velocity_model.velocity[:, :, 0],
                          shading='auto', cmap='seismic_r', alpha=0.7, vmin=1.5, vmax=4.5)
        plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    
    # Plot OBS stations
    coords, station_codes = obs_array.get_all_coords('relative')
    for i, code in enumerate(station_codes):
        info = obs_array.stations[code]
        ax.plot(coords[i, 0], coords[i, 1], 's', markersize=12, color='blue',
               markeredgecolor='white', markeredgewidth=2, alpha=0.9, zorder=10)
        ax.text(coords[i, 0], coords[i, 1] + 0.8, code, ha='center', va='bottom',
               fontsize=10, fontweight='bold', color='blue',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot earthquake if provided
    if earthquake_loc is not None:
        ax.plot(earthquake_loc[0], earthquake_loc[1], 'r*', markersize=25,
               markeredgecolor='black', markeredgewidth=2,
               label=f'Earthquake (z={earthquake_loc[2]:.1f} km)', zorder=11)
        ax.legend(loc='upper right', fontsize=11)
    
    # Add caldera outline
    caldera_circle = plt.Circle((0, 0), 4, fill=False, color='black',
                                linewidth=3, linestyle='--', alpha=0.8, label='Caldera')
    ax.add_patch(caldera_circle)
    
    ax.set_xlabel('X (km) - East from Caldera Center', fontsize=12)
    ax.set_ylabel('Y (km) - North from Caldera Center', fontsize=12)
    ax.set_title('Axial Seamount OBS Array', fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.set_ylim(-6, 8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"OBS array map saved to: {save_path}")
    
    return fig, ax


def plot_obs_array_3d(obs_array, earthquake_loc=None, figsize=(12, 10), save_path=None):
    """
    Plot 3D view of OBS array.
    
    Parameters
    ----------
    obs_array : OBSArray
        OBS array object
    earthquake_loc : array-like, optional
        Earthquake location [x, y, z]
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot OBS stations
    coords, station_codes = obs_array.get_all_coords('relative')
    for i, code in enumerate(station_codes):
        ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                  c='blue', marker='s', s=150, edgecolors='white',
                  linewidths=2, alpha=0.9)
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2] + 0.2, code,
               fontsize=9, fontweight='bold')
    
    # Plot earthquake if provided
    if earthquake_loc is not None:
        ax.scatter(earthquake_loc[0], earthquake_loc[1], earthquake_loc[2],
                  c='red', marker='*', s=500, edgecolors='black',
                  linewidths=2, label='Earthquake', zorder=10)
    
    # Add seafloor surface
    x_range = np.linspace(-12, 5, 20)
    y_range = np.linspace(-12, 2, 20)
    X_sea, Y_sea = np.meshgrid(x_range, y_range)
    Z_sea = np.zeros_like(X_sea)
    ax.plot_surface(X_sea, Y_sea, Z_sea, alpha=0.2, color='lightblue')
    
    ax.set_xlabel('X (km) - East', fontsize=11)
    ax.set_ylabel('Y (km) - North', fontsize=11)
    ax.set_zlabel('Depth (km)', fontsize=11)
    ax.set_title('3D View: Axial Seamount OBS Network', fontsize=13, fontweight='bold')
    ax.invert_zaxis()
    if earthquake_loc is not None:
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D OBS array plot saved to: {save_path}")
    
    return fig, ax


def plot_misfit_grid(misfit_grid, grid_coords, true_location=None,
                    located_position=None, figsize=(16, 12), save_path=None):
    """
    Plot misfit grid from grid search as depth slices.
    
    Parameters
    ----------
    misfit_grid : ndarray
        3D array of misfit values
    grid_coords : tuple
        (x_grid, y_grid, z_grid) coordinate arrays
    true_location : array-like, optional
        True earthquake location [x, y, z]
    located_position : array-like, optional
        Located earthquake position [x, y, z]
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    x_grid, y_grid, z_grid = grid_coords
    nz = len(z_grid)
    
    # Determine subplot layout
    n_rows = (nz + 2) // 3
    n_cols = min(3, nz)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    # Plot each depth slice
    for k, z_val in enumerate(z_grid):
        ax = axes[k]
        
        # Contour plot of misfit
        X, Y = np.meshgrid(x_grid, y_grid)
        levels = np.linspace(misfit_grid.min(), misfit_grid.min() * 2, 15)
        cs = ax.contourf(X, Y, misfit_grid[:, :, k].T, levels=levels, cmap='viridis_r')
        ax.contour(X, Y, misfit_grid[:, :, k].T, levels=levels, colors='k',
                  linewidths=0.5, alpha=0.3)
        
        # Mark true location if provided
        if true_location is not None and np.isclose(true_location[2], z_val, atol=0.5):
            ax.plot(true_location[0], true_location[1], 'r*', markersize=20,
                   markeredgecolor='white', markeredgewidth=1.5,
                   label='True location')
        
        # Mark located position if provided
        if located_position is not None and np.isclose(located_position[2], z_val, atol=0.5):
            ax.plot(located_position[0], located_position[1], 'g^', markersize=15,
                   markeredgecolor='white', markeredgewidth=1.5,
                   label='Located')
        
        ax.set_xlabel('X (km)', fontsize=10)
        ax.set_ylabel('Y (km)', fontsize=10)
        ax.set_title(f'Depth = {z_val:.2f} km', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)
    
    # Hide unused subplots
    for k in range(nz, len(axes)):
        axes[k].axis('off')
    
    # Add colorbar
    fig.colorbar(cs, ax=axes, label='RMS Misfit (s)', pad=0.02)
    
    plt.suptitle('Grid Search Results: Misfit at Each Depth',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Misfit grid plot saved to: {save_path}")
    
    return fig, axes


def plot_location_uncertainty(misfit_grid, grid_coords, best_location,
                             confidence_levels=[0.68, 0.95],
                             figsize=(14, 10), save_path=None):
    """
    Plot uncertainty analysis with confidence contours.
    
    Parameters
    ----------
    misfit_grid : ndarray
        3D array of misfit values
    grid_coords : tuple
        (x_grid, y_grid, z_grid) coordinate arrays
    best_location : array-like
        Best-fit location [x, y, z]
    confidence_levels : list
        Confidence levels for contours
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    x_grid, y_grid, z_grid = grid_coords
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Find depth index closest to best location
    z_idx = np.argmin(np.abs(z_grid - best_location[2]))
    
    # Horizontal slice at best depth
    ax = axes[0, 0]
    X, Y = np.meshgrid(x_grid, y_grid)
    misfit_slice = misfit_grid[:, :, z_idx]
    
    # Convert misfit to chi-squared for confidence levels
    min_misfit = misfit_slice.min()
    chi2 = (misfit_slice / min_misfit)**2
    
    levels = [min_misfit * np.sqrt(1 + level) for level in confidence_levels]
    cs = ax.contourf(X, Y, misfit_slice.T, levels=20, cmap='viridis_r', alpha=0.7)
    ax.contour(X, Y, misfit_slice.T, levels=levels, colors=['red', 'orange'],
              linewidths=2)
    ax.plot(best_location[0], best_location[1], 'w*', markersize=15,
           markeredgecolor='black', markeredgewidth=1)
    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Y (km)', fontsize=11)
    ax.set_title(f'Horizontal Uncertainty (z={z_grid[z_idx]:.2f} km)',
                fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(cs, ax=ax, label='RMS Misfit (s)')
    
    # X-Z cross-section
    ax = axes[0, 1]
    y_idx = np.argmin(np.abs(y_grid - best_location[1]))
    X_xz, Z_xz = np.meshgrid(x_grid, z_grid)
    misfit_xz = misfit_grid[:, y_idx, :]
    
    cs = ax.contourf(X_xz, Z_xz, misfit_xz.T, levels=20, cmap='viridis_r', alpha=0.7)
    ax.plot(best_location[0], best_location[2], 'w*', markersize=15,
           markeredgecolor='black', markeredgewidth=1)
    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Depth (km)', fontsize=11)
    ax.set_title(f'X-Z Cross-Section (y={y_grid[y_idx]:.2f} km)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.colorbar(cs, ax=ax, label='RMS Misfit (s)')
    
    # Y-Z cross-section
    ax = axes[1, 0]
    x_idx = np.argmin(np.abs(x_grid - best_location[0]))
    Y_yz, Z_yz = np.meshgrid(y_grid, z_grid)
    misfit_yz = misfit_grid[x_idx, :, :]
    
    cs = ax.contourf(Y_yz, Z_yz, misfit_yz.T, levels=20, cmap='viridis_r', alpha=0.7)
    ax.plot(best_location[1], best_location[2], 'w*', markersize=15,
           markeredgecolor='black', markeredgewidth=1)
    ax.set_xlabel('Y (km)', fontsize=11)
    ax.set_ylabel('Depth (km)', fontsize=11)
    ax.set_title(f'Y-Z Cross-Section (x={x_grid[x_idx]:.2f} km)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.colorbar(cs, ax=ax, label='RMS Misfit (s)')
    
    # 1D misfit profiles
    ax = axes[1, 1]
    
    # Extract 1D profiles through best location
    x_profile = misfit_grid[:, y_idx, z_idx]
    y_profile = misfit_grid[x_idx, :, z_idx]
    z_profile = misfit_grid[x_idx, y_idx, :]
    
    ax.plot(x_grid, x_profile, 'r-', linewidth=2, label='X direction')
    ax.axvline(best_location[0], color='r', linestyle='--', alpha=0.5)
    
    ax2 = ax.twiny()
    ax2.plot(y_grid, y_profile, 'b-', linewidth=2, label='Y direction')
    ax2.axvline(best_location[1], color='b', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('X (km)', fontsize=11, color='r')
    ax2.set_xlabel('Y (km)', fontsize=11, color='b')
    ax.set_ylabel('RMS Misfit (s)', fontsize=11)
    ax.set_title('1D Misfit Profiles', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Location Uncertainty Analysis', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty analysis saved to: {save_path}")
    
    return fig, axes


def plot_anisotropy_field(velocity_model, figsize=(18, 12), save_path=None):
    """
    Plot spatial variation of anisotropy parameters.
    
    Shows how anisotropy strength (percent) and fast direction (phi) vary
    across the model domain, highlighting enhanced anisotropy at ring faults
    and spatially-varying fast directions.
    
    Parameters
    ----------
    velocity_model : AxialVelocityModel
        Velocity model with anisotropic components created
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    vm = velocity_model
    
    if vm.percent_anisotropy is None:
        raise ValueError("Anisotropic model not created. Call create_anisotropic_model() first.")
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    # Plot percent anisotropy at different depths
    depths_to_plot = [0.0, 1.0, 2.0]
    
    for idx, depth in enumerate(depths_to_plot):
        ax = axes[0, idx]
        aniso_slice, (x_coords, y_coords) = vm.get_anisotropy_slice('z', depth, 'percent')
        
        im = ax.contourf(x_coords, y_coords, aniso_slice.T, levels=15,
                        cmap='YlOrRd', vmin=2.0, vmax=7.0)
        ax.contour(x_coords, y_coords, aniso_slice.T, levels=10,
                  colors='k', linewidths=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Anisotropy (%)')
        
        # Add caldera outline
        theta = np.linspace(0, 2*np.pi, 100)
        caldera_x = 1.5 * np.cos(theta)
        caldera_y = 4.0 * np.sin(theta)
        ax.plot(caldera_x, caldera_y, 'k--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('East (km)', fontsize=11)
        ax.set_ylabel('North (km)', fontsize=11)
        ax.set_title(f'Percent Anisotropy at z={depth:.1f} km', fontweight='bold')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    # Plot fast direction (phi) at different depths with orientation bars
    for idx, depth in enumerate(depths_to_plot):
        ax = axes[1, idx]
        phi_slice, (x_coords, y_coords) = vm.get_anisotropy_slice('z', depth, 'phi')
        
        # Plot as colored contours
        im = ax.contourf(x_coords, y_coords, phi_slice.T, levels=15,
                        cmap='twilight', vmin=0, vmax=180)
        plt.colorbar(im, ax=ax, label='φ (degrees)')
        
        # Overlay orientation bars showing fast direction
        # Sample every Nth point for clarity
        skip = 8
        X_sample = x_coords[::skip]
        Y_sample = y_coords[::skip]
        
        for i, x in enumerate(X_sample):
            for j, y in enumerate(Y_sample):
                # Get phi at this location
                phi = phi_slice[i*skip, j*skip]
                # Convert to radians and draw short line segment
                angle_rad = np.radians(phi)
                length = 0.4  # km
                dx = length * np.cos(angle_rad)
                dy = length * np.sin(angle_rad)
                ax.plot([x-dx/2, x+dx/2], [y-dy/2, y+dy/2], 
                       'k-', linewidth=1.5, alpha=0.6)
        
        # Add caldera outline
        ax.plot(caldera_x, caldera_y, 'k--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('East (km)', fontsize=11)
        ax.set_ylabel('North (km)', fontsize=11)
        ax.set_title(f'Fast Direction (φ) at z={depth:.1f} km', fontweight='bold')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    # Plot velocity difference (fast - slow) at different depths
    for idx, depth in enumerate(depths_to_plot):
        ax = axes[2, idx]
        
        # Get velocity slices
        z_idx = np.argmin(np.abs(vm.z - depth))
        v_fast_slice = vm.velocity_fast[:, :, z_idx]
        v_slow_slice = vm.velocity_slow[:, :, z_idx]
        v_diff = v_fast_slice - v_slow_slice
        
        im = ax.contourf(vm.x, vm.y, v_diff.T, levels=15,
                        cmap='RdBu_r', vmin=-0.15, vmax=0.15)
        ax.contour(vm.x, vm.y, v_diff.T, levels=10,
                  colors='k', linewidths=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax, label='ΔV (km/s)')
        
        # Add caldera outline
        theta = np.linspace(0, 2*np.pi, 100)
        caldera_x = 1.5 * np.cos(theta)
        caldera_y = 4.0 * np.sin(theta)
        ax.plot(caldera_x, caldera_y, 'k--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('East (km)', fontsize=11)
        ax.set_ylabel('North (km)', fontsize=11)
        ax.set_title(f'Velocity Difference (V_fast - V_slow) at z={depth:.1f} km',
                    fontweight='bold')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Spatial Distribution of Crustal Anisotropy\n(Strength and Orientation)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Anisotropy field plot saved to: {save_path}")
    
    return fig, axes


def plot_split_ray_comparison(ray_fast, ray_slow, velocity_model, source_loc,
                              receiver_loc, path_difference=None,
                              figsize=(12, 6), save_path=None):
    """
    Plot comparison of fast and slow ray paths through anisotropic medium.
    
    Shows how rays diverge due to velocity differences, demonstrating the
    physical mechanism of shear-wave splitting. Enhanced visualization with
    separation markers and optimized plotting order.
    
    Parameters
    ----------
    ray_fast : ndarray
        Ray path for fast component, shape (npts, 3)
    ray_slow : ndarray
        Ray path for slow component, shape (npts, 3)
    velocity_model : AxialVelocityModel
        Velocity model for background
    source_loc : array-like
        Source location [x, y, z]
    receiver_loc : array-like
        Receiver location [x, y, z]
    path_difference : dict, optional
        Dictionary with path statistics from compute_anisotropic_ray_paths
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    vm = velocity_model
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Compute separation for finding max separation point
    separations = np.linalg.norm(ray_fast - ray_slow, axis=1)
    max_sep_idx = np.argmax(separations)
    max_sep_point_fast = ray_fast[max_sep_idx]
    max_sep_point_slow = ray_slow[max_sep_idx]
    
    # Calculate smart axis limits (tight bounding box with 10% padding)
    all_points = np.vstack([ray_fast, ray_slow])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_pad = max(0.5, x_range * 0.1)
    y_pad = max(0.5, y_range * 0.1)
    z_pad = max(0.2, z_range * 0.1)
    
    # Panel 1: Map view (x-y projection)
    ax = axes[0]
    
    # Plot velocity background
    vel_slice, (x_coords, y_coords) = vm.get_slice('z', 0.5)
    im = ax.contourf(x_coords, y_coords, vel_slice.T, levels=15,
                    cmap='viridis', alpha=0.2, vmin=1.5, vmax=4.5)
    
    # Plot rays - SLOW FIRST (thicker, underneath), FAST SECOND (thinner, on top)
    ax.plot(ray_slow[:, 0], ray_slow[:, 1], 'b-', linewidth=4.0,
           label='Slow ray', alpha=0.6, zorder=7)
    ax.plot(ray_fast[:, 0], ray_fast[:, 1], 'r-', linewidth=2.5,
           label='Fast ray', alpha=1.0, zorder=8)
    
    # Add separation markers at evenly-spaced points
    n_markers = 8
    marker_indices = np.linspace(0, len(ray_fast)-1, n_markers, dtype=int)
    for idx in marker_indices:
        # Draw line connecting fast and slow points
        ax.plot([ray_fast[idx, 0], ray_slow[idx, 0]], 
               [ray_fast[idx, 1], ray_slow[idx, 1]],
               'gray', linewidth=1.0, alpha=0.4, zorder=6)
        # Mark points
        ax.plot(ray_fast[idx, 0], ray_fast[idx, 1], 'ro', markersize=4, zorder=9)
        ax.plot(ray_slow[idx, 0], ray_slow[idx, 1], 'bo', markersize=4, zorder=9)
    
    # Highlight max separation point
    ax.plot([max_sep_point_fast[0], max_sep_point_slow[0]],
           [max_sep_point_fast[1], max_sep_point_slow[1]],
           'orange', linewidth=2.5, alpha=0.8, zorder=9, label='Max separation')
    
    # Mark source and receiver
    ax.plot(source_loc[0], source_loc[1], 'k*', markersize=20,
           markeredgecolor='white', markeredgewidth=1.5, label='Source', zorder=10)
    ax.plot(receiver_loc[0], receiver_loc[1], 'k^', markersize=15,
           markeredgecolor='white', markeredgewidth=1.5, label='Receiver', zorder=10)
    
    # Set fixed axis limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    ax.set_xlabel('East (km)', fontsize=12)
    ax.set_ylabel('North (km)', fontsize=12)
    ax.set_title('Map View (x-y)', fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Panel 2: Cross-section view (x-z projection)
    ax = axes[1]
    
    # Plot velocity background
    y_slice = (source_loc[1] + receiver_loc[1]) / 2
    vel_xz, (x_coords, z_coords) = vm.get_slice('y', y_slice)
    im = ax.contourf(x_coords, z_coords, vel_xz.T, levels=15,
                    cmap='viridis', alpha=0.2, vmin=1.5, vmax=4.5)
    
    # Plot rays - SLOW FIRST (thicker, underneath), FAST SECOND (thinner, on top)
    ax.plot(ray_slow[:, 0], ray_slow[:, 2], 'b-', linewidth=4.0,
           label='Slow ray', alpha=0.6, zorder=7)
    ax.plot(ray_fast[:, 0], ray_fast[:, 2], 'r-', linewidth=2.5,
           label='Fast ray', alpha=1.0, zorder=8)
    
    # Add separation markers
    for idx in marker_indices:
        ax.plot([ray_fast[idx, 0], ray_slow[idx, 0]], 
               [ray_fast[idx, 2], ray_slow[idx, 2]],
               'gray', linewidth=1.0, alpha=0.4, zorder=6)
        ax.plot(ray_fast[idx, 0], ray_fast[idx, 2], 'ro', markersize=4, zorder=9)
        ax.plot(ray_slow[idx, 0], ray_slow[idx, 2], 'bo', markersize=4, zorder=9)
    
    # Highlight max separation point
    ax.plot([max_sep_point_fast[0], max_sep_point_slow[0]],
           [max_sep_point_fast[2], max_sep_point_slow[2]],
           'orange', linewidth=2.5, alpha=0.8, zorder=9, label='Max separation')
    
    # Mark source and receiver
    ax.plot(source_loc[0], source_loc[2], 'k*', markersize=20,
           markeredgecolor='white', markeredgewidth=1.5, label='Source', zorder=10)
    ax.plot(receiver_loc[0], receiver_loc[2], 'k^', markersize=15,
           markeredgecolor='white', markeredgewidth=1.5, label='Receiver', zorder=10)
    
    # Smart axis limits
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(z_min - z_pad, z_max + z_pad)
    
    ax.set_xlabel('East (km)', fontsize=12)
    ax.set_ylabel('Depth (km)', fontsize=12)
    ax.set_title('Cross-Section (x-z)', fontweight='bold', fontsize=13)
    ax.invert_yaxis()
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    
    # Add text box with statistics
    if path_difference is not None:
        stats_text = (f"Max separation: {path_difference['max_separation']*1000:.1f} m\n"
                     f"Mean separation: {path_difference['mean_separation']*1000:.1f} m\n"
                     f"Travel time (fast): {path_difference['travel_time_fast']:.3f} s\n"
                     f"Travel time (slow): {path_difference['travel_time_slow']:.3f} s\n"
                     f"Predicted δt: {path_difference['predicted_dt']*1000:.1f} ms")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Ray Paths Through Anisotropic Medium: Physical Origin of Splitting',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Split ray comparison saved to: {save_path}")
    
    return fig, axes


def plot_measured_vs_true_anisotropy(splitting_results, pair_info, velocity_model=None,
                                    eq_locations=None, true_percent_aniso=4.0, 
                                    figsize=(14, 10), save_path=None):
    """
    Compare measured splitting parameters against known input anisotropy.
    
    Validates that splitting analysis correctly recovers the anisotropy
    parameters built into the velocity model. Now handles spatially-varying
    fast directions by computing expected φ for each ray path.
    
    Parameters
    ----------
    splitting_results : list
        List of splitting result dictionaries from splitting analysis
    pair_info : list
        List of dictionaries with event/station pair information
    velocity_model : AxialVelocityModel, optional
        Velocity model to extract spatially-varying φ
    eq_locations : ndarray, optional
        Earthquake locations to compute path-averaged φ
    true_percent_aniso : float
        True percent anisotropy built into model
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract measurements
    measured_phi = np.array([r['phi'] for r in splitting_results])
    measured_dt = np.array([r['dt'] for r in splitting_results])
    qualities = np.array([r['quality'] for r in splitting_results])
    distances = np.array([p['distance'] for p in pair_info])
    
    # Compute expected φ for each ray path (if velocity model provided)
    expected_phi = np.zeros(len(splitting_results))
    if velocity_model is not None and eq_locations is not None:
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator for phi_fast field
        phi_interpolator = RegularGridInterpolator(
            (velocity_model.x, velocity_model.y, velocity_model.z),
            velocity_model.phi_fast,
            method='linear',
            bounds_error=False,
            fill_value=37.0  # Default to regional stress if out of bounds
        )
        
        # For each event-station pair, compute path-averaged φ
        for idx in range(len(splitting_results)):
            event_idx = pair_info[idx]['event_idx']
            eq_loc = eq_locations[event_idx]
            
            # Sample along straight-line path (approximation)
            # In reality, should follow ray path, but straight line is reasonable
            sta_loc = np.array([pair_info[idx].get('station_x', 0), 
                               pair_info[idx].get('station_y', 0),
                               pair_info[idx].get('station_z', 0)])
            
            # If station location not in pair_info, skip
            if np.all(sta_loc == 0):
                expected_phi[idx] = 37.0  # Default
                continue
            
            # Sample 20 points along path
            t = np.linspace(0, 1, 20)
            path_points = np.outer(1-t, eq_loc) + np.outer(t, sta_loc)
            
            # Get φ at each point
            phi_along_path = phi_interpolator(path_points)
            
            # Average (using circular mean for angles)
            phi_rad = np.radians(phi_along_path * 2)  # Double for 0-180 range
            mean_x = np.mean(np.cos(phi_rad))
            mean_y = np.mean(np.sin(phi_rad))
            expected_phi[idx] = np.degrees(np.arctan2(mean_y, mean_x)) / 2.0
            
            # Normalize to 0-180
            while expected_phi[idx] < 0:
                expected_phi[idx] += 180.0
            while expected_phi[idx] >= 180:
                expected_phi[idx] -= 180.0
    else:
        # Use constant value if no velocity model
        expected_phi[:] = 37.0
    
    # Filter by quality
    good_quality = qualities > 0.5
    
    # Panel 1: Fast direction distribution and comparison
    ax = axes[0, 0]
    ax.hist(measured_phi[good_quality], bins=20, alpha=0.7, color='blue',
           edgecolor='black', label=f'Measured (n={np.sum(good_quality)})')
    if velocity_model is not None:
        ax.hist(expected_phi[good_quality], bins=20, alpha=0.5, color='red',
               edgecolor='black', label='Expected (path-averaged)')
    ax.axvline(np.mean(measured_phi[good_quality]), color='blue', linewidth=2,
              linestyle='--', label=f'Mean measured = {np.mean(measured_phi[good_quality]):.1f}°')
    if velocity_model is not None:
        ax.axvline(np.mean(expected_phi[good_quality]), color='red', linewidth=2,
                  linestyle='--', label=f'Mean expected = {np.mean(expected_phi[good_quality]):.1f}°')
    ax.set_xlabel('Fast Direction φ (degrees)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Fast Direction Distribution', fontweight='bold', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Delay time vs distance
    ax = axes[0, 1]
    scatter = ax.scatter(distances[good_quality], measured_dt[good_quality] * 1000,
                        c=qualities[good_quality], cmap='viridis', s=80,
                        edgecolors='black', linewidth=0.5, alpha=0.7)
    
    # Add expected relationship for true anisotropy
    # δt ≈ (percent_aniso / 100) * (distance / v_avg)
    v_avg = 3.5  # km/s average S-wave velocity
    expected_dt = (true_percent_aniso / 100) * (distances / v_avg) * 1000  # ms
    ax.plot(distances, expected_dt, 'r--', linewidth=2, label='Expected (4% aniso)')
    
    ax.set_xlabel('Ray Path Distance (km)', fontsize=12)
    ax.set_ylabel('Delay Time δt (ms)', fontsize=12)
    ax.set_title('Delay Time vs Distance', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Quality')
    
    # Panel 3: Measured vs Expected φ scatter
    ax = axes[1, 0]
    
    if velocity_model is not None:
        # Scatter plot: measured vs expected
        scatter = ax.scatter(expected_phi[good_quality], measured_phi[good_quality],
                           c=qualities[good_quality], cmap='viridis', s=80,
                           edgecolors='black', linewidth=0.5, alpha=0.7)
        
        # 1:1 line
        phi_range = [0, 180]
        ax.plot(phi_range, phi_range, 'r--', linewidth=2, label='1:1 line')
        
        # Compute circular correlation
        phi_error = np.abs(measured_phi[good_quality] - expected_phi[good_quality])
        phi_error = np.minimum(phi_error, 180 - phi_error)
        mean_error = np.mean(phi_error)
        
        ax.set_xlabel('Expected φ (path-averaged, degrees)', fontsize=12)
        ax.set_ylabel('Measured φ (degrees)', fontsize=12)
        ax.set_title(f'Measured vs Expected φ\n(Mean error: {mean_error:.1f}°)', 
                    fontweight='bold', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 180)
        plt.colorbar(scatter, ax=ax, label='Quality')
    else:
        # Fallback: just show quality vs measurement
        ax.scatter(qualities, measured_phi, s=80, alpha=0.7, color='purple',
                  edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Measurement Quality', fontsize=12)
        ax.set_ylabel('Measured φ (degrees)', fontsize=12)
        ax.set_title('Quality vs Fast Direction', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
    
    # Panel 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute statistics
    phi_mean = np.mean(measured_phi[good_quality])
    phi_std = np.std(measured_phi[good_quality])
    
    dt_mean = np.mean(measured_dt[good_quality]) * 1000
    dt_std = np.std(measured_dt[good_quality]) * 1000
    
    # Expected mean delay time
    mean_distance = np.mean(distances[good_quality])
    expected_dt_mean = (true_percent_aniso / 100) * (mean_distance / v_avg) * 1000
    
    # Compute phi agreement if we have expected values
    if velocity_model is not None:
        expected_phi_mean = np.mean(expected_phi[good_quality])
        phi_error_mean = np.mean(np.minimum(np.abs(measured_phi[good_quality] - expected_phi[good_quality]),
                                           180 - np.abs(measured_phi[good_quality] - expected_phi[good_quality])))
        phi_agreement = f"Mean error: {phi_error_mean:.1f}°"
        phi_status = '✓ Yes' if phi_error_mean < 15 else '✗ No (>{:.0f}°)'.format(phi_error_mean)
    else:
        expected_phi_mean = 37.0
        phi_agreement = "N/A (uniform φ assumed)"
        phi_status = "N/A"
    
    summary_text = f"""
    INPUT ANISOTROPY PARAMETERS:
    ─────────────────────────────────
    Expected φ (path-averaged):   {expected_phi_mean:.1f}° ± varies spatially
    True percent anisotropy:      {true_percent_aniso:.1f}% (base, varies 3-6%)
    
    MEASURED SPLITTING PARAMETERS:
    ─────────────────────────────────
    High quality measurements:    {np.sum(good_quality)}/{len(splitting_results)}
    
    Fast Direction (φ):
      Mean ± Std:                 {phi_mean:.1f}° ± {phi_std:.1f}°
      Agreement with expected:    {phi_agreement}
    
    Delay Time (δt):
      Mean ± Std:                 {dt_mean:.1f} ± {dt_std:.1f} ms
      Expected from ~{true_percent_aniso:.0f}% aniso: {expected_dt_mean:.1f} ms
    
    VALIDATION:
    ─────────────────────────────────
    φ matches path-averaged:      {phi_status}
    δt consistent with input:     {'✓ Yes' if abs(dt_mean - expected_dt_mean) < 25 else '✗ No'}
    
    Mean ray distance:            {mean_distance:.2f} km
    Mean quality factor:          {np.mean(qualities[good_quality]):.3f}
    
    NOTE: φ varies spatially (tangential at ring faults,
          radial near magma, regional elsewhere)
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Validation: Measured vs True Anisotropy Parameters',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Validation plot saved to: {save_path}")
    
    return fig, axes
