#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stub module to replace missing 'general' module dependencies

This module provides minimal implementations of functions from the 'general' module
that are required by shearwavesplit.py to load pickled SWSobs objects.

The pickled objects were created with the original 'general' module, but we can
work around this by providing stub implementations that allow unpickling to succeed.

Usage:
------
import sys
import general_stub
sys.modules['general'] = general_stub.general
sys.modules['general.plotwaveform'] = general_stub.general.plotwaveform
sys.modules['general.util'] = general_stub.general.util
sys.modules['general.projection'] = general_stub.general.projection

Then import shearwavesplit as normal.
"""

import numpy as np
import types

# Create general module and submodules
general = types.ModuleType('general')
general.plotwaveform = types.ModuleType('general.plotwaveform')
general.util = types.ModuleType('general.util')
general.projection = types.ModuleType('general.projection')


def cov_eig(data_array):
    """
    Compute eigenvalues of covariance matrix
    
    Parameters:
    -----------
    data_array : np.ndarray
        N x 2 array of data (typically horizontal seismic components)
    
    Returns:
    --------
    list
        Sorted eigenvalues [lambda1, lambda2] where lambda1 >= lambda2
    """
    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_array.T)
    
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    
    # Return sorted (largest first)
    return sorted(eigenvalues, reverse=True)


def smooth_curve(data, window_len=11, window='hanning'):
    """
    Smooth data using a window function
    
    This is a stub implementation - just returns the original data.
    Only needed if you're doing new analysis, not for loading pickled results.
    """
    if window_len < 3:
        return data
    
    if len(data) < window_len:
        return data
    
    # Simple moving average for stub
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int((window_len-1)/2):int(-(window_len-1)/2)]


def ll2xy(lon_list, lat_list, ref_lon, ref_lat):
    """
    Convert longitude/latitude to local x,y coordinates
    
    This is a stub implementation using simple approximation.
    Only needed if you're computing back-azimuths, not for loading pickled results.
    
    Parameters:
    -----------
    lon_list : list
        Longitudes in degrees
    lat_list : list
        Latitudes in degrees
    ref_lon : float
        Reference longitude
    ref_lat : float
        Reference latitude
    
    Returns:
    --------
    tuple
        ([x_coords], [y_coords]) in km
    """
    R_EARTH = 6371.0  # Earth radius in km
    
    x_coords = []
    y_coords = []
    
    for lon, lat in zip(lon_list, lat_list):
        # Simple equirectangular projection
        x = R_EARTH * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
        y = R_EARTH * np.radians(lat - ref_lat)
        x_coords.append(x)
        y_coords.append(y)
    
    return (x_coords, y_coords)


# Add functions to the modules
general.plotwaveform.cov_eig = cov_eig
general.util.smooth_curve = smooth_curve
general.projection.ll2xy = ll2xy


def register_modules():
    """
    Register stub modules in sys.modules
    
    Call this function before importing shearwavesplit to ensure
    the stub modules are used instead of trying to import the missing
    'general' module.
    """
    import sys
    sys.modules['general'] = general
    sys.modules['general.plotwaveform'] = general.plotwaveform
    sys.modules['general.util'] = general.util
    sys.modules['general.projection'] = general.projection
    print("✓ Stub modules registered for 'general' dependencies")


if __name__ == "__main__":
    # Test the stub module
    print("Testing general_stub module...")
    
    # Test cov_eig
    test_data = np.random.randn(100, 2)
    eigs = cov_eig(test_data)
    print(f"✓ cov_eig works: eigenvalues = {eigs}")
    
    # Test smooth_curve
    test_signal = np.sin(np.linspace(0, 4*np.pi, 100))
    smoothed = smooth_curve(test_signal)
    print(f"✓ smooth_curve works: input length = {len(test_signal)}, output length = {len(smoothed)}")
    
    # Test ll2xy
    lons = [-130.0, -129.5]
    lats = [45.9, 46.0]
    x, y = ll2xy(lons, lats, -130.0, 45.9)
    print(f"✓ ll2xy works: x = {x}, y = {y}")
    
    print("\nAll stub functions working correctly!")
