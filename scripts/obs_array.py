"""
OBS Array Geometry for Axial Seamount

This module manages the Ocean Bottom Seismometer (OBS) array geometry from the
OOI Regional Cabled Array at Axial Seamount, including coordinate transformations
and station information.

Author: Michael Hemmett
Date: December 2025
"""

import numpy as np
import pandas as pd


class OBSArray:
    """
    OBS array geometry and coordinate management for Axial Seamount.
    
    Attributes
    ----------
    stations : dict
        Dictionary of station information with lat, lon, depth
    caldera_center : tuple
        Reference point (lat, lon) for coordinate transformation
    seamount_top_depth : float
        Depth of seamount top below sea level in km
    """
    
    # Real OBS station coordinates from OOI Regional Cabled Array
    STATION_DATA = {
        'AXCC1': {'lat': 45.9553, 'lon': -130.0085, 'depth': 1.516, 'name': 'Central Caldera'},
        'AXEC2': {'lat': 45.9559, 'lon': -129.9736, 'depth': 1.519, 'name': 'Eastern Caldera'},
        'AXID1': {'lat': 45.9343, 'lon': -129.9816, 'depth': 1.522, 'name': 'International District'},
        'AXAS1': {'lat': 45.9369, 'lon': -130.0060, 'depth': 1.520, 'name': 'ASHES Vent Field'}
    }
    
    # Reference coordinates
    CALDERA_CENTER = (45.9600, -130.0000)  # lat, lon
    SEAMOUNT_TOP_DEPTH = 1.4  # km below sea level
    
    # Conversion factors at ~46°N
    LAT_TO_KM = 111.0  # km per degree latitude
    LON_TO_KM = 77.0   # km per degree longitude
    
    def __init__(self, caldera_center=None, seamount_top_depth=None):
        """
        Initialize OBS array with station data.
        
        Parameters
        ----------
        caldera_center : tuple, optional
            (lat, lon) reference point for coordinate transformation
        seamount_top_depth : float, optional
            Depth of seamount top below sea level in km
        """
        self.stations = self.STATION_DATA.copy()
        self.caldera_center = caldera_center or self.CALDERA_CENTER
        self.seamount_top_depth = seamount_top_depth or self.SEAMOUNT_TOP_DEPTH
        
        # Calculate relative coordinates for all stations
        self._compute_relative_coords()
    
    def _compute_relative_coords(self):
        """Compute relative Cartesian coordinates for all stations."""
        caldera_lat, caldera_lon = self.caldera_center
        
        for station_code, info in self.stations.items():
            # Convert lat/lon to relative position in km
            delta_lat = info['lat'] - caldera_lat
            delta_lon = info['lon'] - caldera_lon
            
            x_rel = delta_lon * self.LON_TO_KM  # East-West (positive = East)
            y_rel = delta_lat * self.LAT_TO_KM  # North-South (positive = North)
            z_rel = info['depth'] - self.seamount_top_depth  # Depth relative to seamount top
            
            # Store relative coordinates
            self.stations[station_code]['x'] = x_rel
            self.stations[station_code]['y'] = y_rel
            self.stations[station_code]['z'] = z_rel
    
    def get_station_coords(self, station_code, coord_system='relative'):
        """
        Get coordinates for a specific station.
        
        Parameters
        ----------
        station_code : str
            Station identifier (e.g., 'AXBA1')
        coord_system : str
            'relative' for Cartesian (x, y, z) or 'geographic' for (lat, lon, depth)
        
        Returns
        -------
        coords : tuple
            Station coordinates in requested system
        """
        if station_code not in self.stations:
            raise ValueError(f"Unknown station: {station_code}")
        
        info = self.stations[station_code]
        
        if coord_system == 'relative':
            return (info['x'], info['y'], info['z'])
        elif coord_system == 'geographic':
            return (info['lat'], info['lon'], info['depth'])
        else:
            raise ValueError("coord_system must be 'relative' or 'geographic'")
    
    def get_all_coords(self, coord_system='relative'):
        """
        Get coordinates for all stations.
        
        Parameters
        ----------
        coord_system : str
            'relative' for Cartesian (x, y, z) or 'geographic' for (lat, lon, depth)
        
        Returns
        -------
        coords : ndarray
            Array of shape (n_stations, 3) with coordinates
        station_codes : list
            List of station codes in same order as coords
        """
        station_codes = list(self.stations.keys())
        coords = np.array([self.get_station_coords(code, coord_system) for code in station_codes])
        return coords, station_codes
    
    def get_station_dataframe(self):
        """
        Get all station information as a pandas DataFrame.
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with all station information
        """
        data = []
        for code, info in self.stations.items():
            row = {
                'station': code,
                'name': info['name'],
                'lat': info['lat'],
                'lon': info['lon'],
                'depth_km': info['depth'],
                'x_km': info['x'],
                'y_km': info['y'],
                'z_km': info['z']
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def latlon_to_relative(self, lat, lon, depth=None):
        """
        Convert geographic coordinates to relative Cartesian coordinates.
        
        Parameters
        ----------
        lat, lon : float or array-like
            Geographic coordinates in degrees
        depth : float or array-like, optional
            Depth below sea level in km
        
        Returns
        -------
        x, y, z : float or ndarray
            Relative Cartesian coordinates in km
        """
        caldera_lat, caldera_lon = self.caldera_center
        
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        
        delta_lat = lat - caldera_lat
        delta_lon = lon - caldera_lon
        
        x = delta_lon * self.LON_TO_KM
        y = delta_lat * self.LAT_TO_KM
        
        if depth is not None:
            depth = np.atleast_1d(depth)
            z = depth - self.seamount_top_depth
            return x, y, z
        else:
            return x, y
    
    def relative_to_latlon(self, x, y, z=None):
        """
        Convert relative Cartesian coordinates to geographic coordinates.
        
        Parameters
        ----------
        x, y : float or array-like
            Relative Cartesian coordinates in km
        z : float or array-like, optional
            Relative depth in km
        
        Returns
        -------
        lat, lon, depth : float or ndarray
            Geographic coordinates and depth below sea level
        """
        caldera_lat, caldera_lon = self.caldera_center
        
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        lat = caldera_lat + y / self.LAT_TO_KM
        lon = caldera_lon + x / self.LON_TO_KM
        
        if z is not None:
            z = np.atleast_1d(z)
            depth = z + self.seamount_top_depth
            return lat, lon, depth
        else:
            return lat, lon
    
    def compute_distances(self, x, y, z):
        """
        Compute distances from a point to all stations.
        
        Parameters
        ----------
        x, y, z : float
            Point coordinates in relative Cartesian system (km)
        
        Returns
        -------
        distances : ndarray
            Array of distances to each station in km
        station_codes : list
            List of station codes in same order
        """
        coords, station_codes = self.get_all_coords('relative')
        
        point = np.array([x, y, z])
        distances = np.linalg.norm(coords - point, axis=1)
        
        return distances, station_codes
    
    def print_summary(self):
        """Print summary of OBS array configuration."""
        print("=" * 70)
        print("AXIAL SEAMOUNT OBS ARRAY")
        print("=" * 70)
        print(f"Number of stations: {len(self.stations)}")
        print(f"Caldera center: {self.caldera_center[0]:.4f}°N, {self.caldera_center[1]:.4f}°W")
        print(f"Seamount top depth: {self.seamount_top_depth} km below sea level")
        print(f"\nCoordinate system: x=East, y=North, z=depth relative to seamount top")
        print("\nStation positions:")
        print("-" * 70)
        
        df = self.get_station_dataframe()
        for _, row in df.iterrows():
            print(f"{row['station']:6s} ({row['name']:20s}): "
                  f"({row['x_km']:6.2f}, {row['y_km']:6.2f}, {row['z_km']:6.3f}) km")
        print("=" * 70)
