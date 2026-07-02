#!/usr/bin/env python3
"""
Seismic Geometry Calculations for Shear-Wave Splitting Analysis

This module provides functions to calculate geometric parameters between 
earthquake sources and seismic stations, including back-azimuth and 
angle of incidence calculations needed for shear-wave splitting analysis.

Created for: Axial Seamount Shear-Wave Splitting Project
Author: Michael Hemmett
Date: October 2025
"""

import numpy as np
import pandas as pd


def calculate_epicentral_distance_km(eq_lat, eq_lon, sta_lat, sta_lon):
    """
    Calculate epicentral distance using haversine formula
    
    Parameters:
    -----------
    eq_lat, eq_lon : float
        Earthquake latitude and longitude in degrees
    sta_lat, sta_lon : float
        Station latitude and longitude in degrees
        
    Returns:
    --------
    float
        Epicentral distance in kilometers
    """
    # Convert to radians
    eq_lat_rad = np.radians(eq_lat)
    eq_lon_rad = np.radians(eq_lon)
    sta_lat_rad = np.radians(sta_lat)
    sta_lon_rad = np.radians(sta_lon)
    
    # Haversine formula
    dlat = eq_lat_rad - sta_lat_rad
    dlon = eq_lon_rad - sta_lon_rad
    a = np.sin(dlat/2)**2 + np.cos(sta_lat_rad) * np.cos(eq_lat_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km
    R = 6371.0
    distance_km = R * c
    
    return distance_km


def calculate_back_azimuth(eq_lat, eq_lon, sta_lat, sta_lon):
    """
    Calculate back-azimuth from station to earthquake (0-360°)
    
    Parameters:
    -----------
    eq_lat, eq_lon : float
        Earthquake latitude and longitude in degrees
    sta_lat, sta_lon : float
        Station latitude and longitude in degrees
        
    Returns:
    --------
    float
        Back-azimuth in degrees (0-360°)
    """
    # Convert to radians
    eq_lat_rad = np.radians(eq_lat)
    eq_lon_rad = np.radians(eq_lon)
    sta_lat_rad = np.radians(sta_lat)
    sta_lon_rad = np.radians(sta_lon)
    
    # Calculate back-azimuth
    dlon = eq_lon_rad - sta_lon_rad
    y = np.sin(dlon) * np.cos(eq_lat_rad)
    x = (np.cos(sta_lat_rad) * np.sin(eq_lat_rad) - 
         np.sin(sta_lat_rad) * np.cos(eq_lat_rad) * np.cos(dlon))
    
    back_az = np.degrees(np.arctan2(y, x))
    
    # Normalize to 0-360°
    back_az = (back_az + 360) % 360
    
    return back_az


def calculate_angle_of_incidence(epicentral_dist_km, eq_depth_km, sta_elevation_km=0):
    """
    Calculate angle of incidence from vertical using simple geometric approach
    
    Note: This assumes straight ray path and doesn't account for Earth's
    velocity structure. For more accurate results, use ray tracing software.
    
    Parameters:
    -----------
    epicentral_dist_km : float
        Epicentral distance in kilometers
    eq_depth_km : float
        Earthquake depth in kilometers
    sta_elevation_km : float, optional
        Station elevation in kilometers (negative for below sea level)
        Default is 0
        
    Returns:
    --------
    float
        Angle of incidence from vertical in degrees
    """
    # Convert elevation from meters to km if needed
    if abs(sta_elevation_km) > 100:  # Likely in meters
        sta_elevation_km = sta_elevation_km / 1000.0
    
    # Adjust depth for station elevation (depth below station)
    effective_depth = eq_depth_km - sta_elevation_km
    
    # Angle from vertical = arctan(horizontal_distance / depth)
    angle_of_incidence = np.degrees(np.arctan2(epicentral_dist_km, effective_depth))
    
    return angle_of_incidence


def get_station_earthquake_geometry(station_id, eq_lat, eq_lon, eq_depth_km, station_data):
    """
    Get back-azimuth and angle of incidence for a specific station-earthquake pair
    
    Parameters:
    -----------
    station_id : str
        Station identifier (e.g., 'AXAS2')
    eq_lat, eq_lon : float
        Earthquake latitude and longitude in degrees
    eq_depth_km : float
        Earthquake depth in kilometers
    station_data : pandas.DataFrame
        Station coordinates dataframe with columns:
        ['station_id', 'location_name', 'lat', 'lon', 'elev']
        
    Returns:
    --------
    dict or None
        Dictionary with geometry parameters:
        {
            'station_id': str,
            'station_name': str,
            'epicentral_distance_km': float,
            'back_azimuth_deg': float,
            'angle_of_incidence_deg': float,
            'quality': str ('GOOD', 'FAIR', or 'POOR')
        }
        Returns None if station not found.
    """
    # Find station in dataframe
    station_row = station_data[station_data['station_id'] == station_id]
    
    if len(station_row) == 0:
        print(f"Warning: Station {station_id} not found in station_data")
        return None
    
    station_info = station_row.iloc[0]
    sta_lat = station_info['lat']
    sta_lon = station_info['lon'] 
    sta_elev = station_info['elev']
    
    # Calculate geometry
    epicentral_dist = calculate_epicentral_distance_km(eq_lat, eq_lon, sta_lat, sta_lon)
    back_azimuth = calculate_back_azimuth(eq_lat, eq_lon, sta_lat, sta_lon)
    angle_of_incidence = calculate_angle_of_incidence(epicentral_dist, eq_depth_km, sta_elev)
    
    # Quality assessment for shear-wave splitting
    if angle_of_incidence <= 35:
        quality = 'GOOD'
    elif angle_of_incidence <= 45:
        quality = 'FAIR'
    else:
        quality = 'POOR'
    
    return {
        'station_id': station_id,
        'station_name': station_info['location_name'],
        'epicentral_distance_km': epicentral_dist,
        'back_azimuth_deg': back_azimuth,
        'angle_of_incidence_deg': angle_of_incidence,
        'quality': quality
    }


def calculate_all_station_geometries(eq_lat, eq_lon, eq_depth_km, station_data):
    """
    Calculate geometry for all stations relative to one earthquake
    
    Parameters:
    -----------
    eq_lat, eq_lon : float
        Earthquake latitude and longitude in degrees
    eq_depth_km : float
        Earthquake depth in kilometers
    station_data : pandas.DataFrame
        Station coordinates dataframe
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with geometry calculations for all stations
    """
    results = []
    
    for index, station in station_data.iterrows():
        geometry = get_station_earthquake_geometry(
            station['station_id'], eq_lat, eq_lon, eq_depth_km, station_data
        )
        if geometry:
            results.append(geometry)
    
    return pd.DataFrame(results)


def print_geometry_summary(eq_lat, eq_lon, eq_depth_km, station_data):
    """
    Print a summary of geometry calculations for all stations
    
    Parameters:
    -----------
    eq_lat, eq_lon : float
        Earthquake latitude and longitude in degrees
    eq_depth_km : float
        Earthquake depth in kilometers
    station_data : pandas.DataFrame
        Station coordinates dataframe
    """
    print(f"Station-Earthquake Geometry Calculations")
    print(f"Earthquake: Lat={eq_lat:.4f}°, Lon={eq_lon:.4f}°, Depth={eq_depth_km:.1f}km")
    print("=" * 70)
    
    # Calculate for all stations in the dataset
    for index, station in station_data.iterrows():
        station_id = station['station_id']
        sta_lat = station['lat']
        sta_lon = station['lon']
        sta_elev = station['elev']
        
        # Calculate distances and angles
        epicentral_dist = calculate_epicentral_distance_km(eq_lat, eq_lon, sta_lat, sta_lon)
        back_azimuth = calculate_back_azimuth(eq_lat, eq_lon, sta_lat, sta_lon)
        angle_of_incidence = calculate_angle_of_incidence(epicentral_dist, eq_depth_km, sta_elev)
        
        print(f"\n{station_id} ({station['location_name']}):")
        print(f"  Station coords: {sta_lat:.4f}°, {sta_lon:.4f}°, {sta_elev:.1f}m")
        print(f"  Epicentral distance: {epicentral_dist:.2f} km")
        print(f"  Back-azimuth: {back_azimuth:.1f}°")
        print(f"  Angle of incidence: {angle_of_incidence:.1f}° from vertical")
        
        # Quality assessment for shear-wave splitting
        if angle_of_incidence <= 35:
            quality = "GOOD (≤35°)"
        elif angle_of_incidence <= 45:
            quality = "FAIR (35-45°)"
        else:
            quality = "POOR (>45°)"
        
        print(f"  Quality for splitting: {quality}")
    
    print("\n" + "=" * 70)
    print("Notes:")
    print("- Back-azimuth: Direction FROM station TO earthquake")
    print("- Angle of incidence: Angle from vertical (0° = straight down)")
    print("- For shear-wave splitting: prefer angles ≤35° from vertical")
    print("- Elevation: Negative values = below sea level (ocean bottom seismometers)")


def load_station_data(csv_file_path):
    """
    Load station data from CSV file with proper column names
    
    Parameters:
    -----------
    csv_file_path : str
        Path to CSV file containing station data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns ['station_id', 'location_name', 'lat', 'lon', 'elev']
    """
    station_data = pd.read_csv(csv_file_path)
    station_data.columns = ['station_id', 'location_name', 'lat', 'lon', 'elev']
    return station_data


# Example usage
if __name__ == "__main__":
    # Example: Load station data and calculate geometry
    try:
        # Load station data (adjust path as needed)
        station_data = load_station_data('../data/axial_seamount_stations.csv')
        
        # Example earthquake
        eq_lat = 45.93439
        eq_lon = -130.02092
        eq_depth_km = 1.148
        
        # Print summary for all stations
        print_geometry_summary(eq_lat, eq_lon, eq_depth_km, station_data)
        
        # Example: Get geometry for specific station
        print("\n" + "="*50)
        print("Example: AXAS2 geometry for swspy analysis")
        print("="*50)
        
        axas2_geometry = get_station_earthquake_geometry('AXAS2', eq_lat, eq_lon, eq_depth_km, station_data)
        
        if axas2_geometry:
            print(f"Station: {axas2_geometry['station_id']}")
            print(f"Back-azimuth: {axas2_geometry['back_azimuth_deg']:.1f}°")
            print(f"Angle of incidence: {axas2_geometry['angle_of_incidence_deg']:.1f}°")
            print(f"Quality: {axas2_geometry['quality']}")
            print()
            print("For swspy analysis use:")
            print(f"back_azis_all_stations=[{axas2_geometry['back_azimuth_deg']:.1f}]")
            print(f"receiver_inc_angles_all_stations=[{axas2_geometry['angle_of_incidence_deg']:.1f}]")
        
    except FileNotFoundError:
        print("Station data file not found. Please check the file path.")
        print("Expected: '../data/axial_seamount_stations.csv'")