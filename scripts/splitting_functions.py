"""
Master Shear-Wave Splitting Analysis Functions

This module contains all the functions used in the master shear-wave splitting workflow,
extracted from the noteook for better code organization and reusability.

Author: Michael Hemmett with Claude 4.0
Date: 11-2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import json
from pathlib import Path
from datetime import datetime
from obspy.core.utcdatetime import UTCDateTime
import obspy
from tqdm import tqdm
import swspy
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from scipy.ndimage import zoom, filters
import scipy.ndimage.morphology as morphology
import copy
from scipy import signal
import teanby_clustering as tbc
import sws_methods as swm
import scipy

def cov_eig(data_array):
    """
    Compute covariance matrix eigenvalues and eigenvectors with descending sort.
    
    This function matches the Jurkevics (1988) implementation for polarization analysis.
    Uses np.linalg.eig (not eigh) and sorts eigenvalues in descending order.
    
    Parameters:
    -----------
    data_array : np.ndarray
        Data matrix with shape (n_samples, n_components) where columns are Z, N, E
        
    Returns:
    --------
    tuple
        (eig_vals, eig_vecs) - Eigenvalues and eigenvectors sorted in descending order
    """
    cov_mat = np.cov(np.transpose(data_array))
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    ind_descend = np.argsort(-eig_vals)
    eig_vals = eig_vals[ind_descend]
    eig_vecs = eig_vecs[:, ind_descend]
    
    return eig_vals, eig_vecs


def create_extended_catalog(catalog_df, pre_p_time=1.0, post_s_time=1.0):
    """
    Create extended start and end times for waveform retrieval.
    
    Parameters:
    -----------
    catalog_df : pandas.DataFrame
        Input catalog with datetime column
    pre_p_time : float
        Time before P-pick (seconds) for noise/P-wave analysis
    post_s_time : float
        Time after S-pick (seconds) for S-wave analysis
    
    Returns:
    --------
    pandas.DataFrame
        Catalog with extended starttime and endtime columns
    """
    extended_catalog = catalog_df.copy()
    
    # Convert datetime to UTCDateTime objects
    origin_times = [UTCDateTime(dt) for dt in extended_catalog['datetime']]
    
    # p_time and s_time are already absolute UTCDateTime objects
    p_times = [UTCDateTime(pt) for pt in extended_catalog['p_time']]
    s_times = [UTCDateTime(st) for st in extended_catalog['s_time']]
    
    # Create extended time windows: [P - 1s, S + 1s]
    extended_catalog['starttime'] = [pt - pre_p_time for pt in p_times]
    extended_catalog['endtime'] = [st + post_s_time for st in s_times]
    extended_catalog['origin_utc'] = origin_times
    
    # Add timing parameters for reference
    extended_catalog['pre_p_sec'] = pre_p_time
    extended_catalog['post_s_sec'] = post_s_time
    extended_catalog['total_duration'] = [
        float(endtime - starttime) for starttime, endtime 
        in zip(extended_catalog['starttime'], extended_catalog['endtime'])
    ]
    
    return extended_catalog


def organize_waveform_data(traces_dict, extended_catalog):
    """
    Organize waveform data by event ID and ensure proper timing alignment.
    
    Parameters:
    -----------
    traces_dict : dict
        Dictionary of trace data keyed by event ID
    extended_catalog : pandas.DataFrame
        Catalog with extended timing information
    
    Returns:
    --------
    dict
        Organized waveform data with timing metadata
    """
    organized_data = {}
    
    for idx, row in extended_catalog.iterrows():
        event_id = row['id']
        
        if event_id in traces_dict:
            # Get the traces for this event
            event_traces = traces_dict[event_id]
            
            # Add all metadata needed for QC and analysis
            organized_data[event_id] = {
                'traces': event_traces,
                'origin_time': row['origin_utc'],
                'datetime': row['datetime'],  # For SNR calculations
                'starttime': row['starttime'], 
                'endtime': row['endtime'],
                'pre_p_sec': row['pre_p_sec'],
                'post_s_sec': row['post_s_sec'],
                'magnitude': row['mag'],
                'depth': row['dep'],
                'latitude': row['lat'],
                'longitude': row['lon'],
                'station': row['station'],  # For back-azimuth and station lookups
                'p_arrival_time': row.get('p_time', np.nan),  # For SNR P-window
                's_arrival_time': row.get('s_time', np.nan)   # For SNR S-window
            }
    
    return organized_data


def snr(trace, noise_start_offset=0.5, noise_duration=2.0, 
                  signal_start_offset=0.5, signal_duration=3.0):
    """
    Calculate signal-to-noise ratio for a seismic trace.
    
    Parameters:
    -----------
    trace : obspy.Trace
        Seismic trace data
    noise_start_offset : float
        Seconds from trace start to begin noise window
    noise_duration : float
        Duration of noise window in seconds
    signal_start_offset : float
        Seconds from P-arrival to begin signal window
    signal_duration : float
        Duration of signal window in seconds
    
    Returns:
    --------
    float
        Signal-to-noise ratio
    """
    try:
        sampling_rate = trace.stats.sampling_rate
        
        # Calculate sample indices for noise window
        noise_start_idx = int(noise_start_offset * sampling_rate)
        noise_end_idx = int((noise_start_offset + noise_duration) * sampling_rate)
        
        # Assume P-arrival is at pre_event_time from start
        p_arrival_idx = int(4.0 * sampling_rate)  # 4 seconds pre-event time
        signal_start_idx = p_arrival_idx + int(signal_start_offset * sampling_rate)
        signal_end_idx = signal_start_idx + int(signal_duration * sampling_rate)
        
        # Ensure indices are within trace bounds
        if (noise_end_idx >= len(trace.data) or 
            signal_end_idx >= len(trace.data) or
            noise_start_idx < 0 or signal_start_idx < 0):
            return np.nan
        
        # Calculate RMS amplitudes
        noise_rms = np.sqrt(np.mean(trace.data[noise_start_idx:noise_end_idx]**2))
        signal_rms = np.sqrt(np.mean(trace.data[signal_start_idx:signal_end_idx]**2))
        
        # Avoid division by zero
        if noise_rms == 0:
            return np.inf if signal_rms > 0 else np.nan
        
        return signal_rms / noise_rms
    
    except Exception as e:
        print(f"SNR calculation error: {e}")
        return np.nan

def calculate_incidence_angle(eq_lat, eq_lon, eq_depth, sta_lat, sta_lon):
    """
    Calculate back-azimuth from station to earthquake (0-360°)
    
    Parameters:
    -----------
    eq_lat, eq_lon : float
        Earthquake latitude and longitude in degrees
    eq_depth : float
        Earthquake depth in kilometers
    sta_lat, sta_lon : float
        Station latitude and longitude in degrees
        
    Returns:
    --------
    float
        Incidence angle in degrees (0-90°)
    """
    # Define TauPy model
    model = TauPyModel(model="ak135")

    # Compute epicentral distance in degrees
    dist_m, az, baz = gps2dist_azimuth(eq_lat, eq_lon, sta_lat, sta_lon)
    dist_deg = dist_m / (111319.5)  # approx conversion: 1° ≈ 111 km

    # Get ray paths with incident angle for s phase
    ray_paths = model.get_ray_paths(source_depth_in_km=eq_depth,
                                distance_in_degree=dist_deg,
                                phase_list=["s"])
    
    incidence_angle = ray_paths[0].incident_angle
    
    return incidence_angle


def calculate_incidence_angle_eigenvalue(trace_z, trace_n, trace_e, p_arrival_offset=4.0, 
                                       analysis_window=0.12):
    """
    Calculate incidence angle using eigenvalue decomposition on P-wave window.
    
    This method analyzes the covariance matrix of the three-component P-wave data
    to determine the principal direction of particle motion and calculate the
    incidence angle from the vertical.
    
    Parameters:
    -----------
    trace_z, trace_n, trace_e : obspy.Trace
        Vertical, North, and East component traces
    p_arrival_offset : float
        Time of P-arrival from trace start (seconds)
    analysis_window : float
        Window duration for P-wave analysis (seconds)
    
    Returns:
    --------
    float
        Incidence angle in degrees (0-90°), where 0° is vertical incidence
    """
    try:
        # Get sampling rate and calculate window samples
        fs = trace_z.stats.sampling_rate
        p_start_sample = int(p_arrival_offset * fs)
        window_samples = int(analysis_window * fs)
        p_end_sample = p_start_sample + window_samples
        
        # Extract P-wave windows from all components
        z_window = trace_z.data[p_start_sample:p_end_sample]
        n_window = trace_n.data[p_start_sample:p_end_sample]
        e_window = trace_e.data[p_start_sample:p_end_sample]
        
        # Ensure all windows have same length
        min_length = min(len(z_window), len(n_window), len(e_window))
        z_window = z_window[:min_length]
        n_window = n_window[:min_length]
        e_window = e_window[:min_length]
        
        # Construct data matrix (each row is a component, each column is a time sample)
        data_matrix = np.array([z_window, n_window, e_window])
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(data_matrix)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Principal eigenvector (direction of maximum variance)
        principal_eigenvector = eigenvectors[:, 0]  # First column after sorting
        
        # Calculate incidence angle from vertical
        # The incidence angle is the angle between the principal eigenvector and the vertical (Z) axis
        z_component = abs(principal_eigenvector[0])  # Vertical component
        
        # Ensure z_component is within valid range for arccos
        z_component = np.clip(z_component, 0, 1)
        
        # Incidence angle is the angle from vertical
        incidence_angle = np.rad2deg(np.arccos(z_component))
        
        return incidence_angle
        
    except Exception as e:
        print(f"Eigenvalue incidence angle calculation error: {e}")
        return np.nan


def calculate_incidence_angle_eigenvalue_jurkevics(trace_z, trace_n, trace_e, p_arrival_offset, 
                                                     analysis_window=0.12, p_window_before=0.02):
    """
    Calculate incidence angle using Jurkevics (1988) polarization analysis method.
    
    This method analyzes the covariance matrix of three-component P-wave data using
    the exact Jurkevics formulation. The implementation assumes Z-axis positive upward
    (seismological convention) and forces the principal eigenvector to point downward.
    
    Default window follows run_sws.py convention: starts 0.02s before P-arrival,
    extends 0.1s after P-arrival (total 0.12s window).
    
    Reference: Jurkevics, A. (1988). Polarization analysis of three-component array data.
    
    Parameters:
    -----------
    trace_z, trace_n, trace_e : obspy.Trace
        Vertical (Z-up), North, and East component traces
    p_arrival_offset : float
        Time of P-arrival from trace start (seconds)
    analysis_window : float
        Total window duration for P-wave analysis (seconds), default=0.12s
    p_window_before : float
        Time before P-arrival to start window (seconds), default=0.02s
    
    Returns:
    --------
    float
        Incidence angle in degrees (0-90°), where 0° is vertical incidence
        
    Note:
    -----
    Z-axis convention: Positive upward (standard seismological convention)
    The eigenvector is flipped to point downward if needed before calculating incidence.
    Window timing matches run_sws.py: [P-0.02s, P+0.1s]
    """
    try:
       # Extract P-wave windows from all components
        window_start = p_arrival_offset - p_window_before
        window_end = p_arrival_offset - p_window_before + analysis_window

        #trace_z = trace_z.filter("bandpass", freqmin=5, freqmax=40)
        #trace_z = trace_z.taper(type="hann", max_percentage=0.05)

        #trace_n = trace_n.filter("bandpass", freqmin=5, freqmax=40)
        #trace_n = trace_n.taper(type="hann", max_percentage=0.05)
        
        #trace_e = trace_e.filter("bandpass", freqmin=5, freqmax=40)
        #trace_e = trace_e.taper(type="hann", max_percentage=0.05)
        
        # DEBUG: Check window coverage
        print(f"    Trace Z: {trace_z.stats.starttime} to {trace_z.stats.endtime}")
        print(f"    P-window: {window_start} to {window_end}")
        
        z_data = trace_z.slice(window_start, window_end).data
        n_data = trace_n.slice(window_start, window_end).data
        e_data = trace_e.slice(window_start, window_end).data
        
        # Construct data matrix (N_samples x 3) with Z, N, E columns (Z upward)
        data_zne = np.column_stack([z_data, n_data, e_data])
        
        # Eigenvalue decomposition using Jurkevics method
        eigvals, eigvecs = cov_eig(data_zne)
        eigvec1 = eigvecs[:, 0]
        lambda1, lambda2, lambda3 = eigvals
        
        # Make sure the eigenvector points towards the ground (-Z)
        # If Z component is positive (pointing up), flip the vector
        if eigvec1[0] >= 0:
            eigvec1 = -eigvec1
        
        # Calculate incidence angle from vertical (Jurkevics formula)
        # inc = arccos(|Z-component|) converts from vertical
        inc = np.arccos(np.abs(eigvec1[0])) * 180 / np.pi
        
        return inc
        
    except Exception as e:
        print(f"Jurkevics incidence angle calculation error: {e}")
        return np.nan


def calculate_incidence_angle_for_organized_waveforms(organized_waveforms, stations_df):
    """
    Calculate incidence angle for all events in organized_waveforms and add to the dataset.
    
    All required event metadata (lat, lon, station) is already in organized_waveforms,
    so no external catalog lookup is needed.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    stations_df : pandas.DataFrame
        Station information with coordinates (station, latitude, longitude)
        
    Returns:
    --------
    dict
        Updated organized_waveforms with incidence angle values added to each event
    """
    
    print(f"Calculating incidence angle for {len(organized_waveforms)} events...")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")
        
        # Get event location from event_data
        event_lat = event_data.get('latitude')
        event_lon = event_data.get('longitude')
        event_depth = event_data.get('depth')
        
        if event_lat is None or event_lon is None:
            print(f"  No event location found in event_data")
            event_data['incidence'] = np.nan
            continue
        
        # Get station name from event data
        station_name = event_data.get('station')
        
        if station_name is None:
            print(f"  No station information for event {event_id}")
            event_data['incidence'] = np.nan
            continue
        
        # Find station coordinates
        station_info = stations_df[stations_df['Station ID'] == station_name]
        
        if station_info.empty:
            print(f"  No coordinates found for station {station_name}")
            event_data['incidence'] = np.nan
            continue
        
        station_lat = station_info.iloc[0]['Latitude (°N)']
        station_lon = station_info.iloc[0]['Longitude (°W)']
        
        # Calculate incidence angle (from station to event)
        incidence_angle = calculate_incidence_angle(event_lat, event_lon, event_depth, station_lat, station_lon)
        
        # Add to event_data
        event_data['incidence'] = incidence_angle
        
        print(f"  Station: {station_name}")
        print(f"  Event location: ({event_lat:.4f}, {event_lon:.4f})")
        print(f"  Station location: ({station_lat:.4f}, {station_lon:.4f})")
        print(f"  Incidence angle: {incidence_angle:.2f}°")
        
        success_count += 1
    
    print(f"\n{'='*60}")
    print("Incidence Angle Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid incidence angle: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    incidence_angle_values = [data.get('incidence', np.nan) for data in organized_waveforms.values()]
    valid_incidence_angles = [v for v in incidence_angle_values if not np.isnan(v)]
    
    if valid_incidence_angles:
        print(f"Incidence angle range: {min(valid_incidence_angles):.2f}° to {max(valid_incidence_angles):.2f}°")
        print(f"Mean incidence angle: {np.mean(valid_incidence_angles):.2f}°")
    
    return organized_waveforms


def calculate_incidence_angle_eigenvalue_for_organized_waveforms(organized_waveforms, p_arrival_variable='p_arrival_time', analysis_window=0.5):
    """
    Calculate incidence angle using eigenvalue decomposition for all events in organized_waveforms.
    
    This version uses the eigenvalue decomposition method on the actual P-wave data
    rather than ray tracing calculations, providing a data-driven approach.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    p_arrival_variable : str
        Name of variable containing P-arrival time offset from trace start
    analysis_window : float
        Window duration for P-wave analysis (seconds)
        
    Returns:
    --------
    dict
        Updated organized_waveforms with eigenvalue-based incidence angle values
    """
    
    print(f"Calculating eigenvalue-based incidence angle for {len(organized_waveforms)} events...")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")
        
        # Get traces from event data
        traces = event_data.get('traces')
        if traces is None:
            print(f"  No traces found for event {event_id}")
            event_data['incidence_eigenvalue'] = np.nan
            continue
        
        # Get P-arrival time
        p_arrival_time = event_data.get('p_arrival_time')
        if p_arrival_time is None:
            print(f"  No P-arrival time found for event {event_id}")
            event_data['incidence_eigenvalue'] = np.nan
            continue
        
        try:
            # Sort traces and extract components
            traces_sorted = traces.sort(['channel'])
            
            # Find Z, N, E components
            trace_z = None
            trace_n = None 
            trace_e = None
            
            for tr in traces_sorted:
                channel_last_char = tr.stats.channel[-1].upper()
                if channel_last_char == 'Z':
                    trace_z = tr
                elif channel_last_char == 'N' or channel_last_char == '1':
                    trace_n = tr
                elif channel_last_char == 'E' or channel_last_char == '2':
                    trace_e = tr
            
            if trace_z is None or trace_n is None or trace_e is None:
                print(f"  Missing components for event {event_id}")
                print(f"  Available channels: {[tr.stats.channel for tr in traces_sorted]}")
                event_data['incidence_eigenvalue'] = np.nan
                continue
            
            # Calculate incidence angle using eigenvalue method
            incidence_angle = calculate_incidence_angle_eigenvalue(
                trace_z, trace_n, trace_e, 
                p_arrival_offset=p_arrival_time,
                analysis_window=analysis_window
            )
            
            # Add to event_data
            event_data['incidence_eigenvalue'] = incidence_angle
            
            print(f"  Station: {event_data.get('station', 'Unknown')}")
            print(f"  P-arrival time: {p_arrival_time:.3f}s")
            print(f"  Analysis window: {analysis_window:.3f}s") 
            print(f"  Eigenvalue incidence angle: {incidence_angle:.2f}°")
            
            success_count += 1
            
        except Exception as e:
            print(f"  Error calculating eigenvalue incidence angle for event {event_id}: {e}")
            event_data['incidence_eigenvalue'] = np.nan
    
    print(f"\n{'='*60}")
    print("Eigenvalue Incidence Angle Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid eigenvalue incidence angle: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    incidence_angle_values = [data.get('incidence_eigenvalue', np.nan) for data in organized_waveforms.values()]
    valid_incidence_angles = [v for v in incidence_angle_values if not np.isnan(v)]
    
    if valid_incidence_angles:
        print(f"Eigenvalue incidence angle range: {min(valid_incidence_angles):.2f}° to {max(valid_incidence_angles):.2f}°")
        print(f"Mean eigenvalue incidence angle: {np.mean(valid_incidence_angles):.2f}°")
    
    return organized_waveforms


#def calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms(organized_waveforms, p_arrival_variable='p_arrival_time', analysis_window=0.12):
    """
    Calculate incidence angle using Jurkevics (1988) method for all events in organized_waveforms.
    
    This version uses the Jurkevics polarization analysis method on actual P-wave data,
    providing a data-driven approach with proper eigenvector orientation handling and
    Z-up convention (seismological standard).
    
    Default window follows run_sws.py convention: starts 0.02s before P-arrival,
    extends 0.1s after P-arrival (total 0.12s window).
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    p_arrival_variable : str
        Name of variable containing P-arrival time offset from trace start
    analysis_window : float
        Window duration for P-wave analysis (seconds), default=0.12s
        
    Returns:
    --------
    dict
        Updated organized_waveforms with Jurkevics incidence angle values stored in
        'incidence_eigenvalue_jurkevics' field
    """
    
    print(f"Calculating Jurkevics-based incidence angle for {len(organized_waveforms)} events...")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")
        
        # Get traces from event data
        traces = event_data.get('traces')
        if traces is None:
            print(f"  No traces found for event {event_id}")
            event_data['incidence_eigenvalue_jurkevics'] = np.nan
            continue
        
        # Get P-arrival time
        p_arrival_offset = event_data.get('datetime') + event_data.get(str(p_arrival_variable))
        
        #if p_arrival_time is None:
        #    print(f"  No P-arrival time found for event {event_id}")
        #    event_data['incidence_eigenvalue_jurkevics'] = np.nan
        #    continue

        
        try:
            # Sort traces and extract components
            traces_sorted = traces.sort(['channel'])
            
            # Find Z, N, E components
            trace_z = None
            trace_n = None 
            trace_e = None
            
            for tr in traces_sorted:
                channel_last_char = tr.stats.channel[-1].upper()
                if channel_last_char == 'Z':
                    trace_z = tr
                elif channel_last_char == 'N' or channel_last_char == '1':
                    trace_n = tr
                elif channel_last_char == 'E' or channel_last_char == '2':
                    trace_e = tr
            
            if trace_z is None or trace_n is None or trace_e is None:
                print(f"  Missing components for event {event_id}")
                print(f"  Available channels: {[tr.stats.channel for tr in traces_sorted]}")
                event_data['incidence_eigenvalue_jurkevics'] = np.nan
                continue
            
            # Calculate incidence angle using Jurkevics method
            incidence_angle = calculate_incidence_angle_eigenvalue_jurkevics(
                trace_z, trace_n, trace_e, 
                p_arrival_offset=p_arrival_offset,
                analysis_window=analysis_window
            )
            
            # Add to event_data with Jurkevics-specific field name
            event_data['incidence_eigenvalue_jurkevics'] = incidence_angle
            
            print(f"  Station: {event_data.get('station', 'Unknown')}")
            print(f"  P-arrival time: {p_arrival_offset:.3f}s")
            print(f"  Analysis window: {analysis_window:.3f}s") 
            print(f"  Jurkevics incidence angle: {incidence_angle:.2f}°")
            
            success_count += 1
            
        except Exception as e:
            print(f"  Error calculating Jurkevics incidence angle for event {event_id}: {e}")
            event_data['incidence_eigenvalue_jurkevics'] = np.nan
    
    print(f"\n{'='*60}")
    print("Jurkevics Incidence Angle Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid Jurkevics incidence angle: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    incidence_angle_values = [data.get('incidence_eigenvalue_jurkevics', np.nan) for data in organized_waveforms.values()]
    valid_incidence_angles = [v for v in incidence_angle_values if not np.isnan(v)]
    
    if valid_incidence_angles:
        print(f"Jurkevics incidence angle range: {min(valid_incidence_angles):.2f}° to {max(valid_incidence_angles):.2f}°")
        print(f"Mean Jurkevics incidence angle: {np.mean(valid_incidence_angles):.2f}°")
    
    return organized_waveforms

def calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms(organized_waveforms, 
                                                                            p_arrival_variable='p_arrival_time',
                                                                            analysis_window=0.12):
    """
    Calculate incidence angle using Jurkevics (1988) method for all events in organized_waveforms.
    
    This version uses the Jurkevics polarization analysis method on actual P-wave data,
    providing a data-driven approach with proper eigenvector orientation handling and
    Z-up convention (seismological standard).
    
    Default window follows run_sws.py convention: starts 0.02s before P-arrival,
    extends 0.1s after P-arrival (total 0.12s window).
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    p_arrival_variable : str
        Name of variable containing P-arrival time offset from trace start
    analysis_window : float
        Window duration for P-wave analysis (seconds), default=0.12s
        
    Returns:
    --------
    dict
        Updated organized_waveforms with Jurkevics incidence angle values stored in
        'incidence_eigenvalue_jurkevics' field
    """
    
    print(f"Calculating Jurkevics-based incidence angle for {len(organized_waveforms)} events...")
    print(f"P-arrival variable: {p_arrival_variable}, Analysis window: {analysis_window}s")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")

        # Calculate absolute P-arrival time from datetime and relative offset
        datetime_utc = event_data.get('datetime')
        p_arrival_relative = event_data.get(str(p_arrival_variable))
        
        if datetime_utc is None or p_arrival_relative is None:
            print(f"  Missing datetime or P-arrival time")
            event_data['incidence_eigenvalue_jurkevics'] = np.nan
            continue
        
        # Convert to UTCDateTime and calculate absolute P-arrival
        p_arrival_offset = UTCDateTime(datetime_utc) + float(p_arrival_relative)
        
        # Get traces for this event
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  No traces found for event {event_id}")
            event_data['incidence_eigenvalue_jurkevics'] = np.nan
            continue
        
        # Convert to stream if it's a list
        if isinstance(event_traces, list):
            event_stream = obspy.Stream(event_traces)
        else:
            event_stream = event_traces
        
        print(f"  Found {len(event_stream)} traces")
        
        # Find Z, N, E components
        trace_z = None
        trace_n = None
        trace_e = None
        
        for tr in event_stream:
            component = tr.stats.channel[-1].upper()
            if component == 'Z':
                trace_z = tr
            elif component in ['N', '1']:
                trace_n = tr
            elif component in ['E', '2']:
                trace_e = tr
        
        # Check if we have all three components
        if trace_z is None or trace_n is None or trace_e is None:
            print(f"  Missing components: Z={trace_z is not None}, "
                  f"N={trace_n is not None}, E={trace_e is not None}")
            event_data['incidence_eigenvalue_jurkevics'] = np.nan
            continue
        
        # Perform incidence angle calculation using Jurkevics method
        try:
            incidence_angle = calculate_incidence_angle_eigenvalue_jurkevics(
                trace_z, trace_n, trace_e, 
                p_arrival_offset=p_arrival_offset,
                analysis_window=analysis_window
            )
            
            # Add results to event_data with Jurkevics-specific field name
            event_data['incidence_eigenvalue_jurkevics'] = incidence_angle
            
            print(f"  Results:")
            print(f"    Jurkevics incidence angle: {incidence_angle:.1f}°")
            
            if not np.isnan(incidence_angle):
                success_count += 1
                
        except Exception as e:
            print(f"  Error calculating Jurkevics incidence angle: {e}")
            event_data['incidence_eigenvalue_jurkevics'] = np.nan
    
    print(f"\n{'='*60}")
    print("Jurkevics Incidence Angle Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid Jurkevics incidence angle: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    incidence_angle_values = [data.get('incidence_eigenvalue_jurkevics', np.nan) for data in organized_waveforms.values()]
    valid_incidence_angles = [v for v in incidence_angle_values if not np.isnan(v)]
    
    if valid_incidence_angles:
        print(f"\nJurkevics Incidence Angle Statistics:")
        print(f"  Range: {min(valid_incidence_angles):.1f}° to {max(valid_incidence_angles):.1f}°")
        print(f"  Mean: {np.mean(valid_incidence_angles):.1f}°")
        print(f"  Median: {np.median(valid_incidence_angles):.1f}°")
        
        # Show how many pass typical QC threshold (e.g., < 30°)
        passing_inc = sum(1 for v in valid_incidence_angles if v <= 30.0)
        print(f"  Passing QC (≤30°): {passing_inc}/{len(valid_incidence_angles)} ({100*passing_inc/len(valid_incidence_angles):.1f}%)")
    
    return organized_waveforms


def calculate_rectilinearity(trace_z, trace_n, trace_e, p_arrival_offset=4.0, 
                                 analysis_window=1.0):
    """
    Analyze P-wave rectilinearity using covariance matrix analysis.
    
    Parameters:
    -----------
    trace_z, trace_n, trace_e : obspy.Trace
        Vertical, North, and East component traces
    p_arrival_offset : float
        Time of P-arrival from trace start (seconds)
    analysis_window : float
        Window duration for P-wave analysis (seconds)
    
    Returns:
    --------
    dict
        Rectilinearity metrics including eigenvalue ratios and polarization
    """
    try:
        # Ensure all traces have same sampling rate
        sampling_rate = trace_z.stats.sampling_rate
        if not (trace_n.stats.sampling_rate == sampling_rate and 
                trace_e.stats.sampling_rate == sampling_rate):
            return {'rectilinearity': np.nan}
        
        # Calculate sample indices for P-wave window
        p_arrival_idx = int(p_arrival_offset * sampling_rate)
        window_samples = int(analysis_window * sampling_rate)
        start_idx = p_arrival_idx
        end_idx = p_arrival_idx + window_samples
        
        # Ensure window is within trace bounds
        min_length = min(len(trace_z.data), len(trace_n.data), len(trace_e.data))
        if end_idx >= min_length or start_idx < 0:
            return {'rectilinearity': np.nan}
        
        # Extract P-wave data
        z_data = trace_z.data[start_idx:end_idx]
        n_data = trace_n.data[start_idx:end_idx]
        e_data = trace_e.data[start_idx:end_idx]
        
        # Create data matrix
        data_matrix = np.column_stack([z_data, n_data, e_data])
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(data_matrix.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Calculate rectilinearity (linearity measure)
        rectilinearity = 1 - (eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0])
        
        # Principal polarization vector (largest eigenvalue)
        principal_vector = eigenvectors[:, 0]
        
        
        return {
            'rectilinearity': rectilinearity,
            'eigenvalues': eigenvalues,
            'principal_vector': principal_vector
        }
    
    except Exception as e:
        print(f"P-wave rectilinearity analysis error: {e}")
        return {'rectilinearity': np.nan}


def calculate_rectilinearity_jurkevics(trace_z, trace_n, trace_e, p_arrival_offset, 
                                        analysis_window=0.12, p_window_before=0.02):
    """
    Calculate P-wave rectilinearity using Jurkevics (1988) polarization analysis method.
    
    This implementation uses the exact Jurkevics formulation for rectilinearity:
    rec = 1 - (lambda2 + lambda3) / (2 * lambda1)
    
    The method assumes Z-axis positive upward (seismological convention) and uses
    the cov_eig function for eigenvalue decomposition matching the original implementation.
    
    Default window follows run_sws.py convention: starts 0.02s before P-arrival,
    extends 0.1s after P-arrival (total 0.12s window).
    
    Reference: Jurkevics, A. (1988). Polarization analysis of three-component array data.
    
    Parameters:
    -----------
    trace_z, trace_n, trace_e : obspy.Trace
        Vertical (Z-up), North, and East component traces
    p_arrival_offset : float
        Time of P-arrival from trace start (seconds)
    analysis_window : float
        Total window duration for P-wave analysis (seconds), default=0.12s
    p_window_before : float
        Time before P-arrival to start window (seconds), default=0.02s
    
    Returns:
    --------
    float
        Rectilinearity value (0-1), where 1 indicates perfect linear polarization
        
    Note:
    -----
    Z-axis convention: Positive upward (standard seismological convention)
    Returns only the rectilinearity value, not eigenvalues or eigenvectors.
    Window timing matches run_sws.py: [P-0.02s, P+0.1s]
    """
    try:
        # Apply filtering and tapering to traces
        #trace_z = trace_z.filter("bandpass", freqmin=5, freqmax=40)
        #trace_z = trace_z.taper(type="hann", max_percentage=0.05)

        #trace_n = trace_n.filter("bandpass", freqmin=5, freqmax=40)
        #trace_n = trace_n.taper(type="hann", max_percentage=0.05)

        #trace_e = trace_e.filter("bandpass", freqmin=5, freqmax=40)
        #trace_e = trace_e.taper(type="hann", max_percentage=0.05)

        # Ensure all traces have same sampling rate
        sampling_rate = trace_z.stats.sampling_rate
        if not (trace_n.stats.sampling_rate == sampling_rate and 
                trace_e.stats.sampling_rate == sampling_rate):
            return np.nan
        
        # Extract P-wave windows from all components
        window_start = p_arrival_offset - p_window_before
        window_end = p_arrival_offset - p_window_before + analysis_window
        
        # DEBUG: Check window coverage
        print(f"    Trace Z: {trace_z.stats.starttime} to {trace_z.stats.endtime}")
        print(f"    P-window: {window_start} to {window_end}")
        
        # Extract P-wave data
        z_data = trace_z.slice(window_start, window_end).data
        n_data = trace_n.slice(window_start, window_end).data
        e_data = trace_e.slice(window_start, window_end).data
        
        # Create data matrix (N_samples x 3) with Z, N, E columns (Z upward)
        data_zne = np.column_stack([z_data, n_data, e_data])
        
        # Eigenvalue decomposition using Jurkevics method
        eigvals, eigvecs = cov_eig(data_zne)
        lambda1, lambda2, lambda3 = eigvals
        
        # Calculate rectilinearity using exact Jurkevics formula
        rec = 1 - (lambda2 + lambda3) / (2 * lambda1)
        
        return rec
    
    except Exception as e:
        print(f"Jurkevics rectilinearity calculation error: {e}")
        return np.nan

    
def calculate_rectilinearity_for_organized_waveforms(organized_waveforms,
                                                      p_arrival_variable,
                                                      analysis_window=1.0):
    """
    Calculate P-wave rectilinearity and incidence angle for all events in organized_waveforms.
    
    This function runs the P-wave covariance matrix analysis on all events,
    extracting QC metrics (rectilinearity and incidence angle) needed for 
    quality control filtering.
    
    All required metadata is already in organized_waveforms, so no external 
    catalog lookup is needed.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    p_arrival_offset : float, optional
        Time of P-arrival from trace start (seconds), default=4.0
    analysis_window : float, optional
        Window duration for P-wave analysis (seconds), default=1.0
        
    Returns:
    --------
    dict
        Updated organized_waveforms with P-wave metrics added to each event:
        - 'rectilinearity': P-wave linearity measure (0-1, higher is better)
    """
    
    print(f"Calculating P-wave metrics for {len(organized_waveforms)} events...")
    print(f"P-arrival offset: {p_arrival_variable}s, Analysis window: {analysis_window}s")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")

        p_arrival_offset = event_data.get(str(p_arrival_variable))
        
        # Get traces for this event
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  No traces found for event {event_id}")
            event_data['rectilinearity'] = np.nan
            event_data['incidence'] = np.nan
            event_data['p_wave_azimuth'] = np.nan
            continue
        
        # Convert to stream if it's a list
        if isinstance(event_traces, list):
            event_stream = obspy.Stream(event_traces)
        else:
            event_stream = event_traces
        
        print(f"  Found {len(event_stream)} traces")
        
        # Find Z, N, E components
        trace_z = None
        trace_n = None
        trace_e = None
        
        for tr in event_stream:
            component = tr.stats.channel[-1].upper()
            if component == 'Z':
                trace_z = tr
            elif component in ['N', '1']:
                trace_n = tr
            elif component in ['E', '2']:
                trace_e = tr
        
        # Check if we have all three components
        if trace_z is None or trace_n is None or trace_e is None:
            print(f"  Missing components: Z={trace_z is not None}, "
                  f"N={trace_n is not None}, E={trace_e is not None}")
            event_data['rectilinearity'] = np.nan
            continue
        
        # Perform P-wave rectilinearity analysis
        try:
            p_wave_metrics = calculate_rectilinearity(
                trace_z, trace_n, trace_e, 
                p_arrival_offset=p_arrival_offset,
                analysis_window=analysis_window
            )
            
            # Add results to event_data
            event_data['rectilinearity'] = p_wave_metrics['rectilinearity']
            
            print(f"  Results:")
            print(f"    Rectilinearity: {p_wave_metrics['rectilinearity']:.3f}")
            
            if not np.isnan(p_wave_metrics['rectilinearity']):
                success_count += 1
                
        except Exception as e:
            print(f"  Error calculating P-wave metrics: {e}")
            event_data['rectilinearity'] = np.nan
    
    print(f"\n{'='*60}")
    print("P-Wave Metrics Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid P-wave metrics: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    rect_values = [data.get('rectilinearity', np.nan) for data in organized_waveforms.values()]
    
    valid_rect = [v for v in rect_values if not np.isnan(v)]
    
    if valid_rect:
        print(f"\nRectilinearity Statistics:")
        print(f"  Range: {min(valid_rect):.3f} to {max(valid_rect):.3f}")
        print(f"  Mean: {np.mean(valid_rect):.3f}")
        print(f"  Median: {np.median(valid_rect):.3f}")
        
        # Show how many pass typical QC threshold
        passing_rect = sum(1 for v in valid_rect if v >= 0.7)
        print(f"  Passing QC (≥0.7): {passing_rect}/{len(valid_rect)} ({100*passing_rect/len(valid_rect):.1f}%)")
    
    return organized_waveforms


def calculate_rectilinearity_jurkevics_for_organized_waveforms(organized_waveforms,
                                                                p_arrival_variable='p_arrival_time',
                                                                analysis_window=0.12):
    """
    Calculate P-wave rectilinearity using Jurkevics (1988) method for all events in organized_waveforms.
    
    This function runs the Jurkevics polarization analysis on all events, using the exact
    formula: rec = 1 - (lambda2 + lambda3) / (2 * lambda1) with Z-up convention.
    
    Default window follows run_sws.py convention: starts 0.02s before P-arrival,
    extends 0.1s after P-arrival (total 0.12s window).
    
    Results are stored in the 'rectilinearity_jurkevics' field for comparison with
    other methods.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    p_arrival_variable : str
        Name of variable containing P-arrival time offset from trace start  
    analysis_window : float, optional
        Window duration for P-wave analysis (seconds), default=0.12s
        
    Returns:
    --------
    dict
        Updated organized_waveforms with Jurkevics rectilinearity values stored in
        'rectilinearity_jurkevics' field
    """
    
    print(f"Calculating Jurkevics P-wave rectilinearity for {len(organized_waveforms)} events...")
    print(f"P-arrival variable: {p_arrival_variable}, Analysis window: {analysis_window}s")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")

        p_arrival_offset = UTCDateTime(event_data.get('datetime') + event_data.get(str(p_arrival_variable)))
        
        # Get traces for this event
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  No traces found for event {event_id}")
            event_data['rectilinearity_jurkevics'] = np.nan
            continue
        
        # Convert to stream if it's a list
        if isinstance(event_traces, list):
            event_stream = obspy.Stream(event_traces)
        else:
            event_stream = event_traces
        
        print(f"  Found {len(event_stream)} traces")
        
        # Find Z, N, E components
        trace_z = None
        trace_n = None
        trace_e = None
        
        for tr in event_stream:
            component = tr.stats.channel[-1].upper()
            if component == 'Z':
                trace_z = tr
            elif component in ['N', '1']:
                trace_n = tr
            elif component in ['E', '2']:
                trace_e = tr
        
        # Check if we have all three components
        if trace_z is None or trace_n is None or trace_e is None:
            print(f"  Missing components: Z={trace_z is not None}, "
                  f"N={trace_n is not None}, E={trace_e is not None}")
            event_data['rectilinearity_jurkevics'] = np.nan
            continue
        
        # Perform P-wave rectilinearity analysis using Jurkevics method
        try:
            rectilinearity = calculate_rectilinearity_jurkevics(
                trace_z, trace_n, trace_e, 
                p_arrival_offset=p_arrival_offset,
                analysis_window=analysis_window
            )
            
            # Add results to event_data with Jurkevics-specific field name
            event_data['rectilinearity_jurkevics'] = rectilinearity
            
            print(f"  Results:")
            print(f"    Jurkevics rectilinearity: {rectilinearity:.3f}")
            
            if not np.isnan(rectilinearity):
                success_count += 1
                
        except Exception as e:
            print(f"  Error calculating Jurkevics rectilinearity: {e}")
            event_data['rectilinearity_jurkevics'] = np.nan
    
    print(f"\n{'='*60}")
    print("Jurkevics P-Wave Rectilinearity Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid Jurkevics rectilinearity: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    rect_values = [data.get('rectilinearity_jurkevics', np.nan) for data in organized_waveforms.values()]
    
    valid_rect = [v for v in rect_values if not np.isnan(v)]
    
    if valid_rect:
        print(f"\nJurkevics Rectilinearity Statistics:")
        print(f"  Range: {min(valid_rect):.3f} to {max(valid_rect):.3f}")
        print(f"  Mean: {np.mean(valid_rect):.3f}")
        print(f"  Median: {np.median(valid_rect):.3f}")
        
        # Show how many pass typical QC threshold
        passing_rect = sum(1 for v in valid_rect if v >= 0.7)
        print(f"  Passing QC (≥0.7): {passing_rect}/{len(valid_rect)} ({100*passing_rect/len(valid_rect):.1f}%)")
    
    return organized_waveforms

def perform_splitting_analysis(event_data, first_window_start, last_window_start, first_window_end, last_window_end, n_win, s_pick_uncertainty, plot_results=False):
    """
    Perform shear-wave splitting analysis using data from organized_waveforms.
    
    This function now uses dynamic parameter calculation to optimize the analysis
    window and filtering based on the event's spectral characteristics. All required
    data (traces, timing, metadata) comes from event_data.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing:
        - traces: ObsPy stream with N/E/Z components
        - station: Station name
        - back_azimuth: Geographic back-azimuth
        - incidence_eigenvalue_jurkevics: P-wave incidence angle
        - s_arrival_time: S-wave arrival (seconds from origin)
        - datetime: Event origin time
        - magnitude: Event magnitude
        - All QC metrics (snr_horizontal, rectilinearity, etc.)
    plot_results : bool, optional
        Whether to generate diagnostic plots (default=True)
    
    Returns:
    --------
    tuple
        (result_dict, splitting_obj) containing:
        - result_dict: Dict with splitting parameters and metadata
        - splitting_obj: SWSPy splitting object with sws_result_df and plots
    """
    
    station_name = event_data.get('station', 'UNKNOWN')
    
    try:
        print(f"  Creating SWSPy splitting object with MFAST-like windowing parameters...")
        
        # Create the splitting analysis object using our helper function
        # This handles all the setup: filtering, windowing, parameter optimization
        splitting_obj = create_splitting_analysis(event_data, first_window_start=first_window_start, last_window_start=last_window_start, 
                                                  first_window_end=first_window_end, last_window_end=last_window_end, n_win=n_win, s_pick_uncertainty=s_pick_uncertainty)
        
        # Get dominant period
        Tmid = splitting_obj.Tmid

        # Perform the actual splitting analysis
        print(f"  Running SWSPy splitting measurement...")
        #splitting_obj.perform_sws_analysis(coord_system="ZNE", sws_method="EV_and_XC")
        splitting_obj.perform_sws_analysis(coord_system="LQT", sws_method="EV_and_XC")
        
        # Extract results from sws_result_df DataFrame
        print(f"  Extracting results from sws_result_df...")
        
        if not hasattr(splitting_obj, 'sws_result_df'):
            raise ValueError("splitting_obj does not have 'sws_result_df' attribute after perform_sws_analysis()")
        
        results_df = splitting_obj.sws_result_df
        
        if results_df is None or len(results_df) == 0:
            raise ValueError("sws_result_df is empty or None")
        
        # Extract splitting parameters from the first row (should be only one station)
        result_row = results_df.iloc[0]
        
        # Get splitting parameters (adjust column names based on actual DataFrame)
        phi_best = result_row.get('fast', result_row.get('phi_from_N', np.nan))
        dt_best = result_row.get('lag', result_row.get('dt', np.nan))
        
        # Get error estimates if available
        phi_error = result_row.get('fast_error', result_row.get('phi_err', np.nan))
        dt_error = result_row.get('lag_error', result_row.get('dt_err', np.nan))

        # Get quality metrics if available
        quality = result_row.get('Q_w', result_row.get('Q_w', np.nan))
        
        # Get other metadata from event_data
        avg_snr = event_data.get('snr_horizontal', np.nan)
        magnitude = event_data.get('magnitude', np.nan)
        s_arrival_time = event_data.get('s_arrival_time', np.nan)
        
        result_dict = {
            'station': station_name,
            'phi': float(phi_best) if not pd.isna(phi_best) else np.nan,
            'dt': float(dt_best) if not pd.isna(dt_best) else np.nan,
            'phi_error': float(phi_error) if not pd.isna(phi_error) else np.nan,
            'dt_error': float(dt_error) if not pd.isna(dt_error) else np.nan,
            'quality': float(quality) if not pd.isna(quality) else np.nan,
            'snr_avg': avg_snr,
            's_arrival_time': s_arrival_time,
            'magnitude': magnitude,
            'dominant_period' : Tmid,
            'success': True if not np.isnan(phi_best) and not np.isnan(dt_best) else False,
            'sws_result_df': results_df  # Include full DataFrame for reference
        }
        
        # Generate diagnostic plots
        if plot_results:
            print(f"  Generating diagnostic plots...")
            try:
                splitting_obj.plot()
                plt.show()
            except Exception as plot_error:
                print(f"  Warning: Could not generate plots: {plot_error}")
        
        print(f"  ✓ Splitting analysis complete!")
        print(f"    Fast axis (φ): {result_dict['phi']:.1f}°" if not np.isnan(result_dict['phi']) else "    Fast axis (φ): N/A")
        print(f"    Delay time (δt): {result_dict['dt']:.3f}s" if not np.isnan(result_dict['dt']) else "    Delay time (δt): N/A")
        
        # Return both the result dict and the splitting object
        return result_dict, splitting_obj
    
    except Exception as e:
        import traceback
        error_dict = {
            'station': station_name,
            'phi': np.nan,
            'dt': np.nan,
            'phi_error': np.nan,
            'dt_error': np.nan,
            'quality': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }
        print(f"  ✗ Error: {e}")
        return error_dict, None

def perform_splitting_analysis_orig(event_data, first_window_start, last_window_start, first_window_end, last_window_end, n_win, s_pick_uncertainty, plot_results=False):
    """
    Perform shear-wave splitting analysis using data from organized_waveforms.
    
    This function now uses dynamic parameter calculation to optimize the analysis
    window and filtering based on the event's spectral characteristics. All required
    data (traces, timing, metadata) comes from event_data.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing:
        - traces: ObsPy stream with N/E/Z components
        - station: Station name
        - back_azimuth: Geographic back-azimuth
        - incidence_eigenvalue_jurkevics: P-wave incidence angle
        - s_arrival_time: S-wave arrival (seconds from origin)
        - datetime: Event origin time
        - magnitude: Event magnitude
        - All QC metrics (snr_horizontal, rectilinearity, etc.)
    plot_results : bool, optional
        Whether to generate diagnostic plots (default=True)
    
    Returns:
    --------
    tuple
        (result_dict, splitting_obj) containing:
        - result_dict: Dict with splitting parameters and metadata
        - splitting_obj: SWSPy splitting object with sws_result_df and plots
    """
    
    station_name = event_data.get('station', 'UNKNOWN')
    
    try:
        print(f"  Creating SWSPy splitting object with MFAST-like windowing parameters...")
        
        # Create the splitting analysis object using our helper function
        # This handles all the setup: filtering, windowing, parameter optimization
        splitting_obj = create_splitting_analysis(event_data, first_window_start=first_window_start, last_window_start=last_window_start, 
                                                  first_window_end=first_window_end, last_window_end=last_window_end, n_win=n_win, s_pick_uncertainty=s_pick_uncertainty)
        
        # Get dominant period
        Tmid = splitting_obj.Tmid

        # Perform the actual splitting analysis
        print(f"  Running SWSPy splitting measurement...")
        splitting_obj.perform_sws_analysis(coord_system="ZNE", sws_method="EV_and_XC")
        
        # Extract results from sws_result_df DataFrame
        print(f"  Extracting results from sws_result_df...")
        
        if not hasattr(splitting_obj, 'sws_result_df'):
            raise ValueError("splitting_obj does not have 'sws_result_df' attribute after perform_sws_analysis()")
        
        results_df = splitting_obj.sws_result_df
        
        if results_df is None or len(results_df) == 0:
            raise ValueError("sws_result_df is empty or None")
        
        # Extract splitting parameters from the first row (should be only one station)
        result_row = results_df.iloc[0]
        
        # Get splitting parameters (adjust column names based on actual DataFrame)
        phi_best = result_row.get('fast', result_row.get('phi_from_N', np.nan))
        dt_best = result_row.get('lag', result_row.get('dt', np.nan))
        
        # Get error estimates if available
        phi_error = result_row.get('fast_error', result_row.get('phi_err', np.nan))
        dt_error = result_row.get('lag_error', result_row.get('dt_err', np.nan))
        
        # Get other metadata from event_data
        avg_snr = event_data.get('snr_horizontal', np.nan)
        magnitude = event_data.get('magnitude', np.nan)
        s_arrival_time = event_data.get('s_arrival_time', np.nan)
        
        result_dict = {
            'station': station_name,
            'phi': float(phi_best) if not pd.isna(phi_best) else np.nan,
            'dt': float(dt_best) if not pd.isna(dt_best) else np.nan,
            'phi_error': float(phi_error) if not pd.isna(phi_error) else np.nan,
            'dt_error': float(dt_error) if not pd.isna(dt_error) else np.nan,
            'snr_avg': avg_snr,
            's_arrival_time': s_arrival_time,
            'magnitude': magnitude,
            'dominant_period' : Tmid,
            'success': True if not np.isnan(phi_best) and not np.isnan(dt_best) else False,
            'sws_result_df': results_df  # Include full DataFrame for reference
        }
        
        # Generate diagnostic plots
        if plot_results:
            print(f"  Generating diagnostic plots...")
            try:
                splitting_obj.plot()
                plt.show()
            except Exception as plot_error:
                print(f"  Warning: Could not generate plots: {plot_error}")
        
        print(f"  ✓ Splitting analysis complete!")
        print(f"    Fast axis (φ): {result_dict['phi']:.1f}°" if not np.isnan(result_dict['phi']) else "    Fast axis (φ): N/A")
        print(f"    Delay time (δt): {result_dict['dt']:.3f}s" if not np.isnan(result_dict['dt']) else "    Delay time (δt): N/A")
        
        # Return both the result dict and the splitting object
        return result_dict, splitting_obj
    
    except Exception as e:
        import traceback
        error_dict = {
            'station': station_name,
            'phi': np.nan,
            'dt': np.nan,
            'phi_error': np.nan,
            'dt_error': np.nan,
            'quality': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }
        print(f"  ✗ Error: {e}")
        return error_dict, None


# ============================================================================
# BAILLARD-STYLE SHEAR-WAVE SPLITTING IMPLEMENTATION
# ============================================================================
# Based on Christian Baillard's run_sws.py and shearwavesplit.py
# Implements custom eigenvalue grid search method
# ============================================================================

class MinLambda:
    """
    Storage class for splitting solution from eigenvalue minimum.
    Based on Christian Baillard's MinLambda class from shearwavesplit.py
    """
    def __init__(self):
        self.lag = None           # Delay time in samples
        self.angle = None         # Fast direction in radians (-π/2 to π/2)
        self.lag_error = None     # Error in lag (samples)
        self.angle_error = None   # Error in angle (radians)
        self.lambda_value = None  # λ2 value at minimum (Baillard naming)
        self.quality = None       # Quality metric from mesh
        self.rec = None           # Rectilinearity at minimum
        self.rms = None           # RMS quality metric
        self.row_ind = None       # Row index in mesh
        self.col_ind = None       # Column index in mesh
        
    def to_dict(self, sampling_rate):
        """Convert to dictionary for results"""
        return {
            'dt': self.lag / sampling_rate if self.lag is not None else np.nan,
            'dt_samples': self.lag if self.lag is not None else np.nan,
            'phi': (np.degrees(self.angle)) if self.angle is not None else np.nan,
            'phi_rad': self.angle if self.angle is not None else np.nan,
            'dt_error': self.lag_error / sampling_rate if self.lag_error is not None else np.nan,
            'phi_error': np.degrees(self.angle_error) if self.angle_error is not None else np.nan,
            'lambda2': self.lambda_value if self.lambda_value is not None else np.nan,
            'quality': self.quality if self.quality is not None else np.nan,
            'rec': self.rec if self.rec is not None else np.nan,
            'rms': self.rms if self.rms is not None else np.nan
        }
    
    def match(self, lambda_value=None, lambda_quality=None, lambda_lag=None, lambda_angle=None,
              lambda_lag_error=None, lambda_angle_error=None, lambda_rec=None, **kwargs):
        """
        Check if MinLambda instance matches given criteria.
        From Baillard's shearwavesplit.py MinLambda.match() method.
        
        Parameters:
        -----------
        lambda_*: list with 2 elements
            Range [min, max] for threshold to be applied
            
        Returns:
        --------
        bool
            True if class passes all thresholds
        """
        class_dict = {
            'lambda_value': lambda_value,
            'lag': lambda_lag, 'lag_error': lambda_lag_error,
            'angle': lambda_angle, 'angle_error': lambda_angle_error,
            'rec': lambda_rec, 'quality': lambda_quality
        }
        
        # Return True if all criteria are None
        if all(value is None for value in class_dict.values()):
            return True
        
        # Check each criterion
        return all((getattr(self, key) >= min(list_val)) & (getattr(self, key) <= max(list_val))
                   for (key, list_val) in class_dict.items() if list_val is not None)


def rotate_basis_2_baillard(data, angle_rad, flag_plot=False):
    """
    Rotate coordinate system for 2-component data.
    From shearwavesplit.py line 429
    
    Parameters:
    -----------
    data_array : ndarray [N x 2]
        Two-component data [X, Y]
    angle_rad : float
        Rotation angle in radians (counterclockwise positive)
        
    Returns:
    --------
    ndarray [N x 2]
        Rotated data
    """
    #rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
    #                      [np.sin(angle_rad), np.cos(angle_rad)]])
    #return np.dot(data_array, rot_matrix.T)

     #### Check data
    
    num_col=data.shape[1]

    if num_col!=2:
        raise ValueError('Data has wrong shape, num col is %i'%num_col)
        return
    

    ### Rotation (Change of basis matrix)
    
    sina = np.sin(angle_rad)
    cosa = np.cos(angle_rad)
    
    ##### Check Matrix
    
    R=np.array([[cosa , sina],
             [-sina, cosa]])

    data=data.transpose()
    
    ### Apply product
    
    new_data=np.dot(R,data).transpose()
    data=data.transpose()
    
    #### Plot if asked
    
    if flag_plot:
        x,ax=plt.subplots()
        ax.plot(data[:,0],data[:,1],'ok')
        ax.plot(new_data[:,0],new_data[:,1],'or')
    
    ### Return and plot
    
    return new_data


def shift_2_baillard(data_array, Nsamp):
    """
    Apply time shift to two-component waveforms.
    From shearwavesplit.py line 451
    
    X component shifted backward (earlier in time)
    Y component shifted forward (later in time)
    This "unsplits" the waveforms
    
    Parameters:
    -----------
    data_array : ndarray [N x 2]
        Two-component data [X, Y]
    Nsamp : int
        Shift amount in samples (positive shifts X earlier, Y later)
        
    Returns:
    --------
    ndarray [N x 2]
        Shifted data
    """
    #data_shifted = np.copy(data_array)
    
    # Shift X component backward (roll with negative shift)
    #data_shifted[:, 0] = np.roll(data_array[:, 0], -int(shift))
    
    # Shift Y component forward (roll with positive shift)
    #data_shifted[:, 1] = np.roll(data_array[:, 1], int(shift))
    
    #return data_shifted

    y=data_array[:,1]
    
    y_new=np.zeros(y.shape)
    
    shift=abs(Nsamp)
    if Nsamp==0:
        y_new=y
    elif Nsamp<0:
        y_new[:-shift]=y[shift:]
    else:
        y_new[shift:]=y[:-shift]

    new_data=np.column_stack((data_array[:,0],y_new))
    return new_data


def unsplit_baillard(data_array, lag, angle_rad, flag_plot=False, ax_list=None):
    """
    Unsplit a given data array by applying rotation and shift corrections.
    Based on Baillard's unsplit function from shearwavesplit.py
    
    This function reverses the splitting effect by:
    1. Rotating to the fast/slow coordinate system
    2. Shifting to remove the delay time
    
    Parameters:
    -----------
    data_array : ndarray [N x 2]
        Two-component data [X, Y] (or [E, N])
    lag : float or int
        Lag in samples for "unshifting"
    angle_rad : float
        Angle CCW from X axis for "unrotating" (radians)
    flag_plot : bool, optional
        Whether to generate diagnostic plots (default=False)
    ax_list : list of matplotlib axes, optional
        List of axes for plotting
        
    Returns:
    --------
    tuple
        (data_array, ax_list) where data_array is unsplit [X, Y]
    """
    import copy
    
    lag = int(np.round(lag))
    ini_array = copy.deepcopy(data_array)
    shift = -lag  # If lag is positive, move it to the left
    
    # Apply rotation
    data_array = rotate_basis_2_baillard(data_array, angle_rad)
    
    # Apply shift
    data_array = shift_2_baillard(data_array, shift)
    
    ### Plot if asked
    if flag_plot:
        if ax_list is None:
            fig, ax_list = plt.subplots(3, 1, figsize=[7, 6], sharey=True, sharex=True)
            fig.tight_layout()
        
        ax_list[0].plot(ini_array[:, 0], 'k', label='X')
        ax_list[0].plot(ini_array[:, 1], '--r', label='Y')
        ax_list[1].plot(data_array[:, 0], 'k', label='X')
        ax_list[1].plot(data_array[:, 1], '--r', label='Y')
        ax_list[2].plot(data_array[:, 0], 'k', label='X')
        ax_list[2].plot(-data_array[:, 1], '--r', label='-Y')
        
        for ax in ax_list:
            ax.legend(loc=1)
        
        ### Title
        texts = ['Initial', 'Unsplit', 'Unsplit']
        
        k = -1
        for text in texts:
            k += 1
            ax_list[k].text(0.05, 0.85, text, fontweight='bold', 
                          horizontalalignment='left',
                          verticalalignment='top', 
                          transform=ax_list[k].transAxes)
        
        ax_list[2].set_xlabel('Samples')
    
    ### Return
    return (data_array, ax_list)


def cov_eig_baillard(data_array):
    """
    Compute covariance matrix eigenvalues.
    From shearwavesplit.py line 509
    
    Parameters:
    -----------
    data_array : ndarray [N x 2]
        Two-component data
        
    Returns:
    --------
    eig_vals : ndarray [2]
        Eigenvalues sorted descending [λ1, λ2]
    eig_vecs : ndarray [2 x 2]
        Eigenvectors (columns)
    """
    # Compute covariance matrix
    cov_matrix = np.cov(np.transpose(data_array))
    
    # Compute eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues descending
    ind_descend=np.argsort(-eig_vals)
    eig_vals=eig_vals[ind_descend]
    eig_vecs=eig_vecs[:,ind_descend]
    
    return eig_vals, eig_vecs


def gridsearch_baillard(data_array, shift, angle_rad, cut_s1, cut_s2):
    """
    Single grid search iteration: rotate, shift, cut, compute eigenvalues.
    From shearwavesplit.py line 565
    
    Parameters:
    -----------
    data_array : ndarray [N x 2]
        Two-component data
    shift : int
        Time shift in samples
    angle_rad : float
        Rotation angle in radians
    cut_s1 : int
        Start sample for windowing
    cut_s2 : int
        End sample for windowing
        
    Returns:
    --------
    list [λ1, λ2]
        Eigenvalues at this parameter combination
    """
    # Rotate
    data_array = rotate_basis_2_baillard(data_array, angle_rad)
    
    # Shift
    data_array = shift_2_baillard(data_array, Nsamp=shift)
    
    # Cut/window
    data_array=cut(data_array,cut_s1,cut_s2)
    
    # Compute eigenvalues
    eig_vals, _ = cov_eig_baillard(data_array)
    
    return list(eig_vals)

def cut(data_array,s0=0,s1=None):
    """
    cut data array
    
    Inputs
    ------
        s0: int: start sample for cutting
        s1: int,None: last sample for cutting
        
    Output:
    -------
        np.array
    """
    
    s0=int(np.round(s0))
    
    if s1 is None or s1>data_array.shape[0]:
        s1=data_array.shape[0]
        
    return data_array[s0:s1]


def get_LAMBDAS_baillard(xy_array, min_lag, max_lag, 
                        Nlags=100, Nangles=100,
                        cut_s1=0, cut_s2=None):
    """
    Compute eigenvalue meshes over lag/angle parameter space.
    From shearwavesplit.py line 615 (get_LAMBDAS)
    
    This is the core grid search that tests all combinations of
    delay time (lag) and fast direction (angle) parameters.
    
    Parameters:
    -----------
    xy_array : ndarray [N x 2]
        Two-component waveform data [X, Y]
    min_lag : float
        Minimum lag in samples
    max_lag : float
        Maximum lag in samples
    Nlags : int
        Number of lag values to test
    Nangles : int
        Number of angle values to test
    cut_s1 : int
        Start sample for analysis window
    cut_s2 : int or None
        End sample for analysis window
        
    Returns:
    --------
    LAMBDA1 : ndarray [Nangles x Nlags]
        First eigenvalue mesh
    LAMBDA2 : ndarray [Nangles x Nlags]
        Second eigenvalue mesh (minimized for splitting)
    LAGS : ndarray [Nangles x Nlags]
        Lag values mesh
    ANGLES : ndarray [Nangles x Nlags]
        Angle values mesh (radians)
    """
    print(f"    Grid search: {Nlags} lags × {Nangles} angles = {Nlags*Nangles} evaluations")
    
    min_lag,max_lag=int(np.round(min_lag)),int(np.round(max_lag))

    # Create parameter arrays
    #lags_array = np.linspace(min_lag, max_lag, Nlags)
    lags_array = sample_array(min_lag, max_lag, Nlags)
    angles_array = np.linspace(-np.pi/2, np.pi/2, Nangles)
    
    # Create meshgrids
    LAGS, ANGLES = np.meshgrid(lags_array, angles_array)

    lags=list(LAGS.reshape(-1))
    angles=list(ANGLES.reshape(-1))

    eigs_list=[gridsearch_baillard(xy_array,-lag,angle_rad,cut_s1=cut_s1,cut_s2=cut_s2) for lag,angle_rad in zip(lags,angles)]
    
    # Initialize eigenvalue arrays
    #LAMBDA1 = np.zeros_like(LAGS)
    #LAMBDA2 = np.zeros_like(LAGS)
    
    # Grid search over all parameter combinations
    #for i in range(Nangles):
    #    for j in range(Nlags):
    #        eig_vals = gridsearch_baillard(xy_array, -lags_array[j], 
    #                                      angles_array[i], cut_s1, cut_s2)
    #        LAMBDA1[i, j] = eig_vals[0]
    #        LAMBDA2[i, j] = eig_vals[1]

    eigs_array=np.array(eigs_list)
    LAMBDA1=np.reshape(eigs_array[:,0],LAGS.shape)
    LAMBDA2=np.reshape(eigs_array[:,1],LAGS.shape)
    
    print(f"    λ2 range: {LAMBDA2.min():.2e} to {LAMBDA2.max():.2e}")
    
    return LAMBDA1, LAMBDA2, LAGS, ANGLES


def process_LAMBDAS_baillard(LAMBDA2, LAMBDA1, LAGS, ANGLES,
                             min_lag_thres=None, max_lag_thres=None,
                             min_thres=0.5, min_numbers=2,
                             cont_step=0.05, quality_thres=0.2, zoom_factor=[4, 4],
                             flag_plot=True, ax=None):
    """
    Process LAMBDA2 mesh to select minima and output MinLambda objects.
    Exact replication of Baillard's process_LAMBDAS from shearwavesplit.py line 930
    
    The LAMBDA2 mesh is normalized from 0 to 1 and all thresholds are given 
    in that normalized reference. min_lag_thres ensures the minimum lambda 
    selected will not suffer from border effects.
    
    Parameters:
    -----------
    LAMBDA2, LAMBDA1 : ndarray
        Eigenvalue meshgrids (all same size)
    LAGS, ANGLES : ndarray
        Lag and angle meshgrids
    min_lag_thres : float or None
        Minimum lag threshold (samples)
    max_lag_thres : float or None
        Maximum lag threshold (samples)
    min_thres : float
        Keep only minima under this value (0-1 normalized)
    min_numbers : int
        Maximum number of minima to keep
    cont_step : float
        Step to plot contour around minima
    quality_thres : float
        Value to compute quality of grid
    zoom_factor : list [rows, cols]
        Zooming values to interpolate grid
        If [2,6], multiply rows by 2, columns by 6
    flag_plot : bool
        Whether to generate diagnostic plots
    ax : matplotlib axes
        Axes object for plotting
        
    Returns:
    --------
    MinLambdas : list of MinLambda
        List of splitting solutions
    ax : matplotlib axes
        Axes object if plotting
    """
    print(f"    Processing λ2 surface for minima...")
    
    ### Check thresholds
    if min_lag_thres is None:
        min_lag_thres = np.min(LAGS)
    
    if max_lag_thres is None:
        max_lag_thres = np.max(LAGS)
    
    #### Zoom to ensure good quality of contours
    LAMBDA2 = zoom(LAMBDA2, zoom_factor)
    LAMBDA1 = zoom(LAMBDA1, zoom_factor)
    
    ### Resample LAG grid (Zoom doesn't interpolate properly)
    lagsz = np.linspace(np.min(LAGS), np.max(LAGS), LAGS.shape[1] * zoom_factor[1])
    LAGSZ = np.tile(lagsz, (zoom_factor[0] * LAGS.shape[0], 1))
    
    ### Normalize
    MESH = minmax2zeroone(LAMBDA2)
    
    #### Get quality mesh
    quality = quality_mesh(MESH, threshold=quality_thres)
    
    #### Find all minima (i.e. all local minima)
    min_list = extrema(MESH)

    ### Pre-select minima that are only inside min_lag_thres and max_lag_thres
    min_list = [(ind_row, ind_col, lambda_v) for (ind_row, ind_col, lambda_v) in min_list \
                if (LAGSZ[ind_row, ind_col] >= min_lag_thres) & (LAGSZ[ind_row, ind_col] <= max_lag_thres)]
    
    ### Select properly the minima (below threshold and maximum of X minima)
    min_list = select_minima(min_list, min_thres, min_numbers)
    
    print(f"    Selected {len(min_list)} minima (threshold={min_thres}, max={min_numbers})")
    
    #######################################################
    ### Get contour and errors for each selected minima
    
    ### Get step
    extent_lag = [np.min(LAGS), np.max(LAGS)]
    extent_angle = [np.min(ANGLES), np.max(ANGLES)]
    
    d_lag = float(np.diff(extent_lag) / MESH.shape[1])
    d_angle = float(np.diff(extent_angle) / MESH.shape[0])
    
    ### Make a periodic mesh on angles (vertical) to ensure contours are closed
    PER_MESH = np.vstack((MESH, MESH[1:-1, :], MESH))
    
    ### Start loop
    MinLambdas = []
    
    kk = 0
    for ind_row, ind_col, min_value in min_list:
        kk += 1
        
        ### Compute rectilinearity at the minimum
        rec = 1 - LAMBDA2[ind_row, ind_col] / LAMBDA1[ind_row, ind_col]
        
        ### Modify indices according to new mesh
        ind_row_per = ind_row + MESH.shape[0] - 1
        contour_select, width, height, min_tuple, box, _ = get_valcontour(
            PER_MESH, ind_row_per, ind_col, cont_step, flag_plot=False)
        
        ### Report to LAGS and ANGLES instead of indexes
        lag = np.min(LAGS) + ind_col * d_lag
        angle = np.min(ANGLES) + ind_row * d_angle
        
        lag_error = 999
        angle_error = 999
        
        if contour_select is not None:
            lag_error = width * d_lag
            angle_error = height * d_angle
        
        ### Feed to class
        MinSc = MinLambda()
        MinSc.lag = lag
        MinSc.angle = angle
        MinSc.angle_error = angle_error
        MinSc.lag_error = lag_error
        MinSc.lambda_value = min_value
        MinSc.quality = quality
        MinSc.rec = rec
        
        MinLambdas.append(MinSc)
        
        #### Plot if asked
        if flag_plot:
            if ax is None:  ### Plot things just once in the loop
                fig, ax = plt.subplots()
            if kk == 1:
                ax.imshow(MESH, extent=extent_lag + extent_angle, origin='lower',
                         cmap=plt.cm.get_cmap('jet'), aspect='auto')  # GRID
                CS = ax.contour(MESH, extent=extent_lag + extent_angle, origin='lower',
                               levels=[min_thres], colors='w', linestyles=':')  # CONTOURS
                ax.clabel(CS, [min_thres], fmt='%.2f')
                ax.set_xlabel('LAG [samples]')
                ax.set_ylabel('ANGLE [rad]')
                ax.set_title('Quality = %.2f'%quality)
            
            ax.plot(lag, angle, 'ow')
            ax.text(lag, angle, str(kk))
            ax.axvline(min_lag_thres, 0, 1, color='w', ls='--')
            ax.axvline(max_lag_thres, 0, 1, color='w', ls='--')
            
            if contour_select is not None:
                #### Prepare for plotting
                cont_x = np.min(LAGS) + d_lag * contour_select[:, 0]
                # Don't forget to subtract the number of rows that were added
                cont_y = np.min(ANGLES) + d_angle * (contour_select[:, 1] - MESH.shape[0] + 1)
                cont = np.column_stack((cont_x, cont_y))
                
                ### Box limits
                box = get_box([np.min(cont_x), np.min(cont_y)], lag_error, angle_error)
                [abov_box, middle_box, bott_box] = split_contour(box, [np.pi / 2, -np.pi / 2])
                
                if abov_box is not None:
                    abov_box[:, 1] = abov_box[:, 1] - np.pi
                if bott_box is not None:
                    bott_box[:, 1] = bott_box[:, 1] + np.pi
                
                sub_boxes = [abov_box, middle_box, bott_box]
                sub_boxes = [x for x in sub_boxes if x is not None]
                
                for sub_box in sub_boxes:
                    ax.plot(sub_box[:, 0], sub_box[:, 1], color='w')
                
                #### Make sure cont_y is inside -pi/2, pi/2 for plotting
                [abov_box, middle_box, bott_box] = split_contour(cont, [np.pi / 2, -np.pi / 2])
                
                if abov_box is not None:
                    abov_box[:, 1] = abov_box[:, 1] - np.pi
                if bott_box is not None:
                    bott_box[:, 1] = bott_box[:, 1] + np.pi
                
                sub_boxes = [abov_box, middle_box, bott_box]
                sub_boxes = [x for x in sub_boxes if x is not None]
                
                for sub_box in sub_boxes:
                    ax.plot(sub_box[:, 0], sub_box[:, 1], color='w')
    
    #### Return
    return (MinLambdas, ax)

def sample_array(nstart,nend,Nvalues):
    """
    Made to create an array of integers evenly spaced
    last sample in output array might be different from initial nend
    
    Inputs:
        nstart,nend:int
    """
    
#    nstart=5
#    nend=25
#    N=100
    delta=(nend-nstart)

    reste=delta%Nvalues
    inc=delta//Nvalues
    while reste!=0:
        Nvalues=Nvalues-1
        reste=delta%Nvalues
        inc=delta//Nvalues

    sam_array=np.arange(nstart,nend+1,inc).astype(int)
        
    return sam_array

def minmax2zeroone(array):
    """
    Normalize array to [0,1] based on min and max values.
    From shearwavesplit.py line 669
    
    Parameters:
    -----------
    array : ndarray
        Array to normalize
        
    Returns:
    --------
    ndarray
        Normalized array
    """
    min_value = np.min(array)
    max_value = np.max(array)
    
    array = array - min_value
    array = array / max_value
    return array


def extrema(arr, size_perc=[10, 10], flag_plot=False):
    """
    Find local minima in 2D array using morphological filtering.
    From shearwavesplit.py line 676
    
    Parameters:
    -----------
    arr : ndarray
        2D array to find minima in
    size_perc : list [col_perc, row_perc]
        Footprint size as percentage of array dimensions
    flag_plot : bool
        Whether to plot results
        
    Returns:
    --------
    list of tuples
        [(ind_row, ind_col, min_value), ...] sorted by value
    """
    foot_row = int(arr.shape[0] * size_perc[1] / 100)
    foot_col = int(arr.shape[1] * size_perc[0] / 100)
    
    neighborhood = np.ones((foot_row, foot_col), dtype=bool)
    
    local_min = (filters.minimum_filter(arr, footprint=neighborhood, mode=('wrap', 'constant')) == arr)
    
    background = (arr == 0)
    
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    
    # Remove background from local_min mask to get only peaks
    detected_minima = local_min ^ eroded_background
    
    ind_row, ind_col = np.where(detected_minima)
    min_values = arr[ind_row, ind_col]
    
    # Sort
    ind_row = ind_row[np.argsort(min_values)]
    ind_col = ind_col[np.argsort(min_values)]
    min_values = min_values[np.argsort(min_values)]
    
    # Create tuples
    min_list = list(zip(ind_row, ind_col, min_values))
    labels = [str(kk) for kk in range(len(min_list))]
    
    # Plot if asked
    if flag_plot:
        fig, ax = plt.subplots()
        ax.imshow(arr, plt.cm.get_cmap('jet'))
        ax.plot(ind_col, ind_row, '+w')
        for kk, label in enumerate(labels):
            ax.text(ind_col[kk], ind_row[kk], label)
    
    return min_list


def quality_mesh(array, threshold=0.5, mode='below'):
    """
    Estimate quality of mesh by computing ratio of samples above/below threshold.
    From shearwavesplit.py line 723
    
    Parameters:
    -----------
    array : ndarray
        Input array
    threshold : float
        Limit to compute ratio
    mode : str
        'below' or 'above'
        
    Returns:
    --------
    float
        Quality metric (1 = single spike, 0.5 = lots of noise)
    """
    # Check
    if mode not in ['below', 'above']:
        mode = 'below'
    
    # Normalize
    array = minmax2zeroone(array)
    
    # Compute
    if mode == 'below':
        n_samples = np.sum(array < threshold)
    else:
        n_samples = np.sum(array > threshold)
    
    noise = n_samples / np.size(array)
    quality = (1 - noise)
    
    return quality


def inside_contour(x, y, cont_array):
    """
    Check if point (x, y) is inside polygon defined by contour.
    From shearwavesplit.py line 345
    
    Parameters:
    -----------
    x, y : float
        Point coordinates
    cont_array : ndarray
        Contour vertices
        
    Returns:
    --------
    bool
        True if point inside contour
    """
    # Close cont_array to be sure
    if not np.all(cont_array[0] == cont_array[-1]):
        cont_array = np.vstack((cont_array, cont_array[0]))
    
    # Check
    n = cont_array.shape[0]
    inside = False
    p1x, p1y = cont_array[0, 0], cont_array[0, 1]
    for i in range(1, n):
        p2x, p2y = cont_array[i, 0], cont_array[i, 1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def get_contours(array, levels):
    """
    Extract contour paths at specified levels from 2D array.
    From shearwavesplit.py line 365
    
    Parameters:
    -----------
    array : ndarray
        2D array
    levels : list
        Contour levels
        
    Returns:
    --------
    list
        List of numpy arrays with contour vertices
    """
    # Define contours (vertices given in indexes)
    plt.ioff()
    fig, ax = plt.subplots()
    ax.imshow(array)
    contour_set = ax.contour(array, levels=levels)
    plt.close(fig)
    plt.ion()
    
    # Get contours
    contour_list = []
    # QuadContourSet.collections is deprecated as of Python 3.8
    for p in contour_set.allsegs[0]:
        contour_list.append(p)
    
    return contour_list


def select_contour(x, y, contour_list):
    """
    Select contour that contains point (x, y).
    From shearwavesplit.py line 406
    
    Parameters:
    -----------
    x, y : float
        Point coordinates
    contour_list : list
        List of contour arrays
        
    Returns:
    --------
    ndarray or None
        Selected contour array
    """
    sel_array = None
    for cont_array in contour_list:
        inside = inside_contour(x, y, cont_array)
        if inside is True:
            sel_array = cont_array
            break
    
    return sel_array


def box_contour(data_array):
    """
    Measure size of box contouring the data array.
    From shearwavesplit.py line 388
    
    Parameters:
    -----------
    data_array : ndarray
        Nx2 array
        
    Returns:
    --------
    width, height : float
        Box dimensions
    """
    if data_array is None:
        width = 999
        height = 999
        return width, height
    width = np.max(data_array[:, 0]) - np.min(data_array[:, 0])
    height = np.max(data_array[:, 1]) - np.min(data_array[:, 1])
    
    return width, height


def get_valcontour(array, ind_row, ind_col, cont_step, ax=None, flag_plot=False):
    """
    Compute contour at array[ind_row, ind_col] + cont_step.
    From shearwavesplit.py line 747
    
    Parameters:
    -----------
    array : ndarray
        Input M x N array
    ind_row, ind_col : int
        Indices to evaluate contour
    cont_step : float
        Contour step to add to value
    ax : matplotlib axes
        Axes to plot on
    flag_plot : bool
        Whether to plot
        
    Returns:
    --------
    tuple
        (contour_select, width, height, (ind_row, ind_col, array_value), box, ax)
    """
    array_value = array[ind_row, ind_col]
    
    # Select contour associated to single min
    contour_list = get_contours(array, [array_value + cont_step])
    contour_select = select_contour(ind_col, ind_row, contour_list)
    
    # Get box
    width, height = box_contour(contour_select)
    box = None
    
    if flag_plot:
        if ax is None:
            fig, ax = plt.subplots()
            ax.imshow(array, aspect=0.1, cmap=plt.cm.get_cmap('jet'))
        ax.plot(ind_col, ind_row, 'ow')
        
        if contour_select is not None:
            ax.plot(contour_select[:, 0], contour_select[:, 1], color='w')
            
            # Plot box
            box = get_box([np.min(contour_select[:, 0]), np.min(contour_select[:, 1])], width, height)
            ax.plot(box[:, 0], box[:, 1], color='w')
    
    return (contour_select, width, height, (ind_row, ind_col, array_value), box, ax)


def get_box(lowerleft, width, height):
    """
    Create a box given width, height, and lower left corner.
    From shearwavesplit.py line 783
    
    Parameters:
    -----------
    lowerleft : list
        [x, y] coordinates of lower left corner
    width, height : float
        Box dimensions
        
    Returns:
    --------
    ndarray
        5x2 array with all vertices
    """
    Ac = [lowerleft[0], lowerleft[1]]
    Bc = [Ac[0] + width, Ac[1]]
    Cc = [Bc[0], Bc[1] + height]
    Dc = [Cc[0] - width, Cc[1]]
    
    box = np.array([Ac, Bc, Cc, Dc, Ac])
    
    return box


def split_contour(cont, y_lim_list, flag_plot=False):
    """
    Split contour into multiple subcontours where it crosses y_lim values.
    From shearwavesplit.py line 798
    
    Parameters:
    -----------
    cont : ndarray
        Nx2 contour
    y_lim_list : list or float
        Horizontal separation limits
    flag_plot : bool
        Whether to plot
        
    Returns:
    --------
    list
        List of contour arrays or None
    """
    # Check
    if not isinstance(y_lim_list, list):
        y_lim_list = [y_lim_list]
    else:
        y_lim_list.sort(reverse=True)
    
    # Split
    cont_list = []
    for y_lim in y_lim_list:
        cont = add_y(cont, y_lim)
        (top, bottom) = split_y(cont, y_lim)
        cont_list.append(top)
        cont = bottom
    
    cont_list.append(bottom)
    
    # Plot if asked
    if flag_plot:
        fig, ax = plt.subplots()
        ax.axis('equal')
        
        for cont in cont_list:
            if cont is None:
                continue
            ax.plot(cont[:, 0], cont[:, 1], 'k')
    
    return cont_list


def add_y(cont, y_lim):
    """
    Add interpolated (x, y) points where contour crosses y_lim.
    From shearwavesplit.py line 825
    
    Parameters:
    -----------
    cont : ndarray
        [x, y] contour
    y_lim : float
        Y limit for splitting
        
    Returns:
    --------
    ndarray
        Contour with added points
    """
    if cont is None:
        return None
    
    # Find indexes where contour crosses y_lim
    x = cont[:, 0]
    y = cont[:, 1]
    y_diff = y - y_lim
    
    ind_sel = np.where(np.diff(np.sign(y_diff)) != 0)[0]
    ind_next = ind_sel + 1
    
    ind_pairs = list(zip(ind_sel, ind_next))
    
    # Interpolate and add points to contour
    x_new = []
    y_new = []
    k = 0
    for ind_pair in ind_pairs:
        ind_pair = list(ind_pair)
        x_bord = x[ind_pair]
        y_bord = y[ind_pair]
        x_bord = x_bord[np.argsort(y_bord)]
        y_bord = y_bord[np.argsort(y_bord)]
        
        x_lim = np.interp(y_lim, y_bord, x_bord)
        
        x_new.extend(list(x[k:ind_pair[0] + 1]))
        x_new = x_new + [x_lim]
        y_new.extend(list(y[k:ind_pair[0] + 1]))
        y_new = y_new + [y_lim]
        
        k = ind_pair[1]
    
    x_new.extend(list(x[k:]))
    y_new.extend(list(y[k:]))
    
    x_new = np.array(x_new)
    y_new = np.array(y_new)
    
    return np.column_stack((x_new, y_new))


def split_y(cont, y_lim, flag_plot=False):
    """
    Split contour into 2 subcontours above and below y_lim.
    From shearwavesplit.py line 872
    
    Parameters:
    -----------
    cont : ndarray
        [x, y] contour
    y_lim : float
        Y limit
    flag_plot : bool
        Whether to plot
        
    Returns:
    --------
    tuple
        (cont_above, cont_below), one may be None if no crossing
    """
    # Special case
    if cont is None:
        return (None, None)
    
    # Check position
    x = cont[:, 0]
    y = cont[:, 1]
    y_abov = copy.copy(y)
    y_abov = y[y >= y_lim]
    x_abov = x[y >= y_lim]
    
    y_bott = copy.copy(y)
    y_bott = y[y <= y_lim]
    x_bott = x[y <= y_lim]
    
    # Return if no crossing
    if x_abov.size == 0:
        return (None, cont)
    if x_bott.size == 0:
        return (cont, None)
    
    x_abov = np.hstack((x_abov, x_abov[0]))
    y_abov = np.hstack((y_abov, y_abov[0]))
    
    x_bott = np.hstack((x_bott, x_bott[0]))
    y_bott = np.hstack((y_bott, y_bott[0]))
    
    cont_abov = np.column_stack((x_abov, y_abov))
    cont_bott = np.column_stack((x_bott, y_bott))
    
    # Plot if asked
    if flag_plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, 'k', lw=6)
        ax.plot(x_abov, y_abov, 'r', lw=3)
        ax.plot(x_bott, y_bott, 'y', lw=1)
    
    return (cont_abov, cont_bott)


def select_minima(min_list, min_thres, min_numbers):
    """
    Select lowest minima below threshold up to maximum number.
    From shearwavesplit.py line 913
    
    Parameters:
    -----------
    min_list : list of tuples
        [(ind_row, ind_col, minimum_value), ...]
    min_thres : float
        Keep minima below this value
    min_numbers : int
        Keep maximum of this many minima
        
    Returns:
    --------
    list of tuples
        Selected tuples that fulfill criteria
    """
    min_list = np.array(min_list)
    
    min_list = min_list[min_list[:, 2].argsort()]  # From lowest to highest
    min_list = min_list[min_list[:, 2] <= min_thres]
    min_list = min_list[0:min_numbers]
    
    min_list = [(int(x[0]), int(x[1]), x[-1]) for x in min_list]
    
    return min_list


#def rms_MinLambdas_baillard(xy_array, sw1, sw2, MinLambdas, mode='norm'):
    """
    Compute RMS between unsplit X and Y components.
    Sort MinLambdas by RMS quality (lower is better).
    From shearwavesplit.py line 1115 (rms_MinLambdas)
    
    Parameters:
    -----------
    xy_array : ndarray [N x 2]
        Original waveforms
    sw1 : int
        Start of analysis window
    sw2 : int
        End of analysis window
    MinLambdas : list of MinLambda
        Splitting solutions
    mode : str
        'norm' for normalized RMS, 'raw' for raw RMS
        
    Returns:
    --------
    list of MinLambda
        Sorted by RMS (best first)
    """
    print(f"    Computing RMS quality metrics...")
    
    rmss = []
    for ml in MinLambdas:
        # Apply unsplitting correction
        xy_unsplit, _ = unsplit_baillard(xy_array, ml.lag, ml.angle)
        xy_cut = xy_unsplit[sw1:sw2]
        
        # Compute RMS between X and Y
        if mode == 'norm':
            # Normalize each component to unit amplitude
            x_max = np.max(np.abs(xy_cut[:, 0]))
            y_max = np.max(np.abs(xy_cut[:, 1]))
            
            if x_max > 0 and y_max > 0:
                x_norm = xy_cut[:, 0] / x_max
                y_norm = xy_cut[:, 1] / y_max
                rms = np.sqrt(np.mean((x_norm - y_norm)**2))
            else:
                rms = np.inf
        else:
            # Raw RMS
            rms = np.sqrt(np.mean((xy_cut[:, 0] - xy_cut[:, 1])**2))
        
        ml.rms = rms
        rmss.append(rms)
    
    # Sort by RMS (ascending - lower is better)
    sorted_idx = np.argsort(rmss)
    MinLambdas = [MinLambdas[i] for i in sorted_idx]
    
    print(f"    RMS range: {min(rmss):.4f} to {max(rmss):.4f}")
    
    return MinLambdas

def rms_MinLambdas_baillard(xy_array,sw1,sw2,MinLambdas,mode='norm',flag_plot=False):
    
    """
    Function made to compute the RMS in between the unsplit x and y to assess
    quality of correction. The MinLambdas are then sorted based on minimum RMS.
    The lambda with minimum RMS is first. We return the sorted RMS
    
    Input
    -----
        xy_array: np.array
        sw1,sw2: float: samples to cut the signal
        MinLambdas: list: list of classes 
        mode:str: 'norm' to specfify how to compute RMS
        flag_plot: Bool
        
    Output
    ------
        MinLambdas: list
        
    """

    for kk in range(len(MinLambdas)):
        lag=MinLambdas[kk].lag
        angle=MinLambdas[kk].angle
        xy_unsplit,_= unsplit_baillard(xy_array,lag,angle,flag_plot=flag_plot)
        xy_unsplit_cut=xy_unsplit[sw1:sw2,:]
        rms_1=get_rms(xy_unsplit_cut[:,0],xy_unsplit_cut[:,1],mode=mode)
        rms_2=get_rms(xy_unsplit_cut[:,0],-xy_unsplit_cut[:,1],mode=mode)
        
        min_rms=np.min((rms_1,rms_2))
        
        MinLambdas[kk].rms=min_rms
        
        if flag_plot:
            fig=plt.gcf()
            ax=fig.get_axes()
            for i in range(len(ax)):
                ylim=ax[i].get_ylim()
                ax[i].vlines(sw1,ylim[0],ylim[1])
                ax[i].vlines(sw2,ylim[0],ylim[1])
                if i==1:
                    ax[i].text(0.1,0.1,'RMS=%f'%rms_1, horizontalalignment='center',
                          verticalalignment='center', transform=ax[i].transAxes)
                elif i==2:
                    ax[i].text(0.1,0.1,'RMS=%f'%rms_2, horizontalalignment='center',
                          verticalalignment='center', transform=ax[i].transAxes)
            
        
    ### Sort MinLambdas based on RMS rather than Lambda2 min
        
    rmss=[MinLambda.rms for MinLambda in MinLambdas]
    ind_rmss=np.argsort(rmss)
    
    MinLambdas=[MinLambdas[kk] for kk in ind_rmss] 
    
    return MinLambdas

def get_rms(x,y,mode='norm'):
    """
    Compute RMS of two signals to check similarity
    It is possible to normalize it by the maximum range or not
    """

    rms=np.sqrt(np.mean((x-y)**2))

    # Normalize if asked
    
    if mode=='norm':
        cat_array=np.hstack((x,y))
        rms=rms/(np.max(cat_array)-np.min(cat_array))
  
    return rms


def estimate_dominant_frequency_baillard(xy_array, sampling_rate, sw1, sw2,
                                        freq_min=4.0, freq_max=50.0):
    """
    Estimate dominant frequency using power spectral density.
    Based on Baillard's approach in sws_methods.py
    
    Parameters:
    -----------
    xy_array : ndarray [N x 2]
        Two-component data
    sampling_rate : float
        Sampling rate in Hz
    sw1 : int
        Start of analysis window
    sw2 : int
        End of analysis window
    freq_min : float
        Minimum frequency to consider (Hz)
    freq_max : float
        Maximum frequency to consider (Hz)
        
    Returns:
    --------
    float
        Dominant period in seconds
    """
    
    # Extract window
    xy_window = xy_array[sw1:sw2]
    
    # Compute PSD for both components
    freqs_x, psd_x = signal.welch(xy_window[:, 0], fs=sampling_rate, nperseg=min(256, len(xy_window)//2))
    freqs_y, psd_y = signal.welch(xy_window[:, 1], fs=sampling_rate, nperseg=min(256, len(xy_window)//2))
    
    # Average PSD
    psd_avg = (psd_x + psd_y) / 2
    
    # Find dominant frequency within range
    freq_mask = (freqs_x >= freq_min) & (freqs_x <= freq_max)
    if not np.any(freq_mask):
        # Fallback if no frequencies in range
        return 0.1
    
    psd_in_range = psd_avg[freq_mask]
    freqs_in_range = freqs_x[freq_mask]
    
    dominant_freq = freqs_in_range[np.argmax(psd_in_range)]
    dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.1
    
    return dominant_period

def get_dominant_period_baillard(data, fs, method='welch', nfft=256, num_wind=2, flag_plot=False):
    """
    Function made to compute the dominant period of a 1D np.array using either
    a classical FFT or a multitaper methods
    
    Output
    ------
        dom_period: float: dominant period in samples (can be float)
        dom_freq: float: dominant frequency
    
    """
    
    ### Compute
    
    if method=='welch':
        freq,spec=signal.welch(data, fs=fs,nperseg=len(data)/num_wind,scaling='density',nfft=nfft)
    #elif method=='multitaper' or method=='mtspec':
    #    if not MULTITAPER_AVAILABLE:
    #        raise ImportError("multitaper package is not installed. Install with: pip install multitaper")
        # multitaper package usage: MTSpec(data, nw, k, dt)
        # nw = time-bandwidth product (typically 2-4), k = number of tapers (typically 2*nw - 1)
    #    nw = num_wind if num_wind > 1 else 2.5
    #    k = int(2 * nw - 1)
    #    dt = 1.0 / fs
    #    mtspec_obj = MTSpec(data, nw=nw, k=k, dt=dt, nfft=nfft)
    #    spec = mtspec_obj.rspec()  # Return spectrum
    #    freq = mtspec_obj.freq     # Return frequencies
    #elif method=='classic':
    #    ([freq,spec],_)=spectrum(data,1/fs,nfft=nfft,flag_plot=False)
        
    ### Freq to period
      
    with np.errstate(divide='ignore'):
        period=1/freq*fs
    
    ### Get Dominant period

    dom_period=period[np.argmax(spec)]  
    dom_freq=1/dom_period*fs
    
    ### Plot if asked
    
    if flag_plot:
        x=np.arange(0,len(data))
        alpha=2*np.pi/dom_period
        sin=np.max(data)*np.sin(alpha*x)
            
        fig,ax=plt.subplots(3,1)
        
        ax[0].plot(x,data,color='k',ls='-',lw=1)
        ax[0].plot(x,sin,color='r',ls='--',lw=1)
        
        ax[1].plot(period,spec,color='k',ls='-',lw=1)
        ax[1].axvline(dom_period,color='r',ls='--')
        ax[1].text(0.5, 0.9,'Dom T=%.1f samples'%dom_period, horizontalalignment='center',
                  verticalalignment='center', transform=ax[1].transAxes)
        
        ax[2].plot(freq,spec,color='k',ls='-',lw=1)
        ax[2].axvline(dom_freq,color='r',ls='--')
        ax[2].text(0.5, 0.9,'Dom F=%.1f Hz'%dom_freq, horizontalalignment='center',
                  verticalalignment='center', transform=ax[2].transAxes)
        
    return (dom_period,dom_freq)


def calculate_baillard_window_params(xy_array, s_time, p_time, 
                                    trace_start_time, sampling_rate,
                                    s_window_base=[0.02, 0.3],
                                    min_lag_base=0, max_lag_base=60,
                                    flag_adapt_window=True,
                                    flag_adapt_maxlag=True):
    """
    Calculate S-window and max_lag using Baillard's adaptive method.
    From run_sws.py lines 177-186
    
    Parameters:
    -----------
    xy_array : ndarray [N x 2]
        Two-component waveform data
    s_time : UTCDateTime
        S-wave arrival time
    p_time : UTCDateTime
        P-wave arrival time
    trace_start_time : UTCDateTime
        Trace start time
    sampling_rate : float
        Sampling rate in Hz
    s_window_base : list [before, after]
        Base S-window in seconds
    min_lag_base : int
        Base minimum lag in samples
    max_lag_base : int
        Base maximum lag in samples
    flag_adapt_window : bool
        Whether to adapt window based on dominant period
    flag_adapt_maxlag : bool
        Whether to adapt max_lag based on dominant period
        
    Returns:
    --------
    sw1 : int
        S-window start sample
    sw2 : int
        S-window end sample
    min_lag : int
        Minimum lag in samples
    max_lag : int
        Maximum lag in samples
    T_dom : float
        Dominant period in seconds
    """
    # Calculate sample indices
    fs_window = [0.1, 0.3] 
    fs_window_time=[s_time-fs_window[0],s_time+fs_window[1]] 

    fs_w1=int(round((fs_window_time[0]-trace_start_time)*sampling_rate))
    fs_w2=int(round((fs_window_time[1]-trace_start_time)*sampling_rate))

    s_samples = int(round((s_time - trace_start_time) * sampling_rate))
    p_samples = int(round((p_time - trace_start_time) * sampling_rate))
    mid_samples = int(round(p_samples + (s_samples - p_samples) / 2))

    # Initial window for dominant frequency estimation
    s_window_time = [s_time - s_window_base[0], s_time + s_window_base[1]]
    sw1_init = int(round((s_window_time[0] - trace_start_time) * sampling_rate))
    sw2_init = int(round((s_window_time[1] - trace_start_time) * sampling_rate))
    
    sw1=int(round((s_window_time[0]-trace_start_time)*sampling_rate))
    
    # Ensure sw1 doesn't contaminate with P-wave
    if sw1_init < mid_samples:
        sw1_init = mid_samples

    ### Make sure left side of the window is bigger than P+(S-P)/2
    if fs_w1<mid_samples:
        fs_w1=mid_samples

    ### Cut the data between fs_w1 and fs_w2

    xy_array_dom=xy_array[fs_w1:fs_w2,:]

    ### Get dominant period on X and Y (given in samples)

    (dom_period_x,dom_freq_x)=get_dominant_period_baillard(xy_array_dom[:,0],sampling_rate,flag_plot=False)
    (dom_period_y,dom_freq_y)=get_dominant_period_baillard(xy_array_dom[:,1],sampling_rate,flag_plot=False)

    T_dom=np.mean([dom_period_x,dom_period_y]) # Take the mean dominant period

    ### Adapt window size to perform splitting and max lag allowed

    if flag_adapt_window:
        sw2=int(round(sw1+2*T_dom)) # Check Wuestfeld et al., 2010
    else:
        sw2=int(round((s_window_time[1]-trace_start_time)*sampling_rate))
        
    if flag_adapt_maxlag:
        max_lag=int(round(T_dom))
    
    # Estimate dominant frequency
   #T_dom = estimate_dominant_frequency_baillard(xy_array, sampling_rate, 
    #                                            sw1_init, sw2_init)
    
    print(f"    Dominant period: {T_dom:.3f}s")
    
    # Adapt window if requested (Baillard uses T*0.5 before, T*2 after)
    #if flag_adapt_window:
    #    s_window = [T_dom * 0.5, T_dom * 2.0]
    #    print(f"    Adapted S-window: [{s_window[0]:.3f}s, {s_window[1]:.3f}s]")
    #else:
    #    s_window = s_window_base
    
    # Calculate final window boundaries
    #s_window_time = [s_time - s_window[0], s_time + s_window[1]]
    #sw1 = int(round((s_window_time[0] - trace_start_time) * sampling_rate))
    #sw2 = int(round((s_window_time[1] - trace_start_time) * sampling_rate))
    
    # Ensure sw1 doesn't contaminate with P-wave
    #if sw1 < mid_samples:
    #    sw1 = mid_samples
    #    print(f"    Adjusted sw1 to avoid P-wave contamination")
    
    # Adapt max_lag if requested (Baillard uses 1 dominant period)
    min_lag = min_lag_base
   #if flag_adapt_maxlag:
    #    max_lag = int(T_dom * sampling_rate)
    #    print(f"    Adapted max_lag: {max_lag} samples ({max_lag/sampling_rate:.3f}s)")
    #else:
    #    max_lag = max_lag_base
    
    return sw1, sw2, min_lag, max_lag, T_dom


def _create_baillard_axes_layout():
    """
    Create the 6-panel subplot layout used for Baillard SWS diagnostic plots.
    
    Layout:
    - Upper tier: 3 rows for waveforms (initial, rotated/unsplit, unsplit-inverted)
    - Lower tier: 3 columns (lambda2 surface, particle motion original, particle motion unsplit)
    
    Returns:
    --------
    list
        [ax_ini, ax_un1, ax_un2, ax_lambda2, ax_polar, ax_polar_un]
    """
    from matplotlib import gridspec
    
    # Upper tier: 3x3 GridSpec for waveform plots (3 rows, shared x/y axes)
    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(left=0.1, right=0.8, top=0.92, bottom=0.55, hspace=0)
    
    ax_ini = plt.subplot(gs1[0, :])      # Row 0: Initial waveforms
    ax_un1 = plt.subplot(gs1[1, :], sharex=ax_ini, sharey=ax_ini)  # Row 1: Rotated waveforms
    ax_un2 = plt.subplot(gs1[2, :], sharex=ax_ini, sharey=ax_ini)  # Row 2: Unsplit waveforms
    
    # Lower tier: 2x6 GridSpec for analysis plots
    gs = gridspec.GridSpec(2, 6)
    gs.update(wspace=3.5, hspace=0.3, top=0.45, left=0.1, bottom=0.05)
    
    ax_lambda2 = plt.subplot(gs[0:, 0:2])    # Cols 0-1: Lambda2 contour map
    ax_polar = plt.subplot(gs[0:, 2:4])      # Cols 2-3: Original particle motion
    ax_polar_un = plt.subplot(gs[0:, 4:6])   # Cols 4-5: Unsplit particle motion
    
    return [ax_ini, ax_un1, ax_un2, ax_lambda2, ax_polar, ax_polar_un]


def _plot_lambda2_surface(ax, LAMBDA2, LAMBDA1, LAGS, ANGLES, MinLambdas, 
                         min_lag, max_lag, min_thres=0.5, zoom_factor=[2, 6]):
    """
    Plot Lambda2 eigenvalue surface with contours and marked minima.
    Replicates Baillard's process_LAMBDAS plotting functionality.
    
    Parameters:
    -----------
    ax : matplotlib axes
        Axes to plot on
    LAMBDA2 : ndarray [Nlags x Nangles]
        Smaller eigenvalue grid
    LAMBDA1 : ndarray [Nlags x Nangles]
        Larger eigenvalue grid
    LAGS : ndarray [Nlags x Nangles]
        Lag values grid
    ANGLES : ndarray [Nlags x Nangles]
        Angle values grid
    MinLambdas : list
        List of MinLambda objects with minima
    min_lag : int
        Minimum lag threshold
    max_lag : int
        Maximum lag threshold
    min_thres : float
        Threshold for contours
    zoom_factor : list [rows, cols]
        Interpolation factor for smoothing
    """
    from scipy import ndimage, interpolate
    
    # Normalize LAMBDA2 to [0, 1]
    #LAMBDA2_norm = LAMBDA2 / np.max(LAMBDA2)
    
    # Zoom (interpolate) for smooth contours
    MESH = minmax2zeroone(LAMBDA2)
    
    # Get extent for plotting
    extent_lag = [np.min(LAGS), np.max(LAGS)]
    extent_angle = [np.min(ANGLES), np.max(ANGLES)]
    extent = extent_lag + extent_angle
    
    if ax is None: ### Plot things just once in the loop
        fig,ax=plt.subplots()
    if kk==1:
        ax.imshow(MESH,extent=extent_lag+extent_angle,origin='lower',cmap=plt.cm.get_cmap('jet'),aspect='auto') # GRID
        CS=ax.contour(MESH,extent=extent_lag+extent_angle,origin='lower',levels=[min_thres],colors='w',linestyles=':') # CONTOURS
        ax.clabel(CS, [min_thres],fmt='%.2f')
        ax.set_xlabel('LAG [samples]')
        ax.set_ylabel('ANGLE [rad]')
        ax.set_title('Quality = %.2f'%quality)
        
    ax.plot(lag,angle,'ow')
    ax.text(lag,angle,str(kk))
    ax.axvline(min_lag_thres,0,1,color='w',ls='--')
    ax.axvline(max_lag_thres,0,1,color='w',ls='--')
    
    if contour_select is not None:
    
        #### Prepare for plotting
    
        cont_x=np.min(LAGS)+d_lag*contour_select[:,0]
        cont_y=np.min(ANGLES)+d_angle*(contour_select[:,1]-MESH.shape[0]+1) # Don't forget to substract the number of column that were added
        cont=np.column_stack((cont_x,cont_y))
        
        ### Box limits
        
        box=get_box([np.min(cont_x),np.min(cont_y)],lag_error,angle_error)
        [abov_box,middle_box,bott_box]=split_contour(box,[np.pi/2,-np.pi/2])

        if abov_box is not None:
            abov_box[:,1]=abov_box[:,1]-np.pi
        if bott_box is not None:
            bott_box[:,1]=bott_box[:,1]+np.pi
        
        sub_boxes=[abov_box,middle_box,bott_box]
        sub_boxes=[x for x in sub_boxes if x is not None]
        
        for sub_box in sub_boxes:
            ax.plot(sub_box[:,0],sub_box[:,1],color='w')
    
#                #### Make sure cont_y is inside -pi/2 , pi/2 for plotting
#                
        [abov_box,middle_box,bott_box]=split_contour(cont,[np.pi/2,-np.pi/2])

        if abov_box is not None:
            abov_box[:,1]=abov_box[:,1]-np.pi
        if bott_box is not None:
            bott_box[:,1]=bott_box[:,1]+np.pi
        
        sub_boxes=[abov_box,middle_box,bott_box]
        sub_boxes=[x for x in sub_boxes if x is not None]
        
        for sub_box in sub_boxes:
            ax.plot(sub_box[:,0],sub_box[:,1],color='w')


def _plot_baillard_waveforms(ax_list, xy_array, xy_unsplitted, sw1, sw2, 
                             s_samples, lag, angle, rms, sampling_rate):
    """
    Plot 3 rows of waveforms: initial, unsplit, unsplit with inverted Y.
    Replicates Baillard's unsplit() plotting and run_sws.py waveform display.
    
    Parameters:
    -----------
    ax_list : list [ax_ini, ax_un1, ax_un2]
        Three axes for waveform plots
    xy_array : ndarray [N x 2]
        Original waveform data [X, Y]
    xy_unsplitted : ndarray [N x 2]
        Unsplit waveform data [X, Y]
    sw1 : int
        S-window start sample
    sw2 : int
        S-window end sample
    s_samples : int
        S-arrival sample index
    lag : int
        Splitting lag in samples
    angle : float
        Splitting angle in radians
    rms : float
        RMS quality metric
    sampling_rate : float
        Sampling rate in Hz
    """
    import matplotlib.patches as mpatches
    
    ax_ini, ax_un1, ax_un2 = ax_list
    
    # Plot initial waveforms (ax_ini)
    ax_ini.plot(xy_array[:, 0], 'k', label='X', lw=1)
    ax_ini.plot(xy_array[:, 1], '--r', label='Y', lw=1)
    ax_ini.legend(loc=1, fontsize=8)
    ax_ini.text(0.03, 0.85, 'Initial', fontweight='bold', 
                transform=ax_ini.transAxes, fontsize=10)
    
    # Add S-window rectangle (light gray background)
    y_bottom, y_top = ax_ini.get_ylim()
    height = y_top - y_bottom
    rec = mpatches.Rectangle((sw1, y_bottom), width=(sw2 - sw1), height=height,
                             facecolor='0.9', zorder=-1)
    ax_ini.add_patch(rec)
    
    # Add S label
    #plt.text(sw1 + (sw2 - sw1) * 0.05, y_top - height * 0.05, 'S',
    #         weight='bold', horizontalalignment='left', verticalalignment='top',
    #         fontsize=10)
    
    # Add vertical line at S-pick
    ax_ini.axvline(x=s_samples, linestyle='-', color='k', lw=1, zorder=0)
    
    # Title with lag and angle
    ax_ini.set_title(f'Lag={lag:.1f}, Angle={angle:.2f} rad', 
                     fontweight='bold', fontsize=11)
    
    ax_ini.set_ylim(-100, 100)
    
    # Plot unsplit waveforms (ax_un1)
    ax_un1.plot(xy_unsplitted[:, 0], 'k', label='X', lw=1)
    ax_un1.plot(xy_unsplitted[:, 1], '--r', label='Y', lw=1)
    ax_un1.legend(loc=1, fontsize=8)
    ax_un1.text(0.03, 0.85, 'Unsplit', fontweight='bold', 
                transform=ax_un1.transAxes, fontsize=10)
    
    ax_un1.set_ylim(-100, 100)
    
    # Plot unsplit with inverted Y (ax_un2)
    ax_un2.plot(xy_unsplitted[:, 0], 'k', label='X', lw=1)
    ax_un2.plot(-xy_unsplitted[:, 1], '--r', label='-Y', lw=1)
    ax_un2.legend(loc=1, fontsize=8)
    ax_un2.text(0.03, 0.85, 'Unsplit', fontweight='bold', 
                transform=ax_un2.transAxes, fontsize=10)
    
    # Add RMS text to bottom plot
    ax_un2.text(0.1, 0.1, f'RMS = {rms:.3f}', transform=ax_un2.transAxes,
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax_un2.set_ylim(-100, 100)
    
    # Zoom to S-window region with buffer
    #buffer = max(100, int((sw2 - sw1) * 0.3))
    #for ax in [ax_ini, ax_un1, ax_un2]:
    #    ax.set_xlim([sw1 - buffer, sw2 + buffer])
    
    ax_un2.set_xlabel('Sample Index', fontweight='bold')


def _plot_particle_motion_baillard(ax, xy_data, xlabel='X', ylabel='Y', title='',
                                   num_points=500, cmap='coolwarm'):
    """
    Plot particle motion diagram with time-based color coding.
    Replicates Baillard's plot_particle_motion() from plotwaveform.py.
    
    Parameters:
    -----------
    ax : matplotlib axes
        Axes to plot on
    xy_data : ndarray [N x 2]
        Particle motion data [X, Y]
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    num_points : int
        Number of interpolation points for smooth curve
    cmap : str
        Colormap name
    """
    from scipy import interpolate
    
    x_array = xy_data[:, 0]
    y_array = xy_data[:, 1]
    
    # Create time array for color coding
    time_array = np.arange(len(x_array))
    
    # Interpolate to smooth curve
    if len(x_array) > 3:
        # Calculate cumulative distance along path
        dx = np.diff(x_array)
        dy = np.diff(y_array)
        ds = np.sqrt(dx**2 + dy**2)
        ds_cumsum = np.concatenate(([0], np.cumsum(ds)))
        
        # Create interpolation functions
        try:
            f_x = interpolate.interp1d(ds_cumsum, x_array, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
            f_y = interpolate.interp1d(ds_cumsum, y_array, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
            
            # Generate smooth curve
            ds_new = np.linspace(0, ds_cumsum[-1], num_points)
            x_graph = f_x(ds_new)
            y_graph = f_y(ds_new)
            
            # Interpolate time values
            time_out = np.interp(ds_new, ds_cumsum, time_array)
            
        except:
            # Fallback to original data
            x_graph = x_array
            y_graph = y_array
            time_out = time_array
    else:
        x_graph = x_array
        y_graph = y_array
        time_out = time_array
    
    # Plot crosshairs
    ax.axvline(0, linestyle=':', color='k', lw=1, zorder=0)
    ax.axhline(0, linestyle=':', color='k', lw=1, zorder=0)
    
    # Plot path as black line with red outline
    ax.plot(x_graph, y_graph, '-r', lw=3, zorder=1, alpha=0.8)  # Red outline
    ax.plot(x_graph, y_graph, '-k', lw=1.5, zorder=2)  # Black path
    
    # Add direction arrow at center
    if len(x_graph) > 10:
        ind_clos = len(x_graph) // 2
        # Calculate direction vector
        ind_before = max(0, ind_clos - 5)
        ind_after = min(len(x_graph) - 1, ind_clos + 5)
        dx = x_graph[ind_after] - x_graph[ind_before]
        dy = y_graph[ind_after] - y_graph[ind_before]
        
        # Normalize
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0:
            dx /= mag
            dy /= mag
            # Scale arrow
            scale = (np.max(x_graph) - np.min(x_graph)) * 0.1
            ax.quiver(x_graph[ind_clos], y_graph[ind_clos], 
                     dx * scale, dy * scale,
                     facecolor='k', edgecolor='none', 
                     scale=1, scale_units='xy', width=0.01, zorder=3)
        
        # Mark center point
        ax.plot(x_graph[ind_clos], y_graph[ind_clos], 'ok', markersize=4, zorder=3)
    
    # Set equal aspect ratio with 1.1x magnification
    ax.set_aspect('equal', 'box')
    mag = 1.1
    max_range = max(np.max(np.abs(x_graph)), np.max(np.abs(y_graph))) * mag
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    
    # Labels
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=10)


def plot_baillard_splitting_results(xy_array, LAMBDA1, LAMBDA2, LAGS, ANGLES, 
                                    MinLambdas, sw1, sw2, s_samples, 
                                    min_lag, max_lag, sampling_rate,
                                    solution_index=0,
                                    station_name='', event_id=None,
                                    output_dir=None):
    """
    Create Baillard-style 6-panel diagnostic plot for shear-wave splitting.
    
    Replicates Christian Baillard's plotting workflow from run_sws.py:
    - 3 waveform panels (initial, unsplit, unsplit-inverted)
    - Lambda2 contour surface with marked minima
    - 2 particle motion diagrams (original, unsplit)
    
    Parameters:
    -----------
    xy_array : ndarray [N x 2]
        Original waveform data [X, Y]
    LAMBDA1 : ndarray [Nlags x Nangles]
        Larger eigenvalue grid
    LAMBDA2 : ndarray [Nlags x Nangles]
        Smaller eigenvalue grid
    LAGS : ndarray [Nlags x Nangles]
        Lag values grid
    ANGLES : ndarray [Nlags x Nangles]
        Angle values grid
    MinLambdas : list
        List of MinLambda objects with splitting solutions
    sw1 : int
        S-window start sample
    sw2 : int
        S-window end sample
    s_samples : int
        S-arrival sample index
    min_lag : int
        Minimum lag threshold
    max_lag : int
        Maximum lag threshold
    sampling_rate : float
        Sampling rate in Hz
    solution_index : int
        Which solution to plot (0 = best)
    station_name : str
        Station name for title
    event_id : int or str
        Event ID for filename
    output_dir : str or None
        Directory to save figure (if None, display interactively)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Get solution to plot
    if not MinLambdas or solution_index >= len(MinLambdas):
        print(f"Warning: Solution index {solution_index} out of range, using 0")
        solution_index = 0
    
    ml = MinLambdas[solution_index]
    
    # Create figure and axes layout
    fig = plt.figure(figsize=[8.1, 7.8])
    ax_list = _create_baillard_axes_layout()
    [ax_ini, ax_un1, ax_un2, ax_lambda2, ax_polar, ax_polar_un] = ax_list
    
    # 1. Plot Lambda2 surface
    #_plot_lambda2_surface(ax_lambda2, LAMBDA2, LAMBDA1, LAGS, ANGLES, 
    #                     MinLambdas, min_lag, max_lag, min_thres=0.5)
    
    _, ax_lambda2 = process_LAMBDAS_baillard(LAMBDA2, LAMBDA1, LAGS, ANGLES, 
                                            min_lag_thres=min_lag, max_lag_thres=max_lag,
                                            min_thres=0.5, min_numbers=2,
                                            cont_step=0.01, quality_thres=0.2, zoom_factor=[2, 6],
                                            flag_plot=True, ax=ax_lambda2)
    
    # 2. Unsplit waveforms
    xy_unsplitted, _ = unsplit_baillard(xy_array, ml.lag, ml.angle)
    
    # 3. Plot waveforms
    _plot_baillard_waveforms([ax_ini, ax_un1, ax_un2], xy_array, xy_unsplitted,
                            sw1, sw2, s_samples, ml.lag, ml.angle, ml.rms,
                            sampling_rate)
    
    # 4. Plot particle motion diagrams
    # Cut to S-window
    xy_array_cut = xy_array[sw1:sw2, :]
    xy_unsplit_cut = xy_unsplitted[sw1:sw2, :]
    
    _plot_particle_motion_baillard(ax_polar, xy_array_cut, 
                                   xlabel='X', ylabel='Y', title='Initial')
    _plot_particle_motion_baillard(ax_polar_un, xy_unsplit_cut,
                                   xlabel='X', ylabel='Y', title='Unsplit')
    
    # Add overall title
    if station_name:
        title = f'Station: {station_name}'
        if event_id is not None:
            title += f' | Event: {event_id}'
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    
    # Use explicit spacing instead of tight_layout for consistent sizing
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95)
    
    # Save or display
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"{station_name}_event{event_id}_splitting.png"
        filepath = output_path / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"    Saved plot to: {filepath}")
        plt.close(fig)
    
    return fig


def perform_splitting_analysis_baillard(event_data, 
                                        s_window=[0.02, 0.3],
                                        min_lag=0, max_lag=60,
                                        Nlags=60, Nangles=90,
                                        flag_adapt_window=True,
                                        flag_adapt_maxlag=True,
                                        min_thres=0.5,
                                        min_numbers=2,
                                        plot_results=False,
                                        output_dir=None):
    """
    Perform shear-wave splitting analysis using Baillard's method.
    
    This replicates the workflow from run_sws.py and shearwavesplit.py:
    1. Extract horizontal components
    2. Calculate adaptive window parameters
    3. Perform eigenvalue grid search
    4. Find and rank minima
    5. Assess quality via RMS
    6. Return best solution(s)
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms
    s_window : list [before, after]
        Base S-window in seconds [time before S, time after S]
    min_lag : int
        Minimum delay time in samples
    max_lag : int
        Maximum delay time in samples
    Nlags : int
        Number of lag values to test
    Nangles : int
        Number of angle values to test
    flag_adapt_window : bool
        Whether to adapt window based on dominant period
    flag_adapt_maxlag : bool
        Whether to adapt max_lag based on dominant period
    min_thres : float
        Threshold for minimum selection (0-1)
    min_numbers : int
        Maximum number of minima to keep
    plot_results : bool
        Whether to generate diagnostic plots (default: False)
    output_dir : str or None
        Directory to save plots (if None and plot_results=True, displays interactively)
        
    Returns:
    --------
    dict
        Result dictionary with splitting parameters
    """
    
    station_name = event_data.get('station', 'UNKNOWN')
    
    try:
        print(f"  → Baillard splitting analysis for {station_name}")
        
        # Extract horizontal components
        event_traces = event_data.get('traces', [])
        if not event_traces:
            raise ValueError("No traces in event_data")
        
        if isinstance(event_traces, list):
            st = obspy.Stream(event_traces)
        else:
            st = event_traces
        
        # Get E and N components
        st_x = st.select(channel='??E')
        st_y = st.select(channel='??N')
        
        if len(st_x) == 0 or len(st_y) == 0:
            raise ValueError("Missing E or N component")
        
        trace_x = st_x[0]
        trace_y = st_y[0]
        
        # Create xy_array [N_samples x 2]
        xy_array = np.column_stack([trace_x.data, trace_y.data])
        
        # Get timing information
        trace_start_time = trace_x.stats.starttime
        sampling_rate = trace_x.stats.sampling_rate
        event_time = UTCDateTime(event_data['datetime'])
        s_time = event_time + event_data['s_arrival_time']
        p_time = event_time + event_data.get('p_arrival_time', event_data['s_arrival_time'] - 1.0)
        
        print(f"    Sampling rate: {sampling_rate} Hz")
        print(f"    Trace length: {len(xy_array)} samples ({len(xy_array)/sampling_rate:.2f}s)")
        
        # Calculate window parameters
        print(f"  → Calculating adaptive window parameters...")
        sw1, sw2, min_lag, max_lag, T_dom = calculate_baillard_window_params(
            xy_array, s_time, p_time, trace_start_time, sampling_rate,
            s_window_base=s_window, min_lag_base=min_lag, max_lag_base=max_lag,
            flag_adapt_window=flag_adapt_window, flag_adapt_maxlag=flag_adapt_maxlag)
        
        print(f"    Analysis window: samples {sw1} to {sw2} ({(sw2-sw1)/sampling_rate:.2f}s)")
        print(f"    Lag range: {min_lag} to {max_lag} samples ({min_lag/sampling_rate:.3f}s to {max_lag/sampling_rate:.3f}s)")
        
        # Baillard adds some padding to lag range in case of extrema at edges, we will replicate that
        add_lag = 3

        # Grid search
        print(f"  → Running eigenvalue grid search...")
        LAMBDA1, LAMBDA2, LAGS, ANGLES = get_LAMBDAS_baillard(
            xy_array, min_lag-add_lag, max_lag+add_lag, Nlags, Nangles, sw1, sw2)
        
        # Process minima
        print(f"  → Processing minima...")
        MinLambdas, _ = process_LAMBDAS_baillard(
            LAMBDA2, LAMBDA1, LAGS, ANGLES,
            min_lag_thres=min_lag, max_lag_thres=max_lag,
            min_thres=0.5, min_numbers=2,
            cont_step=0.01, quality_thres=0.2, zoom_factor=[2, 6],
            flag_plot=False, ax=None)
        
        if not MinLambdas:
            raise ValueError("No valid minima found")
        
        # Quality assessment via RMS
        print(f"  → Assessing quality via RMS...")
        MinLambdas = rms_MinLambdas_baillard(xy_array, sw1, sw2, MinLambdas, mode='norm')
        
        # Generate diagnostic plot if requested

        if plot_results:
            print(f"  → Generating diagnostic plot...")
            event_id = event_data.get('origin_time')
            try:
                s_samples = int(round((s_time - trace_start_time) * sampling_rate))

                fig = plot_baillard_splitting_results(
                    xy_array, LAMBDA1, LAMBDA2, LAGS, ANGLES,
                    MinLambdas, sw1, sw2, s_samples,
                    min_lag, max_lag, sampling_rate,
                    solution_index=0,
                    station_name=station_name, event_id=event_id,
                    output_dir=output_dir
                )
            except Exception as plot_error:
                print(f"  ⚠ Warning: Plotting failed: {plot_error}")
        
        # Get best solution
        best_solution = MinLambdas[0]
        
        # Create result dictionary
        result_dict = {
            'success': True,
            'phi': np.degrees(best_solution.angle),
            'phi_rad': best_solution.angle,
            'dt': best_solution.lag / sampling_rate,
            'dt_samples': best_solution.lag,
            'phi_error': np.degrees(best_solution.angle_error) if best_solution.angle_error else np.nan,
            'dt_error': best_solution.lag_error / sampling_rate if best_solution.lag_error else np.nan,
            'rms': best_solution.rms,
            'lambda2': best_solution.lambda_value,
            'T_dom': T_dom,
            'window_samples': [sw1, sw2],
            'lag_range': [min_lag, max_lag],
            'num_solutions': len(MinLambdas),
            'all_solutions': [ml.to_dict(sampling_rate) for ml in MinLambdas],
            'station': station_name,
            'back_azimuth': event_data.get('back_azimuth', np.nan),
            'incidence': event_data.get('incidence_eigenvalue_jurkevics', np.nan),
            'snr_horizontal': event_data.get('snr_horizontal', np.nan),
            'rectilinearity': event_data.get('rectilinearity_jurkevics', np.nan),
            'magnitude': event_data.get('magnitude', np.nan),
            'method': 'baillard'
        }
        
        print(f"  ✓ Baillard splitting analysis complete!")
        print(f"    Fast axis (φ): {result_dict['phi']:.1f}° ± {result_dict['phi_error']:.1f}°")
        print(f"    Delay time (δt): {result_dict['dt']:.3f}s ± {result_dict['dt_error']:.3f}s")
        print(f"    RMS quality: {result_dict['rms']:.4f}")
        print(f"    Found {len(MinLambdas)} solutions")
        
        return result_dict
        
    except Exception as e:
        import traceback
        print(f"  ✗ Error in Baillard splitting: {e}")
        print(traceback.format_exc())
        
        error_dict = {
            'success': False,
            'phi': np.nan,
            'dt': np.nan,
            'phi_error': np.nan,
            'dt_error': np.nan,
            'rms': np.nan,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'station': station_name,
            'method': 'baillard'
        }
        
        return error_dict


def apply_quality_control(organized_waveforms, qc_thresholds):
    """
    Apply quality control filters to organized_waveforms and return only events that pass all thresholds.
    
    This function assumes that all QC metrics have already been calculated and stored in organized_waveforms:
    - snr_horizontal: S-wave signal-to-noise ratio
    - rectilinearity: P-wave linearity measure
    - incidence: P-wave angle from vertical
    - magnitude: Event magnitude (optional)
    - back_azimuth: Geographic back-azimuth (for reference)
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing traces and all QC metrics
    qc_thresholds : dict
        Quality control threshold values with keys:
        - 'min_snr': Minimum horizontal SNR (e.g., 2.0)
        - 'min_rectilinearity': Minimum P-wave rectilinearity (e.g., 0.7)
        - 'max_incidence': Maximum incidence angle in degrees (e.g., 30.0)
        
    Returns:
    --------
    dict
        Filtered organized_waveforms containing only events that pass ALL QC thresholds
    """
    
    print(f"\n{'='*60}")
    print("Applying Quality Control Filters")
    print(f"{'='*60}")
    print(f"Initial events: {len(organized_waveforms)}")
    print(f"\nQC Thresholds:")
    print(f"  Minimum SNR: {qc_thresholds.get('min_snr', 2.0)}")
    print(f"  Minimum Rectilinearity: {qc_thresholds.get('min_rectilinearity', 0.7)}")
    print(f"  Maximum Incidence: {qc_thresholds.get('max_incidence', 30.0)}°")
    
    # Extract thresholds with defaults
    min_snr = qc_thresholds.get('min_snr', 2.0)
    min_rectilinearity = qc_thresholds.get('min_rectilinearity', 0.7)
    max_incidence = qc_thresholds.get('max_incidence', 30.0)
    
    # Track QC statistics
    qc_stats = {
        'total': len(organized_waveforms),
        'failed_snr': 0,
        'failed_rectilinearity': 0,
        'failed_incidence': 0,
        'failed_missing_data': 0,
        'passed': 0
    }
    
    filtered_waveforms = {}
    
    print(f"\n{'='*60}")
    print("Checking Individual Events")
    print(f"{'='*60}")
    
    for event_id, event_data in organized_waveforms.items():
        # Track failure reasons for this event
        failure_reasons = []
        
        # Check for missing QC metrics
        snr_horizontal = event_data.get('snr_horizontal', np.nan)
        rectilinearity = event_data.get('rectilinearity_jurkevics', np.nan)
        incidence = event_data.get('incidence_eigenvalue_jurkevics', np.nan)
        
        # Check if any required metrics are missing
        if np.isnan(snr_horizontal):
            failure_reasons.append('Missing SNR')
            qc_stats['failed_missing_data'] += 1
        
        if np.isnan(rectilinearity):
            failure_reasons.append('Missing rectilinearity')
            qc_stats['failed_missing_data'] += 1
        
        if np.isnan(incidence):
            failure_reasons.append('Missing incidence')
            qc_stats['failed_missing_data'] += 1
        
        # Apply QC thresholds
        if not np.isnan(snr_horizontal) and snr_horizontal < min_snr:
            failure_reasons.append(f'SNR too low ({snr_horizontal:.2f} < {min_snr})')
            qc_stats['failed_snr'] += 1
        
        if not np.isnan(rectilinearity) and rectilinearity < min_rectilinearity:
            failure_reasons.append(f'Rectilinearity too low ({rectilinearity:.3f} < {min_rectilinearity})')
            qc_stats['failed_rectilinearity'] += 1
        
        if not np.isnan(incidence) and incidence > max_incidence:
            failure_reasons.append(f'Incidence too high ({incidence:.1f}° > {max_incidence}°)')
            qc_stats['failed_incidence'] += 1
        
        # Event passes if no failure reasons
        if len(failure_reasons) == 0:
            filtered_waveforms[event_id] = event_data
            qc_stats['passed'] += 1
            print(f"✓ Event {event_id}: PASS")
            print(f"    SNR={snr_horizontal:.2f}, Rect={rectilinearity:.3f}, Inc={incidence:.1f}°")
        else:
            print(f"✗ Event {event_id}: FAIL - {', '.join(failure_reasons)}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Quality Control Summary")
    print(f"{'='*60}")
    print(f"Total events processed: {qc_stats['total']}")
    print(f"Events passed: {qc_stats['passed']} ({100*qc_stats['passed']/qc_stats['total']:.1f}%)")
    print(f"Events failed: {qc_stats['total'] - qc_stats['passed']}")
    print(f"\nFailure Breakdown:")
    print(f"  Failed SNR threshold: {qc_stats['failed_snr']}")
    print(f"  Failed rectilinearity threshold: {qc_stats['failed_rectilinearity']}")
    print(f"  Failed incidence threshold: {qc_stats['failed_incidence']}")
    print(f"  Missing QC data: {qc_stats['failed_missing_data']}")
    print(f"\nReturning {len(filtered_waveforms)} events for splitting analysis")
    
    return filtered_waveforms


def perform_splitting_on_organized_waveforms(organized_waveforms, first_window_start, last_window_start, first_window_end, last_window_end, n_win, s_pick_uncertainty, mode='swspy',
                                             plot_results=False):
    """
    Perform shear-wave splitting analysis on all events in organized_waveforms.
    
    This function assumes organized_waveforms has been filtered by apply_quality_control()
    and contains only events that pass QC thresholds. It extracts horizontal components
    and performs splitting analysis using the back_azimuth for coordinate rotation.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing QC-filtered traces and metadata
        Must have: traces, back_azimuth, station, and all other event metadata

    mode : str
        Splitting analysis method to use: 'swspy' or 'baillard' (default: 'swspy')
        
    Returns:
    --------
    dict
        Dictionary with event IDs as keys, each containing:
        {
            event_id: {
                'result': result_dict,  # Splitting parameters and metadata
                'splitting_obj': splitting_obj  # SWSPy splitting object
            },
            ...
        }
    """
    
    print(f"\n{'='*60}")
    print("Performing Shear-Wave Splitting Analysis")
    print(f"{'='*60}")
    print(f"Processing {len(organized_waveforms)} QC-filtered events...")
    
    event_results = {}
    
    # Track statistics
    stats = {
        'total_events': len(organized_waveforms),
        'missing_components': 0,
        'missing_back_azimuth': 0,
        'splitting_errors': 0,
        'successful_splits': 0
    }
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\n{'─'*60}")
        print(f"Event {event_id}")
        print(f"{'─'*60}")
        
        # Get traces
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  ✗ No traces found")
            stats['missing_components'] += 1
            continue
        
        # Convert to stream if needed
        if isinstance(event_traces, list):
            event_stream = obspy.Stream(event_traces)
        else:
            event_stream = event_traces
        
        # Find N and E components
        trace_n = None
        trace_e = None
        trace_z = None
        
        for tr in event_stream:
            component = tr.stats.channel[-1].upper()
            if component in ['N', '1']:
                trace_n = tr
            elif component in ['E', '2']:
                trace_e = tr
            elif component == 'Z':
                trace_z = tr
        
        # Check if we have horizontal components
        if trace_n is None or trace_e is None:
            print(f"  ✗ Missing horizontal components (N: {trace_n is not None}, E: {trace_e is not None})")
            stats['missing_components'] += 1
            continue
        
        print(f"  ✓ Found horizontal components")
        
        # Check for back-azimuth
        back_azimuth = event_data.get('back_azimuth')
        if back_azimuth is None or np.isnan(back_azimuth):
            print(f"  ✗ Missing back-azimuth")
            stats['missing_back_azimuth'] += 1
            continue
        
        print(f"  ✓ Back-azimuth: {back_azimuth:.2f}°")
        
        # Get station and other metadata
        station_name = event_data.get('station', 'UNKNOWN')
        magnitude = event_data.get('magnitude', np.nan)
        snr_horizontal = event_data.get('snr_horizontal', np.nan)
        rectilinearity = event_data.get('rectilinearity_jurkevics', np.nan)
        incidence = event_data.get('incidence_eigenvalue_jurkevics', np.nan)
        
        print(f"  Station: {station_name}")
        print(f"  Magnitude: {magnitude:.1f}")
        print(f"  SNR: {snr_horizontal:.2f}")
        print(f"  Rectilinearity: {rectilinearity:.3f}")
        print(f"  Incidence: {incidence:.1f}°")
        
        print(f"\n  → Running splitting analysis...")
        
        try:
           
            if mode=='baillard':
                # Call Baillard splitting function
                splitting_result = perform_splitting_analysis_baillard(
                    event_data,
                    s_window=[0.02,0.3],
                    min_lag=0,
                    max_lag=60,
                    Nlags=60,
                    Nangles=90,
                    flag_adapt_window=True,
                    flag_adapt_maxlag=True,
                    min_thres=0.5,
                    min_numbers=2,
                    plot_results=plot_results
                )
                
                # Baillard returns dict only (no splitting_obj like SWSPy)
                splitting_obj = None

            # Hemmett-adapted SWSPy-like implementation that draws on MFAST
            elif mode=='swspy':
                # Call SWSPy-like splitting function
                splitting_result, splitting_obj = perform_splitting_analysis(
                    event_data, first_window_start, last_window_start, first_window_end, last_window_end, n_win, s_pick_uncertainty, plot_results=plot_results
                )

            elif mode=='teanby_baillard':
                # Call Teanby clustering with Baillard method
                splitting_result = tbc.teanby_clustering_analysis(
                    event_data,
                    T_beg_1=0.02, T_end_0=0.3,
                    dT_beg=0.005, dT_end=0.005,
                    N_beg=10, N_end=10,
                    min_lag=0, max_lag=60,
                    Nlags=60, Nangles=90,
                    flag_adapt_window=True,
                    flag_adapt_maxlag=True,
                    min_thres=0.5,
                    min_numbers=2,
                    plot_results=plot_results,
                    output_dir=None,
                    N_c_min=5
                )
                splitting_obj = None

            if splitting_result.get('success', False):
                # Add additional metadata to result
                splitting_result['event_id'] = event_id
                splitting_result['back_azimuth'] = back_azimuth
                splitting_result['snr_horizontal'] = snr_horizontal
                splitting_result['rectilinearity_jurkevics'] = rectilinearity
                splitting_result['incidence_eigenvalue_jurkevics'] = incidence
                splitting_result['event_lat'] = event_data.get('latitude')
                splitting_result['event_lon'] = event_data.get('longitude')
                splitting_result['event_depth'] = event_data.get('depth')
                splitting_result['event_origin_time'] = event_data.get('origin_time')
                splitting_result['event_datetime'] = event_data.get('datetime')
                splitting_result['s_arrival_time'] = event_data.get('s_arrival_time')
                splitting_result['first_window_start'] = first_window_start
                splitting_result['last_window_start'] = last_window_start
                splitting_result['first_window_end'] = first_window_end
                splitting_result['last_window_end'] = last_window_end
                splitting_result['s_pick_uncertainty'] = s_pick_uncertainty
                splitting_result['n_win'] = n_win
                
                # Store both result dict and splitting object for later use
                event_results[event_id] = {
                    'result': splitting_result,
                    'splitting_obj': splitting_obj
                }
                stats['successful_splits'] += 1
                
                print(f"  ✓ SUCCESS!")
                if not np.isnan(splitting_result['phi']):
                    print(f"    Fast axis (φ): {splitting_result['phi']:.1f}°")
                else:
                    print(f"    Fast axis (φ): N/A")
                    
                if not np.isnan(splitting_result['dt']):
                    print(f"    Delay time (δt): {splitting_result['dt']:.3f}s")
                else:
                    print(f"    Delay time (δt): N/A")
                    
                if not np.isnan(splitting_result.get('phi_error', np.nan)):
                    print(f"    φ error: ±{splitting_result['phi_error']:.1f}°")
                if not np.isnan(splitting_result.get('dt_error', np.nan)):
                    print(f"    δt error: ±{splitting_result['dt_error']:.3f}s")
                if 'dominant_period' in splitting_result:
                    print(f"    Dominant period: {splitting_result['dominant_period']:.3f}s")
            else:
                error_msg = splitting_result.get('error', 'Unknown error')
                print(f"  ✗ FAILED: {error_msg}")
                if 'traceback' in splitting_result:
                    print(f"  Traceback:\n{splitting_result['traceback']}")
                stats['splitting_errors'] += 1
                
        except Exception as e:
            import traceback
            print(f"  ✗ ERROR during splitting: {e}")
            print(f"  Traceback:\n{traceback.format_exc()}")
            stats['splitting_errors'] += 1
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("Splitting Analysis Summary")
    print(f"{'='*60}")
    print(f"Total events processed: {stats['total_events']}")
    print(f"Successful splits: {stats['successful_splits']} ({100*stats['successful_splits']/stats['total_events']:.1f}%)")
    print(f"\nFailure breakdown:")
    print(f"  Missing components: {stats['missing_components']}")
    print(f"  Missing back-azimuth: {stats['missing_back_azimuth']}")
    print(f"  Splitting errors: {stats['splitting_errors']}")
    
    # Calculate splitting parameter statistics if we have results
    if event_results:

        if mode!='swspy':
            # Extract phi and dt values from nested structure
            phi_values = [r['result']['phi_rad'] for r in event_results.values() 
                        if not np.isnan(r['result']['phi_rad'])]
            dt_values = [r['result']['dt'] for r in event_results.values() 
                        if not np.isnan(r['result']['dt'])]
        
        else:
            # Extract phi and dt values from nested structure
            phi_values = [r['result']['phi'] for r in event_results.values() 
                        if not np.isnan(r['result']['phi'])]
            dt_values = [r['result']['dt'] for r in event_results.values() 
                        if not np.isnan(r['result']['dt'])]
        
        if phi_values and dt_values:
            print(f"\n{'─'*60}")
            print("Splitting Parameter Statistics")
            print(f"{'─'*60}")
            print(f"Fast axis direction (φ):")
            print(f"  Mean: {np.mean(phi_values):.1f}° ± {np.std(phi_values):.1f}°")
            print(f"  Range: {np.min(phi_values):.1f}° to {np.max(phi_values):.1f}°")
            print(f"  Median: {np.median(phi_values):.1f}°")
            
            print(f"\nDelay time (δt):")
            print(f"  Mean: {np.mean(dt_values):.3f} ± {np.std(dt_values):.3f}s")
            print(f"  Range: {np.min(dt_values):.3f}s to {np.max(dt_values):.3f}s")
            print(f"  Median: {np.median(dt_values):.3f}s")
            
            # Add to stats
            stats['phi_mean'] = float(np.mean(phi_values))
            stats['phi_std'] = float(np.std(phi_values))
            stats['dt_mean'] = float(np.mean(dt_values))
            stats['dt_std'] = float(np.std(dt_values))
        else:
            print(f"\n  Note: No valid splitting parameters for statistics")
    
    # Return the event_results dictionary directly
    # Each entry contains both 'result' and 'splitting_obj'
    return event_results


def process_all_events(waveform_data, extended_catalog, qc_thresholds):
    """
    Process all events through the complete splitting analysis workflow.
    
    Parameters:
    -----------
    waveform_data : dict
        Organized waveform data by event ID
    extended_catalog : pandas.DataFrame
        Extended catalog with timing information
    qc_thresholds : dict
        Quality control threshold values
    
    Returns:
    --------
    dict
        Complete results including QC and splitting analysis
    """
    
    all_results = {
        'qc_results': {},
        'splitting_results': {},
        'summary_stats': {}
    }
    
    processed_count = 0
    qc_passed_count = 0
    splitting_success_count = 0
    
    print(f"Processing {len(waveform_data)} events...")
    
    for event_id in tqdm(waveform_data.keys(), desc="Processing events"):
        event_info = waveform_data[event_id]
        event_traces = event_info['traces']
        
        if not event_traces:
            continue
            
        processed_count += 1
        
        # Apply quality control
        qc_result = apply_quality_control(event_traces, event_info, qc_thresholds)
        all_results['qc_results'][event_id] = qc_result
        
        if qc_result['overall_pass']:
            qc_passed_count += 1
            
            # Organize traces by station and component for splitting analysis
            traces_by_station = {}
            for trace in event_traces:
                station = trace.stats.station
                component = trace.stats.channel[-1]
                
                if station not in traces_by_station:
                    traces_by_station[station] = {}
                traces_by_station[station][component] = trace
            
            # Perform splitting analysis on stations with horizontal components
            event_splitting_results = {}
            
            for station, components in traces_by_station.items():
                # Check if we have horizontal components
                trace_n = components.get('N') or components.get('1')
                trace_e = components.get('E') or components.get('2')
                
                if trace_n and trace_e:
                    # Check if this station passed QC
                    if (station in qc_result['station_results'] and 
                        qc_result['station_results'][station]['snr_pass']):
                        
                        splitting_result = perform_splitting_analysis(
                            trace_n, trace_e, event_info, station
                        )
                        
                        if splitting_result['success']:
                            event_splitting_results[station] = splitting_result
                            splitting_success_count += 1
            
            all_results['splitting_results'][event_id] = event_splitting_results
    
    # Calculate summary statistics
    all_results['summary_stats'] = {
        'total_events': len(waveform_data),
        'processed_events': processed_count,
        'qc_passed_events': qc_passed_count,
        'splitting_success_count': splitting_success_count,
        'qc_pass_rate': qc_passed_count / processed_count if processed_count > 0 else 0,
        'splitting_success_rate': splitting_success_count / qc_passed_count if qc_passed_count > 0 else 0
    }
    
    return all_results

    
def compile_results_dataframe(splitting_results, extended_catalog):
    """
    Compile splitting results into a structured DataFrame for analysis.
    
    Parameters:
    -----------
    splitting_results : dict
        Splitting analysis results by event containing 'result' and 'splitting_obj'
    extended_catalog : pandas.DataFrame
        Extended catalog with event metadata
    
    Returns:
    --------
    pandas.DataFrame
        Compiled results with splitting parameters and metadata
    """
    
    results_list = []
    
    for event_id, event_result in splitting_results.items():
        # Extract the result dictionary (not 'splitting_obj')
        split_data = event_result.get('result')
        
        if split_data is None or not split_data.get('success', False):
            continue
        
        # Get event metadata from catalog
        event_row = extended_catalog[extended_catalog['id'] == event_id]
        if event_row.empty:
            # Try using the metadata stored in split_data
            event_lat = split_data.get('event_lat', np.nan)
            event_lon = split_data.get('event_lon', np.nan)
            event_depth = split_data.get('event_depth', np.nan)
        else:
            event_meta = event_row.iloc[0]
            event_lat = event_meta.get('lat', np.nan)
            event_lon = event_meta.get('lon', np.nan)
            event_depth = event_meta.get('dep', np.nan)
        
        # Compile result entry
        result_entry = {
            'event_id': event_id,
            'station': split_data.get('station', 'UNKNOWN'),
            'phi': split_data.get('phi', np.nan),
            'dt': split_data.get('dt', np.nan),
            'phi_error': split_data.get('phi_error', np.nan),
            'dt_error': split_data.get('dt_error', np.nan),
            'avg_snr': split_data.get('snr_avg', split_data.get('snr_horizontal', np.nan)),
            'magnitude': split_data.get('magnitude', np.nan),
            'event_lat': event_lat,
            'event_lon': event_lon,
            'event_depth': event_depth,
            'window_duration': split_data.get('window_duration', np.nan),
            'quality': split_data.get('quality', 'unknown'),
            'back_azimuth': split_data.get('back_azimuth', np.nan),
            'rectilinearity': split_data.get('rectilinearity', np.nan),
            'incidence': split_data.get('incidence', np.nan)
        }
        
        results_list.append(result_entry)
    
    if results_list:
        return pd.DataFrame(results_list)
    else:
        print("Warning: No successful splitting results found to compile")
        return pd.DataFrame()


def create_splitting_plots(results_df):
    """
    Create comprehensive plots of splitting analysis results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Compiled splitting results
    """
    
    if results_df.empty:
        print("No results available for plotting")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Fast axis directions (phi) histogram
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(results_df['phi'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Fast Axis Direction φ (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fast Axis Directions')
    plt.axvline(results_df['phi'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["phi"].mean():.1f}°')
    plt.legend()
    
    # Plot 2: Delay times (dt) histogram  
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(results_df['dt'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Delay Time δt (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delay Times')
    plt.axvline(results_df['dt'].mean(), color='red', linestyle='--',
                label=f'Mean: {results_df["dt"].mean():.3f}s')
    plt.legend()
    
    # Plot 3: Phi vs Magnitude
    ax3 = plt.subplot(2, 3, 3)
    scatter = plt.scatter(results_df['magnitude'], results_df['phi'], 
                         c=results_df['dt'], cmap='viridis', alpha=0.6)
    plt.xlabel('Magnitude')
    plt.ylabel('Fast Axis Direction φ (degrees)')
    plt.title('φ vs Magnitude (colored by δt)')
    plt.colorbar(scatter, label='δt (seconds)')
    
    # Plot 4: Delay time vs Magnitude
    ax4 = plt.subplot(2, 3, 4)
    plt.scatter(results_df['magnitude'], results_df['dt'], alpha=0.6, color='orange')
    plt.xlabel('Magnitude')
    plt.ylabel('Delay Time δt (seconds)')
    plt.title('Delay Time vs Magnitude')
    
    # Plot 5: Station-wise splitting parameters
    ax5 = plt.subplot(2, 3, 5)
    station_means = results_df.groupby('station').agg({
        'phi': 'mean',
        'dt': 'mean'
    }).reset_index()
    
    plt.scatter(station_means['phi'], station_means['dt'], s=100, alpha=0.7)
    for idx, row in station_means.iterrows():
        plt.annotate(row['station'], (row['phi'], row['dt']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Mean Fast Axis Direction φ (degrees)')
    plt.ylabel('Mean Delay Time δt (seconds)')
    plt.title('Station-Average Splitting Parameters')
    
    # Plot 6: Quality metrics
    ax6 = plt.subplot(2, 3, 6)
    plt.scatter(results_df['avg_snr'], results_df['dt'], alpha=0.6, color='green')
    plt.xlabel('Average SNR')
    plt.ylabel('Delay Time δt (seconds)')
    plt.title('Delay Time vs Signal Quality')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics table
    print(f"\n{'='*60}")
    print("SPLITTING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total measurements: {len(results_df)}")
    print(f"Unique stations: {results_df['station'].nunique()}")
    print(f"Unique events: {results_df['event_id'].nunique()}")
    print(f"\nFast Axis Direction (φ):")
    print(f"  Mean: {results_df['phi'].mean():.1f}° ± {results_df['phi'].std():.1f}°")
    print(f"  Range: {results_df['phi'].min():.1f}° to {results_df['phi'].max():.1f}°")
    print(f"\nDelay Time (δt):")
    print(f"  Mean: {results_df['dt'].mean():.3f} ± {results_df['dt'].std():.3f} seconds")
    print(f"  Range: {results_df['dt'].min():.3f} to {results_df['dt'].max():.3f} seconds")
    print(f"\nQuality Metrics:")
    print(f"  Average SNR: {results_df['avg_snr'].mean():.1f}")
    print(f"  Magnitude range: {results_df['magnitude'].min():.1f} to {results_df['magnitude'].max():.1f}")


def create_analysis_documentation(results, qc_thresholds, results_df):
    """
    Create comprehensive documentation of the analysis workflow and parameters.
    
    Parameters:
    -----------
    results : dict
        Complete analysis results
    qc_thresholds : dict
        Quality control parameters used
    results_df : pandas.DataFrame
        Compiled splitting results
    
    Returns:
    --------
    dict
        Documentation dictionary
    """
    
    doc = {
        'analysis_info': {
            'workflow': 'Master Shear-Wave Splitting Analysis',
            'date_created': datetime.now().isoformat(),
            'software_versions': {
                'python': '3.x',
                'obspy': 'latest',
                'swspy': 'latest'
            }
        },
        'parameters': {
            'quality_control': qc_thresholds,
            'time_windows': {
                'pre_event_time': 4.0,
                'post_event_time': 15.0,
                'p_wave_analysis_window': 1.0,
                's_wave_base_window': 2.0
            },
            'splitting_analysis': {
                'phi_range': '(-90, 95) degrees in 5° steps',
                'dt_range': '(0.0, 0.25) seconds in 0.01s steps',
                'window_scaling': 'Dynamic based on magnitude'
            }
        },
        'statistics': results.get('summary_stats', {}),
        'file_locations': {
            'input_catalog': '../data/final_catalog.csv',
            'trace_data': '../scripts/all_earthquakes_trace_data.pkl',
            'output_results': '../results/master_splitting_results.csv'
        }
    }
    
    if not results_df.empty:
        doc['result_statistics'] = {
            'phi_stats': {
                'mean': float(results_df['phi'].mean()),
                'std': float(results_df['phi'].std()),
                'min': float(results_df['phi'].min()),
                'max': float(results_df['phi'].max())
            },
            'dt_stats': {
                'mean': float(results_df['dt'].mean()),
                'std': float(results_df['dt'].std()),
                'min': float(results_df['dt'].min()),
                'max': float(results_df['dt'].max())
            },
            'measurement_counts': {
                'total_measurements': len(results_df),
                'unique_stations': int(results_df['station'].nunique()),
                'unique_events': int(results_df['event_id'].nunique())
            }
        }
    
    return doc


def export_results_multiple_formats(results, results_df, documentation):
    """
    Export results in multiple formats for different use cases.
    """
    
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. CSV export (main results)
    if not results_df.empty:
        csv_path = output_dir / 'master_splitting_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Results exported to CSV: {csv_path}")
    
    # 2. JSON export (complete results with metadata)
    json_path = output_dir / f'complete_analysis_results_{timestamp}.json'
    
    # Convert numpy types to Python native types for JSON serialization
    json_results = {}
    for key, value in results.items():
        if key == 'splitting_results':
            # Convert splitting results
            json_results[key] = {}
            for event_id, stations in value.items():
                json_results[key][event_id] = {}
                for station, data in stations.items():
                    # Convert numpy values
                    clean_data = {}
                    for k, v in data.items():
                        if isinstance(v, np.ndarray):
                            clean_data[k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            clean_data[k] = v.item()
                        else:
                            clean_data[k] = v
                    json_results[key][event_id][station] = clean_data
        else:
            json_results[key] = value
    
    with open(json_path, 'w') as f:
        json.dump({'results': json_results, 'documentation': documentation}, f, indent=2)
    print(f"Complete results exported to JSON: {json_path}")
    
    # 3. Documentation export
    doc_path = output_dir / f'analysis_documentation_{timestamp}.json'
    with open(doc_path, 'w') as f:
        json.dump(documentation, f, indent=2)
    print(f"Analysis documentation exported: {doc_path}")
    
    # 4. Summary report
    report_path = output_dir / f'analysis_summary_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("AXIAL SEAMOUNT SHEAR-WAVE SPLITTING ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Analysis Date: {documentation['analysis_info']['date_created']}\n")
        f.write(f"Workflow: {documentation['analysis_info']['workflow']}\n\n")
        
        stats = documentation['statistics']
        f.write("PROCESSING STATISTICS:\n")
        f.write(f"Total events processed: {stats.get('processed_events', 'N/A')}\n")
        f.write(f"Events passing QC: {stats.get('qc_passed_events', 'N/A')}\n")
        f.write(f"Successful splitting analyses: {stats.get('splitting_success_count', 'N/A')}\n")
        f.write(f"QC pass rate: {stats.get('qc_pass_rate', 0):.1%}\n")
        f.write(f"Splitting success rate: {stats.get('splitting_success_rate', 0):.1%}\n\n")
        
        if 'result_statistics' in documentation:
            result_stats = documentation['result_statistics']
            f.write("SPLITTING PARAMETERS:\n")
            f.write(f"Fast axis direction (φ): {result_stats['phi_stats']['mean']:.1f}° ± {result_stats['phi_stats']['std']:.1f}°\n")
            f.write(f"Delay time (δt): {result_stats['dt_stats']['mean']:.3f} ± {result_stats['dt_stats']['std']:.3f} seconds\n")
            f.write(f"Total measurements: {result_stats['measurement_counts']['total_measurements']}\n")
            f.write(f"Stations analyzed: {result_stats['measurement_counts']['unique_stations']}\n")
    
    print(f"Summary report exported: {report_path}")


def parse_phase_file(filename):
    """
    Parse the phase file format used at Axial Seamount:
    Event lines: # YYYY MM DD HH MM SS.ss LAT LON DEPTH MAG ... EVENT_ID E
    Phase lines: STATION ARRIVAL_TIME WEIGHT PHASE_TYPE QUALITY
    
    Returns:
    --------
    events_df : pandas.DataFrame
        DataFrame with earthquake event information
    phases_df : pandas.DataFrame  
        DataFrame with phase picks for each event
    """
    
    events = []
    phases = []
    current_event_id = None
    current_event_info = None
    
    print(f"Parsing phase file: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} lines...")
    
    for line_num, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        parts = stripped.split()
        
        # Check if this is an event line (starts with #)
        if stripped.startswith('#') and len(parts) >= 10:
            try:
                year = int(parts[1])
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                second = float(parts[6])
                lat = float(parts[7])
                lon = float(parts[8])
                depth = float(parts[9])
                
                # Find the event ID (second to last element, before 'E')
                event_id = parts[-2] if len(parts) >= 2 else str(len(events))
                
                # Create UTC datetime string
                datetime_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:06.3f}Z"
                
                event_info = {
                    'event_id': event_id,
                    'year': year,
                    'month': month,
                    'day': day, 
                    'hour': hour,
                    'minute': minute,
                    'second': second,
                    'datetime_str': datetime_str,
                    'lat': lat,
                    'lon': lon,
                    'depth': depth
                }
                
                # Add magnitude if available (typically at index 10)
                if len(parts) > 10:
                    try:
                        event_info['magnitude'] = float(parts[10])
                    except ValueError:
                        event_info['magnitude'] = np.nan
                else:
                    event_info['magnitude'] = np.nan
                
                events.append(event_info)
                current_event_info = event_info
                current_event_id = event_id
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse event line {line_num + 1}: {stripped}")
                continue
                
        else:
            # This should be a phase pick line if we have a current event
            if current_event_id is not None and current_event_info is not None and len(parts) >= 4:
                try:
                    phase_info = {
                        'event_id': current_event_id,
                        'event_datetime': current_event_info['datetime_str'],
                        'event_lat': current_event_info['lat'],
                        'event_lon': current_event_info['lon'],
                        'event_depth': current_event_info['depth'],
                        'event_magnitude': current_event_info['magnitude'],
                        'station': parts[0],
                        'arrival_time': float(parts[1]),
                        'weight': float(parts[2]) if parts[2] != '-1.000' else np.nan,
                        'phase_type': parts[3],
                        'quality': parts[4] if len(parts) > 4 else ''
                    }
                    
                    phases.append(phase_info)
                    
                except (ValueError, IndexError) as e:
                    # Skip problematic phase lines
                    continue
    
    # Convert to DataFrames
    events_df = pd.DataFrame(events)
    phases_df = pd.DataFrame(phases)
    
    # Convert event_id to numeric if possible
    if len(events_df) > 0:
        try:
            events_df['event_id'] = pd.to_numeric(events_df['event_id'])
            phases_df['event_id'] = pd.to_numeric(phases_df['event_id'])
        except:
            pass  # Keep as string if conversion fails
    
    return events_df, phases_df

def create_organized_catalog(events_df, phases_df):
    """
    Create organized catalog where each row represents one event at one station
    with both P- and S-wave picks and arrival times.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        Event information from phase file
    phases_df : pandas.DataFrame
        Phase picks from phase file
    
    Returns:
    --------
    pandas.DataFrame
        Organized catalog with events × stations structure
    """
    
    print("Creating organized catalog with events × stations structure...")
    
    # Get all unique stations and events
    all_stations = phases_df['station'].unique()
    all_events = events_df['event_id'].unique()
    
    print(f"Events: {len(all_events)}, Stations: {len(all_stations)}")
    print(f"Stations in network: {sorted(all_stations)}")
    
    final_rows = []
    
    for event_id in all_events:
        # Get event metadata
        event_info = events_df[events_df['event_id'] == event_id].iloc[0]
        
        # Get all phase picks for this event
        event_phases = phases_df[phases_df['event_id'] == event_id]
        
        # Group phase picks by station
        for station in all_stations:
            station_phases = event_phases[event_phases['station'] == station]
            
            # Initialize row with event information
            row = {
                'id': event_id,
                'year': event_info['year'],
                'datetime': event_info['datetime_str'],
                'lat': event_info['lat'],
                'lon': event_info['lon'],
                'dep': event_info['depth'],  # Use 'dep' to match old catalog format
                'mag': event_info['magnitude'],
                'station': station,
                'total_picks': len(station_phases),
                'p_arrival_time': np.nan,
                'p_weight': np.nan,
                'p_quality': '',
                's_arrival_time': np.nan,
                's_weight': np.nan,
                's_quality': ''
            }
            
            # Fill in phase pick information if available
            if len(station_phases) > 0:
                # P-wave information
                p_phases = station_phases[station_phases['phase_type'] == 'P']
                if len(p_phases) > 0:
                    p_pick = p_phases.iloc[0]  # Take first P pick if multiple
                    row['p_arrival_time'] = p_pick['arrival_time']
                    row['p_weight'] = p_pick['weight']
                    row['p_quality'] = p_pick['quality']
                
                # S-wave information
                s_phases = station_phases[station_phases['phase_type'] == 'S']
                if len(s_phases) > 0:
                    s_pick = s_phases.iloc[0]  # Take first S pick if multiple
                    row['s_arrival_time'] = s_pick['arrival_time']
                    row['s_weight'] = s_pick['weight']
                    row['s_quality'] = s_pick['quality']
            
            final_rows.append(row)
    
    catalog_df = pd.DataFrame(final_rows)
    
    # Filter to only keep rows with both P and S picks
    has_both_picks = (~catalog_df['p_arrival_time'].isna()) & (~catalog_df['s_arrival_time'].isna())
    catalog_filtered = catalog_df[has_both_picks].copy()
    
    print(f"Total event-station pairs: {len(catalog_df)}")
    print(f"Pairs with both P and S picks: {len(catalog_filtered)}")

    return catalog_filtered

def organize_stream_by_events(stream, catalog, time_tolerance=5.0):
    """
    Organize the stream data by matching traces to catalog events.
    
    Parameters:
    -----------
    stream : obspy.Stream
        The full stream containing all waveform data
    catalog : pandas.DataFrame
        Catalog of events with datetime and station information
    time_tolerance : float
        Tolerance in seconds for matching trace start times to event times
        
    Returns:
    --------
    dict
        Dictionary mapping event indices to their corresponding stream data
    """
    event_streams = {}
    
    print(f"Organizing stream data for {len(catalog)} events...")
    
    for i, (idx, event) in enumerate(catalog.iterrows()):
        event_time = UTCDateTime(event['datetime'])
        station_id = event['station']
        
        # Find traces that match this event's time and station
        matching_traces = []
        
        for tr in stream.select(station=str(station_id)):
            # Check if trace start time is close to event time
            time_diff = abs(tr.stats.starttime - event_time)
            
            if time_diff <= time_tolerance:
                matching_traces.append(tr.copy())
        
        if len(matching_traces) >= 3:  # Need at least Z, N, E components
            event_stream = obspy.Stream(matching_traces)
            event_streams[idx] = event_stream
            print(f"  Event {i+1}: Found {len(matching_traces)} traces for {station_id} at {event_time}")
        else:
            print(f"  Event {i+1}: Only {len(matching_traces)} traces found for {station_id} at {event_time}")
            event_streams[idx] = None
    
    return event_streams
    
#def compute_snr_for_event(event_stream, event_row, component='E'):
    """
    Calculate SNR for a specific component using S-wave windows.
    
    Parameters:
    -----------
    event_stream : obspy.Stream or list
        Stream or list of traces for the event
    event_row : dict or pandas.Series
        Event metadata including S and P arrival times
    component : str
        Component to calculate SNR for ('E', 'N', or 'Z')
        
    Returns:
    --------
    float
        Signal-to-noise ratio for S-wave window
    """
    try:
        # Convert to stream if it's a list
        if isinstance(event_stream, list):
            event_stream = obspy.Stream(event_stream)
        
        # Check if S-arrival time exists
        if 's_arrival_time' not in event_row or pd.isna(event_row['s_arrival_time']):
            print(f"  No S-arrival time for {component}")
            return np.nan
        
        event_time = UTCDateTime(event_row['datetime'])
        s_pick = event_time + float(event_row['s_arrival_time'])
        print(f"  S-pick time: {s_pick}")
        
        # Get S-P delay for adaptive noise window sizing
        sp_delay = None
        if 'p_arrival_time' in event_row and not pd.isna(event_row['p_arrival_time']):
            sp_delay = event_row['s_arrival_time'] - event_row['p_arrival_time']
            print(f"  S-P delay: {sp_delay:.2f}s")
        
        # Find component trace
        trace = None
        for tr in event_stream:
            if tr.stats.channel[-1].upper() == component.upper():
                trace = tr.copy()
                break
        
        if trace is None:
            print(f"  No {component} component trace found")
            return np.nan
        
        print(f"  Found {component} trace: {trace.stats.starttime} to {trace.stats.endtime}")
        
        # Apply filtering
        trace.filter("bandpass", freqmin=5, freqmax=40)
        trace.taper(type="hann", max_percentage=0.05)
        
        # Define adaptive time windows
        noise_window = 0.4
        #noise_window = min(0.4, sp_delay / 2) if sp_delay else 0.4
        signal_window = 0.2
        
        noise_start = s_pick - noise_window  # BEFORE S-arrival
        noise_end = s_pick                   # UP TO S-arrival
        signal_start = s_pick                # FROM S-arrival
        signal_end = s_pick + signal_window  # AFTER S-arrival
        
        print(f"  Noise window: {noise_start} to {noise_end} ({noise_window:.2f}s)")
        print(f"  Signal window: {signal_start} to {signal_end} ({signal_window:.2f}s)")
        
        # Check trace coverage
        if (trace.stats.starttime > noise_start or trace.stats.endtime < signal_end):
            print(f"  Trace doesn't cover required windows")
            print(f"  Trace: {trace.stats.starttime} to {trace.stats.endtime}")
            return np.nan
        
        # Extract data for noise and signal windows
        noise_data = trace.slice(noise_start, noise_end).data
        signal_data = trace.slice(signal_start, signal_end).data
        
        print(f"  Noise data points: {len(noise_data)}")
        print(f"  Signal data points: {len(signal_data)}")
        
        if len(noise_data) == 0 or len(signal_data) == 0:
            print(f"  Empty noise or signal window")
            return np.nan
        
        # Calculate RMS amplitudes
        noise_rms = np.sqrt(np.mean(noise_data**2))
        signal_rms = np.sqrt(np.mean(signal_data**2))
        
        print(f"  Noise RMS: {noise_rms:.6f}")
        print(f"  Signal RMS: {signal_rms:.6f}")
        
        if noise_rms == 0:
            print(f"  Zero noise level")
            return np.nan
        
        snr = signal_rms / noise_rms
        print(f"  SNR: {snr:.2f}")
        return float(snr)
        
    except Exception as e:
        print(f"  Error calculating SNR for {component}: {e}")
        return np.nan
    
# ...existing code...

def check_index(ind,data):
    
    if ind<0:
        ind=0
    elif ind>=len(data):
        ind=len(data)-1
        
    return ind

def SNR_pick(data,ind_center,N_left,N_right,mode='mean',flag_plot=False):
    """
    Cmpute SNR around given pick based on absolute values
    """
    abs_data=np.abs(data)
    ind_left=check_index(ind_center-N_left,abs_data)
    ind_right=check_index(ind_center+N_right,abs_data)
    if mode=='mean':
        right=np.mean(abs_data[ind_center:ind_right])
    elif mode=='max':
        right=np.max(abs_data[ind_center:ind_right])
    else:
        raise ValueError('mode has to be mean or max')
    left=np.mean(abs_data[ind_left:ind_center+1])
    
    ratio=right/left
    
    if flag_plot:
        _,ax=plt.subplots(2,1,sharex=True)
        ax[0].plot(data,color='k',ls='-',lw=1)
        ax[1].plot(abs_data,color='k',ls='-',lw=1)
        ax[1].axvline(x=ind_center,color='green')
        ax[1].axvline(x=ind_center-N_left,color='r')
        ax[1].axvline(x=ind_center+N_right,color='r')
        ax[1].text(0.1, 0.9,'Ratio=%.1f'%ratio, horizontalalignment='center',
          verticalalignment='center', transform=ax[1].transAxes)
        
        ### Cosmetic
        ax[0].set_ylabel('Data')
        ax[1].set_ylabel('abs(Data)')
    
    return ratio

def compute_snr_for_event(event_stream, event_row, component='E', 
                          N_left=80, N_right=40, mode='mean'):
    """
    Calculate SNR using Baillard's absolute-value method for comparison.
    
    Parameters:
    -----------
    event_stream : obspy.Stream
        Stream with traces
    event_row : dict
        Event metadata with s_arrival_time
    component : str
        Component ('E', 'N', or 'Z')
    N_left : int
        Number of samples before S-pick for noise window (default=80 @ 200Hz = 0.4s)
    N_right : int
        Number of samples after S-pick for signal window (default=40 @ 200Hz = 0.2s)
    mode : str
        'mean' for average or 'max' for peak amplitude
    
    Returns:
    --------
    float
        SNR ratio (signal/noise)
    """
    try:
        # Convert to stream if needed
        if isinstance(event_stream, list):
            event_stream = obspy.Stream(event_stream)
        
        # Find component trace
        trace = None
        for tr in event_stream:
            if tr.stats.channel[-1].upper() == component.upper():
                trace = tr.copy()
                break
        
        if trace is None:
            return np.nan
        
        # Apply same filtering as your method

        # Added redefinition of trace in case that is messing with downstream calculations
        #trace = trace.filter("bandpass", freqmin=5, freqmax=40)
        #trace = trace.taper(type="hann", max_percentage=0.05)
        
        # Calculate S-pick sample index
        event_time = UTCDateTime(event_row['datetime'])
        s_pick = event_time + float(event_row['s_arrival_time'])
        
        # Convert to sample index - potentially problematic if trace starts before actual start time
        # Except this may be necessary, to ensure the sampling looks correct
        ind_center = int(round((s_pick - trace.stats.starttime) * trace.stats.sampling_rate))

        # Convert to sample index - here s_arrival_time is time in seconds since event start
        # ind_center = int((float(event_row['s_arrival_time']) * trace.stats.sampling_rate))

        # Baillard implementation of SNR_pick in run_sws.py
        #N_left = ind_center - N_left
        #N_right = N_right - ind_center
        
        # Define noise/signal windows - potentially problematic method
        #ind_left = max(0, ind_center - N_left)
        #ind_right = min(len(trace.data), ind_center + N_right)
        
        # Extract absolute values
        abs_data = np.abs(trace.data)

        # Define noise/signal windows using check_index function from Baillard
        ind_left=check_index(ind_center-N_left,abs_data)
        ind_right=check_index(ind_center+N_right,abs_data)
        
        # Calculate noise (left) and signal (right)
        noise_val = np.mean(abs_data[ind_left:ind_center+1])
        
        if mode == 'mean':
            signal_val = np.mean(abs_data[ind_center:ind_right])
        elif mode == 'max':
            signal_val = np.max(abs_data[ind_center:ind_right])
        else:
            raise ValueError("mode must be 'mean' or 'max'")
        
        # Compute ratio
        if noise_val == 0:
            return np.nan
        
        snr = signal_val / noise_val
        return float(snr)
        
    except Exception as e:
        print(f"  Error calculating Baillard-style SNR: {e}")
        return np.nan


#def compute_snr_for_event_baillard(event_stream, event_row, component='E', 
#                                   N_left=80, N_right=40, mode='mean',
#                                   apply_filter=False, check_p_contamination=True):    
#    """
#    Calculate SNR using Baillard's method with full implementation including
#    P-wave contamination check and filtering.
#    
#    This function follows the exact Baillard workflow from run_sws.py:
#    1. Apply 5-40 Hz bandpass filter
#    2. Calculate sample indices for P and S picks
#    3. Check for P-wave contamination in noise window
#    4. Compute SNR using absolute-value method
#    
#    Parameters:
#    -----------
#    event_stream : obspy.Stream
#        Stream with traces
#    event_row : dict
#        Event metadata with datetime, s_arrival_time, and optionally p_arrival_time
#    component : str
#        Component to analyze ('E', 'N', or 'Z')
#    N_left : int
#        Number of samples before S-pick for noise window (default=80 @ 200Hz = 0.4s)
#    N_right : int
#        Number of samples after S-pick for signal window (default=40 @ 200Hz = 0.2s)
#    mode : str
#        'mean' for average amplitude or 'max' for peak amplitude
#    apply_filter : bool
#        Whether to apply 5-40 Hz bandpass filter (default=True, matches Baillard)
#    check_p_contamination : bool
#        Whether to adjust noise window to avoid P-wave coda (default=True, matches Baillard)
#    
#    Returns:
#    --------
#    float
#        SNR ratio (signal/noise)
#    """
#    try:
#        # Convert to stream if needed
#       if isinstance(event_stream, list):
#           event_stream = obspy.Stream(event_stream)
#        
#        # Find component trace
#        trace = None
#        for tr in event_stream:
#            if tr.stats.channel[-1].upper() == component.upper():
#                break
#        
#       if trace is None:
#            print(f"  No {component} component found")
#            return np.nan
#        
#        print(f"  Component: {component}")
#        
#        # Apply Baillard's filtering if requested
#        if apply_filter:
#            trace.taper(type="hann", max_percentage=0.05)
#            trace.filter("bandpass", freqmin=5, freqmax=40)
#            print(f"  Applied 5-40 Hz bandpass filter")
#        
#        # Extract timing information
#        trace_start_time = trace.stats.starttime
#        sampling_rate = trace.stats.sampling_rate
#        event_time = UTCDateTime(event_row['datetime'])
#        
#        # Calculate S-pick absolute time and sample index
#        s_arrival_offset = float(event_row['s_arrival_time'])
#        s_pick = event_time + s_arrival_offset

#        # This essentially offsets the originao s_arrival_offset by 4 seconds
#        # Trace starts 4 seconds earlier than origin time to give extended window for SNR
#        # Allows proper samples to be picked with respect to the trace
#        s_samples = int(round((s_pick - trace_start_time) * sampling_rate))
#        
#        print(f"  S-pick time: {s_pick}")
#        print(f"  S-pick sample index: {s_samples}")
#        
#        # Calculate P-pick sample index if available (for contamination check)
#        p_samples = None
#        mid_samples = None
#        if check_p_contamination and 'p_arrival_time' in event_row:
#            p_arrival_offset = event_row.get('p_arrival_time')
#            if p_arrival_offset is not None and not np.isnan(p_arrival_offset):
#                p_pick = event_time + float(p_arrival_offset)
#                p_samples = int(round((p_pick - trace_start_time) * sampling_rate))
#                mid_samples = int(round(p_samples + (s_samples - p_samples) / 2))
#                print(f"  P-pick sample index: {p_samples}")
#                print(f"  Midpoint (P-S): {mid_samples}")
#        
#        # Define initial SNR window boundaries (in samples)
#        s_snr_w1 = s_samples - N_left   # Noise window start
#        s_snr_w2 = s_samples + N_right  # Signal window end
#
#        ### Make sure left side of the window is bigger than P+(S-P)/2
#        if s_snr_w1<mid_samples:
#            s_snr_w1=mid_samples
#
#        # Additional Baillard time window constraints
#        s_window=[0.02, 0.3]
#        s_window_time = [s_pick - s_window[0], s_pick + s_window[1]]
#        sw2 = int(round((s_window_time[1] - trace_start_time) * sampling_rate))
#            
#        if s_snr_w2>sw2:
#            s_snr_w2=sw2
#        
#        print(f"  Initial noise window: samples {s_snr_w1} to {s_samples}")
#        print(f"  Initial signal window: samples {s_samples} to {s_snr_w2}")
#        
#       # Apply P-wave contamination check (Baillard's method)
#        if check_p_contamination and mid_samples is not None:
#            if s_snr_w1 < mid_samples:
#                s_snr_w1 = mid_samples
#                print(f"  ⚠ Adjusted noise window to avoid P-wave contamination")
#                print(f"  New noise window start: sample {s_snr_w1}")
#        
#        # Use check_index to ensure boundaries are valid
#        abs_data = np.abs(trace.data)
#        ind_left = check_index(s_snr_w1, abs_data)
#        ind_right = check_index(s_snr_w2, abs_data)
#        
#        # Verify we have valid window ranges
#        if ind_left >= s_samples:
#            print(f"  ✗ Noise window is empty after P-contamination adjustment")
#            return np.nan
#        
#        if s_samples >= ind_right:
#            print(f"  ✗ Signal window is empty")
#            return np.nan
#        
#        # Calculate noise (left of S-pick)
#        noise_val = np.mean(abs_data[ind_left:s_samples+1])
#        
#        # Calculate signal (right of S-pick)
#        if mode == 'mean':
#            signal_val = np.mean(abs_data[s_samples:ind_right])
#        elif mode == 'max':
#            signal_val = np.max(abs_data[s_samples:ind_right])
#        else:
#            raise ValueError("mode must be 'mean' or 'max'")
#        
#        print(f"  Noise amplitude: {noise_val:.6e}")
#        print(f"  Signal amplitude: {signal_val:.6e}")
#        
#        # Compute SNR ratio
#        if noise_val == 0:
#            print(f"  ✗ Zero noise level")
#            return np.nan
#        
#        snr = signal_val / noise_val
#        print(f"  ✓ SNR: {snr:.2f}")
#        
#        return float(snr)
#        
#    except Exception as e:
#        print(f"  ✗ Error calculating Baillard SNR: {e}")
#        import traceback
#        traceback.print_exc()
#        return np.nan
    

def compute_snr_for_event_baillard_orig(event_stream, event_row, 
                                   N_left=80, N_right=40, mode='mean',
                                   apply_filter=False, check_p_contamination=True,
                                   flag_adapt_window=True, flag_adapt_maxlag=True):
    
    s_window=[0.02,0.3]
    fs_window=[0.1,0.3]
    s_snr_window=[0.4,0.2]

    st_x=event_stream.select(channel='??E')
    st_y=event_stream.select(channel='??N')
    st_xy=st_x+st_y
    xy_array=stream2data(st_xy)

    p_time = UTCDateTime(event_row['datetime']) + float(event_row['p_arrival_time'])
    s_time = UTCDateTime(event_row['datetime']) + float(event_row['s_arrival_time'])
    fs_window_time=[s_time-fs_window[0],s_time+fs_window[1]] 
    s_window_time=[s_time-s_window[0],s_time+s_window[1]] 
    trace_start_time=st_xy[0].stats.starttime
    sampling_rate=st_xy[0].stats.sampling_rate
    sampling_rate=event_stream[0].stats.sampling_rate

    s_samples=int(round((s_time-trace_start_time)*sampling_rate)) # in samples
    p_samples=int(round((p_time-trace_start_time)*sampling_rate)) # in samples
    fs_w1=int(round((fs_window_time[0]-trace_start_time)*sampling_rate))
    fs_w2=int(round((fs_window_time[1]-trace_start_time)*sampling_rate))
    mid_samples=int(round(p_samples+(s_samples-p_samples)/2))
    sw1=int(round((s_window_time[0]-trace_start_time)*sampling_rate))
    sw2=int(round((s_window_time[1]-trace_start_time)*sampling_rate))

    if fs_w1<mid_samples:
        fs_w1=mid_samples
                
        ### Cut the data between fs_w1 and fs_w2
        
        xy_array_dom=xy_array[fs_w1:fs_w2,:]
        
        ### Get dominant period on X and Y (given in samples)
        
        (dom_period_x,dom_freq_x)=get_dominant_period_baillard(xy_array_dom[:,0],sampling_rate,flag_plot=False)
        (dom_period_y,dom_freq_y)=get_dominant_period_baillard(xy_array_dom[:,1],sampling_rate,flag_plot=False)
    
        dom_period=np.mean([dom_period_x,dom_period_y]) # Take the mean dominant period
        
        ### Adapt window size to perform splitting and max lag allowed
        
        if flag_adapt_window:
            sw2=int(round(sw1+2*dom_period)) # Check Wuestfeld et al., 2010
        else:
            sw2=int(round((s_window_time[1]-trace_start_time)*sampling_rate))
            
        if flag_adapt_maxlag:
            max_lag=int(round(dom_period))

    s_snr_window_time=[s_time-s_snr_window[0],s_time+s_snr_window[1]] 
    s_snr_w1=int(round((s_snr_window_time[0]-trace_start_time)*sampling_rate))
    s_snr_w2=int(round((s_snr_window_time[1]-trace_start_time)*sampling_rate))

    ### Make sure left side of the window is bigger than P+(S-P)/2
    if s_snr_w1<mid_samples:
        s_snr_w1=mid_samples
        
    if s_snr_w2>sw2:
        s_snr_w2=sw2

    ### Compute

    snr_x=SNR_pick(xy_array[:,0],s_samples,
                        s_samples-s_snr_w1,s_snr_w2-s_samples,mode='mean')
    
    print(f"  SNR for East component: {snr_x:.2f}")
    snr_y=SNR_pick(xy_array[:,1],s_samples,
                        s_samples-s_snr_w1,s_snr_w2-s_samples,mode='mean')
    print(f"  SNR for North component: {snr_y:.2f}")

    ### Feed obs

    s_snr=np.mean((snr_x,snr_y))

    return s_snr

def compute_snr_for_event_baillard(event_stream, event_row, 
                                   N_left=80, N_right=40, mode='mean',
                                   apply_filter=False, check_p_contamination=True,
                                   flag_adapt_window=True, flag_adapt_maxlag=True):
    
    if len(event_stream[0].data) != len(event_stream[1].data) or len(event_stream[0].data) != len(event_stream[2].data) or len(event_stream[1].data) != len(event_stream[2].data):
        print(f"Event {key}: Traces have different lengths, skipping SNR calculation")
        s_snr = np.nan
        return s_snr
    
    event_stream.detrend("linear")
    event_stream.taper(max_percentage=0.05, type='hann')
    event_stream.filter('bandpass', freqmin=5.0, freqmax=40.0)

    s_window = [0.02, 0.3]
    fs_window = [0.1, 0.3]
    s_snr_window = [0.4, 0.2]

    st_x = event_stream.select(channel='??E')
    st_y = event_stream.select(channel='??N')
    st_xy = st_x + st_y

    if len(st_x) == len(st_y) and len(st_x) > 0:
        xy_array = stream2data(st_xy)
    else:
        print(f"Event {key}: Mismatch in number of E and N traces, skipping SNR calculation")
        s_snr = np.nan
        return s_snr

    p_time = UTCDateTime(event_row['datetime']) + float(event_row['p_arrival_time'])
    s_time = UTCDateTime(event_row['datetime']) + float(event_row['s_arrival_time'])
    fs_window_time = [s_time - fs_window[0], s_time + fs_window[1]]
    s_window_time = [s_time - s_window[0], s_time + s_window[1]]
    trace_start_time = st_xy[0].stats.starttime
    sampling_rate = event_stream[0].stats.sampling_rate

    s_samples = int(round((s_time - trace_start_time) * sampling_rate))
    p_samples = int(round((p_time - trace_start_time) * sampling_rate))
    fs_w1 = int(round((fs_window_time[0] - trace_start_time) * sampling_rate))
    fs_w2 = int(round((fs_window_time[1] - trace_start_time) * sampling_rate))
    mid_samples = int(round(p_samples + (s_samples - p_samples) / 2))
    sw1 = int(round((s_window_time[0] - trace_start_time) * sampling_rate))
    max_lag = 60  # initialize with default before potential adaptation

    # Always clamp fs_w1 to avoid P-wave contamination
    if fs_w1 < mid_samples:
        fs_w1 = mid_samples

    # Always calculate dominant period (not gated on fs_w1 < mid_samples)
    xy_array_dom = xy_array[fs_w1:fs_w2, :]

    if xy_array_dom.shape[0] < 2:
        print(f"Event {key}: Not enough data points in dominant period window, skipping SNR calculation")
        s_snr = np.nan
        return s_snr

    (dom_period_x, dom_freq_x) = get_dominant_period_baillard(
        xy_array_dom[:, 0], sampling_rate, flag_plot=False)
    (dom_period_y, dom_freq_y) = get_dominant_period_baillard(
        xy_array_dom[:, 1], sampling_rate, flag_plot=False)
    
    if dom_freq_x <= 0 or dom_freq_y <= 0:
        print(f"Event {key}: Invalid dominant frequency (dom_freq_x={dom_freq_x}, dom_freq_y={dom_freq_y}), skipping SNR calculation")
        s_snr = np.nan
        return s_snr

    dom_period = np.mean([dom_period_x, dom_period_y])

    if np.isnan(dom_period) or dom_period <= 0:
        print(f"Event {key}: Invalid dominant period ({dom_period}), skipping SNR calculation")
        s_snr = np.nan
        return s_snr

    if flag_adapt_window:
        sw2 = int(round(sw1 + 2 * dom_period))
    else:
        sw2 = int(round((s_window_time[1] - trace_start_time) * sampling_rate))

    if flag_adapt_maxlag:
        max_lag = int(round(dom_period))

    # SNR window calculation - always runs with correct sw2
    s_snr_window_time = [s_time - s_snr_window[0], s_time + s_snr_window[1]]
    s_snr_w1 = int(round((s_snr_window_time[0] - trace_start_time) * sampling_rate))
    s_snr_w2 = int(round((s_snr_window_time[1] - trace_start_time) * sampling_rate))

    if s_snr_w1 < mid_samples:
        s_snr_w1 = mid_samples

    if s_snr_w2 > sw2:
        s_snr_w2 = sw2

    # Guard against degenerate windows
    if s_snr_w1 >= s_samples:
        print(f"  Warning: noise window start ({s_snr_w1}) >= S-pick ({s_samples}), returning NaN")
        return np.nan
    if s_snr_w2 <= s_samples:
        print(f"  Warning: signal window end ({s_snr_w2}) <= S-pick ({s_samples}), returning NaN")
        return np.nan

    snr_x = SNR_pick(xy_array[:, 0], s_samples,
                     s_samples - s_snr_w1, s_snr_w2 - s_samples, mode='mean')
    print(f"  SNR for East component: {snr_x:.2f}")

    snr_y = SNR_pick(xy_array[:, 1], s_samples,
                     s_samples - s_snr_w1, s_snr_w2 - s_samples, mode='mean')
    print(f"  SNR for North component: {snr_y:.2f}")

    s_snr = np.mean((snr_x, snr_y))

    return s_snr

# ...existing code...

def calculate_snr_for_organized_waveforms_orig(organized_waveforms):
    """
    Calculate SNR for all events in organized_waveforms and add to the dataset.
    
    All required metadata (S/P arrival times, datetime) is already in organized_waveforms,
    so no external catalog lookup is needed.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
        
    Returns:
    --------
    dict
        Updated organized_waveforms with SNR values added to each event
    """
    
    print(f"Calculating SNR for {len(organized_waveforms)} events in organized_waveforms...")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\n{'='*60}")
        print(f"Processing event {event_id}...")
        print(f"{'='*60}")
        
        # Get traces for this event
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  No traces found for event {event_id}")
            event_data['snr_e'] = np.nan
            event_data['snr_n'] = np.nan
            event_data['snr_horizontal'] = np.nan
            continue
        
        print(f"  Found {len(event_traces)} traces for this event")
        
        # Calculate SNR for horizontal components using event_data directly
        print("\n  Calculating SNR for E component:")
        snr_horizontal = compute_snr_for_event_baillard(event_traces.copy(), event_data)
  
        event_data['snr_horizontal'] = snr_horizontal
        
        # Print summary
        print(f"\n  Final SNR Results:")

        if not np.isnan(snr_horizontal):
            print(f"    Horizontal average: {snr_horizontal:.2f}")
        else:
            print("    Horizontal average: N/A")

        if not np.isnan(snr_horizontal):
            success_count += 1
    
    print(f"\n{'='*60}")
    print("SNR Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid SNR: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    snr_values = [data.get('snr_horizontal', np.nan) for data in organized_waveforms.values()]
    valid_snr = [v for v in snr_values if not np.isnan(v)]
    
    if valid_snr:
        print(f"SNR range: {min(valid_snr):.2f} to {max(valid_snr):.2f}")
        print(f"Mean SNR: {np.mean(valid_snr):.2f}")
        print(f"Median SNR: {np.median(valid_snr):.2f}")
    
    return organized_waveforms

# Redefine calculate_snr_for_organized_waveforms to skip events that return empty spec sequence, then remove them from organized_waveforms

def calculate_snr_for_organized_waveforms(organized_waveforms):
    """
    Calculate SNR for all events in organized_waveforms and add to the dataset.
    
    All required metadata (S/P arrival times, datetime) is already in organized_waveforms,
    so no external catalog lookup is needed.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
        
    Returns:
    --------
    dict
        Updated organized_waveforms with SNR values added to each event
    """
    
    print(f"Calculating SNR for {len(organized_waveforms)} events in organized_waveforms...")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\n{'='*60}")
        print(f"Processing event {event_id}...")
        print(f"{'='*60}")
        
        # Get traces for this event
        event_traces = event_data.get('traces', [])
        if not event_traces:
            print(f"  No traces found for event {event_id}")
            event_data['snr_e'] = np.nan
            event_data['snr_n'] = np.nan
            event_data['snr_horizontal'] = np.nan
            continue
        
        print(f"  Found {len(event_traces)} traces for this event")
        
        # Calculate SNR for horizontal components using event_data directly

        try:
            print("\n  Calculating SNR:")
            snr_horizontal = compute_snr_for_event_baillard(event_traces.copy(), event_data)
    
        except Exception as e:

            print(f"  Error computing SNR for event {event_id}: {e}")
            snr_horizontal = np.nan

        event_data['snr_horizontal'] = snr_horizontal
        
        # Print summary
        print(f"\n  Final SNR Results:")

        if not np.isnan(snr_horizontal):
            print(f"    Horizontal average: {snr_horizontal:.2f}")
        else:
            print("    Horizontal average: N/A")

        if not np.isnan(snr_horizontal):
            success_count += 1
    
    print(f"\n{'='*60}")
    print("SNR Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid SNR: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    snr_values = [data.get('snr_horizontal', np.nan) for data in organized_waveforms.values()]
    valid_snr = [v for v in snr_values if not np.isnan(v)]
    
    if valid_snr:
        print(f"SNR range: {min(valid_snr):.2f} to {max(valid_snr):.2f}")
        print(f"Mean SNR: {np.mean(valid_snr):.2f}")
        print(f"Median SNR: {np.median(valid_snr):.2f}")
    
    else:
        # Remove events that have NaN SNR values from organized_waveforms
        print("No valid SNR values found, removing events with NaN SNR from organized_waveforms...")
        events_to_remove = [eid for eid, data in organized_waveforms.items() if np.isnan(data.get('snr_horizontal', np.nan))]
        for eid in events_to_remove:
            del organized_waveforms[eid]
        print(f"Removed {len(events_to_remove)} events with NaN SNR values")
        print(f"Remaining events after SNR QC: {len(organized_waveforms)}")
    return organized_waveforms

#def calculate_back_azimuth(eq_lat, eq_lon, sta_lat, sta_lon):
    """
    Calculate back-azimuth from earthquake to station using Cartesian projection method
    (consistent with Christian's get_baz_trigo implementation)
    
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
    # Use fixed reference point (ini_lon, ini_lat) as in Christian's implementation
    ini_lon = -130.1
    ini_lat = 45.9
    
    # Project to Cartesian coordinates (km) using fixed reference point
    cos_ref_lat = np.cos(np.radians(ini_lat))
    x_event = (eq_lon - ini_lon) * 111.32 * cos_ref_lat  # longitude to km
    y_event = (eq_lat - ini_lat) * 110.54                # latitude to km
    x_sta = (sta_lon - ini_lon) * 111.32 * cos_ref_lat
    y_sta = (sta_lat - ini_lat) * 110.54
    
    # Calculate back-azimuth using Cartesian coordinates (earthquake to station)
    baz_rad = np.arctan2((y_event - y_sta), (x_event - x_sta))
    
    # Ensure positive angle (0 to 2π)
    baz_rad = baz_rad + 2 * np.pi if baz_rad < 0 else baz_rad
    
    # Convert to degrees
    back_az = np.degrees(baz_rad)
    
    return back_az

def calculate_back_azimuth(eq_lat, eq_lon, sta_lat, sta_lon):
    """
    Calculate back-azimuth using the same convention as Baillard's get_baz_trigo:
    compute trigo angle via ll2xy (atan2(dy,dx), radians CCW from +X/East),
    then convert to degrees clockwise from North (0-360°).
    """
    import projection as gproj  # uses same projection helper as shearwavesplit.get_baz_trigo

    # Use the same reference point used by Baillard's code (if different, match it)
    ini_lon = -130.1
    ini_lat = 45.9

    # Use ll2xy to get projected coordinates (matches Baillard)
    [x_event, x_sta], [y_event, y_sta] = gproj.ll2xy([eq_lon, sta_lon], [eq_lat, sta_lat], ini_lon, ini_lat)

    # trigo angle: atan2(dy, dx) -> radians, CCW from +X (East)
    baz_trigo = np.arctan2((y_event - y_sta), (x_event - x_sta))
    if baz_trigo < 0:
        baz_trigo += 2 * np.pi

    # convert to degrees clockwise from North (standard back-azimuth)
    back_az = (90.0 - np.degrees(baz_trigo)) % 360.0
    #back_az = np.rad2deg(baz_trigo)

    return back_az


def calculate_back_azimuth_for_organized_waveforms(organized_waveforms, stations_df):
    """
    Calculate back-azimuth for all events in organized_waveforms and add to the dataset.
    
    All required event metadata (lat, lon, station) is already in organized_waveforms,
    so no external catalog lookup is needed.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary with event IDs as keys, containing event data, traces, and metadata
    stations_df : pandas.DataFrame
        Station information with coordinates (station, latitude, longitude)
        
    Returns:
    --------
    dict
        Updated organized_waveforms with back-azimuth values added to each event
    """
    
    print(f"Calculating back-azimuth for {len(organized_waveforms)} events...")
    
    success_count = 0
    
    for event_id, event_data in organized_waveforms.items():
        print(f"\nProcessing event {event_id}...")
        
        # Get event location from event_data
        event_lat = event_data.get('latitude')
        event_lon = event_data.get('longitude')
        
        if event_lat is None or event_lon is None:
            print(f"  No event location found in event_data")
            event_data['back_azimuth'] = np.nan
            continue
        
        # Get station name from event data
        station_name = event_data.get('station')
        
        if station_name is None:
            print(f"  No station information for event {event_id}")
            event_data['back_azimuth'] = np.nan
            continue
        
        # Find station coordinates
        station_info = stations_df[stations_df['Station ID'] == station_name]
        
        if station_info.empty:
            print(f"  No coordinates found for station {station_name}")
            event_data['back_azimuth'] = np.nan
            continue
        
        station_lat = station_info.iloc[0]['Latitude (°N)']
        station_lon = station_info.iloc[0]['Longitude (°W)']
        
        # Calculate back-azimuth (from station to event)
        back_az = calculate_back_azimuth(event_lat, event_lon, station_lat, station_lon)
        
        # Add to event_data
        event_data['back_azimuth'] = back_az
        
        print(f"  Station: {station_name}")
        print(f"  Event location: ({event_lat:.4f}, {event_lon:.4f})")
        print(f"  Station location: ({station_lat:.4f}, {station_lon:.4f})")
        print(f"  Back-azimuth: {back_az:.2f}°")
        
        success_count += 1
    
    print(f"\n{'='*60}")
    print("Back-Azimuth Calculation Complete")
    print(f"{'='*60}")
    print(f"Events with valid back-azimuth: {success_count}/{len(organized_waveforms)}")
    
    # Calculate statistics
    back_az_values = [data.get('back_azimuth', np.nan) for data in organized_waveforms.values()]
    valid_back_az = [v for v in back_az_values if not np.isnan(v)]
    
    if valid_back_az:
        print(f"Back-azimuth range: {min(valid_back_az):.2f}° to {max(valid_back_az):.2f}°")
        print(f"Mean back-azimuth: {np.mean(valid_back_az):.2f}°")
    
    return organized_waveforms

import numpy as np
from obspy.signal.spectral_estimation import PPSD

# Check for optional mtspec library
try:
    from mtspec import mtspec
    MTSPEC_AVAILABLE = True
except ImportError:
    MTSPEC_AVAILABLE = False
    print("mtspec not available, using ObsPy PPSD method")

def estimate_dominant_period(trace, method='obspy'):
    """
    Estimate the dominant period of a seismic trace using spectral analysis.
    
    Parameters:
    -----------
    trace : obspy.Trace
        Input seismic trace
    method : str
        'obspy' for ObsPy PPSD or 'mtspec' for multitaper method
        
    Returns:
    --------
    float
        Dominant period in seconds
    """
    if trace is None or len(trace.data) == 0:
        return np.nan
    
    # Preprocess trace
    trace_copy = trace.copy()
    trace_copy.detrend("linear")
    trace_copy.taper(type="hann", max_percentage=0.05)
    
    try:
        if method == 'mtspec' and MTSPEC_AVAILABLE:
            # Multitaper spectral estimation
            spec, freq = mtspec(
                data=trace_copy.data, 
                delta=trace_copy.stats.delta, 
                time_bandwidth=4, 
                number_of_tapers=7
            )
            dominant_freq = freq[np.argmax(spec)]
        else:
            # ObsPy PPSD method (fallback)
            df = trace_copy.stats.sampling_rate
            nfft = 2 ** int(np.ceil(np.log2(len(trace_copy.data))))
            freq, power = PPSD(trace_copy.data, NFFT=nfft, Fs=df, method="multitaper")
            dominant_freq = freq[np.argmax(power)]
        
        # Convert to dominant period
        t_dom = 1.0 / dominant_freq if dominant_freq > 0 else np.nan
        return t_dom
    
    except Exception as e:
        print(f"Error estimating dominant period: {e}")
        return np.nan

def calculate_dynamic_parameters(event_data, s_arrival_buffer=1.0):
    """
    Calculate dynamic analysis parameters for an event using spectral analysis.
    
    All required data (traces, datetime, s_arrival_time) is in the event_data from organized_waveforms.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing traces and metadata
    s_arrival_buffer : float
        Buffer around S-arrival for analysis window (seconds)
        
    Returns:
    --------
    dict
        Dictionary containing dynamic parameters
    """
    try:
        # Get traces from event_data
        event_traces = event_data.get('traces', [])
        if not event_traces:
            raise ValueError("No traces in event_data")
        
        # Convert to stream if needed
        if isinstance(event_traces, list):
            event_stream = obspy.Stream(event_traces)
        else:
            event_stream = event_traces
        
        # Get S-arrival time
        event_time = UTCDateTime(event_data['datetime'])
        s_pick = event_time + float(event_data['s_arrival_time'])
        
        # Extract traces around S-arrival for spectral analysis
        analysis_start = s_pick - s_arrival_buffer
        analysis_end = s_pick + s_arrival_buffer
        
        # Get horizontal components for analysis
        trace_e = None
        trace_n = None
        
        for tr in event_stream:
            component = tr.stats.channel[-1].upper()
            if component in ['E', '2']:
                trace_e = tr.slice(analysis_start, analysis_end)
            elif component in ['N', '1']:
                trace_n = tr.slice(analysis_start, analysis_end)
        
        # Calculate dominant periods for both horizontal components
        t_dom_e = estimate_dominant_period(trace_e) if trace_e and len(trace_e.data) > 10 else np.nan
        t_dom_n = estimate_dominant_period(trace_n) if trace_n and len(trace_n.data) > 10 else np.nan
        
        # Use average or the valid one
        if not np.isnan(t_dom_e) and not np.isnan(t_dom_n):
            t_dom = (t_dom_e + t_dom_n) / 2
        elif not np.isnan(t_dom_e):
            t_dom = t_dom_e
        elif not np.isnan(t_dom_n):
            t_dom = t_dom_n
        else:
            t_dom = 0.1  # Default fallback (10 Hz dominant frequency)
        
        # Calculate dynamic window parameters
        # Window should be several dominant periods long, but within reasonable bounds
        dynamic_window_length = max(0.5, min(2.0, 4 * t_dom))  # 4 periods, but between 0.5-2.0s
        
        # Calculate optimal filter frequencies
        # Center frequency around 1/T_dom, with reasonable bandwidth
        center_freq = 1.0 / t_dom
        freq_min = max(1.0, center_freq * 0.5)  # Lower bound at least 1 Hz
        freq_max = min(50.0, center_freq * 2.5)  # Upper bound at most 50 Hz
        
        return {
            't_dom': t_dom,
            't_dom_e': t_dom_e,
            't_dom_n': t_dom_n,
            'dynamic_window_length': dynamic_window_length,
            'optimal_freq_min': freq_min,
            'optimal_freq_max': freq_max,
            'center_frequency': center_freq
        }
        
    except Exception as e:
        print(f"Error calculating dynamic parameters: {e}")
        return {
            't_dom': 0.1,
            'dynamic_window_length': 1.0,
            'optimal_freq_min': 5.0,
            'optimal_freq_max': 40.0,
            'center_frequency': 10.0
        }
    
def create_splitting_analysis(event_data, first_window_start, last_window_start, first_window_end, last_window_end, n_win, s_pick_uncertainty):
    """
    Create a SWSPy splitting object from organized_waveforms event data.
    
    All required data (traces, station, back_azimuth, incidence, s_arrival_time) is 
    contained in event_data from organized_waveforms.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing:
        - traces: ObsPy stream with waveform data
        - station: Station name/ID
        - back_azimuth: Back-azimuth from event to station (degrees)
        - incidence_eigenvalue_jurkevics: P-wave incidence angle (degrees from vertical)
        - s_arrival_time: S-wave arrival time (seconds from origin)
        - datetime: Event origin time (UTCDateTime compatible string)
    use_dynamic_params : bool, optional
        Whether to calculate and use dynamic windowing/filtering parameters
        based on spectral analysis (default=True)
        
    Returns:
    --------
    swspy.splitting object
        Splitting object ready for analysis with optimized parameters
    """
    
    # Get traces from event_data
    event_traces = event_data.get('traces', [])
    if not event_traces:
        raise ValueError("No traces in event_data")
    
    # Convert to stream if needed
    if isinstance(event_traces, list):
        stream = obspy.Stream(event_traces)
    else:
        stream = event_traces.copy()
    
    # Apply optimal filtering to the stream
    #print(f"  Applying bandpass filter: {freq_min:.1f}-{freq_max:.1f} Hz")
    stream_filtered = stream.copy()
    #stream_filtered.filter("bandpass", freqmin=freq_min, freqmax=freq_max)
    
    # Extract required metadata from event_data
    station_name = event_data.get('station', 'UNKNOWN')
    back_azimuth = event_data.get('back_azimuth')
    incidence_angle = event_data.get('incidence_eigenvalue_jurkevics')
    
    # Validate required fields
    if back_azimuth is None or np.isnan(back_azimuth):
        raise ValueError(f"Missing or invalid back_azimuth for station {station_name}")
    if incidence_angle is None or np.isnan(incidence_angle):
        raise ValueError(f"Missing or invalid incidence angle for station {station_name}")
    
    # Calculate S-arrival absolute time
    event_time = UTCDateTime(event_data['datetime'])
    s_arrival_time = float(event_data['s_arrival_time'])
    s_arrival_absolute = event_time + s_arrival_time
    
    print(f"  Creating SWSPy splitting object...")
    print(f"    Station: {station_name}")
    print(f"    Back-azimuth: {back_azimuth:.2f}°")
    print(f"    Incidence: {incidence_angle:.2f}°")
    print(f"    S-arrival: {s_arrival_absolute}")
    
    # Create splitting object with SWSPy using exact pattern from user
    # origin_times = [UTCDateTime(event_data['datetime'])]
    #P_phase_arrival_times=[p_arrival_absolute],

    splitting_event = swspy.splitting.create_splitting_object(
        stream_filtered, 
        stations_in=[station_name],
        back_azis_all_stations=[back_azimuth],
        receiver_inc_angles_all_stations=[incidence_angle],
        S_phase_arrival_times=[s_arrival_absolute],
        origin_times = [event_time],
        first_window_start=first_window_start, 
        last_window_start=last_window_start, 
        first_window_end=first_window_end, 
        last_window_end=last_window_end, n_win=n_win, 
        s_pick_uncertainty=s_pick_uncertainty
    )
    
    return splitting_event
    
def create_splitting_analysis_orig(event_data, first_window_start, last_window_start, first_window_end, last_window_end, n_win, s_pick_uncertainty):
    """
    Create a SWSPy splitting object from organized_waveforms event data.
    
    All required data (traces, station, back_azimuth, incidence, s_arrival_time) is 
    contained in event_data from organized_waveforms.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing:
        - traces: ObsPy stream with waveform data
        - station: Station name/ID
        - back_azimuth: Back-azimuth from event to station (degrees)
        - incidence_eigenvalue_jurkevics: P-wave incidence angle (degrees from vertical)
        - s_arrival_time: S-wave arrival time (seconds from origin)
        - datetime: Event origin time (UTCDateTime compatible string)
    use_dynamic_params : bool, optional
        Whether to calculate and use dynamic windowing/filtering parameters
        based on spectral analysis (default=True)
        
    Returns:
    --------
    swspy.splitting object
        Splitting object ready for analysis with optimized parameters
    """
    
    # Get traces from event_data
    event_traces = event_data.get('traces', [])
    if not event_traces:
        raise ValueError("No traces in event_data")
    
    # Convert to stream if needed
    if isinstance(event_traces, list):
        stream = obspy.Stream(event_traces)
    else:
        stream = event_traces.copy()
    
    # Apply optimal filtering to the stream
    #print(f"  Applying bandpass filter: {freq_min:.1f}-{freq_max:.1f} Hz")
    stream_filtered = stream.copy()
    #stream_filtered.filter("bandpass", freqmin=freq_min, freqmax=freq_max)
    
    # Extract required metadata from event_data
    station_name = event_data.get('station', 'UNKNOWN')
    back_azimuth = event_data.get('back_azimuth')
    incidence_angle = event_data.get('incidence_eigenvalue_jurkevics')
    
    # Validate required fields
    if back_azimuth is None or np.isnan(back_azimuth):
        raise ValueError(f"Missing or invalid back_azimuth for station {station_name}")
    if incidence_angle is None or np.isnan(incidence_angle):
        raise ValueError(f"Missing or invalid incidence angle for station {station_name}")
    
    # Calculate S-arrival absolute time
    event_time = UTCDateTime(event_data['datetime'])
    s_arrival_time = float(event_data['s_arrival_time'])
    s_arrival_absolute = event_time + s_arrival_time
    
    print(f"  Creating SWSPy splitting object...")
    print(f"    Station: {station_name}")
    print(f"    Back-azimuth: {back_azimuth:.2f}°")
    print(f"    Incidence: {incidence_angle:.2f}°")
    print(f"    S-arrival: {s_arrival_absolute}")
    
    # Create splitting object with SWSPy using exact pattern from user
    # origin_times = [UTCDateTime(event_data['datetime'])]
    #P_phase_arrival_times=[p_arrival_absolute],

    splitting_event = swspy.splitting.create_splitting_object(
        stream_filtered, 
        stations_in=[station_name],
        back_azis_all_stations=[back_azimuth],
        receiver_inc_angles_all_stations=[incidence_angle],
        S_phase_arrival_times=[s_arrival_absolute],
        first_window_start=first_window_start, 
        last_window_start=last_window_start, 
        first_window_end=first_window_end, 
        last_window_end=last_window_end, n_win=n_win, 
        s_pick_uncertainty=s_pick_uncertainty
    )
    
    return splitting_event

def plot_three_component_seismogram(event_data, event_id=None, figsize=(12, 8), 
                                   time_window=(0, 5), filter_data=True, 
                                   lowfreq=5.0, highfreq=40.0):
    """
    Plot 3-component seismogram with P- and S-wave arrival markers for an event from organized_waveforms.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms containing:
        - traces: ObsPy stream with Z/N/E components
        - station: Station name
        - p_arrival_time: P-wave arrival time (seconds from origin)
        - s_arrival_time: S-wave arrival time (seconds from origin)
        - magnitude: Event magnitude (optional)
        - datetime: Event datetime (optional)
    event_id : str or int, optional
        Event ID for plot title (extracted from event_data if not provided)
    figsize : tuple, optional
        Figure size (width, height) in inches, default=(12, 8)
    time_window : tuple, optional
        Time window to display (start, end) in seconds, default=(0, 5)
    filter_data : bool, optional
        Whether to apply bandpass filter, default=True
    lowfreq : float, optional
        Low-frequency corner for bandpass filter (Hz), default=5.0
    highfreq : float, optional
        High-frequency corner for bandpass filter (Hz), default=40.0
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    
    # Get traces from event_data
    event_traces = event_data.get('traces', [])
    if not event_traces:
        print("No traces found in event_data")
        return None
    
    # Convert to stream if needed
    if isinstance(event_traces, list):
        stream = obspy.Stream(event_traces)
    else:
        stream = event_traces.copy()
    
    # Apply filter if requested
    if filter_data:
        stream.filter("bandpass", freqmin=lowfreq, freqmax=highfreq)
    
    # Get event metadata
    station = event_data.get('station', 'UNKNOWN')
    if event_id is None:
        event_id = event_data.get('event_id', 'UNKNOWN')
    
    p_arrival_time = event_data.get('p_arrival_time', np.nan)
    s_arrival_time = event_data.get('s_arrival_time', np.nan)
    magnitude = event_data.get('magnitude', np.nan)
    datetime_str = event_data.get('datetime', '')
    
    # Setup plot
    fig = plt.figure(figsize=figsize)
    
    components = {'Z': 'Vertical', 'N': 'North-South', 'E': 'East-West'}
    colors = {'Z': 'black', 'N': 'black', 'E': 'black'}
    
    # Plot each component
    for i, comp in enumerate(components.keys(), 1):
        # Find the component (handle both N/E and 1/2 naming)
        comp_trace = None
        for tr in stream:
            channel_comp = tr.stats.channel[-1].upper()
            if comp == 'Z' and channel_comp == 'Z':
                comp_trace = tr
                break
            elif comp == 'N' and channel_comp in ['N', '1']:
                comp_trace = tr
                break
            elif comp == 'E' and channel_comp in ['E', '2']:
                comp_trace = tr
                break
        
        plt.subplot(3, 1, i)
        
        if comp_trace is not None:
            # Plot the trace
            times = comp_trace.times()
            plt.plot(times, comp_trace.data, color=colors.get(comp, 'black'), linewidth=1)
            
            # Set time limits
            plt.xlim(time_window[0], time_window[1])
            
            # Add P-arrival marker if available
            if not np.isnan(p_arrival_time):
                if time_window[0] <= p_arrival_time <= time_window[1]:
                    plt.axvline(x=p_arrival_time, color='blue', linestyle='--', 
                               linewidth=1.5, label='P-arrival', alpha=0.7)
                    # Position text slightly above the data
                    y_pos = comp_trace.data.max() * 0.85
                    plt.text(p_arrival_time, y_pos, 'P', color='blue', 
                            fontsize=12, fontweight='bold', ha='center')
            
            # Add S-arrival marker if available
            if not np.isnan(s_arrival_time):
                if time_window[0] <= s_arrival_time <= time_window[1]:
                    plt.axvline(x=s_arrival_time, color='red', linestyle='--', 
                               linewidth=1.5, label='S-arrival', alpha=0.7)
                    # Position text slightly above the data
                    y_pos = comp_trace.data.max() * 0.85
                    plt.text(s_arrival_time, y_pos, 'S', color='red', 
                            fontsize=12, fontweight='bold', ha='center')
            
            plt.title(f"{components[comp]} Component ({comp})", fontsize=11)
            plt.ylabel("Amplitude", fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add legend on first subplot
            if i == 1:
                handles = []
                labels = []
                if not np.isnan(p_arrival_time):
                    handles.append(plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5))
                    labels.append('P-arrival')
                if not np.isnan(s_arrival_time):
                    handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5))
                    labels.append('S-arrival')
                if handles:
                    plt.legend(handles, labels, loc='upper right', fontsize=9)
        else:
            plt.xlim(time_window[0], time_window[1])
            plt.text(0.5, 0.5, f"No {components[comp]} component data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=10)
            plt.ylabel("Amplitude", fontsize=10)
            plt.grid(True, alpha=0.3)
        
        # Only add x-label to bottom plot
        if i == 3:
            plt.xlabel("Time (s)", fontsize=10)
    
    # Create title with event information
    title_parts = [f"Event {event_id} at Station {station}"]
    if not np.isnan(magnitude):
        title_parts.append(f"M{magnitude:.1f}")
    if datetime_str:
        title_parts.append(datetime_str)
    
    title = " | ".join(title_parts)
    if filter_data:
        title += f" | Filtered {lowfreq}-{highfreq} Hz"
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    return fig


def plot_multiple_events_seismograms(organized_waveforms, event_ids=None, max_events=4, **kwargs):
    """
    Plot 3-component seismograms for multiple events from organized_waveforms.
    
    Parameters:
    -----------
    organized_waveforms : dict
        Dictionary of events from organized_waveforms
    event_ids : list, optional
        List of specific event IDs to plot. If None, plots first max_events
    max_events : int, optional
        Maximum number of events to plot if event_ids not specified, default=4
    **kwargs : dict
        Additional keyword arguments passed to plot_three_component_seismogram()
        
    Returns:
    --------
    list
        List of figure objects created
    """
    
    # Determine which events to plot
    if event_ids is None:
        # Plot first max_events
        event_ids = list(organized_waveforms.keys())[:max_events]
    
    figures = []
    
    for event_id in event_ids:
        if event_id not in organized_waveforms:
            print(f"Warning: Event {event_id} not found in organized_waveforms")
            continue
        
        event_data = organized_waveforms[event_id]
        
        # Add event_id to event_data if not present
        if 'event_id' not in event_data:
            event_data['event_id'] = event_id
        
        print(f"\nPlotting event {event_id}...")
        fig = plot_three_component_seismogram(event_data, event_id=event_id, **kwargs)
        
        if fig is not None:
            figures.append(fig)
            plt.show()
        else:
            print(f"Failed to create plot for event {event_id}")
    
    print(f"\nCreated {len(figures)} seismogram plots")
    return figures

def stream2data(st):
    """
    Convert stream into MxN numpy array
    N= stream number
    M= sample number
    
    Input:
        st: obspy stream object: must have same number of samples to be stacked into a matrix
        
    Output:
        big_data: MxN np.array with data
    """
    big_data=[]
    for trace in st:
        data=trace.data
        big_data.append(data)
        
    big_data=np.column_stack(big_data)
    
    
    return big_data


def regenerate_lambda_surface_for_event(event_data, 
                                        s_window=[0.02, 0.3],
                                        min_lag=0, max_lag=60,
                                        Nlags=60, Nangles=90,
                                        flag_adapt_window=True,
                                        flag_adapt_maxlag=True):
    """
    Regenerate the lambda surface (LAMBDA1, LAMBDA2, LAGS, ANGLES) for a given event.
    
    This function extracts the horizontal components, calculates adaptive window parameters,
    and performs the eigenvalue grid search to generate the parameter space for visualization.
    
    Parameters:
    -----------
    event_data : dict
        Event data from organized_waveforms
    s_window : list [before, after]
        Base S-window in seconds [time before S, time after S]
    min_lag : int
        Minimum delay time in samples
    max_lag : int
        Maximum delay time in samples
    Nlags : int
        Number of lag values to test
    Nangles : int
        Number of angle values to test
    flag_adapt_window : bool
        Whether to adapt window based on dominant period
    flag_adapt_maxlag : bool
        Whether to adapt max_lag based on dominant period
        
    Returns:
    --------
    dict containing:
        - LAMBDA1, LAMBDA2: eigenvalue surfaces
        - LAGS, ANGLES: coordinate meshgrids
        - sw1, sw2: analysis window samples
        - min_lag, max_lag: actual lag range used
        - sampling_rate: trace sampling rate
        - xy_array: horizontal component data
    """
    
    # Extract horizontal components
    event_traces = event_data.get('traces', [])
    if not event_traces:
        raise ValueError("No traces in event_data")
    
    if isinstance(event_traces, list):
        st = obspy.Stream(event_traces)
    else:
        st = event_traces
    
    # Get E and N components
    st_x = st.select(channel='??E')
    st_y = st.select(channel='??N')
    
    if len(st_x) == 0 or len(st_y) == 0:
        raise ValueError("Missing E or N component")
    
    trace_x = st_x[0]
    trace_y = st_y[0]
    
    # Create xy_array [N_samples x 2]
    xy_array = np.column_stack([trace_x.data, trace_y.data])
    
    # Get timing information
    trace_start_time = trace_x.stats.starttime
    sampling_rate = trace_x.stats.sampling_rate
    event_time = UTCDateTime(event_data['datetime'])
    s_time = event_time + event_data['s_arrival_time']
    p_time = event_time + event_data.get('p_arrival_time', event_data['s_arrival_time'] - 1.0)
    
    # Calculate window parameters
    sw1, sw2, min_lag, max_lag, T_dom = calculate_baillard_window_params(
        xy_array, s_time, p_time, trace_start_time, sampling_rate,
        s_window_base=s_window, min_lag_base=min_lag, max_lag_base=max_lag,
        flag_adapt_window=flag_adapt_window, flag_adapt_maxlag=flag_adapt_maxlag)
    
    # Baillard adds padding to lag range
    add_lag = 3
    
    # Grid search
    LAMBDA1, LAMBDA2, LAGS, ANGLES = get_LAMBDAS_baillard(
        xy_array, min_lag-add_lag, max_lag+add_lag, Nlags, Nangles, sw1, sw2)
    
    return {
        'LAMBDA1': LAMBDA1,
        'LAMBDA2': LAMBDA2,
        'LAGS': LAGS,
        'ANGLES': ANGLES,
        'sw1': sw1,
        'sw2': sw2,
        'min_lag': min_lag,
        'max_lag': max_lag,
        'sampling_rate': sampling_rate,
        'xy_array': xy_array,
        'T_dom': T_dom
    }


def plot_lambda_comparison(lambda_data, 
                           baillard_result, 
                           swspy_result,
                           event_data,
                           min_thres=0.5,
                           quality_thres=0.2,
                           zoom_factor=[4, 4],
                           figsize=(14, 10)):
    """
    Plot lambda surface with both Baillard and SWSPy solutions overlaid.
    
    This function visualizes the parameter space (delay time vs fast direction)
    with the lambda2 eigenvalue surface, showing both splitting solutions and
    their uncertainty regions to assess method agreement.
    
    Parameters:
    -----------
    lambda_data : dict
        Output from regenerate_lambda_surface_for_event()
    baillard_result : dict
        Result dictionary from Baillard analysis
    swspy_result : dict
        Result dictionary from SWSPy analysis
    event_data : dict
        Event metadata from passing_waveforms
    min_thres : float
        Threshold for contour plotting (0-1 normalized)
    quality_thres : float
        Quality threshold for mesh quality calculation
    zoom_factor : list
        Zoom factors for interpolation [rows, cols]
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib figure
        The generated figure
    """
    
    # Extract lambda surface data
    LAMBDA2 = lambda_data['LAMBDA2']
    LAMBDA1 = lambda_data['LAMBDA1']
    LAGS = lambda_data['LAGS']
    ANGLES = lambda_data['ANGLES']
    min_lag = lambda_data['min_lag']
    max_lag = lambda_data['max_lag']
    sampling_rate = lambda_data['sampling_rate']
    
    # Zoom for better contour quality
    LAMBDA2_zoom = zoom(LAMBDA2, zoom_factor)
    LAMBDA1_zoom = zoom(LAMBDA1, zoom_factor)
    
    # Resample LAG grid
    lagsz = np.linspace(np.min(LAGS), np.max(LAGS), LAGS.shape[1] * zoom_factor[1])
    LAGSZ = np.tile(lagsz, (zoom_factor[0] * LAGS.shape[0], 1))
    
    # Normalize
    MESH = minmax2zeroone(LAMBDA2_zoom)
    
    # Get quality
    quality = quality_mesh(MESH, threshold=quality_thres)
    
    # Set up extent for plotting
    extent_lag = [np.min(LAGS), np.max(LAGS)]
    extent_angle = [np.min(ANGLES), np.max(ANGLES)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot lambda2 surface
    im = ax.imshow(MESH, extent=extent_lag + extent_angle, origin='lower',
                   cmap=plt.cm.get_cmap('magma'), aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized λ₂', fontsize=12, fontweight='bold')
    
    # Plot contours
    CS = ax.contour(MESH, extent=extent_lag + extent_angle, origin='lower',
                   levels=[min_thres], colors='white', linestyles=':', linewidths=2)
    ax.clabel(CS, [min_thres], fmt='%.2f', fontsize=10)
    
    # Plot lag thresholds
    ax.axvline(min_lag, 0, 1, color='white', ls='--', linewidth=2, alpha=0.7, label='Lag bounds')
    ax.axvline(max_lag, 0, 1, color='white', ls='--', linewidth=2, alpha=0.7)
    
    # Convert SWSPy result to Baillard coordinate system
    # SWSPy uses degrees clockwise from North
    # Baillard uses radians counterclockwise from East
    # Conversion: rad = (90 - deg) * pi/180
    swspy_phi_deg = -1 * (swspy_result['phi'] + 90)
    swspy_phi_rad = swspy_phi_deg * np.pi/180
    swspy_dt_samples = swspy_result['dt'] * sampling_rate
    
    # Baillard result (already in correct units)
    baillard_phi_rad = baillard_result['phi_rad']
    baillard_dt_samples = baillard_result['dt_samples']
    
    # Plot Baillard solution
    ax.plot(baillard_dt_samples, baillard_phi_rad, 'o', 
            color='red', markersize=15, markeredgewidth=1,
            markeredgecolor='black', 
            label='Baillard', zorder=10)
    ax.text(baillard_dt_samples, baillard_phi_rad + 0.05, 'B',
            color='white', fontsize=14, fontweight='bold', 
            ha='center', va='bottom', zorder=11)
    
    # Plot SWSPy solution
    ax.plot(swspy_dt_samples, swspy_phi_rad, 'o', 
            color='blue', markersize=15, markeredgewidth=1,
            markeredgecolor='black', label='SWSPy', zorder=10)
    ax.text(swspy_dt_samples, swspy_phi_rad + 0.05, 'S',
            color='white', fontsize=14, fontweight='bold',
            ha='center', va='bottom', zorder=11)
    
    # Add error bars if available
    if not np.isnan(baillard_result.get('phi_error', np.nan)) and not np.isnan(baillard_result.get('dt_error', np.nan)):
        phi_err_rad = np.radians(baillard_result['phi_error'])
        dt_err_samples = baillard_result['dt_error'] * sampling_rate
        ax.errorbar(baillard_dt_samples, baillard_phi_rad,
                   xerr=dt_err_samples, yerr=phi_err_rad,
                   color='red', linewidth=2, capsize=5, capthick=2,
                   alpha=0.7, zorder=9)
        
    # Add error bars for SWSPy if available
    if not np.isnan(swspy_result.get('phi_error', np.nan)) and not np.isnan(swspy_result.get('dt_error', np.nan)):
        swspy_phi_err_rad = np.radians(swspy_result['phi_error'])
        swspy_dt_err_samples = swspy_result['dt_error'] * sampling_rate
        ax.errorbar(swspy_dt_samples, swspy_phi_rad,
                   xerr=swspy_dt_err_samples, yerr=swspy_phi_err_rad,
                   color='blue', linewidth=2, capsize=5, capthick=2,
                   alpha=0.7, zorder=9)
    
    # Labels and title
    ax.set_xlabel('Delay Time (samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fast Direction (radians)', fontsize=12, fontweight='bold')
    
    station = event_data.get('station', 'UNKNOWN')
    event_time = event_data.get('datetime', 'UNKNOWN')
    depth = event_data.get('depth', np.nan)
    
    title = (f"Splitting Parameter Space Comparison\n"
            f"Station: {station} | Event: {event_time}\n"
            f"Depth: {depth:.2f} km | Quality: {quality:.2f}\n"
            f"Baillard: φ={baillard_result['phi']:.1f}°, δt={baillard_result['dt']:.3f}s | "
            f"SWSPy: φ={swspy_phi_deg:.1f}°, δt={swspy_result['dt']:.3f}s")
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, color='white', linestyle=':')
    
    plt.tight_layout()
    
    return fig

def plot_fast_direction_rose_orig(results_dict, title="Fast Direction Distribution", 
                              nbins=18, figsize=(8, 8), color='steelblue',
                              edgecolor='black', linewidth=0.5):
    """
    Create a polar rose plot (histogram) of fast directions from splitting results.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of splitting results with event_id as keys
    title : str
        Title for the plot
    nbins : int
        Number of angular bins (default 18 = 10° bins for ±90°)
    figsize : tuple
        Figure size (width, height)
    color : str
        Color for the histogram bars
    edgecolor : str
        Color for bar edges
    linewidth : float
        Width of bar edges
    """
    # Extract fast directions (phi) from results
    fast_directions = []
    for event_id, result in results_dict.items():
        phi = result['result']['phi']  # in degrees
        # Convert to radians
        phi_rad = np.deg2rad(phi)
        fast_directions.append(phi_rad)
    
    fast_directions = np.array(fast_directions)
    
    # Create polar histogram
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Create histogram bins (-pi/2 to pi/2 for -90° to +90°)
    bins = np.linspace(-np.pi/2, np.pi/2, nbins + 1)
    
    # Calculate histogram
    counts, bin_edges = np.histogram(fast_directions, bins=bins)
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Width of each bar
    width = np.pi / nbins
    
    # Create the rose plot
    bars = ax.bar(bin_centers, counts, width=width, bottom=0.0,
                   color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=0.7)
    
    # Set theta direction (clockwise from North)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Set angular limits (-90° to +90°)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    
    # Set radial ticks
    ax.set_rlabel_position(0)
    
    # Add degree labels
    tick_labels = ['-90°', '-60°', '-30°', '0°', '30°', '60°', '90°']
    tick_positions = np.deg2rad([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 375)
    
    # Add title with statistics
    n_measurements = len(fast_directions)
    # Calculate circular mean for ±90° range
    mean_direction = np.rad2deg(np.arctan2(np.sin(fast_directions).sum(), 
                                           np.cos(fast_directions).sum()))
    
    title_text = f"{title}\nN = {n_measurements}, Mean = {mean_direction:.1f}°"
    ax.set_title(title_text, va='bottom', fontsize=12, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig, ax

def plot_splitting_timeseries_smooth(results_dict, qc_metrics_df, station='AXAS2', 
                                     figsize=(14, 8), x_width_days=5, x_overlap=0.95,
                                     y_width_phi=5, y_width_dt=2, y_overlap=0.95,
                                     sigma=2.0, sampling_rate=200.0):
    """
    Create smoothed 2D histogram time-series plots inspired by Baillard's approach.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of splitting results with event_id as keys
    qc_metrics_df : pd.DataFrame
        DataFrame with event metadata including origin_time
    station : str
        Station name for title
    figsize : tuple
        Figure size (width, height)
    x_width_days : float
        Width of moving time window in days
    x_overlap : float
        Overlap fraction for time windows (0.95 = 95% overlap)
    y_width_phi : float
        Bin width for phi in radians
    y_width_dt : int
        Bin width for dt in samples (1 sample = 1/sampling_rate seconds)
    y_overlap : float
        Overlap for smoothing in y-direction
    sigma : float
        Gaussian smoothing parameter
    sampling_rate : float
        Sampling rate in Hz (default 200 Hz)
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates
    from scipy.ndimage import gaussian_filter
    
    # Extract data from results
    data_list = []
    for event_id, result in results_dict.items():
        if event_id in qc_metrics_df['event_id'].values:
            origin_time = qc_metrics_df.loc[qc_metrics_df['event_id'] == event_id, 'origin_time'].values[0]
            phi = result['result']['phi']
            dt = result['result']['dt']
            
            # Convert phi from degrees to radians and normalize to -pi/2 to +pi/2
            phi_rad = np.deg2rad(phi)
            phi_rad_norm = ((phi_rad + np.pi/2) % np.pi) - np.pi/2
            
            # Convert dt from seconds to samples
            dt_samples = dt * sampling_rate
            
            data_list.append({
                'time': pd.to_datetime(str(origin_time)),
                'phi_rad': phi_rad_norm,
                'phi_deg': phi,
                'dt_samples': dt_samples,
                'dt_seconds': dt
            })
    
    df = pd.DataFrame(data_list).sort_values('time')
    
    if len(df) == 0:
        print("No data to plot")
        return
    
    # Create figure with gridspec layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, hspace=0.3, top=0.9, bottom=0.1, 
                          left=0.08, right=0.95)
    
    ax_phi = plt.subplot(gs[0])
    ax_dt = plt.subplot(gs[1], sharex=ax_phi)
    
    # Convert datetime to matplotlib date numbers
    time_nums = mdates.date2num(df['time'])
    
    # === PHI PLOT (in radians) ===
    # Create bins
    time_range = (time_nums.min(), time_nums.max())
    n_time_bins = int((time_range[1] - time_range[0]) / (x_width_days * (1 - x_overlap)))
    n_time_bins = max(20, min(n_time_bins, 100))  # Reasonable limits
    
    # Phi bins from -pi/2 to +pi/2 radians (-1.57 to +1.57)
    phi_bins = np.arange(-np.pi/2, np.pi/2 + y_width_phi*np.pi/180, y_width_phi*np.pi/180)
    n_phi_bins = len(phi_bins) - 1
    
    # Create 2D histogram for phi
    H_phi, xedges_phi, yedges_phi = np.histogram2d(
        time_nums, df['phi_rad'], 
        bins=[n_time_bins, phi_bins],
        range=[time_range, None]
    )
    
    # Normalize by column (each time bin) - "norm_y=True" in Baillard's code
    H_phi_norm = H_phi.copy()
    for i in range(H_phi.shape[0]):
        col_sum = H_phi[i, :].sum()
        if col_sum > 0:
            H_phi_norm[i, :] = H_phi[i, :] / col_sum
    
    # Apply Gaussian smoothing
    H_phi_smooth = gaussian_filter(H_phi_norm, sigma=sigma)
    
    # Mask zeros
    H_phi_smooth = np.ma.masked_where(H_phi_smooth < 0.001, H_phi_smooth)
    
    # Plot with imshow for smooth appearance
    extent_phi = [xedges_phi[0], xedges_phi[-1], yedges_phi[0], yedges_phi[-1]]
    im_phi = ax_phi.imshow(H_phi_smooth.T, 
                           origin='lower',
                           aspect='auto',
                           extent=extent_phi,
                           cmap='magma',
                           interpolation='bilinear',  # Smooth interpolation
                           alpha=0.9)
    
    # Colorbar
    cbar_phi = plt.colorbar(im_phi, ax=ax_phi, pad=0.01)
    cbar_phi.set_label('Normalized Density', fontsize=10)
    
    # Format phi axis with radians
    ax_phi.set_ylabel('Fast Direction φ (rad)', fontsize=12, fontweight='bold')
    ax_phi.set_ylim(-np.pi/2, np.pi/2)
    ax_phi.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_phi.grid(True, alpha=0.3, linestyle='--', color='white')

      # Add axvline dashed white line at time = 2015-04-24 05:00:00
    event_time_line = pd.to_datetime('2015-04-24 05:00:00')
    ax_phi.axvline(mdates.date2num(event_time_line), color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Set y-ticks in radians
    phi_ticks_rad = np.array([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2])
    ax_phi.set_yticks(phi_ticks_rad)
    
    # Add moving average
    window_size = max(5, len(df) // 10)
    if len(df) >= window_size:
        df['phi_rad_ma'] = df['phi_rad'].rolling(window=window_size, center=True).mean()
        ax_phi.plot(df['time'], df['phi_rad_ma'], 'w-', linewidth=2.5, alpha=0.9)
        ax_phi.plot(df['time'], df['phi_rad_ma'], 'black', linewidth=2,
                   label=f'{window_size}-event moving avg', alpha=0.8)
        ax_phi.legend(loc='upper right', fontsize=9, facecolor='white', 
                     edgecolor='white', framealpha=0.7)
    
    # === DT PLOT (in samples) ===
    dt_max_samples = df['dt_samples'].quantile(0.98)
    # Create bins in samples
    dt_bins = np.arange(0, min(40, dt_max_samples) + y_width_dt, y_width_dt)  # 40 samples = 0.2 sec at 200 Hz
    n_dt_bins = len(dt_bins) - 1
    
    # Create 2D histogram for dt
    H_dt, xedges_dt, yedges_dt = np.histogram2d(
        time_nums, df['dt_samples'],
        bins=[n_time_bins, dt_bins],
        range=[time_range, None]
    )
    
    # Normalize by column
    H_dt_norm = H_dt.copy()
    for i in range(H_dt.shape[0]):
        col_sum = H_dt[i, :].sum()
        if col_sum > 0:
            H_dt_norm[i, :] = H_dt[i, :] / col_sum
    
    # Apply Gaussian smoothing
    H_dt_smooth = gaussian_filter(H_dt_norm, sigma=sigma)
    
    # Mask zeros
    H_dt_smooth = np.ma.masked_where(H_dt_smooth < 0.001, H_dt_smooth)
    
    # Plot with imshow
    extent_dt = [xedges_dt[0], xedges_dt[-1], yedges_dt[0], yedges_dt[-1]]
    im_dt = ax_dt.imshow(H_dt_smooth.T,
                         origin='lower',
                         aspect='auto',
                         extent=extent_dt,
                         cmap='magma',
                         interpolation='bilinear',
                         alpha=0.9)
    
    # Colorbar
    cbar_dt = plt.colorbar(im_dt, ax=ax_dt, pad=0.01)
    cbar_dt.set_label('Normalized Density', fontsize=10)

    # Add axvline dashed white line at time = 2015-04-24 05:00:00
    event_time_line = pd.to_datetime('2015-04-24 05:00:00')
    ax_dt.axvline(mdates.date2num(event_time_line), color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Format dt axis with samples
    ax_dt.set_ylabel('Delay Time δt (samples)', fontsize=12, fontweight='bold')
    ax_dt.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax_dt.set_ylim(0, 25)  # Show up to 25 samples (0.125 sec at 200 Hz)
    ax_dt.grid(True, alpha=0.3, linestyle='--', color='white')
    
    # Add secondary y-axis for seconds
    #ax_dt_sec = ax_dt.secondary_yaxis('right', functions=(
    #    lambda x: x / sampling_rate,  # samples to seconds
    #    lambda x: x * sampling_rate   # seconds to samples
    #))
    #ax_dt_sec.set_ylabel('δt (s)', fontsize=10)
    
    # Add moving average for dt
    if len(df) >= window_size:
        df['dt_samples_ma'] = df['dt_samples'].rolling(window=window_size, center=True).mean()
        ax_dt.plot(df['time'], df['dt_samples_ma'], 'w-', linewidth=2.5, alpha=0.9)
        ax_dt.plot(df['time'], df['dt_samples_ma'], 'black', linewidth=2,
                  label=f'{window_size}-event moving avg', alpha=0.8)
        ax_dt.legend(loc='upper right', fontsize=9, facecolor='white',
                    edgecolor='white', framealpha=0.7)
    
    # Format x-axis with dates
    date_formatter = DateFormatter('%Y-%m-%d')
    ax_dt.xaxis.set_major_formatter(date_formatter)
    
    # Auto-adjust date locator
    days_span = (df['time'].max() - df['time'].min()).days
    if days_span > 60:
        ax_dt.xaxis.set_major_locator(mdates.MonthLocator())
    elif days_span > 14:
        ax_dt.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    else:
        ax_dt.xaxis.set_major_locator(mdates.DayLocator())
    
    plt.setp(ax_dt.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax_phi.xaxis.get_majorticklabels(), visible=False)
    
    # Add title with statistics (in both degrees and radians)
    phi_mean_deg = df['phi_deg'].mean()
    phi_std_deg = df['phi_deg'].std()
    dt_mean_sec = df['dt_seconds'].mean()
    dt_std_sec = df['dt_seconds'].std()
    
    title = (f"Splitting Parameter - Station {station}\n"
             f"N = {len(df)} events | "
             f"φ: {phi_mean_deg:.1f}° ± {phi_std_deg:.1f}° "
             f"({np.deg2rad(phi_mean_deg):.2f} ± {np.deg2rad(phi_std_deg):.2f} rad) | "
             f"δt: {dt_mean_sec:.3f} ± {dt_std_sec:.3f} s "
             f"({dt_mean_sec*sampling_rate:.1f} ± {dt_std_sec*sampling_rate:.1f} samples)")
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig, (ax_phi, ax_dt), df

# Make a new list of phi results from SWSPy, converted to Baillard convention for direct comparison
# phi_baillard = ((90 + phi + 90) % 180) - 90
def convert_phi_baillard(phi_deg):
    return -1 * (((90 + phi_deg + 90) % 180) - 90)


def plot_splitting_timeseries_baillard_style(
        results_dict,
        qc_metrics_df,
        station='AXAS2',
        figsize=(14, 8),
        x_width_days=5,
        x_overlap=0.97,
        y_width_phi=5,
        y_width_dt=1,
        sampling_rate=200.0,
        sigma_time=2.0,
        sigma_y=0.8):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter
    from scipy.ndimage import gaussian_filter

    # --------------------------------------------------
    # Extract data
    # --------------------------------------------------
    data_list = []
    for event_id, result in results_dict.items():
        if event_id in qc_metrics_df['event_id'].values:
            origin_time = qc_metrics_df.loc[
                qc_metrics_df['event_id'] == event_id,
                'origin_time'
            ].values[0]

            phi = result['result']['phi']
            dt = result['result']['dt']

            phi_rad = np.deg2rad(phi)
            phi_rad = ((phi_rad + np.pi/2) % np.pi) - np.pi/2
            dt_samples = dt * sampling_rate

            data_list.append({
                'time': pd.to_datetime(str(origin_time)),
                'phi_rad': phi_rad,
                'dt_samples': dt_samples
            })

    df = pd.DataFrame(data_list).sort_values('time')
    if len(df) == 0:
        print("No data to plot.")
        return

    time_nums = mdates.date2num(df['time'])

    # --------------------------------------------------
    # Sliding window density function
    # --------------------------------------------------
    def sliding_density(values, y_bins):

        t_min = time_nums.min()
        t_max = time_nums.max()

        step = x_width_days * (1 - x_overlap)
        centers = np.arange(
            t_min + x_width_days/2,
            t_max - x_width_days/2,
            step
        )

        H = np.zeros((len(centers), len(y_bins) - 1))

        for i, c in enumerate(centers):
            mask = (
                (time_nums >= c - x_width_days/2) &
                (time_nums <= c + x_width_days/2)
            )

            if np.sum(mask) > 0:
                hist, _ = np.histogram(values[mask], bins=y_bins)
                H[i, :] = hist

        # Smooth (anisotropic)
        H = gaussian_filter(H, sigma=(sigma_time, sigma_y))

        # Column normalization AFTER smoothing
        for i in range(H.shape[0]):
            s = H[i, :].sum()
            if s > 0:
                H[i, :] /= s

        H = np.ma.masked_where(H <= 0, H)

        return H, centers

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------
    fig, (ax_phi, ax_dt) = plt.subplots(
        2, 1, figsize=figsize, sharex=True
    )

    # --------------------------------------------------
    # PHI
    # --------------------------------------------------
    phi_bins = np.arange(
        -np.pi/2,
        np.pi/2 + np.deg2rad(y_width_phi),
        np.deg2rad(y_width_phi)
    )

    H_phi, t_centers = sliding_density(
        df['phi_rad'].values,
        phi_bins
    )

    extent_phi = [
        t_centers[0], t_centers[-1],
        phi_bins[0], phi_bins[-1]
    ]

    im_phi = ax_phi.imshow(
        H_phi.T,
        origin='lower',
        aspect='auto',
        extent=extent_phi,
        cmap='magma',
        interpolation='nearest'
    )

    ax_phi.set_ylabel('Fast dir (rad)', fontsize=12)
    ax_phi.set_ylim(-np.pi/2, np.pi/2)
    ax_phi.grid(False)

    # eruption line
    eruption_time = pd.to_datetime('2015-04-24 05:00:00')
    ax_phi.axvline(
        mdates.date2num(eruption_time),
        color='white',
        linestyle=':',
        linewidth=2
    )

    # --------------------------------------------------
    # DT
    # --------------------------------------------------
    dt_bins = np.arange(0, 30, y_width_dt)

    H_dt, t_centers = sliding_density(
        df['dt_samples'].values,
        dt_bins
    )

    extent_dt = [
        t_centers[0], t_centers[-1],
        dt_bins[0], dt_bins[-1]
    ]

    im_dt = ax_dt.imshow(
        H_dt.T,
        origin='lower',
        aspect='auto',
        extent=extent_dt,
        cmap='magma',
        interpolation='nearest'
    )

    ax_dt.set_ylabel('Lag (samples)', fontsize=12)
    ax_dt.set_xlabel('Date', fontsize=12)
    ax_dt.set_ylim(0, 30)
    ax_dt.grid(False)

    ax_dt.axvline(
        mdates.date2num(eruption_time),
        color='white',
        linestyle=':',
        linewidth=2
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------
    ax_dt.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax_dt.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_dt.xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle(
        f'{station}',
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()
    return fig, (ax_phi, ax_dt), df

def plot_splitting_timeseries_smooth_before_after(results_dict, qc_metrics_df, station='AXAS2', 
                                                  figsize=(14, 8), x_width_days=5, x_overlap=0.95,
                                                  y_width_phi=5, y_width_dt=2, y_overlap=0.95,
                                                  sigma=2.0, sampling_rate=200.0,
                                                  eruption_time='2015-04-24 05:00:00'):
    """
    Create smoothed 2D histogram time-series plots with separate rolling averages before/after eruption.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of splitting results with event_id as keys
    qc_metrics_df : pd.DataFrame
        DataFrame with event metadata including origin_time
    station : str
        Station name for title
    figsize : tuple
        Figure size (width, height)
    x_width_days : float
        Width of moving time window in days
    x_overlap : float
        Overlap fraction for time windows (0.95 = 95% overlap)
    y_width_phi : float
        Bin width for phi in radians
    y_width_dt : int
        Bin width for dt in samples (1 sample = 1/sampling_rate seconds)
    y_overlap : float
        Overlap for smoothing in y-direction
    sigma : float
        Gaussian smoothing parameter
    sampling_rate : float
        Sampling rate in Hz (default 200 Hz)
    eruption_time : str
        Eruption time for splitting the rolling averages (default: '2015-04-24 05:00:00')
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates
    from scipy.ndimage import gaussian_filter
    
    # Convert eruption time to pandas Timestamp
    eruption_timestamp = pd.to_datetime(eruption_time)
    
    # Extract data from results
    data_list = []
    for event_id, result in results_dict.items():
        if event_id in qc_metrics_df['event_id'].values:
            origin_time = qc_metrics_df.loc[qc_metrics_df['event_id'] == event_id, 'origin_time'].values[0]
            phi = result['result']['phi']
            dt = result['result']['dt']
            
            # Convert phi from degrees to radians and normalize to -pi/2 to +pi/2
            phi_rad = np.deg2rad(phi)
            phi_rad_norm = ((phi_rad + np.pi/2) % np.pi) - np.pi/2
            
            # Convert dt from seconds to samples
            dt_samples = dt * sampling_rate
            
            data_list.append({
                'time': pd.to_datetime(str(origin_time)),
                'phi_rad': phi_rad_norm,
                'phi_deg': phi,
                'dt_samples': dt_samples,
                'dt_seconds': dt
            })
    
    df = pd.DataFrame(data_list).sort_values('time')
    
    if len(df) == 0:
        print("No data to plot")
        return
    
    # Split data into before and after eruption
    df_before = df[df['time'] < eruption_timestamp].copy()
    df_after = df[df['time'] >= eruption_timestamp].copy()
    
    # Create figure with gridspec layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, hspace=0.3, top=0.9, bottom=0.1, 
                          left=0.08, right=0.95)
    
    ax_phi = plt.subplot(gs[0])
    ax_dt = plt.subplot(gs[1], sharex=ax_phi)
    
    # Convert datetime to matplotlib date numbers
    time_nums = mdates.date2num(df['time'])
    
    # === PHI PLOT (in radians) ===
    # Create bins
    time_range = (time_nums.min(), time_nums.max())
    n_time_bins = int((time_range[1] - time_range[0]) / (x_width_days * (1 - x_overlap)))
    n_time_bins = max(20, min(n_time_bins, 100))  # Reasonable limits
    
    # Phi bins from -pi/2 to +pi/2 radians (-1.57 to +1.57)
    phi_bins = np.arange(-np.pi/2, np.pi/2 + y_width_phi*np.pi/180, y_width_phi*np.pi/180)
    n_phi_bins = len(phi_bins) - 1
    
    # Create 2D histogram for phi
    H_phi, xedges_phi, yedges_phi = np.histogram2d(
        time_nums, df['phi_rad'], 
        bins=[n_time_bins, phi_bins],
        range=[time_range, None]
    )
    
    # Normalize by column (each time bin) - "norm_y=True" in Baillard's code
    H_phi_norm = H_phi.copy()
    for i in range(H_phi.shape[0]):
        col_sum = H_phi[i, :].sum()
        if col_sum > 0:
            H_phi_norm[i, :] = H_phi[i, :] / col_sum
    
    # Apply Gaussian smoothing
    H_phi_smooth = gaussian_filter(H_phi_norm, sigma=sigma)
    
    # Mask zeros
    H_phi_smooth = np.ma.masked_where(H_phi_smooth < 0.001, H_phi_smooth)
    
    # Plot with imshow for smooth appearance
    extent_phi = [xedges_phi[0], xedges_phi[-1], yedges_phi[0], yedges_phi[-1]]
    im_phi = ax_phi.imshow(H_phi_smooth.T, 
                           origin='lower',
                           aspect='auto',
                           extent=extent_phi,
                           cmap='magma',
                           interpolation='bilinear',  # Smooth interpolation
                           alpha=0.9)
    
    # Colorbar
    cbar_phi = plt.colorbar(im_phi, ax=ax_phi, pad=0.01)
    cbar_phi.set_label('Normalized Density', fontsize=10)
    
    # Format phi axis with radians
    ax_phi.set_ylabel('Fast Direction φ (rad)', fontsize=12, fontweight='bold')
    ax_phi.set_ylim(-np.pi/2, np.pi/2)
    ax_phi.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_phi.grid(True, alpha=0.3, linestyle='--', color='white')

    # Add axvline dashed white line at eruption time
    ax_phi.axvline(mdates.date2num(eruption_timestamp), color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Set y-ticks in radians
    phi_ticks_rad = np.array([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2])
    ax_phi.set_yticks(phi_ticks_rad)
    
    # Add separate moving averages for before and after eruption
    window_size = max(5, len(df) // 10)
    
    # Before eruption rolling average (phi)
    if len(df_before) >= 3:
        window_before = max(3, len(df_before) // 5)
        df_before['phi_rad_ma'] = df_before['phi_rad'].rolling(window=window_before, center=True, min_periods=1).mean()
        ax_phi.plot(df_before['time'], df_before['phi_rad_ma'], 'w-', linewidth=2.5, alpha=0.9)
        ax_phi.plot(df_before['time'], df_before['phi_rad_ma'], 'blue', linewidth=2, 
                   label=f'Before: {window_before}-event moving avg', alpha=0.8)
    
    # After eruption rolling average (phi)
    if len(df_after) >= 3:
        window_after = max(3, len(df_after) // 5)
        df_after['phi_rad_ma'] = df_after['phi_rad'].rolling(window=window_after, center=True, min_periods=1).mean()
        ax_phi.plot(df_after['time'], df_after['phi_rad_ma'], 'w-', linewidth=2.5, alpha=0.9)
        ax_phi.plot(df_after['time'], df_after['phi_rad_ma'], 'red', linewidth=2, 
                   label=f'After: {window_after}-event moving avg', alpha=0.8)
    
    ax_phi.legend(loc='upper right', fontsize=9, facecolor='white', 
                 edgecolor='white', framealpha=0.7)
    
    # === DT PLOT (in samples) ===
    dt_max_samples = df['dt_samples'].quantile(0.98)
    # Create bins in samples
    dt_bins = np.arange(0, min(40, dt_max_samples) + y_width_dt, y_width_dt)  # 40 samples = 0.2 sec at 200 Hz
    n_dt_bins = len(dt_bins) - 1
    
    # Create 2D histogram for dt
    H_dt, xedges_dt, yedges_dt = np.histogram2d(
        time_nums, df['dt_samples'],
        bins=[n_time_bins, dt_bins],
        range=[time_range, None]
    )
    
    # Normalize by column
    H_dt_norm = H_dt.copy()
    for i in range(H_dt.shape[0]):
        col_sum = H_dt[i, :].sum()
        if col_sum > 0:
            H_dt_norm[i, :] = H_dt[i, :] / col_sum
    
    # Apply Gaussian smoothing
    H_dt_smooth = gaussian_filter(H_dt_norm, sigma=sigma)
    
    # Mask zeros
    H_dt_smooth = np.ma.masked_where(H_dt_smooth < 0.001, H_dt_smooth)
    
    # Plot with imshow
    extent_dt = [xedges_dt[0], xedges_dt[-1], yedges_dt[0], yedges_dt[-1]]
    im_dt = ax_dt.imshow(H_dt_smooth.T,
                         origin='lower',
                         aspect='auto',
                         extent=extent_dt,
                         cmap='magma',
                         interpolation='bilinear',
                         alpha=0.9)
    
    # Colorbar
    cbar_dt = plt.colorbar(im_dt, ax=ax_dt, pad=0.01)
    cbar_dt.set_label('Normalized Density', fontsize=10)

    # Add axvline dashed white line at eruption time
    ax_dt.axvline(mdates.date2num(eruption_timestamp), color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Format dt axis with samples
    ax_dt.set_ylabel('Delay Time δt (samples)', fontsize=12, fontweight='bold')
    ax_dt.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax_dt.set_ylim(0, 25)  # Show up to 25 samples (0.125 sec at 200 Hz)
    ax_dt.grid(True, alpha=0.3, linestyle='--', color='white')
    
    # Add separate moving averages for before and after eruption (dt)
    # Before eruption rolling average (dt)
    if len(df_before) >= 3:
        window_before = max(3, len(df_before) // 5)
        df_before['dt_samples_ma'] = df_before['dt_samples'].rolling(window=window_before, center=True, min_periods=1).mean()
        ax_dt.plot(df_before['time'], df_before['dt_samples_ma'], 'w-', linewidth=2.5, alpha=0.9)
        ax_dt.plot(df_before['time'], df_before['dt_samples_ma'], 'blue', linewidth=2,
                  label=f'Before: {window_before}-event moving avg', alpha=0.8)
    
    # After eruption rolling average (dt)
    if len(df_after) >= 3:
        window_after = max(3, len(df_after) // 5)
        df_after['dt_samples_ma'] = df_after['dt_samples'].rolling(window=window_after, center=True, min_periods=1).mean()
        ax_dt.plot(df_after['time'], df_after['dt_samples_ma'], 'w-', linewidth=2.5, alpha=0.9)
        ax_dt.plot(df_after['time'], df_after['dt_samples_ma'], 'red', linewidth=2,
                  label=f'After: {window_after}-event moving avg', alpha=0.8)
    
    ax_dt.legend(loc='upper right', fontsize=9, facecolor='white',
                edgecolor='white', framealpha=0.7)
    
    # Format x-axis with dates
    date_formatter = DateFormatter('%Y-%m-%d')
    ax_dt.xaxis.set_major_formatter(date_formatter)
    
    # Auto-adjust date locator
    days_span = (df['time'].max() - df['time'].min()).days
    if days_span > 60:
        ax_dt.xaxis.set_major_locator(mdates.MonthLocator())
    elif days_span > 14:
        ax_dt.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    else:
        ax_dt.xaxis.set_major_locator(mdates.DayLocator())
    
    plt.setp(ax_dt.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax_phi.xaxis.get_majorticklabels(), visible=False)
    
    # Add title with statistics (in both degrees and radians)
    phi_mean_deg = df['phi_deg'].mean()
    phi_std_deg = df['phi_deg'].std()
    dt_mean_sec = df['dt_seconds'].mean()
    dt_std_sec = df['dt_seconds'].std()
    
    # Add before/after statistics
    stats_lines = [f"N = {len(df)} events total"]
    if len(df_before) > 0:
        stats_lines.append(f"Before: N={len(df_before)}, φ={df_before['phi_deg'].mean():.1f}°±{df_before['phi_deg'].std():.1f}°, δt={df_before['dt_seconds'].mean():.3f}±{df_before['dt_seconds'].std():.3f}s")
    if len(df_after) > 0:
        stats_lines.append(f"After: N={len(df_after)}, φ={df_after['phi_deg'].mean():.1f}°±{df_after['phi_deg'].std():.1f}°, δt={df_after['dt_seconds'].mean():.3f}±{df_after['dt_seconds'].std():.3f}s")
    
    title = (f"Splitting Parameter - Station {station}\n" + 
             " | ".join(stats_lines))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig, (ax_phi, ax_dt), df

def plot_splitting_timeseries_around_eruption(results_dict, qc_metrics_df, 
                                               eruption_time, station='AXAS2',
                                               hours_before=24, hours_after=24,
                                               figsize=(14, 8), x_width_hours=2, x_overlap=0.95,
                                               y_width_phi=5, y_width_dt=2,
                                               sigma=2.0, sampling_rate=200.0):
    """
    Create smoothed 2D histogram time-series plots focused on eruption timing.
    Time axis shows hours relative to eruption (negative = before, positive = after).
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of splitting results with event_id as keys
    qc_metrics_df : pd.DataFrame
        DataFrame with event metadata including origin_time
    eruption_time : str or pd.Timestamp
        Eruption onset time (e.g., "2015-04-24 05:00:00")
    station : str
        Station name for title
    hours_before : float
        Hours before eruption to include (default: 24)
    hours_after : float
        Hours after eruption to include (default: 24)
    figsize : tuple
        Figure size (width, height)
    x_width_hours : float
        Width of moving time window in hours
    x_overlap : float
        Overlap fraction for time windows
    y_width_phi : float
        Bin width for phi in degrees
    y_width_dt : int
        Bin width for dt in samples
    sigma : float
        Gaussian smoothing parameter
    sampling_rate : float
        Sampling rate in Hz (default 200 Hz)
    """
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import gaussian_filter
    
    # Convert eruption time to pandas Timestamp
    eruption_time = pd.to_datetime(eruption_time)
    
    # Define time window
    start_time = eruption_time - pd.Timedelta(hours=hours_before)
    end_time = eruption_time + pd.Timedelta(hours=hours_after)
    
    # Extract data from results within time window
    data_list = []
    for event_id, result in results_dict.items():
        if event_id in qc_metrics_df['event_id'].values:
            origin_time = pd.to_datetime(str(qc_metrics_df.loc[
                qc_metrics_df['event_id'] == event_id, 'origin_time'].values[0]))
            
            # Filter by time window
            if start_time <= origin_time <= end_time:
                phi = result['result']['phi']
                dt = result['result']['dt']
                
                # Calculate hours relative to eruption (negative = before)
                hours_from_eruption = (origin_time - eruption_time).total_seconds() / 3600.0
                
                # Convert phi from degrees to radians and normalize to -pi/2 to +pi/2
                phi_rad = np.deg2rad(phi)
                phi_rad_norm = ((phi_rad + np.pi/2) % np.pi) - np.pi/2
                
                # Convert dt from seconds to samples
                dt_samples = dt * sampling_rate
                
                data_list.append({
                    'hours_from_eruption': hours_from_eruption,
                    'phi_rad': phi_rad_norm,
                    'phi_deg': phi,
                    'dt_samples': dt_samples,
                    'dt_seconds': dt
                })
    
    df = pd.DataFrame(data_list).sort_values('hours_from_eruption')
    
    if len(df) == 0:
        print(f"No data found in time window [{start_time} to {end_time}]")
        return None, None, None
    
    print(f"Found {len(df)} events in window ({hours_before}h before to {hours_after}h after eruption)")
    
    # Create figure with gridspec layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, hspace=0.3, top=0.9, bottom=0.1, 
                          left=0.08, right=0.95)
    
    ax_phi = plt.subplot(gs[0])
    ax_dt = plt.subplot(gs[1], sharex=ax_phi)
    
    # Time range in hours
    time_vals = df['hours_from_eruption'].values
    time_range = (time_vals.min(), time_vals.max())
    
    # === PHI PLOT (in radians) ===
    # Create bins
    n_time_bins = int((time_range[1] - time_range[0]) / (x_width_hours * (1 - x_overlap)))
    n_time_bins = max(20, min(n_time_bins, 100))
    
    # Phi bins from -pi/2 to +pi/2 radians
    phi_bins = np.arange(-np.pi/2, np.pi/2 + y_width_phi*np.pi/180, y_width_phi*np.pi/180)
    
    # Create 2D histogram for phi
    H_phi, xedges_phi, yedges_phi = np.histogram2d(
        time_vals, df['phi_rad'], 
        bins=[n_time_bins, phi_bins],
        range=[time_range, None]
    )
    
    # Normalize by column (each time bin)
    H_phi_norm = H_phi.copy()
    for i in range(H_phi.shape[0]):
        col_sum = H_phi[i, :].sum()
        if col_sum > 0:
            H_phi_norm[i, :] = H_phi[i, :] / col_sum
    
    # Apply Gaussian smoothing
    H_phi_smooth = gaussian_filter(H_phi_norm, sigma=sigma)
    H_phi_smooth = np.ma.masked_where(H_phi_smooth < 0.001, H_phi_smooth)
    
    # Plot
    extent_phi = [xedges_phi[0], xedges_phi[-1], yedges_phi[0], yedges_phi[-1]]
    im_phi = ax_phi.imshow(H_phi_smooth.T, 
                           origin='lower',
                           aspect='auto',
                           extent=extent_phi,
                           cmap='magma',
                           interpolation='bilinear',
                           alpha=0.9)
    
    # Colorbar
    cbar_phi = plt.colorbar(im_phi, ax=ax_phi, pad=0.01)
    cbar_phi.set_label('Normalized Density', fontsize=10)
    
    # Format phi axis
    ax_phi.set_ylabel('Fast Direction φ (rad)', fontsize=12, fontweight='bold')
    ax_phi.set_ylim(-np.pi/2, np.pi/2)
    ax_phi.axhline(0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_phi.grid(True, alpha=0.3, linestyle='--', color='white')
    
    # Set y-ticks in radians
    phi_ticks_rad = np.array([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2])
    ax_phi.set_yticks(phi_ticks_rad)
    
    # Add eruption line (dashed white)
    ax_phi.axvline(0, color='white', linestyle='--', alpha=0.9, linewidth=2.5, 
                   label='Eruption Onset', zorder=10)
    ax_phi.legend(loc='upper right', fontsize=9, facecolor='white', 
                 edgecolor='white', framealpha=0.7)
    
    # === DT PLOT (in samples) ===
    dt_max_samples = df['dt_samples'].quantile(0.98)
    dt_bins = np.arange(0, min(40, dt_max_samples) + y_width_dt, y_width_dt)
    
    # Create 2D histogram for dt
    H_dt, xedges_dt, yedges_dt = np.histogram2d(
        time_vals, df['dt_samples'],
        bins=[n_time_bins, dt_bins],
        range=[time_range, None]
    )
    
    # Normalize by column
    H_dt_norm = H_dt.copy()
    for i in range(H_dt.shape[0]):
        col_sum = H_dt[i, :].sum()
        if col_sum > 0:
            H_dt_norm[i, :] = H_dt[i, :] / col_sum
    
    # Apply Gaussian smoothing
    H_dt_smooth = gaussian_filter(H_dt_norm, sigma=sigma)
    H_dt_smooth = np.ma.masked_where(H_dt_smooth < 0.001, H_dt_smooth)
    
    # Plot
    extent_dt = [xedges_dt[0], xedges_dt[-1], yedges_dt[0], yedges_dt[-1]]
    im_dt = ax_dt.imshow(H_dt_smooth.T,
                         origin='lower',
                         aspect='auto',
                         extent=extent_dt,
                         cmap='magma',
                         interpolation='bilinear',
                         alpha=0.9)
    
    # Colorbar
    cbar_dt = plt.colorbar(im_dt, ax=ax_dt, pad=0.01)
    cbar_dt.set_label('Normalized Density', fontsize=10)
    
    # Format dt axis
    ax_dt.set_ylabel('Delay Time δt (samples)', fontsize=12, fontweight='bold')
    ax_dt.set_xlabel('Hours Relative to Eruption', fontsize=12, fontweight='bold')
    ax_dt.set_ylim(0, 25)
    ax_dt.grid(True, alpha=0.3, linestyle='--', color='white')
    
    # Add secondary y-axis for seconds
    ax_dt_sec = ax_dt.secondary_yaxis('right', functions=(
        lambda x: x / sampling_rate,
        lambda x: x * sampling_rate
    ))
    ax_dt_sec.set_ylabel('δt (s)', fontsize=10)
    
    # Add eruption line (dashed white)
    ax_dt.axvline(0, color='white', linestyle='--', alpha=0.9, linewidth=2.5,
                  label='Eruption Onset', zorder=10)
    ax_dt.legend(loc='upper right', fontsize=9, facecolor='white',
                edgecolor='white', framealpha=0.7)
    
    # Format x-axis
    ax_dt.set_xlim(-hours_before, hours_after)
    ax_phi.set_xlim(-hours_before, hours_after)
    
    plt.setp(ax_phi.xaxis.get_majorticklabels(), visible=False)
    
    # Add title with statistics
    phi_mean_deg = df['phi_deg'].mean()
    phi_std_deg = df['phi_deg'].std()
    dt_mean_sec = df['dt_seconds'].mean()
    dt_std_sec = df['dt_seconds'].std()
    
    title = (f"Splitting Parameters Around Eruption - Station {station}\n"
             f"Eruption: {eruption_time.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
             f"N = {len(df)} events | "
             f"φ: {phi_mean_deg:.1f}° ± {phi_std_deg:.1f}° | "
             f"δt: {dt_mean_sec:.3f} ± {dt_std_sec:.3f} s")
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig, (ax_phi, ax_dt), df

def plot_splitting_scatter_around_eruption(results_dict, qc_metrics_df, 
                                           eruption_time, station='AXAS2',
                                           hours_before=24, hours_after=24,
                                           figsize=(14, 10), window_hours=3,
                                           marker_size=30, marker_alpha=0.6,
                                           sampling_rate=200.0):
    """
    Create scatter plots of splitting parameters around eruption with rolling averages.
    Time axis shows hours relative to eruption (negative = before, positive = after).
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of splitting results with event_id as keys
    qc_metrics_df : pd.DataFrame
        DataFrame with event metadata including origin_time
    eruption_time : str or pd.Timestamp
        Eruption onset time (e.g., "2015-04-24 05:00:00")
    station : str
        Station name for title
    hours_before : float
        Hours before eruption to include (default: 24)
    hours_after : float
        Hours after eruption to include (default: 24)
    figsize : tuple
        Figure size (width, height)
    window_hours : float
        Rolling window size in hours for moving average
    marker_size : float
        Size of scatter plot markers
    marker_alpha : float
        Transparency of scatter markers (0-1)
    sampling_rate : float
        Sampling rate in Hz (default 200 Hz)
    """
    import matplotlib.gridspec as gridspec
    
    # Convert eruption time to pandas Timestamp
    eruption_time = pd.to_datetime(eruption_time)
    
    # Define time window
    start_time = eruption_time - pd.Timedelta(hours=hours_before)
    end_time = eruption_time + pd.Timedelta(hours=hours_after)
    
    # Extract data from results within time window
    data_list = []
    for event_id, result in results_dict.items():
        if event_id in qc_metrics_df['event_id'].values:
            origin_time = pd.to_datetime(str(qc_metrics_df.loc[
                qc_metrics_df['event_id'] == event_id, 'origin_time'].values[0]))
            
            # Filter by time window
            if start_time <= origin_time <= end_time:
                phi = result['result']['phi']
                dt = result['result']['dt']
                
                # Calculate hours relative to eruption (negative = before)
                hours_from_eruption = (origin_time - eruption_time).total_seconds() / 3600.0
                
                # Convert dt from seconds to samples
                dt_samples = dt * sampling_rate
                
                data_list.append({
                    'hours_from_eruption': hours_from_eruption,
                    'phi_deg': phi,
                    'dt_samples': dt_samples,
                    'dt_seconds': dt
                })
    
    df = pd.DataFrame(data_list).sort_values('hours_from_eruption')
    
    if len(df) == 0:
        print(f"No data found in time window [{start_time} to {end_time}]")
        return None, None, None
    
    print(f"Found {len(df)} events in window ({hours_before}h before to {hours_after}h after eruption)")
    
    # Create figure with gridspec layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, hspace=0.3, top=0.87, bottom=0.08, 
                          left=0.1, right=0.95)
    
    ax_phi = plt.subplot(gs[0])
    ax_dt = plt.subplot(gs[1], sharex=ax_phi)
    
    # Calculate rolling averages
    # Convert window size to number of events (approximate)
    time_span = df['hours_from_eruption'].max() - df['hours_from_eruption'].min()
    events_per_hour = len(df) / time_span if time_span > 0 else 1
    window_size = max(3, int(window_hours * events_per_hour))
    
    df['phi_rolling'] = df['phi_deg'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['dt_rolling'] = df['dt_samples'].rolling(window=window_size, center=True, min_periods=1).mean()
    
    # === PHI SCATTER PLOT ===
    # Scatter plot
    ax_phi.scatter(df['hours_from_eruption'], df['phi_deg'], 
                   s=marker_size, alpha=marker_alpha, color='steelblue', 
                   edgecolors='black', linewidth=0.5, label='Individual measurements')
    
    # Rolling average
    ax_phi.plot(df['hours_from_eruption'], df['phi_rolling'], 
                color='red', linewidth=2.5, label=f'{window_hours}h rolling average', zorder=10)
    
    # Eruption line
    ax_phi.axvline(0, color='black', linestyle='--', linewidth=2, 
                   label='Eruption Onset', alpha=0.8, zorder=5)
    
    # Formatting
    ax_phi.set_ylabel('Fast Direction φ (°)', fontsize=12, fontweight='bold')
    ax_phi.set_ylim(-90, 90)
    ax_phi.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax_phi.grid(True, alpha=0.3, linestyle='--')
    ax_phi.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add horizontal bands for reference
    ax_phi.axhspan(-90, -60, alpha=0.05, color='blue')
    ax_phi.axhspan(-30, 30, alpha=0.05, color='green')
    ax_phi.axhspan(60, 90, alpha=0.05, color='blue')
    
    # === DT SCATTER PLOT ===
    # Scatter plot
    ax_dt.scatter(df['hours_from_eruption'], df['dt_samples'], 
                  s=marker_size, alpha=marker_alpha, color='coral', 
                  edgecolors='black', linewidth=0.5, label='Individual measurements')
    
    # Rolling average
    ax_dt.plot(df['hours_from_eruption'], df['dt_rolling'], 
               color='darkred', linewidth=2.5, label=f'{window_hours}h rolling average', zorder=10)
    
    # Eruption line
    ax_dt.axvline(0, color='black', linestyle='--', linewidth=2, 
                  label='Eruption Onset', alpha=0.8, zorder=5)
    
    # Formatting
    ax_dt.set_ylabel('Delay Time δt (samples)', fontsize=12, fontweight='bold')
    ax_dt.set_xlabel('Hours Relative to Eruption', fontsize=12, fontweight='bold')
    ax_dt.set_ylim(0, max(40, df['dt_samples'].quantile(0.98) * 1.1))
    ax_dt.grid(True, alpha=0.3, linestyle='--')
    ax_dt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add secondary y-axis for seconds
    ax_dt_sec = ax_dt.secondary_yaxis('right', functions=(
        lambda x: x / sampling_rate,
        lambda x: x * sampling_rate
    ))
    ax_dt_sec.set_ylabel('δt (s)', fontsize=11)
    
    # Format x-axis
    ax_dt.set_xlim(-hours_before, hours_after)
    ax_phi.set_xlim(-hours_before, hours_after)
    
    # Remove x-labels from top plot
    plt.setp(ax_phi.xaxis.get_majorticklabels(), visible=False)
    
    # Add statistics text boxes
    phi_mean_deg = df['phi_deg'].mean()
    phi_std_deg = df['phi_deg'].std()
    dt_mean_sec = df['dt_seconds'].mean()
    dt_std_sec = df['dt_seconds'].std()
    
    # Split statistics by before/after eruption
    df_before = df[df['hours_from_eruption'] < 0]
    df_after = df[df['hours_from_eruption'] >= 0]
    
    stats_text = (
        f"All: φ = {phi_mean_deg:.1f}° ± {phi_std_deg:.1f}°, "
        f"δt = {dt_mean_sec:.3f} ± {dt_std_sec:.3f}s\n"
    )
    
    if len(df_before) > 0:
        phi_before = df_before['phi_deg'].mean()
        dt_before = df_before['dt_seconds'].mean()
        stats_text += f"Before: φ = {phi_before:.1f}°, δt = {dt_before:.3f}s (n={len(df_before)})\n"
    
    if len(df_after) > 0:
        phi_after = df_after['phi_deg'].mean()
        dt_after = df_after['dt_seconds'].mean()
        stats_text += f"After: φ = {phi_after:.1f}°, δt = {dt_after:.3f}s (n={len(df_after)})"
    
    # Add title
    title = (f"Splitting Parameters Around Eruption - Station {station}\n"
             f"Eruption: {eruption_time.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
             f"Total N = {len(df)} events\n"
             f"{stats_text}")
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()

    plt.show()

    return fig, (ax_phi, ax_dt), df

def save_results_csv(results, file_name, file_loc='/Users/mhemmett/Seismology/axial-splitting-ml/results/', mode='swspy'):
    # Create dataframe from results_swspy with event metadata and processing parameters

    # Initialize lists for each column
    data_rows = []

    for event_id, event_result in results.items():
        result = event_result['result']
        
        # Extract splitting measurements
        phi = result.get('phi')
        dt = result.get('dt')
        phi_error = result.get('phi_error')
        dt_error = result.get('dt_error')

        quality = result.get('quality')
        
        event_datetime = result.get('event_datetime')
        event_origin_time = result.get('event_origin_time')
        s_arrival_time = result.get('s_arrival_time')
        event_lat = result.get('event_lat')
        event_lon = result.get('event_lon')
        event_depth = result.get('event_depth')
        back_azimuth = result.get('back_azimuth')
        snr_horizontal = result.get('snr_horizontal')
        rectilinearity = result.get('rectilinearity_jurkevics')
        incidence = result.get('incidence_eigenvalue_jurkevics')

        if mode=='swspy':
            first_window_start = result.get('first_window_start')
            last_window_start = result.get('last_window_start')
            first_window_end = result.get('first_window_end')
            last_window_end = result.get('last_window_end')
            n_win = result.get('n_win')
            s_pick_uncertainty = result.get('s_pick_uncertainty')
            dominant_period = result.get('dominant_period')

            # Calculate numerical values for window start and end times
            first_window_start_pre_S_pick =  first_window_start * s_pick_uncertainty
            last_window_start_pre_S_pick = last_window_start * s_pick_uncertainty
            first_window_end_post_S_pick =  first_window_end * dominant_period
            last_window_end_post_S_pick = last_window_end * dominant_period

        # Create row dictionary
        row = {
            'event_id': event_id,
            'event_datetime': event_datetime,
            's_arrival_time' : s_arrival_time,
            'event_origin_time' : event_origin_time,
            'event_lat': event_lat,
            'event_lon': event_lon,
            'event_depth': event_depth,
            'back_azimuth': back_azimuth,
            'incidence': incidence,
            'snr_horizontal': snr_horizontal,
            'rectilinearity': rectilinearity,
            'phi': phi,
            'phi_error': phi_error,
            'dt': dt,
            'dt_error': dt_error,
            'quality': quality,
        }
        
        if mode == 'swspy':
            row.update({
                # Processing parameters (constant values as specified)
                'dominant_period': dominant_period,
                'first_window_start_pre_S_pick': first_window_start_pre_S_pick,
                'last_window_start_pre_S_pick': last_window_start_pre_S_pick,
                'first_window_end_post_S_pick': first_window_end_post_S_pick,
                'last_window_end_post_S_pick': last_window_end_post_S_pick,
                'n_windows': n_win,
                'start_window_step_size': (first_window_start_pre_S_pick - last_window_start_pre_S_pick) / n_win,
                'end_window_step_size': (last_window_end_post_S_pick - first_window_end_post_S_pick) / n_win,
            })

        data_rows.append(row)

    # Create DataFrame
    results_df = pd.DataFrame(data_rows)

    # Display the dataframe
    print(f"\nSplitting Results DataFrame - {len(results_df)} events")
    print("=" * 100)
    print(results_df.to_string())

    # Save to CSV
    output_file = str(file_loc + file_name + '.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")

# Extract all data from organized_waveforms as a dataframe for saving as a csv
import os

def save_passing_waveforms(passing_waveforms, output_dir='passing_waveforms_data'):
    """
    Save passing_waveforms to CSV (metadata) and mseed files (waveforms)
    
    Parameters:
    -----------
    passing_waveforms : dict
        Dictionary with event_id keys and event data values
    output_dir : str
        Directory to save the files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    waveforms_dir = os.path.join(output_dir, 'waveforms')
    os.makedirs(waveforms_dir, exist_ok=True)
    
    # Extract metadata
    metadata_rows = []
    
    for event_id, event_data in passing_waveforms.items():
        origin_time_str = str(event_data.get('origin_time'))
        # Save waveforms to mseed file
        waveform_file = os.path.join(waveforms_dir, f'event_{origin_time_str}.mseed')
        event_data['traces'].write(waveform_file, format='MSEED')
        
        # Extract metadata
        row = {
            'event_id': event_id,
            'waveform_file': waveform_file,
            'station': event_data.get('station'),
            'origin_time': str(event_data.get('origin_time')),
            'datetime': str(event_data.get('datetime')),
            's_arrival_time': (event_data.get('s_arrival_time')),
            'p_arrival_time': (event_data.get('p_arrival_time')),
            'latitude': event_data.get('latitude'),
            'longitude': event_data.get('longitude'),
            'depth': event_data.get('depth'),
            'back_azimuth': event_data.get('back_azimuth'),
            'snr_horizontal': event_data.get('snr_horizontal'),
            'rectilinearity_jurkevics': event_data.get('rectilinearity_jurkevics'),
            'incidence_eigenvalue_jurkevics': event_data.get('incidence_eigenvalue_jurkevics'),
            'magnitude': event_data.get('magnitude'),
        }
        metadata_rows.append(row)
    
    # Create DataFrame and save to CSV
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_file = os.path.join(output_dir, 'passing_waveforms_metadata.csv')
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"✓ Saved {len(metadata_rows)} events")
    print(f"  Metadata: {metadata_file}")
    print(f"  Waveforms: {waveforms_dir}/")
    
    return metadata_df


# Reload passing_waveforms from saved CSV and waveform files
def load_passing_waveforms(output_dir='passing_waveforms_data'):
    """
    Load passing_waveforms from CSV (metadata) and mseed files (waveforms)
    
    Parameters:
    -----------
    output_dir : str
        Directory containing the saved files
        
    Returns:
    --------
    dict
        Reconstructed passing_waveforms dictionary
    """
    import obspy
    from obspy.core import UTCDateTime
    
    # Load metadata CSV
    metadata_file = os.path.join(output_dir, 'passing_waveforms_metadata.csv')
    metadata_df = pd.read_csv(metadata_file)
    
    # Reconstruct passing_waveforms dictionary
    passing_waveforms_reloaded = {}
    
    for idx, row in metadata_df.iterrows():
        event_id = int(row['event_id'])
        
        # Load waveforms from mseed file
        waveform_file = row['waveform_file']
        traces = obspy.read(waveform_file)
        
        # Reconstruct event data dictionary
        event_data = {
            'traces': traces,
            'station': row['station'],
            'origin_time': UTCDateTime(row['origin_time']),
            'datetime': UTCDateTime(row['datetime']),
            's_arrival_time': row['s_arrival_time'],
            'p_arrival_time': row['p_arrival_time'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'depth': row['depth'],
            'back_azimuth': row['back_azimuth'],
            'snr_horizontal': row['snr_horizontal'],
            'rectilinearity_jurkevics': row['rectilinearity_jurkevics'],
            'incidence_eigenvalue_jurkevics': row['incidence_eigenvalue_jurkevics'],
            'magnitude': row['magnitude'] if not pd.isna(row['magnitude']) else None,
        }
        
        passing_waveforms_reloaded[event_id] = event_data
    
    print(f"✓ Loaded {len(passing_waveforms_reloaded)} events from saved files")
    print(f"  Source: {output_dir}/")
    
    return passing_waveforms_reloaded

def plot_movehisto2d_orig1(x,y,
                x_label='X',y_label='Y',title='',ax=None,vmax=None,
                **movehisto2d_kwargs):
    """
    Function made to plot an histo2d but using a moving window in both directions, this ensure better
    consistency between neighbor bins. 
    The histogram can also work when data is an obspy.UTCDateTime array, then the x_start and x_end must
    be given in UTCDateTime as well and the x_width should be given in seconds.
    UTCDatetime are converted to timestamps (seconds since 1970)
    
    Inputs
    ------
        x,y: np.array: arrays containing the data to apply histogram on (x can be UTCDateTime)
        x_width,y_width: float: width of the bins (in seconds for UTCDateTime)
        [x,y]_[start,end]: float: start and end for histogram edges
        [x,y]_over: float in [0,1]: overlap for windows [1 = full overlap]
        flag_y_norm: Boolean: True to normalize by the maximum in each column
        flag_resample: Boolean: Enable resampling to uniform grid (required for smoothing)
        flag_filter: Boolean: Enable Gaussian filtering (requires flag_resample=True)
        x_filter_per, y_filter_per: float in [0,100]: width percentage for smoothing
        filter_mode: str or list: 'nearest' or 'wrap' for edge handling
        [x,y]_label: str
        
    Ouputs
    ------
        ax, im, X, Y, Z, x_bins, x_diffs
        
    Comments:
    ---------
        Uses Baillard's movehisto2d_bin for resampling and filtering
        
    UsedIn
    ------
        SWSCat.plot_movehisto2d_time, wrapper functions
    """
    
    ### Check if is made of UTCDateTimes
    
    time_flag=False
    if isinstance(x[0],UTCDateTime):
        time_flag=True
        print('X is in UTCDateTime')
        if movehisto2d_kwargs.get('x_mode','window')=='window':
            print('Remember width should be given in seconds, otherwise memory error')
    
    ### Modify x_start and x_end, and x if x is time and convert to timestamps
    x_start=movehisto2d_kwargs.get('x_start',None)
    x_end=movehisto2d_kwargs.get('x_end',None)
    y_start=movehisto2d_kwargs.get('y_start',None)
    y_end=movehisto2d_kwargs.get('y_end',None)
    
    if time_flag:
        if (x_start is not None) & (not isinstance(x_start,UTCDateTime)):
            raise ValueError('x_start must be given in obspy.UTCDateTime')
        if (x_end is not None) & (not isinstance(x_end,UTCDateTime)):
            raise ValueError('x_end must be given in obspy.UTCDateTime')
        x=np.array([value.timestamp for value in x]) # (seconds since 1970-01-01T00:00:00)
        x_start=x_start.timestamp if x_start is not None else None
        x_end=x_end.timestamp if x_end is not None else None
        movehisto2d_kwargs['x_start']=x_start
        movehisto2d_kwargs['x_end']=x_end

    ### Bin the data (smoothing handled inside movehisto2d_bin via resampling + filtering)
    
    (X,Y,Z,x_bins,x_diffs)=swm.movehisto2d_bin(x,y,**movehisto2d_kwargs)
    
    ########################
    #### Start plotting ####

    #### Grid spec
    
    bottom=0.15 if time_flag else 0.1
    
    ### Checks
    
    if ax is None:
        fig,ax = plt.subplots(gridspec_kw={'bottom':bottom,'left':0.15})
        
    if time_flag:
        X=np.array(swm.timestamp2matplotlib(X.ravel())).reshape(X.shape) # transform for plotting
        x_bins=swm.timestamp2matplotlib(x_bins)
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=30 ,ha='right')
        ax.set_xlim(swm.timestamp2matplotlib([x_start,x_end]))
    else:
        ax.set_xlim([x_start,x_end])
       
    (Xm,Ym)=swm.XY2XY_pcolormesh(X,Y) # To ensure Pcolormesh will be centered on bins

    #im=ax.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'),rasterized=True)
    im=ax.pcolormesh(Xm,Ym,Z,cmap=plt.cm.get_cmap('magma'),rasterized=True,vmax=vmax)
    
    ax.set_ylim([y_start,y_end])
    
    ax.set_aspect('auto')
    if time_flag:
        ax.xaxis_date()
            
    ### Cosmetic
    
    ax.set_ylabel(y_label) 
    if not time_flag:
        ax.set_xlabel(x_label) 
   
    ###### Return
    
    return (ax,im,X,Y,Z,x_bins,x_diffs)

def plot_dt_timeseries_movehisto_orig(results_df, time_column='event_datetime',
                                  x_width=5*24*3600, x_overlap=0.95,
                                  y_width=2, y_overlap=0.9,
                                  sampling_rate=200.0,
                                  figsize=(14, 6),
                                  station='AXAS2',
                                  flag_smooth=True,
                                  x_filter_per=10,
                                  y_filter_per=10):
    """
    Plot delay time (dt) over time using moving window 2D histogram with Baillard-style smoothing.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with splitting results
    time_column : str
        Name of time column (default 'event_datetime')
    x_width : float
        Width of time window in seconds (default 5 days = 5*24*3600)
    x_overlap : float
        Overlap fraction for time windows (0.95 = 95% overlap)
    y_width : float
        Bin width for dt in samples
    y_overlap : float
        Overlap for y-direction
    sampling_rate : float
        Sampling rate in Hz (default 200 Hz)
    figsize : tuple
        Figure size
    station : str
        Station name for title
    flag_smooth : bool
        Apply Baillard-style smoothing via resampling (default True)
    x_filter_per : float
        Gaussian filter width percentage in time direction (default 10)
    y_filter_per : float
        Gaussian filter width percentage in y direction (default 10)
    """
    import sws_methods as swm
    # Extract data
    df = results_df.copy()
    times = pd.to_datetime(df[time_column])
    dt_samples = df['dt'].values * sampling_rate
    
    # Convert to UTCDateTime for movehisto2d
    x = np.array([UTCDateTime(t) for t in times])
    y = dt_samples
    
    # Set time range
    x_start = UTCDateTime(times.min())
    x_end = UTCDateTime(times.max())
    y_start = 0
    y_end = 30

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with Baillard-style resampling and smoothing
    (ax, im, X, Y, Z, x_bins, x_diffs) = plot_movehisto2d(
        x, y,
        x_label='Date',
        y_label='Delay Time δt (samples)',
        ax=ax,
        x_width=x_width,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        x_over=x_overlap,
        y_over=y_overlap,
        y_width=y_width,
        flag_y_norm=True,  # Normalize by column (Baillard's norm_y)
        flag_resample=flag_smooth,  # Enable resampling for smoothing
        flag_filter=flag_smooth,  # Enable Gaussian filtering after resampling
        x_filter_per=x_filter_per,  # Smoothing percentage in time
        y_filter_per=y_filter_per,  # Smoothing percentage in y
        filter_mode='nearest',  # Non-cyclic edge handling
        vmax=None
    )
    
    # Add eruption line
    eruption_time = UTCDateTime(2015, 4, 24, 6)
    ax.axvline(eruption_time.matplotlib_date, color='white', 
               linestyle='--', alpha=0.7, linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Normalized Density', fontsize=10)
    
    # Add title with statistics
    dt_mean = df['dt'].mean()
    dt_std = df['dt'].std()
    title = (f"Delay Time - Station {station}\n"
             f"N = {len(df)} events | "
             f"δt: {dt_mean:.3f} ± {dt_std:.3f} s "
             f"({dt_mean*sampling_rate:.1f} ± {dt_std*sampling_rate:.1f} samples)")
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_phi_timeseries_movehisto_orig(results_df, time_column='event_datetime',
                                   x_width=5*24*3600, x_overlap=0.95,
                                   y_width=9, y_overlap=0.9,
                                   figsize=(14, 6),
                                   station='AXAS2',
                                   flag_smooth=True,
                                   x_filter_per=10,
                                   y_filter_per=10):
    """
    Plot fast direction (phi) over time using moving window 2D histogram with Baillard-style smoothing.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with splitting results
    time_column : str
        Name of time column (default 'event_datetime')
    x_width : float
        Width of time window in seconds (default 5 days = 5*24*3600)
    x_overlap : float
        Overlap fraction for time windows (0.95 = 95% overlap)
    y_width : float
        Bin width for phi in degrees (default 9 = 180/20)
    y_overlap : float
        Overlap for y-direction
    figsize : tuple
        Figure size
    station : str
        Station name for title
    flag_smooth : bool
        Apply Baillard-style smoothing via resampling (default True)
    x_filter_per : float
        Gaussian filter width percentage in time direction (default 10)
    y_filter_per : float
        Gaussian filter width percentage in y direction (default 10)
    """
    import sws_methods as swm
    
    # Extract data
    df = results_df.copy()
    times = pd.to_datetime(df[time_column])
    phi_deg = df['phi'].values
    
    # Convert to UTCDateTime for movehisto2d
    x = np.array([UTCDateTime(t) for t in times])
    y = phi_deg
    
    # Set time range
    x_start = UTCDateTime(times.min())
    x_end = UTCDateTime(times.max())
    y_start = -90
    y_end = 90
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with Baillard-style resampling and smoothing
    (ax, im, X, Y, Z, x_bins, x_diffs) = plot_movehisto2d(
        x, y,
        x_label='Date',
        y_label='Fast Direction φ (°)',
        ax=ax,
        x_width=x_width,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        x_over=x_overlap,
        y_over=y_overlap,
        y_width=y_width,
        flag_y_norm=True,  # Normalize by column (Baillard's norm_y)
        flag_resample=flag_smooth,  # Enable resampling for smoothing
        flag_filter=flag_smooth,  # Enable Gaussian filtering after resampling
        x_filter_per=x_filter_per,  # Smoothing percentage in time
        y_filter_per=y_filter_per,  # Smoothing percentage in y
        filter_mode=['wrap', 'nearest'],  # 'wrap' for cyclic phi, 'nearest' for time
        vmax=None
    )
    
    # Add eruption line
    eruption_time = UTCDateTime(2015, 4, 24, 6)
    ax.axvline(eruption_time.matplotlib_date, color='white', 
               linestyle='--', alpha=0.7, linewidth=2)
    
    # Add horizontal line at 0
    ax.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Normalized Density', fontsize=10)
    
    # Add title with statistics
    phi_mean = df['phi'].mean()
    phi_std = df['phi'].std()
    title = (f"Fast Direction - Station {station}\n"
             f"N = {len(df)} events | "
             f"φ: {phi_mean:.1f}° ± {phi_std:.1f}° "
             f"({np.deg2rad(phi_mean):.2f} ± {np.deg2rad(phi_std):.2f} rad)")
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

# Plot the rose plot data before the eruption and afterwards - updated plotting function

def plot_fast_direction_rose_eruption_comparison_orig(results_dict, eruption_time=None, 
                                                  title_prefix="Fast Direction Distribution",
                                                  nbins=36, figsize=(16, 7), color='steelblue',
                                                  edgecolor='black', linewidth=0.5):
    """
    Create side-by-side 360° rose plots comparing fast directions before and after eruption.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of splitting results with event_id as keys
    eruption_time : UTCDateTime
        Time of eruption onset (default: 2015-04-24T06:00:00)
    title_prefix : str
        Prefix for plot titles
    nbins : int
        Number of angular bins (default 36 = 10° bins for 360°)
    figsize : tuple
        Figure size (width, height) for combined plot
    color : str
        Color for the histogram bars
    edgecolor : str
        Color for bar edges
    linewidth : float
        Width of bar edges
    """
    if eruption_time is None:
        eruption_time = UTCDateTime(2015, 4, 24, 6)  # Nooner and Chadwick 2016
    
    # Split results into before and after eruption
    results_before = {}
    results_after = {}
    
    for event_id, result in results_dict.items():
        event_time = UTCDateTime(result['result']['event_datetime'])
        if event_time < eruption_time:
            results_before[event_id] = result
        else:
            results_after[event_id] = result
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    
    # Before eruption plot (left)
    ax1 = fig.add_subplot(121, projection='polar')
    plot_rose_subplot(results_before, ax1, f"{title_prefix}\nBefore Eruption", 
                      nbins, color, edgecolor, linewidth)
    
    # After eruption plot (right)
    ax2 = fig.add_subplot(122, projection='polar')
    plot_rose_subplot(results_after, ax2, f"{title_prefix}\nAfter Eruption", 
                      nbins, color, edgecolor, linewidth)
    
    plt.tight_layout()
    return fig, (ax1, ax2), (results_before, results_after)


def plot_rose_subplot(results_dict, ax, title, nbins, color, edgecolor, linewidth):
    """
    Helper function to plot rose diagram on a given axis.
    """
    # Extract fast directions (phi) from results
    fast_directions = []
    for event_id, result in results_dict.items():
        phi = result['result']['phi']  # in degrees (-90 to +90)
        # Convert to 0-360 range and add 180° symmetry
        phi_0_180 = phi + 90  # Convert to 0-180 range
        phi_rad_1 = np.deg2rad(phi_0_180)
        phi_rad_2 = np.deg2rad(phi_0_180 + 180)  # Add 180° symmetric value
        fast_directions.append(phi_rad_1)
        fast_directions.append(phi_rad_2)
    
    if len(fast_directions) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        return
    
    fast_directions = np.array(fast_directions)
    
    # Create histogram bins (0 to 2π for 0° to 360°)
    bins = np.linspace(0, 2*np.pi, nbins + 1)
    
    # Calculate histogram
    counts, bin_edges = np.histogram(fast_directions, bins=bins)
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Width of each bar
    width = 2 * np.pi / nbins
    
    # Create the rose plot
    bars = ax.bar(bin_centers, counts, width=width, bottom=0.0,
                   color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=1.0)
    
    # Set theta direction (clockwise from North)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Set radial ticks
    ax.set_rlabel_position(45)
    
    # Add degree labels for full 360°
    tick_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    tick_positions = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Adjust y-limit
    max_count = counts.max()
    ax.set_ylim(0, max_count * 1.2)
    
    # Add title with statistics
    n_measurements = len(fast_directions) // 2  # Divide by 2 since we doubled for symmetry
    # Calculate circular mean
    original_directions = []
    for event_id, result in results_dict.items():
        phi = result['result']['phi']
        original_directions.append(np.deg2rad(phi))
    original_directions = np.array(original_directions)
    mean_direction = np.rad2deg(np.arctan2(np.sin(original_directions).sum(), 
                                           np.cos(original_directions).sum()))
    
    title_text = f"{title}\nN = {n_measurements}, Mean = {mean_direction:.1f}°"
    ax.set_title(title_text, va='bottom', fontsize=12, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_fast_direction_rose(results_dict, title="Fast Direction Distribution", 
                              nbins=36, figsize=(8, 8), color='steelblue',
                              edgecolor='black', linewidth=0.5):
    """
    Create a 360° polar rose plot (histogram) of fast directions with 180° symmetry.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of splitting results with event_id as keys
    title : str
        Title for the plot
    nbins : int
        Number of angular bins (default 36 = 10° bins for 360°)
    figsize : tuple
        Figure size (width, height)
    color : str
        Color for the histogram bars
    edgecolor : str
        Color for bar edges
    linewidth : float
        Width of bar edges
    """
    # Extract fast directions (phi) from results
    fast_directions = []
    for event_id, result in results_dict.items():
        phi = result['result']['phi']  # in degrees (-90 to +90)
        # Convert to 0-360 range and add 180° symmetry
        # Map -90 to 90 range to 0 to 180, then add symmetric values
        phi_0_180 = phi + 90  # Convert to 0-180 range
        phi_rad_1 = np.deg2rad(phi_0_180)
        phi_rad_2 = np.deg2rad(phi_0_180 + 180)  # Add 180° symmetric value
        fast_directions.append(phi_rad_1)
        fast_directions.append(phi_rad_2)
    
    fast_directions = np.array(fast_directions)
    
    # Create polar histogram
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Create histogram bins (0 to 2π for 0° to 360°)
    bins = np.linspace(0, 2*np.pi, nbins + 1)
    
    # Calculate histogram
    counts, bin_edges = np.histogram(fast_directions, bins=bins)
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Width of each bar
    width = 2 * np.pi / nbins
    
    # Create the rose plot
    bars = ax.bar(bin_centers, counts, width=width, bottom=0.0,
                   color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=0.7)
    
    # Set theta direction (clockwise from North)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Set radial ticks
    ax.set_rlabel_position(45)
    
    # Add degree labels for full 360°
    tick_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    tick_positions = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Adjust y-limit (counts are doubled due to symmetry)
    max_count = counts.max()
    ax.set_ylim(0, max_count * 1.2)
    
    # Add title with statistics
    n_measurements = len(fast_directions) // 2  # Divide by 2 since we doubled for symmetry
    # Calculate circular mean for original ±90° range
    original_directions = []
    for event_id, result in results_dict.items():
        phi = result['result']['phi']
        original_directions.append(np.deg2rad(phi))
    original_directions = np.array(original_directions)
    mean_direction = np.rad2deg(np.arctan2(np.sin(original_directions).sum(), 
                                           np.cos(original_directions).sum()))
    
    title_text = f"{title}\nN = {n_measurements}, Mean = {mean_direction:.1f}°"
    ax.set_title(title_text, va='bottom', fontsize=12, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig, ax

## Do it for every AXAS2 file
def output_plots_orig(filename, first_start, last_start, first_end, last_end, station):
    results_df = pd.read_csv(filename)

    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: UTCDateTime(x))

    # Convert results_df event_datetime to datetime for plotting
    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: x.datetime)

    fig, (ax1, ax2), (results_before, results_after) = plot_fast_direction_rose_eruption_comparison(
        results_df,
        title_prefix="Fast Direction Distribution, " + str(station) + ", SWSPy: " + str(first_start) + "-"+ str(last_start) + "σ, " + str(first_end) + "-" + str(last_end) + " Tmid End",
        nbins=36,  # 10° bins
        color='steelblue',
        figsize=(16, 7)
    )
    plt.show()

    results_df = pd.read_csv(filename)

    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: UTCDateTime(x))

    # Filter results_df to be +/- 24 hrs of the eruption onset
    eruption_time = UTCDateTime(2015, 4, 24, 6)
    time_window = 24 * 3600  # 24 hours in seconds
    start_time = eruption_time - time_window
    end_time = eruption_time + time_window
    mask = (results_df['event_datetime'] >= start_time) & (results_df['event_datetime'] <= end_time)
    results_df = results_df[mask]

    # Convert results_df event_datetime to datetime for plotting
    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: x.datetime)

    results_df_baillard_phi = results_df.copy()
    # Conversion formula: phi_ccw_from_E = 90° - phi_cw_from_N
    results_df_baillard_phi['phi'] = 90 - results_df_baillard_phi['phi']

    # Handle wrapping: keep values in -90° to +90° range
    # If result > 90°, subtract 180°
    # If result < -90°, add 180°
    results_df_baillard_phi['phi'] = results_df_baillard_phi['phi'].apply(
        lambda x: x - 180 if x > 90 else (x + 180 if x < -90 else x)
    )

    # Now plot on AXEC2 data

    fig, ax = plot_dt_timeseries_movehisto(
        results_df_baillard_phi, 
        station=str(station) + " " + str(first_start) + "-"+ str(last_start),
        x_width=100,  # 1 day time window (shorter)
        x_overlap=0.9,
        y_width=1,          # 1 sample bins (finer)
        y_overlap=0.9,
        flag_smooth=True,
        x_filter_per=5,     # Less smoothing in time (5%)
        y_filter_per=5,      # Less smoothing in dt (5%)
        figsize=(6,8)
    )
    plt.show()

    fig, ax = plot_phi_timeseries_movehisto(
        results_df_baillard_phi, 
        station=str(station) + " " + str(first_start) + "-"+ str(last_start),
        x_width=100,  # 1 day time window (shorter)
        x_overlap=0.9,
        y_width=180/40,          # 5 degree bins (finer)
        y_overlap=0.9,
        flag_smooth=True,
        x_filter_per=5,     # Less smoothing in time (5%)
        y_filter_per=5,      # Less smoothing in phi (5%)
        figsize=(6,8)
    )
    plt.show()

    return results_df

##### New definitions for updated plots

def plot_movehisto2d_orig(x,y,
                x_label='X',y_label='Y',title='',ax=None,vmax=None,
                **movehisto2d_kwargs):
    """
    Function made to plot an histo2d but using a moving window in both directions, this ensure better
    consistency between neighbor bins. 
    The histogram can also work when data is an obspy.UTCDateTime array, then the x_start and x_end must
    be given in UTCDateTime as well and the x_width should be given in seconds.
    UTCDatetime are converted to timestamps (seconds since 1970)
    
    Inputs
    ------
        x,y: np.array: arrays containing the data to apply histogram on (x can be UTCDateTime)
        x_width,y_width: float: width of the bins (in seconds for UTCDateTime)
        [x,y]_[start,end]: float: start and end for histogram edges
        [x,y]_over: float in [0,1]: overlap for windows [1 = full overlap]
        flag_y_norm: Boolean: True to normalize by the maximum in each column
        flag_resample: Boolean: Enable resampling to uniform grid (required for smoothing)
        flag_filter: Boolean: Enable Gaussian filtering (requires flag_resample=True)
        x_filter_per, y_filter_per: float in [0,100]: width percentage for smoothing
        filter_mode: str or list: 'nearest' or 'wrap' for edge handling
        [x,y]_label: str
        
    Ouputs
    ------
        ax, im, X, Y, Z, x_bins, x_diffs
        
    Comments:
    ---------
        Uses Baillard's movehisto2d_bin for resampling and filtering
        
    UsedIn
    ------
        SWSCat.plot_movehisto2d_time, wrapper functions
    """
    
    ### Check if is made of UTCDateTimes
    
    time_flag=False
    if isinstance(x[0],UTCDateTime):
        time_flag=True
        print('X is in UTCDateTime')
        if movehisto2d_kwargs.get('x_mode','window')=='window':
            print('Remember width should be given in seconds, otherwise memory error')
    
    ### Modify x_start and x_end, and x if x is time and convert to timestamps
    x_start=movehisto2d_kwargs.get('x_start',None)
    x_end=movehisto2d_kwargs.get('x_end',None)
    y_start=movehisto2d_kwargs.get('y_start',None)
    y_end=movehisto2d_kwargs.get('y_end',None)
    
    if time_flag:
        if (x_start is not None) & (not isinstance(x_start,UTCDateTime)):
            raise ValueError('x_start must be given in obspy.UTCDateTime')
        if (x_end is not None) & (not isinstance(x_end,UTCDateTime)):
            raise ValueError('x_end must be given in obspy.UTCDateTime')
        x=np.array([value.timestamp for value in x]) # (seconds since 1970-01-01T00:00:00)
        x_start=x_start.timestamp if x_start is not None else None
        x_end=x_end.timestamp if x_end is not None else None
        movehisto2d_kwargs['x_start']=x_start
        movehisto2d_kwargs['x_end']=x_end

    ### Bin the data (smoothing handled inside movehisto2d_bin via resampling + filtering)
    
    (X,Y,Z,x_bins,x_diffs)=swm.movehisto2d_bin(x,y,**movehisto2d_kwargs)

    # Added to fix edge artifacts issues - bins on edges have less data, highly sensitive to outliers
    # Drop first and last x-bins (edge time windows) before plotting
    if X.shape[1] > 2:          # only if we have at least 3 columns
        X = X[:, 1:-1]
        Y = Y[:, 1:-1]
        Z = Z[:, 1:-1]
        x_bins = x_bins[1:-1]
        x_diffs = x_diffs[1:-1]
    
    ########################
    #### Start plotting ####

    #### Grid spec
    
    bottom=0.15 if time_flag else 0.1
    
    ### Checks
    
    if ax is None:
        fig,ax = plt.subplots(gridspec_kw={'bottom':bottom,'left':0.15})
        
    if time_flag:
        X=np.array(swm.timestamp2matplotlib(X.ravel())).reshape(X.shape) # transform for plotting
        x_bins=swm.timestamp2matplotlib(x_bins)
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=30 ,ha='right')
        ax.set_xlim(swm.timestamp2matplotlib([x_start,x_end]))
    else:
        ax.set_xlim([x_start,x_end])
       
    (Xm,Ym)=swm.XY2XY_pcolormesh(X,Y) # To ensure Pcolormesh will be centered on bins

    #im=ax.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'),rasterized=True)
    im=ax.pcolormesh(Xm,Ym,Z,cmap=plt.cm.get_cmap('inferno'),rasterized=True,vmax=vmax)
    
    ax.set_ylim([y_start,y_end])
    
    ax.set_aspect('auto')
    if time_flag:
        ax.xaxis_date()
            
    ### Cosmetic
    
    ax.set_ylabel(y_label) 
    if not time_flag:
        ax.set_xlabel(x_label) 
   
    ###### Return
    
    return (ax,im,X,Y,Z,x_bins,x_diffs)

def plot_movehisto2d(x, y,
                     x_label='X', y_label='Y', title='', ax=None, vmax=None,
                     taper_fraction=0.05, cmap_param='Blues',
                     **movehisto2d_kwargs):
    """
    Plot a 2D moving-window histogram with improved edge handling.

    Improvements over original:
    - Normalisation flag is intercepted here so raw counts always come back
      from movehisto2d_bin, enabling proper column-sum normalisation.
    - Time axis padded by one full x_width before binning, then cropped back,
      so edge windows always have a full window of data.
    - Column-sum normalisation (rather than column-max) for stable density
      estimates regardless of how many events fall in each time window.
    - Cosine taper applied after normalisation to smoothly suppress edge
      columns without affecting the interior.

    Parameters
    ----------
    x, y : np.array
        Data arrays (x can be UTCDateTime).
    x_label, y_label, title : str
        Axis and plot labels.
    ax : matplotlib Axes, optional
        Axes to plot into; created if None.
    vmax : float, optional
        Colour scale maximum.
    taper_fraction : float
        Fraction of columns at each end of the time axis to cosine-taper
        after normalisation. Default 0.05 (5%).
    **movehisto2d_kwargs
        Passed through to movehisto2d_bin.
    """

    # ------------------------------------------------------------------
    # UTCDateTime handling
    # ------------------------------------------------------------------
    time_flag = isinstance(x[0], UTCDateTime)
    if time_flag:
        print('X is in UTCDateTime')
        if movehisto2d_kwargs.get('x_mode', 'window') == 'window':
            print('Remember width should be given in seconds, otherwise memory error')

    x_start = movehisto2d_kwargs.get('x_start', None)
    x_end   = movehisto2d_kwargs.get('x_end',   None)
    y_start = movehisto2d_kwargs.get('y_start',  None)
    y_end   = movehisto2d_kwargs.get('y_end',    None)

    if time_flag:
        if (x_start is not None) and not isinstance(x_start, UTCDateTime):
            raise ValueError('x_start must be given in obspy.UTCDateTime')
        if (x_end is not None) and not isinstance(x_end, UTCDateTime):
            raise ValueError('x_end must be given in obspy.UTCDateTime')
        x       = np.array([v.timestamp for v in x])
        x_start = x_start.timestamp if x_start is not None else None
        x_end   = x_end.timestamp   if x_end   is not None else None
        movehisto2d_kwargs['x_start'] = x_start
        movehisto2d_kwargs['x_end']   = x_end

    # ------------------------------------------------------------------
    # Intercept flag_y_norm so we always get raw counts back from
    # movehisto2d_bin and handle normalisation ourselves below.
    # ------------------------------------------------------------------
    #flag_y_norm = movehisto2d_kwargs.get('flag_y_norm', False)
    #movehisto2d_kwargs['flag_y_norm'] = False

    # ------------------------------------------------------------------
    # Pad the time axis by one full window width so edge bins always
    # have a complete window of data. Store original bounds for cropping.
    # ------------------------------------------------------------------
    x_width = movehisto2d_kwargs.get('x_width', None)

    original_x_start = x_start if x_start is not None else np.min(x)
    original_x_end   = x_end   if x_end   is not None else np.max(x)

    if x_width is not None:
        movehisto2d_kwargs['x_start'] = original_x_start - x_width
        movehisto2d_kwargs['x_end']   = original_x_end   + x_width

    # ------------------------------------------------------------------
    # Bin (always returns raw counts due to flag_y_norm=False above)
    # ------------------------------------------------------------------
    X, Y, Z, x_bins, x_diffs = swm.movehisto2d_bin(x, y, **movehisto2d_kwargs)

    # ------------------------------------------------------------------
    # Crop back to the original time range.
    # x_bins and X may differ in length when flag_resample=True, so
    # build separate masks from their own coordinate arrays.
    # ------------------------------------------------------------------
    keep_bins = (x_bins >= original_x_start) & (x_bins <= original_x_end)
    x_bins    = x_bins[keep_bins]
    x_diffs   = x_diffs[keep_bins]

    x_coords  = X[0, :]
    keep_grid = (x_coords >= original_x_start) & (x_coords <= original_x_end)
    X = X[:, keep_grid]
    Y = Y[:, keep_grid]
    Z = Z[:, keep_grid]

    # ------------------------------------------------------------------
    # Column-sum normalisation: each time column becomes a relative
    # density so the colour scale reflects where events cluster in phi
    # or dt regardless of how many events that window contains.
    # A smoothed denominator prevents a single outlier in a low-count
    # window from dominating.
    # ------------------------------------------------------------------
    #if flag_y_norm:
    #    col_sums = np.sum(Z, axis=0)
    #    col_sums_smooth = scipy.ndimage.gaussian_filter1d(
    #        col_sums, sigma=max(1, int(0.03 * Z.shape[1]))
    #    )
    #    col_sums_smooth[col_sums_smooth <= 0] = 1
    #    Z = Z / col_sums_smooth[None, :]

    # ------------------------------------------------------------------
    # Cosine taper on edge columns only, applied after normalisation.
    # Interior columns with few events are left intact — the resampling
    # and Gaussian filter inside movehisto2d_bin already interpolate
    # across them smoothly.
    # ------------------------------------------------------------------
    n_cols    = Z.shape[1]
    taper_len = max(2, int(taper_fraction * n_cols))
    taper     = np.ones(n_cols)
    taper[:taper_len]  = np.sin(np.linspace(0, np.pi / 2, taper_len))
    taper[-taper_len:] = np.sin(np.linspace(np.pi / 2, 0,  taper_len))
    Z = Z * taper[None, :]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    bottom = 0.15 if time_flag else 0.1

    if ax is None:
        fig, ax = plt.subplots(gridspec_kw={'bottom': bottom, 'left': 0.15})

    if time_flag:
        X      = np.array(swm.timestamp2matplotlib(X.ravel())).reshape(X.shape)
        x_bins = swm.timestamp2matplotlib(x_bins)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.set_xlim(swm.timestamp2matplotlib([original_x_start, original_x_end]))
    else:
        ax.set_xlim([original_x_start, original_x_end])

    Xm, Ym = swm.XY2XY_pcolormesh(X, Y)
    im = ax.pcolormesh(Xm, Ym, Z,
                       cmap=plt.cm.get_cmap(str(cmap_param)),
                       rasterized=True,
                       vmax=vmax)

    ax.set_ylim([y_start, y_end])
        
    ax.set_aspect('auto')

    if time_flag:
        ax.xaxis_date()

    ax.set_ylabel(y_label,fontsize=250)

    #if cmap_param == "Reds":
    #    ax.yaxis.tick_right()
    #    ax.yaxis.set_label_position("right")
    
    #if not time_flag:
    ax.set_xlabel(x_label, fontsize=250)

    ax.tick_params(axis='x', labelsize=200)
    ax.tick_params(axis='y', labelsize=200)

    return ax, im, X, Y, Z, x_bins, x_diffs



    
def plot_fast_direction_rose_eruption_comparison(results_df, eruption_start=None, eruption_end=None,
                                                  title_prefix="Fast Direction Distribution",
                                                  nbins=36, figsize=(16, 7), color='black',
                                                  edgecolor='black', linewidth=0.5,
                                                  time_column='event_datetime'):
    """
    Create side-by-side 360° rose plots comparing fast directions before and after eruption.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with splitting results containing 'phi' and time columns
    eruption_time : UTCDateTime or str
        Time of eruption onset (default: 2015-04-24T06:00:00)
    title_prefix : str
        Prefix for plot titles
    nbins : int
        Number of angular bins (default 36 = 10° bins for 360°)
    figsize : tuple
        Figure size (width, height) for combined plot
    color : str
        Color for the histogram bars
    edgecolor : str
        Color for bar edges
    linewidth : float
        Width of bar edges
    time_column : str
        Name of the time column in results_df (default 'event_datetime')
    """
    if eruption_start is None:
        eruption_start = UTCDateTime(2015, 4, 24, 6)
    elif not isinstance(eruption_start, UTCDateTime):
        eruption_start= UTCDateTime(eruption_start)

    if eruption_end is None:
        eruption_end = UTCDateTime(2015, 5, 19, 0)
    elif not isinstance(eruption_end, UTCDateTime):
        eruption_end = UTCDateTime(eruption_end)
    
    # Convert time column to UTCDateTime for comparison
    df_time = results_df[time_column].apply(lambda x: UTCDateTime(x) if not isinstance(x, UTCDateTime) else x)
    
    results_before = results_df[df_time < eruption_start]

    mask = (df_time >= eruption_start) & (df_time <= eruption_end)
    results_during = results_df[mask]
 
    results_after = results_df[df_time > eruption_end]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    
    # Before eruption plot (left)
    ax1 = fig.add_subplot(131, projection='polar')
    plot_rose_subplot(results_before, ax1, 'Pre-Eruption',
                      nbins, color, edgecolor, linewidth)
    
    # After eruption plot (right)
    ax2 = fig.add_subplot(132, projection='polar')
    plot_rose_subplot(results_during, ax2, 'Syn-Eruption', 
                      nbins, color, edgecolor, linewidth)

    ax3 = fig.add_subplot(133, projection='polar')
    plot_rose_subplot(results_after, ax3, 'Post-Eruption',
                      nbins, color, edgecolor, linewidth)

    fig.suptitle(f"{title_prefix}")
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3), (results_before, results_during, results_after)


def plot_rose_subplot(results_df, ax, title, nbins, color, edgecolor, linewidth):
    """
    Helper function to plot rose diagram on a given axis.
    """
    if len(results_df) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        return
    
    fast_directions = []
    #for phi in results_df['phi'].values:  # in degrees (-90 to +90)
        # Convert to 0-360 range and add 180° symmetry
    #    phi_0_180 = phi + 90  # Convert to 0-180 range
    #    phi_rad_1 = np.deg2rad(phi_0_180)
    #    phi_rad_2 = np.deg2rad(phi_0_180 + 180)  # Add 180° symmetric value
    #    fast_directions.append(phi_rad_1)
    #    fast_directions.append(phi_rad_2)
    

    for phi in results_df['phi'].values:  # in degrees (-90 to +90)
        phi_adjusted = phi % 360  # Wrap to 0-360 (handles negatives correctly)
        phi_rad_1 = np.deg2rad(phi_adjusted)
        phi_rad_2 = np.deg2rad((phi_adjusted + 180) % 360)
        fast_directions.append(phi_rad_1)
        fast_directions.append(phi_rad_2)

    fast_directions = np.array(fast_directions)
    
    # Create histogram bins (0 to 2π for 0° to 360°)
    bins = np.linspace(0, 2*np.pi, nbins + 1)
    
    # Calculate histogram
    counts, bin_edges = np.histogram(fast_directions, bins=bins)
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Width of each bar
    width = 2 * np.pi / nbins


    # Create the rose plot, coloring bins in the Juan de Fuca Ridge range (320-340°) red
    ridge_ranges = [(320, 350), (140, 170)]
    bar_colors = []
    for center in bin_centers:
        center_deg = np.rad2deg(center) % 360
        is_ridge = any(start <= center_deg <= end for start, end in ridge_ranges)
        bar_colors.append('red' if is_ridge else color)

    ax.bar(bin_centers, counts, width=width, bottom=0.0,
            color=bar_colors, edgecolor=edgecolor, linewidth=linewidth, alpha=0.7)

    
    # Set theta direction (clockwise from North)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Set radial ticks
    ax.set_rlabel_position(45)
    
    # Add degree labels for full 360°
    tick_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    tick_positions = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Adjust y-limit
    max_count = counts.max()
    ax.set_ylim(0, max_count * 1.2)
    
    # Add title with statistics
    n_measurements = len(fast_directions) // 2  # Divide by 2 since we doubled for symmetry
    # Calculate circular mean
    original_directions = np.deg2rad(results_df['phi'].values)
    mean_direction = np.rad2deg(np.arctan2(np.sin(original_directions).sum(), 
                                           np.cos(original_directions).sum()))
    
    title_text = f"{title}"
    ax.set_title(title_text, va='bottom', fontsize=12, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_dt_timeseries_movehisto(results_df, time_column='event_datetime',
                                  x_width=5*24*3600, x_overlap=0.95,
                                  y_width=2, y_overlap=0.9,
                                  sampling_rate=200.0,
                                  figsize=(14*8, 6*8),
                                  station='AXAS2',
                                  flag_smooth=True,
                                  x_filter_per=10,
                                  y_filter_per=10):
    """
    Plot delay time (dt) over time using moving window 2D histogram with Baillard-style smoothing.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with splitting results
    time_column : str
        Name of time column (default 'event_datetime')
    x_width : float
        Width of time window in seconds (default 5 days = 5*24*3600)
    x_overlap : float
        Overlap fraction for time windows (0.95 = 95% overlap)
    y_width : float
        Bin width for dt in samples
    y_overlap : float
        Overlap for y-direction
    sampling_rate : float
        Sampling rate in Hz (default 200 Hz)
    figsize : tuple
        Figure size
    station : str
        Station name for title
    flag_smooth : bool
        Apply Baillard-style smoothing via resampling (default True)
    x_filter_per : float
        Gaussian filter width percentage in time direction (default 10)
    y_filter_per : float
        Gaussian filter width percentage in y direction (default 10)
    """
    import sws_methods as swm
    # Extract data
    df = results_df.copy()
    times = pd.to_datetime(df[time_column])
    dt_samples = df['dt'].values * sampling_rate
    
    # Convert to UTCDateTime for movehisto2d
    x = np.array([UTCDateTime(t) for t in times])
    y = dt_samples
    
    # Set time range
    x_start = UTCDateTime(times.min())
    x_end = UTCDateTime(times.max())
    y_start = 0
    y_end = 30

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with Baillard-style resampling and smoothing
    (ax, im, X, Y, Z, x_bins, x_diffs) = plot_movehisto2d(
        x, y,
        x_label='Date',
        y_label='Delay Time δt (samples)',
        ax=ax,
        cmap_param='Blues',
        x_width=x_width,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        x_over=x_overlap,
        y_over=y_overlap,
        y_width=y_width,
        flag_y_norm=True,  # Normalize by column (Baillard's norm_y)
        flag_resample=flag_smooth,  # Enable resampling for smoothing
        flag_filter=flag_smooth,  # Enable Gaussian filtering after resampling
        x_filter_per=x_filter_per,  # Smoothing percentage in time
        y_filter_per=y_filter_per,  # Smoothing percentage in y
        filter_mode='nearest',  # Non-cyclic edge handling
        vmax=None
    )
    
    # Add eruption line
    eruption_start_time = UTCDateTime(2015, 4, 24, 6)
    ax.axvline(eruption_start_time.matplotlib_date, color='black', 
               linestyle='--', alpha=0.9, linewidth=15, label='Eruption Onset')
    
    ax.scatter(datetime(2015, 4, 24, 6), 15, marker='*', color='black', s=20000, zorder=5)
    

    eruption_end_time = UTCDateTime(2015, 5, 19, 0)
    ax.axvline(eruption_end_time.matplotlib_date, color='black', 
               linestyle='--', alpha=0.9, linewidth=15, label='Eruption End')
    
    ax.scatter(datetime(2015, 5, 19, 0), 15, marker='*', color='black', s=20000, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Normalized Density', fontsize=240)
    cbar.ax.tick_params(labelsize=190)
    cbar.set_ticks([0.0, 1.0])
    
    # Add title with statistics
    dt_mean = df['dt'].mean()
    dt_std = df['dt'].std()
    title = (f"Delay Time - Station {station}\n"
             f"N = {len(df)} events | "
             f"δt: {dt_mean:.3f} ± {dt_std:.3f} s "
             f"({dt_mean*sampling_rate:.1f} ± {dt_std*sampling_rate:.1f} samples)")
    #ax.set_title(title, fontsize=50, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_phi_timeseries_movehisto(results_df, time_column='event_datetime',
                                   x_width=5*24*3600, x_overlap=0.95,
                                   y_width=9, y_overlap=0.9,
                                   figsize=(14*8, 6*8),
                                   station='AXAS2',
                                   flag_smooth=True,
                                   x_filter_per=10,
                                   y_filter_per=10):
    """
    Plot fast direction (phi) over time using moving window 2D histogram with Baillard-style smoothing.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with splitting results
    time_column : str
        Name of time column (default 'event_datetime')
    x_width : float
        Width of time window in seconds (default 5 days = 5*24*3600)
    x_overlap : float
        Overlap fraction for time windows (0.95 = 95% overlap)
    y_width : float
        Bin width for phi in degrees (default 9 = 180/20)
    y_overlap : float
        Overlap for y-direction
    figsize : tuple
        Figure size
    station : str
        Station name for title
    flag_smooth : bool
        Apply Baillard-style smoothing via resampling (default True)
    x_filter_per : float
        Gaussian filter width percentage in time direction (default 10)
    y_filter_per : float
        Gaussian filter width percentage in y direction (default 10)
    """
    import sws_methods as swm
    
    # Extract data
    df = results_df.copy()
    times = pd.to_datetime(df[time_column])
    phi_deg = df['phi'].values
    
    # Convert to UTCDateTime for movehisto2d
    x = np.array([UTCDateTime(t) for t in times])
    y = phi_deg
    
    # Set time range
    x_start = UTCDateTime(times.min())
    x_end = UTCDateTime(times.max())
    y_start = -90
    y_end = 90
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with Baillard-style resampling and smoothing
    (ax, im, X, Y, Z, x_bins, x_diffs) = plot_movehisto2d(
        x, y,
        x_label='Date',
        y_label='Fast Direction φ (°)',
        ax=ax,
        cmap_param='Reds',
        x_width=x_width,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        x_over=x_overlap,
        y_over=y_overlap,
        y_width=y_width,
        flag_y_norm=True,  # Normalize by column (Baillard's norm_y)
        flag_resample=flag_smooth,  # Enable resampling for smoothing
        flag_filter=flag_smooth,  # Enable Gaussian filtering after resampling
        x_filter_per=x_filter_per,  # Smoothing percentage in time
        y_filter_per=y_filter_per,  # Smoothing percentage in y
        #filter_mode=['nearest', 'wrap'],  # 'wrap' for cyclic phi, 'nearest' for time
        filter_mode = ['wrap', 'nearest'],
        vmax=None
    )
    
    # Add eruption line
    eruption_start_time = UTCDateTime(2015, 4, 24, 6)
    ax.axvline(eruption_start_time.matplotlib_date, color='black', 
            linestyle='--', alpha=0.9, linewidth=10, label='Eruption Onset')
    
    ax.scatter(datetime(2015, 4, 24, 6), 0,  marker='*', color='black', s=20000, zorder=5)

    eruption_end_time = UTCDateTime(2015, 5, 19, 0)
    ax.axvline(eruption_end_time.matplotlib_date, color='black', 
               linestyle='--', alpha=0.9, linewidth=10, label='Eruption End')

    ax.scatter(datetime(2015, 5, 19, 0), 0,  marker='*', color='black', s=20000, zorder=5)
    
    # Add horizontal line at 0
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Normalized Density', fontsize=150)
    cbar.ax.tick_params(labelsize=100)
    
    # Add title with statistics
    phi_mean = df['phi'].mean()
    phi_std = df['phi'].std()
    title = (f"Fast Direction - Station {station}\n"
             f"N = {len(df)} events | "
             f"φ: {phi_mean:.1f}° ± {phi_std:.1f}° "
             f"({np.deg2rad(phi_mean):.2f} ± {np.deg2rad(phi_std):.2f} rad)")
    #ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

## Do it for every AXAS2 file
def output_plots(filename, first_start, last_start, first_end, last_end, station):
    results_df = pd.read_csv(filename)


    #results_df_baillard_phi = results_df.copy()
    # Conversion formula: phi_ccw_from_E = 90° - phi_cw_from_N
    #results_df_baillard_phi['phi'] = 90 - results_df_baillard_phi['phi']

    # Handle wrapping: keep values in -90° to +90° range
    # If result > 90°, subtract 180°
    # If result < -90°, add 180°
    #results_df_baillard_phi['phi'] = results_df_baillard_phi['phi'].apply(
    #    lambda x: x - 180 if x > 90 else (x + 180 if x < -90 else x)
    #)

    #results_df_baillard_phi['event_datetime'] = results_df['event_datetime'].apply(lambda x: UTCDateTime(x))

    fig, (ax1, ax2, ax3), (results_before, results_during, results_after) = plot_fast_direction_rose_eruption_comparison(
        results_df, eruption_start = None, eruption_end = None,
        title_prefix="Fast Direction Distribution, " + str(station),
        nbins=36,  # 10° bins
        color='steelblue',
        figsize=(16, 7)
    )
    plt.show()

    results_df = pd.read_csv(filename)

    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: UTCDateTime(x))

    # Filter results_df to be +/- 24 hrs of the eruption onset
    eruption_time = UTCDateTime(2015, 4, 24, 6)
    time_window = 24 * 3600  # 24 hours in seconds
    start_time = eruption_time - time_window
    end_time = eruption_time + time_window
    mask = (results_df['event_datetime'] >= start_time) & (results_df['event_datetime'] <= end_time)
    results_df = results_df[mask]

    # Convert results_df event_datetime to datetime for plotting
    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: x.datetime)

    results_df_baillard_phi = results_df.copy()
    # Conversion formula: phi_ccw_from_E = 90° - phi_cw_from_N
    results_df_baillard_phi['phi'] = 90 - results_df_baillard_phi['phi']

    # Handle wrapping: keep values in -90° to +90° range
    # If result > 90°, subtract 180°
    # If result < -90°, add 180°
    results_df_baillard_phi['phi'] = results_df_baillard_phi['phi'].apply(
        lambda x: x - 180 if x > 90 else (x + 180 if x < -90 else x)
    )

    # Now plot on AXEC2 data

    fig, ax = plot_dt_timeseries_movehisto(
        results_df_baillard_phi, 
        station=str(station) + ", SWSPy: " + str(first_start) + "-"+ str(last_start) + "σ, " + str(first_end) + "-" + str(last_end) + " Tmid End",
        x_width=200,  # 600 seconds
        x_overlap=0.9,
        y_width=1,          # 1 sample bins (finer)
        y_overlap=0.9,
        flag_smooth=True,
        x_filter_per=5,     # Less smoothing in time (5%)
        y_filter_per=3,      # Less smoothing in dt (5%)
        figsize=(6*6,8*6)
    )
    plt.show()

    fig, ax = plot_phi_timeseries_movehisto(
        results_df_baillard_phi,
        station=str(station) + ", SWSPy: " + str(first_start) + "-"+ str(last_start) + "σ, " + str(first_end) + "-" + str(last_end) + " Tmid End",
        x_width=200,  # 600 seconds
        x_overlap=0.9,
        y_width=180/40,          # 5 degree bins (finer)
        y_overlap=0.9,
        flag_smooth=True,
        x_filter_per=5,     # Less smoothing in time (5%)
        y_filter_per=3,      # Less smoothing in phi (5%)
        figsize=(6*6,8*6)
    )
    plt.show()

    return results_df


def plot_cosine_similarity_vs_time(
        station_data,
        nbins=36,
        output_filename=None,
        title="Cosine Similarity to 2015 Pre-Eruption Distribution (dt-weighted)",
        time_column='event_datetime',
        figsize=(16, 6)):
    """
    Line plot of the dt-weighted rose cosine similarity (relative to each
    station's own 2015 pre-eruption distribution) over annual/sub-annual
    time periods.

    Parameters
    ----------
    station_data : dict
        {station_name: (df_old, df_new)}
        df_old -- DataFrame covering 2015-2021 (required)
        df_new -- DataFrame covering 2022-2026, or None
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    ESTART = pd.Timestamp("2015-04-24 06:00:00")
    EEND   = pd.Timestamp("2015-05-19 00:00:00")

    def _prep(df, col):
        df = df.copy()
        def _to_dt(x):
            try:
                from obspy.core.utcdatetime import UTCDateTime as _U
                if isinstance(x, _U):
                    return x.datetime
            except ImportError:
                pass
            ts = pd.Timestamp(x)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            return ts.to_pydatetime()
        df["_dt"]   = pd.to_datetime(df[col].apply(_to_dt), utc=False)
        df["_year"] = df["_dt"].dt.year
        return df

    def _hist(phi, dt, nbins):
        if len(phi) == 0:
            return np.zeros(nbins)
        da, dw = [], []
        for p, w in zip(phi, dt):
            p = p % 360
            da.extend([np.deg2rad(p), np.deg2rad((p + 180) % 360)])
            dw.extend([w, w])
        counts, _ = np.histogram(np.array(da), bins=np.linspace(0, 2 * np.pi, nbins + 1),
                                 weights=np.array(dw))
        return counts

    def _cs(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))

    # Time periods: (label, x_date, filter_fn_on_df_with__dt__year, use_new_df)
    periods_old = [
        ("2015\nPre",    pd.Timestamp("2015-02-12"),
         lambda d: d["_dt"] < ESTART, False),
        ("2015\nDuring", pd.Timestamp("2015-05-06"),
         lambda d: (d["_dt"] >= ESTART) & (d["_dt"] <= EEND), False),
        ("2015\nPost",   pd.Timestamp("2015-08-20"),
         lambda d: (d["_dt"] > EEND) & (d["_year"] == 2015), False),
        ("2016", pd.Timestamp("2016-07-01"), lambda d: d["_year"] == 2016, False),
        ("2017", pd.Timestamp("2017-07-01"), lambda d: d["_year"] == 2017, False),
        ("2018", pd.Timestamp("2018-07-01"), lambda d: d["_year"] == 2018, False),
        ("2019", pd.Timestamp("2019-07-01"), lambda d: d["_year"] == 2019, False),
        ("2020", pd.Timestamp("2020-07-01"), lambda d: d["_year"] == 2020, False),
        ("2021", pd.Timestamp("2021-07-01"), lambda d: d["_year"] == 2021, False),
    ]
    periods_new = [
        ("2022", pd.Timestamp("2022-07-01"), lambda d: d["_year"] == 2022, True),
        ("2023", pd.Timestamp("2023-07-01"), lambda d: d["_year"] == 2023, True),
        ("2024", pd.Timestamp("2024-07-01"), lambda d: d["_year"] == 2024, True),
        ("2025", pd.Timestamp("2025-07-01"), lambda d: d["_year"] == 2025, True),
        ("2026", pd.Timestamp("2026-07-01"), lambda d: d["_year"] == 2026, True),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(station_data), 1)))

    for (station, (df_old, df_new)), color in zip(station_data.items(), colors):
        dp_old = _prep(df_old, time_column)
        dp_new = _prep(df_new, time_column) if df_new is not None else None

        ref = dp_old[dp_old["_dt"] < ESTART].dropna(subset=["phi", "dt"])
        ref_h = _hist(ref["phi"].values, np.abs(ref["dt"].values), nbins)

        all_periods = periods_old + (periods_new if dp_new is not None else [])
        xs, ys = [], []
        for _label, xdate, mask_fn, use_new in all_periods:
            src = dp_new if use_new else dp_old
            if src is None:
                continue
            sub = src[mask_fn(src)].dropna(subset=["phi", "dt"])
            h = _hist(sub["phi"].values, np.abs(sub["dt"].values), nbins)
            xs.append(xdate)
            ys.append(_cs(h, ref_h))

        xs_arr = np.array(xs, dtype="datetime64[ns]")
        ys_arr = np.array(ys, dtype=float)
        valid  = ~np.isnan(ys_arr)
        if valid.sum() == 0:
            continue
        ax.plot(xs_arr[valid], ys_arr[valid], marker="o", label=station,
                color=color, linewidth=2, markersize=6, zorder=3)
        # mark N/A periods with an open circle at y=0
        if (~valid).sum():
            ax.scatter(xs_arr[~valid], np.zeros((~valid).sum()),
                       marker="x", color=color, s=40, zorder=3)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="cos sim = 1")
    ax.axvspan(ESTART, EEND, alpha=0.12, color="red", label="2015 eruption", zorder=1)
    ax.axvline(pd.Timestamp("2024-01-01"), color="orange", linestyle=":", linewidth=1.5,
               alpha=0.8, label="2024 inflation onset", zorder=2)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_filename:
        fig.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_filename}")

    return fig, ax
