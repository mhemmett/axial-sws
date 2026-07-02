#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use("Agg")

# ## 1. Import Requirements
# 

# # Master Shear-Wave Splitting Workflow for Axial Seamount
# 
# This notebook provides a complete, clean workflow from raw earthquake catalog and waveform data to shear-wave splitting analysis results. The workflow follows proper sequencing and includes all necessary quality control measures. 
# 
# Instead of using catalog from Wilcock and Zhang or ML DD, we use the nlloc file for all stations from Christian's results.
# 
# ## Workflow Overview
# 
# 1. **Data Loading & Initial Setup** - Load earthquake catalog and station metadata
# 2. **Extended Time Window Creation** - Create proper time windows for waveform retrieval
# 3. **Waveform Data Retrieval** - Download seismic data with extended windows
# 4. **Quality Control Filters** - P-wave rectilinearity, SNR, and incidence angle filtering
# 5. **Geometric Calculations** - Back-azimuth and distance calculations
# 6. **Shear-Wave Splitting Analysis** - Dynamic parameter estimation and SWSPy analysis
# 7. **Results Processing & Visualization** - Compile and visualize splitting parameters
# 
# ## Key Improvements
# - Extended catalog creation moved to proper early position
# - Updated P-wave polarization analysis for true incidence angles
# - Integrated SNR calculations with proper S-wave timing
# - Clean separation of quality control steps
# 

# In[ ]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import obspy
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.event import read_events
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import swspy (installed via pip in axial-sws conda env)
import swspy

# Add scripts directory to path for custom modules
sys.path.append('.')
from get_all_traces import get_station_traces_batch
from splitting_functions import *
from teanby_clustering import *

# Set up plotting
plt.rcParams['figure.figsize'] = (12, 8)

print("Libraries imported successfully")
print(f"ObsPy version: {obspy.__version__}")
print(f"SWSPy available: {'Yes' if 'swspy' in sys.modules else 'No'}")
print(f"SWSPy location: {swspy.__file__}")


# In[ ]:


# Load Kaiwen's ML DD catalog
catalog_file = '../data/mldd_catalog_2015_2021.csv'

catalog = pd.read_csv(catalog_file)

#Load station information from Christian's data
stations_file = '../data/stations_axial.llz'
#Read llz file - reads like a text file with space delimiter
stations_df = pd.read_csv(stations_file, delim_whitespace=True, header=None, names=['Longitude (°W)', 'Latitude (°N)', 'Elevation (m)', 'Station ID'])
# Convert elevation column to m from km
stations_df['Elevation (m)'] = stations_df['Elevation (m)']*1000

print(f"Stations in catalog: {catalog['station'].value_counts()}")


# In[ ]:


# Look at AXEC1 station
axec1_catalog = catalog[catalog['station'] == 'AXEC1'].copy()

# Reset index to ensure clean indexing
axec1_catalog = axec1_catalog.reset_index(drop=True)

axec1_catalog['datetime'] = pd.to_datetime(axec1_catalog['event_datetime'], format='ISO8601')

#print(f"Total AXAS2 events in catalog: {len(axas2_catalog)}")
print(f"Total AXEC1 events in catalog: {len(axec1_catalog)}")

print(f"\nDate range of test catalog:")
print(f"Start: {axec1_catalog['datetime'].min()}")
print(f"End: {axec1_catalog['datetime'].max()}")

print(axec1_catalog)


# In[ ]:


test_catalog = axec1_catalog.copy()

print(f"\nDate range of test catalog:")
print(f"Start: {test_catalog['datetime'].min()}")
print(f"End: {test_catalog['datetime'].max()}")


# In[ ]:


# Convert to UTCDateTime
test_catalog['datetime'] = test_catalog['datetime'].apply(lambda x: UTCDateTime(x))

# Format p_time and s_time as UTCDateTime of datetime + p_arrival_time and s_arrival_time, respectively
# Format p_time and s_time as UTCDateTime of datetime + p_time and s_time, respectively
test_catalog['p_time'] = test_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_p'], axis=1)
test_catalog['s_time'] = test_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_s'], axis=1)

print("Successfully converted p_time and s_time to absolute UTCDateTime")
print(f"Sample p_time: {test_catalog['p_time'].iloc[0]}")
print(f"Sample s_time: {test_catalog['s_time'].iloc[0]}")


# In[ ]:


catalog = test_catalog.copy()


# In[ ]:


catalog['event_id'] = catalog.index


# ## 3. Extended Time Window Creation
# 
# This step creates extended time windows for waveform retrieval. This is critical for proper analysis and must happen early in the workflow, before any quality control that depends on waveform data.
# 

# In[ ]:


# Create extended time windows for proper waveform analysis
print("Creating extended time windows for waveform retrieval...")

# Apply extended windowing
extended_catalog = create_extended_catalog(catalog, pre_p_time=1.0, post_s_time=2.0)

print(f"Extended catalog created with {len(extended_catalog)} events")
print(f"Time windows: {extended_catalog['total_duration'].iloc[0]} seconds total")
print(f"Pre-event: {extended_catalog['pre_p_sec'].iloc[0]}s, Post-event: {extended_catalog['post_s_sec'].iloc[0]}s")
# Display sample of extended timing
print("\nSample timing windows:")
sample_cols = ['event_id', 'datetime', 'starttime', 'endtime', 'total_duration']
print(extended_catalog[sample_cols].head())


# In[ ]:


# Remove leading 'OO' from station names
extended_catalog['station'] = extended_catalog['station'].str.replace('OO', '', regex=False)


# In[ ]:


print(extended_catalog)


# In[ ]:


# Rename event_magnitude to mag
catalog = catalog.rename(columns={'event_magnitude': 'mag'})
print(f"Renamed columns: {catalog.columns.tolist()}")


# ## 4. Waveform Data Retrieval
# 
# This section retrieves seismic waveform data using the extended time windows. We'll load the existing trace data and organize it for processing.
# 

# In[ ]:


test_catalog = extended_catalog


# In[ ]:


# Replace catalog id with index
test_catalog['id'] = test_catalog.index


# In[ ]:


def get_station_traces_batch(df, filename, starttime, endtime, station_id, batch_size=250):
    """
    Fast bulk retrieval of waveform data with intelligent batched fallback.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with event information
    filename : str
        Output filename (without extension)
    starttime : str
        Column name for start time
    endtime : str
        Column name for end time
    station_id : str
        Column name for station ID
    
    Returns:
    --------
    obspy.Stream : All retrieved traces
    
    Performance:
    - Bulk success: ~10-15 seconds for 250 events
    - Batched fallback: ~30-60 seconds (250 events per batch)
    - Individual fallback: Only for failed batches
    """
    from obspy.clients.fdsn import Client
    from obspy.core.utcdatetime import UTCDateTime
    from obspy import Stream
    import time
    
    client = Client("https://service.earthscope.org")
    all_traces = Stream()

    # BATCHED- 250 events per batch
    batch_size = batch_size
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(df))
        batch_df = df.iloc[batch_start_idx:batch_end_idx]
        
        print(f"\n{'─'*60}")
        print(f"BATCH {batch_idx + 1}/{total_batches}")
        print(f"Events {batch_start_idx + 1} to {batch_end_idx} ({len(batch_df)} events)")
        print(f"{'─'*60}")
        
        # Build bulk request for this batch
        batch_bulk_list = []
        for _, row in batch_df.iterrows():
            t_start = UTCDateTime(row[str(starttime)]) - 0.5
            t_final = UTCDateTime(row[str(endtime)]) + 0.5
            current_station = row[str(station_id)]
            
            if current_station == 'AXEC1':
                batch_bulk_list.append(('OO', 'AXEC1', '', 'EHE', t_start, t_final))
                batch_bulk_list.append(('OO', 'AXEC1', '', 'EHN', t_start, t_final))
                batch_bulk_list.append(('OO', 'AXEC1', '', 'EHZ', t_start, t_final))
        
        # Try batch bulk request
        batch_start_time = time.time()
        batch_stream = client.get_waveforms_bulk(batch_bulk_list)
        batch_elapsed = time.time() - batch_start_time
        
        all_traces += batch_stream
        
        print(f"✓ Batch {batch_idx + 1} SUCCESS: {len(batch_stream)} traces in {batch_elapsed:.1f}s")
        print(f"  Expected: {len(batch_bulk_list)}, Retrieved: {len(batch_stream)}")
        
        if len(batch_stream) < len(batch_bulk_list):
            missing = len(batch_bulk_list) - len(batch_stream)
            print(f"  ⚠ Warning: {missing} traces missing from this batch")
            

            
            print(f"\n{'='*60}")
            print(f"BATCHED FALLBACK COMPLETE")
            print(f"{'='*60}")
            print(f"Total traces retrieved: {len(all_traces)}")
    
    # Save results
    if all_traces:
        print(f"\nSaving {len(all_traces)} traces to {filename}.mseed...")
        all_traces.write(str(filename) + ".mseed", format="MSEED")
        print(f"✓ File saved successfully")
        
        # Summary statistics
        print(f"\n{'='*60}")
        print(f"RETRIEVAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total events processed: {len(df)}")
        print(f"Total traces retrieved: {len(all_traces)}")
        print(f"Expected traces (max): {len(df) * 3}")
        print(f"Success rate: {len(all_traces)/(len(df)*3)*100:.1f}%")
        
    else:
        print(f"\n⚠ WARNING: No traces retrieved!")
    
    return all_traces


# In[ ]:


# Divide the test catalog into batches and retrieve waveforms for each batch
batch_size = 250
total_events = len(test_catalog)
total_batches = (total_events + batch_size - 1) // batch_size

all_results = []
for i in range(total_batches):
    batch_start_idx = i * batch_size
    batch_end_idx = min(batch_start_idx + batch_size, total_events)
    batch_df = test_catalog.iloc[batch_start_idx:batch_end_idx]

    batch_filename = f'../data/axial_mldd_2015_2021_axec1_batch_{i+1}'

    if os.path.exists(f'{batch_filename}.mseed'):
        print(f"Batch {i+1}/{total_batches} already exists ({batch_filename}.mseed), skipping download...")
        continue

    print(f"\n{'='*60}")
    print(f"PROCESSING BATCH {i + 1}/{total_batches}")
    print(f"Events {batch_start_idx + 1} to {batch_end_idx} ({len(batch_df)} events)")
    print(f"{'='*60}")

    # Retrieve waveforms for this batch
    batch_traces = get_station_traces_batch(batch_df, filename=batch_filename,
                                            starttime='starttime', endtime='endtime', station_id='station',
                                            batch_size=batch_size)


# In[ ]:


# Retrieve waveforms for all events in the test catalog using get_all_traces function
#print("Retrieving waveforms for all events in the test catalog...")
#waveforms = get_station_traces_batch(test_catalog, 'axial_mldd_2015_2021_axec1', 'starttime', 'endtime', 'station', batch_size=100)


# In[ ]:


# Combine all batches into a single stream
waveforms = obspy.Stream()
for i in range(total_batches):
    batch_filename = f'../data/axial_mldd_2015_2021_axec1_batch_{i+1}.mseed'
    if os.path.exists(batch_filename):
        batch_stream = obspy.read(batch_filename)
        waveforms += batch_stream
        print(f"Loaded {len(batch_stream)} traces from {batch_filename}")
    else:
        print(f"Warning: {batch_filename} not found, skipping...")


# In[ ]:


# Load waveforms from mseed file with obspy
#waveforms_file = 'axial_mldd_2015_2021_axec1.mseed'
#waveforms = obspy.read(waveforms_file)


# In[ ]:


# Associate waveforms with events in the catalog
print("Organizing waveforms by events...")
waveform_dict = organize_stream_by_events(waveforms, test_catalog)


# In[ ]:


# Rename columns
test_catalog = test_catalog.rename(columns={'event_depth': 'dep', 
                                            'event_lat': 'lat', 
                                            'event_lon': 'lon'})


# In[ ]:


test_catalog = test_catalog.rename(columns={'event_magnitude': 'mag'})


# In[ ]:


test_catalog


# In[ ]:


# Organize waveforms by event ID
print("Organizing waveforms by event ID...")
organized_waveforms = organize_waveform_data(waveform_dict, test_catalog)


# In[ ]:


# Format s_arrival_time and p_arrival_time as difference between arrival times and origin time
print("Formatting s_arrival_time and p_arrival_time as differences from origin time...")
for eid in organized_waveforms.keys():
    organized_waveforms[eid]['s_arrival_time'] = (UTCDateTime(test_catalog.loc[test_catalog['id'] == eid, 's_time'].values[0]) - 
                                                  UTCDateTime(test_catalog.loc[test_catalog['id'] == eid, 'datetime'].values[0]))
    organized_waveforms[eid]['p_arrival_time'] = (UTCDateTime(test_catalog.loc[test_catalog['id'] == eid, 'p_time'].values[0]) - 
                                                  UTCDateTime(test_catalog.loc[test_catalog['id'] == eid, 'datetime'].values[0]))


# In[ ]:


# For all traces in organized_waveforms, taper and filter in-place
print("Tapering and filtering all traces in organized_waveforms...")
events_to_remove = []
try:
    for eid in organized_waveforms.keys():
        for tr in organized_waveforms[eid]['traces']:
            tr.detrend("linear") # to avoid weird start and end amplitudes
            tr.taper(max_percentage=0.05, type='hann')
            tr.filter('bandpass', freqmin=5.0, freqmax=40.0)
except Exception as e:
    print(f"Error during waveform processing: {e}")
    if type(organized_waveforms[eid]['traces']) == type(None):
        events_to_remove.append(eid)
    print(f"Events with issues: {events_to_remove}")

print("Waveform retrieval and organization complete.")


# In[ ]:


# Find streams that are NoneType and remove from organized_waveforms
print("Checking for NoneType streams in organized_waveforms...")
events_to_remove = []
for eid in organized_waveforms.keys():
    if type(organized_waveforms[eid]['traces']) == type(None):
        events_to_remove.append(eid)
if events_to_remove:
    print(f"Removing {len(events_to_remove)} events with NoneType streams: {events_to_remove}")
    for eid in events_to_remove:
        del organized_waveforms[eid]
else:
    print("No NoneType streams found in organized_waveforms.")


# In[ ]:


# Remove duplicate traces from organized_waveforms
print("Checking for and removing duplicate traces in organized_waveforms...")

for eid in organized_waveforms.keys():
    # Get the stream for this event
    st = organized_waveforms[eid]['traces']
    
    # Check if there are duplicates
    try:
        if type(st) == type(None):
            print(f"Event {eid}: No traces found (NoneType)")
            continue

        else:
            if len(st) > 3:
                print(f"Event {eid}: Found {len(st)} traces (expected 3)")
                
                # Create a new stream with unique traces based on channel code
                unique_traces = {}
                for tr in st:
                    channel = tr.stats.channel
                    # Keep the first occurrence of each channel
                    if channel not in unique_traces:
                        unique_traces[channel] = tr
                
                # Replace the stream with deduplicated traces
                organized_waveforms[eid]['traces'] = obspy.Stream(traces=list(unique_traces.values()))
                print(f"  Reduced to {len(organized_waveforms[eid]['traces'])} unique traces")
    except Exception as e:
        print(f"Error processing event {eid}: {e}")
        organized_waveforms[eid]['traces'] = st[:3]  # Fallback to first 3 traces if error occurs

# Verify the results
print("\nVerification of trace counts after deduplication:")
trace_counts = {}
for eid in organized_waveforms.keys():
    count = len(organized_waveforms[eid]['traces'])
    trace_counts[count] = trace_counts.get(count, 0) + 1

print(f"Events with 3 traces: {trace_counts.get(3, 0)}")
if any(k != 3 for k in trace_counts.keys()):
    print("Events with unexpected trace counts:")
    for count, num_events in trace_counts.items():
        if count != 3:
            print(f"  {num_events} events with {count} traces")
else:
    print("All events have exactly 3 traces (E, N, Z)")


# In[ ]:


# Remove events that do not have exactly 3 traces
print("\nRemoving events that do not have exactly 3 traces...")
events_to_remove = []
for eid in organized_waveforms.keys():
    if len(organized_waveforms[eid]['traces']) != 3:
        events_to_remove.append(eid)

    # also remove events with any trace that has zero length (indicating a retrieval issue) or empty traces
    for tr in organized_waveforms[eid]['traces']:
        if tr.stats.npts == 0:
            print(f"Event {eid} has a trace with zero length, marking for removal")
            events_to_remove.append(eid)
            break

for eid in events_to_remove:
    del organized_waveforms[eid]


# ## 5. Quality Control Pipeline
# 
# This section implements comprehensive quality control measures including P-wave rectilinearity analysis, signal-to-noise ratio calculations, and incidence angle filtering.
# 

# In[ ]:


# Define quality control thresholds
QC_THRESHOLDS = {
    'min_snr': 2.0,           # Minimum S-wave signal-to-noise ratio
    'min_rectilinearity': 0.7, # Minimum P-wave rectilinearity
    'max_incidence': 30.0,     # Maximum incidence angle (degrees)
}

print("Quality control functions loaded successfully")
print(f"QC Thresholds: {QC_THRESHOLDS}")


# In[ ]:


# Check that all traces for same event have same length, and remove events that do not meet this criterion
print("Checking that all traces for the same event have the same length...")
events_to_remove = []
for eid in organized_waveforms.keys():
    trace_lengths = [tr.stats.npts for tr in organized_waveforms[eid]['traces']]
    if len(set(trace_lengths)) != 1:
        print(f"Event {eid} has traces of different lengths: {trace_lengths}, marking for removal")
        events_to_remove.append(eid)


# In[ ]:


# Check if any traces are length zero, and if so mark those events for removal
print("Checking for traces with zero length...")
events_to_remove = []
for eid in organized_waveforms.keys():
    for tr in organized_waveforms[eid]['traces']:
        if tr.stats.npts == 0:
            print(f"Event {eid} has a trace with zero length, marking for removal")
            events_to_remove.append(eid)
            break


# In[ ]:


if events_to_remove:
    print(f"Removing {len(events_to_remove)} events that do not have traces of the same length or have zero-length traces: {events_to_remove}")
    for eid in events_to_remove:
        del organized_waveforms[eid]
else:
    print("All events have traces of the same length and no zero-length traces found.")


# In[ ]:


# Calculate quality control metrics for organized waveforms
print("Calculating quality control metrics for organized waveforms...")

# 1. Calculate S-wave SNR
organized_waveforms = calculate_snr_for_organized_waveforms(organized_waveforms)

# 2. Calculate geographic back-azimuth, for coordinate rotation later
organized_waveforms = calculate_back_azimuth_for_organized_waveforms(organized_waveforms, stations_df)

# 3. Calculate incidence angle
organized_waveforms = calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms(organized_waveforms, p_arrival_variable='p_arrival_time', analysis_window=0.12)

#4. Calculate P-wave rectilinearity
organized_waveforms = calculate_rectilinearity_jurkevics_for_organized_waveforms(organized_waveforms, p_arrival_variable='p_arrival_time', analysis_window=0.12)


# In[ ]:


# Define passing_waveforms as those that meet all QC thresholds
passing_waveforms = apply_quality_control(organized_waveforms, QC_THRESHOLDS)


# In[ ]:


# Save the passing_waveforms
metadata_df = save_passing_waveforms(passing_waveforms, output_dir='passing_waveforms_data_mldd_2015_2021_axec1')
print(metadata_df.head())


# In[ ]:


#############


# ## 6. Shear-Wave Splitting Analysis
# 
# This section implements the core shear-wave splitting analysis using SWSPy with dynamic parameter estimation and comprehensive quality assessment.
# 

# In[ ]:


# Reload the data
passing_waveforms_reloaded = load_passing_waveforms(output_dir='passing_waveforms_data_mldd_2015_2021_axec1')

# Verify the reload worked correctly
print(f"Reloaded events: {len(passing_waveforms_reloaded)}")
print(f"\nSample reloaded event (ID: {list(passing_waveforms_reloaded.keys())[0]}):")
sample_event = passing_waveforms_reloaded[list(passing_waveforms_reloaded.keys())[0]]
print(f"  Station: {sample_event['station']}")
print(f"  Origin time: {sample_event['origin_time']}")
print(f"  Number of traces: {len(sample_event['traces'])}")
print(f"  Back azimuth: {sample_event['back_azimuth']:.2f}°")


# In[ ]:


passing_waveforms = passing_waveforms_reloaded.copy()


# In[ ]:


keys = passing_waveforms.keys()


# In[ ]:


for key in keys:
    st = passing_waveforms[key]['traces']
    st.detrend("linear")
    st.taper(type="hann", max_percentage=0.05)
    st.filter('bandpass', freqmin=5.0, freqmax=40.0)


# In[ ]:


batch_size = len(passing_waveforms) / 100


# In[ ]:


import numpy as np
# Determine the number of batches
num_batches = 100
# Get the keys from the dictionary
keys = list(passing_waveforms.keys())
# Split the keys into batches
key_batches = np.array_split(keys, num_batches)

# Create a list of dictionaries, where each dictionary is a batch of waveforms
batched_waveforms = []
for batch_keys in key_batches:
    batch = {key: passing_waveforms[key] for key in batch_keys}
    batched_waveforms.append(batch)

all_results = []
for i, batch in enumerate(batched_waveforms):
    result_filename = f'splitting_results_mldd_2015_2021_axec1_batch_{i+1}.csv'
    if os.path.exists(result_filename):
        print(f"Batch {i+1}/{num_batches} results already exist ({result_filename}), skipping...")
        continue

    print(f"Processing batch {i+1}/{num_batches} with {len(batch)} waveforms")
    results_swspy = perform_splitting_on_organized_waveforms(
        batch,
        first_window_start=2,
        last_window_start=1,
        first_window_end=1.5,
        last_window_end=2.5,
        n_win=10,
        s_pick_uncertainty=0.0395, # Kaiwen's MLdd catalog S-pick uncertainty
        plot_results=False
    )
    all_results.append(results_swspy)
    # Optionally, save intermediate results
    save_results_csv(results_swspy, file_name=result_filename)

# Combine results if necessary
final_results = {}
for res in all_results:
    final_results.update(res)

save_results_csv(final_results, file_name='splitting_results_mldd_2015_2021_axec1_all_batches.csv')


# In[ ]:




