#!/usr/bin/env python
# coding: utf-8
import matplotlib
matplotlib.use("Agg")

# ## 1. Import Requirements
#
# # Master Shear-Wave Splitting Workflow for Axial Seamount — AXCC1 (Real-Time, 2022-2026)
#
# Same pipeline as AXCC1 2015-2021, adapted for the MLDD-RT catalog:
#   - Channels: HHE, HHN, HHZ
#   - Catalog: mldd_catalog_2022_2026_both_picks.csv

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

# Use local swspy from repo instead of pip-installed version
swspy_local_path = os.path.abspath('../swspy')
if swspy_local_path not in sys.path:
    sys.path.insert(0, swspy_local_path)
import swspy

# Add scripts directory to path for custom modules
sys.path.append('.')
from get_all_traces import get_station_traces_batch
from splitting_functions import *
from teanby_clustering import *

plt.rcParams['figure.figsize'] = (12, 8)

print("Libraries imported successfully")
print(f"ObsPy version: {obspy.__version__}")
print(f"SWSPy available: {'Yes' if 'swspy' in sys.modules else 'No'}")
print(f"SWSPy location: {swspy.__file__}")


# ## 2. Load catalog and filter for AXCC1

catalog_file = '../data/mldd_catalog_2022_2026_both_picks.csv'
catalog = pd.read_csv(catalog_file)

stations_file = '../data/stations_axial.llz'
stations_df = pd.read_csv(stations_file, delim_whitespace=True, header=None,
                          names=['Longitude (°W)', 'Latitude (°N)', 'Elevation (m)', 'Station ID'])
stations_df['Elevation (m)'] = stations_df['Elevation (m)'] * 1000

print(f"Stations in catalog: {catalog['station'].value_counts()}")

axcc1_catalog = catalog[catalog['station'] == 'AXCC1'].copy()
axcc1_catalog = axcc1_catalog.reset_index(drop=True)
axcc1_catalog['datetime'] = pd.to_datetime(axcc1_catalog['event_datetime'], format='ISO8601')

print(f"Total AXCC1 events in catalog: {len(axcc1_catalog)}")
print(f"\nDate range:")
print(f"  Start: {axcc1_catalog['datetime'].min()}")
print(f"  End:   {axcc1_catalog['datetime'].max()}")

test_catalog = axcc1_catalog.copy()


# ## 3. Convert times and build extended catalog

test_catalog['datetime'] = test_catalog['datetime'].apply(lambda x: UTCDateTime(x))
test_catalog['p_time'] = test_catalog.apply(
    lambda row: UTCDateTime(row['datetime']) + row['arrival_time_p'], axis=1)
test_catalog['s_time'] = test_catalog.apply(
    lambda row: UTCDateTime(row['datetime']) + row['arrival_time_s'], axis=1)

print("Successfully converted p_time and s_time to absolute UTCDateTime")
print(f"Sample p_time: {test_catalog['p_time'].iloc[0]}")
print(f"Sample s_time: {test_catalog['s_time'].iloc[0]}")

test_catalog['event_id'] = test_catalog.index

# Create extended time windows for waveform retrieval
print("Creating extended time windows for waveform retrieval...")
extended_catalog = create_extended_catalog(test_catalog, pre_p_time=1.0, post_s_time=2.0)

print(f"Extended catalog created with {len(extended_catalog)} events")
print(f"Time windows: {extended_catalog['total_duration'].iloc[0]} seconds total")

# Remove leading 'OO' from station names
extended_catalog['station'] = extended_catalog['station'].str.replace('OO', '', regex=False)

# Rename columns
extended_catalog = extended_catalog.rename(columns={'event_magnitude': 'mag'})

test_catalog = extended_catalog
test_catalog['id'] = test_catalog.index
print(test_catalog)


# ## 4. Waveform retrieval (batched, with resume support)

def get_station_traces_batch_axcc1(df, filename, starttime, endtime, station_id, batch_size=250):
    """
    Batched bulk waveform retrieval for AXCC1 (HHE/HHN/HHZ channels).
    """
    from obspy.clients.fdsn import Client
    from obspy.core.utcdatetime import UTCDateTime
    from obspy import Stream
    import time

    client = Client("https://service.earthscope.org")
    all_traces = Stream()

    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(df))
        batch_df = df.iloc[batch_start_idx:batch_end_idx]

        print(f"\n{'─'*60}")
        print(f"BATCH {batch_idx + 1}/{total_batches}")
        print(f"Events {batch_start_idx + 1} to {batch_end_idx} ({len(batch_df)} events)")
        print(f"{'─'*60}")

        batch_bulk_list = []
        for _, row in batch_df.iterrows():
            t_start = UTCDateTime(row[str(starttime)]) - 0.5
            t_final = UTCDateTime(row[str(endtime)]) + 0.5
            current_station = row[str(station_id)]

            if current_station == 'AXCC1':
                batch_bulk_list.append(('OO', 'AXCC1', '', 'HHE', t_start, t_final))
                batch_bulk_list.append(('OO', 'AXCC1', '', 'HHN', t_start, t_final))
                batch_bulk_list.append(('OO', 'AXCC1', '', 'HHZ', t_start, t_final))

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
        print(f"BATCHED RETRIEVAL COMPLETE")
        print(f"{'='*60}")
        print(f"Total traces retrieved so far: {len(all_traces)}")

    if all_traces:
        print(f"\nSaving {len(all_traces)} traces to {filename}.mseed...")
        all_traces.write(str(filename) + ".mseed", format="MSEED")
        print(f"✓ File saved successfully")

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


batch_size = 250
total_events = len(test_catalog)
total_batches = (total_events + batch_size - 1) // batch_size

for i in range(total_batches):
    batch_start_idx = i * batch_size
    batch_end_idx = min(batch_start_idx + batch_size, total_events)
    batch_df = test_catalog.iloc[batch_start_idx:batch_end_idx]

    batch_filename = f'../data/axial_mldd_2022_2026_axcc1_batch_{i+1}'

    if os.path.exists(f'{batch_filename}.mseed'):
        print(f"Batch {i+1}/{total_batches} already exists ({batch_filename}.mseed), skipping download...")
        continue

    print(f"\n{'='*60}")
    print(f"PROCESSING BATCH {i + 1}/{total_batches}")
    print(f"Events {batch_start_idx + 1} to {batch_end_idx} ({len(batch_df)} events)")
    print(f"{'='*60}")

    get_station_traces_batch_axcc1(batch_df, filename=batch_filename,
                                   starttime='starttime', endtime='endtime', station_id='station',
                                   batch_size=batch_size)


# ## 5. QC pipeline (combine batches → organize → filter → save)

passing_waveforms_dir = 'passing_waveforms_data_mldd_2022_2026_axcc1'
passing_waveforms_metadata = os.path.join(passing_waveforms_dir, 'passing_waveforms_metadata.csv')

if os.path.exists(passing_waveforms_metadata):
    print(f"Passing waveforms already saved at {passing_waveforms_dir}, skipping to splitting analysis...")
else:
    waveforms = obspy.Stream()
    for i in range(total_batches):
        batch_filename = f'../data/axial_mldd_2022_2026_axcc1_batch_{i+1}.mseed'
        if os.path.exists(batch_filename):
            batch_stream = obspy.read(batch_filename)
            waveforms += batch_stream
            print(f"Loaded {len(batch_stream)} traces from {batch_filename}")
        else:
            print(f"Warning: {batch_filename} not found, skipping...")

    print("Organizing waveforms by events...")
    waveform_dict = organize_stream_by_events(waveforms, test_catalog)

    test_catalog = test_catalog.rename(columns={'event_depth': 'dep',
                                                'event_lat': 'lat',
                                                'event_lon': 'lon'})
    test_catalog = test_catalog.rename(columns={'event_magnitude': 'mag'}, errors='ignore')

    print("Organizing waveforms by event ID...")
    organized_waveforms = organize_waveform_data(waveform_dict, test_catalog)

    print("Formatting arrival times as offsets from origin time...")
    for eid in organized_waveforms.keys():
        organized_waveforms[eid]['s_arrival_time'] = (
            UTCDateTime(test_catalog.loc[test_catalog['id'] == eid, 's_time'].values[0]) -
            UTCDateTime(test_catalog.loc[test_catalog['id'] == eid, 'datetime'].values[0]))
        organized_waveforms[eid]['p_arrival_time'] = (
            UTCDateTime(test_catalog.loc[test_catalog['id'] == eid, 'p_time'].values[0]) -
            UTCDateTime(test_catalog.loc[test_catalog['id'] == eid, 'datetime'].values[0]))

    print("Tapering and filtering all traces...")
    events_to_remove = []
    try:
        for eid in organized_waveforms.keys():
            for tr in organized_waveforms[eid]['traces']:
                tr.detrend("linear")
                tr.taper(max_percentage=0.05, type='hann')
                tr.filter('bandpass', freqmin=5.0, freqmax=40.0)
    except Exception as e:
        print(f"Error during waveform processing: {e}")
        if type(organized_waveforms[eid]['traces']) == type(None):
            events_to_remove.append(eid)

    # Remove NoneType streams
    events_to_remove = []
    for eid in organized_waveforms.keys():
        if type(organized_waveforms[eid]['traces']) == type(None):
            events_to_remove.append(eid)
    if events_to_remove:
        print(f"Removing {len(events_to_remove)} events with NoneType streams")
        for eid in events_to_remove:
            del organized_waveforms[eid]

    # Remove duplicate traces
    for eid in organized_waveforms.keys():
        st = organized_waveforms[eid]['traces']
        try:
            if type(st) == type(None):
                continue
            if len(st) > 3:
                unique_traces = {}
                for tr in st:
                    channel = tr.stats.channel
                    if channel not in unique_traces:
                        unique_traces[channel] = tr
                organized_waveforms[eid]['traces'] = obspy.Stream(traces=list(unique_traces.values()))
        except Exception as e:
            print(f"Error processing event {eid}: {e}")
            organized_waveforms[eid]['traces'] = st[:3]

    # Remove events without exactly 3 traces or with zero-length traces
    print("\nRemoving events that do not have exactly 3 traces...")
    events_to_remove = []
    for eid in organized_waveforms.keys():
        if len(organized_waveforms[eid]['traces']) != 3:
            events_to_remove.append(eid)
        for tr in organized_waveforms[eid]['traces']:
            if tr.stats.npts == 0:
                events_to_remove.append(eid)
                break
    for eid in events_to_remove:
        del organized_waveforms[eid]

    # Remove events with mismatched trace lengths
    print("Checking that all traces for the same event have the same length...")
    events_to_remove = []
    for eid in organized_waveforms.keys():
        trace_lengths = [tr.stats.npts for tr in organized_waveforms[eid]['traces']]
        if len(set(trace_lengths)) != 1:
            events_to_remove.append(eid)
        for tr in organized_waveforms[eid]['traces']:
            if tr.stats.npts == 0:
                events_to_remove.append(eid)
                break
    if events_to_remove:
        for eid in events_to_remove:
            del organized_waveforms[eid]

    QC_THRESHOLDS = {
        'min_snr': 2.0,
        'min_rectilinearity': 0.7,
        'max_incidence': 30.0,
    }

    print("Calculating QC metrics...")
    organized_waveforms = calculate_snr_for_organized_waveforms(organized_waveforms)
    organized_waveforms = calculate_back_azimuth_for_organized_waveforms(organized_waveforms, stations_df)
    organized_waveforms = calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms(
        organized_waveforms, p_arrival_variable='p_arrival_time', analysis_window=0.12)
    organized_waveforms = calculate_rectilinearity_jurkevics_for_organized_waveforms(
        organized_waveforms, p_arrival_variable='p_arrival_time', analysis_window=0.12)

    passing_waveforms = apply_quality_control(organized_waveforms, QC_THRESHOLDS)

    metadata_df = save_passing_waveforms(passing_waveforms, output_dir=passing_waveforms_dir)
    print(metadata_df.head())

# end of QC pipeline block


# ## 6. Shear-wave splitting analysis

passing_waveforms_reloaded = load_passing_waveforms(output_dir='passing_waveforms_data_mldd_2022_2026_axcc1')

print(f"Reloaded events: {len(passing_waveforms_reloaded)}")
sample_event = passing_waveforms_reloaded[list(passing_waveforms_reloaded.keys())[0]]
print(f"  Station: {sample_event['station']}")
print(f"  Origin time: {sample_event['origin_time']}")
print(f"  Number of traces: {len(sample_event['traces'])}")
print(f"  Back azimuth: {sample_event['back_azimuth']:.2f}°")

passing_waveforms = passing_waveforms_reloaded.copy()

for key in passing_waveforms.keys():
    st = passing_waveforms[key]['traces']
    st.detrend("linear")
    st.taper(type="hann", max_percentage=0.05)
    st.filter('bandpass', freqmin=5.0, freqmax=40.0)

num_batches = 100
keys = list(passing_waveforms.keys())
key_batches = np.array_split(keys, num_batches)

batched_waveforms = []
for batch_keys in key_batches:
    batch = {key: passing_waveforms[key] for key in batch_keys}
    batched_waveforms.append(batch)

results_dir = '../results'
os.makedirs(results_dir, exist_ok=True)

all_results = []
for i, batch in enumerate(batched_waveforms):
    base_name = f'splitting_results_mldd_2022_2026_axcc1_batch_{i+1}.csv'
    result_filename = os.path.join(results_dir, base_name)
    legacy_filename = result_filename + '.csv'
    if os.path.exists(result_filename) or os.path.exists(legacy_filename):
        print(f"Batch {i+1}/{num_batches} results already exist, skipping...")
        continue

    print(f"Processing batch {i+1}/{num_batches} with {len(batch)} waveforms")
    results_swspy = perform_splitting_on_organized_waveforms(
        batch,
        first_window_start=2,
        last_window_start=1,
        first_window_end=1.5,
        last_window_end=2.5,
        n_win=10,
        s_pick_uncertainty=0.0395,
        plot_results=False
    )
    all_results.append(results_swspy)
    save_results_csv(results_swspy, file_name=result_filename)

final_results = {}
for res in all_results:
    final_results.update(res)

save_results_csv(final_results,
                 file_name=os.path.join(results_dir, 'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'))
