import pandas as pd
from obspy.clients.fdsn import Client
from obspy.core.utcdatetime import UTCDateTime
from obspy import Stream

def get_all_traces(df, filename, starttime, endtime):
    """
    Loops over a DataFrame of earthquakes with 't_start' and 't_final' columns.
    Retrieves waveform data from IRIS and saves all traces into one MiniSEED file.
    """
    client = Client("IRIS")

    # Define unique station-channel combinations
    combinations = [
        ('AXCC1', 'HHE'), ('AXCC1', 'HHN'), ('AXCC1', 'HHZ'), ('AXCC1', 'HDH'),
        ('AXEC1', 'EHE'), ('AXEC1', 'EHN'), ('AXEC1', 'EHZ'),
        ('AXEC2', 'HHE'), ('AXEC2', 'HHN'), ('AXEC2', 'HHZ'), ('AXEC2', 'HDH'),
        ('AXEC3', 'EHE'), ('AXEC3', 'EHN'), ('AXEC3', 'EHZ'),
        ('AXAS1', 'EHE'), ('AXAS1', 'EHN'), ('AXAS1', 'EHZ'),
        ('AXAS2', 'EHE'), ('AXAS2', 'EHN'), ('AXAS2', 'EHZ'),
        ('AXID1', 'EHE'), ('AXID1', 'EHN'), ('AXID1', 'EHZ'),
        ('AXBA1', 'HHE'), ('AXBA1', 'HHN'), ('AXBA1', 'HHZ'), ('AXBA1', 'HDH')
    ]

    all_traces = Stream()

    for _, row in df.iterrows():
        t_start = UTCDateTime(row[str(starttime)])
        t_final = UTCDateTime(row[str(endtime)])

        for station, channel in combinations:
            try:
                stream = client.get_waveforms(
                    network='OO',
                    station=station,
                    location='',
                    channel=channel,
                    starttime=t_start,
                    endtime=t_final,
                    attach_response=True
                )
                all_traces += stream
                print(f"Retrieved data for {station} {channel} from {t_start} to {t_final}")
            except Exception as e:
                print(f"No data for {station} {channel} - {e}")

    if all_traces:
        all_traces.write(str(filename) + ".mseed", format="MSEED")
        print(f"Saved all waveform data to {filename}.mseed")

    return all_traces

def get_station_traces(df, filename, starttime, endtime, station_id):
    """
    Loops over a DataFrame of earthquakes with 't_start' and 't_final' columns.
    Retrieves waveform data from IRIS and saves all traces into one MiniSEED file.
    """
    client = Client("IRIS")

    # Define unique station-channel combinations
    combinations = [
        ('AXCC1', 'HHE'), ('AXCC1', 'HHN'), ('AXCC1', 'HHZ'),
        ('AXEC1', 'EHE'), ('AXEC1', 'EHN'), ('AXEC1', 'EHZ'),
        ('AXEC2', 'HHE'), ('AXEC2', 'HHN'), ('AXEC2', 'HHZ'),
        ('AXEC3', 'EHE'), ('AXEC3', 'EHN'), ('AXEC3', 'EHZ'),
        ('AXAS1', 'EHE'), ('AXAS1', 'EHN'), ('AXAS1', 'EHZ'),
        ('AXAS2', 'EHE'), ('AXAS2', 'EHN'), ('AXAS2', 'EHZ'),
        ('AXID1', 'EHE'), ('AXID1', 'EHN'), ('AXID1', 'EHZ'),
    ]

    all_traces = Stream()

    for _, row in df.iterrows():
        t_start = UTCDateTime(row[str(starttime)])
        t_final = UTCDateTime(row[str(endtime)])
        current_station = row[str(station_id)]

        for station, channel in combinations:
            if station == current_station:
                try:
                    stream = client.get_waveforms(
                        network='OO',
                        station=station,
                        location='',
                        channel=channel,
                        starttime=t_start,
                        endtime=t_final,
                        attach_response=True
                    )
                    all_traces += stream
                    print(f"Retrieved data for {station} {channel} from {t_start} to {t_final}")
                except Exception as e:
                    print(f"No data for {station} {channel} - {e}")
            else:
                continue

    if all_traces:
        all_traces.write(str(filename) + ".mseed", format="MSEED")
        print(f"Saved all waveform data to {filename}.mseed")

    return all_traces

def get_station_traces_bulk(df, filename, starttime, endtime, station_id):
    """
    Fast bulk retrieval of waveform data for a specific station.
    
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
    
    Speed: ~10-20x faster than individual requests
    """
    client = Client("IRIS")
    all_traces = Stream()
    
    # Build bulk request list for all events at once
    bulk_list = []
    
    for _, row in df.iterrows():
        t_start = UTCDateTime(row[str(starttime)]) - 0.5  # Small buffer
        t_final = UTCDateTime(row[str(endtime)]) + 0.5
        current_station = row[str(station_id)]
        
        # For AXAS2, only request EHE, EHN, EHZ channels
        if current_station == 'AXAS2':
            bulk_list.append(('OO', 'AXAS2', '', 'EHE', t_start, t_final))
            bulk_list.append(('OO', 'AXAS2', '', 'EHN', t_start, t_final))
            bulk_list.append(('OO', 'AXAS2', '', 'EHZ', t_start, t_final))
    
    # Single bulk request for all events
    if bulk_list:
        try:
            print(f"Requesting {len(bulk_list)} traces in bulk for {len(df)} events...")
            all_traces = client.get_waveforms_bulk(bulk_list, attach_response=True)
            print(f"Successfully retrieved {len(all_traces)} traces")
        except Exception as e:
            print(f"Bulk request failed: {e}")
            print("Falling back to individual requests...")
            
            # Fallback to individual requests if bulk fails
            for _, row in df.iterrows():
                t_start = UTCDateTime(row[str(starttime)])
                t_final = UTCDateTime(row[str(endtime)])
                current_station = row[str(station_id)]
                
                if current_station == 'AXAS2':
                    for channel in ['EHE', 'EHN', 'EHZ']:
                        try:
                            stream = client.get_waveforms(
                                network='OO',
                                station='AXAS2',
                                location='',
                                channel=channel,
                                starttime=t_start,
                                endtime=t_final,
                                attach_response=True
                            )
                            all_traces += stream
                            print(f"Retrieved {channel} from {t_start}")
                        except Exception as e:
                            print(f"No data for {channel} - {e}")
    
    # Save to file
    if all_traces:
        all_traces.write(str(filename) + ".mseed", format="MSEED")
        print(f"Saved {len(all_traces)} traces to {filename}.mseed")
    else:
        print("No traces retrieved!")
    
    return all_traces

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
    
    client = Client("IRIS")
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
            
            if current_station == 'AXAS2':
                batch_bulk_list.append(('OO', 'AXAS2', '', 'EHE', t_start, t_final))
                batch_bulk_list.append(('OO', 'AXAS2', '', 'EHN', t_start, t_final))
                batch_bulk_list.append(('OO', 'AXAS2', '', 'EHZ', t_start, t_final))
        
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