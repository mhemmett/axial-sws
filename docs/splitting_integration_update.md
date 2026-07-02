# Splitting Analysis Integration Update

## Summary of Changes

Updated the shear-wave splitting workflow to properly integrate `create_splitting_analysis()` with `perform_splitting_analysis()`, ensuring that the SWSPy splitting object is created before attempting to perform the analysis.

## Key Changes

### 1. Updated `create_splitting_analysis()` Function

**Purpose:** Creates a properly configured SWSPy splitting object from `organized_waveforms` event data.

**Signature:**
```python
def create_splitting_analysis(event_data, use_dynamic_params=True)
```

**What it does:**
1. Extracts traces from `event_data['traces']`
2. Calculates dynamic parameters (if `use_dynamic_params=True`)
3. Applies optimal bandpass filtering to the stream
4. Extracts required metadata: `station`, `back_azimuth`, `incidence`, `s_arrival_time`
5. Creates SWSPy splitting object using the exact pattern:
   ```python
   splitting_event = swspy.splitting.create_splitting_object(
       stream_filtered, 
       stations_in=[event_data['station']],
       back_azis_all_stations=[event_data['back_azimuth']],
       receiver_inc_angles_all_stations=[event_data['incidence']],
       S_phase_arrival_times=[s_arrival_absolute]
   )
   ```
6. Sets analysis parameters on the splitting object (window lengths, time shifts, etc.)
7. Stores dynamic parameters in `splitting_event.dynamic_params` for reference

**Returns:** Configured `swspy.splitting` object ready for analysis

### 2. Updated `perform_splitting_analysis()` Function

**Major Changes:**
- **Removed:** Manual trace extraction, filtering, windowing, and parameter calculation
- **Added:** Call to `create_splitting_analysis()` at the beginning
- **Simplified:** Now just creates the splitting object and runs the analysis

**New Workflow:**
```python
def perform_splitting_analysis(event_data, use_dynamic_params=True):
    # 1. Create splitting object (handles all setup)
    splitting_obj = create_splitting_analysis(event_data, use_dynamic_params)
    
    # 2. Run the splitting analysis
    splitting_obj.split()
    
    # 3. Extract and return results
    return results_dict
```

**Result Extraction:**
The function now includes flexible result extraction to handle different SWSPy versions:
- First tries: `splitting_obj.fast_direction`, `splitting_obj.delay_time`
- Then tries: `splitting_obj.results` dictionary
- Fallback: `splitting_obj.phi`, `splitting_obj.dt`

## Complete Workflow

The complete splitting analysis workflow now follows this pattern:

```
organized_waveforms (filtered by QC)
         ↓
perform_splitting_on_organized_waveforms()
         ↓
    [For each event]
         ↓
perform_splitting_analysis(event_data, use_dynamic_params=True)
         ↓
    ┌─────────────────────────────────────────────────────┐
    │ create_splitting_analysis(event_data)               │
    │   ├→ Extract traces from event_data                 │
    │   ├→ calculate_dynamic_parameters()                 │
    │   │    ├→ estimate_dominant_period()                │
    │   │    ├→ Compute window length                     │
    │   │    └→ Compute filter band                       │
    │   ├→ Apply optimal filtering to stream              │
    │   ├→ Extract metadata (station, back_az, inc, s_arr)│
    │   ├→ Create SWSPy splitting object                  │
    │   └→ Set analysis parameters on object              │
    │                                                       │
    │ Returns: splitting_obj                              │
    └─────────────────────────────────────────────────────┘
         ↓
    splitting_obj.split()  # Run the actual analysis
         ↓
    Extract results (phi, dt, errors, quality)
         ↓
Results with dynamic parameters and metadata
```

## Function Dependencies

```
perform_splitting_analysis()
    └── create_splitting_analysis()
        └── calculate_dynamic_parameters()
            └── estimate_dominant_period()
```

## Required Data in event_data

The `event_data` dictionary from `organized_waveforms` must contain:

```python
{
    'traces': obspy.Stream,           # Waveform data (N/E/Z components)
    'station': str,                   # Station name/ID
    'back_azimuth': float,            # Back-azimuth (degrees)
    'incidence': float,               # P-wave incidence angle (degrees)
    's_arrival_time': float,          # S-arrival time (seconds from origin)
    'datetime': str,                  # Event origin time (UTCDateTime compatible)
    'magnitude': float,               # Event magnitude (for metadata)
    'snr_horizontal': float,          # Horizontal SNR (for metadata)
    'latitude': float,                # Event latitude
    'longitude': float,               # Event longitude
    'depth': float                    # Event depth
}
```

## Benefits of This Architecture

### 1. **Separation of Concerns**
- `create_splitting_analysis()`: Handles object creation and setup
- `perform_splitting_analysis()`: Handles execution and result extraction
- `calculate_dynamic_parameters()`: Handles parameter optimization

### 2. **Flexibility**
- Can call `create_splitting_analysis()` separately if you want to inspect the object before running
- Can run multiple analyses on the same splitting object with different parameters

### 3. **Maintainability**
- Clear function boundaries
- Easy to debug (can examine splitting object state)
- Single source of truth for SWSPy object creation

### 4. **Consistency**
- All splitting objects created the same way
- Dynamic parameters calculated consistently
- Reduces code duplication

## Usage Example

```python
import splitting_functions as sf

# 1. Load and organize data
organized_waveforms = sf.organize_waveform_data(all_traces, catalog)

# 2. Calculate QC metrics
organized_waveforms = sf.calculate_snr_for_organized_waveforms(organized_waveforms)
organized_waveforms = sf.calculate_back_azimuth_for_organized_waveforms(
    organized_waveforms, stations_df
)
organized_waveforms = sf.calculate_p_wave_metrics_for_organized_waveforms(
    organized_waveforms
)

# 3. Apply quality control
qc_thresholds = {
    'min_snr': 2.0,
    'min_rectilinearity': 0.7,
    'max_incidence': 30.0
}
filtered_waveforms = sf.apply_quality_control(organized_waveforms, qc_thresholds)

# 4. Perform splitting analysis (now properly creates splitting objects)
splitting_results = sf.perform_splitting_on_organized_waveforms(filtered_waveforms)

# 5. Access results
for event_id, result in splitting_results['event_results'].items():
    print(f"Event {event_id}:")
    print(f"  φ = {result['phi']:.1f}° ± {result['phi_error']:.1f}°")
    print(f"  δt = {result['dt']:.3f}s ± {result['dt_error']:.3f}s")
```

## Alternative Usage: Manual Object Creation

You can also create and run splitting objects manually:

```python
# Get one event from organized_waveforms
event_data = filtered_waveforms['some_event_id']

# Create splitting object
splitting_obj = sf.create_splitting_analysis(event_data, use_dynamic_params=True)

# Inspect the object
print(f"Window pre-S: {splitting_obj.overall_win_start_pre_fast_S_pick}s")
print(f"Max time shift: {splitting_obj.max_t_shift_s}s")
print(f"Number of windows: {splitting_obj.n_win}")

# Run analysis
splitting_obj.split()

# Access results (adjust attribute names based on your SWSPy version)
print(f"Fast direction: {splitting_obj.fast_direction}°")
print(f"Delay time: {splitting_obj.delay_time}s")
```

## Troubleshooting

### Issue: "module object is not callable"

**Cause:** Stale import or incorrect import statement

**Solution:**
1. Restart Jupyter kernel
2. Ensure you're using: `from splitting_functions import *` or explicit imports
3. Reload module: 
   ```python
   import importlib
   import splitting_functions
   importlib.reload(splitting_functions)
   ```

### Issue: Missing SWSPy results attributes

**Cause:** Different SWSPy versions use different attribute names

**Solution:** The `perform_splitting_analysis()` function now includes fallback logic to handle multiple naming conventions. If needed, check your SWSPy documentation and adjust the result extraction code in `perform_splitting_analysis()`.

### Issue: "Missing or invalid back_azimuth"

**Cause:** Back-azimuth not calculated or is NaN

**Solution:** Ensure you run `calculate_back_azimuth_for_organized_waveforms()` before filtering/splitting:
```python
organized_waveforms = sf.calculate_back_azimuth_for_organized_waveforms(
    organized_waveforms, stations_df
)
```

## Notes

- The lint warnings about missing `mtspec` and `obspy.signal.spectral_estimation` imports are expected if those packages aren't installed. The code includes fallbacks.
- Dynamic parameters are optional - set `use_dynamic_params=False` to use fixed parameters.
- The splitting object creation follows the exact pattern you specified, ensuring compatibility with your SWSPy version.

---

**Last Updated:** November 19, 2025  
**Status:** ✅ Complete - Ready for Testing
