# Shear Wave Splitting Analysis - Package Updates

## Changes Made to Address Missing Dependencies

### Problem
The original shear wave splitting code had dependencies on:
1. A missing `general` module (containing utilities for the original project)
2. The deprecated `mtspec` package (no longer maintained)

### Solutions Implemented

#### 1. General Module Workaround

Created stub implementations for the missing `general` module. The module contained utility functions that are registered before importing `shearwavesplit`:

**Key functions replaced:**
- `general.plotwaveform.cov_eig`: Eigenvalue computation for covariance matrices
- `general.util.smooth_curve`: Data smoothing (stub implementation)
- `general.projection.ll2xy`: Lat/lon to x/y projection (stub implementation)

**Implementation:**
- A standalone module: `scripts/general_stub.py`
- Inline stubs in notebooks for immediate use

#### 2. Multitaper Replacement for mtspec

Replaced the deprecated `mtspec` package with the modern `multitaper` package.

**Changes in `scripts/shearwavesplit.py`:**
```python
# OLD:
import mtspec
spec, freq = mtspec.mtspec(data, 1/fs, num_wind, nfft=nfft)

# NEW:
from multitaper import MTSpec
mtspec_obj = MTSpec(data, nw=nw, k=k, dt=dt, nfft=nfft)
spec = mtspec_obj.rspec()
freq = mtspec_obj.freq
```

**Key parameters:**
- `nw`: Time-bandwidth product (typically 2-4, controls frequency resolution)
- `k`: Number of tapers (typically 2*nw - 1, controls variance reduction)
- `dt`: Sample interval (1/sampling_rate)

### Installation Requirements

```bash
# Install modern multitaper package
pip install multitaper

# Other required packages
pip install obspy numpy matplotlib pandas scipy
```

### Files Modified

1. **scripts/shearwavesplit.py**
   - Replaced `import mtspec` with conditional `multitaper` import
   - Updated `get_dominant_period()` function to use `MTSpec` class
   - Added backward compatibility: both 'mtspec' and 'multitaper' method names work

2. **scripts/general_stub.py** (NEW)
   - Standalone module providing stub implementations
   - Can be imported and registered before `shearwavesplit`

3. **notebooks/shear_wave_splitting_workflow_christian.ipynb**
   - Added inline stub module creation
   - Added multitaper import handling
   - Added documentation cell explaining dependencies

4. **notebooks/shear_wave_splitting_workflow.ipynb**
   - Same updates as above

### Usage

#### Option 1: Use the Notebooks Directly
The notebooks now have all workarounds built-in. Just run the cells in order:

```python
# Cell 1 automatically:
# - Creates general module stubs
# - Handles multitaper import
# - Imports shearwavesplit

# Cell 2 loads pickle data
with open('data/AXAS2.clean.cat.pickle', 'rb') as f:
    sws_data = pickle.load(f)
```

#### Option 2: Use the Stub Module in Scripts
For standalone Python scripts:

```python
import sys
sys.path.insert(0, 'path/to/scripts')

# Register stub modules
import general_stub
general_stub.register_modules()

# Now import shearwavesplit
import shearwavesplit as sws
```

### Testing the Changes

To verify everything works:

```bash
# Test the stub module
cd scripts
python3 general_stub.py

# Should output:
# ✓ cov_eig works: eigenvalues = [...]
# ✓ smooth_curve works: ...
# ✓ ll2xy works: ...
# All stub functions working correctly!
```

### What Works Now

✅ **Loading pickle files**: All `SWSobs` objects load successfully
✅ **Extracting parameters**: Fast directions, delay times, rectilinearity
✅ **Basic analysis**: Statistical summaries, distributions
✅ **Visualization**: Histograms, scatter plots, rose diagrams
✅ **Quality control**: Filtering by rectilinearity and other metrics
✅ **Export**: Save results to CSV files

### What Requires Full Dependencies

⚠️ **New splitting analysis**: Computing new splitting measurements from raw waveforms requires:
- Full implementation of projection functions for back-azimuth calculations
- Proper smoothing functions for data processing
- The multitaper package for spectral analysis

The stub implementations are sufficient for **working with pre-computed results** stored in pickle files, but not for generating new splitting measurements from scratch.

### Backward Compatibility

The code maintains backward compatibility:
- The `method='mtspec'` parameter still works (internally uses multitaper)
- Existing pickle files load without modification
- All original attribute names and data structures preserved

### Future Improvements

If you need to perform new splitting analysis (not just load existing results):

1. **Install multitaper**:
   ```bash
   pip install multitaper
   ```

2. **Implement full projection functions**: Replace the stub `ll2xy` with proper geodetic calculations:
   ```python
   from pyproj import Transformer
   # or
   from obspy.geodetics import gps2dist_azimuth
   ```

3. **Implement proper smoothing**: Replace stub with actual smoothing algorithm if needed for your analysis.

### Questions or Issues?

If you encounter errors:
1. Check that `multitaper` is installed: `pip show multitaper`
2. Verify obspy is available: `python3 -c "import obspy; print(obspy.__version__)"`
3. Make sure you're running from the correct directory with `sys.path` set properly
4. Check the inline stub code is executed before importing `shearwavesplit`

### Credits

- Original `shearwavesplit` module: @baillard
- Pickle file format: SWSobs and MinLambda classes
- Multitaper replacement: Uses the modern `multitaper` package by M. Prieto et al.
- Workaround implementation: Michael Hemmett with assistance from Claude (Anthropic)
