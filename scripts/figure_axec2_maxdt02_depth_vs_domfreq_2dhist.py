"""
2D histogram of earthquake depth vs. dominant frequency, from the newest AXEC2
mfast try_filters run (max_t_shift_s=0.2s, run_axec2_mfast_filters_maxdt02_batches.py),
using all batches completed so far.

Dominant frequency (Hz) = 200.0 / chosen_filter_dom_period_samples (200 Hz sampling
rate, same convention used throughout this repo, e.g. traveltime_anisotropy_
axec2_domperiod_filtered.py's dom_period_s = chosen_filter_dom_period_samples / 200.0).
Depth (km) comes from the raw mfast-filters metadata (event hypocenter depth), merged
on event_id.

No QC/quality filtering applied beyond what's already implicit in a event having a
chosen_filter_dom_period_samples value (i.e. every event that reached the splitting
measurement stage) -- this is a diagnostic of the pre-processing dominant-period/filter
selection, not of splitting measurement quality.

Dashed white reference lines at 10 Hz (dominant frequency) and 1.0 km (depth).

Produces: figure_axec2_maxdt02_depth_vs_domfreq_2dhist.pdf

Run with:
    python3 figure_axec2_maxdt02_depth_vs_domfreq_2dhist.py
"""

import glob
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                        'raw_axec2_all_batches_mfast_filters_metadata.csv')
OUT_PDF = os.path.join(HERE, 'figure_axec2_maxdt02_depth_vs_domfreq_2dhist.pdf')

FREQ_LINE_HZ = 10.0
DEPTH_LINE_KM = 1.0

DEPTH_MIN = 0.0
DEPTH_MAX = 2.0
FREQ_MIN = 3.0
FREQ_MAX = 23.0
NBINS = 80

print('Loading AXEC2 max_dt=0.2 batches completed so far...')
batch_files = glob.glob(os.path.join(
    RESULTS_DIR, 'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_*.csv'))
dfs = [pd.read_csv(f) for f in batch_files]
dfs = [d for d in dfs if len(d) > 0]
df = pd.concat(dfs, ignore_index=True)
print(f'  {len(batch_files)} batches, {len(df):,} measurements loaded')

df = df.dropna(subset=['chosen_filter_dom_period_samples'])
df['dom_freq_hz'] = 200.0 / df['chosen_filter_dom_period_samples']

meta = pd.read_csv(META_CSV)[['event_id', 'depth']]
df = df.merge(meta, on='event_id', how='left').dropna(subset=['depth'])
print(f'  {len(df):,} events with depth + dominant frequency')

df = df[(df['depth'] >= DEPTH_MIN) & (df['depth'] <= DEPTH_MAX) &
        (df['dom_freq_hz'] >= FREQ_MIN) & (df['dom_freq_hz'] <= FREQ_MAX)]
print(f'  {len(df):,} events within plot range '
      f'(depth {DEPTH_MIN}-{DEPTH_MAX}km, freq {FREQ_MIN}-{FREQ_MAX}Hz)')

fig, ax = plt.subplots(figsize=(7.5, 6.5))

h = ax.hist2d(df['dom_freq_hz'].values, df['depth'].values,
              bins=[np.linspace(FREQ_MIN, FREQ_MAX, NBINS + 1), np.linspace(DEPTH_MIN, DEPTH_MAX, NBINS + 1)],
              cmap='inferno', norm=LogNorm(), cmin=1)

ax.axvline(FREQ_LINE_HZ, color='white', linestyle='--', linewidth=1.3,
           label=f'{FREQ_LINE_HZ:.0f} Hz')
ax.axhline(DEPTH_LINE_KM, color='white', linestyle='--', linewidth=1.3,
           label=f'{DEPTH_LINE_KM:.1f} km')

ax.invert_yaxis()
ax.set_xlabel('Dominant frequency (Hz)', fontsize=11, fontweight='bold')
ax.set_ylabel('Depth (km)', fontsize=11, fontweight='bold')
ax.set_title(f'AXEC2 (mfast try_filters, max_dt=0.2s): depth vs. dominant frequency\n'
             f'N={len(df):,}, batches 1-{len(batch_files)}', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, framealpha=0.85, labelcolor='black')

cb = fig.colorbar(h[3], ax=ax)
cb.set_label('Event count (log scale)', fontsize=9)

fig.tight_layout()
fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
print(f'\nWrote {OUT_PDF}')
