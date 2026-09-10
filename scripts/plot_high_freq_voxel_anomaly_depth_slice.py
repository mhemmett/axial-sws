"""
Depth-slice map of the 3D lat/lon/depth voxel grid (0.25 km cubes) showing where the
mfast try_filters results have a significantly elevated fraction of high-dominant-
frequency (>=10 Hz) earthquakes, vs. low-frequency (<10 Hz).

Background/station/fault styling copied from the existing tomography scripts
(sws_tomography_inversion.py's _bathy/_sta_markers/_fault_traces idiom): grayscale
MBARI bathymetry GeoTIFF, gold station triangles, red ring-fault strike lines
projected to the slice depth via their 66-deg dip.

Voxels require >=20 events; significance is a one-sided binomial test of each
voxel's high-freq count against the overall high-freq fraction, Bonferroni-corrected
across all tested voxels.

Run with:
    python3 plot_high_freq_voxel_anomaly_depth_slice.py
"""

import glob
import math
import re
import warnings

import numpy as np
import pandas as pd
import tifffile
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binomtest

warnings.filterwarnings('ignore')

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))


def ll2xy(lat, lon):
    return ((np.asarray(lon) - INI_LON) * KM_PER_DEG_LON,
            (np.asarray(lat) - INI_LAT) * KM_PER_DEG_LAT)


STATION_FILE = 'raw_axec2_all_batches_data/../../data/stations_axial.llz'
STATION_FILE = '../data/stations_axial.llz'
STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1': 'AS1', 'AXAS2': 'AS2', 'AXCC1': 'CC1',
               'AXEC1': 'EC1', 'AXEC2': 'EC2', 'AXEC3': 'EC3'}
LABEL_OFFSET = {'AXAS1': (-0.20, -0.55), 'AXAS2': (-0.45, -0.55),
                'AXCC1': (-0.75, 0.25), 'AXEC1': (0.15, 0.30),
                'AXEC2': (0.15, 0.05), 'AXEC3': (0.30, -0.28)}

VOXEL = 0.25

X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0

# ── Load AXEC2 mfast results + metadata, classify by dominant frequency ────────

result_files = glob.glob('production_axec2_mfast_filters_lqt_pykonal_results/'
                          'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_*.csv')
batch_nums = sorted(int(re.search(r'batch_(\d+)\.csv', f).group(1)) for f in result_files)
files = [f'production_axec2_mfast_filters_lqt_pykonal_results/'
         f'splitting_results_mldd_2015_2021_axec2_mfast_filters_batch_{n}.csv' for n in batch_nums]

meta = pd.read_csv('raw_axec2_all_batches_mfast_filters_data/raw_axec2_all_batches_mfast_filters_metadata.csv')
meta_ll = meta[['event_id', 'latitude', 'longitude', 'depth']]

dfs = [pd.read_csv(f) for f in files]
dfs = [d for d in dfs if len(d) > 0]
events = pd.concat(dfs, ignore_index=True)
events = events[events['success'] == True].copy()
events = events[events['dt'] > 0]
events = events.dropna(subset=['chosen_filter_dom_period_samples'])
events['dom_freq'] = 200.0 / events['chosen_filter_dom_period_samples']
events = events.merge(meta_ll, on='event_id', how='left').dropna(subset=['latitude', 'longitude', 'depth'])
events['is_high'] = events['dom_freq'] >= 10.0

x, y = ll2xy(events['latitude'].values, events['longitude'].values)
events['x_km'] = x
events['y_km'] = y

events['ix'] = np.floor(events['x_km'] / VOXEL).astype(int)
events['iy'] = np.floor(events['y_km'] / VOXEL).astype(int)
events['iz'] = np.floor(events['depth'] / VOXEL).astype(int)

overall_p = events['is_high'].mean()

grp = events.groupby(['ix', 'iy', 'iz']).agg(n=('is_high', 'size'), n_high=('is_high', 'sum')).reset_index()
grp = grp[grp['n'] >= 20].copy()
grp['frac_high'] = grp['n_high'] / grp['n']

n_tests = len(grp)
grp['pval'] = [binomtest(int(r.n_high), int(r.n), overall_p, alternative='greater').pvalue
               for r in grp.itertuples()]
grp['sig_bonf'] = grp['pval'] < (0.05 / max(n_tests, 1))

grp['x_center'] = (grp['ix'] + 0.5) * VOXEL
grp['y_center'] = (grp['iy'] + 0.5) * VOXEL
grp['depth_center'] = (grp['iz'] + 0.5) * VOXEL

print(f'Batches: {batch_nums[0]}-{batch_nums[-1]} (N={len(batch_nums)})')
print(f'Total events: {len(events)}, overall high-freq fraction: {overall_p:.4f}')
print(f'Voxels with >=20 events: {n_tests}, Bonferroni-significant: {grp["sig_bonf"].sum()}')

sig = grp[grp['sig_bonf']].sort_values('pval')
print(sig[['x_center', 'y_center', 'depth_center', 'n', 'n_high', 'frac_high', 'pval']].to_string(index=False))

# ── Station coordinates ────────────────────────────────────────────────────────

_sta = pd.read_csv(STATION_FILE, sep=r'\s+', names=['lon', 'lat', 'elev_km', 'station'],
                    engine='python').set_index('station')
_sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'], _sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)

# ── Bathymetry (grayscale MBARI GeoTIFF, matching sws_tomography_inversion.py) ─

BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
          'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS = None
_p = PILImage.open(BATHY)
_t = _p.tag_v2
_olon, _olat = _t[33922][3], _t[33922][4]
_pl, _pb = _t[33550][0], _t[33550][1]
_nc, _nr = _p.size
_p.close()
_c0 = max(0, int(((INI_LON + X_START / KM_PER_DEG_LON) - _olon) / _pl) - 2)
_c1 = min(_nc, int(((INI_LON + X_END / KM_PER_DEG_LON) - _olon) / _pl) + 2)
_r0 = max(0, int((_olat - (INI_LAT + Y_END / KM_PER_DEG_LAT)) / _pb) - 2)
_r1 = min(_nr, int((_olat - (INI_LAT + Y_START / KM_PER_DEG_LAT)) / _pb) + 2)
_rgb = tifffile.imread(BATHY)[_r0:_r1, _c0:_c1]
_ds = max(1, max(_rgb.shape[:2]) // 1024)
_rgb = _rgb[::_ds, ::_ds]
_gray = np.dot(_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
_ext = [(_olon + _c0 * _pl - INI_LON) * KM_PER_DEG_LON, (_olon + _c1 * _pl - INI_LON) * KM_PER_DEG_LON,
        (_olat - _r1 * _pb - INI_LAT) * KM_PER_DEG_LAT, (_olat - _r0 * _pb - INI_LAT) * KM_PER_DEG_LAT]


def _bathy(ax):
    ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', zorder=0)


def _sta_markers(ax):
    for sta, row in _sta.iterrows():
        ax.plot(row['x'], row['y'], '^', ms=6, mfc='#FFD700', mec='k', mew=0.7, zorder=12)
        dx, dy = LABEL_OFFSET.get(sta, (0.12, 0.12))
        ax.text(row['x'] + dx, row['y'] + dy, STA_DISPLAY.get(sta, sta), fontsize=7, zorder=13, color='#FFD700',
                fontweight='bold')


def _fault_traces(ax, depth_km=0.0):
    """Both ring faults strike N30 W (330 deg), dip 66 deg outward from caldera."""
    DEPTH_TOP = 0.1
    COT66 = math.cos(math.radians(66)) / math.sin(math.radians(66))
    horiz = (depth_km - DEPTH_TOP) * COT66
    faults = [
        (8.5, 4.0, 330., 60.),   # east: cx,cy,strike,dip_azimuth
        (7.0, 4.0, 330., 240.),  # west
    ]
    for cx, cy, strike, dip_az in faults:
        da = math.radians(dip_az)
        cx_z = cx + horiz * math.sin(da)
        cy_z = cy + horiz * math.cos(da)
        st = math.radians(strike)
        L = 8.
        x1, y1 = cx_z - L / 2 * math.sin(st), cy_z - L / 2 * math.cos(st)
        x2, y2 = cx_z + L / 2 * math.sin(st), cy_z + L / 2 * math.cos(st)
        ax.plot([x1, x2], [y1, y2], '-', color='#cc0000', lw=1.1, alpha=0.8, zorder=11)


# ── Depth slices to plot: the anomaly cluster's own depth band, plus context ───

DEPTH_SLICES = sorted(sig['depth_center'].round(3).unique()) if len(sig) else [1.375, 1.625]

with PdfPages('high_freq_voxel_anomaly_depth_slices.pdf') as pdf:
    for dz in DEPTH_SLICES:
        slice_voxels = grp[np.isclose(grp['depth_center'], dz)]

        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        _bathy(ax)
        _fault_traces(ax, depth_km=dz)
        _sta_markers(ax)

        sc = ax.scatter(slice_voxels['x_center'], slice_voxels['y_center'],
                         c=slice_voxels['frac_high'] * 100, s=slice_voxels['n'] * 3,
                         cmap='Reds', vmin=0, vmax=max(60, slice_voxels['frac_high'].max() * 100),
                         edgecolor='k', linewidth=0.5, zorder=15, alpha=0.9)

        sig_here = slice_voxels[slice_voxels['sig_bonf']]
        ax.scatter(sig_here['x_center'], sig_here['y_center'],
                   facecolor='none', edgecolor='lime', linewidth=2.0, s=sig_here['n'] * 3 + 60,
                   zorder=16, label='Bonferroni-significant')

        cb = fig.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label('% high-frequency ($\\geq$10 Hz) events')

        ax.set_xlim(X_START, X_END)
        ax.set_ylim(Y_START, Y_END)
        ax.set_xlabel('Local x (km)')
        ax.set_ylabel('Local y (km)')
        ax.set_title(f'AXEC2 mfast results, batches {batch_nums[0]}-{batch_nums[-1]}\n'
                      f'High-freq fraction by voxel, depth = {dz:.3f} km slice (0.25 km grid, N>=20/voxel)\n'
                      f'overall high-freq fraction = {overall_p*100:.1f}%')
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f'\nSaved {len(DEPTH_SLICES)}-page PDF: high_freq_voxel_anomaly_depth_slices.pdf')
