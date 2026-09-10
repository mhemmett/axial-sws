#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deformation_geometry_stress_hemmett_newdata.py

Re-run of deformation_geometry_stress_hemmett.py's "simple" Baillard deformation-model
comparison (def_syn_6.xyzuvw / def_pre_1.xyzuvw -- the Scenario Syn-6 / Pre-1 best-fit
DMODELS runs, per this session's memory) against OUR splitting observations, using the new
mfast max_dt=0.2s data for AXCC1/AXEC1/AXEC2/AXEC3/AXAS1.

AXAS2 has no new-data rerun yet, so it is KEPT ON ITS OLD DATA/OLD QC (phi_error<20 deg,
dt_error<0.04s, no quality or dt<T_dom/2 cut -- exactly the original script's filter) and
clearly distinguished from the 5 new-data stations wherever it appears (legend, print output).
Only pre-eruption and syn-eruption events are used (same 2 periods as the sibling script), so
only each new-data station's 2015-2021 file is needed (2022-2026 is irrelevant here).

QC for the 5 new-data stations (per explicit user confirmation, matching this session's
established new-data convention rather than the original script's now-outdated filter):
    quality >= 0.5, dt < T_dom/2 (cycle-skip-risk cut, per-event dominant period),
    phi_error < 20 deg, dt_error < 0.05s.
AXEC2's 2015-2021 half is still the in-progress partial maxdt02 re-run (whatever batches have
completed so far); its event lat/lon/depth come from raw_axec2_all_batches_mfast_filters_data/
raw_axec2_all_batches_mfast_filters_metadata.csv (merged on event_id), since the batch CSVs
don't carry location directly (same convention as this session's other newdata scripts).

Everything else (DMODELS sigma1 field, ray tracing + per-voxel circular-median phi, fault/
geometry lines, plot layout) is identical to the sibling script -- see its docstring for the
full method description, not repeated here.

Produces:
    deformation_figures/def_syn_6_geometry_stress_hemmett_newdata.pdf  (period='syn')
    deformation_figures/def_syn_7_geometry_stress_hemmett_newdata.pdf  (period='syn')
    deformation_figures/def_syn_9_geometry_stress_hemmett_newdata.pdf  (period='syn')
    deformation_figures/def_pre_1_geometry_stress_hemmett_newdata.pdf  (period='pre')

Legend is drawn with an explicit high zorder so it always renders on top of the observed-phi
quivers (previously the quivers could sit on top of/obscure the legend box).

Run with:
    python3 deformation_geometry_stress_hemmett_newdata.py
"""

import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image as PILImage
import tifffile

import deformation_util as adutil
import GMT as ggmt
import projection as gproj

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pykonal_raytracer import BaillardRayTracer

disp_dir = os.path.join(HERE, '..', 'Axial_Deformation') + os.sep
output_dir = os.path.join(HERE, 'deformation_figures') + os.sep
RESULTS_DIR = os.path.join(HERE, '..', 'results') + os.sep
TRANSFER_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer') + os.sep
AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')
AXEC2_META_CSV = os.path.join(HERE, 'raw_axec2_all_batches_mfast_filters_data',
                              'raw_axec2_all_batches_mfast_filters_metadata.csv')
VELOCITY_FILE = os.path.join(HERE, '..', 'data', 'AXIAL_MODEL_3P_VELOCITY.S.mod.buf')

STATION_LIST = ['AXCC1', 'AXEC1', 'AXEC2', 'AXEC3', 'AXAS1', 'AXAS2']  # no AXID1
NEWDATA_STATIONS = {'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3', 'AXAS1'}
OLDDATA_STATIONS = {'AXAS2'}   # no new-data rerun yet -- stays on old data/old QC

ini_lon = -130.1
ini_lat = 45.9
idz = 0
poisson = 0.25
mu_el = 1

x_lim = [5, 11]
y_lim = [1, 8]

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

# New-data QC (5 new-data stations) -- this session's established convention.
QW_MIN = 0.5
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.05

# Old-data QC (AXAS2 only) -- unchanged from the original sibling script.
OLD_PHI_ERR_MAX = 20.0
OLD_DT_ERR_MAX = 0.04

# Ray-voxelization grid - same convention as the sibling script / sws_percent_anisotropy.py
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VOXEL_XY = 0.30
NX = int(round((X_END - X_START) / VOXEL_XY))
NY = int(round((Y_END - Y_START) / VOXEL_XY))
COUNT_MIN = 5

OLD_STATION_FILES = {
    'AXAS2': ['splitting_results_mldd_2015_2021_axas2.csv'],
}

NEWDATA_STATION_FILES_2015_2021 = {
    'AXCC1': 'splitting_results_AXCC1_2015_2021_all_batches.csv',
    'AXEC1': 'splitting_results_AXEC1_2015_2021_all_batches.csv',
    'AXEC3': 'splitting_results_AXEC3_2015_2021_all_batches.csv',
    'AXAS1': 'splitting_results_AXAS1_2015_2021_all_batches.csv',
}

# Full fault_coord_dic (deformation_analysis.py) - filtered per period by substring match
FAULT_COORD_DIC = {
    'dike_syn_1':     {'center':[8.2 ,5.2],'angle':89, 'length':[0,4]},
    'fault_pre_syn_2':{'center':[8.2 ,5.2],'angle':-62,'length':[0,1.5]},
    'fault_pre_syn_3':{'center':[8.75,3.79],'angle':118,'length':[0,1.5]},
    'dike_pre_1':     {'center':[9.05,4.4],'angle':97, 'length':[0,3.7]},
    'dike_pre_2':     {'center':[9.05,4.4],'angle':-99,'length':[0,5]},
    'fault_pre_1':    {'center':[7.12,4.88],'angle':-68,'length':[0,2]},
    'fault_pre_2':    {'center':[6.92,2.6],'angle':112,'length':[0,2]},
}


def circular_median_phi(phi_deg):
    """Circular median of axially-symmetric (mod 180) fast directions, doubled-angle trick."""
    if len(phi_deg) == 0:
        return np.nan
    angles = 2.0 * np.radians(phi_deg)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return float(np.degrees(np.arctan2(s, c)) / 2.0) % 180.0


def load_station_events_old(station, period):
    """Original sibling-script loader (AXAS2 only): old data, old QC (phi_error<20 deg,
    dt_error<0.04s, no quality or dt<T_dom/2 cut)."""
    paths = [p if os.path.isabs(p) else os.path.join(RESULTS_DIR, p)
             for p in OLD_STATION_FILES[station]]
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    df = df.dropna(subset=['phi', 'dt', 'phi_error', 'dt_error', 'event_lat', 'event_lon',
                            'event_depth'])
    df = df[(df['dt'] > 0) & (df['phi_error'] < OLD_PHI_ERR_MAX) & (df['dt_error'] < OLD_DT_ERR_MAX)]
    t = pd.to_datetime(df['event_datetime'], utc=True)
    if period == 'pre':
        df = df[t < ERUPTION_START]
    else:
        df = df[(t >= ERUPTION_START) & (t < ERUPTION_END)]
    return df


def load_station_events_newdata(station, period):
    """New mfast max_dt=0.2s data loader (5 new-data stations): quality>=0.5, dt<T_dom/2,
    phi_error<20 deg, dt_error<0.05s. Only pre/syn-eruption events are needed, so only the
    2015-2021 file(s) are loaded. Returns a dataframe with event_lat/event_lon/event_depth/
    event_datetime columns (renamed from the new-data schema) so downstream ray-tracing code
    is unchanged from the sibling script."""
    if station == 'AXEC2':
        batch_files = sorted(glob.glob(os.path.join(
            AXEC2_2015_2021_DIR,
            'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_*.csv')))
        batch_dfs = [pd.read_csv(f) for f in batch_files]
        batch_dfs = [d for d in batch_dfs if len(d) > 0]
        meta = pd.read_csv(AXEC2_META_CSV)[['event_id', 'latitude', 'longitude', 'depth']]
        for d in batch_dfs:
            d['dominant_period'] = d['chosen_filter_dom_period_samples'] / 200.0
        df = pd.concat(batch_dfs, ignore_index=True)
        df = df.merge(meta, on='event_id', how='left')
    else:
        df = pd.read_csv(os.path.join(TRANSFER_DIR, NEWDATA_STATION_FILES_2015_2021[station]))

    df = df[df['success'] == True].copy()
    df = df[df['dt'] > 0]
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'quality', 'dominant_period',
                            'phi_error', 'dt_error'])
    df = df[(df['quality'] >= QW_MIN) &
            (df['dt'] < df['dominant_period'] / 2.0) &
            (df['phi_error'] < PHI_ERR_MAX) &
            (df['dt_error'] < DT_ERR_MAX)]

    df = df.rename(columns={'latitude': 'event_lat', 'longitude': 'event_lon',
                            'depth': 'event_depth', 'datetime': 'event_datetime'})

    t = pd.to_datetime(df['event_datetime'], utc=True, format='ISO8601')
    if period == 'pre':
        df = df[t < ERUPTION_START]
    else:
        df = df[(t >= ERUPTION_START) & (t < ERUPTION_END)]
    return df


def load_station_events(station, period):
    if station in OLDDATA_STATIONS:
        return load_station_events_old(station, period)
    return load_station_events_newdata(station, period)


def voxelize_ray_2d(ray_xyz):
    """Distinct 2D (ix,iy) voxel columns a ray passes through (collapsed over depth)."""
    mid = 0.5*(ray_xyz[:-1] + ray_xyz[1:])
    ix = ((mid[:,0]-X_START)/VOXEL_XY).astype(int)
    iy = ((mid[:,1]-Y_START)/VOXEL_XY).astype(int)
    z  = mid[:,2]
    valid = (ix>=0)&(ix<NX)&(iy>=0)&(iy<NY)&(z>=0)&(z<=Z_MAX)
    return set(zip(ix[valid].tolist(), iy[valid].tolist()))


def build_ray_traced_phi_field(period, tracer, sta_xy):
    """Ray-trace all QC'd events for `period` across the 6 stations, and return
    (X_centers, Y_centers, PHI_median, COUNT) 2D arrays on the voxel grid."""
    phi_lists = {}  # (ix,iy) -> [phi_az]
    n_total = 0
    n_traced = 0
    for sta in STATION_LIST:
        df = load_station_events(sta, period)
        n_total += len(df)
        for _, row in df.iterrows():
            eq_x, eq_y = gproj.ll2xy(row['event_lon'], row['event_lat'], ini_lon, ini_lat)
            eq_z = float(row['event_depth'])
            if eq_z < 0 or eq_z > Z_MAX:
                continue
            try:
                ray = tracer.trace(sta, float(eq_x), float(eq_y), eq_z)
            except Exception:
                continue
            phi_az = float(row['phi']) % 180.0
            for vox in voxelize_ray_2d(ray):
                phi_lists.setdefault(vox, []).append(phi_az)
            n_traced += 1
    print(f"    {period}: {n_traced}/{n_total} events traced, {len(phi_lists)} voxel columns hit")

    PHI = np.full((NX,NY), np.nan)
    CNT = np.zeros((NX,NY), dtype=int)
    for (ix,iy), vals in phi_lists.items():
        CNT[ix,iy] = len(vals)
        if len(vals) >= COUNT_MIN:
            PHI[ix,iy] = circular_median_phi(vals)
    return PHI, CNT


def build_page(disp_filename, period, tracer, sta_xy, station_dic, bathy):
    print(f"\n=== {disp_filename} (period={period}) ===")

    ### DMODELS model field
    (X,Y,Z,Ux,Uy,Uz) = adutil.read_disp_file(os.path.join(disp_dir,disp_filename), flag_plot=False)
    X_slice=X[:,:,idz]; Y_slice=Y[:,:,idz]
    Ux_slice=Ux[:,:,idz]; Uy_slice=Uy[:,:,idz]
    SIGMA1 = adutil.compute_sigma1_2d(X_slice,Y_slice,Ux_slice,Uy_slice,poisson,mu_el)

    Vx = SIGMA1[:,:,0]; Vy = SIGMA1[:,:,1]
    phi_cal = np.degrees(np.arctan2(Vy, Vx))
    phi_cal = np.where(phi_cal>=90, phi_cal-180, phi_cal)
    phi_cal = np.where(phi_cal<=-90, phi_cal+180, phi_cal)
    az_cal = (-phi_cal+90) % 360

    STEP = 6
    Xq = X_slice[::STEP,::STEP]; Yq = Y_slice[::STEP,::STEP]
    r = np.radians(az_cal[::STEP,::STEP])
    Uq, Vq = np.sin(r), np.cos(r)

    ### Ray-traced observed phi field
    print("  Ray-tracing observed events and computing per-voxel circular median phi...")
    PHI_obs, CNT_obs = build_ray_traced_phi_field(period, tracer, sta_xy)

    ### Fault/geometry lines active for this period
    fault_subset = {k:v for k,v in FAULT_COORD_DIC.items() if period in k}

    ### Plot
    fig, ax = plt.subplots(figsize=[7,6])
    if bathy is not None:
        _gray, _ext = bathy
        ax.imshow(_gray, origin='upper', extent=_ext, aspect='auto', cmap='gray', alpha=0.8, zorder=0)

    ### Model principal-stress tick marks (blue)
    qkw = dict(scale=18, width=0.0035, headlength=0, headaxislength=0, headwidth=0,
               pivot='middle', color='blue', alpha=0.7, zorder=3)
    ax.quiver(Xq, Yq,  Uq,  Vq, **qkw)
    ax.quiver(Xq, Yq, -Uq, -Vq, **qkw)

    ### Feature/geometry lines - grayscale
    for key, feat in fault_subset.items():
        ggmt.plot_box(feat['center'], feat['angle'], feat['length'], width_prof=None,
                      ax=ax, flag_label=False, key=key, flag_plot=True, fontsize=12,
                      color='0.25', ls='-', lw=1.8, flag_cross=False)

    ### Observed fast direction - ray-traced, per-voxel circular median (red)
    obs_qkw = dict(scale=18, width=0.004, headlength=0, headaxislength=0, headwidth=0,
                   pivot='middle', color='red', alpha=0.9, zorder=7)
    iix, iiy = np.where(~np.isnan(PHI_obs))
    if len(iix):
        xo = X_START + (iix+0.5)*VOXEL_XY
        yo = Y_START + (iiy+0.5)*VOXEL_XY
        ro = np.radians(PHI_obs[iix,iiy])
        uo, vo = np.sin(ro), np.cos(ro)
        ax.quiver(xo, yo,  uo,  vo, **obs_qkw)
        ax.quiver(xo, yo, -uo, -vo, **obs_qkw)
    n_vox = len(iix)

    ### Stations - new-data stations in yellow, old-data (AXAS2) in white
    for sta in STATION_LIST:
        lon,lat = station_dic[sta]['lon'], station_dic[sta]['lat']
        x_sta,y_sta = gproj.ll2xy(lon,lat,ini_lon,ini_lat)
        color = '#FFD700' if sta in NEWDATA_STATIONS else '#FFFFFF'
        ax.plot(x_sta, y_sta, '^', ms=9, mfc=color, mec='black', mew=1.0, zorder=12)
        ax.text(x_sta+0.12, y_sta+0.12, sta, fontsize=7.5, fontweight='bold', zorder=13)

    ### Cosmetic
    ax.set_aspect('equal')
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]')
    period_label = 'Syn-eruption' if period=='syn' else 'Pre-eruption'
    disp_tag = os.path.splitext(disp_filename)[0]
    ax.set_title(f'{disp_tag}: modelled principal stress direction vs.\n'
                 f'ray-traced observed $\\phi$ ({period_label}, N_vox={n_vox})\n'
                 f'new mfast max_dt=0.2s data: AXCC1/AXEC1/AXEC2/AXEC3/AXAS1; '
                 f'AXAS2: old data (no rerun yet)')

    legend_elements=[
        mlines.Line2D([0],[0],color='blue',lw=1.8,label='Model $\\sigma_1$ (compression)'),
        mlines.Line2D([0],[0],color='0.25',lw=1.8,label='Dike/fault geometry'),
        mlines.Line2D([0],[0],color='red',lw=2.0,
                      label=f'Observed $\\phi$ (ray-traced median, N$\\geq${COUNT_MIN}/voxel)'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFD700', mec='k', ms=8,
                      label='New-data station'),
        mlines.Line2D([], [], marker='^', color='w', mfc='#FFFFFF', mec='k', ms=8,
                      label='Old-data station (AXAS2)'),
    ]
    leg = ax.legend(handles=legend_elements, fontsize=7, loc='lower right', framealpha=0.95)
    leg.set_zorder(100)   # above the observed-phi quivers (zorder=7) and everything else

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name_pdf = output_dir + f'{disp_tag}_geometry_stress_hemmett_newdata.pdf'
    fig.savefig(name_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name_pdf}")


def main():
    station_dic = ggmt.read_stationfile()
    sta_xy = {}
    for sta in STATION_LIST:
        lon,lat = station_dic[sta]['lon'], station_dic[sta]['lat']
        x,y = gproj.ll2xy(lon,lat,ini_lon,ini_lat)
        sta_xy[sta] = (float(x), float(y))

    print("Precomputing PyKonal FMM travel-time fields for all stations...")
    tracer = BaillardRayTracer(bathy_file=VELOCITY_FILE)
    for sta,(sx,sy) in sta_xy.items():
        tracer.precompute_station(sta, sx, sy)

    print("Loading bathymetry...")
    KM_PER_DEG_LAT = 111.32
    KM_PER_DEG_LON = 111.32*np.cos(np.radians(ini_lat))
    BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
             'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
    bathy = None
    if os.path.exists(BATHY):
        PILImage.MAX_IMAGE_PIXELS=None
        _p=PILImage.open(BATHY); _t=_p.tag_v2
        _olon,_olat=_t[33922][3],_t[33922][4]; _pl,_pb=_t[33550][0],_t[33550][1]
        _nc,_nr=_p.size; _p.close()
        _c0=max(0,int(((ini_lon+x_lim[0]/KM_PER_DEG_LON)-_olon)/_pl)-2)
        _c1=min(_nc,int(((ini_lon+x_lim[1]/KM_PER_DEG_LON)-_olon)/_pl)+2)
        _r0=max(0,int((_olat-(ini_lat+y_lim[1]/KM_PER_DEG_LAT))/_pb)-2)
        _r1=min(_nr,int((_olat-(ini_lat+y_lim[0]/KM_PER_DEG_LAT))/_pb)+2)
        _rgb=tifffile.imread(BATHY)[_r0:_r1,_c0:_c1]
        _ds=max(1,max(_rgb.shape[:2])//1024); _rgb=_rgb[::_ds,::_ds]
        _gray=np.dot(_rgb[...,:3].astype(np.float32),[0.299,0.587,0.114]).astype(np.uint8)
        _ext=[(_olon+_c0*_pl-ini_lon)*KM_PER_DEG_LON,(_olon+_c1*_pl-ini_lon)*KM_PER_DEG_LON,
              (_olat-_r1*_pb-ini_lat)*KM_PER_DEG_LAT,(_olat-_r0*_pb-ini_lat)*KM_PER_DEG_LAT]
        bathy = (_gray, _ext)
    else:
        print(f"  WARNING: bathymetry file not found at {BATHY} - skipping background")

    build_page('def_syn_6.xyzuvw', 'syn', tracer, sta_xy, station_dic, bathy)
    build_page('def_syn_7.xyzuvw', 'syn', tracer, sta_xy, station_dic, bathy)
    build_page('def_syn_9.xyzuvw', 'syn', tracer, sta_xy, station_dic, bathy)
    build_page('def_pre_1.xyzuvw', 'pre', tracer, sta_xy, station_dic, bathy)


if __name__ == '__main__':
    main()
