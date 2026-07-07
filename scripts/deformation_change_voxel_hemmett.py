#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voxel-resolved version of deformation_change_pre1_syn6_hemmett.py's linear-fit+RMS
pre-to-syn CHANGE comparison. That script used one point per station (6 points,
from get_station_obs_hemmett's per-station median). This script instead uses every
spatial voxel with sufficient ray coverage in BOTH the pre- and syn-eruption
periods (ray-traced, per-voxel median dt / circular-median phi - same method as
deformation_geometry_stress_hemmett.py's observed field), giving a much larger,
spatially resolved sample (tens to low hundreds of voxels instead of 6 stations)
for the linear fit and RMS.

For each qualifying voxel (>=COUNT_MIN rays in both periods, within the map's
plotting domain):
    delta(sigma1-sigma2)_model = (syn_6 model norm at voxel) - (pre_1 model norm at voxel)
    delta(dt)_obs               = (syn median dt at voxel)    - (pre median dt at voxel)
    delta(Phi_cal)_model        = 180-wrapped change in modelled compression azimuth
    delta(Phi_obs)_obs          = 180-wrapped change in ray-traced median phi

Output: deformation_figures/pre_1_vs_syn_6_change_voxel_hemmett.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import deformation_util as adutil
import GMT as ggmt
import projection as gproj

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pykonal_raytracer import BaillardRayTracer

disp_dir = os.path.join(HERE, '..', 'Axial_Deformation') + os.sep
output_dir = os.path.join(HERE, 'deformation_figures') + os.sep
RESULTS_DIR = os.path.join(HERE, '..', 'results') + os.sep
VELOCITY_FILE = os.path.join(HERE, '..', 'data', 'AXIAL_MODEL_3P_VELOCITY.S.mod.buf')

STATION_LIST = ['AXCC1','AXEC1','AXEC2','AXEC3','AXAS1','AXAS2']  # no AXID1

ini_lon=-130.1
ini_lat=45.9
idz=0
poisson=0.25
mu_el=1

x_lim=[5,11]
y_lim=[1,8]

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')
PHI_ERR_MAX = 20.0
DT_ERR_MAX = 0.04

# Ray-voxelization grid - same convention as sws_percent_anisotropy.py /
# deformation_geometry_stress_hemmett.py
X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
Z_MAX = 4.0
VOXEL_XY = 0.30
NX = int(round((X_END-X_START)/VOXEL_XY))
NY = int(round((Y_END-Y_START)/VOXEL_XY))
COUNT_MIN = 5

STATION_FILES = {
    'AXCC1': ['splitting_results_mldd_2015_2021_axcc1_all_batches.csv'],
    'AXEC1': ['splitting_results_mldd_2015_2021_axec1_all_batches.csv'],
    'AXEC2': sorted(__import__('glob').glob(os.path.join(RESULTS_DIR,
                'splitting_results_mldd_2015_2021_axec2_batch_*.csv.csv'))),
    'AXEC3': ['splitting_results_mldd_2015_2021_axec3_all_batches.csv'],
    'AXAS1': ['splitting_results_mldd_2015_2021_axas1.csv'],
    'AXAS2': ['splitting_results_mldd_2015_2021_axas2.csv'],
}


def circular_median_phi(phi_deg):
    if len(phi_deg) == 0:
        return np.nan
    angles = 2.0*np.radians(phi_deg)
    s, c = np.mean(np.sin(angles)), np.mean(np.cos(angles))
    return float(np.degrees(np.arctan2(s, c))/2.0) % 180.0


def wrap_angle_diff(a1, a0):
    """180-deg-aware angle difference a1-a0, wrapped to [-90,90]."""
    return (a1 - a0 + 90) % 180 - 90


def load_station_events(station, period):
    paths = [p if os.path.isabs(p) else os.path.join(RESULTS_DIR, p) for p in STATION_FILES[station]]
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    df = df.dropna(subset=['phi','dt','phi_error','dt_error','event_lat','event_lon','event_depth'])
    df = df[(df['dt'] > 0) & (df['phi_error'] < PHI_ERR_MAX) & (df['dt_error'] < DT_ERR_MAX)]
    t = pd.to_datetime(df['event_datetime'], utc=True)
    if period == 'pre':
        df = df[t < ERUPTION_START]
    else:
        df = df[(t >= ERUPTION_START) & (t < ERUPTION_END)]
    return df


def voxelize_ray_2d(ray_xyz):
    mid = 0.5*(ray_xyz[:-1] + ray_xyz[1:])
    ix = ((mid[:,0]-X_START)/VOXEL_XY).astype(int)
    iy = ((mid[:,1]-Y_START)/VOXEL_XY).astype(int)
    z  = mid[:,2]
    valid = (ix>=0)&(ix<NX)&(iy>=0)&(iy<NY)&(z>=0)&(z<=Z_MAX)
    return set(zip(ix[valid].tolist(), iy[valid].tolist()))


def build_ray_traced_fields(period, tracer):
    """Per-voxel median dt, circular-median phi, and count for `period`."""
    phi_lists = {}
    dt_lists = {}
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
            dt_val = float(row['dt'])
            for vox in voxelize_ray_2d(ray):
                phi_lists.setdefault(vox, []).append(phi_az)
                dt_lists.setdefault(vox, []).append(dt_val)
            n_traced += 1
    print(f"    {period}: {n_traced}/{n_total} events traced, {len(phi_lists)} voxel columns hit")

    PHI = np.full((NX,NY), np.nan)
    DT  = np.full((NX,NY), np.nan)
    CNT = np.zeros((NX,NY), dtype=int)
    for vox, vals in phi_lists.items():
        ix,iy = vox
        CNT[ix,iy] = len(vals)
        if len(vals) >= COUNT_MIN:
            PHI[ix,iy] = circular_median_phi(vals)
            DT[ix,iy] = np.median(dt_lists[vox])
    return PHI, DT, CNT


def model_fields(disp_filename):
    """Native-grid model norm(sigma1-sigma2) and compression azimuth for one DMODELS file."""
    (X,Y,Z,Ux,Uy,Uz) = adutil.read_disp_file(os.path.join(disp_dir,disp_filename), flag_plot=False)
    X_slice=X[:,:,idz]; Y_slice=Y[:,:,idz]
    Ux_slice=Ux[:,:,idz]; Uy_slice=Uy[:,:,idz]
    SIGMA1 = adutil.compute_sigma1_2d(X_slice,Y_slice,Ux_slice,Uy_slice,poisson,mu_el)
    Vx = SIGMA1[:,:,0]; Vy = SIGMA1[:,:,1]
    norm = np.sqrt(Vx**2+Vy**2)
    phi_cal = np.degrees(np.arctan2(Vy, Vx))
    phi_cal = np.where(phi_cal>=90, phi_cal-180, phi_cal)
    phi_cal = np.where(phi_cal<=-90, phi_cal+180, phi_cal)
    az_cal = (-phi_cal+90) % 360
    return X_slice, Y_slice, norm, az_cal


def sample_nearest(X_slice, Y_slice, field, x, y):
    dist2 = (X_slice-x)**2 + (Y_slice-y)**2
    i,j = np.unravel_index(np.argmin(dist2), dist2.shape)
    return field[i,j]


def linear(x, a, b):
    x = np.asarray(x)
    return a*x + b


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

    print("Ray-tracing pre-eruption events...")
    PHI_pre, DT_pre, CNT_pre = build_ray_traced_fields('pre', tracer)
    print("Ray-tracing syn-eruption events...")
    PHI_syn, DT_syn, CNT_syn = build_ray_traced_fields('syn', tracer)

    print("Loading DMODELS model fields (def_pre_1, def_syn_6)...")
    Xg_pre, Yg_pre, norm_pre, az_pre = model_fields('def_pre_1.xyzuvw')
    Xg_syn, Yg_syn, norm_syn, az_syn = model_fields('def_syn_6.xyzuvw')

    ### Build per-voxel change dataset
    rows = []
    for ix in range(NX):
        for iy in range(NY):
            if CNT_pre[ix,iy] < COUNT_MIN or CNT_syn[ix,iy] < COUNT_MIN:
                continue
            xc = X_START + (ix+0.5)*VOXEL_XY
            yc = Y_START + (iy+0.5)*VOXEL_XY
            if not (x_lim[0] <= xc <= x_lim[1] and y_lim[0] <= yc <= y_lim[1]):
                continue

            model_norm_pre = sample_nearest(Xg_pre, Yg_pre, norm_pre, xc, yc)
            model_norm_syn = sample_nearest(Xg_syn, Yg_syn, norm_syn, xc, yc)
            model_az_pre = sample_nearest(Xg_pre, Yg_pre, az_pre, xc, yc)
            model_az_syn = sample_nearest(Xg_syn, Yg_syn, az_syn, xc, yc)

            d_stress = model_norm_syn - model_norm_pre
            d_dt = DT_syn[ix,iy] - DT_pre[ix,iy]
            d_phi_cal = wrap_angle_diff(model_az_syn, model_az_pre)
            d_phi_obs = wrap_angle_diff(PHI_syn[ix,iy], PHI_pre[ix,iy])

            rows.append(dict(x=xc, y=yc, d_stress=d_stress, d_dt=d_dt,
                              d_phi_cal=d_phi_cal, d_phi_obs=d_phi_obs))

    n_vox = len(rows)
    print(f"\n{n_vox} voxels with >= {COUNT_MIN} rays in BOTH periods, within plotting domain")

    d_dt_ms = np.array([r['d_dt']*1000 for r in rows])
    d_stress = np.array([r['d_stress'] for r in rows])
    d_phi_obs = np.array([r['d_phi_obs'] for r in rows])
    d_phi_cal = np.array([r['d_phi_cal'] for r in rows])

    ### Fit 1: delta dt (obs) vs delta stress (model)
    popt1, _ = curve_fit(linear, d_dt_ms, d_stress)
    rms1 = np.sqrt(np.mean((linear(d_dt_ms, *popt1) - d_stress)**2))

    ### Fit 2: delta Phi_obs vs delta Phi_cal (linear fit, not just 1:1 reference,
    ### since with many voxels a genuine regression is meaningful)
    popt2, _ = curve_fit(linear, d_phi_obs, d_phi_cal)
    rms2 = np.sqrt(np.mean((linear(d_phi_obs, *popt2) - d_phi_cal)**2))

    print(f"Panel 1 fit: slope={popt1[0]:.4f}/ms  intercept={popt1[1]:.4f}  RMS={rms1:.3f}")
    print(f"Panel 2 fit: slope={popt2[0]:.4f}  intercept={popt2[1]:.2f}deg  RMS={rms2:.2f}deg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[9, 4.2])
    plt.subplots_adjust(wspace=0.35)

    ### Panel 1
    ax1.scatter(d_dt_ms, d_stress, s=14, c='k', alpha=0.5, edgecolors='none')
    x_fit = np.array([d_dt_ms.min(), d_dt_ms.max()])
    ax1.plot(x_fit, linear(x_fit, *popt1), '--r', zorder=5)
    ax1.text(0.05, 0.92, f'N={n_vox}\nslope: {popt1[0]:.4f} /ms\nRMS: {rms1:.3f}',
             transform=ax1.transAxes, va='top', fontsize=8)
    ax1.axhline(0, color='0.6', lw=0.8, ls=':')
    ax1.axvline(0, color='0.6', lw=0.8, ls=':')
    ax1.set_xlabel(r'$\Delta \delta t_{obs}$ (syn $-$ pre) [ms]')
    ax1.set_ylabel(r'$\Delta(\sigma_1-\sigma_2)_{model}$ (syn $-$ pre)')
    ax1.set_title('Change in stress vs. change in delay time\n(per-voxel, ray-traced)')

    ### Panel 2
    ax2.scatter(d_phi_obs, d_phi_cal, s=14, c='k', alpha=0.5, edgecolors='none')
    lims = [-90, 90]
    ax2.plot(lims, lims, '--', color='0.6', zorder=4, label='1:1')
    x_fit2 = np.array(lims)
    ax2.plot(x_fit2, linear(x_fit2, *popt2), '--r', zorder=5, label='fit')
    ax2.text(0.05, 0.92, f'N={n_vox}\nslope: {popt2[0]:.3f}\nRMS: {rms2:.1f}$\\degree$',
             transform=ax2.transAxes, va='top', fontsize=8)
    ax2.axhline(0, color='0.6', lw=0.8, ls=':')
    ax2.axvline(0, color='0.6', lw=0.8, ls=':')
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel(r'$\Delta \Phi_{obs}$ (syn $-$ pre) [°]')
    ax2.set_ylabel(r'$\Delta \Phi_{cal}$ (syn $-$ pre) [°]')
    ax2.set_title('Change in fast direction:\nobserved vs. modeled (per-voxel)')
    ax2.legend(fontsize=7, loc='lower right')

    fig.suptitle(r'def_pre_1 $\rightarrow$ def_syn_6 (voxel-resolved, ray-traced observations)', y=1.03)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name_pdf = output_dir + 'pre_1_vs_syn_6_change_voxel_hemmett.pdf'
    fig.savefig(name_pdf, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nSaved {name_pdf}")


if __name__ == '__main__':
    main()
