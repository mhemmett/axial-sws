#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-to-syn eruption CHANGE comparison between def_pre_1.xyzuvw and def_syn_6.xyzuvw,
using this repo's own splitting-pipeline observations (get_station_obs_hemmett) - the
same data source used in deformation_analysis_hemmett.py's _sigma1_hemmett.pdf plots,
for the 6 production stations (AXID1 excluded).

For each station, computes:
    delta(sigma1-sigma2)_model = (syn_6 norm_cal) - (pre_1 norm_cal)
    delta(dt)_obs              = (syn dt_obs)      - (pre dt_obs)
    delta(Phi_cal)_model       = 180-wrapped angle change in modeled compressive axis
    delta(Phi_obs)_obs         = 180-wrapped angle change in observed fast direction

and makes two scatter plots:
    1) delta(dt)_obs (x) vs delta(sigma1-sigma2)_model (y)
    2) delta(Phi_obs) (x) vs delta(Phi_cal) (y)

Output: pre_1_vs_syn_6_change_hemmett.pdf in deformation_figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import deformation_util as adutil
import GMT as ggmt
import projection as gproj
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
disp_dir = os.path.join(HERE, '..', 'Axial_Deformation') + os.sep
output_dir = os.path.join(HERE, 'deformation_figures') + os.sep

STATION_LIST = ['AXCC1','AXEC1','AXEC2','AXEC3','AXAS1','AXAS2']

ini_lon=-130.1
ini_lat=45.9
idz=0
poisson=0.25
mu_el=1


def wrap_angle_diff(a1, a0):
    """180-deg-aware angle difference a1-a0, wrapped to [-90,90]."""
    d = (a1 - a0 + 90) % 180 - 90
    return d


def get_model_cal(disp_file):
    """Replicates the station-extraction logic inside plot_sigma1_stations,
    returning {station: {'norm':..., 'fast':...}} for the given displacement file."""
    (X,Y,Z,Ux,Uy,Uz) = adutil.read_disp_file(disp_file, flag_plot=False)
    X_slice=X[:,:,idz]; Y_slice=Y[:,:,idz]
    Ux_slice=Ux[:,:,idz]; Uy_slice=Uy[:,:,idz]

    SIGMA1 = adutil.compute_sigma1_2d(X_slice,Y_slice,Ux_slice,Uy_slice,poisson,mu_el)

    station_dic = ggmt.read_stationfile()
    dic_sta_cal = {}
    for station in STATION_LIST:
        lon,lat = station_dic[station]['lon'], station_dic[station]['lat']
        x_sta,y_sta = gproj.ll2xy(lon,lat,ini_lon,ini_lat)
        DIST=(X_slice-x_sta)**2+(Y_slice-y_sta)**2
        i_sta,j_sta=np.unravel_index(DIST.argmin(), DIST.shape)
        vector=SIGMA1[i_sta,j_sta]
        (rho, phi_cal)=gproj.cart2pol(vector[0], vector[1])
        phi_cal*=180/np.pi
        if phi_cal>=90:
            phi_cal-=180
        elif phi_cal<=-90:
            phi_cal+=180
        az_cal=gproj.trigo2az(phi_cal)
        norm=np.sqrt(np.sum(vector**2))
        dic_sta_cal[station] = {'norm': norm, 'fast': az_cal}
    return dic_sta_cal


print("Loading pre_1 model...")
cal_pre = get_model_cal(os.path.join(disp_dir, 'def_pre_1.xyzuvw'))
print("Loading syn_6 model...")
cal_syn = get_model_cal(os.path.join(disp_dir, 'def_syn_6.xyzuvw'))

print("Loading observed splitting-pipeline results (pre/syn)...")
obs_pre = adutil.get_station_obs_hemmett(period='pre')
obs_syn = adutil.get_station_obs_hemmett(period='syn')

rows = []
for sta in STATION_LIST:
    d_stress = cal_syn[sta]['norm'] - cal_pre[sta]['norm']
    d_dt = obs_syn[sta]['lag'] - obs_pre[sta]['lag']  # seconds
    d_phi_cal = wrap_angle_diff(cal_syn[sta]['fast'], cal_pre[sta]['fast'])
    d_phi_obs = wrap_angle_diff(obs_syn[sta]['fast'], obs_pre[sta]['fast'])
    rows.append(dict(station=sta, d_stress=d_stress, d_dt=d_dt,
                      d_phi_cal=d_phi_cal, d_phi_obs=d_phi_obs))
    print(f"  {sta}: d(sigma1-sigma2)={d_stress:+.3f}  d(dt)={d_dt*1000:+.1f}ms  "
          f"d(Phi_cal)={d_phi_cal:+.1f}deg  d(Phi_obs)={d_phi_obs:+.1f}deg")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8.5, 3.6])
plt.subplots_adjust(wspace=0.35)

### Panel 1: delta dt (obs) vs delta stress (model)

def linear(x, a, b):
    x = np.asarray(x)
    return a*x + b

d_dt_ms = np.array([r['d_dt']*1000 for r in rows])
d_stress = np.array([r['d_stress'] for r in rows])

for r in rows:
    ax1.plot(r['d_dt']*1000, r['d_stress'], 'ok')
    ax1.annotate(r['station'], (r['d_dt']*1000, r['d_stress']),
                 xytext=(4,4), textcoords='offset points', fontsize=8)

popt, pcov = curve_fit(linear, d_dt_ms, d_stress)
fit_model = linear(d_dt_ms, *popt)
rms = np.sqrt(np.mean((fit_model - d_stress)**2))

x_fit = np.array([d_dt_ms.min(), d_dt_ms.max()])
ax1.plot(x_fit, linear(x_fit, *popt), '--r', zorder=-2)
ax1.text(0.05, 0.92, f'slope: {popt[0]:.4f} /ms\nRMS: {rms:.3f}',
          transform=ax1.transAxes, va='top', fontsize=8)

ax1.axhline(0, color='0.6', lw=0.8, ls=':')
ax1.axvline(0, color='0.6', lw=0.8, ls=':')
ax1.set_xlabel(r'$\Delta \delta t_{obs}$ (syn $-$ pre) [ms]')
ax1.set_ylabel(r'$\Delta(\sigma_1-\sigma_2)_{model}$ (syn $-$ pre)')
ax1.set_title('Change in stress vs. change in delay time')

### Panel 2: delta Phi_obs vs delta Phi_cal
for r in rows:
    ax2.plot(r['d_phi_obs'], r['d_phi_cal'], 'ok')
    ax2.annotate(r['station'], (r['d_phi_obs'], r['d_phi_cal']),
                 xytext=(4,4), textcoords='offset points', fontsize=8)
lims = [-90, 90]
ax2.plot(lims, lims, '--r', zorder=-2)
ax2.axhline(0, color='0.6', lw=0.8, ls=':')
ax2.axvline(0, color='0.6', lw=0.8, ls=':')
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_aspect('equal', 'box')
ax2.set_xlabel(r'$\Delta \Phi_{obs}$ (syn $-$ pre) [°]')
ax2.set_ylabel(r'$\Delta \Phi_{cal}$ (syn $-$ pre) [°]')
ax2.set_title('Change in fast direction:\nobserved vs. modeled')

fig.suptitle(r'def_pre_1 $\rightarrow$ def_syn_6', y=1.04)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
name_pdf = output_dir + 'pre_1_vs_syn_6_change_hemmett.pdf'
fig.savefig(name_pdf, format='pdf', dpi=300, bbox_inches='tight')
print(f"\nSaved {name_pdf}")
