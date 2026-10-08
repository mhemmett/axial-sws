#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank every Baillard DMODELS scenario (Axial_Deformation/def_{pre,syn}_N.xyzuvw) against the
Grade 3 splitting observations (get_station_obs_grade3), using Baillard's own per-grid metrics
from deformation_util.plot_sigma1_stations:

    rms_norm : RMS of the linear fit (sigma1-sigma2)_model ~ a * dt_obs[ms] + b   (6 stations)
    rms_fast : RMS of Phi_obs - Phi_cal, Phi_cal wrapped to within +-90 deg of Phi_obs

and, for every pre_i -> syn_j pair, the change metrics of deformation_change_pre1_syn6_*.py:

    rms_dphi_1to1 : RMS of dPhi_cal - dPhi_obs (residual from the 1:1 line, 180-wrapped)
    rms_dstress   : RMS of the linear fit d(sigma1-sigma2)_model ~ a * d(dt)_obs[ms] + b

Each fast-direction metric is computed twice: with the circular-mean observed fast direction
('fast', primary) and with the legacy linear median ('fast_median', what the _hemmett
scripts used), so the effect of the median's 0/180 wrap bias on the scenario ranking is
visible. Reference: RMS of a uniformly random axial misfit on [-90, 90] is 90/sqrt(3) = 52 deg.

Outputs (deformation_figures/):
    scenario_ranking_grade3_grids.csv
    scenario_ranking_grade3_pairs.csv
"""

import os
import glob
import re
import numpy as np
import pandas as pd

import deformation_util as adutil
import GMT as ggmt
import projection as gproj

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
    return (a1 - a0 + 90) % 180 - 90


def get_model_cal(disp_file):
    """Same station extraction as plot_sigma1_stations / deformation_change_pre1_syn6_*.py."""
    (X,Y,Z,Ux,Uy,Uz) = adutil.read_disp_file(disp_file, flag_plot=False)
    X_slice=X[:,:,idz]; Y_slice=Y[:,:,idz]
    SIGMA1 = adutil.compute_sigma1_2d(X_slice,Y_slice,Ux[:,:,idz],Uy[:,:,idz],poisson,mu_el)

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
        dic_sta_cal[station] = {'norm': np.sqrt(np.sum(vector**2)),
                                'fast': gproj.trigo2az(phi_cal)}
    return dic_sta_cal


def linfit_rms(x, y):
    a, b = np.polyfit(x, y, 1)
    return a, float(np.sqrt(np.mean((a*x + b - y)**2)))


def grid_label(path):
    m = re.match(r'def_(pre|syn)_(\d+)\.xyzuvw', os.path.basename(path))
    return m.group(1), int(m.group(2))


obs = {p: adutil.get_station_obs_grade3(period=p) for p in ['pre','syn']}

files = sorted(glob.glob(disp_dir + 'def_*.xyzuvw'), key=lambda f: grid_label(f))
cal = {}
for f in files:
    per, k = grid_label(f)
    cal[(per, k)] = get_model_cal(f)
    print(f"loaded {os.path.basename(f)}")

### Per-grid absolute metrics

grid_rows = []
for (per, k), dic_cal in cal.items():
    o = obs[per]
    lag_ms = np.array([o[s]['lag']*1000 for s in STATION_LIST])
    norm = np.array([dic_cal[s]['norm'] for s in STATION_LIST])
    slope, rms_norm = linfit_rms(lag_ms, norm)
    row = dict(grid=f'{per}_{k}', period=per, slope_norm_per_ms=slope, rms_norm=rms_norm)
    for key in ['fast', 'fast_median']:
        res = np.array([wrap_angle_diff(dic_cal[s]['fast'], o[s][key]) for s in STATION_LIST])
        row[f'rms_{key}'] = float(np.sqrt(np.mean(res**2)))
    for s in STATION_LIST:
        row[f'res_{s}'] = wrap_angle_diff(dic_cal[s]['fast'], o[s]['fast'])
    grid_rows.append(row)
grids = pd.DataFrame(grid_rows).sort_values(['period','rms_fast'])

### Pre -> syn change metrics

pair_rows = []
pre_keys = sorted(k for (p, k) in cal if p == 'pre')
syn_keys = sorted(k for (p, k) in cal if p == 'syn')
d_dt_ms = np.array([(obs['syn'][s]['lag'] - obs['pre'][s]['lag'])*1000 for s in STATION_LIST])
for i in pre_keys:
    for j in syn_keys:
        cp, cs = cal[('pre', i)], cal[('syn', j)]
        d_stress = np.array([cs[s]['norm'] - cp[s]['norm'] for s in STATION_LIST])
        d_phi_cal = np.array([wrap_angle_diff(cs[s]['fast'], cp[s]['fast']) for s in STATION_LIST])
        slope, rms_dstress = linfit_rms(d_dt_ms, d_stress)
        row = dict(pair=f'pre_{i}->syn_{j}', slope_dstress_per_ms=slope, rms_dstress=rms_dstress)
        for key in ['fast', 'fast_median']:
            d_phi_obs = np.array([wrap_angle_diff(obs['syn'][s][key], obs['pre'][s][key])
                                  for s in STATION_LIST])
            res = wrap_angle_diff(d_phi_cal, d_phi_obs)
            row[f'rms_dphi_1to1_{key}'] = float(np.sqrt(np.mean(res**2)))
        pair_rows.append(row)
pairs = pd.DataFrame(pair_rows).sort_values('rms_dphi_1to1_fast')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
grids.to_csv(output_dir + 'scenario_ranking_grade3_grids.csv', index=False)
pairs.to_csv(output_dir + 'scenario_ranking_grade3_pairs.csv', index=False)

pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda v: f'{v:.3f}')
print('\nObserved Grade 3 fast directions (circular mean | median) and dt:')
for p in ['pre','syn']:
    for s in STATION_LIST:
        v = obs[p][s]
        print(f"  {p} {s}: N={v['n']:5d}  {v['fast']:6.1f} | {v['fast_median']:6.1f} deg  "
              f"dt={v['lag']*1000:.0f} ms")
print('\nPer-grid metrics (sorted by rms_fast within period):')
print(grids.to_string(index=False))
print('\nPre->syn pairs (sorted by rms_dphi_1to1_fast):')
print(pairs.to_string(index=False))
