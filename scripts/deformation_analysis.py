#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New version of ARTICLE_deformation.py (Christian Baillard, 2019), fixed to run in this repo:

  - broken imports (from general import GMT / import general.projection) pointed at our real
    flat modules.
  - hardcoded paths (disp_dir, output_dir, grid_dir - all under /home/baillard/Dropbox/...)
    replaced with paths into this repo: DMODELS output is in ../Axial_Deformation/ (the
    dike+spheroid displacement grids from axial_comb_V0.m/axial_dike_V0.m), output figures go
    to deformation_figures/.
  - matplotlib API fixes: plt.cm.get_cmap() -> plt.colormaps[...] (removed in matplotlib>=3.9),
    savefig(quality=100) -> dropped (removed kwarg).
  - caldera outline (ggmt.plot_lines default -> caldera_smooth.ll) dropped: no such file
    exists anywhere in this repo or in Axial_Deformation/, and unlike the fissure/flow
    overlays this call wasn't gated by a flag in the original script. Flagged to the user
    rather than fabricated.
  - fissure/flow overlays disabled (flag_fissure/flag_flow = False): those files
    (Axial_2015_Fissures_points.ll etc.) also don't exist in this repo.
  - dropped: a leftover throwaway test figure at the top of the original script, and a dead
    period-detection if/elif immediately overwritten by the next line - neither did anything.

Original docstring: Created on Wed May 29 13:57:41 2019, @author: baillard
    Script made to process and plot the deformation on Axial Seamount
"""

import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pickle
import glob
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import glob

import deformation_util as adutil
from obspy import UTCDateTime
import shearwavesplit as sws
import sws_methods as swm
import GMT as ggmt
import projection as gproj
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol

### Parameters

mpl.rcParams['pdf.fonttype']=42 ## Very important to edit the text in illustrator

HERE = os.path.dirname(os.path.abspath(__file__))

disp_dir = os.path.join(HERE, '..', 'Axial_Deformation') + os.sep
output_dir = os.path.join(HERE, 'deformation_figures') + os.sep

flag_fissure=False  # Axial_2015_Fissures_points.ll etc. don't exist in this repo
flag_flow=False     # Axial_2015_Flows_points.ll doesn't exist in this repo
fissure_files=[]
flow_files=[]

x_lim=[5,11]
y_lim=[1,8]
sigma1_max=4
ini_lon=-130.1
ini_lat=45.9
idz=0
plt.ioff()


fault_coord_dic={
        'dike_syn_1':{'center':[8.2 ,5.2],'angle':89,'length':[0,4]},
        'fault_pre_syn_2':{'center':[8.2 ,5.2],'angle':-62,'length':[0,1.5]},
        'fault_pre_syn_3':{'center':[8.75 ,3.79],'angle':118,'length':[0,1.5]},
        'dike_pre_1':{'center':[9.05 ,4.4],'angle':97,'length':[0,3.7]},
        'dike_pre_2':{'center':[9.05 ,4.4],'angle':-99,'length':[0,5]},
        'fault_pre_1':{'center':[7.12 ,4.88],'angle':-68,'length':[0,2]},
        'fault_pre_2':{'center':[6.92 ,2.6],'angle':112,'length':[0,2]},

        }

fault_plot_dic={
        'dike_syn_1':{'lw':2,'color':'k','ls':'-'},
        'fault_pre_syn_2':{'lw':2,'color':'k','ls':'--'},
        'fault_pre_syn_3':{'lw':2,'color':'k','ls':'--'},
        'dike_pre_1':{'lw':2,'color':'k','ls':'-'},
        'dike_pre_2':{'lw':2,'color':'k','ls':'-'},
        'fault_pre_1':{'lw':2,'color':'k','ls':'--'},
        'fault_pre_2':{'lw':2,'color':'k','ls':'-'},
        }

### Check output

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
### Make list of files

file_ins=glob.glob(disp_dir+'*.xyzuvw')
print(f"Found {len(file_ins)} displacement grid files in {disp_dir}")

### Process

(xc,yc)=adutil.get_fault_coord(8.2,5.2,1.5,-62,80,1,flag_plot=False)

### Define colormap

light_jet = ggmt.cmap_map(lambda x: x/2+0.5, plt.colormaps['jet'])


### Loop
k_file=0

for disp_file in file_ins:

    plt.close('all')
    k_file+=1

    ### Read file

    (X,Y,Z,Ux,Uy,Uz)=adutil.read_disp_file(disp_file,flag_plot=True)
    disp_filename=disp_file.split('/')[-1]
    plt.close('all')  # close the read_disp_file diagnostic figure too

    ### Get period (def_pre_N.xyzuvw / def_syn_N.xyzuvw -> 'pre' / 'syn')

    period=disp_filename[4:7]

    X_slice=X[:,:,idz]
    Y_slice=Y[:,:,idz]
    Ux_slice=Ux[:,:,idz]
    Uy_slice=Uy[:,:,idz]
    Uz_slice=Uz[:,:,idz]

    ### Compute Strain and Strain tensor

    x_sta=8
    y_sta=6
    mu_el=1
    poisson=0.25
    strain=adutil.get_strain_2d(x_sta,y_sta,X_slice,Y_slice,Ux_slice,Uy_slice)
    STRA=adutil.compute_strain_2d(X_slice,Y_slice,Ux_slice,Uy_slice)
    STRE=adutil.compute_stress_2d(X_slice,Y_slice,Ux_slice,Uy_slice,poisson,mu_el)
    SIGMA1=adutil.compute_sigma1_2d(X_slice,Y_slice,Ux_slice,Uy_slice,poisson,mu_el)
    EPSILON1=adutil.compute_epsilon1_2d(X_slice,Y_slice,Ux_slice,Uy_slice)

    ### Get Sigma at station


    axuz=adutil.plot_elev_stations(X_slice,Y_slice,2*Uz_slice,period=period)
    name_pdf=output_dir+disp_filename.split('.')[0]+'_uz.pdf'
    plt.savefig(name_pdf,format='pdf',dpi=300,bbox_inches='tight')


    ### Get displacement

    [ax_norm,ax_fast]=adutil.plot_sigma1_stations(X_slice,Y_slice,SIGMA1,period=period,
    lag_mode='ms',lag_lim=[0,80])
    ax_norm.set_xlabel(r'$\delta t$ [ms]')
    name_pdf=output_dir+disp_filename.split('.')[0]+'_sigma1.pdf'
    plt.savefig(name_pdf,format='pdf',dpi=300,bbox_inches='tight')

    ### Prepare

    U_sig=SIGMA1[:,:,0]
    V_sig=SIGMA1[:,:,1]
    U_eps=EPSILON1[:,:,0]
    V_eps=EPSILON1[:,:,1]

    ###

    SIGMA1_norm=np.sqrt(SIGMA1[:,:,0]**2+SIGMA1[:,:,1]**2)
    sigma1_max=np.median(SIGMA1_norm)+1*np.std(SIGMA1_norm)
    U_sig[SIGMA1_norm>=sigma1_max]=0
    V_sig[SIGMA1_norm>=sigma1_max]=0

    ### Reduce

    times=2
    X_quiv=adutil.reduce(X_slice,times=times)
    Y_quiv=adutil.reduce(Y_slice,times=times)
    U_sig_quiv=adutil.reduce(U_sig,times=times)
    V_sig_quiv=adutil.reduce(V_sig,times=times)
    U_eps_quiv=adutil.reduce(U_eps,times=times)
    V_eps_quiv=adutil.reduce(V_eps,times=times)
    Uh_slice=np.sqrt(Ux_slice**2+Uy_slice**2)

    ################
    ### Plot #######

    param='sig'
    U_quiv=eval('U_%s_quiv'%param)
    V_quiv=eval('V_%s_quiv'%param)

    fig,ax=plt.subplots()

    ### Plot mesh

    im=ax.pcolormesh(X_slice,Y_slice,Uh_slice,cmap=light_jet,vmax=None,rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    cax=ggmt.get_cax(ax)
    plt.colorbar(im,cax=cax)
    cax.set_ylabel('Uh [m]')


    ### Plot caldera - SKIPPED: no caldera outline file exists anywhere in this repo
    ### (ggmt.plot_lines's own default pointed at Baillard's caldera_smooth.ll, which
    ### doesn't exist here either; unlike fissures/flows this had no flag in the original).

    ### Plot stations

    ggmt.plot_stations(ax=ax,mfc='w',ms=8,mec='k',alpha=1,color='k',name=True)

    ### Plot flow

    if flag_flow:
        for flow_file in flow_files:
            ggmt.plot_lines(line_file=flow_file,ax=ax,
                            color='blue',lw=0.8,bg_color='w',bg_lw=3)
    ### Plot fissure

    if flag_fissure:
        for fissure_file in fissure_files:
            ggmt.plot_lines(line_file=fissure_file,ax=ax,
                            color='yellow',lw=1.5,bg_color='k',bg_lw=2.5)

    ### Plot faults


    for k_fault,fault in enumerate(fault_coord_dic.keys()):
        if period not in fault:
            continue
        center=fault_coord_dic[fault]['center']
        angle_deg=fault_coord_dic[fault]['angle']
        len_prof=fault_coord_dic[fault]['length']
        color=fault_plot_dic[fault]['color']
        lw=fault_plot_dic[fault]['lw']
        ls=fault_plot_dic[fault]['ls']
        ggmt.plot_box(center,angle_deg,len_prof,width_prof=None,ax=ax,flag_label=False,
                 key=str(k_fault),flag_plot=True,fontsize=12,color=color,ls=ls,lw=lw,flag_cross=False)


    ### Plot arrows

    ax.quiver(X_quiv,Y_quiv,U_quiv,V_quiv,scale=10,width=0.003,pivot='mid',headaxislength=0,
              headlength=0,zorder=20,lw=0,rasterized=True)


    ### Cosmetic

    ax.set_title('Max. compressive stress axis')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')

    name_pdf=output_dir+disp_filename.split('.')[0]+'_map.pdf'
    plt.savefig(name_pdf,format='pdf',dpi=300,bbox_inches='tight')

    print(f"[{k_file}/{len(file_ins)}] {disp_filename} (period={period}) -> 3 PDFs saved")

print(f"\nDone. Figures in {output_dir}")
