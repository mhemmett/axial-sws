#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:57:41 2019

@author: baillard

Scipt made to process and plot the deformation on Axial Szamount

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

import ARTICLE_deformation_util as adutil
from obspy import UTCDateTime
import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
import general.projection as gproj
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol 

### Parameters

mpl.rcParams['pdf.fonttype']=42 ## Very important to edit the text in illustrator

fig,ax=plt.subplots()
ax.plot(0,0,'ok')
ax.set_xlabel('sdf')

disp_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/\
Axial_Deformation/DMODELS/A_Matlab_functions/Axial_Deformation/'
disp_filename='def_2_dike_1_sph.xyzuvw'
disp_filename='def_0_dike_1_sph.xyzuvw'
disp_filename='def_3_dike_1_sph_ntnt.xyzuvw'
disp_filename='def_pre_2.xyzuvw'
#disp_filename='def_3_dike_1_sph_tnt.xyzuvw'

output_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/FIGURE_deformation/'
grid_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/'
fissure_files=[grid_dir+'Axial_2015_Fissures_points.ll',grid_dir+'Axial_2011_Fissures_points.ll']
flow_files=[grid_dir+'Axial_2015_Flows_points.ll']

flag_fissure=True
flag_flow=True
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
    os.mkdir(output_dir)
### Make list of files

file_ins=glob.glob(disp_dir+'*.xyzuvw')
### Process


(xc,yc)=adutil.get_fault_coord(8.2,5.2,1.5,-62,80,1,flag_plot=False)

### Define colormap

light_jet = ggmt.cmap_map(lambda x: x/2+0.5, plt.cm.get_cmap('jet'))


### Loop
k_file=0

for disp_file in file_ins:

    plt.close('all')
    k_file+=1
    
#    if k_file>=2:
#        sys.exit()

    ### Read file
    
    (X,Y,Z,Ux,Uy,Uz)=adutil.read_disp_file(disp_file,flag_plot=True)
    disp_filename=disp_file.split('/')[-1]
    
    ### Get period
    
    if 'pre' in disp_filename:
        period='pre'
    elif 'syn' in disp_filename:
        period='post'
        
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
    plt.savefig(name_pdf,format='pdf',frameon=False,dpi=300,quality=100,bbox_inches='tight')
    
    
    ### Get displacement
    
    [ax_norm,ax_fast]=adutil.plot_sigma1_stations(X_slice,Y_slice,SIGMA1,period=period,
    lag_mode='ms',lag_lim=[0,80])
    ax_norm.set_xlabel('$\delta t$ [ms]')
    name_pdf=output_dir+disp_filename.split('.')[0]+'_sigma1.pdf'
    plt.savefig(name_pdf,format='pdf',frameon=False,dpi=300,quality=100,bbox_inches='tight')
    
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
    
    
    ### Plot caldera
    
    ggmt.plot_lines(ax=ax,color='k',lw=1,ini_lon=-130.1,ini_lat=45.9)  
      
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
    plt.savefig(name_pdf,format='pdf',dpi=300,quality=100,bbox_inches='tight')
    
    #fig,ax=plt.subplots()
    #im=ax.pcolormesh(X_slice,Y_slice,Uh_slice,cmap=plt.cm.get_cmap('jet'))
    #ax.set_aspect('equal')
    #ax.quiver(X_quiv,Y_quiv,U_eps_quiv,V_eps_quiv,scale=5,pivot='mid',headaxislength=0)
    #plt.colorbar(im)



