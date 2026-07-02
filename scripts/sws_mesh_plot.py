#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:16:38 2019

@author: baillard
"""


import pickle
import sws_methods as swm
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import numpy as np
import general.GMT as ggmt
import general.projection as gproj
import os
import sys
import numpy.ma as ma
import general.cube_V1 as cube
import copy
from scipy.ndimage.filters import gaussian_filter
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
import scipy
import glob


def create_plotgrid(nrows,ncols):
    fig=plt.figure(figsize=[9/np.sqrt(2),9])
    
    gs_main = GridSpec(nrows, ncols, figure=fig)
    gs_main.update(hspace=0.1,wspace=0.05)
    
    ax_list=[]
    for k_col in range(ncols):
        ax_col=[] 
        for k_row in range(nrows):
            ax = plt.Subplot(fig, gs_main[k_row,k_col]) 
            ax_col.append(ax)
            fig.add_subplot(ax)
            
        ax_list.append(ax_col)
            
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)     
    
    return (fig,ax_list)




### Parameters

file_ins=['/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/STAT_MESHES/AXEC2_FAST_PRE_dx50_R300_N100.pickle',
          '/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/STAT_MESHES/AXEC2_FAST_AFT_dx50_R300_N100.pickle']

stations=['AXAS1','AXAS2','AXEC1','AXEC2','AXEC3']
dir_mesh='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/STAT_MESHES/'
periods=['PRE','AFT']

dir_fig='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/FIG_MESHES/'

depths=['z=0.5','z=1','z=1.5']
axis_map=[4,11,1,10]
flag_save=True

### Select files

station_paths=[]
for station in stations:
    for parameter in ['LAG','FAST']:
        station_periods=[]
        for period in periods:
            prefix=station+'_'+parameter+'_'+period+'*.pickle'
            station_periods.extend(glob.glob(dir_mesh+prefix))
    
        station_paths.append(station_periods)
            
### Check 

dir_fig=os.path.join(dir_fig,'')
if not os.path.exists(dir_fig):
    os.mkdir(dir_fig)


### Plot dictionaries

dic_plot_lag_0={'vmin':0,'vmax':30,'num_levels':14,'cmap_str':'jet',
                      'gaussian_val':0.2,'type_plot':'contourf','c_center':None,
                      'cbar_label':'Lag [samples]','cbar_fmt':'%.0f','cbar_step':5}


dic_plot_fast_0={'vmin':-90,'vmax':90,'num_levels':15,'cmap_str':'hsv',
                      'gaussian_val':0.2,'type_plot':'contourf','c_center':0,
                      'cbar_label':'$\phi$ [°]','cbar_fmt':'%.0f','cbar_step':15}


dic_plots={'AXAS1':{'FAST':dic_plot_fast_0,'LAG':dic_plot_lag_0},
           'AXAS2':{'FAST':dic_plot_fast_0,'LAG':dic_plot_lag_0},
           'AXEC1':{'FAST':dic_plot_fast_0,'LAG':dic_plot_lag_0},
           'AXEC2':{'FAST':dic_plot_fast_0,'LAG':dic_plot_lag_0},
           'AXEC3':{'FAST':dic_plot_fast_0,'LAG':dic_plot_lag_0}}



for file_ins in station_paths:
    plt.close('all')

    ### Define subplots
    
    nrows=len(depths)
    ncols=len(file_ins)
    
    (fig,ax_list)=create_plotgrid(nrows,ncols)
    
    k_col=-1
    for file_in in file_ins:
        k_col+=1
        ax_col=ax_list[k_col]
        station,parameter,period=file_in.split('/')[-1].split('_')[0:3]
         
        dic_plot=dic_plots[station][parameter]
        ### Read pickle file
        
        mesh_dic=pickle.load(open(file_in,'rb'))
        X=mesh_dic['X_mesh']
        Y=mesh_dic['Y_mesh']
        Z=mesh_dic['Z_mesh']
        D=mesh_dic['MEDIAN_mesh']
        C=mesh_dic['COUNT_mesh']
        
        ### Convert radian to degress if FAST direction is wanted
        
        if parameter=='FAST':
            D=D*180/np.pi
            
        ### Retrieve parameters for plotting
        
        cmap=plt.cm.get_cmap(dic_plot['cmap_str'])
        vmin=dic_plot['vmin']
        vmax=dic_plot['vmax']
        c_center=dic_plot['c_center']
        type_plot=dic_plot['type_plot']
        cbar_label=dic_plot['cbar_label']
        cbar_fmt=dic_plot['cbar_fmt']
        gaussian_val=dic_plot['gaussian_val']
        num_levels=dic_plot['num_levels']
        cbar_step=dic_plot['cbar_step']
            
        ### LOOP on depth
        
        k_row=-1
        for depth in depths:
            k_row+=1
            ax=ax_col[k_row]
    
            ### Plot data
            (ax,h1)=cube.plot_contour(X,Y,Z,D,slice_param=depth,cmap=cmap,
             ax=ax,c_center=c_center,vmin=vmin,vmax=vmax,
             zoom_val=3,type_plot=type_plot,linewidths=None,gaussian_val=gaussian_val,num_levels=num_levels)
    
            ### Plot masks
            (ax,h2)=cube.plot_contour(X,Y,Z,C,slice_param=depth,type_plot='contour',
            zoom_val=2,gaussian_val=0.8,levels=[50],colors='k',ax=ax,linewidths=0.5)
            ax.clabel(h2,inline=1, fmt='%.0f',fontsize=8,)
            (ax,_)=cube.plot_contour(X,Y,Z,C,slice_param=depth,type_plot='contourf',levels=[-50,15],colors='w',ax=ax,linewidths=0)
    
            ### Cosmetic
            
            ggmt.plot_lines(ax=ax,lw=1)
            ggmt.plot_stations(ax=ax)
            
            ax.axis(axis_map)
            ax.text(0.05, 0.02,'%s km' %(depth), transform=ax.transAxes,
              fontsize=10, va='bottom',ha='left')
            
            if k_row==0:
                ax.set_title(parameter+' '+period)
                
            if k_row==nrows-1 and k_col==ncols-1:
                cax=ggmt.get_cax(ax,cax_x0=1.03,cax_y0=0,cax_width=0.05,cax_height=1)
                ticks,_=cube.smart_arange(vmin,vmax,cbar_step)
                plt.colorbar(h1,cax=cax,ticks=ticks,format=FormatStrFormatter(cbar_fmt))
                cax.set_ylabel(cbar_label)
            
    
        ### Save for each file
    
    fig.suptitle(station)

    if flag_save:
        fig_name='%s_%s.pdf'%(station,parameter)
        plt.savefig(dir_fig+fig_name,format='pdf',bbox_inches='tight',dpi=300)

