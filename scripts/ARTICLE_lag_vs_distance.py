#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:01:25 2018

@author: baillard
"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
from obspy import UTCDateTime
import glob
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib as mpl

import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol 
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator


### Parameters


station_list=['AXCC1','AXEC1','AXEC2','AXEC3','AXAS2','AXAS1','AXID1']
#station_list=['AXCC1','AXEC1','AXEC2']
#station_list=['AXAS2']
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat'
suffix='.clean.cat.pickle'
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
x_start=UTCDateTime(2015,1,24,6) # Nooner and Chadwick 2016
x_end=UTCDateTime(2015,7,24,6)
x_end=UTCDateTime(2017,2,1,1)
sampling_rate=200
ncols=2
nrows=np.ceil(len(station_list)/ncols)
cbar_step=0.5
num_levels=15
#param='fast' # or lag
param='lag' # or lag

dic_lag={'mode':'ms','y_start':0,'y_end':150,'y_over':0.9,'y_width':10,
         'y_label':'$\delta t$ [ms]','cbar_step':0.5,'vmax':3.82,'vmin':1,'y_cycle':None}

dic_fast={'mode':'azimuth','y_start':0,'y_end':180,'y_over':0.9,'y_width':10,
         'y_label':'$\Phi$ [°]','cbar_step':0.5,'vmax':3.58,'vmin':1,'y_cycle':[0,180]}

dic_all={'lag':dic_lag,'fast':dic_fast}

dist_start=0
dist_end=5
dist_width=0.3
dist_over=0.95

#minor_locator_angle = AutoMinorLocator(5)
minor_locator_lag = AutoMinorLocator(5)
minor_locator_dist = AutoMinorLocator(10)

plt.ion()

### Create plot

plt.close('all')
(fig,ax_array)=ggmt.create_plotgrid(nrows,ncols,clean_ticks=True)
ax_array=ax_array.reshape(-1)


### START PROCESS

k_sta=-1
max_val=0


for station_name in station_list:
    k_sta+=1

    ### Define file
    
    file_in= os.path.join(cat_dir,'')+station_name+suffix

    ### Read file for each station
    
    Cat=swm.read_pickle(file_in)
    
    ### Get proper parameters and convert if necessary
    
    elems_dic=Cat.get_dic()
    y=elems_dic[param]
    dists=elems_dic['hyp_dist']
    
    mode=dic_all[param]['mode']
    if mode=='ms':
        y=y/sampling_rate*1000
    elif mode=='s':
        y=y/sampling_rate
    elif mode=='azimuth':
        y=y*180/np.pi
        y=swm.trigo2azimuth(y)
    elif mode=='trigo':
        y=y*180/np.pi


    ### Plot large
    
    y_over=dic_all[param]['y_over']
    y_start=dic_all[param]['y_start']
    y_end=dic_all[param]['y_end']
    y_width=dic_all[param]['y_width']
    vmin=dic_all[param]['vmin']
    vmax=dic_all[param]['vmax']
    y_cycle=dic_all[param]['y_cycle']
    (X,Y,Z,x_bins,x_diffs)=swm.movehisto2d_bin(dists,y,
                              x_over=dist_over,y_over=y_over,x_mode='window',
                              x_start=dist_start,x_end=dist_end,x_width=dist_width,
                              y_start=y_start,y_end=y_end,y_width=y_width,y_cycle=y_cycle,
                              flag_y_norm=False,flag_resample=True,flag_filter=True)
    
    (Xm,Ym)=swm.XY2XY_pcolormesh(X,Y)
    
    ax=ax_array[k_sta]
    Z_out=np.log10(Z)
    Z_out[Z_out<=1.0]=np.nan
    
    ### Get max
    bounds= np.linspace(vmin,vmax,num_levels)
    if np.nanmax(Z_out)>max_val:
        max_val=np.nanmax(Z_out)
    im=ax.contourf(X,Y,Z_out,levels=bounds,cmap=plt.cm.get_cmap('jet'),
                   vmin=vmin,vmax=vmax)
    for c in im.collections:
        c.set_rasterized(True)
        
    im=ax.contour(X,Y,Z_out,levels=bounds,colors='k',linewidths=0.5)
    for c in im.collections:
        c.set_rasterized(True)
#    im=ax.pcolormesh(X,Y,Z_out,cmap=plt.cm.get_cmap('jet'),rasterized=True,
#                   vmin=vmin,vmax=vmax)

 
#    ticks,_=cube.smart_arange(vmin,vmax,cbar_step)
#    plt.colorbar(h1,cax=cax,ticks=ticks,format=FormatStrFormatter(cbar_fmt))
    #plt.colorbar(im,vmin=1,vmax=3.8,cax=cax)

    ### Cosmetic
    
    t=ax.text(0.02, 0.98,'%s' %(station_name), transform=ax.transAxes,
              fontsize=10, va='top',ha='left',weight='bold')
    t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.xaxis.set_minor_locator(minor_locator_dist)
    ax.yaxis.set_minor_locator(minor_locator_lag)
 
    
    if k_sta % ncols == 0:
        ax.set_ylabel(dic_all[param]['y_label'])
        
    if (nrows==1) or (k_sta >= ncols*(nrows-1)):
        ax.set_xlabel('Distance [km]')
    
    if k_sta==len(station_list)-1:
        cax=ggmt.get_cax(ax,cax_x0=1.03,cax_y0=0,cax_width=0.05,cax_height=1)
        cmap=plt.cm.get_cmap('jet')
   
        (ticks,_ )= swm.smart_arange(vmin,vmax,cbar_step)
        ticks= np.arange(vmin,vmax,cbar_step)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                        norm=norm,
                                        ticks=ticks)
        
        cax.set_ylabel('Obs. log scale')
    
if len(ax_array)>len(station_list):
    ax_array[-1].remove()
    
    
plt.savefig('ARTICLE_%s_vs_distance.pdf'%(param),format='pdf',dpi=300,frameon=False,bbox_inches='tight')
plt.savefig('ARTICLE_%s_vs_distance.png'%(param),format='png',dpi=300,frameon=False,bbox_inches='tight')

#    plt.setp(ax_mesh_lagz.get_yticklabels(), visible=False)
#    plt.setp(ax_mesh_anglez.get_yticklabels(), visible=False)
#    
#    ax_mesh_lagz.set_ylabel('')
#    ax_mesh_anglez.set_ylabel('')
#    ax_mesh_anglez.set_xlabel('Hours')
#    ax_mesh_lagz.set_xlabel('Hours')
#    ax_mesh_lag.set_xlabel('Dates')
#    ax_mesh_angle.set_xlabel('Dates')
#    ax_mesh_lagz.minorticks_on()
#    ax_mesh_anglez.minorticks_on()
#    ax_mesh_lagz.yaxis.set_minor_locator(minor_locator_lag)
#    ax_mesh_anglez.yaxis.set_minor_locator(minor_locator_angle)
#    ax_mesh_anglez.tick_params(axis='y',which='minor',bottom='off')
#    ax_mesh_lagz.tick_params(axis='y',which='minor',bottom='off')
#    
#    ax_mesh_lag.minorticks_on()
#    ax_mesh_angle.minorticks_on()
#    ax_mesh_lag.yaxis.set_minor_locator(minor_locator_lag)
#    ax_mesh_angle.yaxis.set_minor_locator(minor_locator_angle)
#
#    ax_mesh_angle.tick_params(axis='x',which='minor',bottom='off')
#    ax_mesh_lag.tick_params(axis='x',which='minor',bottom='off')
#    
#    ax_mesh_lag.text(0.01, 0.99,'%s'%station_name, horizontalalignment='left',
#          verticalalignment='top', transform=ax_mesh_lag.transAxes,color='w',fontweight='bold')
#    
#    ax_mesh_angle.text(0.01, 0.99,'%s'%station_name, horizontalalignment='left',
#          verticalalignment='top', transform=ax_mesh_angle.transAxes,color='w',fontweight='bold')
#    
#
#    plt.savefig('ARTICLE_time_variations_samples_ms_%i_%s.pdf'%(time_width_unzoom,station_name),format='pdf',quality=300,frameon=False,bbox_inches='tight')
#    plt.savefig('ARTICLE_time_variations_samples_ms_%i_%s.png'%(time_width_unzoom,station_name),format='png',quality=300,frameon=False,bbox_inches='tight')
#    

plt.ion()
#NCat.plot_movehist2d_time('angle',level=2,y_width=np.pi/20,mode='imshow_window',x_width=10,x_over=0.95,norm_y=True,smooth=False)
#NCat.plot_movehist2d_time('lag',level=2,y_width=2,mode='imshow_window',x_width=0.1,x_over=0,norm_y=True,smooth=False)
#New_Cat.plot_movehist2d_time('angle',level=2,y_width=np.pi/20,mode='pcolor_sample',x_width=500,x_over=0.95,norm_y=True,smooth=False)
#



