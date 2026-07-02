#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:39:26 2018

@author: baillard

Script to plot events from rays0 on top of bathymetry

"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import os

import general.GMT as ggmt
import general.util as gutil
import general.projection as gproj
from obspy.io.nlloc.core import read_nlloc_hyp
from obspy import UTCDateTime
from obspy.core.event.catalog import Catalog
import sws_methods as swm



# %%
### Parameters

flag_save=False
flag_fissure=True
flag_flow=True
flag_amc=True
flag_cross=True
ini_lon=-130.1
ini_lat=45.9

lon_lim=[-130.13,-129.89]
lat_lim=[45.88,46.05]
lon_lim=[-130.07,-129.94]
lat_lim=[45.9,46.01]
x_ticks=[0,5,10,15]
y_ticks=[0,5,10,15]
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
cross_keys=[1]

netcdf_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/Axial_42m_bis.grd'
cmap_file='/home/baillard/Dropbox/_Moi/GMT/Palettes/Bathy_Blue_Orange.pickle'
nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/CATALOG/AXIAL.HYP.FINAL_3D_V2.nlloc'
grid_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/'
profile_file='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/DATA/ARTICLE_profiles.txt'
amc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/AMC_contour.ll'
fissure_files=[grid_dir+'Axial_2015_Fissures_points.ll',grid_dir+'Axial_2011_Fissures_points.ll']
vents='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/Axial_vent_sites.llz'
flow_files=[grid_dir+'Axial_2015_Flows_points.ll']

#%%
###############
### PREPARE ###
###############

### Define eq colors

rgb_list=ggmt.data2rgb(np.linspace(0,1,3),cmap=plt.cm.get_cmap('rainbow'))

### Load Catalog file from .pickle or from .nlloc

cat_pickle_name=nlloc_file.split('/')[-1].split('.nlloc')[0]+'.pickle'

if not os.path.exists(cat_pickle_name):
    Cat=read_nlloc_hyp(nlloc_file)
    pickle.dump(Cat,open(cat_pickle_name,'wb'))
else:
    Cat=pickle.load(open(cat_pickle_name,'rb'))

### Select only phase with more than 9
    
sel_events=[x for x in Cat.events if x.origins[0].quality.used_phase_count>9]
Cat_sel=Catalog(events=sel_events)

### Get x,y,z,otime in arrays

lon_eq=np.array([x.origins[0].longitude for x in Cat_sel])
lat_eq=np.array([x.origins[0].latitude for x in Cat_sel])
z_eq=np.array([x.origins[0].depth/1000 for x in Cat_sel])
time_eq=np.array([x.origins[0].time for x in Cat_sel])
x_eq,y_eq=gproj.ll2xy(lon_eq,lat_eq,ini_lon,ini_lat)

### Select arrays based on time periods

data=np.column_stack((x_eq,y_eq,z_eq,time_eq))
data_pre=data[time_eq<=starteruption_time,:]
data_syn=data[(time_eq>starteruption_time) & (time_eq<=enderuption_time),:]
data_pos=data[time_eq>enderuption_time,:]


lon_labels,lat_labels=gproj.xy2ll(x_ticks,y_ticks,ini_lon,ini_lat)

# %%
############
### PLOT ###
############

### Plot Map

cmap=pickle.load(open(cmap_file,'rb'))
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap', cmaplist, 50)

plt.close('all')

if flag_cross:
    fig,[ax,ax_cross]=plt.subplots(2,1,gridspec_kw={'height_ratios': [2, 1]},figsize=[4.2,8])
else:
    fig,ax=plt.subplots(figsize=[6,6])

### Plot bathy

(_,im)=ggmt.plot_netcdf(netcdf_file,ini_lon=ini_lon,ini_lat=ini_lat,lon_lim=lon_lim,lat_lim=lat_lim,cmap=cmap,shade=True,ax=ax)
#(_,im)=ggmt.plot_netcdf(netcdf_file,lon_lim=lon_lim,lat_lim=lat_lim,cmap=cmap,shade=True,ax=ax)

### Plot colorbar
cax=ggmt.get_cax(ax,cax_x0=0.01,cax_y0=0.01,cax_width=0.05,cax_height=0.4)
plt.colorbar(im,cax=cax)
cax.tick_params(labelsize=8)
cax.set_ylabel('Depth [m]')

### Plot caldera

ggmt.plot_lines(ax=ax,color='k',lw=1,ini_lon=-130.1,ini_lat=45.9)

### Plot amc

if flag_amc:
    ggmt.plot_lines(line_file=amc_file,ax=ax,
                        color='r',lw=1)
    
    
### Plot eqs

k_eq=-1
for data_sel in [data_pre,data_syn,data_pos]:
    k_eq+=1
    x_sel=data_sel[:,0]
    y_sel=data_sel[:,1]
    z_sel=data_sel[:,2]

    mfc=rgb_list[k_eq,:]
    ax.plot(x_sel,y_sel,marker='.',ls='None',mec='None',mfc=mfc,ms=2,alpha=1,rasterized=True)
    
### Plot stations

swm.plot_stations(ax=ax,mfc='w',ms=8,mec='k',alpha=1)

### Plot fissure

if flag_fissure:
    for fissure_file in fissure_files:
        ggmt.plot_lines(line_file=fissure_file,ax=ax,
                        color='yellow',lw=1.5,bg_color='k',bg_lw=2.5)

### Plot flow
        
if flag_flow:
    for flow_file in flow_files:
        ggmt.plot_lines(line_file=flow_file,ax=ax,
                        color='blue',lw=0.8,bg_color='w',bg_lw=1.6)
        
          
# %%     
### Plot cross-sections if asked

if flag_cross:
    
    
    ### Plot cross_sections

    inc=0.1
    k_cross=-1
    for cross_key in cross_keys:
        center,angle_deg,len_prof,width_prof=ggmt.get_profileparam(profile_file,cross_key)
        ggmt.plot_box(center,angle_deg,len_prof,width_prof,ax=ax,key=str(cross_key),fontsize=6)
        k_cross+=1
        k_eq=-1
        for data_sel in [data_pre,data_syn,data_pos]:
            k_eq+=1 
            mfc=rgb_list[k_eq,:]
            
            proj_data_vent,_=gproj.project(data_sel,center,angle_deg,len_prof,width_prof)
            ax_cross.plot(proj_data_vent[:,0],proj_data_vent[:,2],
                    clip_on=True,ls='none',marker='.',mfc=mfc,mec='none',mew=0,ms=3,rasterized=True)
            
        ### Plot AMC
        
        ggmt.plot_cross_surface(center,angle_deg,len_prof,width_prof,
                       surface_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/AMC_clipped_1525.llzd',
                       ax=ax_cross,
                   mfc='r',mec='r',markersize=3,alpha=1,color_col_num=None,linestyle='-',vmin=None,vmax=None,cmap=plt.cm.get_cmap('gray'),
                   color='r',lw=2,num_points=100)
        
        ### Plot stations
        
        ggmt.plot_cross_stations(center,angle_deg,len_prof,width_prof,ax=ax_cross,
                       marker='^',mfc='w',mec='k',mew=1,alpha=1,ms=10,clip_on=False)
        
        ### Cosmetic
        
        ax_cross.set_ylim([2.5,0])
        ax_cross.set_aspect('equal','box')
        ax_cross.set_xlabel('X [km]')
        ax_cross.set_ylabel('Z [km]')
      
#fig.set_size_inches([4.2,8])        
# %%

if flag_save:    
    name_png='ARTICLE_map_earthquakes.png'
    name_pdf='ARTICLE_map_earthquakes.pdf'
    plt.savefig(name_png,format='png',dpi=300,frameon=False)
    plt.savefig(name_pdf,format='pdf',dpi=300,frameon=False)
        



