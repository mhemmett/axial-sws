#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:39:26 2018

@author: baillard

Script to plot events from rays0 on top of bathymetry

"""


from lotos.LOTOS_class import Catalog
from importlib import reload 
from general.GMT import plot_netcdf,xy2ll_axes,add_scale
import pickle
from obspy.io.nlloc.core import read_nlloc_hyp
import matplotlib.pyplot as plt
import general.GMT as ggmt
from lotos.model_3d.vgrid import read_raypaths,read
from general.GMT import get_profileparam,plot_box,plot_circle,get_cax,plot_stations,plot_lines,plot_cross_stations
import numpy as np
from general.projection import ll2xy,xy2ll
from matplotlib.text import Text
import sys
import copy
import general.util as gutil
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

plt.close('all')

### Parameters

flag_save=True
ini_lon=-130.1
ini_lat=45.9
netcdf_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/Axial_42m_bis.grd'
cmap_file='/home/baillard/Dropbox/_Moi/GMT/Palettes/Bathy_Blue_Orange.pickle'

#ray_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/DATA/CATALOG/rays0_3894.txt'

grid_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/'
ray_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_WW/data/rays1.dat'
ray_density='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_WW/data/ray_paths_1.dat'
model_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_WW/data/dv_v21.dat'
profile_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/AXIAL_profiles_ortho.txt'
amc_contour='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/AMC_contour.ll'
fissures=[grid_dir+'Axial_2015_Fissures_points.ll',grid_dir+'Axial_2011_Fissures_points.ll']
vents='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/Axial_vent_sites.llz'
flows=[grid_dir+'Axial_2015_Flows_points.ll']

lon_lim=[-130.13,-129.89]
lat_lim=[45.88,46.05]
cross_slices=[1,2]
model_box=[0,0,15,15] #[x0,y0,width,height]
x_ticks=[0,5,10,15]
y_ticks=[0,5,10,15]

lon_labels,lat_labels=xy2ll(x_ticks,y_ticks,ini_lon,ini_lat)

### Read

Ray=Catalog()
Ray.read(ray_file,'bin')
DV=read(model_file,'lotos')
DENS=read_raypaths(ray_density,DV.grid_spec)

### Read contour and convert

lonc,latc=np.loadtxt(amc_contour,unpack=True)
xc,yc=ll2xy(lonc,latc,ini_lon,ini_lat)
lon_vent,lat_vent=np.loadtxt(vents,unpack=True,usecols=[0,1])
x_vent,y_vent=ll2xy(lon_vent,lat_vent,ini_lon,ini_lat)


### Plot Map

cmap=pickle.load(open(cmap_file,'rb'))
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap', cmaplist, 50)

plt.close('all')
fig,ax=plt.subplots()

(_,im)=plot_netcdf(netcdf_file,ini_lon=ini_lon,ini_lat=ini_lat,lon_lim=lon_lim,lat_lim=lat_lim,cmap=cmap,shade=True,ax=ax)
(_,im)=plot_netcdf(netcdf_file,lon_lim=lon_lim,lat_lim=lat_lim,cmap=cmap,shade=True,ax=ax)

sys.exit()

cax=get_cax(ax,cax_x0=0.01,cax_y0=0.01,cax_width=0.05,cax_height=0.4)
plt.colorbar(im,cax=cax)
cax.tick_params(labelsize=8)
cax.set_ylabel('Depth [m]')


### Plot caldera

plot_lines(ax=ax,color='k',lw=1,ini_lon=-130.1,ini_lat=45.9)
    
### Plot events

    
xlim=ax.get_xlim()
ylim=ax.get_ylim()
Ray.read_line()
ax=Ray.plot_map(ax=ax,ms=1,mfc='k',mew=0,plot_stations=False,plot_lines=False)



#### Plot AMC contour

ax.plot(xc,yc,'-r',lw=1)
ax.set_xlim(xlim)
ax.set_ylim(ylim)


             
        
### Plot Fissures
for file_in in fissures:
    data_list=gutil.read_datafile(file_in)
    for single_array in data_list:
        x,y=ll2xy(single_array[:,0],single_array[:,1],ini_lon,ini_lat)
        ax.plot(x,y,'k',lw=2.5)
        ax.plot(x,y,'yellow',lw=1.5)

#### Plot Lava Flow 

for file_in in flows:
    data_list=gutil.read_datafile(file_in)
    for single_array in data_list:
        x,y=ll2xy(single_array[:,0],single_array[:,1],ini_lon,ini_lat)
        ax.plot(x,y,'blue',lw=0.8)

### Plot stations

plot_stations(ax=ax,marker='^',mfc='w',ms=8,mec='k',alpha=1)

### Plot vents

ax.plot(x_vent,y_vent,linestyle='none',marker='*',mfc='magenta',mec='k',mew=0.5,ms=8)


### Plot boxes
for cross_key in cross_slices:
    center,angle_deg,len_prof,width_prof=get_profileparam(profile_file,cross_key)
    plot_box(center,angle_deg,len_prof,width_prof,ax=ax,key=str(cross_key))
    

### Plot seismic lines
    
ggmt.plot_seismic_lines(ax=ax,flag_cross=True,color='k')

### Plot model Box

rect = patches.Rectangle(model_box[0:2],model_box[2],model_box[3],linewidth=1,edgecolor='k',facecolor='none')
ax.add_patch(rect)

### Cosmetic  

ax.set_xticks([0,5,10,15])
ax.set_yticks([0,5,10,15])
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')




### Define new_cmap

cmap_colors=[(1,1,1),(1,0.8,0.8),(1,0.5,0.5),(1,0.2,0.2),(0.7,0.2,0.2)]
user_cm = LinearSegmentedColormap.from_list('tata', cmap_colors, N=100)
#user_cm=plt.cm.get_cmap('jet')

### Plot cross_sections

v_values={1:[0,220],2:[0,800]}
inc=0.1
kk=-1
for cross_key in cross_slices:
    
    kk+=1

    ### Density plot
    center,angle_deg,len_prof,width_prof=get_profileparam(profile_file,cross_key)
    
    (ax_cross,h_im)=DENS.plot_contour(center=center,angle_deg=angle_deg,len_prof=len_prof,width_prof=[0.1,0.1],
                               vmin=0,vmax=150,cmap=user_cm,inc=inc,filled=True,linewidths=None,
                               gaussian_val=0.8,zoom_val=2,c_extend=False,extend='max',levels=list(np.arange(0,160,10)),alpha=1)
    
    ### Mask
    (ax_cross,_)=DENS.plot_contour(center=center,angle_deg=angle_deg,len_prof=len_prof,width_prof=[0.1,0.1],
                               inc=inc,colors='w',levels=[-10,1],cmap=None,ax=ax_cross,filled=True,linewidths=None,
                               gaussian_val=0.8,zoom_val=2)
    
    f=plt.gcf()
    f.set_size_inches([4.6,2.4])
    ax_cross.set_ylim((2.5,0))
    ax_cross.set_title('')
    
    

    
    ### Get z_lim

    Ray.plot_cross(center,angle_deg,len_prof,width_prof,map_plot=False,ax=ax_cross,mfc='k',mew=0,ms=1.2)
    plot_cross_stations(center,angle_deg,len_prof,width_prof,ax=ax_cross,
                       marker='^',mfc='w',mec='k',mew=1,alpha=1,ms=10,clip_on=False)
#,ms_eq=1,color='k',mew_eq=0
    cax = get_cax(ax_cross)
    plt.colorbar(h_im,cax=cax)
    cax.set_ylabel('Rays/bin')


if flag_save:    
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        name='ARTICLE_figure_1_'+str(fig_num)+'.pdf'
        plt.savefig(name,format='pdf',transparent=False,bbox_inches='tight')
        


xy2ll_axes(ax,ini_lon,ini_lat,lon_inc=0.05,lat_inc=0.05)
if flag_save:    
    plt.figure(1)
    name='ARTICLE_figure_lonlat.pdf'
    plt.savefig(name,format='pdf',transparent=False,bbox_inches='tight')
