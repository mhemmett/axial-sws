#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:50:08 2019

@author: baillard

We use the following nomenclatures for the time periods

_PRE: before the eruption
_SYN: during the eruption
_POS: after the end of eruption
_AFT: after the start of eruption (_SYN+_POS)

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

### Parameters

station_name='AXEC2'
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat/'
pickle_file='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat/AXEC2.cat.pickle'
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
ini_lon=-130.1
ini_lat=45.9
dir_mesh='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/STAT_MESHES/'

### Mesh parameters

step=0.05 # in km # distance seperation of the mesh
x_start,x_end=4,12 # limits of the mesh
y_start,y_end=0,12 
z_start,z_end=0,2
x_step,y_step,z_step=step,step,step
dis_lim=0.3 # distance limit to compute the median,standard deviation or whatever
num_lim=100 # number of elements to take around the bin

### Check path

dir_mesh=os.path.join(dir_mesh,'')
if not os.path.exists(dir_mesh):
    os.mkdir(dir_mesh)


### Create full path

file_in= os.path.join(cat_dir,'')+station_name+'.cat.pickle'

### Read pickle into catalog

Cat=swm.read_pickle(file_in)

### Get arrays of data

elems_dic=Cat.get_dic(minlambda_select='min')

### Select arrays based on time periods

data=np.column_stack((elems_dic['x'],
                      elems_dic['y'],
                      elems_dic['z'],
                      elems_dic['lag'],
                      elems_dic['fast']))

data_pre=data[elems_dic['s_time']<=starteruption_time,:]
data_syn=data[(elems_dic['s_time']>starteruption_time) & (elems_dic['s_time']<=enderuption_time),:]
data_pos=data[elems_dic['s_time']>enderuption_time,:]
data_aft=data[elems_dic['s_time']>starteruption_time,:]

### Compute median/mean/std for each of these periods

prefix=station_name

### LOOP over periods
for suffix in ['PRE','SYN','POS','AFT']:
    data_period=eval('data_'+suffix.lower())
    x=data_period[:,0]
    y=data_period[:,1]
    z=data_period[:,2]
    lag=data_period[:,3]
    fast=data_period[:,4]
    
    ### LOOP over data type
    for parameter in  ['LAG','FAST']:
        d=eval(parameter.lower())
        ### Compute Grid
        
        mesh_dic=swm.xyzd2mesh(x,y,z,d,
              x_start=x_start,x_end=x_end,
              y_start=y_start,y_end=y_end,
              z_start=z_start,z_end=z_start,
              x_step=x_step,y_step=y_step,z_step=z_step,
              dis_lim=dis_lim,num_lim=num_lim,verbose=True)
        
        
        param_dic={'x_start':x_start,'x_end':x_end,
                   'y_start':y_start,'y_end':y_end,
                   'z_start':z_start,'z_end':z_end,
                   'x_step':x_step,'y_step':y_step,'z_step':z_step,
                   'dis_lim':dis_lim,'num_lim':num_lim}
        
                   
        
        ### Store to file
        
        ### Create file name
        
        mesh_file='%s_%s_%s_dx%.0f_R%.0f_N%i.pickle'\
        %(prefix,parameter,suffix,x_step*1000,dis_lim*1000,num_lim)
        
        full_mesh_file=dir_mesh+mesh_file
        
        pickle.dump(mesh_dic,open(full_mesh_file,'wb'))
        
         
            
    
    
#MEDIAN_mesh=mesh_dic['MEDIAN_mesh']
#COUNT_mesh=mesh_dic['COUNT_mesh']
#X_mesh=mesh_dic['X_mesh']
#Y_mesh=mesh_dic['Y_mesh']
#
#MA = ma.array(MEDIAN_mesh, mask = COUNT_mesh<=10)
#
##MA = ma.array(STD_mesh, mask = COUNT_mesh<=20)
#
#plt.close('all')
#
#fig,ax=plt.subplots()
##im=ax.pcolormesh(X_mesh[:,:,0],Y_mesh[:,:,0],D_mesh[:,:,1],antialiased=True,shading='flat',cmap=plt.cm.get_cmap('jet'),vmax=30)
#im=ax.pcolormesh(X_mesh[:,:,0],Y_mesh[:,:,0],MA[:,:,0],
#                 antialiased=True,shading='flat',cmap=plt.cm.get_cmap('jet'))
#cbar=plt.colorbar(im)
#cbar.ax.set_ylabel('Lags [samples]')
#ggmt.plot_lines(ax=ax,color='k',lw=2,ini_lon=-130.1,ini_lat=45.9)
#ggmt.plot_stations(ax=ax)
#
##plt.plot(x,y,'ok',ms=0.1)  
#ax.set_aspect('equal','box')  


#pickle.dump([xs,ys,zs,lags],open('AXEC2_post.xyzd','wb'))
#plt.close('all')
#          
#
#fig,ax=plt.subplots()
#
#ax.hist(lags)
#
#fig,ax=plt.subplots()
#
#ax.scatter(xs,ys,s=3,c=lags,cmap=plt.cm.get_cmap('jet'),vmax=30)
##ax.scatter(xs,ys,s=5,c=fasts,cmap=plt.cm.get_cmap('hsv'))
#ax.set_aspect('equal','datalim')
#
#plt.colorbar()
##
##plot_cross(x,y,z,center,angle_deg,len_prof,width_prof,s=z**2,c=y,ax=None,
##           cmap=plt.cm.get_cmap('jet'))
##    


