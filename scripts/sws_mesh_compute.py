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

station_names=['AXCC1','AXEC1','AXEC2','AXEC3','AXAS2','AXAS1','AXID1']
station_names=['AXEC1','AXEC2','AXEC3','AXAS2','AXAS1','AXID1']
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat/'
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
ini_lon=-130.1
ini_lat=45.9
dir_mesh='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/STAT_MESHES_ARTICLE/'
suffix_sws='.clean.cat.pickle'

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

for station_name in station_names:
    ### Create full path
    
    file_in= os.path.join(cat_dir,'')+station_name+suffix_sws
    
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
                  z_start=z_start,z_end=z_end,
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
        
         
            
    
    
