#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:33:21 2019

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
from scipy.interpolate import interp1d
#import test_distribution as td
import shearwavesplit as sw 


### Parameters

station_name='AXAS1'
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat/'
file_level='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/PRESSURE/2013-2015-LTplot.txt'
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
ini_lon=-130.1
ini_lat=45.9
mode='samples' # window
#mode='window' # window
x_over=0.7
x_width=86400/12
x_width=1000
### Create full path

file_in= os.path.join(cat_dir,'')+station_name+'.cat.pickle'

### Read pickle into catalog
Cat=swm.read_pickle(file_in)
Cat=Cat.select(lambda_lag=[0,45])


### Get arrays of data

elems_dic=Cat.get_dic(minlambda_select='min')
station_dic=swm.read_stationfile()

### Select arrays based on baz

data=np.column_stack((elems_dic['x'],
                      elems_dic['y'],
                      elems_dic['z'],
                      elems_dic['baz_trigo'],
                      elems_dic['s_time'],
                      elems_dic['lag'],
                      elems_dic['fast']))

x=elems_dic['s_time']

#y=elems_dic['lag']
#y_start=0
#y_end=30
#y_width=1
#
#fig,ax=plt.subplots()
#ax.plot(x,y,'ok',ms=0.1)
#
#fig,ax=plt.subplots()
#ax.hist(y,np.linspace(0,40,40))
#
#sys.exit()

y=elems_dic['fast']*180/np.pi
y_start=-90
y_end=90
y_width=5
#

fig,ax=plt.subplots()
ax.plot(x,y,'ok',ms=0.1)

fig,ax=plt.subplots()
ax.hist(y,np.arange(-89.5,98.5,2))

sys.exit()





### Initialize
y_stds=np.zeros_like(x_bins)
y_medians=np.zeros_like(x_bins)
y_means=np.zeros_like(x_bins)
y_maxs=np.zeros_like(x_bins)

y_stds_g=np.zeros_like(x_bins)
y_means_g=np.zeros_like(x_bins)


x=np.array(swm.obspytime2matplotlib(x))
x_lefts=np.array(swm.obspytime2matplotlib(x_lefts))
x_rights=np.array(swm.obspytime2matplotlib(x_rights))
x_bins=np.array(swm.obspytime2matplotlib(x_bins))

### Start bining



k_x=-1

flag_plot=False

for x_left,x_right in zip(x_lefts,x_rights):
    plt.close('all')
    print('processing %i%i'%(k_x,len(x_lefts)))
  
    k_x+=1
    y_select=y[(x>=x_left) & (x<x_right)] # select data 
    
    
    ### Reject outliers
    
    #y_select=swm.zscores(y_select,thres=1.5,mode='modified',flag_plot=False)
    
    ### Compute stat
    
    median_y=np.median(y_select)
    means_y=np.mean(y_select)
    std_y=np.std(y_select)
    
    y_medians[k_x]=median_y
    y_means[k_x]=means_y
    y_stds[k_x]=std_y
    
    ### Compute histogram
    
    (y_bins,counts)=td.centered_histo(y_select,x_width=y_width,x_start=y_start,x_end=y_end,ax=None,flag_plot=flag_plot)

    ### Interpolate
    
    f_interp = interp1d(y_bins,counts,kind='quadratic')
    y_interp=np.linspace(np.min(y_bins),np.max(y_bins),500)
    counts_interp=f_interp(y_interp)
    y_maxs[k_x]=y_interp[np.argmax(counts_interp)]
    
    if flag_plot:
        plt.plot(y_interp,counts_interp)
    
#    ### Fit to Gaussian
#    
#    (keep_param,num_gauss)=td.fit_multi_norm(y_interp,counts_interp,max_fit=3,flag_plot=flag_plot)
#    if flag_plot:
#        plt.pause(1)
#
#    if keep_param==[]:
#        continue
#
#    (stds_n,means_n,max_gauss_n)=td.find_std(keep_param,y_start,y_end)
#    
#    y_stds_g[k_x]=stds_n[0]
#    y_means_g[k_x]=means_n[0]


    
plt.close('all')    

#datenum_array=swm.obspytime2matplotlib(x)   
fig,ax=plt.subplots()
ax.plot(x,y,'ok',ms=0.1)
ax.errorbar(x_bins,y_medians, yerr=y_stds,fmt='o',zorder=-10,color='r',capsize=2,elinewidth=0.5,ms=3)
ax.errorbar(x_bins,y_means_g, yerr=y_stds_g,fmt='o',zorder=-10,color='blue',capsize=2,elinewidth=0.5,ms=3)

ax.plot(x_bins,y_medians,'-r',lw=2)
ax.plot(x_bins,y_means_g,'-',color='blue',lw=2)
ax.plot(x_bins,y_maxs,'-',color='green',lw=2)
#ax.plot(datenum_array,y_means,'or')
#ax.plot(datenum_array,y,'ok',ms=0.1)
ax.xaxis_date()
#fig.autofmt_xdate()

fig.autofmt_xdate()

fig,ax=plt.subplots()
ax.plot(x_bins,y_maxs,marker='o',ls='none',color='green')
#ax.plot(datenum_array,y_means,'or')
#ax.plot(datenum_array,y,'ok',ms=0.1)
ax.xaxis_date()
#(level_obstimes,levels)=read_levels(file_level)
#level_mtimes=swm.obspytime2matplotlib(level_obstimes)
#
#ax.plot(level_mtimes,levels,'or')