#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:39:54 2019

@author: baillard
"""

from obspy.io.nlloc.core import read_nlloc_hyp
import sws_methods as swm
import general.plotwaveform as pltwf
import sys
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy
import os

# %%
def movehisto2d_bin(x,y,x_width=None,y_width=None,
                    x_mode='window',
                x_start=None,y_start=None,x_end=None,y_end=None,
                x_over=0.9,y_over=0.9,
                flag_y_norm=False,
                flag_resample=False,dx_resample=None,dy_resample=None,
                flag_filter=False,x_filter_per=0.5,y_filter_per=0.5,filter_mode='nearest'):
    """
    2019-05-16
    Function made to bin but using a moving window in both directions, this ensure better
    consistency between neighbor bins. 
    Two options for defining intervals in the x direction are possible.
    x_mode=['window','sample']. In 'window' mode the x bins are defined every x_width 
    (classical moving window). In 'sample' mode, the x bins are defined evrey x_width samples
    X_bins are centered, i.e. for each bin we look in the interval [x-x_width/2,x+x_width/2]
    
    Inputs
    ------
        x,y: np.array: arrays containing the data to apply histogram on
        x_width,y_width: float: width of the bins in x units or in number of samples
        [x,y]_[start,end]: float: start and end for histogram edges
        [x,y]_over: float in [0,1]: overlap for windows [1 = full overlap]
        flag_y_norm: bool: Should the ys be normalized by the max
        flag_resample: bool: Apply resampling (using scipy.griddata)
        dx_resample: float: resample every dx_resample
        flag_filter: bool: Apply gaussian filtering (using scipy.ndimage.gaussian_filter)
        x_filter_per: float: percentage of the range to be used as window size for the gaussian filter
            the bigger, the smoother
        filter_mode: str or sequence: define how the gaussian shoud behave 'nearest' or 'wrap' for cyclic
            can be ['nearest','wrap'] for different behavior in the two axis
        
    Ouputs
    ------
         [X,Y,Z]: np.array: 2D arrays containing bining
    
    Comment
    ------
        If you want to pcolormesh use sws_methods.XY2XY_mesh to increase the shape of X and Y, so that bin
        are centered
    """

    #### Check

    x=np.asarray(x)
    y=np.asarray(y)
    
    y=y[np.argsort(x)]
    x=x[np.argsort(x)]
    
    
    x_start=np.min(x) if x_start is None else x_start
    y_start=np.min(y) if y_start is None else y_start
    x_end=np.max(x) if x_end is None else x_end
    y_end=np.max(y) if y_end is None else y_end
    
    dx_resample=(x_end-x_start)/1000 if dx_resample is None else dx_resample
    dy_resample=(y_end-y_start)/1000 if dy_resample is None else dy_resample
    
    if x_mode=='sample':
        x_width=int(len(x)/100) if x_width is None else int(x_width) # Every n samples
    elif x_mode=='window':
        x_width=(x_end-x_start)/20 if x_width is None else x_width # Window size
    y_width=(y_end-y_start)/20 if y_width is None else y_width
    
    ### Define binining for X
    
    if x_mode=='sample':
        x_step=int(round((1-x_over)*x_width))
        if x_step==0:
            x_step=1
        x_ind_bins=np.arange(0,len(x)-1,x_step)
        x_ind_lefts=x_ind_bins-int(x_width/2)
        x_ind_lefts[x_ind_lefts<=0]=0
        x_ind_rights=x_ind_bins+int(x_width/2)
        x_ind_rights[x_ind_rights>=len(x)-1]=len(x)-1
        x_lefts=x[x_ind_lefts]
        x_rights=x[x_ind_rights]
        x_bins=x[x_ind_bins]

    elif x_mode=='window':
        x_step=(1-x_over)*x_width
        (x_bins,x_step)=swm.smart_arange(x_start,x_end,x_step)
        x_width=x_step/(1-x_over)
        x_rights=x_bins+x_step/2
        x_lefts=x_rights-x_width 
        
    ### Define binining for Y (is window by default)
    
    y_step=(1-y_over)*y_width
    (y_bins,y_step)=swm.smart_arange(y_start,y_end,y_step)
    y_width=y_step/(1-y_over)
    y_rights=y_bins+y_step/2
    y_lefts=y_rights-y_width 

    ### Define meshes
    
    X,Y=np.meshgrid(x_bins,y_bins)
    Z=np.zeros_like(X)
        
    ##################
    ### Start Counting
    
    k_x=-1
    for x_left,x_right in zip(x_lefts,x_rights):
        k_x+=1
        k_y=-1
        y_select=y[(x>=x_left) & (x<x_right)] # select data along x
        
        for y_left,y_right in zip(y_lefts,y_rights):
            k_y+=1
            counter=len(y_select[(y_select>=y_left) & (y_select<=y_right)]) # Count numbers of elements  
            Z[k_y,k_x]=counter # store
        
        
    ###################
    ### Resample
    
    if flag_resample:
        (y_resample,dy_resample)=swm.smart_arange(y_start,y_end,dy_resample)
        (x_resample,dx_resample)=swm.smart_arange(x_start,x_end,dx_resample)
        #print(dy_resample,dx_resample)
        X_resample,Y_resample=np.meshgrid(x_resample,y_resample)
        Z_resample=scipy.interpolate.griddata((X.ravel(),Y.ravel()), Z.ravel(), (X_resample, Y_resample),
                                              fill_value=0)
        ### fill_value is important in order to avoid nans
        Z=Z_resample
        
        #### Filter
    
        if flag_filter:
            
            gaussian_x=x_filter_per*X_resample.shape[0]/100
            gaussian_y=y_filter_per*X_resample.shape[1]/100
            #print(gaussian_x,gaussian_y)
            Z_gaussian=scipy.ndimage.gaussian_filter(Z,[gaussian_y,gaussian_x],mode=filter_mode)
            
            Z=Z_gaussian
            
        X=X_resample
        Y=Y_resample
        
    #### Normalize by max
    
    if flag_y_norm:
        
        max_y=np.max(Z,axis=0)[None,:]
        max_y[max_y<=0]=1
        Z=Z/max_y

    ### Return
    
    return (X,Y,Z)

# %%

station_name='AXEC1'
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat/'
file_level='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/PRESSURE/2013-2015-LTplot.txt'


file_in= os.path.join(cat_dir,'')+station_name+'.clean.cat.pickle'

### Read pickle into catalog
Cat=swm.read_pickle(file_in)

### Get arrays of data

elems_dic=Cat.get_dic(minlambda_select='min')

### Select arrays based on baz

data=np.column_stack((elems_dic['x'],
                      elems_dic['y'],
                      elems_dic['z'],
                      elems_dic['baz_trigo'],
                      elems_dic['s_time'],
                      elems_dic['lag'],
                      elems_dic['fast']))

x=elems_dic['lag']
y=elems_dic['fast']*180/np.pi

# %%
plt.ion()
(X,Y,Zc)=swm.movehisto2d_bin(x,y,x_width=200,y_width=10,
                    x_mode='sample',
                x_start=0,y_start=-90,x_end=30,y_end=90,
                x_over=0.9,y_over=0.9,
                flag_y_norm=False,y_cycle=[-90,90],
                flag_resample=True,dx_resample=None,dy_resample=None,
                flag_filter=True,x_filter_per=0.5,y_filter_per=0.5,filter_mode='nearest')

(X,Y,Z)=swm.movehisto2d_bin(x,y,x_width=200,y_width=10,
                    x_mode='sample',
                x_start=0,y_start=-90,x_end=30,y_end=90,
                x_over=0.9,y_over=0.9,
                flag_y_norm=False,y_cycle=None,
                flag_resample=True,dx_resample=None,dy_resample=None,
                flag_filter=True,x_filter_per=0.5,y_filter_per=0.5,filter_mode='nearest')


plt.close('all')

fig,ax=plt.subplots()
plt.plot(x,y,'ok',ms=2,alpha=0.2)

fig,ax=plt.subplots()
im=ax.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'),rasterized=True)
ax.set_xlim([0,30])
plt.colorbar(im)

fig,ax=plt.subplots()
im=ax.pcolormesh(X,Y,Zc,cmap=plt.cm.get_cmap('jet'),rasterized=True)
ax.set_xlim([0,30])
plt.colorbar(im)