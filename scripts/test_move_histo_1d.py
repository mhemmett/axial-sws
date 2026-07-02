#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:45:28 2019

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
import scipy.stats
import time
import copy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import scipy.optimize
import test_distribution as td

### Parameters

file_in='test_fasts.pickle'
x_over=0
x_width=86400/12
x_width=5


### Load

plt.close('all')
x=pickle.load(open(file_in,'rb'))
#
#period_x=np.hstack((x-180,x,x+180))
#
#x=copy.deepcopy(period_x)

def movehisto1d(x,x_width,x_over=0.5,x_start=None,x_end=None,norm=None,
                flag_plot=False):
    """
    Function made to make a histogram using overlapping bins,
    it ensures continuity and smoothness of the resulting plot
    
    Input
    -----
    
    norm: ['max','intergral','initial']
    
    """
    ### Define bining
    
    x_start=np.min(x)
    x_end=np.max(x)
    x_step=(1-x_over)*x_width
    x_lefts,_=swm.smart_arange(x_start,x_end-x_width,x_step)
    x_rights=x_lefts+x_width
    x_bins=x_lefts+(x_rights-x_lefts)/2
    dx=x_bins[1]-x_bins[0]
    print(x_rights[0]-x_lefts[0])
    ### Process
    counts=np.zeros_like(x_bins)
    k_x=-1
    for x_left,x_right in zip(x_lefts,x_rights):
        k_x+=1
        
        x_select=x[(x>=x_left) & (x<x_right)]
        counts[k_x]=len(x_select)
        
    ### Normalize
    
    if norm=='max':
        counts=counts/np.max(counts)
    elif norm=='integral':
        counts=counts/(np.sum(counts)*dx)
    elif norm=='initial':
        counts=counts/len(x)

    ### Plot if asked

    if flag_plot:
        fig,ax=plt.subplots()
        ax.bar(x_bins,counts,width=dx)    
        
        ### Classic histogram
        
        x_edge_c,dxc=swm.smart_arange(x_start,x_end,x_width)
        counts_c,_=np.histogram(x,bins=x_edge_c)
        x_bins_c=x_edge_c[:-1]+dxc/2
        ax.bar(x_bins_c,counts_c,width=dxc,facecolor='r',alpha=0.8)
        
    return (x_bins,counts)



def centered_histo(x,x_bins=None,x_width=None,x_start=None,x_end=None,ax=None,flag_plot=False):
    """
    Like classical numpy histogram but with bins centered instead of edges given
    """
    
    ### Initialize
    
    x_start=np.min(x) if x_start is None else x_start
    x_end=np.max(x) if x_end is None else x_end
    if x_width is None:
        x_width=(x_end-x_start)/20
        
    if x_bins!=None:
        x_bins=np.asarray(x_bins)
        dx=x_bins[1]-x_bins[0]
        x_edges=np.hstack((x_bins-dx/2,x_bins[-1]+dx/2))
    else:
        x_edges,dx=swm.smart_arange(x_start,x_end,x_width)
        x_bins=x_edges[:-1]+dx/2
        
    ### Process
    
    counts,_=np.histogram(x,bins=x_edges)

    
    ### Plot if asked
    if flag_plot:
        if ax is None:
            fig,ax=plt.subplots()
        ax.bar(x_bins,counts,width=dx,facecolor='r',alpha=0.8)
        
    ### Return
    
    return (x_bins,counts)

    
### Plot
    
plt.close('all')

(x_bins,counts)=centered_histo(x,x_width=None,x_start=-90,x_end=90,ax=None,flag_plot=True)

### Interpolate

f = interp1d(x_bins,counts,kind='quadratic')
x_new=np.linspace(np.min(x_bins),np.max(x_bins),500)
interp_counts=f(x_new)

plt.plot(x_new,interp_counts)

### Fit to Gaussian

(a,b)=td.fit_multi_norm(x_new,interp_counts,max_fit=5,flag_plot=True)
         
sys.exit()

fig,ax=plt.subplots()
ax.plot(x_bins,counts)


####

smooth=savgol_filter(counts,5,3,mode='wrap')
ax.plot(x_bins,smooth)
xx=np.linspace(-90,90,300)
yy=np.interp(xx, x_bins, smooth,period=180)
ax.plot(xx,yy)


(a,b)=td.fit_multi_norm(xx,yy,max_fit=10,flag_plot=True)
