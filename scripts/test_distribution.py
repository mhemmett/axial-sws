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

import scipy.optimize

### Parameters

#file_in='test_lags_2.pickle'
#
#
#### Load
#
#y=pickle.load(open(file_in,'rb'))
#
#y_bins=np.linspace(np.min(y),np.max(y),500)
#x_bins=np.linspace(0,20,40)
#[a,b]=np.histogram(y,bins=x_bins)
#
#RMS=[]
#for y_bin in y_bins:
#    rms=np.sqrt(np.mean((y-y_bin)**2))
#    RMS.append(rms)
#    
#plt.close('all')
#plt.hist(y,x_bins)
#
#
#x_data=x_bins[:-1]+(x_bins[1]-x_bins[0])/2
#y_data=a


def multi_norm(x, *params):
    norm=np.zeros_like(x)
    for kk in range(0,len(params),3):

        mean=params[kk]
        std=params[kk+1]
        scale=params[kk+2]
        
        #norm += scale*scipy.stats.cauchy.pdf(x, loc=mean ,scale=std)
        norm += scale*scipy.stats.norm.pdf(x, loc=mean ,scale=std)
        #norm += scale * np.exp( -((x - mean)/std)**2)
    return norm

def fit_multi_norm(x,y,max_fit=10,flag_plot=True,max_std=None):
    
    ### Find max to constrain first gaussian
    
    mean_0=x[np.argmax(y)]
    dx=x[1]-x[0]
    std_0=dx*2
    scale_0=np.sum(y)*dx

    x_min=np.min(x)
    x_max=np.max(x)
    
    if max_std is None:
        max_std=(x_max-x_min)/3
    
    
    ### Start process
    
    ini_param=[mean_0,std_0,scale_0]
#    ini_bound_l=[x_min,0,0]
#    ini_bound_r=[x_max,max_std,np.inf]
    ini_bound_l=[0,0,0]
    ini_bound_r=[30,max_std,np.inf]
    ini_bound=(ini_bound_l,ini_bound_r)

    keep_param=[]
    param=ini_param
    bound=ini_bound
    num_gauss=0

    for kk in range(max_fit):
        try:
            print(param,bound)
            fitted_params,pcov = scipy.optimize.curve_fit(multi_norm,x, y, p0=param,bounds=bound)
            num_gauss+=1
            keep_param=list(fitted_params)
            param=list(fitted_params)+[x[0]+(x[-1]-x[0])/2,1,1]
            ini_bound_l=ini_bound_l+[x_min,0,0]
            ini_bound_r=ini_bound_r+[x_max,max_std,np.inf]
            
            bound=(ini_bound_l,ini_bound_r)

        except Exception as e: 
  
            param=param+[x[0]+(x[-1]-x[0])/2,1,1]
            ini_bound_l=ini_bound_l+[x_min,0,0]
            ini_bound_r=ini_bound_r+[x_max,max_std,np.inf]
            bound=(ini_bound_l,ini_bound_r)
    
    ### Plot if asked
    
    if flag_plot:
        fig,ax=plt.subplots()
        xx=np.linspace(x[0],x[-1],200)
        ax.plot(x,y,'r')
        ax.plot(xx,multi_norm(xx,*keep_param),'k')
        
        for values in get_params(keep_param):
            ax.plot([values[0],values[0]],[0,multi_norm(values[0],*keep_param)],'--k')
            
            ax.plot([values[0]-values[1],values[0]+values[1]],
                    [0.7*multi_norm(values[0],*keep_param),0.7*multi_norm(values[0],*keep_param)],'--k')
            ax.set_xlim([x[0],x[-1]])
        ### Plot means
        
        
        ax.set_title('Number of gaussians: %i'%num_gauss)
        
    ### Return
    
    return (keep_param,num_gauss)
        
def get_params(param):
    """
    Rearrange parameters to get a list per gaussian
    """
    A=np.array(param).reshape(-1,3)
    
    list_param=[list(x) for x in A]

    return list_param # [mean,std,scale]...

def find_std(param,xmin,xmax):
    """
    Function made to find the standard deviation associated with 
    maximum gaussian
    """
    xx=np.linspace(xmin,xmax,100)
    ### Initialize
    stds=[]
    max_gauss=[]
    means=[]
    
    ### Go through gaussians
    
    for values in get_params(param):
        y=multi_norm(xx,*values)
        max_gauss.append(np.max(y))
        stds.append(values[1])
        means.append(values[0])
        
        
    ### Sort for highest gaussian to lowest
    
    max_gauss=np.array(max_gauss)
    stds=np.array(stds)
    means=np.array(means)
    
    sort_inds = max_gauss.argsort()
    stds_n=stds[sort_inds[::-1]] # from biggest to lowest
    means_n=means[sort_inds[::-1]] # from biggest to lowest
    max_gauss_n=max_gauss[sort_inds[::-1]] 
    
    return (stds_n,means_n,max_gauss_n)
    

def centered_histo(x,x_bins=None,x_width=None,x_start=None,x_end=None,x_over=0,ax=None,flag_plot=False):
    """
    Like classical numpy histogram but with bins centered instead of edges given
    """
    
    
    x_step=(1-x_over)*x_width
    (x_bins,x_step)=swm.smart_arange(x_start,x_end,x_step)
    x_width=x_step/(1-x_over)
    x_rights=x_bins+x_width/2
    x_lefts=x_bins-x_width/2

    ### Define meshes
    
    counts=np.zeros_like(x_bins)
        
    ##################
    ### Start Counting
    
    k_x=-1
    for x_left,x_right in zip(x_lefts,x_rights):
        k_x+=1
       
        counter=len(x[(x>=x_left) & (x<=x_right)]) # Count numbers of elements  
        counts[k_x]=counter # store

    ### Plot if asked
    if flag_plot:
        if ax is None:
            fig,ax=plt.subplots()
        ax.plot(x_bins,counts,'or',mfc='r',mec='k')
        
    ### Return
    
    return (x_bins,x_lefts,x_rights,counts)






