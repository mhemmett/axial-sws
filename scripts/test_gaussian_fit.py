#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:12:12 2019

@author: baillard
"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
from obspy import UTCDateTime
import glob
import matplotlib.gridspec as gridspec
import test_distribution as tdist
from scipy.interpolate import interp1d

import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol 
from matplotlib.text import Text
import scipy

### Parameters

plt.close('all')
file_in='xy.pickle'
A=pickle.load(open('xy.pickle','rb'))
mode='max'
x_bins=A[:,0]
counts=A[:,1]
flag_plot=True

### Find Local maxi

y=counts
x=x_bins

def multi_norm(x, *params):
    """
    Used in decompose gaussian to plot and fit a gaussian
    params should be [mean1,std1,scale1,mean2,std2,scale2...]
    """
    norm=np.zeros_like(x)
    for kk in range(0,len(params),3):

        mean=params[kk]
        std=params[kk+1]
        scale=params[kk+2]
        
        #norm += scale*scipy.stats.cauchy.pdf(x, loc=mean ,scale=std)
        norm += scale*scipy.stats.norm.pdf(x, loc=mean ,scale=std)
        #norm += scale * np.exp( -((x - mean)/std)**2)
    return norm

def find_extrema(y,mode='max',flag_plot=False):
    """
    Function made to locate local minima and maxima
    """
    if mode=='max':
        bool_exts=np.r_[True, y[1:] > y[:-1]] & np.r_[y[:-1] > y[1:], True]
    else:
        bool_exts=np.r_[True, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], True]
    
    inds=np.linspace(0,len(y)-1,len(y),dtype=int)

    ind_exts=inds[bool_exts]
    ind_exts=ind_exts[(ind_exts>0) & (ind_exts<len(y)-1)]
    y_exts=y[ind_exts]
    
    
    if flag_plot:
        fig,ax=plt.subplots()
        
        ax.plot(inds,y,'r')
        ax.plot(ind_exts,y_exts,'+k')
    
    return (y_exts,ind_exts)


### Plot

def decompose_gaussians(x,y,flag_sort=True,flag_plot=False,ax=None):
    """
    Function made to decompose a distribution (conatinung only 
    positive values, i.e. histogram) into multiple gaussians and 
    return the standad devation and mean of each gaussians.
    Gaussians are centered on each local maxima
    
    Input
    -----
        x,y: np.array: input data
        flaf_sort: boolean: if True will sort the output valus from max y to min y
        flag_plot: boolean: if True will plot the data
        
    Output:
    ------
        x_maxs: np.array: values of the max of the gaussian
        std_maxs: np.array: std the gaussian
    """
    
    ### Find local extrema
    
    (_,ind_maxs)=find_extrema(y,mode='max',flag_plot=False)
    (_,ind_mins)=find_extrema(y,mode='min',flag_plot=False)
    
    ### Pad ind_mins with 0 and len
    
    ind_mins=np.hstack((0,ind_mins,len(y)-1))
    
    if flag_plot:
        if ax is None:
            fig,ax=plt.subplots()
        ax.plot(x,y,'-k',lw=0.5)
        xx=np.linspace(x[0],x[-1],200)
        colors_curve=ggmt.data2rgb(list(range(len(ind_maxs))),cmap=plt.cm.get_cmap('rainbow'))
        
    x_maxs=[]
    std_maxs=[]
    y_maxs=[]
    
    ### For each ind_max find closest min to the left and to the rights
    
    for k_max,ind_max in enumerate(ind_maxs):
        diffs=ind_mins-ind_max
        ind_left=np.max(diffs[diffs<0])+ind_max
        ind_right=np.min(diffs[diffs>0])+ind_max
        
        ### Find closest ind
        
        diff_clos=np.min([np.abs(ind_max-ind_left),np.abs(ind_max-ind_right)])
        ind_left_c=ind_max-diff_clos
        ind_right_c=ind_max+diff_clos
        
        ### Select data
        
        x_sel=x[ind_left_c:ind_right_c]
        y_sel=y[ind_left_c:ind_right_c]
        
        ### fit
        
        x_max=x[ind_max]
        y_max=y[ind_max]
        param=[x_max,1,1]
        ini_bound_l=[x_max-np.abs(0.01*x_max),0,0]
        ini_bound_r=[x_max+np.abs(0.01*x_max),30,np.inf]
        bound=(ini_bound_l,ini_bound_r)
        
        try:
            fitted_params,pcov = scipy.optimize.curve_fit(multi_norm,x_sel, y_sel, p0=param,bounds=bound)
        except:
            continue
        keep_param=list(fitted_params)
        
        std_max=fitted_params[1]
        
        x_maxs.append(x_max)
        std_maxs.append(std_max)
        y_maxs.append(y_max)
        
        if flag_plot:
            ax.plot(x_sel,y_sel,'k',lw=3)
            ax.plot(x_sel,y_sel,'r',color=colors_curve[k_max],lw=1.5)
            ax.plot(xx,multi_norm(xx,*keep_param),'-k',lw=0.5)
            ax.plot(xx,multi_norm(xx,*keep_param),':k',color=colors_curve[k_max],lw=2)
            
            ax.text(x_max,y_max+np.max(y)*0.03,'$%.0f \pm %.0f \sigma$'%(x_max,std_max),ha='center',clip_on=True)
    
    ### Cosmetic 
        
    if flag_plot:
        ylim=ax.get_ylim()
        ax.set_ylim([ylim[0],ylim[1]*1.1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    ### Sort from biggets to lowest max instead of left to righ
    y_maxs=np.array(y_maxs)
    x_maxs=np.array(x_maxs)
    std_maxs=np.array(std_maxs)
    
    if flag_sort==True:
        ind_sorts=np.argsort(-y_maxs)
        y_maxs=y_maxs[ind_sorts]
        x_maxs=x_maxs[ind_sorts]
        std_maxs=std_maxs[ind_sorts]
            
    return (x_maxs,std_maxs)    


#decompose_gaussians(x,y,flag_sort=True,flag_plot=True)