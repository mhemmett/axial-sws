#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:19:39 2018

@author: baillard
"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
import glob
import time
import copy
from obspy import UTCDateTime
from scipy.ndimage import zoom,gaussian_filter
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import datetime as dt

import general.util as gutil
import shearwavesplit as sws


def movehisto2d(x,y,x_width,y_width,
                x_start=None,y_start=None,x_end=None,y_end=None,
                x_over=0.7,y_over=0.7,
                norm_y=False,smooth=True,gaussian_x_per=0.5,gaussian_y_per=0.5,
                x_label='X',y_label='Y',text_list=''):
    """
    Function made to plot an histo2d but using a moving window in both directionsn, this ensure better
    consistency between neighbor bins.
    The histogram can also work when data is an obspy.UTCDateTime array, then the x_start and x_end must
    be given in UTCDateTime as well and the x_width should be given in days
    
    Inputs
    ------
        x,y: np.array: arrays containing the data to apply histogram on (x can be UTCDateTime)
        x_width,y_width: float: width of the bins (in days for UTCDateTime)
        [x,y]_[start,end]: float: start and end for histogram edges
        [x,y]_over: float in [0,1]: overlap for windows [1 = full overlap]
        norm_y: Boolean: True to normalize by the maximum in each column
        smooth: Boolean: True for smoothin (Gaussian Filter)
        gaussian_[x,y]_per: float in [0,100]: width percentage for smoothing (100= filter size equal to data range)
        [x,y]_label: str
        text_list: list,str: titles to be added to the right corner of the figure
        
    Ouputs
    ------
        ax_list: plt.axes: object associated to the 4 plots (top,text,mesh,right)
    
    Comments:
    ---------
        To be added to the general functions (plot module?)
    """
    
    ### Paramaters
    
#    x_start=None
#    x_end=None
#    x_width=30
#    y_width=1
#    x_over=0.95
#    y_start=0
#    y_end=20
#    y_over=0.95
#    norm_y=True
#    smooth=True
#    gaussian_x_per=0.5 # percent of in x direction
#    gaussian_y_per=0.5 # percent of in x direction
#    x_label='Days'
#    y_label='Lags [samples]'
#    text_list=''

            
    def timestamp2matplotlib(values):
        values = list(map(dt.datetime.fromtimestamp, values))
        values = list(mdates.date2num(values))
        return values
    
    x=np.asarray(x)
    y=np.asarray(y)
    
    x_pixels=1000 # Number of elements in Z for imshow
    y_pixels=1000
    
    ### Convert Obspy.UTCDatetime to timestamp
    
    time_flag=False
    if isinstance(x[0],UTCDateTime):
        time_flag=True
        
    ### Check
    
    gaussian_x=gaussian_x_per*x_pixels/100 # Convert percentage to number or pixels
    gaussian_y=gaussian_y_per*y_pixels/100
    
    x_start=np.min(x) if x_start is None else x_start
    y_start=np.min(y) if y_start is None else y_start
    x_end=np.max(x) if x_end is None else x_end
    y_end=np.max(y) if y_end is None else y_end
    
    if time_flag:
        x=np.array([value.timestamp for value in x])
        x_width=x_width*86400 ## width must be given in days for time plots
        
        if not all((isinstance(x_start,UTCDateTime),isinstance(x_end,UTCDateTime))):
            raise ValueError('x_start and x_end must be given in obspy.UTCDateTime')
    
        x_start=x_start.timestamp
        x_end=x_end.timestamp
    
    
    ### Define bin edges 
    
    x_step=(1-x_over)*x_width
    y_step=(1-y_over)*y_width
    
    y_lefts=np.arange(y_start,y_end,y_step)
    y_mesh=y_lefts+y_width-y_step/2
    y_lefts=y_lefts[y_mesh<=y_end]
    y_mesh=y_mesh[y_mesh<=y_end]
    
    x_lefts=np.arange(x_start,x_end,x_step)
    x_mesh=x_lefts+x_width-x_step/2
    x_lefts=x_lefts[x_mesh<=x_end]
    x_mesh=x_mesh[x_mesh<=x_end]
    
    ### Define meshes
    
    X,Y=np.meshgrid(x_mesh,y_mesh)
    Z=np.zeros_like(X)
        
    ### Start Counting
    
    k_x=-1
    for x_left in x_lefts:
        k_x+=1
        k_y=-1
        x_right=x_left+x_width
        y_select=y[(x>x_left) & (x<=x_right)] # select data 
        
        for y_left in y_lefts:
            k_y+=1
            y_right=y_left+y_width
            counter=len(y_select[(y_select>y_left) & (y_select<=y_right)]) # Count numbers of elements  
            Z[k_y,k_x]=counter # store
        
    #### Zoom and filter
       
    if smooth:
    
        x_factor=x_pixels/Z.shape[1]
        y_factor=y_pixels/Z.shape[0]
        
        Z_smooth=zoom(Z,[y_factor,x_factor],order=1) # row, column
        Z_smooth=gaussian_filter(Z_smooth,[gaussian_y,gaussian_x])
        
        ### Redefine x,y sizes
        
        x_mesh=np.linspace(x_mesh[0],x_mesh[-1],x_pixels)
        y_mesh=np.linspace(y_mesh[0],y_mesh[-1],y_pixels)
        
        Z=Z_smooth
        
    
    #### Normalize 
        
    hist_y=np.max(Z,axis=0)
    hist_x=np.max(Z,axis=1)
    
    if norm_y:
        
        max_y=np.max(Z,axis=0)[None,:]
        max_y[max_y<=0]=1
        Z=Z/max_y
    
    
    #####################
    #### Start plotting

    plt.figure()
      
    #### Grid spec
    
    bottom=0.15 if time_flag else 0.1
    
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[4, 1],
                           height_ratios=[1, 4],
                           bottom=bottom)
    
    gs.update(hspace=0.05,wspace=0.05)
    
    ax_mesh = plt.subplot(gs[2])
    ax_right = plt.subplot(gs[3],sharey=ax_mesh)
    ax_top = plt.subplot(gs[0],sharex=ax_mesh)
    ax_text = plt.subplot(gs[1])
    #ax_text.axis('off')
    
    ### Text
    
    sws.tolist(text_list)
    text_str=''
    for word in text_list:
        text_str+=word+'\n'
    ax_text.text(0.5,0.5,text_str,ha='left',va='center',weight='bold')
    ax_text.axis('off')
    
    ### Top
    
    if time_flag:
        ax_top.plot(np.array(timestamp2matplotlib(x_mesh)),hist_y,'k')
    else:
        ax_top.plot(x_mesh,hist_y,'k')
    ax_top.xaxis_date()
    ax_top.set_ylabel('Max. Obs.')    
    ax_top.xaxis.set_label_position('top') 
    ax_top.xaxis.tick_top()
    if time_flag:
        plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=-30 ,ha='right')
    
    ### Right
    
    ax_right.plot(hist_x,y_mesh,'k') 
    ax_right.set_xlabel('Max. Obs.')    
    ax_right.yaxis.set_label_position('right') 
    ax_right.yaxis.tick_right()
    
    ### Mesh
    
    ax_mesh.set_ylabel(y_label) 
    if not time_flag:
        ax_mesh.set_xlabel(x_label) 
    extent_x=[x_mesh[0],x_mesh[-1]]
    
    if time_flag:
        extent_x = timestamp2matplotlib(extent_x)
        ax_mesh.xaxis_date()
        plt.setp( ax_mesh.xaxis.get_majorticklabels(), rotation=30 ,ha='right')
    
    extent_y=[y_mesh[0],y_mesh[-1]]
    extent=extent_x+extent_y
    ax_mesh.imshow(Z,cmap=plt.get_cmap('jet'),extent=extent,origin='lowerleft')   
    ax_mesh.set_ylim(extent_y)
    ax_mesh.set_xlim(extent_x)
    ax_mesh.axis('auto')
    
    ###### Return
    
    return [ax_top,ax_text,ax_mesh,ax_right]