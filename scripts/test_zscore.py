#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:35:51 2019

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
import warnings


def zscores(data,thres=1,mode='original'):
      
    if mode=='original':
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val==0:
            z_scores=data*0
            warnings.warn('sdf')
        else:
            z_scores = (data - mean_val) / std_val
        
    elif mode=='modified':
        median_val=np.median(data)
        MAD=np.median(np.abs(data - median_val))
        if MAD==0:
            z_scores=data*0
        else:
            z_scores =  0.6745*(data - median_val) / MAD
    
    else:
        raise ValueError('mode is either original or modified')
        
    return data[np.abs(z_scores) < thres]
        


### Plot


data=np.random.normal(10, 2, 100)
data=np.ones(100)

#add=np.random.normal(20, 2, 5000)
#data=np.hstack((data,add))


thres=1.5
data_or=zscores(data,thres=thres,mode='original')
data_mo=zscores(data,thres=thres,mode='modified')

bins=np.linspace(0,30,60)

plt.close('all')

fig,ax=plt.subplots()
ax.hist(data,bins=bins,color='r')
ax.hist(data_or,bins=bins,color='k',alpha=0.5)

fig,ax=plt.subplots()
ax.hist(data,bins=bins,color='r')
ax.hist(data_mo,bins=bins,color='k',alpha=0.5)