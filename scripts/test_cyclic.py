#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:22:32 2019

@author: baillard
"""

import sws_methods as swm
import general.plotwaveform as pltwf
import sys
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy
import os
import pickle
import time


###

y=pickle.load(open('test_cyclic.pickle','rb'))

plt.close('all')
plt.hist(y)


def extend_periodic_array(y,cycle_border,perc_ext=10):
    """
    Function made to extend periodic data so that we don't have border
    artifacts when are doing the histogram
    We replicate some of the samples to extend the range
    """
    

    #Numpy converting range of angles from (-Pi, Pi) to (0, 2*Pi)
    #(angles + 2 * np.pi) % (2 * np.pi)
    T=np.diff(cycle_border)
    alpha=cycle_border[0]
    
    ### Clean values outside borders to avoid counting events twice
    
    y_in=((y-alpha) %T)+alpha
    
    ### Triplicate
    
    y_mid=y_in[(y_in>alpha) & (y_in<alpha+T)]
    
    y_tri=np.hstack((y_in-T,y_mid,y_in+T))
    
    y_fin=y_tri[(y_tri>alpha*(1+perc_ext/100)) & (y_tri<(alpha+T)*(1+perc_ext/100))]
    
    return y_fin

plt.close('all')
cycle_border=[-90,90]
y_fin=extend_periodic_array(y,cycle_border,perc_ext=70)
bins=np.linspace(-200,200,90)
fig,ax=plt.subplots()
ax.hist(y,bins)

ax.hist(y_fin,bins,color='r',alpha=0.5)

