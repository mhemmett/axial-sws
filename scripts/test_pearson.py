#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:55:58 2019

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from obspy.clients.fdsn import Client
from obspy import read_inventory
import sys
import mtspec
import scipy
from obspy import UTCDateTime
from matplotlib.gridspec import GridSpec
import general.plotwaveform as pltwf
import warnings
import pickle
import glob
from shearwavesplit import spectrum
import shearwavesplit as sw
from scipy.stats import pearsonr


#### Parameters

flag_plot=True
array_dir='test/AX*'
nfft=256
fs=200
method='welch'
method='mstpec'

#### Get files and load xy_array

data_files=glob.glob(array_dir)
num=35
xy_array,sw1,sw2=pickle.load(open(data_files[num],'rb'))
fs=200
delta_t=1/200
xy_array=xy_array[sw1:sw2,:]
x_array=xy_array[:,0]
y_array=xy_array[:,1]


plt.close('all')
fig,ax=plt.subplots()
ax.plot(x_array)
ax.plot(y_array,'--r')
x=x_array
y=y_array

def get_rms(x,y,mode='norm'):
    """
    Compute RMS of two signals to check similarity
    """


#y=3*x
rms=np.sqrt(np.mean((x-y)**2))

#normalize

cat_array=np.hstack((x,y))
rms_norm=rms/(np.max(cat_array)-np.min(cat_array))
print(rms)
print(rms_norm)

(a,b)=pearsonr(x_array,2*x_array)
print(a)