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
x_array=xy_array[:,0]
y_array=xy_array[:,1]


data=x_array[sw1:sw2]
data=y_array[sw1:sw2]
#xx=np.linspace(0,50,51)
#data=0.8*np.sin(2*np.pi/40*xx)+np.sin(2*np.pi/5*xx)


plt.close('all')

#ax.plot(y_array)
num_wind=2
sw.get_dominant_period(data,fs,method='welch',nfft=256,num_wind=num_wind,flag_plot=True)
sw.get_dominant_period(data,fs,method='classic',nfft=256,num_wind=num_wind,flag_plot=True)
