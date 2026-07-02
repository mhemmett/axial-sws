#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:54:13 2019

@author: baillard
"""


import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
from sws_methods import SNR_pick


plt.close('all')
#### Parameters

flag_plot=True
array_dir='test/AX*'


#### Get files and load xy_array

data_files=glob.glob(array_dir)
num=6
xy_array,sw1,sw2=pickle.load(open(data_files[num],'rb'))

x_array=xy_array[:,0]
y_array=xy_array[:,1]

### Compute SNR

N_left=30
SNR_pick(x_array,sw1,N_left,30,flag_plot=True,mode='mean')
SNR_pick(y_array,sw1,N_left,30,flag_plot=True,mode='mean')
