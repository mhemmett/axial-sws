#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:33:21 2019

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
from scipy.interpolate import interp1d
import test_distribution as td
import shearwavesplit as sw 


### Parameters

station_name='AXAS1'
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat/'
file_level='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/PRESSURE/2013-2015-LTplot.txt'
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
ini_lon=-130.1
ini_lat=45.9
mode='samples' # window
#mode='window' # window
x_over=0.7
x_width=86400/12
x_width=1000
### Create full path

file_in= os.path.join(cat_dir,'')+station_name+'.cat.pickle'

### Read pickle into catalog
Cat=swm.read_pickle(file_in)
Cat=Cat.select(lambda_lag=[2,60])


### Get arrays of data

elems_dic=Cat.get_dic(minlambda_select='min')
station_dic=swm.read_stationfile()

### Get period

domP_x=[x.dom_freq['x']*200 for x in Cat.obs]
domP_y=[x.dom_freq['y']*200 for x in Cat.obs]

domP=[max([a,b]) for a,b in zip(domP_x,domP_y)]

### Select arrays based on baz

data=np.column_stack((elems_dic['x'],
                      elems_dic['y'],
                      elems_dic['z'],
                      elems_dic['baz_trigo'],
                      elems_dic['s_time'],
                      elems_dic['lag'],
                      elems_dic['fast']))

x=elems_dic['s_time']

lags=elems_dic['lag']


#### Plot

fig,ax=plt.subplots()

ax.hist(lags,100)
ax.set_xlabel('Lags')

fig,ax=plt.subplots()

ax.plot(lags,domP,'+k')


[X,Y,Z]=swm.movehisto2d_bin(lags,domP,x_width=1,y_width=2,
                x_start=None,y_start=None,x_end=None,y_end=None,
                x_over=0,y_over=0,
                norm_y=False,smooth=False,gaussian_x_per=0,gaussian_y_per=0)

fig,ax=plt.subplots()
ax.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'))
ax.set_xlabel('Lags')
ax.set_ylabel('Max_Lag')
