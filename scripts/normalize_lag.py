#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:50:08 2019

@author: baillard
"""

import pickle
import sws_methods as swm
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import numpy as np

pickle_file='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat/AXEC1.cat.pickle'
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)

plt.close('all')

### Read pickle

Cat=swm.read_pickle(pickle_file)

lags=[]
delays=[]
dists=[]

for obs in Cat.obs:
    if obs.s_time>=starteruption_time:
        continue
    dist=np.sqrt(obs.epi_dist**2+obs.event_depth**2)
    delay=obs.s_time-obs.p_time
    if delay<=0:
        continue
    for lamb in obs.MinLambdas:
        lag=lamb.lag
        delays.append(delay)
        lags.append(lag)
        dists.append(dist)
    
plt.close('all')
fig,ax=plt.subplots()    
lags=np.array(lags)
delays=np.array(delays)
dists=np.array(dists)
ax.plot(dists,lags,'+r')


(X,Y,Z)=swm.movehisto2d_bin(dists,lags/dists,x_width=0.1,y_width=1,
                x_start=None,y_start=None,x_end=None,y_end=50,
                x_over=0.9,y_over=0.9,
                norm_y=False,smooth=False,gaussian_x_per=0.5,gaussian_y_per=0.5)
    
    
fig,ax=plt.subplots()      
ax.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'),rasterized=True)

(X,Y,Z)=swm.movehisto2d_bin(dists,lags,x_width=0.1,y_width=1,
                x_start=None,y_start=None,x_end=None,y_end=None,
                x_over=0.9,y_over=0.9,
                norm_y=False,smooth=False,gaussian_x_per=0.5,gaussian_y_per=0.5)
    

fig,ax=plt.subplots()      
ax.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'),rasterized=True)




fig,ax=plt.subplots()     
ax.hist(lags,100)
fig,ax=plt.subplots()   
ax.hist(lags/dists,300)
