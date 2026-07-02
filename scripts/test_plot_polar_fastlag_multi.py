#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:47:50 2018

@author: baillard

Program made to plot the density polar plots and the rose plots 
showing lags and fast directions for each of the station and for different time
periods

"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
from obspy import UTCDateTime
import glob

import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol 

from general.util import figs2pdf


### Parameters

station='AXEC1'

stations=['AXCC1','AXEC1','AXEC2','AXEC3','AXAS2','AXAS1','AXID1']
stations=['AXAS2']

start_time=UTCDateTime(2015,1,1)
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
end_time=UTCDateTime(2018,1,1)
suffix='.clean.cat.pickle' # to choose the cleaned catalog (esp. for AXCC1 and AXID1)
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat'
plt.ion()

#### Define time intervals

time_inters=[
        [start_time,starteruption_time],
        [starteruption_time,enderuption_time],
        [enderuption_time,end_time]]

hypdist_inters=[[0,1],[0,2],[0,3]]

#### Create colors for all stations
rgb_list=ggmt.data2rgb(np.linspace(0,1,len(stations)),cmap=plt.cm.get_cmap('jet'))

######################
### Start Plotting ###

#(ax_main,ax_insets,theta_edges,r_edges,r_inner,r_width)=get_ax_polarinsets(ax_main=None,num_ax=6)

for k_sta,station_name in enumerate(stations):
    
    ### Read pickle into catalog
    
    file_in= os.path.join(cat_dir,'')+station_name+'.clean.cat.pickle'
    Cat=swm.read_pickle(file_in)
    plt.close('all')
    
    fig,ax=plt.subplots(1,2,figsize=(16, 8))
    ax_polar=Cat.plot_polar_fastlag_multi(num_ax=6,
                                           colorbar=False,y_width=8,x_over=0.9,y_over=0.9,x_end=30,x_start=0,
                                           lagticks=np.arange(0,30,10),
                                     flag_resample=True,flag_filter=True,title='dsf',label_mode='azimuth',ax=ax[0])
    ax_polar=Cat.plot_polar_fastlag_multi(num_ax=6,
                                           colorbar=False,y_width=8,x_over=0.9,y_over=0.9,x_end=30,x_start=0,
                                     flag_resample=True,flag_filter=True,title='dsf',label_mode='azimuth',ax=ax[1])
    
#            
#    for kk,ax_inset in enumerate(ax_insets):
#        ax_inset.get_yaxis().set_visible(True)
#        Cat.plot_polar_fastlag(ax=ax_inset,colorbar=False,
#                                      show_axis=False,x_width=2,y_width=8,
#                                      cmap=plt.cm.get_cmap('jet'),label_mode='trigo')

            