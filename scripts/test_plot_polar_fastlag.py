#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:10:09 2019

@author: baillard
"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import matplotlib
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
stations=['AXID1']

start_time=UTCDateTime(2015,1,1)
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
end_time=UTCDateTime(2018,1,1)
suffix='.clean.cat.pickle' # to choose the cleaned catalog (esp. for AXCC1 and AXID1)
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat'

time_inters=[
        [start_time,starteruption_time],
        [starteruption_time,enderuption_time],
        [enderuption_time,end_time]]

station_name=stations[0]
file_in= os.path.join(cat_dir,'')+station_name+'.clean.cat.pickle'
Cat=swm.read_pickle(file_in)
NCat=Cat.select(obs_stime=time_inters[2])
#NCat=Cat.select()
elems_dic=NCat.get_dic()

### Select arrays 

x=elems_dic['lag']
y=elems_dic['fast']*180/np.pi # in degrees
#(X,Y,Z)=swm.movehisto2d_bin(x,y,x_width=x_width,y_width=y_width,y_start=y_start,y_end=y_end,y_cycle=y_cycle)
plt.close('all')
ax_polar=NCat.plot_polar_fastlag(colorbar=False,y_width=8,x_over=0.95,y_over=0.95,x_end=30,x_start=0,
                                 flag_resample=True,flag_filter=True,title='dsf',label_mode='azimuth')

