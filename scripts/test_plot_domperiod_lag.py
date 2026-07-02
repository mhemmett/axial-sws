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
#import test_distribution as td
import shearwavesplit as sw 


### Parameters

station_name='AXEC3'
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat/'

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

file_in= os.path.join(cat_dir,'')+station_name+'.clean.cat.pickle'

### Read pickle into catalog
Cat=swm.read_pickle(file_in)

NCat=Cat.select(lambda_select=['rms','min'],obs_hyp_dist=[0,0.5])

sys.exit()
#Cat=Cat.select(lambda_select=['rms','min'])

### Get arrays of data

elems_dic=Cat.get_dic(minlambda_select='min')
station_dic=swm.read_stationfile()

### Select arrays based on baz

data=np.column_stack((elems_dic['x'],
                      elems_dic['y'],
                      elems_dic['z'],
                      elems_dic['baz_trigo'],
                      elems_dic['s_time'],
                      elems_dic['lag'],
                      elems_dic['fast']))

x=elems_dic['lag']
y=elems_dic['dom_period']
z=elems_dic['s_time']
z=np.array(swm.obspytime2matplotlib(z))

plt.close('all')

xbins=np.linspace(0,30,100)
plt.plot((y/2)*200,x,'ok',ms=3,alpha=0.05)
plt.plot(xbins,xbins,'--r')
plt.plot(xbins,2*xbins,'--g')
plt.xlim([0,30])
plt.ylim([0,30])

fig,ax=plt.subplots()
ax.plot(z,y,'ok',ms=3,alpha=0.05)
#ax.plot(datenum_array,y_means,'or')
#ax.plot(datenum_array,y,'ok',ms=0.1)
ax.xaxis_date()
#fig.autofmt_xdate()

fig.autofmt_xdate()

starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
x_start_zoom=UTCDateTime(2015,4,22,6) # Nooner and Chadwick 2016
x_end_zoom=UTCDateTime(2015,4,26,6)
x_start=UTCDateTime(2015,1,24,6) # Nooner and Chadwick 2016
x_end=UTCDateTime(2015,7,24,6)
lag_start=0
lag_end=30
fast_start=-90
fast_end=90


Cat.plot_movehisto2d_time('dom_period',minlambda_select='min', 
                              x_label='X',y_label='Y',title='',ax=None,
                              x_over=0.95,y_over=0.95,x_mode='window',
                              x_start=x_start,x_end=x_end,x_width=5*86400,
                              y_start=0,y_end=0.30,y_width=0.01,
                              flag_y_norm=True,flag_resample=True,flag_filter=True)
