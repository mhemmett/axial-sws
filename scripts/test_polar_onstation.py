#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:03:16 2018

@author: baillard
"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
import glob
import time
from obspy import UTCDateTime

from scipy.ndimage import zoom,gaussian_filter
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import datetime as dt
import copy
import inspect
import datetime as dt
from matplotlib.ticker import FuncFormatter


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class,get_projection_names

import general.projection as gproj
import general.util as gutil
import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
    
import matplotlib as mpl

#plt.close('all')
#
##### Parameters
##
pickle_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_40_1_cat'
pickle_files=glob.glob(pickle_dir+'/AXAS1*pickle')
pickle_files.sort()
Cat=swm.read_pickle(pickle_files)

starteruption_time=UTCDateTime(2015,4,24,9)
enderuption_time=UTCDateTime(2015,5,19)
begin_time=UTCDateTime(2015,1,1,1)
end_time=UTCDateTime(2018,1,1,1)

lag_end=30
Cat_pre=Cat.select(lambda_select=['rec','max'],obs_contour_keys=None,lambda_lag=[0,lag_end],obs_stime=[begin_time,starteruption_time])
Cat_syn=Cat.select(lambda_select=['rec','max'],lambda_lag=[0,lag_end],obs_stime=[starteruption_time,enderuption_time])
Cat_post=Cat.select(lambda_select=['rec','max'],lambda_lag=[0,lag_end],obs_stime=[enderuption_time,end_time])

plt.close('all')
ax=Cat_pre.plot_events(cmap=plt.cm.get_cmap('jet'))
#ax=Cat.plot_events(cmap=plt.cm.get_cmap('jet'))
Cat.plot_contours(ax=ax,facecolor='none')
#Cat.plot_polar_fastlag(x_width=2,x_over=0.7,cmap=plt.cm.get_cmap('jet'))
Cat_pre.plot_polar_fastlag_multi(x_width=2,num_ax=8,x_end=lag_end,y_width=np.pi/40,x_over=0.7,cmap=plt.cm.get_cmap('jet'))
Cat_pre.plot_polar_fastlag(x_width=2,x_end=lag_end,y_width=np.pi/40,x_over=0.7,cmap=plt.cm.get_cmap('jet'))
Cat_pre.plot_polar_histofast()
Cat_syn.plot_polar_fastlag(x_width=2,x_end=30,y_width=np.pi/40,x_over=0.7,cmap=plt.cm.get_cmap('jet'))
Cat_post.plot_polar_fastlag(x_width=2,x_end=30,y_width=np.pi/40,x_over=0.7,cmap=plt.cm.get_cmap('jet'))


plt.figure()

lag=Cat_pre.get_array('lag',level=2)
epi_dist=Cat_pre.get_array('epi_dist',level=1)

#lag=Cat.get_array('lag',level=2)
#epi_dist=Cat.get_array('epi_dist',level=1)

plt.hist2d(epi_dist,lag,bins=70,vmax=40)
plt.colorbar()

plt.figure()
ps_time=[x.s_time-x.p_time for x in Cat_pre.obs]
plt.hist(ps_time,100)