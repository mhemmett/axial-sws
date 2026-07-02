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
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol 

import matplotlib as mpl

#plt.close('all')
#
##### Parameters
##
pickle_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat'
pickle_files=glob.glob(pickle_dir+'/AXEC2*pickle')
pickle_files.sort()
Cat=swm.read_pickle(pickle_files)


#plt.close('all')
#fig,ax_main=plt.subplots()
#ax_main.set_xlim([0,10])
#ax_main.set_ylim([0,10])



#####

New_Cat=Cat.select(lambda_select=['rec','max'],lambda_lag=[0,40])


#lon=[x.event_lon for x in New_Cat.obs]
#lat=[x.event_lat for x in New_Cat.obs]
#z=[x.event_depth for x in New_Cat.obs]
#
#x,y=gproj.ll2xy(lon,lat,-130.1,45.9)
#
#pickle.dump([x,y,z],open('events.xyz','wb'))

#sys.exit()




plt.close('all')
plt.ion()

#(ax_main,ax_insets,ax_middle)=New_Cat.plot_polar_histofast_multi(edgecolor='k',facecolor='r',lw=0.2,norm=False,num_ax=8)
#(ax_main,_,_)=New_Cat.plot_polar_fastlag_multi(num_ax=8,x_over=0.9,y_over=0.9,x_end=40)


(ax_main,ax_insets,ax_middle,ax_top)=New_Cat.plot_polar_histofast_multi(edgecolor='k',facecolor='r',lw=0.2,norm=False,num_ax=3)
#(ax_main,ax_insets,ax_middle,ax_top)=New_Cat.plot_polar_fastlag_multi(num_ax=3,theta_rot=0,x_over=0,y_over=0,x_end=40,smooth=True)

sys.exit()




sys.exit()
#ax_middle.axis('off')

ax_map=New_Cat.plot_events(flag_rose=True,num_slice=8,cmap=plt.cm.get_cmap('jet'))
ax_map.axis('equal')
ax_map.set_xlim([2,13])
ax_map.set_ylim([2,10])
New_Cat.plot_movehist2d_time('lag',level=2,y_width=2,mode='pcolor_sample',x_width=300,x_over=0.95,norm_y=True,smooth=False)
New_Cat.plot_movehist2d_time('lag',level=2,y_width=2,mode='imshow_window',x_width=0.1,x_over=0,norm_y=True,smooth=False)
New_Cat.plot_movehist2d_time('angle',level=2,y_width=np.pi/20,mode='pcolor_sample',x_width=500,x_over=0.95,norm_y=True,smooth=False)


#plot_rose(ax=plt.gca(),num_slice=89,r_edges=3)
#fig,ax_main=plt.subplots()
##ax_main.figure.canvas.draw()
#ax_main.axis('equal')
#ax_main.set_xlim([-5,5])
#ax_main.set_ylim([-5,5])

#
#New_Cat.plot_polar_fastlag_multi(num_ax=8,x_over=0.9,y_over=0.9,x_end=20)
#
#p_times=Cat.get_array('p_time')
#s_times=Cat.get_array('s_time')
#ps_time=s_times-p_times
#ps_time=ps_time.astype(float)
#
#plt.close('all')
#
#plt.hist(ps_time)