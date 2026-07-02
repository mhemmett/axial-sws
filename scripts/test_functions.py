#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:06:13 2018

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

import general.projection as gproj
import general.util as gutil
import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
    
import matplotlib as mpl

#plt.close('all')
#
##### Parameters
#
pickle_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_40_1_cat'
pickle_files=glob.glob(pickle_dir+'/AXEC2*pickle')
pickle_files.sort()
Cat=swm.read_pickle(pickle_files)

#Cat.select(lambda_select=['rec','max'])

#Cat.plot_obstime(day_width=30)
#
#file_in='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/STATIONS/stations_axial.llz'
#contour_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/caldera_smooth.ll'
#

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class,get_projection_names


plt.close('all')
fig,ax_main=plt.subplots()
ax_main.set_xlim([0,10])
ax_main.set_ylim([0,10])


def get_ax_inset(ax,x_center,y_center,width):

    ax_val= inset_axes(ax, width=width/2, height=width/2, loc=3, 
                       bbox_to_anchor=(x_center,y_center),
                       bbox_transform=ax.transData, 
                       borderpad=0.0, axes_class=get_projection_class("polar"))
    
    ax_val.patch.set_alpha(0)

    return ax_val


#
ax_1=get_ax_inset(ax_main,5,5,5)
#ax_2=get_ax_inset(ax_main,5,5,5)
ax_1.figure.canvas.draw()
#print(ax_va.get_position())
#Cat.plot_polar_events(ax=[ax_1,ax_2])
#fig,ax_1=plt.subplots(subplot_kw=dict(projection='polar'))
Cat.plot_polar_fastlag(ax=ax_1,colorbar=True,x_over=0,y_over=0,x_start=0)




#(a,b)=Cat.plot_polar_events(ax=ax_sub)
#
#
##from matplotlib.axes.Axes import update_from
#aa=update_from(ax_sub)
