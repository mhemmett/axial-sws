#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:16:38 2019

@author: baillard
"""


from obspy.io.nlloc.core import read_nlloc_hyp
import sws_methods as swm
import general.plotwaveform as pltwf
import sys
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy
import os

from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec



station_name='AXAS2'
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat/'
file_level='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/PRESSURE/2013-2015-LTplot.txt'


file_in= os.path.join(cat_dir,'')+station_name+'.clean.cat.pickle'

### Read pickle into catalog
Cat=swm.read_pickle(file_in)

### Get arrays of data

elems_dic=Cat.get_dic(minlambda_select='min')

### Select arrays based on baz

data=np.column_stack((elems_dic['x'],
                      elems_dic['y'],
                      elems_dic['z'],
                      elems_dic['baz_trigo'],
                      elems_dic['s_time'],
                      elems_dic['lag'],
                      elems_dic['fast']))

x=elems_dic['s_time']
xs=np.array([value.timestamp for value in x])
y=elems_dic['fast']
y=elems_dic['fast']*180/np.pi
y=swm.trigo2azimuth(y)



def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


# %
plt.close('all')
x_end=UTCDateTime(2015,7,24,6)
x_start=UTCDateTime(2015,1,24,6) 
x_width=500
x_over=0.95
y_over=0.95
#(ax,im,X,Y,Z,x_bins,x_diffs)=swm.plot_movehisto2d(x,y,y_end=30,x_width=x_width,flag_y_norm=True,x_end=x_end,x_start=x_start,
#x_over=x_over,y_over=y_over,x_mode='sample',flag_resample=True,flag_filter=True,y_cycle=None)


#ax=Cat.plot_movehisto2d_time(y_param='fast',
#                             window_unit='hour',window_bottom=10,angle_mode='azimuth',
#                             y_end=180,y_start=0,x_width=x_width,y_width=10,flag_y_norm=True,x_end=x_end,x_start=x_start,
#x_over=x_over,y_over=y_over,x_mode='sample',flag_resample=False,flag_filter=False,y_cycle=[0,180])

x_width=200
x_over=0.9
y_over=0.9

x_endz=UTCDateTime(2015,4,24,6)+12*3600
x_startz=UTCDateTime(2015,4,24,6)-12*3600
ax=Cat.plot_movehisto2d_time(y_param='fast',
                             window_unit='minute',window_bottom=1,angle_mode='azimuth',
                             y_end=180,y_start=0,x_width=x_width,y_width=10,flag_y_norm=True,x_end=x_endz,x_start=x_startz,
x_over=x_over,y_over=y_over,x_mode='sample',flag_resample=True,flag_filter=True,y_cycle=[0,180])


### Define time windows lenths

#window_unit='minute'
#
#if window_unit=='minute':
#    x_diffs/=60
#elif window_unit=='hour':
#    x_diffs/=3600
#elif window_unit=='day':
#    x_diffs/=86400
#elif window_unit=='second':
#    x_diffs=x_diffs
#else:
#    raise ValueError('window unit must be either, day, hour, minute, second')
#
#yr_label='Window size [%s]'%window_unit[0]
#    
#
#ax_win = ax.twinx()
#ax_win.set_yscale("log")
#
#p3, = ax_win.plot(x_bins, x_diffs/3600, "k-",lw=3)
#p3, = ax_win.plot(x_bins, x_diffs/3600, "w-",lw=1)
#ax_win.set_ylim(bottom=None)


#
#host.set_xlim(0, 2)
#host.set_ylim(0, 2)
#par1.set_ylim(0, 4)
#par2.set_ylim(1, 65)
#
#host.set_xlabel("Distance")
#host.set_ylabel("Density")
#par1.set_ylabel("Temperature")
#par2.set_ylabel("Velocity")
#
#host.yaxis.label.set_color(p1.get_color())
#par1.yaxis.label.set_color(p2.get_color())
#par2.yaxis.label.set_color(p3.get_color())
#
#tkw = dict(size=4, width=1.5)
#host.tick_params(axis='y', colors=p1.get_color(), **tkw)
#par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
#par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
#host.tick_params(axis='x', **tkw)
#

#(ax,im,X,Y,Z)=plot_movehisto2d(x,y,y_end=30,x_width=500,flag_y_norm=True,
#                 x_over=0.9,y_over=0.9,x_mode='sample',
#                 flag_resample=False,dx_resample=86400,flag_filter=False)

# 
