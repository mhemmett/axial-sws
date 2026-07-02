#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:43:15 2018

@author: baillard
"""

import numpy as np
import os,sys

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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class,get_projection_names

plt.close('all')
fig,ax_main=plt.subplots()
ax_main.axis('equal')
ax_main.set_xlim([-5,5])
ax_main.set_ylim([-5,5])
fig.canvas.draw()

width=2
height=2
x_center=0
y_center=0

new_width,_=(ax_main.transData.transform([width,0])-ax_main.transData.transform([0,0]))/ax_main.figure.dpi
_,new_height=(ax_main.transData.transform([0,height])-ax_main.transData.transform([0,0]))/ax_main.figure.dpi
print(width,height)


### Process
ax_val= inset_axes(ax_main, width=new_width, height=new_height, loc=10, 
                   bbox_to_anchor=(x_center,y_center),
                   bbox_transform=ax_main.transData,  axes_class=get_projection_class('polar'),
                   borderpad=0.0)

    #### Set background