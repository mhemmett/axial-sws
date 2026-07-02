#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:27:11 2018

@author: baillard
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import general.GMT as ggmt
import general.util as gutil
import general.projection as gproj
import glob
import sws_vertices as swv
import sws_methods as swm
import shearwavesplit as sws
import time
import sys

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pickle
import copy


[x,y,z]=pickle.load(open('events.xyz','rb'))


pickle_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_60_1_cat'
pickle_files=glob.glob(pickle_dir+'/AXEC2*pickle')
pickle_files.sort()
start=time.time()
Cat=swm.read_pickle(pickle_files)



New_Cat=Cat.select(lambda_select=['rec','max'],obs_contour_keys=['2'],obs_epi_dist=[0,3])
plt.close('all')
New_Cat.plot_movehist2d_time('lag',level=2,y_width=2,y_end=50,norm_y=True,x_width=500,mode='imshow_sample')
New_Cat.plot_movehist2d_time('angle',level=2,y_width=np.pi/30,y_end=np.pi/2,norm_y=True,x_width=1000,mode='imshow_sample')

ax=New_Cat.plot_events(color='k')

ps_time=[x.s_time-x.p_time for x in New_Cat.obs]

A.plot_contours(ax=ax,contour_keys=['0','1','3','8','10'],facecolor='r',textcolor='w')