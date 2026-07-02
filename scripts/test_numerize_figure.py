#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:04:44 2018

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

import general.util as gutil
import general.projection as gproj
import general.util as gutil
import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
import matplotlib.image as mpimg

plt.close('all')

file_in='./lava_2015.pickle'

ref,contour=pickle.load(open(file_in,'rb'))

x=contour[:,0]
y=contour[:,1]

ll=ref[0,:]
ur=ref[1,:]

ll_prime=[-130.033,45.967]
ur_prime=[-130,46]

dx_prime=ur_prime[0]-ll_prime[0]
dx=ur[0]-ll[0]
dy_prime=ur_prime[1]-ll_prime[1]
dy=ur[1]-ll[1]

x_prime=(x-ll[0])*(dx_prime/dx)+ll_prime[0]
y_prime=(y-ll[1])*(dy_prime/dy)+ll_prime[1]

x_prime=x_prime[1:]
y_prime=y_prime[1:]

x_prime,y_prime=gproj.ll2xy(x_prime,y_prime,ini_lon=-130.1,ini_lat=45.9)


x_s,y_s=gutil.smooth_curve(x_prime,y_prime,periodic=True,smoothness=0.01,flag_plot=True)
x_s=x_s-0.1

### Convert back to lon,lat

lon_s,lat_s=gproj.xy2ll(x_s,y_s,ini_lon=-130.1,ini_lat=45.9)

###

new_data=np.column_stack((lon_s,lat_s))

np.savetxt('/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/axial_lavaflow_2015.ll',new_data,fmt='%12.5f')

sys.exit()
plt.scatter(x_prime,y_prime,c=range(len(y_prime)))
plt.axis('equal')


#
file_im='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/FIG/2015_lava_flow.png'
#
#
img=mpimg.imread(file_im)
plt.imshow(img)
#plt.grid()
#print('Pick two points for reference frame')
#ref=plt.ginput(n=2)
#print('Pick contour')
#contour=plt.ginput(n=0,timeout=0)
