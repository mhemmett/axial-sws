#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:43:37 2019

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

import numpy.ma as ma
from scipy.ndimage import zoom,gaussian_filter
import pickle
import sws_methods as swm
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import numpy as np
import general.GMT as ggmt
import general.projection as gproj

### Parameters

file_in='data.xyzd'
file_in='AXEC2_post.xyzd'


### Read pickle

[xs,ys,zs,lags]=pickle.load(open(file_in,'rb'))

x=xs
y=ys
z=zs
d=lags

x=np.array(xs)
y=np.array(ys)
z=np.array(zs)
d=np.array(lags)



dic_mesh=swm.xyzd2mesh(x,y,z,d,
              x_start=2,x_end=12,
              y_start=0,y_end=10,
              z_start=0,z_end=2,
              x_step=0.1,y_step=0.1,z_step=0.1,
              dis_lim=0.3,num_lim=100,verbose=True)

MEDIAN_mesh=dic_mesh['MEDIAN_mesh']
MEAN_mesh=dic_mesh['MEAN_mesh']
MEANZ_mesh=dic_mesh['MEANZ_mesh']
COUNT_mesh=dic_mesh['COUNT_mesh']
X_mesh=dic_mesh['X_mesh']
Y_mesh=dic_mesh['Y_mesh']

MA = ma.array(MEDIAN_mesh, mask = COUNT_mesh<=10)
MA = ma.array(MEAN_mesh, mask = COUNT_mesh<=10)
MA = ma.array(MEANZ_mesh, mask = COUNT_mesh<=10)
#MA = ma.array(STD_mesh, mask = COUNT_mesh<=20)

plt.close('all')

fig,ax=plt.subplots()
#im=ax.pcolormesh(X_mesh[:,:,0],Y_mesh[:,:,0],D_mesh[:,:,1],antialiased=True,shading='flat',cmap=plt.cm.get_cmap('jet'),vmax=30)
im=ax.pcolormesh(X_mesh[:,:,0],Y_mesh[:,:,0],MA[:,:,10],
                 antialiased=True,shading='flat',cmap=plt.cm.get_cmap('jet'))
cbar=plt.colorbar(im)
cbar.ax.set_ylabel('Lags [samples]')
ggmt.plot_lines(ax=ax,color='k',lw=2,ini_lon=-130.1,ini_lat=45.9)
ggmt.plot_stations(ax=ax)

#plt.plot(x,y,'ok',ms=0.1)  
ax.set_aspect('equal','box')  


fig,ax=plt.subplots()
im=ax.pcolormesh(X_mesh[:,:,0],Y_mesh[:,:,0],COUNT_mesh[:,:,10],antialiased=True,shading='flat',cmap=plt.cm.get_cmap('jet'),vmin=5,vmax=6)
plt.colorbar(im)
plt.plot(x,y,'ok',ms=0.1)  
ax.set_aspect('equal','box')  