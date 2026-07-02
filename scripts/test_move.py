#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:23:56 2018

@author: baillard
"""


import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
import glob
import time
from obspy import UTCDateTime

from scipy.ndimage import gaussian_filter

from scipy.ndimage.interpolation import zoom
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import datetime as dt
import copy
import inspect

from scipy.interpolate import griddata

import general.util as gutil
import shearwavesplit as sws
from sws_methods import movehisto2d 


import numpy as np





plt.close('all')
[lag,time]=pickle.load(open('data.test','rb'))





#x=np.array([value.timestamp for value in time])
x=time
### Make sur time is sorted

#time=time-time[0]
#time=time/86400
#time=time.astype('float')

#x=time
y=lag

x_width=int(200) # samples
y_width=1
mode='pcolor_sample'
ax=movehisto2d(x,y,x_width,y_width,mode,
                x_start=None,y_start=0,x_end=None,y_end=20,
                x_over=0,y_over=0.9,
                norm_y=True,smooth=True,gaussian_x_per=0.5,gaussian_y_per=0.0,
                x_label='X',y_label='Y',text_list=['fdh'])
movehisto2d(x,y,1,y_width,'imshow_window',
                x_start=None,y_start=0,x_end=None,y_end=20,
                x_over=0,y_over=0.9,
                norm_y=True,smooth=False,gaussian_x_per=0.5,gaussian_y_per=0.5,
                x_label='X',y_label='Y',text_list=['fdh'])
sys.exit()




x_over=0.8
y_over=0.8

x_step=int((1-x_over)*x_width)
y_step=(1-y_over)*y_width

x_start=np.min(x)
x_end=np.max(x)
y_start=np.min(y)
y_end=np.max(y)

#### Build x_mesh

x_ind_lefts=np.arange(0,len(x)-x_width,x_step)
x_lefts=x[x_ind_lefts]
x_rights=x[x_ind_lefts+x_width]
x_mesh=np.zeros_like(x_rights)
x_mesh[1:]=x_rights[:-1]+np.diff(x_rights)/2
x_mesh[0]=(x_lefts[0]+x_rights[0])/2

#### Build y_mesh

y_lefts=np.arange(y_start,y_end,y_step)
y_mesh=y_lefts+y_width-y_step/2
y_lefts=y_lefts[y_mesh<=y_end]
y_mesh=y_mesh[y_mesh<=y_end]
    
### Define meshes
    
X,Y=np.meshgrid(x_mesh,y_mesh)
Z=np.zeros_like(X)
    
### Start Counting

k_x=-1
for x_left,x_right in zip(x_lefts,x_rights):
    k_x+=1
    k_y=-1
    y_select=y[(x>x_left) & (x<=x_right)] # select data 

    for y_left in y_lefts:
        k_y+=1
        y_right=y_left+y_width
        counter=len(y_select[(y_select>y_left) & (y_select<=y_right)]) # Count numbers of elements  
        Z[k_y,k_x]=counter # storeZ=zoom(Z,6)




#extent_x=[x_mesh[0],x_mesh[-1]]
#extent_y=[y_mesh[0],y_mesh[-1]]
#    
#extent=extent_x+extent_y
#fig,ax_mesh=plt.subplots()
#plt.imshow(Z,cmap=plt.cm.get_cmap('jet'),extent=extent)
#
#ax_mesh.set_ylim(extent_y)
#ax_mesh.set_xlim(extent_x)

max_y=np.nanmax(Z,axis=0)[None,:]
max_y[max_y<=0]=1
Z=Z/max_y
plt.close('all')
fig,ax=plt.subplots(gridspec_kw={'bottom':0.2})
from matplotlib.ticker import FuncFormatter
import datetime as dt

timestamp2str(x[-1])


formatter = FuncFormatter(millions)


#Z=gaussian_filter(Z,[1,4])
ax.xaxis.set_major_formatter(formatter)
ax.imshow(Z,cmap=plt.cm.get_cmap('jet'),origin='lowerleft')


plt.setp( ax.xaxis.get_majorticklabels(), rotation=30 ,ha='right')
plt.axis('auto')
#
plt.figure()
plt.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'))
plt.axis('auto')
    
sys.exit()

x_ind_start=np.argmin(abs(x-x_start))
x_ind_end=np.argmin(abs(x-x_end))

x_ind_left=x_ind_start
x_ind_right=x_ind_left+x_width

x_left=x[x_ind_left]
x_right=x[x_ind_right]


y_lefts=np.arange(y_start,y_end,y_step)
y_mesh=y_lefts+y_width-y_step/2
y_lefts=y_lefts[y_mesh<=y_end]
y_mesh=y_mesh[y_mesh<=y_end]

x_lefts=[]
x_rights=[]
y_grid=[]
z_grid=[]
while x_ind_right<=x_ind_end:
    y_select=y[x_ind_left:x_ind_right]


    for y_left in y_lefts:
    
                y_right=y_left+y_width
                counter=len(y_select[(y_select>y_left) & (y_select<=y_right)]) 
                x_lefts.append(x[x_ind_left])
                x_rights.append(x[x_ind_right])
                y_grid.append(y_left)
                z_grid.append(counter)
                
    x_ind_left+=x_step
    x_ind_right+=x_step
#    
#x_mesh_rights=x_rights
#x_mesh_lefts=x_mesh_rights[:-1]+0.01*np.diff(x_mesh_rights)
#x_mesh_lefts=[x_lefts[0]]+list(x_mesh_lefts)
#x_grid=x_mesh_lefts+x_mesh_rights
    
x_grid=x_rights


#plt.scatter(x_grid,y_grid,c=z_grid,cmap=plt.cm.get_cmap('jet'))
xy_grid=np.column_stack((x_grid,y_grid))

x_mesh=np.linspace(x_start,x_end,500)
y_mesh=np.linspace(y_start,y_end,1000)
X,Y=np.meshgrid(x_mesh,y_mesh)
z_interp=griddata(xy_grid,z_grid,np.column_stack((X.ravel(),Y.ravel())),method='cubic')

Z=np.reshape(z_interp,X.shape)


#Z=zoom(Z,4)


#
max_y=np.nanmax(Z,axis=0)[None,:]
max_y[max_y<=0]=1
Z=Z/max_y

plt.imshow(Z,cmap=plt.cm.get_cmap('jet'),origin='lowerleft')

    



def movehisto2d(x,y,x_width,y_width,
                x_start=None,y_start=None,x_end=None,y_end=None,
                x_over=0.9,y_over=0.9,
                norm_y=False,smooth=True,gaussian_x_per=0.5,gaussian_y_per=0.5,
                x_label='X',y_label='Y',text_list=''):
    """
    Function made to plot an histo2d but using a moving window in both directionsn, this ensure better
    consistency between neighbor bins.
    The histogram can also work when data is an obspy.UTCDateTime array, then the x_start and x_end must
    be given in UTCDateTime as well and the x_width should be given in days
    
    Inputs
    ------
        x,y: np.array: arrays containing the data to apply histogram on (x can be UTCDateTime)
        x_width,y_width: float: width of the bins (in days for UTCDateTime)
        [x,y]_[start,end]: float: start and end for histogram edges
        [x,y]_over: float in [0,1]: overlap for windows [1 = full overlap]
        norm_y: Boolean: True to normalize by the maximum in each column
        smooth: Boolean: True for smoothin (Gaussian Filter)
        gaussian_[x,y]_per: float in [0,100]: width percentage for smoothing (100= filter size equal to data range)
        [x,y]_label: str
        text_list: list,str: titles to be added to the right corner of the figure
        
    Ouputs
    ------
        ax_list: plt.axes: object associated to the 4 plots (top,text,mesh,right)
    
    Comments:
    ---------
        To be added to the general functions (plot module?)
    """
    
    ### Paramaters
    
#    x_start=None
#    x_end=None
#    x_width=30
#    y_width=1
#    x_over=0.95
#    y_start=0
#    y_end=20
#    y_over=0.95
#    norm_y=True
#    smooth=True
#    gaussian_x_per=0.5 # percent of in x direction
#    gaussian_y_per=0.5 # percent of in x direction
#    x_label='Days'
#    y_label='Lags [samples]'
#    text_list=''

            
    def timestamp2matplotlib(values):
        values = list(map(dt.datetime.fromtimestamp, values))
        values = list(mdates.date2num(values))
        return values
    
    x=np.asarray(x)
    y=np.asarray(y)
    
    x_pixels=1000 # Number of elements in Z for imshow
    y_pixels=1000
    
    ### Convert Obspy.UTCDatetime to timestamp
    
    time_flag=False
    if isinstance(x[0],UTCDateTime):
        time_flag=True
        
    ### Check
    
    gaussian_x=gaussian_x_per*x_pixels/100 # Convert percentage to number or pixels
    gaussian_y=gaussian_y_per*y_pixels/100
    
    x_start=np.min(x) if x_start is None else x_start
    y_start=np.min(y) if y_start is None else y_start
    x_end=np.max(x) if x_end is None else x_end
    y_end=np.max(y) if y_end is None else y_end
    
    if time_flag:
        x=np.array([value.timestamp for value in x])
        x_width=x_width*86400 ## width must be given in days for time plots
        
        if not all((isinstance(x_start,UTCDateTime),isinstance(x_end,UTCDateTime))):
            raise ValueError('x_start and x_end must be given in obspy.UTCDateTime')
    
        x_start=x_start.timestamp
        x_end=x_end.timestamp
    
    
    def count_timewindow(x_start,y_start,x_end,x_width,y_width,x_over,y_over):
        
        ### Define bin edges 
        
        x_step=(1-x_over)*x_width
        y_step=(1-y_over)*y_width
        
        y_lefts=np.arange(y_start,y_end,y_step)
        y_mesh=y_lefts+y_width-y_step/2
        y_lefts=y_lefts[y_mesh<=y_end]
        y_mesh=y_mesh[y_mesh<=y_end]
        
        x_lefts=np.arange(x_start,x_end,x_step)
        x_mesh=x_lefts+x_width-x_step/2
        x_lefts=x_lefts[x_mesh<=x_end]
        x_mesh=x_mesh[x_mesh<=x_end]
        
        ### Define meshes
        
        X,Y=np.meshgrid(x_mesh,y_mesh)
        Z=np.zeros_like(X)
            
        ### Start Counting
        
        k_x=-1
        for x_left in x_lefts:
            k_x+=1
            k_y=-1
            x_right=x_left+x_width
            y_select=y[(x>x_left) & (x<=x_right)] # select data 
            
            for y_left in y_lefts:
                k_y+=1
                y_right=y_left+y_width
                counter=len(y_select[(y_select>y_left) & (y_select<=y_right)]) # Count numbers of elements  
                Z[k_y,k_x]=counter # store
                
        return X,Y,Z
            
        #### Zoom and filter
           