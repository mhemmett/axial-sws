#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:45:20 2019

@author: baillard

"""


import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.ndimage import zoom,gaussian_filter
import pickle
import sws_methods as swm
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import numpy as np
import general.GMT as ggmt
import general.projection as gproj



x=np.linspace(0,5,10)
y=np.linspace(10,15,5)
z=np.linspace(20,25,2)


X,Y,Z=np.meshgrid(x,y,z,indexing='ij')

mode='mean'
x_start,x_end=0,100
y_start,y_end=0,100
z_start,z_end=None,None
x_width,y_width,z_width=None,None,100
x_over,y_over,z_over=0,0,0
gaussian_mode='nearest'
zoom_factors=[1,1,1] # nx,ny,nz
gaussian_factors=[0.05,0.05,0.05] # gx,gy,gz
#gaussian_factors=[0,0,0] # gx,gy,gz
shading='flat'
num_points=100
x=np.random.normal(20,5,num_points)
y=np.random.normal(60,10,num_points)
z=np.random.rand(num_points)*100

d=x**2+y**2+z**2

def xyzd2cube(x,y,z,d,mode='count',
              x_start=None,x_end=None,
              y_start=None,y_end=None,
              z_start=None,z_end=None,
              x_width=None,y_width=None,z_width=None,
              zoom_factors=[1,1,1],gaussian_factors=[0,0,0],
              shading='flat'):

    #################
    ### Functions ###
    #################
    
    def smart_arange(x_start,x_end,x_step): 
        """
        Function made to slighly arange the step and create an array that
        starts exactly at x_start and finished exactly at x_step
        
        Inputs
        ------
            x_start,x_end: float: start and end of array
            x_step: float: desired step that can be changed
        """
        x_out=np.linspace(x_start,x_end,int(round((x_end-x_start)/x_step+1)))
        print('sdfsdfsdf')
        print(int(round((x_end-x_start)/x_step+1)))
        print(x_out)
        new_step=x_out[1]-x_out[0]
        
        return (x_out,new_step)
        
    
    
    
    ### Paramters
    
#    mode='mean'
#    x_start,x_end=0,100
#    y_start,y_end=0,100
#    z_start,z_end=None,None
#    x_width,y_width,z_width=None,None,100
#    x_over,y_over,z_over=0,0,0
#    gaussian_mode='nearest'
#    zoom_factors=[1,1,1] # nx,ny,nz
#    gaussian_factors=[0.05,0.05,0.05] # gx,gy,gz
#    #gaussian_factors=[0,0,0] # gx,gy,gz
#    shading='flat'
    
    ### Create synthetic x,y,z,d
    

    
    #####################
    ### Put into cube ###
    #####################
    
    ### Check
    
    x_start=np.min(x) if x_start is None else x_start
    y_start=np.min(y) if y_start is None else y_start
    z_start=np.min(z) if z_start is None else z_start
    x_end=np.max(x) if x_end is None else x_end
    y_end=np.max(y) if y_end is None else y_end
    z_end=np.max(z) if z_end is None else z_end
    
    def_num=20
    x_width=(x_end-x_start)/def_num if x_width is None else x_width
    y_width=(y_end-y_start)/def_num if y_width is None else y_width
    z_width=(z_end-z_start)/def_num if z_width is None else z_width
    
    ### Define properly steps
    
    x_step=(1-x_over)*x_width
    y_step=(1-y_over)*y_width
    z_step=(1-z_over)*z_width
    
    (x_mesh,x_step)=smart_arange(x_start,x_end,x_step)
    print('dsfsd')
    (y_mesh,y_step)=smart_arange(y_start,y_end,y_step)
    print(z_start,z_end)
    (z_mesh,z_step)=smart_arange(z_start,z_end,z_step)
    print('dsfsd')
    #x_width=x_step/(1-x_over)
    #y_width=y_step/(1-y_over)
    #print('New x_width and y_width: %f %f'%(x_width,y_width))
    
    ### For x
    
    #x_lefts=x_mesh-x_width/2
    #x_rights=x_mesh+x_width/2
    #y_lefts=y_mesh-y_width/2
    #y_rights=y_mesh+y_width/2
    #z_lefts=z_mesh-z_width/2
    #z_rights=z_mesh+z_width/2
    
    x_lefts=x_mesh
    x_rights=x_mesh+x_width
    y_lefts=y_mesh
    y_rights=y_mesh+y_width
    z_lefts=z_mesh
    z_rights=z_mesh+z_width
    
    
    ### Define meshes
    
    X,Y,Z=np.meshgrid(x_mesh,y_mesh,z_mesh)
    D=np.zeros_like(X)
        
    ### Start Counting
    
    k_x=-1
    for x_left,x_right in zip(x_lefts,x_rights):
        k_x+=1
        k_y=-1
        for y_left,y_right in zip(y_lefts,y_rights):
            k_y+=1
            k_z=-1
            for z_left,z_right in zip(z_lefts,z_rights):
                k_z+=1
                d_select=d[(x>x_left) & (x<=x_right) &\
                            (y>y_left) & (y<=y_right) &\
                            (z>z_left) & (z<=z_right)]
                
                ### Do the math
                if mode=='count':
                    out=len(d_select)
                else:
                    out=eval('np.nan'+mode+'(d_select)')
                    print(out)
         
                D[k_y,k_x,k_z]=out # uses cartesian 'xy' indexing and not 'ij', easier for plotting
                
    
    ### Increase numbers of bins and interpolate the matrices
    
    ### As we use cartesian 'xy' convetion we need to reorder from 'x,y,z' to 'y,x,z'
                
    if zoom_factors==[1.0,1.0,1.0]:
        X_smooth,Y_smooth,Z_smooth,D_smooth=X,Y,Z,D
    else:
        zoom_factors=[zoom_factors[1],zoom_factors[0],zoom_factors[2]]
        
        Z_smooth=zoom(Z,zoom_factors,order=1) # row, column
        X_smooth=zoom(X,zoom_factors,order=1) # row, column
        Y_smooth=zoom(Y,zoom_factors,order=1) # row, column
        D_smooth=zoom(D,zoom_factors,order=1,mode='nearest') # row, column
    
    ### Filter if required
    
    gaussian_factors=[gaussian_factors[1],gaussian_factors[0],gaussian_factors[2]]
    gaussian_factors=gaussian_factors*np.array(D_smooth.shape)
    D_smooth=gaussian_filter(D_smooth,gaussian_factors,mode=gaussian_mode)
    
    return (X_smooth,Y_smooth,Z_smooth,D_smooth)


#####################
pickle_file='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat/AXEC2.cat.pickle'
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
ini_lon=-130.1
ini_lat=45.9

### Read pickle into catalog

Cat=swm.read_pickle(pickle_file)


xs,ys,zs,lags,fasts=[],[],[],[],[]

for obs in Cat.obs:
    if obs.s_time>=starteruption_time:
        continue
    lon=obs.event_lon
    lat=obs.event_lat
    
    x,y=gproj.ll2xy(lon,lat,ini_lon,ini_lat)
    
    lag=obs.MinLambdas[0].lag
    fast=obs.MinLambdas[0].angle
    
    zs.append(obs.event_depth)
    xs.append(x)
    ys.append(y)
    lags.append(lag)
    fasts.append(fast)



xs=np.array(xs)
ys=np.array(ys)
zs=np.array(zs)
lags=np.array(lags)



(X_smooth,Y_smooth,Z_smooth,D_smooth)=xyzd2cube(xs,ys,zs,lags,
x_start=5,x_end=11,
y_start=2,y_end=9,
x_width=0.3,y_width=0.3,z_width=2,mode='median')
    


plt.close('all')

fig,ax=plt.subplots()
im=ax.pcolormesh(X_smooth[:,:,0],Y_smooth[:,:,0],D_smooth[:,:,0],antialiased=True,shading='flat',cmap=plt.cm.get_cmap('jet'),vmax=30,vmin=0)
#im=ax.pcolormesh(X[:,:,0],Y[:,:,0],D[:,:,0],antialiased=True,shading=shading)
#im=ax.pcolormesh(X_smooth[:,:,0],Y_smooth[:,:,0],D_smooth[:,:,0],antialiased=True,shading='flat')
plt.colorbar(im)
#plt.plot(xs,ys,'ok',ms=1)  
ax.set_aspect('equal','box')    

ggmt.plot_lines(ax=ax,color='k',lw=2,ini_lon=-130.1,ini_lat=45.9)







