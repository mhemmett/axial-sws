#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:23:45 2019

@author: baillard
"""

from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import general.projection as gproj
import general.cube_V1 as cube
import sys
import copy


angle_deg=45
center=[0,0]
len_prof=[0,14.142]
width_prof=[0.5,0.5]

x=np.linspace(0,10,30)
y=np.linspace(0,5,20)
z=np.linspace(0,20,10)

X,Y,Z=np.meshgrid(x,y,z)

D=copy.copy(Z)
D[X>5]=Z[X>5]+20

Xs,Ys,Ds=cube.get_slice(X,Y,Z,D,'z=4')


plt.pcolormesh(Xs,Ys,Ds)
plt.colorbar()
data=np.column_stack((X.reshape(-1),Y.reshape(-1),Z.reshape(-1),D.reshape(-1)))

proj_data,select_boolean=gproj.project(data,center,angle_deg,len_prof,width_prof)



plt.close('all')



#(ax,h1)=cube.plot_contour(X,Y,Z,D,slice_param=None,
#             center=center,angle_deg=angle_deg,len_prof=len_prof,width_prof=width_prof,step=None,
#             ax=None,
#             coef_std=None,c_center=None,
#             filled=True,linewidths=0.5,levels=None,vmin=None,vmax=None,
#             zoom_val=1,gaussian_val=0,perc_change=False,type_cross='mean',
#             c_extend=False,extend='neither')

(ax,h1)=cube.plot_contour(X,Y,Z,D,slice_param='z=4',
             ax=None,
             coef_std=None,c_center=None,
             filled=True,linewidths=0.5,levels=None,vmin=None,vmax=None,
             zoom_val=1,gaussian_val=0,perc_change=False,type_cross='mean',
             c_extend=False,extend='neither')