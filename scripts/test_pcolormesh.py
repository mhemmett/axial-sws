#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:37:26 2019

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

x=[0,1,5,10]
y=np.linspace(5,20,10)

plt.close('all')
X,Y=np.meshgrid(x,y)

def XY2XY_pcolormesh(X,Y):
    """
    Function made to change the coordinates of X and Y so that
    bins of pcolormesh are properly centered on the values and not the edge
    can now plot pcolormesh(Xe,Ye,Z)
    """
    
    diff_x=np.diff(np.append(X[0,:],X[0,-1]+X[0,-2]))/2
    x_edges=X[0,:]+diff_x
    x_edges=np.insert(x_edges,0,X[0,0]-diff_x[0])
    
    diff_y=np.diff(np.append(Y[:,0],Y[-1,0]+Y[-2,0]))/2
    y_edges=Y[:,0]+diff_y
    y_edges=np.insert(y_edges,0,Y[0,0]-diff_y[0])
    
    X_e,Y_e=np.meshgrid(x_edges,y_edges)
    
    return (X_e,Y_e)

Z=np.cos(X)

fig,ax=plt.subplots()
ax.pcolormesh(X,Y,Z,vmin=-1,vmax=1)

fig,ax=plt.subplots()
ax.pcolormesh(Z,vmin=-1,vmax=1)

