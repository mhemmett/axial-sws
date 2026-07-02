#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:38:42 2018

@author: baillard
"""


import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal
import sys
import copy
import time
from scipy.ndimage import zoom
from scipy.ndimage.filters import minimum_filter,maximum_filter
import logging

from shearwavesplit import get_LAMBDAS,split,unsplit,LAMBDA2lagangle,get_contours,select_contour,box_contour
import general.util as gutil


#### Define circle


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

angles=np.linspace(-np.pi/4,-np.pi/4+2*np.pi,100)
r=np.ones_like(angles)

x,y=pol2cart(angles,r)

plt.close('all')
plt.scatter(x,y,c=angles)
plt.axis('equal')


def split_contour(cont,y_lim_list,flag_plot=False):
    """
    Function made to split a contour in multiple subcontours
    """
    
    if not isinstance(y_lim_list,list):
        y_lim_list=[y_lim_list]
    else:
        y_lim_list.sort(reverse=True)
        
    print(y_lim_list)
    cont_list=[]
    for y_lim in y_lim_list:
        cont=add_y(cont,y_lim)
        (top,bottom)=split_y(cont,y_lim)
        cont_list.append(top)
        cont=bottom
  
    cont_list.append(bottom)
    
    #### Plot if asked
    if flag_plot:
        fig,ax=plt.subplots()
        ax.axis('equal')
        
        for cont in cont_list:
            if cont is None:
                continue
            ax.plot(cont[:,0],cont[:,1],'k')
        
        
    return cont_list
    
def add_y(cont,y_lim):
    
    if cont is None:
        return None
    
    x=cont[:,0]
    y=cont[:,1]
    y_diff=y-y_lim
    
    ind_sel=np.where(np.diff(np.sign(y_diff))!=0)[0]
    ind_next=ind_sel+1
    
    ind_pairs=list(zip(ind_sel,ind_next))
    
    x_new=[]
    y_new=[]
    k=0
    for ind_pair in ind_pairs:
        ind_pair=list(ind_pair)
        x_bord=x[ind_pair]
        y_bord=y[ind_pair]
        x_bord=x_bord[np.argsort(y_bord)]
        y_bord=y_bord[np.argsort(y_bord)]
        
        x_lim=np.interp(y_lim,y_bord,x_bord)
        plt.plot(x_bord,y_bord,'ok')
        plt.plot(x_lim,y_lim,'or')
        
        x_new.extend(list(x[k:ind_pair[0]+1]))
        x_new=x_new+[x_lim]
        y_new.extend(list(y[k:ind_pair[0]+1]))
        y_new=y_new+[y_lim]

        k=ind_pair[1]
        
    
    x_new.extend(list(x[k:]))
    y_new.extend(list(y[k:]))
    
    x_new=np.array(x_new)
    y_new=np.array(y_new)
    
    return np.column_stack((x_new,y_new))


def split_y(cont,y_lim,flag_plot=False):
    
    if cont is None:
        return (None,None)

    x=cont[:,0]
    y=cont[:,1]
    y_abov=copy.copy(y)    
    y_abov=y[y>=y_lim]
    x_abov=x[y>=y_lim]    
    
    y_bott=copy.copy(y)    
    y_bott=y[y<=y_lim]
    x_bott=x[y<=y_lim]   
    
    if x_abov.size==0:
        return (None,cont)
    if x_bott.size==0:
        return (cont,None)
    
   
    x_abov=np.hstack((x_abov,x_abov[0]))
    y_abov=np.hstack((y_abov,y_abov[0]))
    
    x_bott=np.hstack((x_bott,x_bott[0]))
    y_bott=np.hstack((y_bott,y_bott[0]))
        

    
    cont_abov=np.column_stack((x_abov,y_abov))
    cont_bott=np.column_stack((x_bott,y_bott))
    
    if flag_plot:
        fig,ax=plt.subplots()
        ax.plot(x,y,'k',lw=6)
        ax.plot(x_abov,y_abov,'r',lw=3)
        ax.plot(x_bott,y_bott,'y',lw=1)
        
    return (cont_abov,cont_bott)

    
#y_lim=list(np.linspace(-1,1,10))
#data=np.column_stack((x,y))
#
#cont_list=split_contour(data,y_lim,flag_plot=True)
#


