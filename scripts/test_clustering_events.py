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

import general.GMT as ggmt

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class,get_projection_names
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import DBSCAN,KMeans,SpectralClustering

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)

### Parameters

file_in='events.xyz'

[x,y,z]=pickle.load(open(file_in,'rb'))
z=np.asarray(z)
ax.plot(x,y,'ok',ms=2,alpha=0.05,mec='none')
ax.axis('equal')
sys.exit()
#plt.axis([4,12,2,10])

### Subevents

#ind_sub=np.random.randint(0,len(x),len(x))
ind_sub=np.arange(len(x),dtype='int')
x_sub=x[ind_sub]
y_sub=y[ind_sub]
z_sub=z[ind_sub]
#ax.scatter(x_sub,y_sub,z_sub)
ax.plot(x_sub,y_sub,'ok',ms=0.5,mfc='k')
ax.axis('equal')

### Cluster

X=np.column_stack((x_sub,y_sub))

def dbscan_twoclusters(X):
    n_clusters=1
    perc_eps=2
    while n_clusters<2:
        perc_min_samples=0.07
        
        x_spread=np.max(X[:,0])-np.min(X[:,0])
        y_spread=np.max(X[:,1])-np.min(X[:,1])
        dist_spread=np.sqrt(x_spread**2+y_spread**2)
        
        eps=dist_spread*perc_eps/100
        min_samples=X.shape[0]*perc_min_samples/100
        print(eps,min_samples)
        db = DBSCAN(eps=eps, min_samples=min_samples,algorithm='auto').fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        perc_eps-=0.05
        
    X_out=get_X(db,X)
    
    return X_out
    
def get_X(db,X_in):
    X_out=[]
    len_out=[]
    labels = db.labels_
    unique_labels = set(labels[labels>=0])
    for label in unique_labels:
        ind_select= (labels == label)
        X_out.append(X_in[ind_select,:])
        len_out.append(len(ind_select))
        
    ### sorrt
    X_out=[X_out[ind] for ind in np.argsort(len_out)]
    return X_out
    
def plot_X(X_list):

    fig,ax = plt.subplots()
    colors=ggmt.data2rgb(range(len(X_list)),cmap=plt.cm.get_cmap('hsv'))
    
    for kk,color in enumerate(colors):
        xy=X_list[kk]
        ax.plot(xy[:,0],xy[:,1],'o',mfc=color,mec='none',alpha=1,ms=1)
        
    ax.axis('equal')
    
def multi_dbscan(X,num_max=20):
    new_X=[]
    while len(new_X)<num_max:
        X_out=dbscan_twoclusters(X)
        plt.close('all')
     
        plot_X(X_out)
        plt.pause(5)
          
        new_X.append(X_out[1:])
        X=X_out[0]
        print(len(new_X))
     
    return [X]+new_X
        

A=multi_dbscan(X,num_max=20)

#db = DBSCAN(eps=5, min_samples=40,algorithm='auto').fit(X)
#centers=np.array([[8.5,2.7],[6.8,4.5],[8.5,5.2],[7.8,2.8],[8.1,5.7]])
#centers=np.array([[8.5,2.7],[8.1,5.7]])
#len_c=centers.shape[0]
#db = KMeans(n_clusters=len_c,init=centers,n_init=1,max_iter=300).fit(X)
#db = KMeans(n_clusters=len_c,n_init=10,max_iter=300).fit(X)
#db = SpectralClustering(n_clusters=2).fit(X)
#labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#print(n_clusters_)
#
##ax.plot(x_sub,y_sub,'or')
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.axis('equal')
#
#unique_labels = set(labels[labels>=0])
#colors=ggmt.data2rgb(range(len(unique_labels)),cmap=plt.cm.get_cmap('hsv'))
#
#for label,color in zip(unique_labels,colors):
#    ind_select= (labels == label)
#    xy=X[ind_select,:]
#    print(len(xy))
#    
#    ax.plot(xy[:,0],xy[:,1],'o',mfc=color,mec='none',alpha=1,ms=1)
#    


