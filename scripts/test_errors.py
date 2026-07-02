#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:01:39 2018

@author: baillard

Script made to test errors estimates
"""


import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal
import sys
import copy
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import time
from scipy.signal.windows import hann
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

from scipy.ndimage import zoom

from shearwavesplit import get_LAMBDAS,split,unsplit,process_LAMBDAS,dominant_freq,spectrum
import general.util as gutil
import shearwavesplit as sws

def get_dom_period(xy_array,p_samples,s_samples,delta_t,n_samples_freq):
    
    delta_ps=int(round(s_samples-p_samples))
    
    ### Define windows for dom_period computation
    
    rf_1=int(round(n_samples_freq[0])) # relative freq
    rf_2=int(round(n_samples_freq[1]))
    
    ### Make sure left frequency windows is smaller than delta_ps/2 
    rf_1=int(round(delta_ps/2)) if rf_1 >= int(round(delta_ps/2)) else rf_1 
    
    nf_1=s_samples-rf_1
    nf_2=s_samples+rf_2
    
    ### Make sure nf_2 is not longer than data
    
    nf_2=xy_array.shape[0] if nf_2>xy_array.shape[0] else nf_2
    
    ### Cut the array for freq
    
    xy_array_f=xy_array[nf_1:nf_2,:]
    
    ### Compute dominant period
    
    xy_array_w=xy_array_f*hann(xy_array_f.shape[0])[:,None]
    
    spectrum(xy_array_w[:,0],delta_t=delta_t,flag_plot=True)
    spectrum(xy_array_w[:,1],delta_t=delta_t,flag_plot=True)
    spectrum(xy_array_w[:,0]+xy_array_w[:,1],delta_t=delta_t,flag_plot=True)
    domP_x=1/(dominant_freq(xy_array_w[:,0],delta_t=delta_t)*delta_t)
    domP_y=1/(dominant_freq(xy_array_w[:,1],delta_t=delta_t)*delta_t)
    #domP=1/(dominant_freq(xy_array_w[:,0]+xy_array_w[:,1],delta_t=delta_t)*delta_t)
    
    domP=max([domP_x,domP_y]) ## Take the biggest period

    return (domP,(domP_x,domP_y))
  

plt.close('all')
#### Parameters

flag_plot=True
array_dir='test/AX*'
min_lag=0
max_lag=14
Nlags=70
Nangles=100

#### Get files and load xy_array

data_files=glob.glob(array_dir)

#### Select one file

num=2
data=pickle.load(open(data_files[num],'rb'))
print(data_files[num])
xy_array,sw1,sw2=data


(domP,(domP_x,domP_y))=get_dom_period(xy_array,0,sw1,1/200,[20,60])

sw2=int(round(sw1+2*domP))
max_lag=int(round(1*domP))+1
print('window_size',sw2-sw1)
print('dominant periods',domP_x,domP_y,domP)
print('max_lag:',max_lag)

### plot data

if flag_plot:
    fig,ax=plt.subplots(2,1)
    
    for i in range(len(ax)):
        ax[i].plot(xy_array[:,i])
        ylim=ax[i].get_ylim()
        ax[i].vlines(sw1,ylim[0],ylim[1])
        ax[i].vlines(sw2,ylim[0],ylim[1])
#    
##### Create synthetics
##    
#xx_array=np.column_stack((xy_array[:,0],xy_array[:,0]))    
#
#angle_rad=2*np.pi/2
#angle_rad=-np.pi/4
#lag=20
#
#xx_splitted=split(xx_array,lag,angle_rad,flag_plot=True)
#xx_unsplitted=unsplit(xx_splitted,lag,angle_rad,flag_plot=True)
#
#
#if flag_plot:
#    fig,ax=plt.subplots(2,1)
#    
#    for i in range(len(ax)):
#        ax[i].plot(xx_array[:,i])
#        ylim=ax[i].get_ylim()
#        ax[i].vlines(sw1,ylim[0],ylim[1])
#        ax[i].vlines(sw2,ylim[0],ylim[1])
#    
#
#xy_array=copy.deepcopy(xx_splitted)   


#points = 500
#a = 10
#vec2 = signal.ricker(points, a)
#plt.plot(vec2)

#### Get Lambdas
        
(LAMBDA1, LAMBDA2, LAGS,ANGLES)=get_LAMBDAS(xy_array,min_lag,max_lag,Nlags=Nlags,Nangles=Nangles,
cut_s1=sw1,cut_s2=sw2)

lag_extent=[min_lag,max_lag]

#LAMBDA2=zoom(LAMBDA2,[2,4])
#
#min_list=pick_minima(LAMBDA2,flag_plot=True)
#min_list,_=sws.extrema(LAMBDA2,mode='constant',size=20)
#
#x,y,z=zip(*min_list)
#
#plt.figure()
#plt.imshow(LAMBDA2,cmap=plt.cm.get_cmap('jet'))
#plt.plot(y,x,'+w')
#sys.exit()

        
start=time.time()
(MinLambdas,ax)=process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                   min_thres=0.5,min_numbers=2,cont_step=0.01,quality_thres=0.2,zoom_factor=[2,6],
                   flag_plot=True)
end=time.time()
print(MinLambdas)


#xx_unsplitted=unsplit(xy_array,lag_min,angle_min,flag_plot=True)
### Take highest quality and unsplit

for kk in range(len(MinLambdas)):
    lag=MinLambdas[kk].lag
    print(lag)
    angle=MinLambdas[kk].angle
    xx_unsplitted=unsplit(xy_array,lag,angle,flag_plot=True)
    
    


    


