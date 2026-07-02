#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:07:17 2018

@author: baillard
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
import matplotlib.gridspec as gridspec
from general.plotwaveform import plot_patch,plot_particle_motion

plt.close('all')

#### Parameters

num=17
flag_plot=True
array_dir='test/AX*'
min_lag=0
max_lag=60
Nlags=70
Nangles=100

#### Get files and load xy_array

data_files=glob.glob(array_dir)

#### Select one file and dispacth

data=pickle.load(open(data_files[num],'rb'))

xy_array,sw1,sw2=data

xy_array_cut=sws.cut(xy_array,s0=sw1,s1=sw2)
        
### Process
(LAMBDA1, LAMBDA2, LAGS,ANGLES)=get_LAMBDAS(xy_array,min_lag,max_lag,Nlags=Nlags,Nangles=Nangles,
cut_s1=sw1,cut_s2=sw2)




#### Define subplots

plt.close('all')

### Define subplots

edge_1=0.4
edge_2=0.65
wspace_1=0.03
wspace_2=0.005
fig=plt.figure(figsize=(10,2.6))
gs1 = gridspec.GridSpec(3, 1,left=0.06, right=edge_1-wspace_1, top=0.9, bottom=0.2,hspace=0)
gs2 = gridspec.GridSpec(1, 1,left=edge_1+wspace_1, right=edge_2-wspace_2, top=0.9, bottom=0.2)
gs3 = gridspec.GridSpec(1, 1,left=edge_2+wspace_2, right=0.95, top=0.9, bottom=0.2)

ax_wave_1= plt.subplot(gs1[0])
ax_wave_2= plt.subplot(gs1[1],sharey=ax_wave_1,sharex=ax_wave_1)
ax_wave_3= plt.subplot(gs1[2],sharey=ax_wave_1,sharex=ax_wave_1)

ax_lambda= plt.subplot(gs3[0])
ax_polar= plt.subplot(gs2[0])

ax_wave=[ax_wave_1,ax_wave_2,ax_wave_3]


### Plot

(MinLambdas,ax)=process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                   min_thres=0.5,min_numbers=1,cont_step=0.01,quality_thres=0.2,zoom_factor=[2,6],
                   flag_plot=True,ax=ax_lambda)


lag=MinLambdas[0].lag
angle=MinLambdas[0].angle

(xy_unsplit,_)=sws.unsplit(xy_array,lag,angle,flag_plot=True,ax_list=ax_wave)
xy_unsplit_cut=sws.cut(xy_unsplit,s0=sw1,s1=sw2)

plot_particle_motion(xy_unsplit_cut,time_array=None,ax=ax_polar,vmin=None,vmax=None,xlabel='X',ylabel='Y')


### Cosmetics

plot_patch(ax_wave_1,start=sw1,width=sw2-sw1,facecolor='0.9',zorder=-1,label='S')
ax_wave_1.axvline(x=sw1+4,linestyle='-',color='k',lw=1,zorder=0)


ax_wave_2.yaxis.set_visible(False)
ax_wave_3.yaxis.set_visible(False)
ax_wave_1.xaxis.set_visible(False)
ax_wave_2.xaxis.set_visible(False)


ax_polar.set_title('Unsplit')
ax_lambda.yaxis.set_label_position("right")
ax_lambda.yaxis.tick_right()
ax_wave_1.set_xlim([sw1-100,sw2+50])
ax_lambda.set_title(r'Min. $\lambda_{2}$')

plt.savefig('POSTER_lag_effect.pdf',format='pdf',bbox_inches='tight' ,quality=300)


#plot_particle_motion(xy_array_unsplit_cut,time_array=None,ax=ax_polar_un,vmin=None,vmax=None,xlabel='X',ylabel='Y')
                             
##xx_unsplitted=unsplit(xy_array,lag_min,angle_min,flag_plot=True)
#### Take highest quality and unsplit
#
#for kk in range(len(MinLambdas)):
#    lag=MinLambdas[kk].lag
#    print(lag)
#    angle=MinLambdas[kk].angle
#    xx_unsplitted=unsplit(xy_array,lag,angle,flag_plot=True)
#    
    


    


