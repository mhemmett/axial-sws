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
max_lag=20
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

middle=0.55
fig=plt.figure(figsize=(9.4,5))
gs1 = gridspec.GridSpec(3, 2,
                       width_ratios=[1,1],left=0.1, right=0.9, top=0.99, bottom=middle+0.05,wspace=0.01,hspace=0)
gs2 = gridspec.GridSpec(1, 3,
                       width_ratios=[1,1,1],left=0.1, right=0.9, top=middle-0.05, bottom=0.1,wspace=0.1)



ax_wave_1= plt.subplot(gs1[0])
ax_wave_2= plt.subplot(gs1[2],sharey=ax_wave_1,sharex=ax_wave_1)
ax_wave_3= plt.subplot(gs1[4],sharey=ax_wave_1,sharex=ax_wave_1)
ax_wavez_1= plt.subplot(gs1[1])
ax_wavez_2= plt.subplot(gs1[3],sharey=ax_wavez_1,sharex=ax_wavez_1)
ax_wavez_3= plt.subplot(gs1[5],sharey=ax_wavez_1,sharex=ax_wavez_1)
ax_lambda= plt.subplot(gs2[0])
ax_polar_1= plt.subplot(gs2[1])
ax_polar_2= plt.subplot(gs2[2],sharex=ax_polar_1, sharey=ax_polar_1)

ax_wave=[ax_wave_1,ax_wave_2,ax_wave_3]
ax_wavez=[ax_wavez_1,ax_wavez_2,ax_wavez_3]

### Plot

(MinLambdas,ax)=process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                   min_thres=0.5,min_numbers=2,cont_step=0.01,quality_thres=0.2,zoom_factor=[2,6],
                   flag_plot=True,ax=ax_lambda)


lag=MinLambdas[0].lag
angle=MinLambdas[0].angle

(xy_unsplit,_)=sws.unsplit(xy_array,lag,angle,flag_plot=True,ax_list=ax_wave)
(xy_unsplit,_)=sws.unsplit(xy_array,lag,angle,flag_plot=True,ax_list=ax_wavez)
xy_unsplit_cut=sws.cut(xy_unsplit,s0=sw1,s1=sw2)

plot_particle_motion(xy_array_cut,time_array=None,ax=ax_polar_1,vmin=None,vmax=None,xlabel='X',ylabel='Y')
plot_particle_motion(xy_unsplit_cut,time_array=None,ax=ax_polar_2,vmin=None,vmax=None,xlabel='X',ylabel='Y')


### Cosmetics

plot_patch(ax_wave_1,start=sw1,width=sw2-sw1,facecolor='0.9',zorder=-1,label='S')
ax_wave_1.axvline(x=sw1+4,linestyle='-',color='k',lw=1,zorder=0)
    
ax_wave_1.set_xlim([0,ax_wave_1.get_xlim()[1]])
plot_patch(ax_wavez_1,start=sw1,width=sw2-sw1,facecolor='0.9',zorder=-1,label='S')
ax_wavez_1.axvline(x=sw1+4,linestyle='-',color='k',lw=1,zorder=0)
                                
plt.setp(ax_polar_1.get_yticklabels(), visible=False)
ax_wavez_1.yaxis.set_visible(False)
ax_wavez_2.yaxis.set_visible(False)
ax_wavez_3.yaxis.set_visible(False)

ax_wave_2.yaxis.set_visible(False)
ax_wave_3.yaxis.set_visible(False)

ax_wave_1.xaxis.set_visible(False)
ax_wave_2.xaxis.set_visible(False)


ax_polar_1.set_ylabel('')
ax_polar_1.set_title('Initial')
ax_polar_2.set_title('Unsplit')
ax_polar_2.yaxis.set_label_position("right")
ax_polar_2.yaxis.tick_right()
ax_wavez_1.set_xlim([sw1-50,sw2+50])
ax_lambda.set_title(r'Min. $\lambda_{2}$')

plt.savefig('POSTER_diagnostic.pdf',format='pdf',bbox_inches='tight' ,quality=300)


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
    


    


