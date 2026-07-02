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
import sws_methods as swm

  

plt.close('all')
#### Parameters

flag_plot=True
array_dir='test/AX*'
min_lag=0
max_lag=60
add_lag=3
Nlags=70
Nangles=180
sampling_rate=200

#### Get files and load xy_array

data_files=glob.glob(array_dir)

#### Select one file

num=70

data=pickle.load(open(data_files[num],'rb'))
print(data_files[num])
xy_array,sw1,sw2=data




### Get dominant period on X and Y (given in samples)

xy_array_dom=xy_array[sw1:sw2,:]

(dom_period_x,dom_freq_x)=sws.get_dominant_period(xy_array_dom[:,0],sampling_rate,flag_plot=True)
(dom_period_y,dom_freq_y)=sws.get_dominant_period(xy_array_dom[:,1],sampling_rate,flag_plot=True)

dom_period=np.mean([dom_period_x,dom_period_y])


sw2=int(round(sw1+2*dom_period))
max_lag=int(round(1*dom_period))+1
print('window_size',sw2-sw1)
print('dominant periods',dom_period_x,dom_period_y,dom_period)
print('max_lag:',max_lag)



### plot data

if flag_plot:
    fig,ax=plt.subplots(2,1)
    
    for i in range(len(ax)):
        ax[i].plot(xy_array[:,i])
        ylim=ax[i].get_ylim()
        ax[i].vlines(sw1,ylim[0],ylim[1])
        ax[i].vlines(sw2,ylim[0],ylim[1])


### Compute SNR
  
plt.close('all')
ratio_x=swm.SNR_pick(xy_array[:,0],sw1,30,sw2-sw1,mode='mean',flag_plot=True)
ratio_y=swm.SNR_pick(xy_array[:,1],sw1,30,sw2-sw1,mode='mean',flag_plot=True)

print('SNR x=%f, y=%f'%(ratio_x,ratio_y))
              

#### Get Lambdas


(LAMBDA1, LAMBDA2, LAGS,ANGLES)=get_LAMBDAS(xy_array,min_lag-add_lag,max_lag+add_lag,Nlags=Nlags,Nangles=Nangles,
    cut_s1=sw1,cut_s2=sw2)



#LAMBDA2=LAMBDA2/LAMBDA1
#
#x,y,z=zip(*min_list)
#
#plt.figure()
#plt.imshow(LAMBDA2,cmap=plt.cm.get_cmap('jet'))
#plt.plot(y,x,'+w')
#sys.exit()


start=time.time()
(MinLambdas,ax)=process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                    min_lag_thres=min_lag,max_lag_thres=max_lag,
                   min_thres=0.5,min_numbers=3,cont_step=0.01,quality_thres=0.2,zoom_factor=[2,6],
                   flag_plot=True)
end=time.time()


#xx_unsplitted=unsplit(xy_array,lag_min,angle_min,flag_plot=True)
### Take highest quality and unsplit




MinLambdas=sws.rms_MinLambdas(xy_array,sw1,sw2,MinLambdas,mode='norm',flag_plot=True)

    


