#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:04:28 2018

@author: baillard

This is the main script to do shearwavesplitting analysis automatically

For shear wave splitting we use the convention of a Nx2 data array with the first column being the X/E component and the second column 
being the Y/N direction
    

"""


import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.patches import Rectangle
import os,pickle,time
import logging
import time,sys
import gc
import traceback
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) ## For scipy.fft


from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.io.nlloc.core import read_nlloc_hyp
from obspy.signal.polarization import particle_motion_odr,vidale_adapt,eigval
from obspy.taup import TauPyModel
from general.util import smooth_curve

from general.plotwaveform import SDS2streams,plot_event,getPickForArrival,getPicks,getPicksAndArrivals,plot_picks,stream2data,jurkevics
from general.plotwaveform import plot_window,plot_particle_motion,plot_particle_motion_stream
import shearwavesplit as sw
import general.plotwaveform as pltwf
import sws_methods as swm


###################"
### Parameters


file_range=[6,7]
#file_range=[13,15]
#file_range=[20,22]
plt.close('all')
sds_root='/home/baillard/SDS/'
nlloc_dir_tmp='SWS_nlloc'
flag_split=False
start_time=UTCDateTime(2015,2,1)
time_delay=5
network_code='OO'
channel_code="EH?,HH?"
p_window=[0.02,0.1] ### for P Polarization analysis
s_window=[0.02,0.3] ### for S Shearwavesplitting 
fs_window=[0.1,0.3] ### To compute dominant period of S waves
s_snr_window=[0.4,0.2]
s_snr_thres=2 ### SNR threshod to keep data
flag_adapt_window=True ### Adapt size of the SWS window based on dominant period (T*2)
flag_adapt_maxlag=True ### Adapt max lagg allows for SWS based on dominant period (T)
stream_window=[1,1]  #### for extraction of the stream, before P and after S
freqmin=5
freqmax=40
flag_plot=False
inc_thres=30 # incidence threshold
rec_thres=0.7 # rectilinearity threshold
log_file='SWS_Final.log'+str(file_range[0])+'_'+str(file_range[1])
event_max=5000


### Obspy Client parameter to get waveforms
network_code="OO"
center="IRIS"
channel_code="EH*,HH*"

###  SWS parameters
min_lag=0
max_lag=60
Nlags=60
Nangles=90
add_lag=3 # Lag in samples to extend the LAMBDAS mesh to avoid extrema to be picked at borders (should not be changed)

print('Make sure that S window is longer than max_lag, otherwise Y component will not be in window')

## Paths
#nlloc_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial_EQ/PROG/NLLOC_MODELS/NLLOC_3H/loc/AXIAL.sum'
nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/CATALOG/AXIAL.PHASE.FINAL_3D_V2.nlloc'
station_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/STATIONS/stations_axial.xyz'
pickle_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2'

##### Initialize logging

logger = logging.getLogger()
logger.handlers=[]
logger.setLevel(logging.INFO)
fh= logging.FileHandler(log_file,mode='w')
sh= logging.StreamHandler()
logger.addHandler(fh)
logger.addHandler(sh)

####

logging.info('Logging file for Shear Wave Splitting')

##### directory for temp files


if not os.path.exists(nlloc_dir_tmp):
    os.makedirs(nlloc_dir_tmp)

###########################################
####  Create directory to store pickles ###
###########################################

if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)
else:
    user_input=input('Following directory already exists:\n%s\nContinue? [y/n]'%pickle_dir)
    if user_input!='y':
        sys.exit('Program terminated by user')
    
##########################################
########## INITIALIZE IRIS CLIENT ########
##########################################
        
client = Client(center) 


################################################
######### Split and Read catalog file ##########
################################################
if flag_split:
    nlloc_list=sw.split_nlloc(nlloc_file,event_max,prefix='split',output_dir=nlloc_dir_tmp)
else:
    index_file=np.arange(file_range[0],int(file_range[1])+1)
    nlloc_list=['./'+nlloc_dir_tmp+'/'+'split_%03i.nlloc'%x for x in index_file]

