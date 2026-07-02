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
import os,pickle,time,glob
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



## Paths
#nlloc_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial_EQ/PROG/NLLOC_MODELS/NLLOC_3H/loc/AXIAL.sum'
nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/CATALOG/AXIAL.PHASE.FINAL_3D_V2.nlloc'

################################################
######### Split and Read catalog file ##########
################################################

index_file=np.arange(file_range[0],int(file_range[1])+1)
nlloc_list=glob.glob('./'+nlloc_dir_tmp+'/'+'split_*.nlloc')

tot_obs=0
k_file=1
for nlloc_file in nlloc_list:
    print('Processing %i'%k_file)
    k_file+=1
    Cat=read_nlloc_hyp(nlloc_file)
    
    
    for i_event in range(len(Cat)):

        single_event=Cat[i_event]
        
        #### Select stations that have both P and S picks 
        
        P_list=pltwf.getPicksAndArrivals(single_event,network_code=None,station_code=None,phase_hint='P')
        S_list=pltwf.getPicksAndArrivals(single_event,network_code=None,station_code=None,phase_hint='S')
        
        PS_list=[[P_pair,S_pair] for P_pair in P_list for S_pair in S_list if P_pair[0].waveform_id.station_code==S_pair[0].waveform_id.station_code]
        
        #################################
        tot_obs+=len(PS_list)
        
    
print(tot_obs)