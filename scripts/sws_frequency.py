#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:45:53 2018

@author: baillard
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

from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.io.nlloc.core import read_nlloc_hyp
from obspy.signal.polarization import particle_motion_odr,vidale_adapt,eigval
from obspy.taup import TauPyModel
from general.util import smooth_curve

from general.plotwaveform import SDS2streams,plot_event,getPickForArrival,getPicks,getPicksAndArrivals,plot_picks,stream2data,jurkevics
from general.plotwaveform import plot_window,plot_particle_motion,plot_particle_motion_stream
import shearwavesplit as sws
import general.plotwaveform as pltwf

from obspy.signal.util import _npts2nfft 
import scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

x=np.linspace(0,100,10000)


acc = lambda t: 5*scipy.sin(2*np.pi*2.0*t) + 5*scipy.sin(2*np.pi*10*t) + 2*scipy.random.random(len(t))

y = acc(x)
#y=np.cos(x)+np.cos(2*x)+np.cos(4.1*x)
plt.close('all')


data=np.column_stack((x,y))

dt=x[1]-x[0]

npts=_npts2nfft(1*len(x))


#class FREQobs(sws.SWSobs):
#    """
#    
#    """
    

def spectrum(y_array,dt,npts=None,flag_plot=False):
    """
    Compute spectrum
    
    Inputs
    ------
        data: np.array: Nx1 data array (y)
        npts: int: number of samples in frequency output, if None is determined
        flag_plot: Boolean
        
    Outputs
    ------
        freq,FFT: np.array: contains freq and power(abs(FFT))
        npts: int
    """
    
    y_array=np.asarray(y_array)
    time_array=np.linspace(0,dt*(len(y_array)-1),len(y_array))

    if npts is None:
        npts=_npts2nfft(len(y_array))
        
    npts=int(npts)
    
    ### Compute FFT
    
    FFTs=pow(abs(scipy.fftpack.fft(y_array, npts)), 2)
    freqs = scipy.fftpack.fftfreq(npts, dt)
    
    freq=freqs[0:int(npts/2)]
    FFT=FFTs[0:int(npts/2)]
    
    #### Plot if asked

    
    if flag_plot:
        fig,(ax_signal,ax_spectr)=plt.subplots(2,1)
        ax_signal.plot(time_array,y_array,'k')
        ax_signal.set_xlabel('Time or Samples')
        ax_spectr.plot(freq,FFT,'k')
        ax_spectr.set_xlabel('Frequency')
        
    ### Return
    
    return ([freq,FFT],npts)
        
def dominant_freq(y_array,dt,npts=None):
    """
    Retrieve dominant frequency from data
    
    Input
    -----
        data: np.array: Nx2 data array (time,y) 
    """
    ([freq,FFT],npts)=spectrum(y_array,dt,npts=npts)
    
    dom_freq=freq[np.argmax(FFT)]
    
    return dom_freq
    
        
plt.close('all')
([freq,FFT],npts)=spectrum(data[:,1],dt,flag_plot=True)
#rint(dominant_freq(data))
