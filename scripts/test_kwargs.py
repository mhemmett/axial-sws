#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:16:38 2019

@author: baillard
"""


from obspy.io.nlloc.core import read_nlloc_hyp
import sws_methods as swm
import general.plotwaveform as pltwf
import sys
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy
import os

from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec


#%%

def plot_movehisto2d(x,y,
                x_label='X',y_label='Y',title='',ax=None,
                **movehisto2d_kwargs):
    
    x_start=movehisto2d_kwargs.get('x_start')
    movehisto2d_kwargs['x_start']=4
    
    return x_start,movehisto2d_kwargs



plot_movehisto2d(x,y,x_label='X',y_label='Y',title='',ax=None,x_start=2)