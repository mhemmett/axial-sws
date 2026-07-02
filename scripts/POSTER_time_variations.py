#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:01:25 2018

@author: baillard
"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
from obspy import UTCDateTime
import glob
import matplotlib.gridspec as gridspec

import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol 
from matplotlib.text import Text



def fit_ax(ax,xhours):
    starteruption_time=UTCDateTime(2015,4,24,6)
    mdates_eruption=swm.obspytime2matplotlib(starteruption_time)[0]
    xdatenum=xhours/24+mdates_eruption
    xticklabels=[Text(x,0,str(y)) for (x,y) in zip(xdatenum,xhours)]
    ax.set_xticks(xdatenum)
    ax.set_xticklabels(xticklabels)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=0 ,ha='center')
    ax.set_xlim([xdatenum[0],xdatenum[-1]])
    
    return ax

### Parameters

stations=['AXAS2','AXAS1','AXEC1','AXEC2','AXEC3','AXID1','AXCC1']
stations=['AXAS2','AXAS1','AXEC1','AXEC2','AXEC3','AXID1','AXCC1']
plt.ioff()

for station in stations:

    pickle_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat'

    pickle_files=glob.glob(pickle_dir+'/'+station+'*pickle')
    pickle_files.sort()
    Cat=swm.read_pickle(pickle_files)
    xhours=np.arange(-24,25,12)
    
    ### Selection of obss
    
    NCat=Cat.select(lambda_select=['rec','max'],lambda_lag=[0,30])
    
    ### Define Times
    starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
    x_start_zoom=UTCDateTime(2015,4,22,6) # Nooner and Chadwick 2016
    x_end_zoom=UTCDateTime(2015,4,26,6)
    x_start=UTCDateTime(2015,1,24,6) # Nooner and Chadwick 2016
    x_end=UTCDateTime(2015,7,24,6)
    
    ### Plots
    
    plt.close('all')

    
    ### Define subplots
    fig=plt.figure(figsize=(12,3))
    gs1 = gridspec.GridSpec(1, 2,
                           width_ratios=[3,1],left=0.05, right=0.49, top=0.99, bottom=0.3,wspace=0.01)
    gs2 = gridspec.GridSpec(1, 2,
                           width_ratios=[3,1],left=0.55, right=0.99, top=0.99, bottom=0.3,wspace=0.01)
    
    ax_mesh_lag= plt.subplot(gs1[0])
    ax_mesh_lagz= plt.subplot(gs1[1],sharey=ax_mesh_lag)
    
    ax_mesh_angle= plt.subplot(gs2[0])
    ax_mesh_anglez= plt.subplot(gs2[1],sharey=ax_mesh_angle)
    
    
    ### Plot large
    NCat.plot_movehist2d_time('lag',ax=ax_mesh_lag,level=2,y_width=2,y_over=0.95,mode='imshow_window',
                              x_start=x_start,x_end=x_end,x_width=5,x_over=0.95,norm_y=True,smooth=True)
    NCat.plot_movehist2d_time('angle',ax=ax_mesh_angle,level=2,y_width=np.pi/20,y_over=0.95,mode='imshow_window',
                              x_start=x_start,x_end=x_end,x_width=5,x_over=0.95,norm_y=True,smooth=True)
    
    ax_mesh_lag.set_ylabel('Lag [samples]')
    
    
    #### Plot zoom
    
    NCat.plot_movehist2d_time('lag',ax=ax_mesh_lagz,level=2,y_width=2,y_over=0.95,mode='pcolor_sample',
                              x_start=x_start_zoom,x_end=x_end_zoom,x_width=100,x_over=0,norm_y=True,smooth=True)
    
    fit_ax(ax_mesh_lagz,xhours)
    
    NCat.plot_movehist2d_time('angle',ax=ax_mesh_anglez,level=2,y_width=np.pi/20,y_over=0.95,mode='pcolor_sample',
                              x_start=x_start_zoom,x_end=x_end_zoom,x_width=100,x_over=0,norm_y=True,smooth=True)
    
    
    fit_ax(ax_mesh_anglez,xhours)
    
    
    plt.setp(ax_mesh_lagz.get_yticklabels(), visible=False)
    plt.setp(ax_mesh_anglez.get_yticklabels(), visible=False)
    
    ax_mesh_lagz.set_ylabel('')
    ax_mesh_anglez.set_ylabel('')
    ax_mesh_anglez.set_xlabel('Hours')
    ax_mesh_lagz.set_xlabel('Hours')
    ax_mesh_lag.set_xlabel('Dates')
    ax_mesh_angle.set_xlabel('Dates')
    ax_mesh_angle.set_ylabel('Fast dir. [rad]')
    ax_mesh_lagz.minorticks_on()
    ax_mesh_anglez.minorticks_on()
    ax_mesh_anglez.tick_params(axis='y',which='minor',bottom='off')
    ax_mesh_lagz.tick_params(axis='y',which='minor',bottom='off')
    
    ax_mesh_lag.minorticks_on()
    ax_mesh_angle.minorticks_on()
    ax_mesh_angle.tick_params(axis='x',which='minor',bottom='off')
    ax_mesh_lag.tick_params(axis='x',which='minor',bottom='off')
    
    
    plt.savefig('POSTER_time_variations_%s.pdf'%(station),format='pdf',quality=300)

#NCat.plot_movehist2d_time('angle',level=2,y_width=np.pi/20,mode='imshow_window',x_width=10,x_over=0.95,norm_y=True,smooth=False)
#NCat.plot_movehist2d_time('lag',level=2,y_width=2,mode='imshow_window',x_width=0.1,x_over=0,norm_y=True,smooth=False)
#New_Cat.plot_movehist2d_time('angle',level=2,y_width=np.pi/20,mode='pcolor_sample',x_width=500,x_over=0.95,norm_y=True,smooth=False)
#



