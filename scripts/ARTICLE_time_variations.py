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
from matplotlib.ticker import AutoMinorLocator



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


station_list=['AXCC1','AXEC1','AXEC2','AXEC3','AXAS2','AXAS1','AXID1']
#station_list=['AXEC3']
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat'
suffix='.clean.cat.pickle'
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
x_start_zoom=starteruption_time-12*3600 # Nooner and Chadwick 2016
x_end_zoom=starteruption_time+12*3600
x_start=UTCDateTime(2015,1,24,6) # Nooner and Chadwick 2016
#x_end=UTCDateTime(2015,7,24,6)
x_end=UTCDateTime(2017,2,1,1)
lag_start=0
lag_end=150
lag_width=10
#time_width_unzoom=1000
time_width_unzoom=200
time_width_zoom=200
lag_mode='ms'
lag_label='$\delta t$ [%s]'%lag_mode
fast_start=0
fast_end=180
xhours=np.arange(-12,13,6)
minor_locator_angle = AutoMinorLocator(5)
minor_locator_lag = AutoMinorLocator(5)

plt.ioff()

### START PROCESS



for station_name in station_list:

    ### Define file
    
    file_in= os.path.join(cat_dir,'')+station_name+suffix

    ### Read file for each station
    
    Cat=swm.read_pickle(file_in)

    
    ### Selection of obss
    
    NCat=Cat

    ### Plot
    
    plt.close('all')
    
    ### Define subplots properly
    
    fig=plt.figure(figsize=(15,3))
    shift=0.1
    gs1 = gridspec.GridSpec(1, 2,
                           width_ratios=[3,1],left=0.05, right=0.5-shift/2, top=0.99, bottom=0.3,wspace=0.3)
    gs2 = gridspec.GridSpec(1, 2,
                           width_ratios=[3,1],left=0.5+shift/2, right=0.99, top=0.99, bottom=0.3,wspace=0.3)
    
    ax_mesh_lag= plt.subplot(gs1[0])
    ax_mesh_lagz= plt.subplot(gs1[1],sharey=ax_mesh_lag)
    
    ax_mesh_angle= plt.subplot(gs2[0])
    ax_mesh_anglez= plt.subplot(gs2[1],sharey=ax_mesh_angle)
    
    
    ### Plot large
    
    NCat.plot_movehisto2d_time('lag',minlambda_select='min', 
                              window_unit='hour',window_bottom=0.001,
                              angle_mode='trigo',lag_mode='ms',sampling_rate=200,
                              x_label='X',y_label='Y',title='',ax=ax_mesh_lag,
                              x_over=0.95,y_over=0.95,x_mode='sample',
                              x_start=x_start,x_end=x_end,x_width=time_width_unzoom,
                              y_start=lag_start,y_end=lag_end,y_width=lag_width,
                              flag_y_norm=True,flag_resample=True,flag_filter=True)
    
    ax_mesh_lag.set_ylabel(lag_label)
    

    NCat.plot_movehisto2d_time('fast',minlambda_select='min', 
                              window_unit=None,window_bottom=0.1,angle_mode='azimuth',
                              x_label='X',y_label='Y',title='',ax=ax_mesh_angle,
                              x_over=0.95,y_over=0.95,x_mode='sample',
                              x_start=x_start,x_end=x_end,x_width=time_width_unzoom,
                              y_start=fast_start,y_end=fast_end,y_width=10,y_cycle=[fast_start,fast_end],
                              flag_y_norm=True,flag_resample=True,flag_filter=True)
            
    

    ax_mesh_angle.set_ylabel('$\Phi$ [°]')
    
   
    
    #### Plot zoom
    
    NCat.plot_movehisto2d_time('lag',minlambda_select='min', 
                              window_unit='minute',window_bottom=1,
                              angle_mode='trigo',lag_mode='ms',sampling_rate=200,
                              x_label='X',y_label='Y',title='',ax=ax_mesh_lagz,
                              x_over=0.95,y_over=0.95,x_mode='sample',
                              x_start=x_start_zoom,x_end=x_end_zoom,x_width=time_width_zoom,
                              y_start=lag_start,y_end=lag_end,y_width=lag_width,
                              flag_y_norm=True,flag_resample=True,flag_filter=True)
    
    
    fit_ax(ax_mesh_lagz,xhours)
    
    NCat.plot_movehisto2d_time('fast',minlambda_select='min', 
                              window_unit=None,window_bottom=1,angle_mode='azimuth',
                              x_label='X',y_label='Y',title='',ax=ax_mesh_anglez,
                              x_over=0.95,y_over=0.95,x_mode='sample',
                              x_start=x_start_zoom,x_end=x_end_zoom,x_width=time_width_zoom,
                              y_start=fast_start,y_end=fast_end,y_width=10,y_cycle=[fast_start,fast_end],
                              flag_y_norm=True,flag_resample=True,flag_filter=True)
    
    fit_ax(ax_mesh_anglez,xhours)
    

    
    ### Cosmetic
    
    
    plt.setp(ax_mesh_lagz.get_yticklabels(), visible=False)
    plt.setp(ax_mesh_anglez.get_yticklabels(), visible=False)
    
    ax_mesh_lagz.set_ylabel('')
    ax_mesh_anglez.set_ylabel('')
    ax_mesh_anglez.set_xlabel('Hours')
    ax_mesh_lagz.set_xlabel('Hours')
    ax_mesh_lag.set_xlabel('Dates')
    ax_mesh_angle.set_xlabel('Dates')
    ax_mesh_lagz.minorticks_on()
    ax_mesh_anglez.minorticks_on()
    ax_mesh_lagz.yaxis.set_minor_locator(minor_locator_lag)
    ax_mesh_anglez.yaxis.set_minor_locator(minor_locator_angle)
    ax_mesh_anglez.tick_params(axis='y',which='minor',bottom='off')
    ax_mesh_lagz.tick_params(axis='y',which='minor',bottom='off')
    
    ax_mesh_lag.minorticks_on()
    ax_mesh_angle.minorticks_on()
    ax_mesh_lag.yaxis.set_minor_locator(minor_locator_lag)
    ax_mesh_angle.yaxis.set_minor_locator(minor_locator_angle)

    ax_mesh_angle.tick_params(axis='x',which='minor',bottom='off')
    ax_mesh_lag.tick_params(axis='x',which='minor',bottom='off')
    
    ax_mesh_lag.text(0.01, 0.99,'%s'%station_name, horizontalalignment='left',
          verticalalignment='top', transform=ax_mesh_lag.transAxes,color='w',fontweight='bold')
    
    ax_mesh_angle.text(0.01, 0.99,'%s'%station_name, horizontalalignment='left',
          verticalalignment='top', transform=ax_mesh_angle.transAxes,color='w',fontweight='bold')
    

    plt.savefig('ARTICLE_time_variations_samples_ms_%i_%s.pdf'%(time_width_unzoom,station_name),format='pdf',quality=300,frameon=False,bbox_inches='tight')
    plt.savefig('ARTICLE_time_variations_samples_ms_%i_%s.png'%(time_width_unzoom,station_name),format='png',quality=300,frameon=False,bbox_inches='tight')
    

plt.ion()
#NCat.plot_movehist2d_time('angle',level=2,y_width=np.pi/20,mode='imshow_window',x_width=10,x_over=0.95,norm_y=True,smooth=False)
#NCat.plot_movehist2d_time('lag',level=2,y_width=2,mode='imshow_window',x_width=0.1,x_over=0,norm_y=True,smooth=False)
#New_Cat.plot_movehist2d_time('angle',level=2,y_width=np.pi/20,mode='pcolor_sample',x_width=500,x_over=0.95,norm_y=True,smooth=False)
#



