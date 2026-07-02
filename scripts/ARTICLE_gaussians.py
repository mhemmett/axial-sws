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
import test_distribution as tdist
from scipy.interpolate import interp1d
import test_gaussian_fit as gfit
import scipy

import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol 
from matplotlib.text import Text


### Parameters

plt.close('all')
station_list=['AXCC1','AXEC1','AXEC2','AXEC3','AXAS2','AXAS1','AXID1']
#station_list=['AXEC2']
#station_name=station_list[0]
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat'
suffix='.clean.cat.pickle'
lag_mode='ms'
angle_mode='azimuth'
sampling_rate=200
flag_save=True
flag_plot=True

plt.ioff()

### Define dictionaries with periods

Tstart_monitoring=UTCDateTime(2015,1,1)
Tend_monitoring=UTCDateTime(2018,1,1)
Tstart=UTCDateTime(2015,1,24,6)
TA=UTCDateTime(2015,3,1,0)
TB=UTCDateTime(2015,5,1,0)
TE1=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
TE2=UTCDateTime(2015,5,19) # End eruption
Tend=UTCDateTime(2015,7,24,6)

list_period_AXCC1=[
        [Tstart,TA],
        [UTCDateTime(2015,4,27,0,0,0),Tend]
        ]

list_period_AXID1=[
        [Tstart,TE1],
        [TE1,TE1+3600*5],
        [TE1+3600*5,TE1+3600*8]
        ]

list_period_AXEC1=[
        [Tstart,TE1],
        [TE1,TE1+3600*1],
        [TE1+3600*1,Tend]
        ]

list_period_AXEC3=[
        [Tstart,TE1],
        [TE1,TE1+3600*1],
        [TE1+3600*1,TE1+3600*5],
        [TE1+3600*5,Tend]
        ]

list_period_AXAS2=[
        [Tstart,TE1],
        [TE1,TE1+3600*2],
        [TE1+3600*2,Tend]
        ]

list_period_AXEC2=[
        [Tstart,TE1],
        [TE1,TE1+3600*1],
        [TE1+3600*1,TE2],
        [TE2,Tend],
        ]

list_period_AXAS1=list_period_AXEC1

list_period_eruption=[
        [Tstart_monitoring,TE1],
        [TE1,TE2],
        [TE2,Tend_monitoring]
        ]

list_period_eruption=[
        [Tstart,TE1],
        [TE1,TE2],
        [TE2,Tend]
        ]

dic_stas={
        'AXCC1':{'periods':list_period_AXCC1},
        'AXEC1':{'periods':list_period_AXEC1},
        'AXEC2':{'periods':list_period_AXEC2},
        'AXEC3':{'periods':list_period_AXEC3},
        'AXAS1':{'periods':list_period_AXAS1},
        'AXAS2':{'periods':list_period_AXAS2},
        'AXID1':{'periods':list_period_AXID1}
        }

dic_stas={
        'AXCC1':{'periods':list_period_eruption},
        'AXEC1':{'periods':list_period_eruption},
        'AXEC2':{'periods':list_period_eruption},
        'AXEC3':{'periods':list_period_eruption},
        'AXAS1':{'periods':list_period_eruption},
        'AXAS2':{'periods':list_period_eruption},
        'AXID1':{'periods':list_period_eruption}
        }

### Start process

########################
### Loop on Stations ###
########################

for station_name in station_list:

    dic_sta=dic_stas[station_name]
    
    ### Read catalog
        
    file_in= os.path.join(cat_dir,'')+station_name+suffix
    Cat=swm.read_pickle(file_in)
    
    #########################
    ### Loop over periods ###
    #########################
    
    for k_per,period in enumerate(dic_sta['periods']):
        plt.close('all')
        
        print(period)
        SCat=Cat.select(obs_stime=period)
        
        ### Build file_name
        
        fig_pdf='ARTICLE_%s_period_%02i.pdf'%(station_name,k_per)
        fig_png='ARTICLE_%s_period_%02i.png'%(station_name,k_per)
     
        ### Read and convert
            
        elems_dic=SCat.get_dic()
        lags=elems_dic['lag']  
        angles=elems_dic['fast']*180/np.pi
        
        if lag_mode=='ms':
            lags=lags/sampling_rate*1000
        elif lag_mode=='s':
            lags=lags/sampling_rate
            
        if angle_mode=='azimuth':
            angles=swm.trigo2azimuth(angles)
        
        ### Prepare plot
            
        fig,[ax_lag,ax_angle]=plt.subplots(2,1)
        dic_lag={'x_start':0,'x_end':150,'x_width':10,'x_over':0.5}
        dic_angle={'x_start':None,'x_end':None,'x_width':15,'x_over':0.5}
        dic_param={'lag':dic_lag,'angle':dic_angle}
        
        ############################
        ### Loop over parameters ###
        ############################
        
        k_plot=-1
        for y,param in zip([lags,angles],['lag','angle']):
            
            x_start=dic_param[param]['x_start']
            x_end=dic_param[param]['x_end']
            x_width=dic_param[param]['x_width']
            x_over=dic_param[param]['x_over']
            
            ### Extend to take into account periodicity
            if param=='angle':
                y=swm.extend_periodic_array(y,[0,180],perc_ext=10)
                
            ### Bins
            
            if x_start is None:
                x_start=np.min(y)
                
            if x_end is None:
                x_end=np.max(y)
            
            (x_bins,x_lefts,x_rights,counts)=swm.centered_histo(y,
            x_start=x_start,x_end=x_end,x_width=x_width,x_over=x_over,flag_plot=False)
        
            ### Interpolate
            f_inter = scipy.interpolate.interp1d(x_bins, counts, kind='quadratic')
            n_x_bins=np.linspace(x_bins[0],x_bins[-1],500,endpoint=True)
            n_counts=f_inter(n_x_bins)
            
            ### Decompose
            if param=='lag':
                swm.decompose_gaussians(n_x_bins,n_counts,ax=ax_lag,flag_sort=True,flag_plot=flag_plot)
            else:
                swm.decompose_gaussians(n_x_bins,n_counts,ax=ax_angle,flag_sort=True,flag_plot=flag_plot)
                
        ### Cosmetic
                
        ax_angle.set_xlabel('$\Phi$ [°]')
        ax_angle.set_xlim([0,180])
        ax_angle.set_ylabel('Obs.')
        
        ax_lag.set_xlabel('$\delta t$ [ms]')
        ax_lag.set_xlim([0,dic_param['lag']['x_end']])
        ax_lag.set_ylabel('Obs.')
        
        format_str='%Y-%m-%dT%H:%M:%S'
        ax_lag.set_title('%s Period: %s -> %s'%(station_name,
                                                period[0].strftime(format_str),period[1].strftime(format_str)),fontsize=10)
            
        ### Save fig
        
        if flag_save:
            plt.tight_layout()
            plt.savefig(fig_pdf,format='pdf',dpi=300,quality=100,bbox_inches='tight',frameon=False)
            plt.savefig(fig_png,format='png',dpi=300,quality=100,bbox_inches='tight',frameon=False)
