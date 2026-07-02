#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:47:50 2018

@author: baillard

Plot to get the polar plots for each eruiptions
"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
from obspy import UTCDateTime
import glob

import shearwavesplit as sws
import sws_methods as swm
from general import GMT as ggmt
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol 

from general.util import figs2pdf



### Parameters

station='AXEC1'
#stations=['AXAS2','AXAS1','AXEC1','AXEC2','AXEC3','AXID1','AXCC1']
stations=['AXAS2','AXAS1','AXEC1','AXEC2','AXEC3','AXID1','AXCC1']



start_time=UTCDateTime(2015,1,1)
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
#starteruption_time=UTCDateTime(2015,4,23) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
end_time=UTCDateTime(2018,1,1)

pickle_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat'



#### Define time intervals

time_inters=[
        [start_time,starteruption_time],
        [starteruption_time,enderuption_time],
        [enderuption_time,end_time]]

#### Create colors for all stations
rgb_list=ggmt.data2rgb(np.linspace(0,1,len(stations)),cmap=plt.cm.get_cmap('jet'))

for qq,station in enumerate(stations):
    
    if station!='AXAS1':
        continue
    
    pickle_files=glob.glob(pickle_dir+'/'+station+'*pickle')
    pickle_files.sort()
    Cat=swm.read_pickle(pickle_files)
    plt.ioff()
    
    ##### Select Catalogs for each time interval
    
    NCats=[]
    for time_inter in time_inters: 
        NCats.append(Cat.select(lambda_select=['rec','max'],lambda_lag=[0,30],obs_stime=time_inter))
        
    ### Plot 
        
    plt.close('all')
    
    for kk,NCat in enumerate(NCats):   
        if kk==2:
            ax_polar=NCat.plot_polar_fastlag(colorbar=False,x_over=0.8,y_over=0.8,x_end=30,smooth=True)
            ax_polar.set_xlabel('')
            ax_polar.tick_params(axis='x', which='major', pad=0)
            fig=plt.gcf()
            fig.set_size_inches([ 3.74,  2.8 ])
            plt.tight_layout()
        else:
            (ax_main,ax_insets,ax_middle,ax_top)=NCat.plot_polar_fastlag_multi(num_ax=6,theta_rot=0,x_over=0.8,y_over=0.8,x_end=30,smooth=True)
    
    for kk,NCat in enumerate(NCats): 
        if kk==2:
            NCat.plot_polar_histofast(edgecolor='k',facecolor=rgb_list[qq][0:3],lw=0.2)
            fig=plt.gcf()
            fig.set_size_inches([ 3.67,  2.63 ])
            plt.tight_layout()
        else:
            (ax_main,ax_insets,ax_middle,ax_top)=NCat.plot_polar_histofast_multi(edgecolor='k',facecolor=rgb_list[qq][0:3],lw=0.2,norm=False,num_ax=6)
        
        
    ### Save plots
        
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        if fig_num<=3:
            name=station+'_fastlag_%02i.pdf'%fig_num
            plt.savefig(name,format='pdf',quality=100,dpi=300)
        else: 
            name=station+'_polarhisto_%02i.pdf'%fig_num
            plt.savefig(name,format='pdf',quality=100,dpi=300)
            
#    if qq==0:
#        sys.exit()
    
        
