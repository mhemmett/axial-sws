#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:47:50 2018

@author: baillard

Program made to plot the density polar plots and the rose plots 
showing lags and fast directions for each of the station and for different time
periods

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

stations=['AXCC1','AXEC1','AXEC2','AXEC3','AXAS2','AXAS1','AXID1']
#stations=['AXAS2']

start_time=UTCDateTime(2015,1,1)
starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)
end_time=UTCDateTime(2018,1,1)
suffix='.clean.cat.pickle' # to choose the cleaned catalog (esp. for AXCC1 and AXID1)
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat'
fig_height=6
plt.ioff()

#### Define time intervals

time_inters=[
        [start_time,starteruption_time],
        [starteruption_time,enderuption_time],
        [enderuption_time,end_time]]

hypdist_inters=[[0,999]]

#### Create colors for all stations
rgb_list=ggmt.data2rgb(np.linspace(0,1,len(stations)),cmap=plt.cm.get_cmap('jet'))

######################
### Start Plotting ###

for k_sta,station_name in enumerate(stations):
    
    ### Read pickle into catalog
    
    file_in= os.path.join(cat_dir,'')+station_name+'.clean.cat.pickle'
    Cat=swm.read_pickle(file_in)
    
    #### Select Catalogs for each depth interval
    
    ZCats=[]
    total_obs=len(Cat.obs)

    for hyp_dist in hypdist_inters: 
        z_suffix='%03i_%03i_m'%(hyp_dist[0]*1000,hyp_dist[1]*1000)
        ZCat=Cat.select(lambda_select=['rms','min'],obs_hyp_dist=hyp_dist,lambda_lag=[0,30])
        ZCats.append(ZCat)
        print('Processing station %s for depth in [%.1f km ,%.1f km]'%(station_name,hyp_dist[0],hyp_dist[1]) )

        ##### Select Catalogs for each time interval
        
        NCats=[]
        total_obs_n=0
        for time_inter in time_inters: 
            NCat=ZCat.select(obs_stime=time_inter)
            NCats.append(NCat)
            total_obs_n+=len(NCat.obs)
            
        ### Plot 
            
        plt.close('all')
        fig,ax=plt.subplots(1,len(NCats),figsize=(len(NCats)*fig_height, fig_height))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=0.05, hspace=None)
        
        for kk,NCat in enumerate(NCats):   
    
            (ax_main,ax_insets,ax_middle,ax_top)=NCat.plot_polar_fastlag_multi(num_ax=6,
                                               colorbar=False,y_width=8,x_over=0.9,y_over=0.9,x_end=30,x_start=0,
                                               lagticks=np.arange(0,30,10),
                                         flag_resample=True,flag_filter=True,title='',label_mode='azimuth',ax=ax[kk])
     

        fig.suptitle(station_name,fontweight='bold')
    
        figname_png='ARTICLE_multi_density_%s_%s.png'%(station_name,z_suffix)
        figname_pdf='ARTICLE_multi_density_%s_%s.pdf'%(station_name,z_suffix)
        plt.savefig(figname_png,format='png',bbox_inches='tight',quality=100,dpi=300,frameon=False)
        plt.savefig(figname_pdf,format='pdf',bbox_inches='tight',quality=100,dpi=300,frameon=False)
        
            

    
        
plt.ion()