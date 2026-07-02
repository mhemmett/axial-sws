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
#stations=['AXID1']

#start_time=UTCDateTime(2015,1,1)
#end_time=UTCDateTime(2018,1,1)
start_time=UTCDateTime(2015,1,24,6) # Nooner and Chadwick 2016
end_time=UTCDateTime(2015,7,24,6)

starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)

suffix='.clean.cat.pickle' # to choose the cleaned catalog (esp. for AXCC1 and AXID1)
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat'
lag_start=0
lag_end=150
lag_width=10
lag_mode='ms'
flag_density=False # Do you want the density plots
flag_histo=True # Polar plots
plt.ioff()

#### Define time intervals

time_inters=[
        [start_time,starteruption_time],
        [starteruption_time,enderuption_time],
        [enderuption_time,end_time]]

hypdist_inters=[[0,1],[1,2],[2,3],[0,999]]
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
    
        ### CReate list of axes
        
        fig_width=len(time_inters)*(flag_histo+flag_density)*2
        fig,ax_main=plt.subplots(   figsize=(fig_width, 3))
       
        #ax_main.axis('equal')
        ax_main.set_aspect("equal")
     
        ax_main.set_xlim([0,11])
        ax_main.set_ylim([-1.5,1.5])
        fig.canvas.draw()
        ax_main.axis('off')
        #
    
    
        #ax_main.axis('off')
        x_inset=0
        y_inset=0
        space_inset=0.4
        w_inset=4
        
        
        ### Density Plots
        
        if flag_density:
            counter=0
            for kk,NCat in enumerate(NCats):   
                ax_polar=get_ax_inset(ax_main,x_inset,y_inset,w_inset,height=None,projection='polar',alpha=1,visible_axis=True)
                x_inset+=(w_inset+space_inset)
                num_obs=len(NCat.obs)
                title=('%i (%.1f'%(num_obs,(num_obs/total_obs)*100))+'%)\n'
                ax_polar=NCat.plot_polar_fastlag(colorbar=False,y_width=8,x_over=0.95,y_over=0.95,
                                                 x_end=lag_end,x_start=lag_start,
                                                 lag_mode='ms',sampling_rate=200,x_width=lag_width,lagticks_step=50,
                                                 flag_resample=True,flag_filter=True,label_mode='azimuth',ax=ax_polar)
        
                ax_polar.set_title(title,fontsize=10)
                ax_polar.set_xlabel('')
               
          
                if counter!=0:
                    ax_polar.set_xticklabels([])
                    
                ax_polar.tick_params(axis='x', which='major', pad=0)
                counter+=1
        
        ### Histo plots
        if flag_histo:
            counter=0
            for kk,NCat in enumerate(NCats): 
                ax_polar=get_ax_inset(ax_main,x_inset,y_inset,w_inset,height=None,projection='polar',alpha=1,visible_axis=True)
                x_inset+=(w_inset+space_inset)
                num_obs=len(NCat.obs)
                title=('%i (%.1f'%(num_obs,(num_obs/total_obs)*100))+'%)\n'
                ax_polar=NCat.plot_polar_histofast(ax=ax_polar,edgecolor='k',facecolor=rgb_list[k_sta][0:3],lw=0.5,
                                                   label_mode='azimuth')
                ax_polar.set_title(title,fontsize=10,pad=-10)
                #if counter!=0:
                ax_polar.set_xticklabels([])
                ax_polar.set_xlabel('')
                counter+=1
            

        fig.suptitle(station_name,fontweight='bold',ha='center')
    
        figname_png='ARTICLE_histo_polar_ms_%s_%s.png'%(station_name,z_suffix)
        figname_pdf='ARTICLE_histo_polar_ms_%s_%s.pdf'%(station_name,z_suffix)
        plt.savefig(figname_png,format='png',bbox_inches='tight',quality=100,dpi=300,frameon=False)
        plt.savefig(figname_pdf,format='pdf',bbox_inches='tight',quality=100,dpi=300,frameon=False)
            

    
        
plt.ion()