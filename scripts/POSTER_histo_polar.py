#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:56:25 2018

@author: baillard
"""

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
    
    pickle_files=glob.glob(pickle_dir+'/'+station+'*pickle')
    pickle_files.sort()
    Cat=swm.read_pickle(pickle_files)
    plt.ioff()
    
    ##### Select Catalogs for each time interval
    
    NCats=[]
    total_obs=0
    for time_inter in time_inters: 
        NCat=Cat.select(lambda_select=['rec','max'],lambda_lag=[0,30],obs_stime=time_inter)
        NCats.append(NCat)
        total_obs+=len(NCat.obs)
        
    ### Plot 
        
    plt.close('all')


    ### create list of axes
    
    
    fig,ax_main=plt.subplots(   figsize=(13, 3))

   
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
    space_inset=0.3
    w_inset=2
    
    counter=0
    for kk,NCat in enumerate(NCats):   
        ax_polar=get_ax_inset(ax_main,x_inset,y_inset,w_inset,height=None,projection='polar',alpha=1,visible_axis=True)
        x_inset+=(w_inset+space_inset)
        num_obs=len(NCat.obs)
        title=('%i (%.0f'%(num_obs,(num_obs/total_obs)*100))+'%)\n'
        ax_polar=NCat.plot_polar_fastlag(ax=ax_polar,colorbar=False,x_over=0.8,y_over=0.8,x_end=30,smooth=True)
        ax_polar.set_title(title,fontsize=10)
        ax_polar.set_xlabel('')
  
        if counter!=0:
            ax_polar.set_xticklabels([])
            
        ax_polar.tick_params(axis='x', which='major', pad=0)
#        fig=plt.gcf()
#        fig.set_size_inches([ 3.74,  2.8 ])
#        plt.tight_layout()
   
        counter+=1

    for kk,NCat in enumerate(NCats): 
        ax_polar=get_ax_inset(ax_main,x_inset,y_inset,w_inset,height=None,projection='polar',alpha=1,visible_axis=True)
        x_inset+=(w_inset+space_inset)
        num_obs=len(NCat.obs)
        title=('%i (%.0f'%(num_obs,(num_obs/total_obs)*100))+'%)\n'
        ax_polar=NCat.plot_polar_histofast(ax=ax_polar,edgecolor='k',facecolor=rgb_list[qq][0:3],lw=0.2)
        ax_polar.set_title(title,fontsize=10)
        ax_polar.set_xticklabels([])
        #ax_polar.tick_params(axis='x', which='major', pad=0)
#
#        fig=plt.gcf()
#        fig.set_size_inches([ 3.74,  2.8 ])
#        plt.tight_layout()
#        counter+=1

#        

    ### Save plots
    

        
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        if fig_num<=3:
            name=station+'_fastlag_%02i.pdf'%fig_num
            plt.savefig(name,format='pdf',bbox_inches='tight',quality=100,dpi=300)
        else: 
            name=station+'_polarhisto_%02i.pdf'%fig_num
            plt.savefig(name,format='pdf',bbox_inches='tight',quality=100,dpi=300)
            
#    if qq==0:
#        sys.exit()
    
        
