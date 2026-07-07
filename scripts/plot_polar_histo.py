#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:22:41 2018

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from general.util import listindir,merge_pdfs
import shearwavesplit as sw
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

### Test plot

plt.close('all')

###### Parameters

pickle_path='/home/baillard/Dropbox/_Moi/Projects/Axial/PROG/SWS/pickles/'


#pickle_file='/home/baillard/Dropbox/_Moi/Projects/Axial/PROG/SWS/pickles/AXEC1_20150315_013012.pickle'

##### Read pickle
#
#pickle_files=listindir(pickle_path,regexp=[''],full=True)
#
#Obs_list=[]
#
#for pickle_file in pickle_files:
#    
#    Obs=pickle.load(open(pickle_file,'rb'))
#    NObs=sw.SWSobs()
#    NObs.__dict__ = Obs.__dict__.copy()
#    Obs_list.append(NObs)


pdf_list=[]

flag_save=False
for station_code in ['AXAS1', 'AXAS2','AXCC1','AXEC1','AXEC2','AXEC3','AXID1']:
#for station_code in ['AXCC1']:
    ### Select for specific station
    
    Obs_sta=[Obs for Obs in Obs_list if Obs.station_code==station_code]
    
    
    ### Get arrays
    
    lags=np.array([Obs.lag for Obs in Obs_sta])
    angles=np.array([Obs.angle for Obs in Obs_sta])
    ang_errors=np.array([Obs.angle_error for Obs in Obs_sta])
    
    #### Define lag windows for polar plots 
    
    window_lag=5
    step_lag=4
    beg_lags=np.arange(0,20,step_lag)
    end_lags=beg_lags+window_lag
    windows=list(zip(beg_lags,end_lags))
    
    ### Apply filters for selecting angles
    
    bool_errors=(ang_errors!=999) 
    
    #### Start plotting and selecting
    
 
    fig,ax_list=plt.subplots(1,len(windows)+1,subplot_kw=dict(projection='polar'),figsize=[ 16.55,   3.57])
    fig.suptitle(station_code)
    sel_angles=angles[bool_errors]
    total=len(sel_angles)
    sw.plot_polar_histo(sel_angles,width_bin_deg=10,ax_polar=ax_list[0],facecolor='r',edgecolor='k')
    ax_list[0].set_title('All\n')
    kk=0
    for start,end in windows:
        kk+=1
        bool_lag=((lags>=start) & (lags<=end))
        
        bool_select=bool_lag & bool_errors
        
        sel_angles=angles[bool_select]
        
        ### Compute percentage
        
        perc=(len(sel_angles)/total) *100
        sw.plot_polar_histo(sel_angles,width_bin_deg=10,ax_polar=ax_list[kk])
        ax=ax_list[kk]
        ax.set_title('$ %i\leq lag \leq%i \ (%i )$\n'%(start,end,perc))
        #ax.set_suptitle('sdf')
        
    fig.subplots_adjust(wspace=0.05)
    plt.tight_layout()
    
    ### Plot histograms of lags 
    fig,ax=plt.subplots(subplot_kw=dict(projection='polar'))
    sel_angles=angles[bool_errors]
    sel_lags=lags[bool_errors]
    sel_angles=np.concatenate((sel_angles,sel_angles+np.pi))
    sel_lags=np.concatenate((sel_lags,sel_lags))
    sel_angles=sel_angles % (2*np.pi)
    ax.plot(sel_angles,sel_lags,'or',markersize=4,alpha=0.5,mec='none')
    plt.pause(3)
    continue
    sys.exit()
    #### save fig
    
    pdf_list.append(station_code+'.pdf')
    plt.savefig(station_code+'.pdf',format='pdf')
    
### Merge_pfs
if flag_save:
    merge_pdfs(pdf_list,'new.pdf')
        
#### Check lags distribution

#lag_bins=np.linspace(0,21,22)-0.5
#lag_centers=lag_bins[:-1]+0.5
#
#nlags,_=np.histogram(lags,bins=lag_bins)
#
#plt.close('all')
#plt.bar(lag_centers,nlags,width=1)
#
#sw.plot_polar_histo(sel_angles,width_bin_deg=10)