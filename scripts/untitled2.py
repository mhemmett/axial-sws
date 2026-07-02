#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:45:03 2018

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from general.util import listindir,merge_pdfs
import shearwavesplit as sw
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from general.GMT import data2rgb,get_cax

### Test plot

plt.close('all')

pdf_list=[]
for station_code in ['AXAS1', 'AXAS2','AXCC1','AXEC1','AXEC2','AXEC3','AXID1']:
    Obs_sta=[Obs for Obs in Obs_list if Obs.station_code==station_code]
    
    ### Get arrays
    
    p_times=np.array([Obs.p_time for Obs in Obs_sta])
    s_times=np.array([Obs.s_time for Obs in Obs_sta])
    ps_delays=s_times-p_times
    ps_delays=ps_delays.astype(float)
    lags=np.array([Obs.lag for Obs in Obs_sta])
    angles=np.array([Obs.angle for Obs in Obs_sta])
    ang_errors=np.array([Obs.angle_error for Obs in Obs_sta])
    
    #plt.hist2d(ps_delays,lags,[40,20],vmax=30)
    #plt.plot(ps_delays,lags,'or',markersize=2)
    
    bool_errors=(ang_errors!=999) 
    #### Filter
    
    bool_lags=(lags<18)
    
    #### 
    
    angles=angles[bool_lags]
    lags=lags[bool_lags]
    
    
    ### Define histo edges
    angle_step=5*np.pi/180
    lag_step=4
    lag_max=20
    
    angle_edges=np.linspace(0,2*np.pi,int(2*np.pi/angle_step)+1)
    lag_edges=np.linspace(0,lag_max,int(lag_max/lag_step)+1)
    
    #### Duplicate and add 180°
    
    angles=np.concatenate((angles,angles+np.pi))
    lags=np.concatenate((lags,lags))
    
    #### Make sure angles are inside 0 360°
    
    angles=angles % (2*np.pi)
    
    ### Compute histograms
    
    histo_array,_,_=np.histogram2d(angles,lags,bins=[angle_edges,lag_edges])
    fig,ax_polar=plt.subplots(subplot_kw=dict(projection='polar'))
    
    vmin=np.min(histo_array)
    vmax=np.max(histo_array)
    width_bar=np.diff(angle_edges)[0]
    
    for i_angle in range(len(angle_edges)-1):
        for i_lag in range(len(lag_edges)-1):
            histo_val=histo_array[i_angle,i_lag]
            facecolor=data2rgb(histo_val,vmin=vmin,vmax=vmax)
            bottom=lag_edges[i_lag]
            top=lag_edges[i_lag+1]
            left_bar=angle_edges[i_angle]
            ax_polar.bar(left_bar,top,bottom=bottom, align='edge',width=width_bar,facecolor=facecolor,edgecolor='none',lw=0.2)
    
    ### Refine plot
    
    ax_polar.yaxis.label.set_color('white')
    ax_polar.tick_params(axis='y', colors='white')
    ax_polar.set_ylim([0,20])
    ax_polar.set_yticks(lag_edges)
    ax_polar.set_title(station_code+'\n')
    
    import matplotlib as mpl
    
    cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    
    
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    #
    cb.set_label('N Obs.')
    name=station_code+'_angle_lag.pdf'
    pdf_list.append(name)
    plt.savefig(name,format='pdf')


merge_pdfs(pdf_list,'new_angle_lag.pdf')

#
#fig,ax=plt.subplots(subplot_kw=dict(projection='polar'))
#
#ax.bar(0,5,bottom=1)