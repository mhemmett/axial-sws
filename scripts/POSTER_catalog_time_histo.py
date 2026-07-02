#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:30:56 2018

@author: baillard
"""

from obspy import read_events
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import general.util as gutil
import matplotlib.dates as mdates
import datetime as dt
import general.GMT as ggmt

#### Functions

file_level='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/PRESSURE/2013-2015-LTplot.txt'

def read_levels(file_level):
    levels=[]
    times=[]
    with open(file_level,'rt') as fic:
        next(fic)
        lines = fic.readlines()
        for line in lines:
            try:
                date,time_str,level=line.split()
                month,day,year=date.split('/')
            except:
                pass
            time=UTCDateTime(year+'/'+month+'/'+day+'T'+time_str)
            level=float(level)
            
            levels.append(level)
            times.append(time)
    
    return (times,levels)
            



def plot_vlines(x_poss,ax_in,marker_color='r',line_color='w',markersize=20,lw=2,mec='k'):
    y_min,y_max=ax_in.get_ylim()
    y_center=(y_min+y_max)/2

    x_poss=gutil.tolist(x_poss)
    
    if marker_color is 'variable':
        marker_color=ggmt.data2rgb(range(len(x_poss)))
    else:
        marker_color=[marker_color]*len(x_poss)
    
    if line_color is 'variable':
        line_color=ggmt.data2rgb(range(len(x_poss)))
    else:
        line_color=[line_color]*len(x_poss)
    
    for idx,x_pos in enumerate(x_poss):
        ax_in.vlines(x_pos,y_min,y_max,colors=line_color[idx],linestyles='dotted',lw=lw)
        ax_in.plot(x_pos,y_center,'*',mfc=marker_color[idx],markersize=markersize,mec=mec)
    
    ax_in.set_ylim([y_min,y_max])
    return ax_in

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)





### Parameters
nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/CATALOG/AXIAL.HYP.FINAL_3D.nlloc'
otimes_file='nlloc_otimes.dat'

starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
enderuption_time=UTCDateTime(2015,5,19)


### Check if exists and load

if not os.path.isfile(otimes_file):
    cat = read_events(nlloc_file)
    otimes=[x.preferred_origin().time for x in cat.events]
    pickle.dump(otimes,open(otimes_file,'wb'))
else:
    otimes=pickle.load(open(otimes_file,'rb'))

### Convert to datetime
    
mtimes=gutil.obspytime2matplotlib(otimes)

start_time=UTCDateTime(2015,1,1,0,0,0)
start_bin=gutil.obspytime2matplotlib(UTCDateTime(2015,1,1,0,0,0))[0]
end_bin=gutil.obspytime2matplotlib(UTCDateTime(2015,9,1))[0]

x_bins=np.arange(start_bin,end_bin,0.04)

(level_obstimes,levels)=read_levels(file_level)
level_mtimes=gutil.obspytime2matplotlib(level_obstimes)


### Plot counts

plt.close('all')
fig,ax=plt.subplots()
[n_events,_,_]=ax.hist(mtimes,x_bins,color='k',edgecolor='none')
n_events=np.array([0]+list(n_events))
ax.xaxis_date()
fig.autofmt_xdate()
cum_events=np.cumsum(n_events)
cum_events=cum_events/cum_events[-1]*100
ax.set_ylabel('Events Count')
ax.set_xlabel('Date')
ax.set_xlim([start_bin,end_bin])

#### Plot cumulative

ax_cum = ax.twinx()
ax_cum.yaxis.tick_left()
ax_cum.set_ylim([0,120])
make_patch_spines_invisible(ax_cum)
ax_cum.spines["left"].set_visible(True)
ax_cum.plot(x_bins,cum_events,'r',lw=1)
ax_cum.yaxis.label.set_color('r')

ax_cum.set_ylabel('Cumalative Sum [%]',labelpad=-40)
ax_cum.yaxis.set_label_position('left')
ax_cum.tick_params(axis='y',direction="in", pad=-25, colors='r',labelleft="on")

#### Plot levels


ax_lev = ax.twinx()
make_patch_spines_invisible(ax_lev)
ax_lev.spines["right"].set_visible(True)
lcolor=[0,0,1]
ax_lev.plot(level_mtimes,levels,color=lcolor,lw=1)
ax_lev.yaxis.label.set_color(lcolor)
ax_lev.set_ylim([1,4.5])
ax_lev.set_ylabel('Change in Seafloor Elevation [m]')
ax_lev.tick_params(axis='y', colors=lcolor)



#### Plot lines

vline_times=[gutil.obspytime2matplotlib(starteruption_time)[0],
             gutil.obspytime2matplotlib(enderuption_time)[0]]

plot_vlines(vline_times,ax_lev,marker_color='w',line_color='k',lw=1,markersize=15,mec='k')
plt.tight_layout()

### Save

plt.savefig('POSTER_catalog_time_histo.pdf',format='pdf')

