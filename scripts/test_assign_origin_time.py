#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:45:36 2019

@author: baillard
"""

from obspy.io.nlloc.core import read_nlloc_hyp
import sws_methods as swm
import general.plotwaveform as pltwf
import sys
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import glob

nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/SWS_nlloc/split_020.nlloc'
sws_file='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_1_cat/AXAS1.cat.pickle'

nlloc_list=glob.glob('/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/SWS_nlloc/split*')



### Build dictionnary with keys based on P times and value giving origin time

dic_o_time={}
counter=0
o_times=[]
p_times=[]
vpvss=[]

for nlloc_file in nlloc_list:
    #nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/SWS_nlloc/split_020.nlloc'
    print('process %s'%nlloc_file)
    Cat_nlloc=read_nlloc_hyp(nlloc_file)
    for i_event in range(len(Cat_nlloc)):
        single_event=Cat_nlloc[i_event]
        o_time=single_event.origins[0].time
        P_list=pltwf.getPicksAndArrivals(single_event,network_code=None,station_code=None,phase_hint='P')
        S_list=pltwf.getPicksAndArrivals(single_event,network_code=None,station_code=None,phase_hint='S')
            
        PS_list=[[P_pair,S_pair] for P_pair in P_list for S_pair in S_list if P_pair[0].waveform_id.station_code==S_pair[0].waveform_id.station_code]
            
        for i_station in range(len(PS_list)):
            counter+=1
            single_PS=PS_list[i_station]
            station_code=single_PS[0][0].waveform_id.station_code
            p_time=UTCDateTime(single_PS[0][0].time)
            s_time=UTCDateTime(single_PS[1][0].time)
            p_travel=p_time-o_time
            s_travel=s_time-o_time
            vpvs=s_travel/p_travel
            vpvss.append(vpvs)
            o_times.append(o_time)
            if station_code=='AXAS1':
                p_times.append(p_time)
            obs_id=station_code+'_'+p_time.strftime('%Y%m%d_%H%M%S')
            
            dic_o_time[obs_id]=o_time

plt.close('all')
fig,ax=plt.subplots()
ax.hist(vpvss,np.linspace(0,4,50))


x=np.array(swm.obspytime2matplotlib(o_times))
fig,ax=plt.subplots()
ax.plot(o_times,vpvss,'ok',ms=1,alpha=0.5,mec='none')
ax.set_ylim([1.2,3.5])

fig,ax=plt.subplots()
ax.plot(x,vpvss,'ok',ms=1,alpha=0.5,mec='none')
ax.set_ylim([1.2,3.5])

ax.xaxis_date()
#fig.autofmt_xdate()

fig.autofmt_xdate()
#Cat_sws=swm.read_pickle(sws_file)
