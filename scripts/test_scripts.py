#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:04:28 2018

@author: baillard
"""

from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.patches import Rectangle

from general.plotwaveform import SDS2streams,plot_event,getPickForArrival,getPicks,getPicksAndArrivals,plot_picks,stream2data,jurkevics
from general.plotwaveform import plot_window,plot_particle_motion
from obspy.io.nlloc.core import read_nlloc_hyp
from obspy.signal.polarization import particle_motion_odr,vidale_adapt,eigval
from obspy.taup import TauPyModel
from general.util import smooth_curve

import sys

plt.close('all')
sds_root='/home/baillard/SDS/'
start_time=UTCDateTime(2015,2,1)
time_delay=5
network_code='OO'
station_code='AXCC1'
channel_code="EH?,HH?"
p_window=[0.02,0.1]

nlloc_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_MODELS/NLLOC_3H/loc/AXIAL.20150301.090055.grid0.loc.hyp'

#### Read hypocenter_file

hyp=read_nlloc_hyp(nlloc_file)

start_time=hyp[0].origins[0].time
single_event=hyp[0]

#A=getPicks(hyp[0].picks,station_code='AXAS1',phase_hint='P')

pair_list=getPicksAndArrivals(single_event,network_code=None,station_code=station_code,phase_hint='P')

p_time=pair_list[0][0].time
time_delay=5
#plot_event(hyp[0],sds_root,15,5,_network_code='OO',_station_code='AXAS1',_channel_code='EH*')

### Extract st for event

st=SDS2streams(sds_root,p_time-1,time_delay,
                network_code=network_code,station_code=station_code,channel_code=channel_code)

st.detrend("linear")
st.taper(max_percentage=0.05, type="hann")
st.filter("bandpass",freqmin=5,freqmax=40)

st.sort(reverse=True)


stream_fig=st.plot(handle=True)

ax_list=stream_fig.get_axes()

ax=ax_list[0]
plot_window(ax,p_time-p_window[0],np.sum(p_window),label='P')
##
plot_picks(stream_fig,st,single_event,phases=['P'])

st_window=st.slice(p_time-p_window[0],p_time+p_window[1])

#kk=-1
#for trace in st_window:
#    kk+=1
#    if kk!=2:
#        trace.data=trace.data/30

st_window.plot(handle=True,type='relative')

ax=plot_particle_motion(st_window,ref_comp='N',ref_time=p_time)

st_synth=st_window.copy()
st_rotate=st.copy()

az,inc,rec=jurkevics(st_synth)


azz,incc,err_azz,err_incc=particle_motion_odr(st_synth)
st_rotate.rotate('ZNE->LQT', back_azimuth=azz, inclination=incc)
st_rotate.plot(handle=True)

#ax=plot_particle_motion(st_rotate,ref_comp='T',ref_time=p_time)

sys.exit()
### Compute covariance matrix

data_array=stream2data(st)

### Make sure data is oriented ZNE (1st,2nd and 3rd Column)



    
    


    
B=np.zeros(data_array.shape)
#B[:,1]=data_array[:,0]
B[:,0]=data_array[:,0]
B=data_array

st_synth=st.copy()
kk=-1
for trace in st_synth:
    kk+=1
    trace.data=B[:,kk]

st_rotate=st_synth.copy()


az,inc,rec=jurkevics(st_synth)

azz,incc,_,_=particle_motion_odr(st_synth)
eig_vals,eig_vecs=cov_eig(B)
leigenv1, leigenv2, leigenv3, rect, plan, dleigenv, drect, dplan=eigval(B[:,2], B[:,1], B[:,0], [1, 1, 1, 1, 1], normf=1.0)



st_rotate.rotate('ZNE->LQT', back_azimuth=azz, inclination=incc)
st_rotate.plot(handle=True)
st_synth.plot(handle=True)
#stream_fig=st.plot(handle=True)
###
#plot_picks(stream_fig,st,single_event,phases=['P'])