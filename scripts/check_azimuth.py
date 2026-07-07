#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:27:46 2018

@author: baillard
"""

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
import pickle
import os
from matplotlib import cm

from general.plotwaveform import SDS2streams,plot_event,getPicksAndArrivals,plot_picks,stream2data,jurkevics
from general.plotwaveform import plot_window,plot_particle_motion,plot_sta_az
from obspy.io.nlloc.core import read_nlloc_hyp
from obspy.signal.polarization import particle_motion_odr,vidale_adapt,eigval
from obspy.taup import TauPyModel
from general.util import smooth_curve
from general.projection import ll2xy
from general.geometry import get_angles
import matplotlib.gridspec as gridspec


import sys

plt.close('all')

### Parameters

sds_root='/home/baillard/SDS/'
start_time=UTCDateTime(2015,2,1)
time_delay=5
network_code='OO'
station_code='AXCC1'
channel_code="EH?,HH?"
p_window=[0.02,0.1]
nlloc_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_MODELS/NLLOC_3H/loc/AXIAL.sum'
station_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/STATIONS/stations_axial.xyz'


### Read nlloc file and dump it to pickle for rapidity

if 'Cat' not in locals():
    file_name='nlloc.pickle'
    if not os.path.isfile(file_name):
        Cat=read_nlloc_hyp(nlloc_file)
        pickle.dump(Cat,open(file_name,'wb'))
    else:
        Cat=pickle.load(open(file_name,'rb'))
        
#####


#### start plotting


single_event=Cat[15]

phase_hint='S'

list_pairs=getPicksAndArrivals(single_event,phase_hint=phase_hint)
az_ray_sta=[Arrival.azimuth for _,Arrival in list_pairs]
az_ray_sta=np.array(az_ray_sta)*np.pi/180
sta_labels=[Pick.waveform_id['station_code'] for Pick,_ in list_pairs]

(ax,dict_cmap)=plot_sta_az(single_event)


for kk in range(len(az_ray_sta)):
    ax.plot((0, az_ray_sta[kk]), ( 0, 1),'-',color=dict_cmap[sta_labels[kk]])
    
    
ax.set_rmin(0)  
sys.exit()
       



        
#### start plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2.0)



        