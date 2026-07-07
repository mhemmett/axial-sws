#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:23:08 2019

@author: baillard
"""

import sws_methods as swm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from obspy.core.utcdatetime import UTCDateTime

### Parameters

station_list=['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3','AXID1']
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat/'
dic_exc_periods={
        'AXAS1':None,
        'AXAS2':None,
        'AXEC1':None,
        'AXEC2':None,
        'AXEC3':None,
        'AXCC1':[[UTCDateTime('2015-03-01T00:00:00'),UTCDateTime('2015-04-27T00:00:00')]],
        'AXID1':[[UTCDateTime('2015-04-24T14:43:00'),UTCDateTime('2015-12-30T00:00:00')]]
        }
suffix_out='.clean.cat.pickle'
suffix_in='.cat.pickle'

### Loop through stations


for station_code in station_list:
    remove_counter=0
    
    print('Process station %s'%station_code)
    
    ### Define names
    
    file_in= os.path.join(cat_dir,'')+station_code+suffix_in
    file_out= os.path.join(cat_dir,'')+station_code+suffix_out

    ### Read pickle into catalog
    
    Cat=swm.read_pickle(file_in)
    new_obs=[]
    
    for obs in Cat.obs:
        
        ### Remove file with no Lambdas (1st)
        
        if len(obs.MinLambdas)==0:
            remove_counter+=1
            continue
        
        ### Remove if not it time period
        
        if dic_exc_periods[station_code] is not None:
            s_time=obs.s_time
            flag_ex=False
            for time_left,time_right in dic_exc_periods[station_code]:
                if (s_time>=time_left) & (s_time<=time_right):
                    flag_ex=True
                    remove_counter+=1
                    continue
            if flag_ex:
                continue
                
        new_obs.append(obs)
    
    print('Removed %i observations'%remove_counter)
    new_Cat=swm.SWScat(obs=new_obs)
    new_Cat.update()
    new_Cat.compute_ini() 
    new_Cat.write_pickle(file_out)

        
        
        
