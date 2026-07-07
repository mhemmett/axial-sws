#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:33:21 2019

@author: baillard


"""

import sws_methods as swm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

### Paraleters

station_name='AXCC1'
cat_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat/'
ini_lon=-130.1
ini_lat=45.9

file_in= os.path.join(cat_dir,'')+station_name+'.clean.cat.pickle'
file_in= os.path.join(cat_dir,'')+station_name+'.cat.pickle'

### Read pickle into catalog

Cat=swm.read_pickle(file_in)



elems_dic=Cat.get_dic(minlambda_select='min')
station_dic=swm.read_stationfile()

### Select arrays based on baz

data=np.column_stack((elems_dic['x'],
                      elems_dic['y'],
                      elems_dic['z'],
                      elems_dic['baz_trigo'],
                      elems_dic['s_time'],
                      elems_dic['lag'],
                      elems_dic['fast']))

x=elems_dic['s_time']

x=np.array(swm.obspytime2matplotlib(x))

lag=elems_dic['lag']
fast=elems_dic['fast']*180/np.pi

lag=np.asarray(lag)
fast=np.asarray(fast)

#x=x[lag<10]
#fast=fast[lag<10]
#lag=lag[lag<10]

#plt.close('all')

fig,ax=plt.subplots(2,1,sharex=True)
ax[0].plot(x,lag,'ok',ms=0.1)
ax[1].plot(x,fast,'ok',ms=0.1)

### Cosmetic

ax[0].set_title('%s'%station_name)
ax[0].set_ylim([0,40])
ax[0].set_ylabel('Lags [samples]')
ax[1].set_ylabel('$\phi$ [°]')
ax[1].set_ylim([-90,90])

ax[1].xaxis_date()
fig.autofmt_xdate()

