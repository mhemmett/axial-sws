#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:33:21 2019

@author: baillard

Function made to concatenate all the individual EQ-STATION pairs of SWS observation
from run_sws.py into a single SWS_cat class, the output is saved into a pickle file
"""

import sws_methods as swm
import matplotlib.pyplot as plt
import numpy as np

### Parameters

station_list=['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3','AXID1']
input_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2'
output_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2_cat'
swm.cat_pickles(input_dir,output_dir,station_list=station_list)

