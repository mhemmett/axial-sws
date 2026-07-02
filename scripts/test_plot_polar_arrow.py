#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:32:30 2019

@author: baillard
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import general.projection as gproj

### Parameters

flag_normalize=True
plt.close('all')
vector=np.array([-3,3])
sign=1
show_axis=True
facecolor='r'
mode='trigo'
rmax=1
width=2



plot_vector(vector)