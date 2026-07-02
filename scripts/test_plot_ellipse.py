#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:38:19 2019

@author: baillard
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import utm
import pyproj


theta=np.linspace(0,2*np.pi,1000)

dx=2
dy=4
num=dx*dy
den=np.sqrt( (dx*np.sin(theta))**2 + (dy*np.cos(theta))**2 )

r=num/den

plt.close('all')
plt.plot(theta,r)