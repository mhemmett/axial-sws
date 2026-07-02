#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:06:04 2018

@author: baillard
"""

import sws_methods as swm
import matplotlib.pyplot as plt

plt.close('all')
fig,ax_main=plt.subplots()
swm.get_ax_polarinsets(ax_main=ax_main,num_ax=6,theta_rot=0)