#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:17:11 2018

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class

d = np.array([[ 1.0, 0.6, 0.8, 0.2, 56890, 98.67],
              [ 0.8, 0.3, 1.0, 0.5, 94948, 98.00],
              [ 1.0, 0.8, 0.1, 0.3, 78483, 97.13]])

fig, ax = plt.subplots()
ax.margins(0.15)


def plot_inset(data, x,y, axis_main, width ):
    ax_sub= inset_axes(axis_main, width=width, height=width, loc=10, 
                       bbox_to_anchor=(x,y),
                       bbox_transform=axis_main.transData, 
                       borderpad=0.0, axes_class=get_projection_class("polar"))

    theta = np.linspace(0.0, 2 * np.pi, 4, endpoint=False)
    radii = [90, 90, 90, 90]
    width = np.pi / 4 * data
    bars = ax_sub.bar(theta, radii, width=width, bottom=0.0)
    ax_sub.set_thetagrids(theta*180/np.pi, frac=1.4)
    ax_sub.set_xticklabels(["v{}".format(i) for i in range(1,5)])
    ax_sub.set_yticks([])


for da in d:    
    plot_inset(da[:4], da[4],da[5], ax, 0.5 )

#plot invisible scatter plot for the axes to autoscale
ax.scatter(d[:,4], d[:,5], s=1, alpha=0.0)

plt.show()