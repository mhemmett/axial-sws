#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 08:49:28 2018

@author: baillard
"""

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt



x = np.arange(0, np.pi, 0.1)
y = np.arange(0, 2*np.pi, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) * np.sin(Y) * 10

colors = [(0,(1, 0, 0)),(0.1, (0, 1, 0)), (1,(0, 0, 1))]


colors=[
        (255,179,186),
        (255,223,186),
        (255,255,186),
        (186,255,201),
        (186,225,255)]

colors=[(r/256,g/256,b/256) for (r,g,b) in colors]


cm=LinearSegmentedColormap.from_list('name',colors) 

plt.close('all')
plt.imshow(Z,cmap=cm)
plt.colorbar()