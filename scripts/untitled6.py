#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:01:09 2018

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy.io.nlloc.core import read_nlloc_hyp

file='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/SWS_tmp/split_001.nlloc'

#Cat=read_nlloc_hyp(file)


class Father():
    
    def __init__(self,sons=None):
        if not sons:
            self.sons = []
        else:
            self.sons = sons
        self.sons=None
    def __getitem__(self, index):
        """
        __getitem__ method of the Catalog object.
        :return: Event objects
        """

        if isinstance(index, slice):
            return self.__class__(sons=self.sons.__getitem__(index))
        else:
            return self.sons.__getitem__(index)
        
    def __str__(self):
        out='sdfe'
        return out
    
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())

class Son():
    def __init__(self):
        self.a=2
        
        
A=Son()
B=Son()
        
F=Father()

F.sons=[A,B]
    

    
#def ax_sws_diagnosis():
#
#    subplot_shape=(5,3)
#    
#    ax_trace_X = plt.subplot2grid(subplot_shape, (0, 0), colspan=2)
#    ax_trace_Y = plt.subplot2grid(subplot_shape, (1, 0), colspan=2)
#    ax_lambda2 = plt.subplot2grid(subplot_shape, (0, 2), rowspan=2)
#    ax_ini = plt.subplot2grid(subplot_shape, (2, 0))
#    ax_un1 = plt.subplot2grid(subplot_shape, (3, 0))
#    ax_un2 = plt.subplot2grid(subplot_shape, (4, 0))
#    ax_polar = plt.subplot2grid(subplot_shape, (3, 1),rowspan=2)
#    ax_polar_un = plt.subplot2grid(subplot_shape, (3, 2),rowspan=2)
#
#    return [ax_trace_X,ax_trace_Y,ax_lambda2,ax_ini,ax_un1,ax_un2,ax_polar,ax_polar_un]
#
#
##
#[ax_trace_X,ax_trace_Y,ax_lambda2,ax_ini,ax_un1,ax_un2,ax_polar,ax_polar_un]=ax_sws_diagnosis()
#
#

