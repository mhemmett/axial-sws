#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:38:15 2018

@author: baillard
"""

from obspy.io.nlloc.core import read_nlloc_hyp
import os
import sys

### Parameters

nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/CATALOG/AXIAL.PHASE.FINAL_3D.nlloc'
event_max=2000
prefix='test'
output_dir='scratch'


def split_nlloc(nlloc_file,event_max,prefix=None,output_dir='split_nlloc'):
    
    if prefix is None:
        prefix=nlloc_file.split('/')[-1]
        
    ### Check
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ### Process
    
    fic=open(nlloc_file,'rt')
    lines=fic.read().splitlines()
    fic.close()
    
    ### Initialize loop
    
    event_counter=0
    file_counter=0
    line_counter=0
    file_names=[]
    
    for line in lines:
        if (event_counter==0) & (line_counter==0):
            file_counter+=1
            file_name='%s_%03i.nlloc'%(prefix,file_counter)
            file_names.append(output_dir+'/'+file_name)
            fic=open(file_name,'wt')
        if line=='':
            event_counter+=1
        line_counter+=1
            
        fic.write(line+'\n')
        if event_counter==event_max:
            event_counter=0
            line_counter=0
            fic.close()
            
    ### Close
    
    fic.close()
    
    ### Return
    
    return file_names
