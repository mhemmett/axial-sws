#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:42:42 2018

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile,copy
import datetime
import os,sys
import copy
import pickle

import nlloc.util as nllocutil
from lotos.model_3d.vgrid import VGrid,read
from lotos.LOTOS_class import Catalog
from general.util import smooth_curve
from general.GMT import plot_lines,plot_stations,plot_box
import dd.util as ddutil
import general.util as gutil
import general.projection as gproj


plt.close('all')

##### Paramters

flag_save=True
center=[5.4,3.8]
angle_deg=18
len_prof=[0,5]
width_prof=[0.2,0.2]
ray_files=[
        ['/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/PROG/NLLOC_MODELS/NLLOC_FA/loc/AXIAL.sum.grid0.loc.hyp','nlloc']
        ]
#ray_files=[
#        ['/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/CATALOG/Axial_hypoDDPhaseInput.dat','dd']
#        ]
ini_lon=-130.1
ini_lat=45.9
coord_ini=[4.7,5.8]
coord_fin=[6,2]
map_lim=[5,10.5,1.5,8]
pickle_load=False
num_points=10
num_rows=5



#### Build profil centers

coord=np.vstack((coord_ini,coord_fin))
x,y=smooth_curve(coord[:,0],coord[:,1],num_points=num_points,k=1)
centers=list(zip(x,y))
num_cols=int(np.ceil(len(centers)/num_rows))

#### Write into file

#fic=open('/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/AXIAL_profiles_ARTICLE_10.txt','wt')
#
#for kk in range(len(centers)):
#    center_str=('center=[%.2f,%.2f]\n'%(centers[kk][0],centers[kk][1]))
#    angle_str=('angle_deg=%.2f\n'%angle_deg)
#    len_str='len_prof=[%.2f,%.2f]\n'%(len_prof[0],len_prof[1])
#    width_str='width_prof=[%.2f,%.2f]\n\n'%(width_prof[0],width_prof[1])
#    
#    ### Print
#    
#    fic.write(center_str)
#    fic.write(angle_str)
#    fic.write(len_str)
#    fic.write(width_str)
#
#fic.close()


#centers=[[10,2.5]]
#angle_deg=110
#len_prof=[0,7]
#width_prof=[1,1]

####  Read files

Rays=[]
for ray_file,ray_type in ray_files:
    pickle_file='ray.'+ray_type+'.pickle'
    if pickle_load:
        Ray=pickle.load(open(pickle_file,'rb'))
    else:      
        if ray_type=='nlloc':
            Ray=nllocutil.read_nlloc_sum(ray_file)
            Ray.ll2xy(ini_lon,ini_lat)
            Ray=Ray.randshift_hypo(0.01)
        elif ray_type=='dd':
            Ray=ddutil.read_dd(ray_file,out_type='nofull')
            Ray.ll2xy(ini_lon,ini_lat)
        pickle.dump(Ray,open(pickle_file,'wb'))
            
    Rays.append(Ray)
    
### check number of phases per events
    
new_Rays=[]
for A in Rays:
    nobs_list=[x.num_phase for x in A.events]
    
    new_events=[x for x in A.events if x.num_phase>9]
    
    B=Catalog()
    B.events=new_events
    new_Rays.append(B)


#### PLOT
    
plt.close('all')

for Ray in new_Rays:

    ##### PLOT MAP
    
    fig_map,ax_map=plt.subplots()
    (ax_map,_)=plot_stations(ax=ax_map)
    (ax_map,_)=plot_lines(ax=ax_map,ini_lon=ini_lon,ini_lat=ini_lat)
    xh,yh,zh=Ray.get_xyz()
    ax_map.plot(xh,yh,'o',mec='none',mfc='0.2',markersize=1,alpha=0.2,rasterized=True)
    ax_map.set_aspect('equal', adjustable='box')
    ax_map.set_xlabel('X [km]')
    ax_map.set_ylabel('Y [km]')
    ax_map.axis(map_lim)
    plt.savefig('test.pdf',format='pdf',quality=300, bbox_inches='tight')
    
    sys.exit()
    kk=-1
    for center in centers:
        kk+=1
        plot_box(center,angle_deg,len_prof,width_prof,ax=ax_map,key=str(kk+1))
        
        
    #### Plot lava flow and fissures
    
    lava_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/'
    lava_files=['Axial_2015_Flows_points.ll']
    lava_files=[lava_dir+x for x in lava_files]
    
    fissure_files=['Axial_2015_Fissures_points.ll','Axial_2011_Fissures_points.ll']
    fissure_files=[lava_dir+x for x in fissure_files]
    
    for file_in in lava_files:
        data=gutil.read_datafile(file_in)
        for single_array in data:
            lon=single_array[:,0]
            lat=single_array[:,1]
            x,y=gproj.ll2xy(lon,lat,ini_lon,ini_lat)
            ax_map.plot(x,y,'b')
            
    for file_in in fissure_files:
        data=gutil.read_datafile(file_in)
        for single_array in data:
            lon=single_array[:,0]
            lat=single_array[:,1]
            x,y=gproj.ll2xy(lon,lat,ini_lon,ini_lat)
            ax_map.plot(x,y,'r')
                           
                
            
    ##### PLOT CROSS
        
    fig,ax_list=plt.subplots(num_rows,num_cols,figsize=(np.array([  6.55,8.03])))
    ax_list=np.transpose(ax_list)
    ax_list=ax_list.reshape(-1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0, hspace=0)
    kk=-1
    for center in centers:
        kk+=1
    
        (_,ax)=Ray.plot_cross(center=center,angle_deg=angle_deg,len_prof=len_prof,width_prof=width_prof,
                       map_plot=False,mec='none',mfc='0.2',markersize=1.5,alpha=0.1,ax=ax_list[kk])
        
        if np.mod(kk+1,num_rows)!=0:
            ax.set_xticks([])
        else:
            ax.set_xlabel('X [km]')
            
        if kk>num_rows-1:
            ax.set_yticks([])
        else:
           ax.set_ylabel('Z [km]')
        
        
        ax.text(0.3,0.3,'key=%i'%(kk+1))
        
        ax.set_aspect('equal','box')
        ax.set_ylim([2.5,0])
        ax.set_xlim([0,5])
        
#### SAVE

if flag_save:
    
    for num in plt.get_fignums():
        plt.figure(num)
        #plt.savefig('Figure_%i_V1.jpg'%num,format='jpg',dpi=500,bbox_inches='tight')
        plt.savefig('Figure_%i_V1.pdf'%num,format='pdf',dpi=300,bbox_inches='tight')
