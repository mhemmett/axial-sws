#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 12:15:49 2018

@author: baillard
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal
import sys
import copy
import time
from scipy.ndimage import zoom
from scipy.ndimage.filters import minimum_filter,maximum_filter
import logging

from shearwavesplit import get_LAMBDAS,split,unsplit,get_contours,select_contour,box_contour
import general.util as gutil


def minmax2zeroone(array):
    """
    Function made to normalize the array to zero-one, based on min and max
    
    Inputs:
        ---
        array: np.array: array to normalize
        
    Outputs
    -------
        array: np.array: normalized array
    """
    
    min_value=np.min(array)
    max_value=np.max(array)

    array=array-min_value
    array=array/max_value
    return array

def extrema(mat,mode='constant',size=20): 
    """
    Find the indices of local extrema (min and max) in the input array.
    
    Inputs
    ------
        array: np.array
        mode: str: see minimum_filter options
        size: float: size of the window sensitivity

    Outputs
    -------
        min_list,max_list: list: list of tuples (ind_row,ind_col,ext_value)
        
    """
    mn = minimum_filter(mat, size=size, mode=mode)
    mx = maximum_filter(mat, size=size, mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    # (mat == mn) true if pixel is equal to the local in
    # Return the indices of the maxima, minima
    min_indarray=np.nonzero(mat == mn) 
    max_indarray=np.nonzero(mat == mx)
    
    ### Order into array
    min_indarray=np.column_stack(min_indarray)
    max_indarray=np.column_stack(max_indarray)
    min_values=np.column_stack((min_indarray,mat[min_indarray[:,0],min_indarray[:,1]]))
    max_values=np.column_stack((max_indarray,mat[max_indarray[:,0],max_indarray[:,1]]))
    
    ### Sort 
    min_values=min_values[min_values[:,2].argsort()] #from lowest min to biggest min
    max_values=max_values[max_values[:,2][::-1].argsort()] #from biggest max to lowest max
    
    ### Make list of tuples
    min_list=[(int(x[0]),int(x[1]),x[-1]) for x in min_values]
    max_list=[(int(x[0]),int(x[1]),x[-1]) for x in max_values]

    return min_list,max_list

def quality_mesh(array,threshold=0.5,mode='below'):
    """
    Function made to estimate the quality of the mesh by computing the ratio of samples
    above/below a given threshold. quality=1 means dirac (one spike),
    
    Inputs
    ----
        array: np.array: input
        threshold: float: limit to compute ratio
        mode: str: 'below','above'
        
    Outputs
    ---
        quality: float: 1= singles spikes,0.5= lots of noise
    
    """
    
    # Check
    if mode not in ['below','above']:
        mode='below'
        
    # Normalize
    array=minmax2zeroone(array)
    
    # Compute
    if mode=='below':
        n_samples=np.sum(array<threshold)
    else:
        n_samples=np.sum(array>threshold)
        
    noise=n_samples/np.size(array)
    quality=(1-noise)

    return quality

def get_valcontour(array,ind_row,ind_col,cont_step,ax=None,flag_plot=False):
    """
    *** Function made to compute the contour that is at array[ind_row,ind_col]+cont_step, then returns the size of the box
    that encompasses this contour.
    Everything is computed using INDICES/SAMPLES and not REAL VALUES
    
    Inputs
    -----
        array: np.array: input M x N array
        ind_row,ind_col: int: indices to evaluate contour
        cont_step: float: contour_step to add to the value
        ax: plt.axes: ax to plot
        flag_plot: boolean: to plot
        
    Outputs
    ------
        contour_select: np.arrays: array of indices
        width,height: float: dimensions of the box (samples)
        (ind_row,ind_col,min_value): tuples: 
        box: np.array: box vertices
        ax: plt.axes:
    
    """

    array_value=array[ind_row,ind_col]
    
    ### Select contour associated to single min
    
    contour_list=get_contours(array,array_value+cont_step)
    contour_select=select_contour(ind_col,ind_row,contour_list)

    ## Get box
    
    width,height=box_contour(contour_select)
    box=None
    
    if flag_plot:
        if ax is None:
            fig,ax=plt.subplots()
            ax.imshow(array,aspect=0.1,cmap=plt.cm.get_cmap('jet'))
        ax.plot(ind_col,ind_row,'ow')
        
        if contour_select is not None:
            ax.plot(contour_select[:,0],contour_select[:,1],color='w')

            ### plot_box
   
            box=get_box([np.min(contour_select[:,0]),np.min(contour_select[:,1])],width,height)
            
            ax.plot(box[:,0],box[:,1],color='w')
        
    return (contour_select,width,height,(ind_row,ind_col,array_value),box,ax)

def get_box(lowerleft,width,height):
    """
    Functin to create a box given width and height and lower left corner
    
    Inputs
    -----
        lowerleft: list: [x,y] coordinates of lowerleft corner of the box
        width,height: float:
            
    Outputs
    ------
        box: np.array: 5x2 array with all vertices
    """
    
    Ac=[lowerleft[0],lowerleft[1]]
    Bc=[Ac[0]+width,Ac[1]]
    Cc=[Bc[0],Bc[1]+height]
    Dc=[Cc[0]-width,Cc[1]]
    
    box=np.array([Ac,Bc,Cc,Dc,Ac])
    
    return box

####################################################################
##### Split contour to allow plotting considering Mesh periodicity
    
def split_contour(cont,y_lim_list,flag_plot=False):
    """
    Function made to split a contour in multiple subcontours where it crosses values in y_lim_list
    It returns sub_contours list, the number of splitted sb_contours is equal to the number of limits
    if y_lim does not cross the contour it returns a None contour
    
    Inputs
    -----
        cont: np.array: Nx2 contour
        y_lim_list: list or float: contains horizontal seperations limits
        flag_plot: bool: True to plot
        
    Outputs
    -------
        cont_list: list of np.arrays or None: contains the arrays defining the sub_contours
    """
    
    ### Check
    if not isinstance(y_lim_list,list):
        y_lim_list=[y_lim_list]
    else:
        y_lim_list.sort(reverse=True)

    ### Split
    cont_list=[]
    for y_lim in y_lim_list:
        cont=add_y(cont,y_lim)
        (top,bottom)=split_y(cont,y_lim)
        cont_list.append(top)
        cont=bottom
  
    cont_list.append(bottom)
    
    #### Plot if asked
    if flag_plot:
        fig,ax=plt.subplots()
        ax.axis('equal')
        
        for cont in cont_list:
            if cont is None:
                continue
            ax.plot(cont[:,0],cont[:,1],'k')
        
    ### Return
    return cont_list
    
def add_y(cont,y_lim):
    """
    Function to add to interpolate (x,y) points where the contour crosses y_lim
    this allows proper splitting afterwards
    
    Inputs
    -----
        cont: np.array: [x,y] contour
        y_lim: float: y limit for splitting
        
    Outputs
    -------
        cont: np.array: [x,y] contour with added points
    """
    
    if cont is None:
        return None
    
    ### Find indexes where contour crosses y_lim
    x=cont[:,0]
    y=cont[:,1]
    y_diff=y-y_lim
    
    ind_sel=np.where(np.diff(np.sign(y_diff))!=0)[0]
    ind_next=ind_sel+1
    
    ind_pairs=list(zip(ind_sel,ind_next))
    
    ### Interpolate and add the points to the contour
    x_new=[]
    y_new=[]
    k=0
    for ind_pair in ind_pairs:
        ind_pair=list(ind_pair)
        x_bord=x[ind_pair]
        y_bord=y[ind_pair]
        x_bord=x_bord[np.argsort(y_bord)]
        y_bord=y_bord[np.argsort(y_bord)]
        
        x_lim=np.interp(y_lim,y_bord,x_bord)
        
        x_new.extend(list(x[k:ind_pair[0]+1]))
        x_new=x_new+[x_lim]
        y_new.extend(list(y[k:ind_pair[0]+1]))
        y_new=y_new+[y_lim]

        k=ind_pair[1]
    
    x_new.extend(list(x[k:]))
    y_new.extend(list(y[k:]))
    
    x_new=np.array(x_new)
    y_new=np.array(y_new)
    
    ### Return
    
    return np.column_stack((x_new,y_new))


def split_y(cont,y_lim,flag_plot=False):
    """
    Split the contour into 2 sub_contours above and below y_lim
    
    Inputs
    -----
        cont: np.array: [x,y]
        y_lim: float
        flag_plot: boolean
        
    Outputs
    ------
        (cont_above,cont_below): np.array: above and below contours, if no crossing, one is None
    """
    
    ### Special case
    if cont is None:
        return (None,None)

    ### Check position
    x=cont[:,0]
    y=cont[:,1]
    y_abov=copy.copy(y)    
    y_abov=y[y>=y_lim]
    x_abov=x[y>=y_lim]    
    
    y_bott=copy.copy(y)    
    y_bott=y[y<=y_lim]
    x_bott=x[y<=y_lim]   
    
    ### Return if no crossing
    if x_abov.size==0:
        return (None,cont)
    if x_bott.size==0:
        return (cont,None)
    
   
    x_abov=np.hstack((x_abov,x_abov[0]))
    y_abov=np.hstack((y_abov,y_abov[0]))
    
    x_bott=np.hstack((x_bott,x_bott[0]))
    y_bott=np.hstack((y_bott,y_bott[0]))
        
    cont_abov=np.column_stack((x_abov,y_abov))
    cont_bott=np.column_stack((x_bott,y_bott))
    
    ### Plot if asked
    if flag_plot:
        fig,ax=plt.subplots()
        ax.plot(x,y,'k',lw=6)
        ax.plot(x_abov,y_abov,'r',lw=3)
        ax.plot(x_bott,y_bott,'y',lw=1)
        
    return (cont_abov,cont_bott)

###########################################################################
####################### Functions made to process the LAMBDA2 MESH
###########################################################################

def select_minima(min_list,min_thres,min_numbers):
    """
    Function made to select the lowest minima (below min_thres) up to a maximum of min_numbers minima
    (ex: if 5 minima satisfies the min_thres criteron, then we'll only take the min_numbers lowest minima)
    
    Inputs:
        ----
        min_list: list of tuples: [(ind_row,ind_col,minimum_value),(...)]
        min_thres: float: keep minima below this values
        min_numbers: float,int : keep only a maximum of min_numbers minima
        
    Outputs:
        ----
        min_list: list of tuples: selected tuples that fullfill critera
    """
    
    #min_thres=0.5
    #min_numbers=2
    
    min_list=np.array(min_list)
    
    min_list=min_list[min_list[:,2].argsort()] #from lowest min to biggest min
    min_list=min_list[min_list[:,2]<=min_thres]
    min_list=min_list[0:min_numbers]
    
    min_list=[(int(x[0]),int(x[1]),x[-1]) for x in min_list]
    
    return min_list


def process_LAMBDA2(LAMBDA2,LAGS,ANGLES,
                   min_thres=0.5,min_numbers=2,cont_step=0.05,quality_thres=0.2,zoom_factor=[4,4],
                   flag_plot=False):
    """
    *** Functiona made to process the LAMBDA2 mesh (or any kind of mesh). It selects the minima and outputs a list 
    of MinLambda Classes.
    The LAMBDA2 mesh is normalized from 0 to 1 and all thresholds are given in that normalized reference
    
    Inputs
    ------
        LAMBDA2, LAGS, ANGLES: np.mesh: mesh grids that have all the same size
        min_thres: float: keep only minima under this value
        min_numbers: int: keep only min_numbers minima or less at the end
        cont_step: float: step to plot the contour around minimas
        quality_thres: float: value to compute the quality of the grid
        zoom_factor: float, 2 elem list: zooming values to interpolate the grid (important to have nice minimas)
            if [2,6] for example, 2 multiplicator will be applied to num rows, 4 to num columns
        flag_plot: boolean: true for plot
    
    Outputs
    -----
        MinLambas: list of MinLambda class
        ax: plt.axes: axes object if plotting
        
    """

    MESH=LAMBDA2 # for clarity
        
    #### Zoom to insure good quality of contours
    
    MESH=zoom(MESH,zoom_factor)
    
    ### Normalize
        
    MESH=minmax2zeroone(MESH)
    
    #### Get quality mesh
    
    quality=quality_mesh(MESH,threshold=quality_thres)
    
    #### Find all minima
    
    min_list,_=extrema(MESH)
    
    ### Select properly the minima (below threshol and maximum of X minima)
    
    min_list=select_minima(min_list,min_thres,min_numbers)
    
    #######################################################
    ### Get contour and errors for each selected minima
    
    ### Get step
    
    extent_lag=[np.min(LAGS),np.max(LAGS)]
    extent_angle=[np.min(ANGLES),np.max(ANGLES)]
    
    d_lag=float(np.diff(extent_lag)/MESH.shape[1])
    d_angle=float(np.diff(extent_angle)/MESH.shape[0])
    
    ### Make a periodic mesh on angles (vertical) to ensure contours are closed
     
    PER_MESH=np.vstack((MESH,MESH[1:-1,:],MESH))
        
    ### start loop
    ax=None
    MinLambdas=[]

    for ind_row,ind_col,min_value in min_list:
        
        ### Modify indices according to new mesh
        
        ind_row_per=ind_row+MESH.shape[0]-1
        contour_select,width,height,min_tuple,box,_=get_valcontour(PER_MESH,ind_row_per,ind_col,cont_step,flag_plot=False)
        
        ### Report to LAGS and ANGLES instead of indexes
            
        lag=np.min(LAGS)+ind_col*d_lag
        angle=np.min(ANGLES)+ind_row*d_angle
        
        lag_error=999
        angle_error=999
        
        if contour_select is not None:
            lag_error=width*d_lag
            angle_error=height*d_angle
            
        ### Feed to class
        
        MinSc=MinLambda()
        MinSc.lag=lag
        MinSc.angle=angle
        MinSc.angle_error=angle_error
        MinSc.lag_error=lag_error
        MinSc.lambda_value=min_value
        MinSc.quality=quality
        
        MinLambdas.append(MinSc)
        
        #### Plot if asked
        
        if flag_plot:
                                        
            if ax is None: ### Plot things just once in the loop
                fig,ax=plt.subplots()
                ax.imshow(MESH,extent=extent_lag+extent_angle,origin='lower',cmap=plt.cm.get_cmap('jet'),aspect=10) # GRID
                CS=ax.contour(MESH,extent=extent_lag+extent_angle,origin='lower',levels=[min_thres],colors='w',linestyles=':') # CONTOURS
                ax.clabel(CS, [min_thres],fmt='%.2f')
                ax.set_xlabel('LAG [samples]')
                ax.set_ylabel('ANGLE [rad]')
                ax.set_title('Quality = %.2f'%quality)
                
            ax.plot(lag,angle,'ow')
            
            if contour_select is not None:
            
                #### Prepare for plotting
            
                cont_x=np.min(LAGS)+d_lag*contour_select[:,0]
                cont_y=np.min(ANGLES)+d_angle*(contour_select[:,1]-MESH.shape[0]+1) # Don't forget to substract the number of column that were added
                cont=np.column_stack((cont_x,cont_y))
                
                ### Box limits
                
                box=get_box([np.min(cont_x),np.min(cont_y)],lag_error,angle_error)
                [abov_box,middle_box,bott_box]=split_contour(box,[np.pi/2,-np.pi/2])
        
                if abov_box is not None:
                    abov_box[:,1]=abov_box[:,1]-np.pi
                if bott_box is not None:
                    bott_box[:,1]=bott_box[:,1]+np.pi
                
                sub_boxes=[abov_box,middle_box,bott_box]
                sub_boxes=[x for x in sub_boxes if x is not None]
                
                for sub_box in sub_boxes:
                    ax.plot(sub_box[:,0],sub_box[:,1],color='w')
           
#                #### Make sure cont_y is inside -pi/2 , pi/2 for plotting
#                
                [abov_box,middle_box,bott_box]=split_contour(cont,[np.pi/2,-np.pi/2])
        
                if abov_box is not None:
                    abov_box[:,1]=abov_box[:,1]-np.pi
                if bott_box is not None:
                    bott_box[:,1]=bott_box[:,1]+np.pi
                
                sub_boxes=[abov_box,middle_box,bott_box]
                sub_boxes=[x for x in sub_boxes if x is not None]
                
                for sub_box in sub_boxes:
                    ax.plot(sub_box[:,0],sub_box[:,1],color='w')
           
      
#            
    #### Return
    
    return (MinLambdas,ax)
    

class MinLambda():
    """
    Class to store the minima obtained from Lambda2 analysis
    
    Used in
    -------
        process_LAMBDA
    """
    def __init__(self):
        self.quality=None
        self.lambda_value=None
        self.lag=None
        self.angle=None
        self.lag_error=None
        self.angle_error=None
        
    def __repr__(self):
        str_cmd=\
        '[quality : %f\n'%self.quality+\
        'lambda_value : %f\n'%self.lambda_value+\
        'lag : %f\n'%self.lag+\
        'angle : %f]\n'%self.angle
        return str_cmd
    
   

#   