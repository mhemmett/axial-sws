#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:19:50 2018

@author: baillard
For shear wave splitting we use the convention of a Nx2 data array with the first column being the X/E component and the second column 
being the Y/N direction
    rm

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.gridspec as gridspec
from plotwaveform import cov_eig
import time
import pickle
from scipy import signal
import sys,os
import copy
import typing
from obspy.signal.util import _npts2nfft 
import scipy
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import time
try:
    from multitaper import MTSpec
    MULTITAPER_AVAILABLE = True
except ImportError:
    MULTITAPER_AVAILABLE = False
    print("Warning: multitaper package not available. Multitaper spectral analysis will not work.")

from scipy.signal.windows import hann
from obspy import UTCDateTime
from scipy.ndimage import zoom,gaussian_filter
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import datetime as dt

import util as gutil
import projection as gproj
import shearwavesplit as sws


from scipy.ndimage import zoom
from scipy.ndimage.filters import minimum_filter,maximum_filter
import logging
from scipy.ndimage import zoom

########################################
########### Core #################
##################################

def tolist(obj):    
    """
    Function made to convert float or list to list
    """
    
    if not isinstance(obj,list):
        if type(obj).__module__=='numpy':
            obj=list(obj)
        else:
            obj=[obj]
    return obj
                

def hold_plot():
    """
    Function made to hold plot in loop
    """
    keyboardClick=False
    while keyboardClick != True:
        keyboardClick=plt.waitforbuttonpress(timeout=-1)

def sample_array(nstart,nend,Nvalues):
    """
    Made to create an array of integers evenly spaced
    last sample in output array might be different from initial nend
    
    Inputs:
        nstart,nend:int
    """
    
#    nstart=5
#    nend=25
#    N=100
    delta=(nend-nstart)

    reste=delta%Nvalues
    inc=delta//Nvalues
    while reste!=0:
        Nvalues=Nvalues-1
        reste=delta%Nvalues
        inc=delta//Nvalues

    sam_array=np.arange(nstart,nend+1,inc).astype(int)
        
    return sam_array

def shift_2(data_array,Nsamp):
    """
    Function made to shift second column by Nsamples to the right (N>0) or to the left (N<0)
    
    Inputs
    ------
        data_array: np.array : Nx2 data array
        Nsamp: int: N samples to shift
        
    Outputs
    ------
        new_data: np.array: (N-Nsamp) x 2 data array
    """

    y=data_array[:,1]
    
    y_new=np.zeros(y.shape)
    
    shift=abs(Nsamp)
    if Nsamp==0:
        y_new=y
    elif Nsamp<0:
        y_new[:-shift]=y[shift:]
    else:
        y_new[shift:]=y[:-shift]

    new_data=np.column_stack((data_array[:,0],y_new))
    return new_data

def cut(data_array,s0=0,s1=None):
    """
    cut data array
    
    Inputs
    ------
        s0: int: start sample for cutting
        s1: int,None: last sample for cutting
        
    Output:
    -------
        np.array
    """
    
    s0=int(np.round(s0))
    
    if s1 is None or s1>data_array.shape[0]:
        s1=data_array.shape[0]
        
    return data_array[s0:s1]


def unsplit(data_array,lag,angle_rad,flag_plot=False,ax_list=None):
    """
    Function made to unsplit a given data array (unrotate, unshift)
    
    Inputs
    -----
        data_array: np.array: [X,Y] (or [E/N]) components
        lag: float,int: lag in samples for "unshifting"
        angle_rad: float: angle CCW from X for "unrotating"
        flag_plot: boolean: True to plot
        
    Outputs
    -------
        data_array: np.array: unsplitted [X,Y]
    """
    
    lag=int(np.round(lag))
    ini_array=copy.deepcopy(data_array)
    shift=-lag # If lag is postive move it to the left
    data_array=rotate_basis_2(data_array,angle_rad)
    data_array=shift_2(data_array,shift)
    
    ### Plot if asked
    
    if flag_plot:
        if ax_list is None:
            fig,ax_list=plt.subplots(3,1,figsize=[ 7,  6],sharey=True,sharex=True)
            fig.tight_layout()
            
       
        ax_list[0].plot(ini_array[:,0],'k',label='X')
        ax_list[0].plot(ini_array[:,1],'--r',label='Y')
        ax_list[1].plot(data_array[:,0],'k',label='X')
        ax_list[1].plot(data_array[:,1],'--r',label='Y')
        ax_list[2].plot(data_array[:,0],'k',label='X')
        ax_list[2].plot(-data_array[:,1],'--r',label='-Y')
        
        for ax in ax_list:
            ax.legend(loc=1)
        
        ### Title
        
        texts=['Initial','Unsplit','Unsplit']
        
        k=-1
        for text in texts:
            k+=1
        
            ax_list[k].text(0.05, 0.9, text, fontweight='bold',horizontalalignment='left',
                   verticalalignment='top', transform=ax_list[k].transAxes)

        ax_list[2].set_xlabel('Samples')
        #

    ### Return
    return (data_array,ax_list)


def split(data_array,lag,angle_rad,flag_plot=False):
    ini_array=copy.deepcopy(data_array)
    shift=lag
    data_array=shift_2(data_array,shift)
    data_array=rotate_data_2(data_array,angle_rad)
    
    
    if flag_plot:
        fig,ax_list=plt.subplots(1,2,figsize=[ 8,  2],sharey=True)
        ax_list[0].plot(ini_array[:,0],'k')
        ax_list[0].plot(ini_array[:,1],'--r')
        ax_list[1].plot(data_array[:,0],'k')
        ax_list[1].plot(data_array[:,1],'--r')
        fig.tight_layout()


    return data_array


def rotate_basis_2(data,angle_rad,flag_plot=False):
    """
    Function made to change basis in 2 dimensions, angle rad is 
    counter clockiwse from E (CCW,trigo convention)
    
    Inputs
    ------
        data: np.array: Nx2 numpy array with (N being the number of samples, x: 1st column, y:2nd column)
        angle_rad: float: angle in radian CCW from Eeast (x)
    
    Outputs:
    -------
        new_data: np.array: coordinates in new basis
    """
    
    #### Check data
    
    num_col=data.shape[1]

    if num_col!=2:
        raise ValueError('Data has wrong shape, num col is %i'%num_col)
        return
    

    ### Rotation (Change of basis matrix)
    
    sina = np.sin(angle_rad)
    cosa = np.cos(angle_rad)
    
    ##### Check Matrix
    
    R=np.array([[cosa , sina],
             [-sina, cosa]])

    data=data.transpose()
    
    ### Apply product
    
    new_data=np.dot(R,data).transpose()
    data=data.transpose()
    
    #### Plot if asked
    
    if flag_plot:
        fig,ax=plt.subplots()
        ax.plot(data[:,0],data[:,1],'ok')
        ax.plot(new_data[:,0],new_data[:,1],'or')
    
    ### Return and plot
    
    return new_data

def rotate_data_2(data,angle_rad,flag_plot=False):
    """
    Function made to rotate data in 2 dimensions, angle rad is 
    counter clockiwse from E (trigo convention)
    
    Inputs
    ------
        data: np.array: Nx2 numpy array with (N being the number of samples, x: 1st column, y:2nd column)
        angle_rad: float: angle in radian CCW from Eeast (x)
    
    Outputs:
    -------
        new_data: np.array: coordinates in new basis
    """
    
    #### Check data
    

    num_col=data.shape[1]

    if num_col!=2:
        raise ValueError('Data has wrong shape, num col is %i'%num_col)
        return
    

    ### Rotation (Change of basis matrix)
    
    sina = np.sin(angle_rad)
    cosa = np.cos(angle_rad)
    
    ##### Check Matrix
    
    R=np.array([[cosa , -sina],
             [sina, cosa]])

    data=data.transpose()
    
    ### Apply product
    
    new_data=np.dot(R,data).transpose()
    data=data.transpose()
    
    #### Plot if asked
    
    if flag_plot:
        fig,ax=plt.subplots()
        ax.plot(data[:,0],data[:,1],'ok')
        ax.plot(new_data[:,0],new_data[:,1],'or')
    
    ### Return and plot
    
    return new_data


def inside_contour(x, y,cont_array):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    
    Input: 
        x,y: float: points to be checked
        cont_array: np.array: contour plot
        
    Output:
        inside: boolean: True if points inside cont_array
        
    """
    
    ### Close cont_array to be sure
    
    if not np.all(cont_array[0] == cont_array[-1] ):
        cont_array=np.vstack((cont_array,cont_array[0]))
        
    ### check
    n = cont_array.shape[0]
    inside = False
    p1x, p1y = cont_array[0,0],cont_array[0,1]
    for i in range(1, n):
        p2x, p2y = cont_array[i,0],cont_array[i,1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_contours(array,levels):
    """
    Function made to return a list of arrays for all the contours in matplotlib.contour.QuadContourSet 
    
    Inputs
    ------
        contour_set=matplotlib.contour.QuadContourSet
        
    Outputs
    -------
        contour_list: list: list of numpy arrays containing x,y coordinates for each contours
    """
    
        
    #### Define contours , vertices will be given in indexes
    
    plt.ioff()
    fig,ax=plt.subplots()
    ax.imshow(array)
    contour_set=ax.contour(array,levels=levels)
    plt.close(fig)
    plt.ion()
        
    ### Get contours
    
    #contour_list=[]
    #for p in contour_set.collections[0].get_paths():
    #    contour_list.append(p.vertices)

    contour_list = []
    # QuadContourSet.collections is deprecated as of Python 3.8
    for p in contour_set.allsegs[0]:
        contour_list.append(p)
    
    return contour_list
        
def select_contour(x,y,contour_list):

    sel_array=None
    for cont_array in contour_list:
        inside=inside_contour(x,y,cont_array)
        if inside is True:
            sel_array=cont_array
            break
        
    return sel_array
    

def box_contour(data_array):
    """
    Mesure the size of the box contouring the data_array
    
    Input:
        data_array: np.array: Nx2 array
        
    Output:
        width,height: floats: width and height of box
    """
    
    if data_array is None: #(UR)
        width=999
        height=999
        return width,height
    width=np.max(data_array[:,0])-np.min(data_array[:,0])
    height=np.max(data_array[:,1])-np.min(data_array[:,1])
    
    return width,height


#############  Functions Made to Plot polar diagrams
    
def plot_polar_histo(trigo_angles,weights=None,width_bin_deg=1,ax_polar=None,facecolor=None,edgecolor='none',lw=0.2,alpha=1,text=False):
    """
    Function made to plot the histograms of angles on a polar plot
    
    Inputs
    ------
    
    trigo_angles: np.array: array of angles in radians define in trigo convention (>0 CCW from X (East))
    weights: np.array or None: weights to be assigned in the histo plot
    ax_polar: matplot axes object
    
    Outputs
    ------
    ax_polar
    """
    
    ### Synthetics
    
#    num_points=20000
#    trigo_angles=np.random.rand(num_points)*1*np.pi
#    weights=np.cos(trigo_angles*5)**2
#    
    ####
    
    ### Ini
    
    if ax_polar is None:
        fig = plt.figure()
        ax_polar = fig.add_axes([0.1 ,0.1, 0.8, 0.8],polar=True)
        
    trigo_angles=np.asarray(trigo_angles)
   
    if weights is None:
        weights=np.ones(trigo_angles.shape)
        
    #### Define angle_step
    
    width_bin=width_bin_deg*np.pi/180
    angle_bins=np.linspace(0,2*np.pi,int(2*np.pi/width_bin)+1)
    
    #### Duplicate and add 180°
#    
#    trigo_angles=np.concatenate((trigo_angles,trigo_angles+np.pi))
#    weights=np.concatenate((weights,weights))
    
    #### Make sure angles are inside 0 360°
    
    trigo_angles=trigo_angles % (2*np.pi)
    ### Histogram
    
    angle_histo,_=np.histogram(trigo_angles,bins=angle_bins,weights=weights)
    
    ### Define plot centers properly
    
    width_bar=np.diff(angle_bins)[0]
    angle_centers=angle_bins[0:-1]+width_bar/2
     #print(np.sum(angle_histo))
    
    #### Plot
    
    if facecolor is None:
        facecolor=np.array([30,144,255])/255
    

    ax_polar.bar(angle_centers,angle_histo, width=width_bar, bottom=0.0,facecolor=facecolor,edgecolor=edgecolor,lw=lw,alpha=alpha)
    ax_polar.set_facecolor('w') # Define BG color
    ax_polar.grid(True,color='0.7',alpha=1,lw=0.2)

    if text:
        angle_max=angle_centers[np.argmax(angle_histo)]
        histo_max=np.max(angle_histo)
        text_rotation=angle_max*180/np.pi-90
        ax_polar.text(angle_max,histo_max*1.05,'%i'%histo_max,rotation=text_rotation,va='center',ha='center',color=facecolor)
    
    return ax_polar

def ramp(x,c1,c2):
    """
    Function made to transform array to ramp 
                   ____________
                  /
    _____________/
                 | |
                c1 c2
    """
    
    y=np.zeros_like(x)
    
    c1,c2=np.sort([c1,c2])
    a=1/(c2-c1)
    b=-a*c1
    
    y[(x>=c1) & (x<=c2)]=a*x[(x>=c1) & (x<=c2)]+b
    
    y[x>c2]=1
    
    return y

def combine_weights(data_array,weights_list):
    """
    Function made to recompute an array from by combining columns and assigning them different weights such
    as:
        
    c=(w1*c1+w2*c2+w3*c3+...)/(w1+w2+w3+...)
    
    Inputs
    -----
    data_array: np.array: NxD [c1,c2,c3...]
    weights_list: list: list of scalar weights to be applied to each column [w1,w2,w3..]
    
    Output
    ------
    
    comb: np.array: Nx1, combination of data_array with proper weights applied
    
    """
    
    weights_list=np.asarray(weights_list)
    if len(weights_list)!=data_array.shape[1]:
        raise ValueError('Dimension problem')
    
    #### Diagonale Square matrix of weights 
    weights_matrix=np.diag(weights_list)
    
    ### Multiply
    
    sc1=np.dot(data_array,weights_matrix)
    
    ### Divide by sum of weights
    
    comb=np.sum(sc1,1)/(np.sum(weights_list))
    
    return comb


###############################################
######### Shear Wave Splitting ################
    

def gridsearch(data_array,shift,angle_rad,cut_s1=0,cut_s2=None):
    """
    *Function made to un-rotate and un-lag the data_array, it cuts the data and then compute the eigenvalues.
    
    *Cut is performed after rotation and lag to ensure no-bias (by padding with zeros for example). 
    
    *Shift is the opposite of lag, if shift negative, that means you are shifting the 
    second column to the left
    
    Inputs:
    -------
        data_array: Nx2 numpt array
        shift: samples (>0 means 2nd column will be shifted to the right)
        angle_rad: angle for basis rotation (positive CCW from East)
        cut_s1,cut_s2: Samples interval for window cutting
        
    Outputs:
    -------
        eigvals: list of eigen values (smallest last, descending order)
    """
    
    ### un-rotate and un_lag
    data_array=rotate_basis_2(data_array,angle_rad)
    data_array=shift_2(data_array,shift)
    
    ### cut
    
    data_array=cut(data_array,cut_s1,cut_s2)

    ### un-rotate

    eig_vals,_=cov_eig(data_array)
    
    return list(eig_vals)
    

def get_LAMBDAS(data_array,min_lag,max_lag,Nlags=100,Nangles=100,cut_s1=0,cut_s2=None):
    """
    Function made to compute the eigenvalues for multiple lags and angles.
    Outputs are of the form of meshgrids for easy plotting
    
    Inputs:
    ------
        data_array: Nx2 numpy array containing the data
        min_lag,max_lag: min/maximum lags in samples
        Nlags: Total number of lags to test
        Nangles: Total number of angles to test
        
    Outputs:
    -------
        LAMBDA1,LAMBDA2: Highest and smallest eigenvalues meshgrids
        LAGS: lag meshgrid (in samples)
        ANGLES: angles meshrid (in radian)
    """

    
    min_lag,max_lag=int(np.round(min_lag)),int(np.round(max_lag))
    
    ##### Define lags and angles arrays to be applied
    
    lags_array=sample_array(min_lag,max_lag,Nlags)
    angles_array=np.linspace(-np.pi/2,np.pi/2,Nangles) # in radian
    
    ### Mesh the lists
    
    LAGS,ANGLES=np.meshgrid(lags_array,angles_array)
    
    lags=list(LAGS.reshape(-1))
    angles=list(ANGLES.reshape(-1))
    
    #### Start gridsearch
    
    #start = time.time()
    eigs_list=[gridsearch(data_array,-lag,angle_rad,cut_s1=cut_s1,cut_s2=cut_s2) for lag,angle_rad in zip(lags,angles)]
    #end = time.time()
    
    #### Build LAMBDA arrays
    #print(end-start)

    eigs_array=np.array(eigs_list)
    LAMBDA1=np.reshape(eigs_array[:,0],LAGS.shape)
    LAMBDA2=np.reshape(eigs_array[:,1],LAGS.shape)

    return (LAMBDA1, LAMBDA2, LAGS,ANGLES)


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

def extrema(arr,size_perc=[10,10],flag_plot=False):
    #neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    
    foot_row=int(arr.shape[0]*size_perc[1]/100)
    foot_col=int(arr.shape[1]*size_perc[0]/100)

    neighborhood = np.ones((foot_row,foot_col),dtype=bool)
    
    local_min = (filters.minimum_filter(arr,footprint=neighborhood,mode=('wrap','constant'))==arr)
    
    background = (arr==0)
    
    eroded_background = morphology.binary_erosion(
         background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    
    ind_row,ind_col=np.where(detected_minima)
    min_values=arr[ind_row,ind_col]
    
    ### Sort
    
    ind_row=ind_row[np.argsort(min_values)]
    ind_col=ind_col[np.argsort(min_values)]
    min_values=min_values[np.argsort(min_values)]
    
    ### Create tuples
    
    min_list= list(zip(ind_row,ind_col,min_values))
    labels=[str(kk) for kk in range(len(min_list))]
    
    
    ### Plot if asked
    
    if flag_plot:
        fig,ax=plt.subplots()
        
        ax.imshow(arr, plt.cm.get_cmap('jet'))
        ax.plot(ind_col,ind_row,'+w')
        for kk,label in enumerate(labels):
            ax.text(ind_col[kk],ind_row[kk],label)
        
        
    return min_list

def extrema_old(mat,mode='constant',size=20): 
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
    
    contour_list=get_contours(array,[array_value+cont_step])
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


def process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                    min_lag_thres=None,max_lag_thres=None,
                   min_thres=0.5,min_numbers=2,cont_step=0.05,quality_thres=0.2,zoom_factor=[4,4],
                   flag_plot=False,ax=None):
    """
    *** Functiona made to process the LAMBDA2 mesh (or any kind of mesh). It selects the minima and outputs a list 
    of MinLambda Classes.
    The LAMBDA2 mesh is normalized from 0 to 1 and all thresholds are given in that normalized reference
    min_lag_thres insures that the miniouù mabda selected will not suffer from boarders effects (
    i.e. a minium located at the border of the grid)
    
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
        min_lag_thres: float: be sure the minimum lambda will be bigger than this
    
    Outputs
    -----
        MinLambas: list of MinLambda class
        ax: plt.axes: axes object if plotting
        
    TODO
    ----
    
    The Minimum is computed on the -90 to 90° Mesh grid, this results in a bias as only a few
    minima are picked close to -90 and 90°. The idea would be to replicate (extend) LAMBDA2 before computing
    the minma
        
    """
    ### Check
    
    if min_lag_thres is None:
          min_lag_thres=np.min(LAGS)
          
    if max_lag_thres is None:
          max_lag_thres=np.max(LAGS)
        
    #### Zoom to insure good quality of contours
    
    LAMBDA2=zoom(LAMBDA2,zoom_factor)
    LAMBDA1=zoom(LAMBDA1,zoom_factor)
    
    ### Resample LAG grid (Zoom doesn't interpolate properly)
    lagsz=np.linspace(np.min(LAGS),np.max(LAGS),LAGS.shape[1]*zoom_factor[1])
    LAGSZ=np.tile(lagsz, (zoom_factor[0]*LAGS.shape[0], 1))
    
    #ANGLESZ=zoom(ANGLES,zoom_factor)
    
    ### Normalize
        
    MESH=minmax2zeroone(LAMBDA2)
    
    #### Get quality mesh
    
    quality=quality_mesh(MESH,threshold=quality_thres)
    
    #### Find all minima (i.e. all local minima)
    
    min_list=extrema(MESH)
 
    ### Pre-select minima that are only inside min_lag_thres and max_lag_thres
    
    min_list=[(ind_row,ind_col,lambda_v) for (ind_row,ind_col,lambda_v) in min_list \
          if (LAGSZ[ind_row,ind_col]>=min_lag_thres) &  (LAGSZ[ind_row,ind_col]<=max_lag_thres)]

    ### Select properly the minima (below threshold and maximum of X minima)
    
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

    MinLambdas=[]

    kk=0
    for ind_row,ind_col,min_value in min_list:
        kk+=1
        
        ### Compute rectilinearity at the minimum
        
        rec=1-LAMBDA2[ind_row,ind_col]/LAMBDA1[ind_row,ind_col]
        
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
        MinSc.rec=rec
        
        MinLambdas.append(MinSc)
        
        #### Plot if asked
        
        if flag_plot:
                                        
            if ax is None: ### Plot things just once in the loop
                fig,ax=plt.subplots()
            if kk==1:
                ax.imshow(MESH,extent=extent_lag+extent_angle,origin='lower',cmap=plt.cm.get_cmap('jet'),aspect='auto') # GRID
                CS=ax.contour(MESH,extent=extent_lag+extent_angle,origin='lower',levels=[min_thres],colors='w',linestyles=':') # CONTOURS
                ax.clabel(CS, [min_thres],fmt='%.2f')
                ax.set_xlabel('LAG [samples]')
                ax.set_ylabel('ANGLE [rad]')
                ax.set_title('Quality = %.2f'%quality)
                
            ax.plot(lag,angle,'ow')
            ax.text(lag,angle,str(kk))
            ax.axvline(min_lag_thres,0,1,color='w',ls='--')
            ax.axvline(max_lag_thres,0,1,color='w',ls='--')
            
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

#######################################
######### Plotting tools ##############
#######################################

def ax_sws_diagnosis():
    """
    Create axes list used for SWS plots
    """
    
    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(left=0.1, right=0.8, top=0.95,bottom=0.55, hspace=0)
    

    ax_ini=plt.subplot(gs1[0,:])
    ax_un1=plt.subplot(gs1[1,:],sharex= ax_ini, sharey= ax_ini)
    ax_un2=plt.subplot(gs1[2,:],sharex= ax_ini, sharey= ax_ini)
    
    gs = gridspec.GridSpec(2,6 )
    gs.update(wspace=1.7,hspace=0.05,top=0.5,left=0.1,bottom=0.1)
    
    ax_lambda2=plt.subplot(gs[0:,0:2])
    ax_polar=plt.subplot(gs[0:,2:4])
    ax_polar_un=plt.subplot(gs[0:,4:6])
#    ax_ini = plt.subplot2grid(subplot_shape, (0, 0),colspan=6)
#    ax_un1 = plt.subplot2grid(subplot_shape, (1, 0),colspan=6)
#    ax_un2 = plt.subplot2grid(subplot_shape, (2, 0),colspan=6)
#    ax_lambda2 = plt.subplot2grid(subplot_shape, (3, 0), rowspan=4,colspan=2)
#    ax_polar = plt.subplot2grid(subplot_shape, (3, 2),rowspan=4,colspan=2)
#    ax_polar_un = plt.subplot2grid(subplot_shape, (3, 4),rowspan=4,colspan=2)

    return [ax_ini,ax_un1,ax_un2,ax_lambda2,ax_polar,ax_polar_un]




#############################
####### Others ##############
#############################
import socket
class retry():
    """
    Decorator that will keep retrying the operation after a timeout.

    Useful for remote operations that are prone to fail spuriously.
    """
    def __init__(self, retries: int, wait_time: float):
        self.retries = retries
        self.wait_time = wait_time

    def __call__(self, f: typing.Callable):
        def wrapped_f(*args, **kwargs):
            for kk in range(self.retries):
                try:
                    retval = f(*args, **kwargs)
                except:
                    time.sleep(self.wait_time)
                    print(self.wait_time)
                    continue
                else:
                    return retval
                raise

        return wrapped_f

def split_nlloc(nlloc_file,event_max,prefix=None,output_dir='split_nlloc'):
    """
    Function made to split nlloc file into smaller file to avoid crash when using obspy.read_nlloc_hyp for
    example. 
    
    Inputs
    ------
        nlloc_file: str: path to nlloc file that needs to be splitted
        event_max: float/int: the file will be splitted into files containing max_events
        prefix: str: prefix to be added to the splitted files
        output_dir: str: dir where you want the files to be stored
        
    Outputs
    -------
        file_names: list: list of the filenames generated
    """
    
    if prefix is None:
        prefix=nlloc_file.split('/')[-1]
        
    ### Check
    
    event_max=int(event_max)
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
            file_name='./'+output_dir+'/'+file_name
            file_names.append(file_name)
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

def spectrum(y_array,delta_t,nfft=None,flag_plot=False):
    """
    Compute spectrum
    
    Inputs
    ------
        data: np.array: Nx1 data array (y)
        npts: int: number of samples in frequency output, if None is determined
        flag_plot: Boolean
        
    Outputs
    ------
        freq,FFT: np.array: contains freq and power(abs(FFT))
        npts: int
    """
    
    y_array=np.asarray(y_array)
    time_array=np.linspace(0,delta_t*(len(y_array)-1),len(y_array))

    if nfft is None:
        nfft=_npts2nfft(len(y_array))
        
    nfft=int(nfft)
    
    ### Compute FFT
    
    FFTs=pow(abs(scipy.fftpack.fft(y_array, nfft)), 2)
    freqs = scipy.fftpack.fftfreq(nfft, delta_t)
    
    freq=freqs[0:int(nfft/2)]
    FFT=FFTs[0:int(nfft/2)]
    
    #### Plot if asked

    
    if flag_plot:
        fig,(ax_signal,ax_spectr)=plt.subplots(2,1)
        ax_signal.plot(time_array,y_array,'k')
        ax_signal.set_xlabel('Time or Samples')
        ax_spectr.plot(freq,FFT,'k')
        ax_spectr.set_xlabel('Frequency')
        
    ### Return
    
    return ([freq,FFT],nfft)
        
def dominant_freq(y_array,delta_t,npts=None):
    """
    Retrieve dominant frequency from data
    
    Input
    -----
        data: np.array: Nx2 data array (time,y) 
    """
    ([freq,FFT],npts)=spectrum(y_array,delta_t,npts=npts)
    
    dom_freq=freq[np.argmax(FFT)]
    
    return dom_freq
    

    
def get_dominant_period(data,fs,method='welch',nfft=256,num_wind=2,flag_plot=False):
    """
    Function made to compute the dominant period of a 1D np.array using either
    a classical FFT or a multitaper methods
    
    Output
    ------
        dom_period: float: dominant period in samples (can be float)
        dom_freq: float: dominant frequency
    
    """
    
    ### Compute
    
    if method=='welch':
        freq,spec=signal.welch(data, fs=fs,nperseg=len(data)/num_wind,scaling='density',nfft=nfft)
    elif method=='multitaper' or method=='mtspec':
        if not MULTITAPER_AVAILABLE:
            raise ImportError("multitaper package is not installed. Install with: pip install multitaper")
        # multitaper package usage: MTSpec(data, nw, k, dt)
        # nw = time-bandwidth product (typically 2-4), k = number of tapers (typically 2*nw - 1)
        nw = num_wind if num_wind > 1 else 2.5
        k = int(2 * nw - 1)
        dt = 1.0 / fs
        mtspec_obj = MTSpec(data, nw=nw, k=k, dt=dt, nfft=nfft)
        spec = mtspec_obj.rspec()  # Return spectrum
        freq = mtspec_obj.freq     # Return frequencies
    elif method=='classic':
        ([freq,spec],_)=spectrum(data,1/fs,nfft=nfft,flag_plot=False)
        
    ### Freq to period
      
    with np.errstate(divide='ignore'):
        period=1/freq*fs
    
    ### Get Dominant period
    
    dom_period=period[np.argmax(spec)]  
    dom_freq=1/dom_period*fs
    
    ### Plot if asked
    
    if flag_plot:
        x=np.arange(0,len(data))
        alpha=2*np.pi/dom_period
        sin=np.max(data)*np.sin(alpha*x)
            
        fig,ax=plt.subplots(3,1)
        
        ax[0].plot(x,data,color='k',ls='-',lw=1)
        ax[0].plot(x,sin,color='r',ls='--',lw=1)
        
        ax[1].plot(period,spec,color='k',ls='-',lw=1)
        ax[1].axvline(dom_period,color='r',ls='--')
        ax[1].text(0.5, 0.9,'Dom T=%.1f samples'%dom_period, horizontalalignment='center',
                  verticalalignment='center', transform=ax[1].transAxes)
        
        ax[2].plot(freq,spec,color='k',ls='-',lw=1)
        ax[2].axvline(dom_freq,color='r',ls='--')
        ax[2].text(0.5, 0.9,'Dom F=%.1f Hz'%dom_freq, horizontalalignment='center',
                  verticalalignment='center', transform=ax[2].transAxes)
        
    return (dom_period,dom_freq)


def get_rms(x,y,mode='norm'):
    """
    Compute RMS of two signals to check similarity
    It is possible to normalize it by the maximum range or not
    """

    rms=np.sqrt(np.mean((x-y)**2))

    # Normalize if asked
    
    if mode=='norm':
        cat_array=np.hstack((x,y))
        rms=rms/(np.max(cat_array)-np.min(cat_array))
  
    return rms

def rms_MinLambdas(xy_array,sw1,sw2,MinLambdas,mode='norm',flag_plot=True):
    
    """
    Function made to compute the RMS in between the unsplit x and y to assess
    quality of correction. The MinLambdas are then sorted based on minimum RMS.
    The lambda with minimum RMS is first. We return the sorted RMS
    
    Input
    -----
        xy_array: np.array
        sw1,sw2: float: samples to cut the signal
        MinLambdas: list: list of classes 
        mode:str: 'norm' to specfify how to compute RMS
        flag_plot: Bool
        
    Output
    ------
        MinLambdas: list
        
    """

    for kk in range(len(MinLambdas)):
        lag=MinLambdas[kk].lag
        angle=MinLambdas[kk].angle
        xy_unsplit,_=unsplit(xy_array,lag,angle,flag_plot=flag_plot)
        xy_unsplit_cut=xy_unsplit[sw1:sw2,:]
        rms_1=sws.get_rms(xy_unsplit_cut[:,0],xy_unsplit_cut[:,1],mode=mode)
        rms_2=sws.get_rms(xy_unsplit_cut[:,0],-xy_unsplit_cut[:,1],mode=mode)
        
        min_rms=np.min((rms_1,rms_2))
        
        MinLambdas[kk].rms=min_rms
        
        if flag_plot:
            fig=plt.gcf()
            ax=fig.get_axes()
            for i in range(len(ax)):
                ylim=ax[i].get_ylim()
                ax[i].vlines(sw1,ylim[0],ylim[1])
                ax[i].vlines(sw2,ylim[0],ylim[1])
                if i==1:
                    ax[i].text(0.1,0.1,'RMS=%f'%rms_1, horizontalalignment='center',
                          verticalalignment='center', transform=ax[i].transAxes)
                elif i==2:
                    ax[i].text(0.1,0.1,'RMS=%f'%rms_2, horizontalalignment='center',
                          verticalalignment='center', transform=ax[i].transAxes)
            
        
    ### Sort MinLambdas based on RMS rather than Lambda2 min
        
    rmss=[MinLambda.rms for MinLambda in MinLambdas]
    ind_rmss=np.argsort(rmss)
    
    MinLambdas=[MinLambdas[kk] for kk in ind_rmss] 
    
    return MinLambdas

#########################################    
########## Classes ######################
#########################################


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
        self.rec=None
        self.rms=None
        
    def __str__(self):
        out='Lambda value:%.2f\n' %self.lambda_value
        out+='Lag [samples]          : %.2f [+/- %.2f]\n'%(self.lag,self.lag_error)
        out+='Fast Dir. [Trigo. rad] : %.2f [+/- %.2f]\n'%(self.angle,self.angle_error)
        out+='Quality: %.2f\n'%self.quality
        out+='Rectlinearity unsplit: %.2f\n'%self.rec
        out+='RMS unsplit %.2f\n'%self.rms
        return out
    
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
        
        
    def __repr__(self):
        str_cmd=\
        '[quality : %f\n'%self.quality+\
        'lambda_value : %f\n'%self.lambda_value+\
        'lag : %f\n'%self.lag+\
        'angle : %f\n'%self.angle+\
        'rec : %f\n'%self.rec
        return str_cmd
    
    def match(self,
                  lambda_value=None,lambda_quality=None,lambda_lag=None,lambda_angle=None,
                  lambda_lag_error=None,lambda_angle_error=None,lambda_rec=None,
                  **kwargs):
        """
        Method made to return True if the MinLambdas class validate all the critera
        All arguments must start with 'Lambda'
        kwargs must be in 'quality', 'lambda_value', 'lag', 'angle', 'lag_error', 'angle_error', 'rec'
        
        Inputs
        ------
            lambda_*: list with 2 elements: Notify the range for the threshold to be applied
            
        Ouputs
        ------
            return: boolean: True if class passes the threshold
        """
        
        ### link parameters to class object
        
        class_dict={
                'lambda_value':lambda_value,
                'lag':lambda_lag,'lag_error':lambda_lag_error,
                'angle':lambda_angle,'angle_error':lambda_angle_error,
                'rec':lambda_rec,'quality':lambda_quality}    
        
        #### Return 
        
        if all(value is None for value in class_dict.values()):
            return True
        
        return all(  (getattr(self,key)>=min(list_val)) & (getattr(self,key)<=max(list_val) ) \
                   for (key, list_val) in class_dict.items() if list_val is not None )
        


class SWSobs():
    """
    Class to used to store Shear Wave Splitting obs
    One PS pair per station = One SWSobs class
    
    """
    def __init__(self):
        self.obs_id=None # Obs id (from STATION and P time)
        self.station_code=None # str 
        self.event_id=None # based
        self.event_lon=None
        self.event_lat=None
        self.event_depth=None
        self.p_time=None # obpsy UTCdatetime
        self.s_time=None #
        self.inc=None # incidence angle in deg (>0 CCW from vertical down)
        self.az=None # azimuth angle in deg (>0 CW from North)
        self.rec=None # rectilinearity
        self.dom_freq=None # Dominant frequency on x and y {'x':float,'y':float}
        self.dom_period=None # Dominant period on x and y {'x':float,'y':float} in seconds
        self.s_snr=None
        self.MinLambdas=None
        self.baz_trigo=None
        self.epi_dist=None
        self.hyp_dist=None
        
    def __eq__(self, other):
        """
        This must be defined together with hash so that 'set' only returns non similar events
        """

        return self.obs_id==other.obs_id
    
    def __hash__(self):
        """
        To allow the use of set
        """
        return hash(self.obs_id)
    

    def __str__(self):
        out='station %s\n'%self.station_code
        out+='S time   : %s\n'%self.s_time.strftime('%Y-%m-%dT%H:%M:%S')
        out+='longitude: %.4f\n'%self.event_lon
        out+='latitude : %.4f\n'%self.event_lat
        out+='depth[km]: %.2f\n'%self.event_depth
        out+='hyp_dist [km]: %.2f\n'%self.hyp_dist
        out+='Number of MinLambdas: %i\n'%len(self.MinLambdas)
        if hasattr(self,'baz_trigo'):
            if self.baz_trigo is not None:
                out+='baz_trigo: %.4f\n'%(self.baz_trigo)
            else:
                out+='baz_trigo: None\n'
        return out
        
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
        
    
    ##### Other Methods ######
    
    

    def match(self,obs_station=None,
                  obs_lon=None,obs_lat=None,obs_depth=None,
                  obs_stime=None,obs_baz_trigo=None,obs_epi_dist=None,
                  obs_hyp_dist=None):
        """
        Method made to return True if the SWSobs class validate all the critera
        All arguments must start with 'obs'
        To be used by SWScat.select
        
        Inputs
        ------
            obs_*: list with 2 elements: Notify the range for the threshold to be applied
            obs_station: str,list of str: Stations to be selected
        Ouputs
        ------
            return: boolean: True if class passes the threshold
        UsedIN
        ------
            sws_methods.SWScat.select
        """
        
        ### If all None then match is True   
        
        keys=[obs_station,obs_lon,obs_lat,obs_depth,obs_stime,obs_baz_trigo,obs_epi_dist,
              obs_hyp_dist]
        if all(key is None for key in keys):
            return True
        
        
        obs_station=tolist(obs_station)
        obs_dict={
                'event_lon':obs_lon,'event_lat':obs_lat,'event_depth':obs_depth,
                's_time':obs_stime,'baz_trigo':obs_baz_trigo,'epi_dist':obs_epi_dist,
                'hyp_dist':obs_hyp_dist,}
        obs_dict_exact={
                'station_code':obs_station}
    
        ### Define booleans list
        
        bool_obs=[]
        for (key, list_val) in obs_dict.items():
            if list_val:
                bool_obs.append((getattr(self,key)>=min(list_val)) & (getattr(self,key)<=max(list_val) ))
            else:
                bool_obs.append(True)
                
        bool_obs_exact=[]
        for (key, list_val) in obs_dict_exact.items():
            if list_val[0]:
                bool_obs_exact.append(getattr(self,key) in list_val)
            else:
                bool_obs_exact.append(True)
                       
        return all(bool_obs+bool_obs_exact)
    
    
    def minlambda_select(self,param,mode):
        """
        Method made to select the only one MinLambda element that has the max/min
        parameter value (lag,rec,rms)
        
        Input
        -----
            param: str: One of MinLambda class attributes (MinLambda._dict_)
            mode: str: max or min desired
        Output
        ------
            select_list: list: List containing one element that fullfilled the criteron
        """
        
        ### Copy
        
    
        minlambda_list=self.MinLambdas
        param_list=[getattr(x,param) for x in minlambda_list]
        
        if mode=='max':
            select_list=[x for x in minlambda_list if getattr(x,param)==max(param_list) ]
        elif mode=='min':
            select_list=[x for x in minlambda_list if getattr(x,param)==min(param_list) ]
        else:
            raise ValueError('Only max or min allowed')
            
        return select_list
      
    def get_baz_trigo(self,sta_lon,sta_lat,ini_lon,ini_lat):
        if getattr(self,'baz_trigo',None) is None:
            [x_event,x_sta],[y_event,y_sta]=gproj.ll2xy([self.event_lon,sta_lon],[self.event_lat,sta_lat],ini_lon,ini_lat)
            baz_trigo=np.arctan2((y_event-y_sta),(x_event-x_sta))
            baz_trigo=baz_trigo+2*np.pi if baz_trigo<0 else baz_trigo
            self.baz_trigo=baz_trigo
        
        return self.baz_trigo
    

    def get_epi_dist(self,sta_lon,sta_lat,ini_lon,ini_lat):
        if getattr(self,'epi_dist',None) is None:
            [x_event,x_sta],[y_event,y_sta]=gproj.ll2xy([self.event_lon,sta_lon],[self.event_lat,sta_lat],ini_lon,ini_lat)
            epi_dist=np.sqrt((x_event-x_sta)**2 + (y_event-y_sta)**2 )
            self.epi_dist=epi_dist

        return self.epi_dist       

    def get_hyp_dist(self,sta_lon,sta_lat,sta_z,ini_lon,ini_lat):
        """
        Function made to compute the hypocenter distance
        
        UsedIn
        ------
        sws_methods.compute_hyp_dist
        """
        if getattr(self,'hyp_dist',None) is None:
            epi_dist=self.get_epi_dist(sta_lon,sta_lat,ini_lon,ini_lat)
            hyp_dist=np.sqrt((epi_dist)**2 + (sta_z-self.event_depth)**2 )
            self.hyp_dist=hyp_dist
        return self.hyp_dist

    
#def LAMBDA2lagangle(LAMBDA2, LAGS,ANGLES,error=True,flag_plot=False):
#    """
#    Function made to retrieve lag and phi giving the minimum lambda.
#    
#    Inputs:
#        ---
#        LAMBDAS,LAGS, ANGLES: np.meshgrids: meshgrids generated by get_LAMBDAS
#        error: bool: Get errors on lags and phi
#        flag_plot: bool: True for plotting
#        
#    Outputs:
#        ---
#        
#    """
#    factor=[2,4]
#    LAMBDA2=zoom(LAMBDA2,factor)
#    LAGS=zoom(LAGS,factor)
#    ANGLES=zoom(ANGLES,factor)
#    
#    #### Lag array
#    
#    lag_array=LAGS[0,:]
#    lag_extent=[np.min(lag_array),np.max(lag_array)]
#    
#    #### Find angles and lags associated with mininmum
#    
#    ind_min=np.unravel_index(LAMBDA2.argmin(), LAMBDA2.shape)
#    
#    lambda2_min=LAMBDA2[ind_min]
#    lag_min=LAGS[ind_min]
#    angle_min=ANGLES[ind_min]
#    
#    #### Compute Error (Walsh)
#    
#    lag_err=0
#    angle_err=0
#    
#    if error:
#    
#        Nsamples=1
#        
#        #### Get lambda 0.95
#        
#        k=2 # num parameters
#        ratio=0.53 # degrees freedom/samples=0.53 (Walsh)
#        v=ratio*Nsamples
#        v=27
#        Fdistri=10#3.3852 # F(2,27-2) http://www.socr.ucla.edu/applets.dir/f_table.html#FTable0.05
#        lambda2_95=lambda2_min * (1 + (k/(v-k))* Fdistri)
#        print('lambda2_min=',lambda2_min)
#        print('lambda2_95=',lambda2_95)
#        LAMBDA2_CONF=LAMBDA2/lambda2_95
#        
#        #### Replicate LAMBDA2 vertically to ensure periodicity, from -90+90 to -270+270
#        
#        LAMBDA2_CONFPER=np.vstack((LAMBDA2_CONF,LAMBDA2_CONF[1:-1,:],LAMBDA2_CONF))
#        
#        #### There are faster way to compute contour without plotting it (UR)
#        
#        plt.ioff()
#       
#        contour_set=plt.contour(LAMBDA2_CONFPER,extent=lag_extent+[-3*np.pi/2,3*np.pi/2],origin='lower',
#                                levels=1)
#        plt.ion()
#        
#        #### Retrieve contour closest to mid point$
#        
#        cont_array=select_contour(lag_min,angle_min,contour_set)
#  
#        lag_err,angle_err=size_box(cont_array)
#        
#        if flag_plot:
#            plt.subplots()
#            plt.imshow(LAMBDA2_CONFPER,extent=lag_extent+[-3*np.pi/2,3*np.pi/2],origin='lower',cmap=plt.cm.get_cmap('jet'),
#                       aspect=10)
#            plt.colorbar()
#            plt.contour(LAMBDA2_CONFPER,extent=lag_extent+[-3*np.pi/2,3*np.pi/2],origin='lower',
#                        levels=1,colors='w')
#            plt.ylim([-np.pi/2,np.pi/2])
#           
#                
#    ### Return
#    
#    return (lag_min,angle_min,lag_err,angle_err,contour_set)
#
#
#        
#def get_lagangle(data_array,min_lag,max_lag,Nlags=100,Nangles=100,error=False,cut_s1=0,cut_s2=None):
#    
#
#    (LAMBDA1, LAMBDA2, LAGS,ANGLES)=get_LAMBDAS(data_array,min_lag,max_lag,Nlags=Nlags,Nangles=Nangles,
#    cut_s1=cut_s1,cut_s2=cut_s2)
#    (lag_min,angle_min,lag_err,angle_err)=LAMBDA2lagangle(LAMBDA2, LAGS,ANGLES,error=error)
#    
#    return (lag_min,angle_min,lag_err,angle_err)
#
# 
#(lag_min,angle_min,lag_err,angle_err)=get_lagangle(data_sp,0,200,Nlags=100,Nangles=100,cut_s1=0,cut_s2=None)


