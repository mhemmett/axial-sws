#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:13:17 2017

@author: baillard
"""

import os 
from PyPDF2 import PdfFileMerger
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import typing
import time


from datetime import datetime, timedelta
from obspy import UTCDateTime

def full_path_list(directory_file,start_with='',end_with=''):
    """
    Funtion returning the list of files present in a given directory
    start_with and end_with could be arrays of string
    """
    
    directory_file=directory_file.rstrip('/')
    
    ### Initialize
    
    file_list=[]
    
    ### Check
    
    if not isinstance(start_with,list) : start_with=[start_with]
    if not isinstance(end_with,list) : end_with=[end_with]
    
    ### Loop
    
    for start_w in start_with:
        for end_w in end_with:
            for file in os.listdir(directory_file):
                if file.startswith(start_w) and file.endswith(end_w):
                    file_list.append(directory_file+'/'+file)
    
  ### Sort
    file_list=sorted(set(file_list))
    return file_list

def listindir(directory,regexp=[''],full=True):
    """
    Replacemet to full-path_list
    
    directory: str
    regexp: list of matching str
    full: boolean, indicate if full path has to be given
    
    Return:
    -------
    
    List of strings
    
    """
    directory=directory.rstrip('/')
    list_files=[]
    for file in os.listdir(directory):
        if regexp==None:
            list_files.append(file)
        elif all(substring in file for substring in regexp):
            list_files.append(file)
     
    list_files=sorted(list_files)
    if full==True:
        list_files=[directory+'/'+file for file in list_files]
    return list_files

def merge_pdfs(list_pdf,output_pdf):
    merger = PdfFileMerger()
    
    for pdf in list_pdf:
        merger.append(open(pdf, 'rb'))
    
    
    with open(output_pdf, 'wb') as fout:
        merger.write(fout)
    
def get_max_field(elem_list):
    """
    function made to return the field width that should be used
    """
    
    max_field=max([len(str(x)) for x in elem_list])
    
    return max_field

def convert_values(input_array,thresholds,new_values):
    
    output_array=np.copy(input_array)
    
    min_val=np.min(input_array)
    max_val=np.max(input_array)
    
    thresholds=np.append(min_val,thresholds)
    thresholds=np.append(thresholds,max_val)
    
    for kk in range(len(thresholds)-1):
        output_array[ (input_array>=thresholds[kk]) & (input_array<=thresholds[kk+1]) ] = new_values[kk]
        
    return output_array


def get_index(list_lines, pattern):
    """
    Function made to retrieve index of lines that contain patterns
    """
    index=[i for i,line in enumerate(list_lines) if pattern in line]
    if len(index)==1:
        index=index[0]
        
    return index
        
def figs2pdf(pdf_file,figure_list=plt.get_fignums()):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_file)
    
    
    for fig in figure_list: ## will open an empty extra figure :(
        pdf.savefig( fig )
     
    pdf.close()
    
def get_colors(num_colors,colormap='jet',flag_plot=False):
    """
    Function made to generate a lits of RGB from a defined colormap
    """

    len_cmap=500
    cmap = plt.cm.get_cmap(colormap, len_cmap)
    
    step=int(len_cmap/(num_colors))
    
    
    rgb_list=[]
    index=int(step/2)
    for kk in range(num_colors):
        print(index)
        rgb_list.append(cmap(index)[:3])
        index+=step
        
     
    if flag_plot:
        
        x=np.linspace(1,10,num_colors)
        y=np.ones((num_colors,1))
        
        for kk in range(num_colors):
            plt.plot(x[kk],y[kk],'o',color=rgb_list[kk])
    
    return rgb_list
        
            
def getPickForArrival(picks, arrival):
    """
    searches list of picks for a pick that matches the arrivals pick_id
    and returns it (empty Pick object otherwise).
    """
    pick = None
    for p in picks:
        if arrival.pick_id == p.resource_id:
            pick = p
            break
    return pick   

def comp_dist(x,y):
    
    ds=np.sqrt((np.diff(x))**2+(np.diff(y))**2)
    ds=np.append(0,ds)
    ds=np.cumsum(ds)
    
    return ds

def smooth_curve(x_in,y_in,periodic=False,num_points=500,smoothness=0.5,flag_plot=False,k=3):
    """
    keywords: smooth, smoothness, 2D, curve
    Function made to smooth a curve
    x_in,y_in: 1D np.array with x and y
    periodic: specify if points define a closed polygon
    num_points: float, number of points
    smoothness: float, specify the degree of smoothing 0= no smoothing
    
    Return
    -----
    
    x_out,y_out
    """
    from scipy.interpolate import splprep,splev
    
    
    x_in=np.asarray(x_in)
    y_in=np.asarray(y_in)
    
    ### Compute distance along curve
    ds=comp_dist(x_in,y_in)

    #### Check where doublons (two consectuive points at the same position will cause
    #### splprep to crash)
    ind_keep=np.where(np.diff(ds)>0)
    ####
    
    if periodic==False:
        per=0
        xp=np.r_[x_in[ind_keep],x_in[-1]]
        yp=np.r_[y_in[ind_keep],y_in[-1]]
    else:
        per=1
        xp=np.r_[x_in[ind_keep],x_in[-1],x_in[0]]
        yp=np.r_[y_in[ind_keep],y_in[-1],y_in[0]]
        

    ### Build a spline representation of the contour
    dist_along=comp_dist(xp,yp)
    spline_val, u = splprep([xp, yp], u=dist_along,per=per,s=smoothness,k=k)
    interp_d = np.linspace(dist_along[0], dist_along[-1], num_points)
    x_out, y_out = splev(interp_d, spline_val)
    
    if flag_plot:
        plt.figure()
        p2,=plt.plot(x_out, y_out, '-or',label='Smoothed')
        p1,=plt.plot(x_in,y_in,'-ok',label='Original')
       
        ax=plt.gca()
        ax.legend()
        plt.axis('equal')


    return x_out,y_out
        
def is_in_polygon(X_val,Y_val,x_polygon,y_polygon,flag_plot=False):
    
    """
    keywords: polygon, inside, point
    Function made to see if points are in polygon or not
    
    x_val,y_val: coordintes of points to evaluate, can be 2D arrays
    x_polygon,y_polygon: coordinates of polygon
    
    Return
    x_inside,y_inside: coordinates of points inside polygon
    boolean: np.array of booleans, True for values inside polygon
    """
    
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    
    X_val=np.asarray(X_val)
    Y_val=np.asarray(Y_val)
    
    ini_shape=X_val.shape
    
    x_val=X_val.reshape(-1)
    y_val=Y_val.reshape(-1)  
    
    #print(ini_shape,x_val)

    list_polygon=[(x,y) for x,y in zip(x_polygon,y_polygon)]
    polygon=Polygon(list_polygon)
    
    x_inside=[]
    y_inside=[]    
    boolean=np.full((len(x_val),),False)
    for index,(x_s,y_s) in enumerate(zip(x_val,y_val)):
        
        point_value = Point(x_s,y_s)
        if polygon.contains(point_value):
            x_inside.append(x_s)
            y_inside.append(y_s)
            boolean[index]=True
            
    ### Plot if asked
    
 
    
    if flag_plot:
        fig,ax=plt.subplots()
        ax.plot(x_val,y_val,'ok')
        ax.plot(x_polygon,y_polygon,'--r')
        ax.plot(x_inside,y_inside,'or',mfc='r')
        plt.axis('equal')
        

    boolean=boolean.reshape(ini_shape)

    
    x_inside=X_val[boolean]
    y_inside=Y_val[boolean]
                
    return x_inside,y_inside,boolean


def read_datafile(file_in):
    """
    Function made to read data files that are composed of columns (x,y,z...)
    and with different contours sections separated by '>' line (like in GMT).
    Depending on the number of contours the output is either an array or a list of arrays.
    Columns in a single array correspond to columns in the file
    Can only be column containing floats
    
    file_in example
    ----------------
        ### #### ###
        ### #### ###
        ### #### ###
        >
        ### #### ###
        ### #### ###
        
    Input
    -----
        file_in: str: name of the data file
    
    Output
    -----
        data: np.array or list of np.array
    """
    data_list=[]
    with open(file_in,'rt') as fic:
        lines = fic.read().splitlines()
        num_col=len(lines[0].split())
        ### Check number of column
        single_list = [ [] for i in range(num_col) ]
    
        for line in lines:
            if line[0]=='>':
                single_array=np.asarray(single_list).transpose()
                data_list.append(single_array)
                single_list = [ [] for i in range(num_col) ]
            else:
                values=[float(x) for x in line.split()]
                for kk in range(num_col):
                    single_list[kk].append(values[kk])
                    
        ### Add if last lines ommitts '>'
        if lines[-1]!='>':
            single_array=np.asarray(single_list).transpose()
            data_list.append(single_array)
            
        ### Return np.array insetad of list if len list=1

    return data_list

    
    
def read_chadwick_file(file_in):
    
    def split_array(single_array):
        v=[]
        for kk in range(single_array.shape[0]):
            value_str=str(single_array[kk,0])+str(single_array[kk,1])
            v.append(value_str)
            
        v=np.array(v)
        
        ind_d=[]
        for kk in range(len(v)):
            ind_val=np.where(v==v[kk])[0]
            if len(ind_val)>=2:
                ind_d.extend(list(ind_val))
                
        if len(ind_d)==0:
            return [single_array]
        ind_d=np.sort(np.unique(ind_d))
        
        
        new_list=[]
        kk=0
        while kk<=len(ind_d)-1:
        
            new_list.append(single_array[ind_d[kk]:ind_d[kk+1],:])
            kk+=2
            
        return new_list
        

    with open(file_in,'rt') as fic:
        lines = fic.read().splitlines()
        num_col=3
        ### Check number of column
        headers=lines[0].split(',')
        col_id=[kk for kk in range(len(headers)) if headers[kk]=='ORIG_FID'][0]
        col_lon=[kk for kk in range(len(headers)) if headers[kk]=='LONGITUDE'][0]
        col_lat=[kk for kk in range(len(headers)) if headers[kk]=='LATITUDE'][0]
        single_list = [ [] for i in range(num_col) ]
    
        for line in lines[1:]:
                values=[float(x) for x in line.split(',')]
                id_val=values[col_id]
                lon=values[col_lon]
                lat=values[col_lat]
             
                single_list[0].append(id_val)
                single_list[1].append(lon)
                single_list[2].append(lat)
                    
    data_array=np.asarray(single_list).transpose()
    

    unique_ids=np.unique(data_array[:,0])
    
    data_list=[]
    for id_col in unique_ids:
        single_array=data_array[data_array[:,0]==id_col,1:]

        new_list=split_array(single_array)

        data_list.extend(new_list)
    
    return data_list

###############################################################
    ##### Convert times
###############################################################

def datenum2datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=days) \
           - timedelta(days=366)
           
def datetime2datenum(dt):
    """
    Converts python datetime object to matlab datenum (days since year 0)
    
    NB: ordinal gives int days since year 1 + day 1
    """
    greg_days=dt.toordinal()
    diff_timedelta=(dt-datetime.fromordinal(greg_days))   
    frac_days=diff_timedelta.days+diff_timedelta.total_seconds()/86400
    return greg_days+frac_days+366

def datenum2obspytime(datenum):
    """
    Convert matlab datenum to obspytime
    """
    
    dt=datenum2datetime(datenum)
    return UTCDateTime(dt)

def obspytime2datetime(obspytime):
    
    return obspytime.datetime

def obspytime2datenum(obspytime):
    
    dt=obspytime2datetime(obspytime)
    
    return datetime2datenum(dt)

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
            for _ in range(self.retries):
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
    
    