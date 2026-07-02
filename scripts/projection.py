#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:13:17 2017

@author: baillard
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import utm
import pyproj

def rotate_z(data,angle_deg,flag_plot=None):
    """
    Function made to change basis along the vertical direction, angle deg is 
    counter clockiwse from x (trigo)
    """
    
    #### Check data
    
    if not isinstance(data,np.ndarray):
        logging.error('Data is not an array')
        return
    
    try:
        num_col=data.shape[1]
    except:
        num_col=data.size
    
    if num_col not in [2,3]:
        logging.error('Data has wrong shape, num col is %i'%num_col)
        return
    
    angle_rad=angle_deg*np.pi/180
    
    ### Rotation (Change of basis matrix)
    
    sina = np.sin(angle_rad)
    cosa = np.cos(angle_rad)
    
    ##### Check Matrix
    
    if num_col==2:
        Rz=np.array([[cosa , sina],
                 [-sina, cosa]])
    else:
        Rz=np.array([[cosa , sina, 0],
                 [-sina, cosa,0],
                 [0 ,0 ,1]])
        
    data=data.transpose()
    
    ### Apply product
    
    new_data=np.dot(Rz,data).transpose()
    
    ### Return and plot
    
    if flag_plot:
        
        plt.figure()
        
        plt.subplot(121)
        plt.title('Un-rotated')
        plt.plot(data[0,:],data[1,:],'or')
        plt.axis('equal')
        plt.grid('on')
        
        plt.subplot(122)
        plt.title('Rotated with angle %.2f (CCW from x)'%(angle_deg))
        plt.plot(new_data[:,0],new_data[:,1],'ob')
        plt.axis('equal')
        plt.grid('on')
    
    
    return new_data

def translate(data,xt=0,yt=0,zt=0,flag_plot=None):
    """
    Function made to translate to +x,+y,+z
    """
    if not isinstance(data,np.ndarray):
        raise TypeError('Provide array as first argument')
    
    try:
        num_col=data.shape[1]
    except:
        num_col=data.size
    
    if num_col==2:
        shift_array=np.array([xt,yt])
    else:
        shift_array=np.array([xt,yt,zt])
    
    new_data=data+shift_array
    
    ### Plot if asked
    
    if flag_plot:
    
        plt.figure()
        
        plt.subplot(121)
        plt.title('Un-translated')
        plt.plot(data[:,0],data[:,1],'or')
        plt.axis('equal')
        plt.grid('on')
        
        plt.subplot(122)
        plt.title('translated with shift dx=%.2f and dy=%.2f '%(xt,yt))
        plt.plot(new_data[:,0],new_data[:,1],'ob')
        plt.axis('equal')
        plt.grid('on')
    
    return new_data

def recenter(data,xo=0,yo=0,zo=0,flag_plot=None):
    
    new_data=translate(data,-xo,-yo,-zo)
    
     ### Plot if asked
    
    if flag_plot:
    
        plt.figure()
        
        ax1=plt.subplot(121)
        plt.title('Un-translated')
        plt.plot(data[:,0],data[:,1],'or')
        plt.axis('equal')
        plt.grid('on')
        
        ax2=plt.subplot(122,sharex=ax1,sharey=ax1)
        plt.title('recentered xo=%.2f and yo=%.2f '%(xo,yo))
        plt.plot(new_data[:,0],new_data[:,1],'ob')
        plt.axis('equal')
        plt.grid('on')
        
    return new_data

def inbox(data,**kwargs):
    
    try:
        num_col=data.shape[1]
    except:
        num_col=data.size
        
    x_border=kwargs.get('x_border')
    y_border=kwargs.get('y_border')
    z_border=kwargs.get('z_border')
    flag_plot=kwargs.get('flag_plot')
 
    
    if x_border is None:
        x_border=[np.min(data[:,0]),np.max(data[:,0])]
    if y_border is None:
        y_border=[np.min(data[:,1]),np.max(data[:,1])]
    if z_border is None:
        z_border=[np.min(data[:,2]),np.max(data[:,2])]
        
    logging.info('x_border: %s'%(str(x_border)))
    logging.info('y_border: %s'%(str(y_border)))
    logging.info('z_border: %s'%(str(z_border)))
        
    bool_x=np.logical_and(data[:,0]>=x_border[0],data[:,0]<=x_border[1])
    bool_y=np.logical_and(data[:,1]>=y_border[0],data[:,1]<=y_border[1])
    bool_z=np.logical_and(data[:,2]>=z_border[0],data[:,2]<=z_border[1])

    bool_array=bool_x & bool_y & bool_z
    
    new_data=data[bool_array,:]
    
    if flag_plot is True:
        plt.figure()
        
        plt.title('Un-translated')
        plt.plot(data[:,0],data[:,1],'or')
        plt.plot(new_data[:,0],new_data[:,1],'ob')
        plt.axis('equal')
        plt.grid('on')

    
    return new_data,bool_array

def project(data,center,angle_deg,len_prof,width_prof):
    """
    Function made to project x,y,z data onto cross section p
    len_prof= list: [lenght_left, length_right]
    width_prof= list: [width_bottom, width_top]
    data can have a lot of columns not only three but it supposes that the first ones are x,y,z
    """
    
    rest_data=data[:,3:]
    keep_data=data[:,:3]
    
    if not isinstance(keep_data,np.ndarray):
        raise TypeError('Provide array as first argument')
        
    if keep_data.ndim!=2:
        raise TypeError('Data array must be 2D')
    
    
    num_initial=keep_data.shape[0]
    logging.info('Number of initial events %i'%num_initial)
    
    #### Do the projection
    
    proj_data=recenter(keep_data,center[0],center[1],0)
    proj_data=rotate_z(proj_data,angle_deg)
    proj_data,bool_array=inbox(proj_data,
                               x_border=[-len_prof[0],len_prof[1]],
                               y_border=[-width_prof[0],width_prof[1]])
                               
    ### Rebuild data

    final_data=np.hstack((proj_data,rest_data[bool_array,:]))      
    ### Return

    num_final=proj_data.shape[0]
    logging.info('Number of initial events %i'%num_final)
    
    return final_data,bool_array
    
def xy2ll(x,y,ini_lon,ini_lat,proj='utm'):
    
    """
    Transform x/y coordinates to lon/lat coordinates
    
    UTM and tmerc (transverse mercator) can be slightly different be careful
    
    proj can be 'utm' or 'tmerc'
    """
    
    x=np.array(x)
    y=np.array(y)

    if proj=='utm':
        zone_utm='%i'%utm.from_latlon(ini_lat, ini_lon)[2]+\
        utm.from_latlon(ini_lat, ini_lon)[3]
        P1=pyproj.Proj(proj=proj, zone=zone_utm, ellps='WGS84')
        
    elif proj=='tmerc':
        P1=pyproj.Proj(proj=proj, ellps='WGS84',lon_0=ini_lon,lat_0=ini_lat)
        
    else:
        raise ValueError('Wrong proj given')

    
    ### Convert
    
    
    x_o,y_o=P1(ini_lon ,ini_lat)
    
    new_x=x*1000+x_o
    new_y=y*1000+y_o

    lon,lat=P1(new_x,new_y,inverse=True)
    
    return lon,lat
    
def ll2xy(lon,lat,ini_lon,ini_lat,proj='utm'):
    
    
    lon=np.asarray(lon)
    lat=np.asarray(lat)
#    if isinstance(lon,list):
#        lon=np.array(lon)
#        
#    if isinstance(lat,list):
#        lat=np.array(lat)
    
    
    if proj=='utm':
        zone_utm='%i'%utm.from_latlon(ini_lat, ini_lon)[2]+\
        utm.from_latlon(ini_lat, ini_lon)[3]
        P1=pyproj.Proj(proj=proj, zone=9, ellps='WGS84')
        #zone = zone_utm
        
    elif proj=='tmerc':
        P1=pyproj.Proj(proj=proj, ellps='WGS84',lon_0=ini_lon,lat_0=ini_lat)
        
    else:
        raise ValueError('Wrong proj given')
        
    x_o,y_o=P1(ini_lon ,ini_lat)
    
    x,y=P1(lon,lat)
    
    new_x=(x-x_o)/1000
    new_y=(y-y_o)/1000

    return new_x,new_y

def is_in_circle(x,y,xo,yo,radius,flag_plot=None):
    """
    Function made to return x,y and boolean of elements that are isinde the 
    circle of center xo and yo with radius.
    x and y are numpy array, xo and yo and radius are floats
    if flag_plot=true, we will plot elements that in the circle
    """
    
    ### Compute val
    
    val=(x-xo)*(x-xo)+(y-yo)*(y-yo)

    ### Get boolean
    
    boolean=val<=radius**2
    x_sel=x[boolean]
    y_sel=y[boolean]
    
    ### Print
    
    logging.info('Number of elemnents selected is %i'%(len(x_sel)))
    
    if flag_plot:
        
    ### Figure
    
        plt.figure()
        plt.plot(x,y,'.',color='g')
        plt.plot(x_sel,y_sel,'.',color='r')

        plt.axis('equal')
    
        
    return x_sel,y_sel,boolean

def is_in_polygon(x,y,x_p,y_p,flag_plot=False):
    """
    Function made to select elements that are inside a given polygon
    
    Input:
    x,y: arrays
        input data
    x_p,y_p, arrays
        coordiantes of the polygon
    """
    from matplotlib import path

    x=np.array(x)
    y=np.array(y)
    
    ### In Path?
    
    p = path.Path(np.column_stack((x_p,y_p)))
    bool_array=p.contains_points(np.column_stack((x,y)))
    x_sel=x[bool_array]
    y_sel=y[bool_array]

    if flag_plot:
        fig,ax=plt.subplots()
        ax.plot(x,y,'+b')
        ax.plot(x_p,y_p,'-r')
        ax.plot(x_sel,y_sel,'ok')
        
    return x_sel,y_sel,bool_array

def get_circle(xo,yo,radius,flag_plot=False):
    """
    Function made to get the coordinates of a circle
    
    Parameter
    ---------
    
    xo,yo,radius: float
        values of center and radius
    """
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2*np.pi, 100)
    
    # the radius of the circle
    
    
    # compute x1 and x2
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    
    x=x+xo
    y=y+yo
    
    if flag_plot:
             
        plt.plot(x,y,'r')
        plt.axis('equal')
        
    return x,y
        
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(hxy,z)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    rcos_theta = r * np.sin(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.cos(el)
    return x, y, z      


def cart2pol(x, y,x0=0,y0=0):
    """
    returns angle between -np.pi and np.pi
    """
    rho = np.sqrt((x-x0)**2 + (y-y0)**2) # radius
    phi = np.arctan2(y-y0, x-x0)
    return(rho, phi)

def pol2cart(rho, phi,x0=0,y0=0):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x+x0, y+y0)
    
def az2trigo(az_angle):
    """
    Function made to convert azimuth angle to trigo angle between 0° and 360°
    """
    
    trigo_angle=90-az_angle
    trigo_angle=trigo_angle%360
    
    return trigo_angle

    
def trigo2az(trigos,unit='degree'):
    """
    Transform values from trigo convention (CCW from x) to azimuth convention
    (CW from N)
    """
    
    if unit=='radian':
        trigos=trigos*180/np.pi
        azimuths=(-trigos+90)%360
        azimuths=azimuths*np.pi/180
    else:
        azimuths=(-trigos+90)%360
    
    return azimuths
    
    

    


   
    