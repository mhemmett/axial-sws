#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New version of ARTICLE_deformation_util.py (Christian Baillard, 2019), fixed to run in this
repo:
  - broken imports (from general import GMT / import general.projection) pointed at our real
    flat modules (GMT.py, projection.py directly in scripts/) - the 'general' package they
    referenced never existed here.
  - plt.cm.get_cmap() replaced with plt.colormaps[...] (removed in matplotlib >=3.9).

All functions used here (ggmt.get_cax, ggmt.read_stationfile, gproj.cart2pol,
gproj.ll2xy, gproj.trigo2az, sws_methods.get_ax_inset/get_ax_polarinsets/pol2cart/cart2pol)
were confirmed to already exist in GMT.py/projection.py/sws_methods.py - no stubbing needed.

Original docstring: Created on Wed May 29 14:22:55 2019, @author: baillard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys
import glob
#import pickle
#import glob
import matplotlib as mpl

import deformation_util as adutil
from obspy import UTCDateTime
#import shearwavesplit as sws
#import sws_methods as swm
import GMT as ggmt
from sws_methods import get_ax_inset,get_ax_polarinsets,pol2cart, cart2pol
from scipy.optimize import curve_fit
import projection as gproj


def read_disp_file(disp_file,flag_plot=False):
    """
    Function made to read the deformaton file given by DMODELS,
    axial_comb_V0.m

    Input
    -----
        def_file: str: deformation file

    Output
    ------
        X,Y,Z: np.array: coordinates of the grid [km]
        Ux,Uy,Uz: np.array: displacement in given direction [m]
    """
    with open(disp_file,'rt') as fic:
        lines=fic.readlines()

    data_lines=lines[2:]
    nx,ny,nz=[int(x) for x in lines[0].split()[1::2]]

    x,y,z,ux,uy,uz=[],[],[],[],[],[]
    for line in data_lines:
        [xs,ys,zs,uxs,uys,uzs]=[float(x) for x in line.split()]
        x.append(xs)
        y.append(ys)
        z.append(zs)
        ux.append(uxs)
        uy.append(uys)
        uz.append(uzs)

    ### Reshape

    X=np.reshape(x,(nx,ny,nz),order='F') # order is to follow matlab convention
    Y=np.reshape(y,(nx,ny,nz),order='F')
    Z=np.reshape(z,(nx,ny,nz),order='F')
    Ux=np.reshape(ux,(nx,ny,nz),order='F')
    Uy=np.reshape(uy,(nx,ny,nz),order='F')
    Uz=np.reshape(uz,(nx,ny,nz),order='F')

    if flag_plot:
        ### Plot

        fig,ax=plt.subplots(1,3,figsize=[3*3,3])

        plt.subplots_adjust(wspace=0.5)
        names=['Ux','Uy','Uz']
        for k_ax,D in enumerate([Ux,Uy,Uz]):
            im=ax[k_ax].pcolormesh(X[:,:,0],Y[:,:,0],D[:,:,0],cmap=plt.colormaps['jet'])
            ax[k_ax].set_aspect('equal','box')
            cax=ggmt.get_cax(ax[k_ax])
            plt.colorbar(im,cax=cax)
            ax[k_ax].set_title(names[k_ax])


    return (X,Y,Z,Ux,Uy,Uz)

def compute_strain_2d(X_slice,Y_slice,Ux_slice,Uy_slice):

    STRA=np.zeros(list(X_slice.shape)+[2,2])
    dx=X_slice[0,1]-X_slice[0,0]
    dy=Y_slice[1,0]-Y_slice[0,0]


    for i in range(1,STRA.shape[0]-1):
        for j in range(1,STRA.shape[1]-1):
            duxdx=(Ux_slice[i,j+1]-Ux_slice[i,j-1])/(2*dx)
            duxdy=(Ux_slice[i+1,j]-Ux_slice[i-1,j])/(2*dy)

            duydx=(Uy_slice[i,j+1]-Uy_slice[i,j-1])/(2*dx)
            duydy=(Uy_slice[i+1,j]-Uy_slice[i-1,j])/(2*dy)

            ### Get strain tensor

            e11=duxdx
            e12=0.5*(duxdy+duydx)
            e21=e12
            e22=duydy

            STRA[i,j]=[[e11,e12],[e21,e22]]

    return STRA

def compute_stress_2d(X_slice,Y_slice,Ux_slice,Uy_slice,poisson,mu_el):

    STRE=np.zeros(list(X_slice.shape)+[2,2])
    dx=X_slice[0,1]-X_slice[0,0]
    dy=Y_slice[1,0]-Y_slice[0,0]


    for i in range(1,STRE.shape[0]-1):
        for j in range(1,STRE.shape[1]-1):
            duxdx=(Ux_slice[i,j+1]-Ux_slice[i,j-1])/(2*dx)
            duxdy=(Ux_slice[i+1,j]-Ux_slice[i-1,j])/(2*dy)

            duydx=(Uy_slice[i,j+1]-Uy_slice[i,j-1])/(2*dx)
            duydy=(Uy_slice[i+1,j]-Uy_slice[i-1,j])/(2*dy)

            ### Get strain tensor

            e11=duxdx
            e12=0.5*(duxdy+duydx)
            e21=e12
            e22=duydy

            stress=adutil.strain2stress(np.array([[e11,e12],[e21,e22]]),poisson,mu_el)

            STRE[i,j]=stress

    return STRE

def compute_sigma1_2d(X_slice,Y_slice,Ux_slice,Uy_slice,poisson,mu_el):
    """
    ### Compressive vector
    # The first eigenvalue is the biggest,
    # but if positive it means that it is extension, compression axis is the one with the
    # smallest eigenvalue
    """

    SIGMA1=np.zeros(list(X_slice.shape)+[2])
    dx=X_slice[0,1]-X_slice[0,0]
    dy=Y_slice[1,0]-Y_slice[0,0]


    for i in range(1,SIGMA1.shape[0]-1):
        for j in range(1,SIGMA1.shape[1]-1):
            duxdx=(Ux_slice[i,j+1]-Ux_slice[i,j-1])/(2*dx)
            duxdy=(Ux_slice[i+1,j]-Ux_slice[i-1,j])/(2*dy)

            duydx=(Uy_slice[i,j+1]-Uy_slice[i,j-1])/(2*dx)
            duydy=(Uy_slice[i+1,j]-Uy_slice[i-1,j])/(2*dy)

            ### Get strain tensor

            e11=duxdx
            e12=0.5*(duxdy+duydx)
            e21=e12
            e22=duydy

            try:
                stress=adutil.strain2stress(np.array([[e11,e12],[e21,e22]]),poisson,mu_el)
                eigvals,eigvecs=adutil.eig_sort(stress)
                coeff_diff=np.abs(eigvals[1]-eigvals[0])
                vector_diff=coeff_diff*eigvecs[:,1]
            except:
                vector_diff=np.array([0,0])

            SIGMA1[i,j]=vector_diff

    return SIGMA1

def compute_epsilon1_2d(X_slice,Y_slice,Ux_slice,Uy_slice):
    """
    ### Compressive vector
    # The first eigenvalue is the biggest,
    # but if positive it means that it is extension, compression axis is the one with the
    # smallest eigenvalue
    """

    EPSILON1=np.zeros(list(X_slice.shape)+[2])
    dx=X_slice[0,1]-X_slice[0,0]
    dy=Y_slice[1,0]-Y_slice[0,0]


    for i in range(1,EPSILON1.shape[0]-1):
        for j in range(1,EPSILON1.shape[1]-1):
            duxdx=(Ux_slice[i,j+1]-Ux_slice[i,j-1])/(2*dx)
            duxdy=(Ux_slice[i+1,j]-Ux_slice[i-1,j])/(2*dy)

            duydx=(Uy_slice[i,j+1]-Uy_slice[i,j-1])/(2*dx)
            duydy=(Uy_slice[i+1,j]-Uy_slice[i-1,j])/(2*dy)

            ### Get strain tensor

            e11=duxdx
            e12=0.5*(duxdy+duydx)
            e21=e12
            e22=duydy

            try:
                eigvals,eigvecs=adutil.eig_sort(np.array([[e11,e12],[e21,e22]]))
                coeff_diff=np.abs(eigvals[0]-eigvals[1])
                vector_diff=coeff_diff*eigvecs[:,0]
            except:
                vector_diff=np.array([0,0])
            EPSILON1[i,j]=vector_diff

    return EPSILON1


def get_strain_2d(x_sta,y_sta,X_slice,Y_slice,Ux_slice,Uy_slice):

    ### Find nearest index

    DIST=(X_slice-x_sta)**2+(Y_slice-y_sta)**2
    i_sta,j_sta=np.unravel_index(DIST.argmin(), DIST.shape)

    ### Compute derivative

    dx=X_slice[i_sta,j_sta+1]-X_slice[i_sta,j_sta]
    dy=Y_slice[i_sta+1,j_sta]-Y_slice[i_sta,j_sta]

    duxdx=(Ux_slice[i_sta,j_sta+1]-Ux_slice[i_sta,j_sta-1])/(2*dx)
    duxdy=(Ux_slice[i_sta+1,j_sta]-Ux_slice[i_sta-1,j_sta])/(2*dy)

    duydx=(Uy_slice[i_sta,j_sta+1]-Uy_slice[i_sta,j_sta-1])/(2*dx)
    duydy=(Uy_slice[i_sta+1,j_sta]-Uy_slice[i_sta-1,j_sta])/(2*dy)

    ### Get strain tensor

    e11=duxdx
    e12=0.5*(duxdy+duydx)
    e21=e12
    e22=duydy

    strain=np.array([[e11,e12],[e21,e22]])

    return strain

def strain2stress(strain,poisson,mu_el):

    lambda_el= (2*poisson*mu_el)/(1-2*poisson)

    stress=lambda_el*strain.trace()*np.eye(2)+2*mu_el*strain

    return stress

def eig_sort(matrix):

    eig_vals,eig_vecs=np.linalg.eig(matrix)
    ind_descend=np.argsort(-eig_vals)
    eig_vals=eig_vals[ind_descend]
    eig_vecs=eig_vecs[:,ind_descend]

    return (eig_vals,eig_vecs)

def plot_vector(vector,sign=1,ax=None,flag_normalize=False,flag_axis=True,
                facecolor='k',width=2,rmax=None,mode='trigo',flag_mirror=True):
    """
    Function made to plot a vector on a polar plot

    Input
    -----
        vector: np.array: 2 elements array specifying x and y
        sign: +/- 1: specify direction of arrows
        ax:
        flag_normalize: normalize the vector
        show_axis: bool: True to show frame
        faceolor
        width
        rmax: float: limit of the polar plot
        mode:str: trigo or azimuth
        mirror: plot mirrot vector
    """

    ###############
    ### Process ###

    ### Norm

    norm=np.sqrt(np.sum(vector**2))
    if flag_normalize:
        vector=vector/norm
        norm=1

    ### Convert to polar coordinates

    (r, phi)=gproj.cart2pol(vector[0],vector[1],x0=0,y0=0)

    if mode=='azimuth':
        phi=gproj.trigo2az(phi,unit='radian')

    ############
    ### Plot ###

    if ax is None:
        fig,ax=plt.subplots(subplot_kw=dict(projection='polar'))

    if not flag_axis:
        ax.axis('off')

    if mode=='azimuth':
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    if rmax is None:
        rmax=norm

    ax.set_rlim([0,rmax])
    r_range=ax.get_rmax()-ax.get_rmin()
    ax.set_rorigin(-0.05*r_range) # Shift origin

    xy_start=(phi, 0) ## Arrow root
    xy_end=(phi, r) ## Arrow head

    if sign==-1: # Revert direction
        xy_start=(phi, r)
        xy_end=(phi, 0)

    headwidth=width*4
    headlength=width*4

    ax.annotate("", xy=xy_end, xytext=xy_start,arrowprops=dict(headwidth=headwidth,
                                                               headlength=headlength,width=width,
                    edgecolor = 'none', facecolor = facecolor, lw = 0.2))

    if flag_mirror:
        ax.annotate("", xy=(xy_end[0]+np.pi,xy_end[1]), xytext=(xy_start[0]+np.pi,xy_start[1]),
                    arrowprops=dict(headwidth=headwidth,headlength=headlength,width=width,
                        edgecolor = 'none', facecolor = facecolor, lw = 0.2))



def reduce(data,times=1):

    k_time=0
    while k_time<times:
        k_time+=1
        data=np.delete(data, np.s_[::2], 1)
        data=np.delete(data, np.s_[::2], 0)

    return data


def get_fault_coord(xa,ya,l,alpha,dip,dz,flag_plot=False):


    #l=2
    #alpha=45
    #dip=15
    #dz=2

    alpha*=np.pi/180
    dip*=np.pi/180

    xb=xa+l*np.cos(alpha)
    yb=ya+l*np.sin(alpha)
    xc=xb+dz/np.tan(dip)*np.sin(alpha)
    yc=yb-dz/np.tan(dip)*np.cos(alpha)

    if flag_plot:
        fig,ax=plt.subplots()
        ax.plot([xa,xb,xc],[ya,yb,yc],'--k')
        ax.text(xa,ya,'A')
        ax.text(xc,yc,'C')
        ax.set_aspect('equal')

    return (xc,yc)


def plot_sigma1_stations(X_slice,Y_slice,SIGMA1,period='syn',
                         lag_mode='sample',sampling_rate=200,lag_lim=None,
                         dic_sta_obs=None,lag_is_time=False):

    def change_angle(angle,mid):
        """
        Function made to transform the angle so that it takes
        into account the fact that there is a 180° periodicity
        """

        angle=angle%180

        if angle>mid+90:
            angle-=180
        elif angle<mid-90:
            angle+=180

        return angle

    def linear(x, a, b):
        """
        Function used for fitting
        """
        x=np.asarray(x)
        return a*x+b


    ### Parameters

    if lag_lim is not None:
        ax_norm_xlim=lag_lim
    ax_norm_ylim=[0,1]
    ax_fast_xlim=[0,180]
    ax_fast_ylim=[0-20,180+20]
    ini_lon=-130.1
    ini_lat=45.9

    ### Get observations

    if dic_sta_obs is None:
        dic_sta_obs=adutil.get_station_obs(period=period)

    station_dic=ggmt.read_stationfile()

    ### Extract sigma1 vector at each station

    dic_sta_cal={}
    for station in station_dic.keys():
        lon,lat=station_dic[station]['lon'],station_dic[station]['lat']
        x_sta,y_sta=gproj.ll2xy(lon,lat,ini_lon,ini_lat)
        DIST=(X_slice-x_sta)**2+(Y_slice-y_sta)**2
        i_sta,j_sta=np.unravel_index(DIST.argmin(), DIST.shape)
        vector=SIGMA1[i_sta,j_sta]
        (rho, phi_cal)=gproj.cart2pol(vector[0], vector[1])
        phi_cal*=180/np.pi
        if phi_cal>=90:
            phi_cal-=180
        elif phi_cal<=-90:
            phi_cal+=180
        az_cal=gproj.trigo2az(phi_cal)
        norm=np.sqrt(np.sum(vector**2))
        dic_sta_cal[station]={}
        dic_sta_cal[station]['norm']=norm
        dic_sta_cal[station]['fast']=az_cal

    ### Plot

    fig,ax=plt.subplots(1,2,figsize=[6.38,3.2])
    plt.subplots_adjust(wspace=0.3)


    [ax_norm,ax_fast]=ax
    fast_bins=np.linspace(0,180,5)
    ax_fast.plot(fast_bins,fast_bins,'--r')
    ax_fast.plot(fast_bins,fast_bins-180,'--r')
    ax_fast.plot(fast_bins,fast_bins+180,'--r')

    lag_obss=[]
    fast_obss=[]
    fast_cals=[]
    norm_cals=[]
    lag_range=ax_norm_xlim[1]-ax_norm_xlim[0]
    fast_range=ax_fast_xlim[1]-ax_fast_xlim[0]

    for station in dic_sta_cal.keys():
        if station not in dic_sta_obs.keys():
            continue

        norm_cal=dic_sta_cal[station]['norm']
        fast_cal=dic_sta_cal[station]['fast']
        lag_obs=dic_sta_obs[station]['lag']
        if lag_obs is not None:
            if lag_is_time:
                # lag_obs is already in seconds (e.g. our splitting pipeline's dt) -
                # no sampling_rate conversion needed, just switch units if requested.
                if lag_mode=='ms':
                    lag_obs=lag_obs*1000
            else:
                if lag_mode=='ms':
                    lag_obs=lag_obs/sampling_rate*1000
                elif lag_mode=='s':
                    lag_obs=lag_obs/sampling_rate

        fast_obs=dic_sta_obs[station]['fast']

        try:
            ax_norm.plot(lag_obs,norm_cal,'ok')
            ax_norm.text(lag_obs+0.05*lag_range,norm_cal,station,ha='left',va='center',fontsize=8)

            norm_cals.append(norm_cal)
            lag_obss.append(lag_obs)
        except:
            pass
        try:
            ax_fast.plot(fast_obs,fast_cal,'ok')
            fast_cals.append(fast_cal)
            fast_obss.append(fast_obs)
            ax_fast.text(fast_obs+0.05*fast_range,fast_cal,station,ha='left',va='center',fontsize=8)
        except:
            pass

    ## Linear fit

    norm_cals=np.array(norm_cals)
    lag_obss=np.array(lag_obss)
    fast_obss=np.array(fast_obss)
    fast_cals=np.array(fast_cals)
    popt, pcov = curve_fit(linear, lag_obss, norm_cals)
    norm_model=linear(lag_obss,*popt)
    rms=np.sqrt(np.mean((norm_model - norm_cals)**2))

    fast_cals_rms=np.array([change_angle(y,x) for x,y in zip(fast_obss,fast_cals)])
    rms_fast=np.sqrt(np.mean((fast_obss - fast_cals_rms)**2))


    ax_norm.plot(ax_norm_xlim,linear(ax_norm_xlim,*popt),'--r',zorder=-2)


    ### Cosmetic

    ax_norm.text(0.1,0.9,'RMS : %.3f'%rms,transform=ax_norm.transAxes)
    ax_norm.set_ylim(ax_norm_ylim)
    ax_norm.set_xlim(ax_norm_xlim)
    ax_norm.set_xlabel('Lag [samples]')
    ax_norm.set_ylabel(r'$\sigma_{1}-\sigma_{2}$')

    ax_fast.text(0.1,0.9,'RMS : %.1f'%rms_fast,transform=ax_fast.transAxes)
    ax_fast.set_aspect('equal','box')
    ax_fast.set_xlim(ax_fast_xlim)
    ax_fast.set_ylim(ax_fast_ylim)
    ax_fast.set_xlabel(r'$\Phi_{obs}$ [°]')
    ax_fast.set_ylabel(r'$\Phi_{cal}$ [°]')

    return (ax_norm,ax_fast)


def plot_elev_stations(X_slice,Y_slice,Uz_slice,period='syn',dic_sta_obs=None):


    ### Parameters

    ini_lon=-130.1
    ini_lat=45.9

    ### Get observations

    if dic_sta_obs is None:
        dic_sta_obs=adutil.get_station_obs(period=period)

    station_dic=ggmt.read_stationfile()

    ### Extract dz vector at each station

    dic_sta_cal={}
    for station in station_dic.keys():
        lon,lat=station_dic[station]['lon'],station_dic[station]['lat']
        x_sta,y_sta=gproj.ll2xy(lon,lat,ini_lon,ini_lat)
        DIST=(X_slice-x_sta)**2+(Y_slice-y_sta)**2
        i_sta,j_sta=np.unravel_index(DIST.argmin(), DIST.shape)
        dz=Uz_slice[i_sta,j_sta]
        dic_sta_cal[station]={}
        dic_sta_cal[station]['dz']=dz


    ### Plot

    fig,ax=plt.subplots(figsize=[6,3])
    obss=[]
    cals=[]
    xticklabels=['']
    k_sta=0
    for station in dic_sta_obs.keys():
        if (station not in dic_sta_cal.keys()) or (dic_sta_obs[station]['dz'] is None) :
            continue
        k_sta+=1
        obs=dic_sta_obs[station]['dz']
        cal=dic_sta_cal[station]['dz']
        obss.append(obs)
        cals.append(cal)
        xticklabels.append(station)
        ax.plot(k_sta,obs,marker='o',mfc='red',ms=7,mec='k')
        ax.plot(k_sta,cal,marker='o',mfc='blue',mec='k',ms=5)

    ### Legend

    custom_lines = [mpl.lines.Line2D([0], [0], color='w',marker='o',mfc='red',ms=7,mec='k'),
                mpl.lines.Line2D([0], [0], color='w',marker='o',mfc='blue',ms=5,mec='k'),]
    ax.legend(custom_lines,['Observed','Modelled'],loc='lower right')

    ax.xaxis.grid(True,ls=':',color='0.5')

    ### Compute chi square

    obss=np.asarray(obss)
    cals=np.asarray(cals)
    rms=np.mean( (obss-cals)**2 )
    ### Cosmetic

    xticklabels.append('')
    # Ticks must match len(xticklabels) exactly (leading + trailing blank + one per station) -
    # older matplotlib silently tolerated the off-by-one here, current matplotlib doesn't.
    ax.set_xticks(np.arange(0,k_sta+2))
    ax.set_xlim([0,k_sta+1])
    ax.set_xticklabels(xticklabels)

    ax.text(0.1,0.9,'RMS : %.3f'%rms,transform=ax.transAxes)
    ax.set_ylabel('Uz [m]')

    return ax


def get_station_obs(period='syn'):
    """
    UsedIn
    -------
    plot_sigma1_stations

    """

    if period not in ['syn','pre','post']:
        raise ValueError('period must be in [syn,pre,post]')


    dic_sta_obs_syn={
            'AXCC1':{'lag':None,'fast':55,'dz':-2.45},
            'AXEC1':{'lag':4,'fast':50,'dz':None},
            'AXEC2':{'lag':9,'fast':135,'dz':-1.06},
            'AXEC3':{'lag':13,'fast':165,'dz':None},
            'AXAS1':{'lag':11,'fast':45,'dz':-2.00},
            'AXAS2':{'lag':4,'fast':160,'dz':None},
            'AXID1':{'lag':7,'fast':15,'dz':-1.39},
                }


    dic_sta_obs_pre={
            'AXCC1':{'lag':9,'fast':80,'dz':+2.45},
            'AXEC1':{'lag':5,'fast':140,'dz':None},
            'AXEC2':{'lag':6,'fast':105,'dz':+1.06},
            'AXEC3':{'lag':4,'fast':70,'dz':None},
            'AXAS1':{'lag':13,'fast':140,'dz':+2.00},
            'AXAS2':{'lag':4,'fast':165,'dz':None},
            'AXID1':{'lag':16,'fast':120,'dz':+1.39},
                }

    dic_sta_obs_post={}

    if period=='pre':
        dic_sta_obs=dic_sta_obs_pre
    elif period=='syn':
        dic_sta_obs=dic_sta_obs_syn
    elif period=='post':
        dic_sta_obs=dic_sta_obs_post

    return dic_sta_obs


def get_station_obs_hemmett(period='syn'):
    """
    Observed fast direction / delay time for the 6 production splitting stations
    (AXID1 excluded - not part of the production catalog), computed from this
    repo's own splitting_functions.py/swspy pipeline results rather than Baillard's
    2019 hardcoded literals. 'dz' (vertical deformation) is not a splitting-pipeline
    output, so it is carried over unchanged from get_station_obs() for the stations
    that have it.

    Data sources: AXAS2 uses the old (fixed 5-40 Hz) full 2015-2021 pipeline
    results, unchanged (no new-data rerun exists for it yet). AXCC1/AXEC1/AXEC2/
    AXEC3/AXAS1 use the NEW mfast max_dt=0.2s data (per-event adaptive bandpass,
    max_t_shift_s=0.20s -- the physically-motivated S-wave-travel-time max delay,
    replacing the earlier max_t_shift_s=0.30s mfast data this function used
    previously):
      AXCC1/AXEC1/AXEC3/AXAS1: mfast_maxdt_pipeline_transfer/splitting_results_
        {AXCC1,AXEC1,AXEC3,AXAS1}_{2015_2021,2022_2026}_all_batches.csv (all
        complete -- AXEC3/AXAS1 reruns finished 2026-07-23, moved here from the
        old-data group this function previously used for them).
      AXEC2: mfast_maxdt_pipeline_transfer/splitting_results_AXEC2_2022_2026_
        all_batches.csv (complete) + our OWN maxdt02 re-run's 2015-2021 batches
        (production_axec2_mfast_filters_maxdt02_lqt_pykonal_results/, now
        COMPLETE - all 497 batches, unlike the earlier still-running state this
        function's docstring previously described), joined against
        raw_axec2_all_batches_mfast_filters_data/raw_axec2_all_batches_mfast_
        filters_metadata.csv for event location where needed (not used here,
        since this function only needs phi/dt/quality/errors/datetime).

    QC: quality>QUALITY_MIN, phi_error<PHI_ERR_MAX, dt_error<DT_ERR_MAX, dt>0 for
    all 6 stations. The 5 new-data stations (AXCC1/AXEC1/AXEC2/AXEC3/AXAS1)
    additionally get the dt<=T_dom/2 cycle-skip-risk cut (per-event dominant
    period -- carried directly as `dominant_period` in the mfast_maxdt_pipeline_
    transfer files, or computed as chosen_filter_dom_period_samples/200.0 for
    the AXEC2 2015-2021 batches, same convention as rose_7period_5stations_
    newdata.py) - AXAS2's old pipeline doesn't track a per-event dominant period
    so this cut can't be applied to it (same caveat as
    figure_all_stations_temporal_fractional_dt.py).

    UsedIn
    -------
    plot_sigma1_stations (fast, lag - lag is in SECONDS, not samples: pass
        lag_is_time=True), plot_elev_stations (dz)
    """

    if period not in ['pre','syn']:
        raise ValueError('period must be in [pre,syn] for get_station_obs_hemmett')

    HERE = os.path.dirname(os.path.abspath(__file__))
    OLD_RESULTS_DIR = '/Users/mhemmett/Seismology/axial-sws/lqt_pykonal_combined_results/'
    NEWDATA_DIR = os.path.join(HERE, '..', 'mfast_maxdt_pipeline_transfer')
    AXEC2_2015_2021_DIR = os.path.join(HERE, 'production_axec2_mfast_filters_maxdt02_lqt_pykonal_results')

    OLD_STATION_FILES = {
        'AXAS2': ['splitting_results_AXAS2_2015_2021_all_batches.csv'],
    }
    NEWDATA_STATION_FILES = {
        'AXCC1': ['splitting_results_AXCC1_2015_2021_all_batches.csv',
                  'splitting_results_AXCC1_2022_2026_all_batches.csv'],
        'AXEC1': ['splitting_results_AXEC1_2015_2021_all_batches.csv',
                  'splitting_results_AXEC1_2022_2026_all_batches.csv'],
        'AXEC2': ['splitting_results_AXEC2_2022_2026_all_batches.csv'],
        'AXEC3': ['splitting_results_AXEC3_2015_2021_all_batches.csv',
                  'splitting_results_AXEC3_2022_2026_all_batches.csv'],
        'AXAS1': ['splitting_results_AXAS1_2015_2021_all_batches.csv',
                  'splitting_results_AXAS1_2022_2026_all_batches.csv'],
    }
    NEWDATA_STATIONS = set(NEWDATA_STATION_FILES)

    ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
    ERUPTION_END = pd.Timestamp('2015-05-19 00:00', tz='UTC')

    PHI_ERR_MAX = 20.0
    DT_ERR_MAX = 0.05   # matches the filter convention established for the new data
                        # elsewhere this session (rose_7period_axcc1_axec1_axec2_newdata.py,
                        # histograms_all_columns_newdata_combined.py, traveltime_anisotropy_
                        # 7period_axcc1_axec1_axec2_newdata.py) -- was 0.04 previously.
    QUALITY_MIN = 0.5

    dz_by_station = get_station_obs(period=period)

    def _load(station):
        if station in NEWDATA_STATIONS:
            paths = [os.path.join(NEWDATA_DIR, f) for f in NEWDATA_STATION_FILES[station]]
            dfs = [pd.read_csv(p) for p in paths]
            if station == 'AXEC2':
                batch_files = sorted(glob.glob(os.path.join(
                    AXEC2_2015_2021_DIR,
                    'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_batch_*.csv')))
                batch_dfs = [pd.read_csv(f) for f in batch_files]
                for d in batch_dfs:
                    if len(d) > 0:
                        d['dominant_period'] = d['chosen_filter_dom_period_samples'] / 200.0
                dfs.extend(d for d in batch_dfs if len(d) > 0)
            df = pd.concat(dfs, ignore_index=True)
        else:
            paths = [os.path.join(OLD_RESULTS_DIR, f) for f in OLD_STATION_FILES[station]]
            dfs = [pd.read_csv(p) for p in paths]
            df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        return df

    dic_sta_obs = {}
    for station in ['AXCC1', 'AXEC1', 'AXEC2', 'AXEC3', 'AXAS1', 'AXAS2']:
        df = _load(station)

        df = df.dropna(subset=['phi', 'dt', 'phi_error', 'dt_error', 'quality'])
        df = df[(df['dt'] > 0) & (df['phi_error'] < PHI_ERR_MAX) &
                (df['dt_error'] < DT_ERR_MAX) & (df['quality'] > QUALITY_MIN)]

        if station in NEWDATA_STATIONS:
            df = df.dropna(subset=['dominant_period'])
            df = df[df['dt'] <= df['dominant_period'] / 2.0]

        t = pd.to_datetime(df['datetime'], utc=True, format='ISO8601')
        if period == 'pre':
            mask = t < ERUPTION_START
        else:
            mask = (t >= ERUPTION_START) & (t < ERUPTION_END)
        sub = df[mask]

        if len(sub) == 0:
            fast, lag = None, None
        else:
            fast = float(np.median(sub['phi'] % 180.0))
            lag = float(np.median(sub['dt']))  # seconds

        dic_sta_obs[station] = {
            'fast': fast,
            'lag': lag,
            'dz': dz_by_station.get(station, {}).get('dz'),
            'n': len(sub),
        }

    return dic_sta_obs
