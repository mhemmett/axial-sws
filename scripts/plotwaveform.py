#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:45:37 2017

@author: baillard

Module made to plot AND process waveforms, that includes polarizaiton analyses, filtering, reading of obsy
streams, particule motions...

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import copy

from obspy.core.event import Event
from obspy.clients.filesystem.sds import Client as sds_client
from obspy.core.utcdatetime import UTCDateTime
import itertools

from GMT import get_cax
from util import smooth_curve
from projection import ll2xy
#from geometry import get_angles

def get_angles(x_sta, y_sta, x_hyp, y_hyp):
    ### Get azimuth from hyp to sta (this is not the back azimuth, just regular azimuth)
    delta_x = x_sta - x_hyp
    delta_y = y_sta - y_hyp
    azimuth = np.arctan2(delta_x, delta_y)
    return azimuth


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

def SDS2streams(sds_root,start_time,time_delay,
                network_code='*',station_code='*',location_code='*',channel_code='*Z'):
    """
    Function made to output streams from SDS database
    sds_client get_waveforms in obspy deos not support commas in codes, so we had to modified it, now it works
    
    Inputs:
        sds_root: str, path to the SDS structure (not including year,network...)
        start_time: obspy.core.utcdatetime.UTCDateTime, beginning of the signal
        time_delay: float, time delay in seconds
        *_code: strings, comma separated if several codes to be given
        
    Outputs:
        st: obspy stream object
        
    Todo:
        - Implemenent comma seprated for networks and locations ids
    """
    
    from obspy.core.stream import Stream
    
    ### Check
    
    if not isinstance(start_time,UTCDateTime):
        raise ValueError('start_time must be obspy UTCDateTime object')
        
    ### Split stations if necessary

    station_codes=station_code.split(',')    
    channel_codes=channel_code.split(',')
    
    #### List of combinations
    
    list_elem=[]
    list_elem.append(station_codes)
    list_elem.append(channel_codes)
    list_comb= list(itertools.product(*list_elem))
    
    ### Read SDS client
    
    client=sds_client(sds_root,sds_type='D',format='MSEED')
    
    ### Get start time
    
    end_time=start_time+time_delay
    
    ### Get waveforms
    
    st=Stream()
    for station_code,channel_code in list_comb: 
        st_tmp= client.get_waveforms(network_code,station_code,location_code,channel_code,start_time,end_time)
        st+=st_tmp
    
    ### Sort
    
    st.sort(keys=['station','channel'])
        
    ### Return
    
    return st
    
def plot_event(single_event,sds_root,time_delay,time_before,_network_code='*',
               _station_code='*',
               _location_code='*',
               _channel_code='*Z'):
    
    """
    multiple codes are comma separated: for example 'BHZ,BHE' , wildcards are allowed
    """
    
    ### Define pick colors
    
    pick_colormap={'P':'k','S':'r'}
    
    if type(single_event) is not type(Event()):
        raise Exception('Event given is not and obspy Event')

    ### Read SDS client
    
    client=sds_client(sds_root,sds_type='D',format='MSEED')
    
    ### Read starttime from preferred origin
    
    origin_pref=[x for x in single_event.origins if x.resource_id==single_event.preferred_origin_id][0]
    _starttime=origin_pref.time-time_before
    _endtime=_starttime+time_delay
    
    st= client.get_waveforms(_network_code,_station_code,_location_code,_channel_code,_starttime,_endtime)
    
    ### Pre-process streams
    
    st.sort(keys=['network','station','channel'])
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="hann")
    st.filter("bandpass",freqmin=5,freqmax=40)
    #st.taper(type="cosine",max_percentage=0.1)
    
    #st.plot(starttime=origin_pref.time, endtime=origin_pref.time+3, fig=fig2)
    
    ### Start plotting waveforms 
    
    
    fig=plt.figure(figsize=(20, 12))
    
    st_num=len(st)
    axs=[]
    
    for i,tr in enumerate(st):
        net, sta, loc, cha = tr.id.split(".")
        if loc=='':
                loc=None
    
        starttime_relative=tr.stats.starttime-_starttime
        sampletimes = np.arange(starttime_relative,
                starttime_relative + (tr.stats.delta * tr.stats.npts),
                tr.stats.delta)
#    
        if len(sampletimes) == tr.stats.npts + 1:
            sampletimes = sampletimes[:-1]
    
        if i == 0:
            ax = fig.add_subplot(st_num, 1, i+1)
        else:
            ax = fig.add_subplot(st_num, 1, i+1, sharex=axs[0])
    
        axs.append(ax)
        ax.plot(sampletimes,tr.data)
        ### Add label
        ax.text(0.1, 0.90, tr.id, transform=ax.transAxes)
    
    fig.subplots_adjust(hspace=0)
    axs[0].set_xlim(0,sampletimes[-1])
    
    ### Plot Picks on top of the waveforms
    
    picks=single_event.picks
    arrivals=single_event.origins[0].arrivals
    
    # Select picks that are in arrivals
    
    picks_sel=[getPickForArrival(picks, arrival) for arrival in arrivals]
    print(picks_sel)
        
    for i,tr in enumerate(st):
        ax=axs[i]
        net,sta,loc,chan=tr.id.split('.')
        if net=='':
            net=None
        if sta=='':
            continue
        if loc=='':
            loc=None
        print(sta,'sf')
        picks_tr=getPicks(picks_sel,net,sta,loc)
        print(picks_tr)
        for pick_tr in picks_tr:
            phase_pick=pick_tr.phase_hint
            rel_time_pick=pick_tr.time-_starttime
           
            ax.axvline(x=rel_time_pick, ymin=0, ymax=1,color=pick_colormap[phase_pick],lw=2)
            
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

def getPicks(picks,network_code=None, station_code=None,phase_hint=None):
    """
    returns all matching picks as list.depending on criterion
    
    Inputs:
        picks: list of events.picks objects
        network_code: String (can only be one network code)
        station_code: String (can only be one station)
        phase_hint: 'P' or 'S' (can only be one at a time)
    
    Outputs:
        sel_pha: List of picks that fulfill the network, station and phase codes
        empty list if criterion to correct
    """
    
    #### Select picks with proper network
    sel_net = []
    for p in picks:
        if network_code is not None:
            if network_code!=p.waveform_id.network_code:
                continue
        sel_net.append(p)
    
    #### Select picks with proper station
    sel_sta =[]
    for p in sel_net:
        if station_code is not None:
            if station_code!=p.waveform_id.station_code:
                continue
        sel_sta.append(p)
    
    #### Select picks with proper phase
    sel_pha =[]
    for p in sel_sta:
        if phase_hint is not None:
            if phase_hint!=p.phase_hint:
                continue
        sel_pha.append(p)
        
    return sel_pha

def getPicksAndArrivals(single_event,network_code=None,station_code=None,phase_hint=None):
    
    """
    Retrieve Pick and Arrival object that fulfill the criterion, Arrivals are
    taken from the preferred origin, empty list if nothing
    """
    
    ### Set
    
    all_picks=single_event.picks
    origin_id=single_event.preferred_origin_id
    
    for origin in single_event.origins:
        if origin.resource_id==origin_id:
            break
    
    ### Select proper picks
    
    sel_picks=getPicks(all_picks,
                       network_code=network_code, station_code=station_code,phase_hint=phase_hint)
    
    #### Get PicksArrivals Pair
    
    pick_arrival_pairs=[]

    for arrival in origin.arrivals:
        pick=getPickForArrival(sel_picks, arrival)
        if pick!=None:
            pair=[pick,arrival]
            pick_arrival_pairs.append(pair)
        
        
    ### Return
    
    return pick_arrival_pairs
        
#### Plotting utils
    
def plot_picks(stream_fig,stream_obj,single_event,phases=['P','S'],dict_colors=None,dict_linestyles=None):

    ### Get axes
    
    ax_list=stream_fig.get_axes()
    
    ### Check
    
    if len(ax_list)!=len(stream_obj):
        raise ValueError('Number of streams in object different from numbers of subplots')
        
    if dict_colors is None:
        dict_colors={'P':'b',
                     'S':'r'
                     }
    if dict_linestyles is None:
        dict_linestyles={'P':'-',
                         'S':'-'
                         }
    
    #### Retrieve pick and plot
    
    kk=-1
    for trace in stream_obj:
        kk+=1
        station_code=stream_obj[kk].stats.station

        ax=ax_list[kk]
        for sel_phase in phases:
            Pick,Arrival=getPicksAndArrivals(single_event,network_code=None,station_code=station_code,phase_hint=sel_phase)[0]
            time_abs_phase=Pick.time.matplotlib_date
            ax.axvline(x=time_abs_phase, ymin=0, ymax=1,color=dict_colors[sel_phase],linestyle=dict_linestyles[sel_phase],lw=1)

    ### Return
    
    return stream_fig


def stream2data(st):
    """
    Convert stream into MxN numpy array
    N= stream number
    M= sample number
    
    Input:
        st: obspy stream object: must have same number of samples to be stacked into a matrix
        
    Output:
        big_data: MxN np.array with data
    """
    big_data=[]
    for trace in st:
        data=trace.data
        big_data.append(data)
        
    big_data=np.column_stack(big_data)
    
    
    return big_data
    
def jurkevics(stream_zne):
    """
    Function made to get the azimuth direction and incidence angle of polarization
    Base on Jurkevics 1988
    
    Inputs:
        stream_zne: obspy stream object, Z must be first, N second, E last
        Z up
    
    Outputs:
        az: float, azimuth of polarization [0,360] (convention: degree positive clockwise from North)
            in case of P analysis, this will indicate the direction of the event
        inc: float, incidence angle of the polarization [0,90] (convention: deegree positive counterclockwise from vertical Down)
            in case of P analysis, this will indicate ray incidence angle. Ray coming horizontally has an inc of 90°
        rec: float, rectilinearity of the wave
    """
    
    #### Check
    
    check_stream3(stream_zne)
    
    ### Process
    
    data_zne=stream2data(stream_zne)
    
    (az,inc,rec,big_eigvec,lambda1)=jurkevics_data(data_zne)
    return (az,inc,rec,big_eigvec)

def jurkevics_sliding(stream_zne,win_width,win_overlap,flag_plot=False):
    """
    Same as function Jurkevics, but using a sliding window to compute the az,rec and incs
    parameters
    
    Input
    -----
        stream_zne: obspy stream object: 
        win_width: float: width in seconds of the sliding window
        win_overlap: float: [0-1] choose overlap of the sliding window, 1 means 100% overlap
        
    Output
    ------
        az_mean,az_std:float,float: azimuth value and error
        inc_mean,inc_std:float,float: inclination and error
        rec_mean,rec_std:float,float: rectilinearity value and error
        ....
        
    """
    
    #### Check
    
    check_stream3(stream_zne)
    
    if win_overlap>1:
        win_overlap=1
    elif win_overlap<0:
        win_overlap=0
    
    ### Initialize

    rsample=stream_zne[0].stats.sampling_rate
    
    data_zne=stream2data(stream_zne)
    
    
    azs = np.full(data_zne.shape[0],np.nan)
    recs = np.full(data_zne.shape[0],np.nan)
    incs = np.full(data_zne.shape[0],np.nan)
    lambda1s = np.full(data_zne.shape[0],np.nan)
    eigvec1s = np.full([3,data_zne.shape[0]],np.nan)

    
    N_width=int(win_width*rsample)
    N_end=N_width
    N_start=0
   
    N_shift=int(N_width*(1-win_overlap))+1    
    
    while N_end<data_zne.shape[0]:
        sub_data=data_zne[N_start:N_end,:]
        
        (az,inc,rec,eigvec1,lambda1)=jurkevics_data(sub_data)
        
        azs[N_end]=az
        recs[N_end]=rec
        incs[N_end]=inc
        lambda1s[N_end]=lambda1
        eigvec1s[:,N_end]=eigvec1
        
        N_start+=N_shift
        N_end+=N_shift
        
    ### Compute stats
    
    az_mean=np.nanmean(azs)
    az_std=np.nanstd(azs)
    inc_mean=np.nanmean(incs)
    inc_std=np.nanstd(incs)
    rec_mean=np.nanmean(recs)
    rec_std=np.nanstd(recs)
    lambda1_mean=np.nanmean(lambda1s)
    lambda1_std=np.nanstd(lambda1s)
    eigvec1_mean=np.nanmean(eigvec1s,axis=1)
    eigvec1_std=np.nanstd(eigvec1s,axis=1)
        
        
    
    if flag_plot:
    
        times=np.arange(data_zne.shape[0])/rsample
        fig,ax=plt.subplots(5,1,sharex=True)
        plt.subplots_adjust(hspace=0)
        ax[2].plot(times,recs,'ok',ms=1)
        ax[1].plot(times,incs,'ok',ms=1)
        ax[0].plot(times,azs,'ok',ms=1)
        ax[3].plot(times,lambda1s,'ok',ms=1)
        
        
        ax[4].plot(times,data_zne)
        
        ### Cosmetic
        
        k=-1
        for prefix in ['az','inc','rec','lambda1']:
            k+=1
            val_mean=eval(prefix+'_mean')
            val_std=eval(prefix+'_std')
            if prefix=='lambda1':
                 ax[k].text(0.05,0.95,'%e +- %e'%(val_mean,val_std),transform=ax[k].transAxes,va='top',ha='left',
          color='r')
            else:
                ax[k].text(0.05,0.95,'%f +- %f'%(val_mean,val_std),transform=ax[k].transAxes,va='top',ha='left',
          color='r')
 
        ax[2].set_ylabel('Rec')
        ax[0].set_ylabel('Az [°]')
        ax[1].set_ylabel('Inc [°]')
        ax[3].set_ylabel('Lambda1')
        ax[4].set_ylabel('Amplitude')
        ax[4].set_xlabel('Time [s]')
        
    
    return (az_mean,az_std,inc_mean,inc_std,rec_mean,rec_std,
            lambda1_mean,lambda1_std,eigvec1_mean,eigvec1_std,
            azs,incs,recs,lambda1s,eigvec1s)
 
def jurkevics_data(data_zne):
    """
    Function made to compute the az,rec and inc of a data array
    WARNING: Data should be Nx3 with columns Z, N and E, otherwise computation of azimuth will be false
    (Z upward)

    """
    
    ### Eigen
    
    eigvals,eigvecs=cov_eig(data_zne)
    eigvec1=eigvecs[:,0]
    lambda1,lambda2,lambda3=eigvals
    
    ### Make sure the eigenvector point towards the ground (-Z)
    
    if eigvec1[0]>=0:
        eigvec1=-eigvec1
    ### az,inc,rec

    az=np.arctan2(eigvec1[2], eigvec1[1])* 180/(np.pi) # Problem in jurkevics paper, this is the right definition
    # atan(Eeast/North)

#
#    if eigvec1[0]>=0:
#        az=az+180
        
    az=az%360 # To be sure angle is in 0-360
        
    inc=np.arccos(np.abs(eigvec1[0]))* 180/(np.pi) # from vertical 
    rec=1-(lambda2+lambda3)/(2*lambda1)

    return (az,inc,rec,eigvec1,lambda1)


def cov_eig(data_array):
    """
    Compute covariance matrix MxD: D being the number of components
    """
    cov_mat=np.cov(np.transpose(data_array))
    eig_vals,eig_vecs=np.linalg.eig(cov_mat)
    ind_descend=np.argsort(-eig_vals)
    eig_vals=eig_vals[ind_descend]
    eig_vecs=eig_vecs[:,ind_descend]
    
    return eig_vals,eig_vecs


def plot_window(ax,start_time,length_sec,facecolor='0.9',zorder=-1,label=''):
    """ 
    Function made to add window on top waveform plot
    
    Inputs:
        ax: matplotlib axes object
        start_time: obspy UTC time object
        length_sec: float: length of window in sec
        label: str
    """
    
    if isinstance(start_time,UTCDateTime):
        start_time=start_time.matplotlib_date
        width=length_sec/(60*60*24)
    else:
        width=length_sec


    plt.sca(ax)
    
    y_bottom,y_top=ax.get_ylim()
    height=y_top-y_bottom

    
    rec=mpatches.Rectangle((start_time,y_bottom),width=width,height=height,facecolor=facecolor,zorder=zorder)
    ax.add_patch(rec)
    
    plt.text(start_time+width*0.05, y_top-height*0.05,label,weight='bold',
             horizontalalignment='left',
             verticalalignment='top')
    
    return ax

def plot_pick(ax,time_pick,color='k',linestyle='-',lw=1,zorder=-1,label=''):
    
    if isinstance(time_pick,UTCDateTime):
        time_pick=time_pick.matplotlib_date
        
    ymin,ymax=ax.get_ylim()
    ax.axvline(x=time_pick, ymin=ymin, ymax=ymax,color=color,ls=linestyle,lw=lw)
    ax.text(time_pick, ymax*0.95,label,verticalalignment='top')
    return ax

def plot_patch(ax,start,width,facecolor='0.9',zorder=-1,label=''):
    """ 
    Function made to add window on top waveform plot (samples)
    
    Inputs:
        ax: matplotlib axes object
        start_time: obspy UTC time object
        length_sec: float: length of window in sec
        label: str
    """

    plt.sca(ax)
    
    y_bottom,y_top=ax.get_ylim()
    height=y_top-y_bottom
    
    rec=mpatches.Rectangle((start,y_bottom),width=width,height=height,facecolor=facecolor,zorder=zorder)
    ax.add_patch(rec)
    
    plt.text(start+width*0.05, y_top-height*0.05,label,weight='bold',
             horizontalalignment='left',
             verticalalignment='top')
    
    return ax

def check_stream3(stream_object):
    """
    Function made to check basically that stream is suitable for polar analysis
    (3 Traces, correct channel names(ZNE or LQT..))
    
    """
    
    ### Check
    
    if len(stream_object)!=3:
        raise ValueError('stream object must have only three components')
    
    channel_dirs=[]
    
    for trace in stream_object:
        channel_dirs.append(trace.stats.channel[-1])
    
    if channel_dirs[0] not in ['Z','1','L']:
        raise ValueError('1st component must be Z')
    if channel_dirs[1] not in ['N','2','Q']:
        raise ValueError('2nd component must be N')
    if channel_dirs[2] not in ['E','3','T']:
        raise ValueError('3rd component must be E')
        
def get_stream_code(stream_object):
    """
    Returns string code of type 'LQT' or 'ZNE' or 'ENZ' or 'EEZZN'...
    """
    channel_dirs=[]
    
    for trace in stream_object:
        channel_dirs.append(trace.stats.channel[-1])
    
    
    code=''.join(channel_dirs)
    
    return code
        
def plot_particle_motion_stream(stream_object,ref_time=None,ref_comp=None,vmin=None,vmax=None):
    """
    """

    
    #### Check
    check_stream3(stream_object)
    
    #### Define default ref_comp
    code=get_stream_code(stream_object)
    
    if code=='ZNE' and ref_comp is None:
        ref_comp='Z'
    
    if code=='LQT' and ref_comp is None:
        ref_comp='L'
    
    if ref_comp not in list(code):
        raise ValueError('ref component projection must be in stream code')

    if ref_time is None:
        ref_time=stream_object[0].stats.starttime
    
    time_shift=ref_time-stream_object[0].stats.starttime
    
    #### Retrieve data and time
    data_array=stream2data(stream_object)
    time_array=stream_object[0].times()
    time_array=time_array-time_shift
     
    #### Broadcast components
    z=data_array[:,0] # Z or L
    y=data_array[:,1] # N or Q (SV)
    x=data_array[:,2] # E or T (SH)
    
    
    if ref_comp in ['Z','L']:
        x_array=x
        y_array=y
        xlabel=code[2] # E or T
        ylabel=code[1] # N or Q
    elif ref_comp in ['N','Q']:
        x_array=x
        y_array=z
        xlabel=code[2] # E or T
        ylabel=code[0] # Z or L
    elif ref_comp in ['E','T']:
        x_array=y
        y_array=z
        xlabel=code[1] # N or Q
        ylabel=code[0] # Z or L
        
    array_2d=np.column_stack((x_array,y_array))
    
    ### plot
    plot_particle_motion(array_2d,time_array=time_array,vmin=vmin,vmax=vmax,xlabel=xlabel,ylabel=ylabel)


def plot_particle_motion(data_array,time_array=None,ax=None,vmin=None,vmax=None,xlabel='X',ylabel='Y',cbar=False):
    
        
    def comp_dist(x,y):
        
        ds=np.sqrt((np.diff(x))**2+(np.diff(y))**2)
        ds=np.append(0,ds)
        ds=np.cumsum(ds)
        
        return ds
    
    x_array=data_array[:,0]
    y_array=data_array[:,1]
    
    flag_time=True
    if time_array is None:
        flag_time=False
    #### Interp non-lin    
    x_graph,y_graph=smooth_curve(x_array,y_array,periodic=False,num_points=500,smoothness=0,k=3)
    ds_ini=comp_dist(x_array,y_array)
    
    ### Interp lin
    x_graph,y_graph=smooth_curve(x_graph,y_graph,periodic=False,num_points=500,smoothness=0,k=1)
  
    ### Interp color array
    
    if time_array is None:
        time_array=np.arange(0,len(x_array))
        
    ds_fin=comp_dist(x_graph,y_graph)
  
    time_out=np.interp(ds_fin,ds_ini,time_array)

    ind_clos=np.argmin(abs(time_out))
    
    ### Start plotting
    
    if ax is None:
        fig,ax=plt.subplots()

    ### Plot lines    
    ax.axvline(linestyle=':',color='k',lw=1,zorder=0)
    ax.axhline(linestyle=':',color='k',lw=1,zorder=0)
    #'RdYlBu'

    #### Define colorbar limits
    
    if vmin is None:
        vmin=np.min(time_out)
    if vmax is None:
        vmax=np.max(time_out)
    
    max_v_value=np.max(np.abs((vmin,vmax)))
    vmin=-max_v_value
    vmax=max_v_value
    
    ### Plot particle motion
    im=ax.scatter(x_graph,y_graph,s=None,c=time_out,cmap=plt.cm.get_cmap('coolwarm'),vmin=vmin,vmax=vmax)
    ax.plot(x_graph,y_graph,'-k',lw=0.5)
    if cbar:
        get_cax(ax,cax_width=0.03,pad=5)
        cbar=plt.colorbar(im)
        if flag_time:
            cbar.ax.set_ylabel('Times [s]')
        else:
            cbar.ax.set_ylabel('Samples')
          
    ### Plot arrow
    
    dx=x_graph[ind_clos+3]-x_graph[ind_clos]
    dy=y_graph[ind_clos+3]-y_graph[ind_clos]
    ax.plot(x_graph[ind_clos],y_graph[ind_clos],'ok')

    ax.quiver(x_graph[ind_clos],y_graph[ind_clos],dx,dy,facecolor='k',edgecolor='none')
    
    ### Plot labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    mag=1.1
    ax.set_xlim([np.min(x_graph)*mag,np.max(x_graph)*mag])
    ax.set_ylim([np.min(y_graph)*mag,np.max(y_graph)*mag])
    ax.set_aspect('equal','box')
    
    return ax


def plot_sta_az(single_event,ax=None,
                station_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/STATIONS/stations_axial_bis.xyz',inset=True,
                dict_cmap=None):
    """
    Function made to plot the geographical azimuth of the stations compared to an hypocenter read from an obspy.event object
    
    Input
    -----
    single_event:obspy.event object: event to be processed containing epicenter and phases
    ax: matplotlib.axes object: polar ax to add the plot
    station_file: str: path to text file containing 4 columns, lon,lat,depth, station_name
    inset: boolean: plot inset of stations or not
    
    Output
    -----
    ax: axes object: contains the plot
    
    """
    
    from general.GMT import get_Ncolors
    
    #### Get hypocenter coordinates
    
    lon_hyp=single_event.preferred_origin().longitude
    lat_hyp=single_event.preferred_origin().latitude
    
    ini_lon=lon_hyp
    ini_lat=lat_hyp
    
    x_hyp=0
    y_hyp=0 # Taken as center

    #### Load station file
            
    lon_sta,lat_sta,z_sta=np.loadtxt(station_file,unpack=True, usecols=[0,1,2])   
    sta_labels=[]
    with open(station_file,'rt') as fic:
        for line in fic:
            sta_labels.append(line.split()[-1])
            
    ### Define cmap_dict for each stations
    
    if dict_cmap is None:
        rgb_list=get_Ncolors(num_colors=len(sta_labels))
        print(rgb_list,sta_labels)
        dict_cmap={sta_labels[kk]:rgb_list[kk] for kk in range(len(sta_labels))}
                
    ### Convert sta to xy
    
    x_sta,y_sta=ll2xy(lon_sta,lat_sta,ini_lon,ini_lat)
    
    ### Get azimuth from hyp to sta (this is not the back azimuth, just regular azimuth)
    
    #angle_sta=get_angles(x_sta,y_sta,x_hyp,y_hyp,mode='geo')
    angle_sta=get_angles(x_sta,y_sta,x_hyp,y_hyp)

    #### Start plotting
    
    if ax is None:
        fig = plt.figure()
        if inset:
            gs = gridspec.GridSpec(3, 4)
            ax_inset=fig.add_subplot(gs[0, 0])
            ax=fig.add_subplot(gs[0:, 1:],projection='polar')
        else: 
            ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_direction(-1)  ### positive CW 
        ax.set_theta_offset(np.pi/2.0) ### from North

    
    ### Plot stations
    for kk in range(len(angle_sta)):
        ax.plot((0, angle_sta[kk]), ( 0, 1),':',color=dict_cmap[sta_labels[kk]],lw=2)
   
    ax.plot(angle_sta,np.ones(angle_sta.shape),'vw',mec='k',markersize=10)
   
    ax.set_rmin(0)

    ### Plot annotations
    kk=-1
    for label in sta_labels:
        kk+=1
        if angle_sta[kk]<=np.pi:
            rotation_text=(np.pi/2-angle_sta[kk])*180/np.pi
        else:
            rotation_text=(np.pi/2-angle_sta[kk])*180/np.pi + 180
        ax.annotate(label,
                    xy=(angle_sta[kk],1.01),xytext=(angle_sta[kk],1.2),horizontalalignment='center',
                    verticalalignment='center',rotation=rotation_text  # theta, radius
                    )   # 

    #### Plot inset
    
    if inset:
        ax_inset.axvline(linestyle=':',color='0.7',lw=1,zorder=0)
        ax_inset.axhline(linestyle=':',color='0.7',lw=1,zorder=0)
        ax_inset.plot(x_hyp,y_hyp,'ok')
        ax_inset.plot(x_sta,y_sta,'vw',mec='k')
        ax_inset.set_xlabel('X [km]')
        ax_inset.set_ylabel('Y [km]')
    ### Return
    
    return (ax,dict_cmap)


def rotate_stream_basis_hor(stream,rot_deg,flag_plot=False):
    """
    Function made to rotate the reference basis horizontal components
    Stream should be 'ZNE' or '123'
    The X or E component should be the last component
    The Y or N component should be the second component
    
    rot_deg: float: angle in degree for basis rotation (CW)
    
    Output
    ------
        r_stream: stream in new basis
    
    Example: if Old north component has a aziumuth of 20° (CW from North) just rotate
    the basis by a angle of 20° to get components in the real basis
    """
    
    ### Check
    check_stream3(stream)
    r_stream=stream.copy()
    data=stream2data(stream)

    data_x=data[:,2]
    data_y=data[:,1]
    data_xy=np.column_stack((data_x,data_y))
        
    ### Basis Rotation 
    
    rot_rad=rot_deg*np.pi/180 # rotation of the basis is equal to azimuth of North

    r_data_xy=rotate_basis_2(data_xy,rot_rad)
    
    ### Feed rotated stream
    
    r_stream[1].data=r_data_xy[:,1] # Y component
    r_stream[2].data=r_data_xy[:,0] # X component
    
    ### Plot if asked
    
    if flag_plot:

        fix,ax=plt.subplots(2,1,sharex=True,sharey=True)
        ax[0].plot(r_data_xy[:,0],'-k',label='New basis')
        ax[1].plot(r_data_xy[:,1],'-k',label='New basis')
        ax[0].plot(data_xy[:,0],'--r',label='Old basis')
        ax[1].plot(data_xy[:,1],'--r',label='Old basis')
        
        ### Cosmetic
        
        ax[0].legend()
        ax[0].set_ylabel('E')
        ax[1].set_ylabel('N')


    return r_stream