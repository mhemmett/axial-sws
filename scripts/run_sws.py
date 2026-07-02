#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:04:28 2018

@author: baillard

This is the main script to do shearwavesplitting analysis automatically

For shear wave splitting we use the convention of a Nx2 data array with the first column being the X/E component and the second column 
being the Y/N direction
    

"""


import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.patches import Rectangle
import os,pickle,time
import logging
import time,sys
import gc
import traceback
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) ## For scipy.fft

import obspy
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.io.nlloc.core import read_nlloc_hyp
from obspy.signal.polarization import particle_motion_odr,vidale_adapt,eigval
from obspy.taup import TauPyModel
from util import smooth_curve

from plotwaveform import SDS2streams,plot_event,getPickForArrival,getPicks,getPicksAndArrivals,plot_picks,stream2data,jurkevics
from plotwaveform import plot_window,plot_particle_motion,plot_particle_motion_stream
import shearwavesplit as sw
import plotwaveform as pltwf
import sws_methods as swm


###################"
### Parameters


file_range=[1,2]
#file_range=[13,15]
#file_range=[20,22]
plt.close('all')
sds_root='/home/baillard/SDS/'
nlloc_dir_tmp='SWS_nlloc'
flag_split=False
start_time=UTCDateTime(2015,2,1)
time_delay=5
network_code='OO'
channel_code="EH?,HH?"
p_window=[0.02,0.1] ### for P Polarization analysis
s_window=[0.02,0.3] ### for S Shearwavesplitting 
fs_window=[0.1,0.3] ### To compute dominant period of S waves
s_snr_window=[0.4,0.2]
s_snr_thres=2 ### SNR threshod to keep data
flag_adapt_window=True ### Adapt size of the SWS window based on dominant period (T*2)
flag_adapt_maxlag=True ### Adapt max lagg allows for SWS based on dominant period (T)
stream_window=[1,1]  #### for extraction of the stream, before P and after S
freqmin=5
freqmax=40
flag_plot=True
inc_thres=30 # incidence threshold
rec_thres=0.7 # rectilinearity threshold
log_file='SWS_Final.log'+str(file_range[0])+'_'+str(file_range[1])
event_max=5000


### Obspy Client parameter to get waveforms
network_code="OO"
center="IRIS"
channel_code="EH*,HH*"

###  SWS parameters
min_lag=0
max_lag=60
Nlags=60
Nangles=90
add_lag=3 # Lag in samples to extend the LAMBDAS mesh to avoid extrema to be picked at borders (should not be changed)

print('Make sure that S window is longer than max_lag, otherwise Y component will not be in window')

## Paths
#nlloc_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial_EQ/PROG/NLLOC_MODELS/NLLOC_3H/loc/AXIAL.sum'
#nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/CATALOG/AXIAL.PHASE.FINAL_3D_V2.nlloc'
#station_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/STATIONS/stations_axial.xyz'
#pickle_dir='/home/baillard/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FINAL_ADAPT_2'
nlloc_file = '../data/AXIAL.PHASE.FINAL_3D_V2.nlloc'
station_file = '../data/stations_axial.xyz'
pickle_dir = '../pickles'

##### Initialize logging

logger = logging.getLogger()
logger.handlers=[]
logger.setLevel(logging.INFO)
fh= logging.FileHandler(log_file,mode='w')
sh= logging.StreamHandler()
logger.addHandler(fh)
logger.addHandler(sh)

####

logging.info('Logging file for Shear Wave Splitting')

##### directory for temp files


if not os.path.exists(nlloc_dir_tmp):
    os.makedirs(nlloc_dir_tmp)

###########################################
####  Create directory to store pickles ###
###########################################

if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)
else:
    user_input=input('Following directory already exists:\n%s\nContinue? [y/n]'%pickle_dir)
    if user_input!='y':
        sys.exit('Program terminated by user')
    
##########################################
########## INITIALIZE IRIS CLIENT ########
##########################################
        
client = Client(center) 


################################################
######### Split and Read catalog file ##########
################################################
if flag_split:
    nlloc_list=sw.split_nlloc(nlloc_file,event_max,prefix='split',output_dir=nlloc_dir_tmp)
else:
    index_file=np.arange(file_range[0],int(file_range[1])+1)
    nlloc_list=['./'+nlloc_dir_tmp+'/'+'split_%03i.nlloc'%x for x in index_file]


#################################
#### START LOOP CATALOG #########
#################################

logging.info('Starting Loop over nlloc catalogs')

for nlloc_file in nlloc_list:
    logging.info('Processing catalog: %s'%(nlloc_file))

    #Cat=read_nlloc_hyp(nlloc_file)
    Cat = obspy.read_events(nlloc_file)
    
    #################################
    #### START LOOP EVENT ###########
    #################################

    #for i_event in range(3985,len(Cat)):
    for i_event in range(len(Cat)):
        # %%
        gc.collect()
    #for i_event in [len(Cat)-2]:
        
        single_event=Cat[i_event]
        logging.info('Processing catalog: %s'%(nlloc_file))
        logging.info('Processing %i event' %(i_event))
        
        #### Select stations that have both P and S picks 
        
        P_list=pltwf.getPicksAndArrivals(single_event,network_code=None,station_code=None,phase_hint='P')
        S_list=pltwf.getPicksAndArrivals(single_event,network_code=None,station_code=None,phase_hint='S')
        
        PS_list=[[P_pair,S_pair] for P_pair in P_list for S_pair in S_list if P_pair[0].waveform_id.station_code==S_pair[0].waveform_id.station_code]
        
        #################################
        ##### START LOOP STATION ########
        #################################
        # %% Cell
        for i_station in range(len(PS_list)):
        #for i_station in [3]:
            single_PS=PS_list[i_station]
            station_code=single_PS[0][0].waveform_id.station_code
            logging.info('Processing station %s'%station_code)
            
            try:
                
                plt.close('all')
                
                ##################################################
                ### Extract stream for station and process it ####
                ##################################################
                
                station_code=single_PS[0][0].waveform_id.station_code
                p_time=UTCDateTime(single_PS[0][0].time) # it's already in UTCDateTime UR
                s_time=UTCDateTime(single_PS[1][0].time)
                
                ps_delay=s_time-p_time
                time_window=ps_delay+np.sum(stream_window)
                
                ##### Initialize class for storing and create IDs
                
                Obs=sw.SWSobs()
                obs_id=station_code+'_'+p_time.strftime('%Y%m%d_%H%M%S') # Use also milliseconds UR
                event_id=single_event.origins[0].time.strftime('%Y%m%d_%H%M%S')
                #o_time , add origin time to compute Vp/Vs (UR)
                
                ##### Initialize storing name and skip iteration if file already exists
                
                pickle_name=obs_id
                pickle_fullname=pickle_dir+'/'+pickle_name+'.pickle'
                
                logging.info('Check id existence')
                start=time.time()
                if os.path.exists(pickle_fullname):
                    logging.info('Following file already exists, skip.\n %s'%(pickle_fullname))
                    logging.info('time spent %f s'%(time.time()-start))
                    continue
                
                logging.info('time spent %f s'%(time.time()-start))
                
                
                #### Get Waveform
                
    #            st_raw=SDS2streams(sds_root,p_time-stream_window[0],time_window,
    #                            network_code=network_code,station_code=station_code,channel_code=channel_code) ### Make sure st is length 3 (UR)
    #            
                logging.info('Get Waveform')
           
        
                start=time.time()
                Retry=sw.retry(100,1) ## retry(retries,wait_time)
                new_get_waveforms=Retry(client.get_waveforms)
                start=time.time()
                st_raw=new_get_waveforms(network=network_code,station=station_code,location='*',channel=channel_code,
                                            starttime=p_time-stream_window[0],endtime=s_time+stream_window[1])
                                                
                logging.info('time spent for waveform       %f s'%(time.time()-start))
                
        
                #### Feed Obs
                
                Obs.obs_id=obs_id
                Obs.event_id=event_id
                Obs.station_code=station_code
                Obs.event_lon=single_event.origins[0].longitude
                Obs.event_lat=single_event.origins[0].latitude
                Obs.event_depth=single_event.origins[0].depth/1000
                Obs.p_time=p_time
                Obs.s_time=s_time
                
                ### Filter stream (make sure that tapering doesn't not affect P and S amplitudes)
                
                st_filter=st_raw.copy()
                st_filter.sort(reverse=True) # to get ZNE
                st_filter.detrend("linear") # to avoid weird start and end amplitudes
                st_filter.taper(max_percentage=0.05, type="hann")
                st_filter.filter("bandpass",freqmin=freqmin,freqmax=freqmax)
                
                if flag_plot:
                    stream_fig=st_filter.plot(handle=True)
                    plot_picks(stream_fig,st_filter,single_event,phases=['P','S'])
                
                
                ##########################################
                ###### P-Polarization analysis ###########
                ##########################################
                
                st_rotate=st_filter.copy()
                
                if flag_plot: ### Plot windows 
                    ax_list=stream_fig.get_axes()
                    ax=ax_list[0]
                    plot_window(ax,p_time-p_window[0],np.sum(p_window),label='P')
                    plot_window(ax_list[1],s_time-s_window[0],np.sum(s_window),label='S')
                    plot_window(ax_list[2],s_time-s_window[0],np.sum(s_window),label='S')
                
                
                #### Cut the stream around P
                    
                st_p=st_filter.slice(p_time-p_window[0],p_time+p_window[1])
                
                if flag_plot:
                    st_p.plot(handle=True,type='relative')
                    ax=plot_particle_motion_stream(st_p,ref_comp='N',ref_time=p_time)
                
                ##### Compute azimuth
                
                (az,inc,rec,big_eigvec)=pltwf.jurkevics(st_p)
                logging.info('az= %.2f, inc= %.2f, rec= %.2f'%(az,inc,rec)) # LOG
                
                #azz,incc,err_azz,err_incc=particle_motion_odr(st_synth)
                
                if flag_plot:
                    st_rotate.rotate('ZNE->LQT', back_azimuth=az, inclination=inc)
                    st_rotate.plot(handle=True)
                
                
                #### Check if pass Thresholds
                    
                if inc>=inc_thres:
                    logging.info('Incidence angle to high, observation skipped, next') # LOG
                    continue #(UR)
                    
                #### Feed SWSobs class
                    
                Obs.rec=rec
                Obs.az=az
                Obs.inc=inc
                
                ################################
                #######  SWS Analysis ##########
                ################################
                
                ### Select the correct components
                
                st_x=st_filter.select(channel='??E')
                st_y=st_filter.select(channel='??N')
                st_xy=st_x+st_y
                xy_array=stream2data(st_xy)
                
                #############################################
                ########## COMPUTE PREDOMINANT PERIOD #######
                #############################################
                
                ### Define properly windows sw1 and sw2 based on dominant period
                
                fs_window_time=[s_time-fs_window[0],s_time+fs_window[1]] 
                s_window_time=[s_time-s_window[0],s_time+s_window[1]] 
                trace_start_time=st_xy[0].stats.starttime
                sampling_rate=st_xy[0].stats.sampling_rate
                s_samples=int(round((s_time-trace_start_time)*sampling_rate)) # in samples
                p_samples=int(round((p_time-trace_start_time)*sampling_rate)) # in samples
                fs_w1=int(round((fs_window_time[0]-trace_start_time)*sampling_rate))
                fs_w2=int(round((fs_window_time[1]-trace_start_time)*sampling_rate))
                mid_samples=int(round(p_samples+(s_samples-p_samples)/2))
                sw1=int(round((s_window_time[0]-trace_start_time)*sampling_rate))
                
                ### Make sure left side of the window is bigger than P+(S-P)/2
                if fs_w1<mid_samples:
                    fs_w1=mid_samples
                
                ### Cut the data between fs_w1 and fs_w2
                
                xy_array_dom=xy_array[fs_w1:fs_w2,:]
                
                ### Get dominant period on X and Y (given in samples)
                
                (dom_period_x,dom_freq_x)=sw.get_dominant_period(xy_array_dom[:,0],sampling_rate,flag_plot=False)
                (dom_period_y,dom_freq_y)=sw.get_dominant_period(xy_array_dom[:,1],sampling_rate,flag_plot=False)
            
                dom_period=np.mean([dom_period_x,dom_period_y]) # Take the mean dominant period
                
                ### Adapt window size to perform splitting and max lag allowed
                
                if flag_adapt_window:
                    sw2=int(round(sw1+2*dom_period)) # Check Wuestfeld et al., 2010
                else:
                    sw2=int(round((s_window_time[1]-trace_start_time)*sampling_rate))
                    
                if flag_adapt_maxlag:
                    max_lag=int(round(dom_period))

                Obs.dom_period={'x':dom_period_x/sampling_rate,'y':dom_period_y/sampling_rate}
                
                #################### END DOMINANT PERIOD ####################
            
                #########################################
                ####### COMPUTE SNR AROUND S PICK #######
                #########################################
                
                ### Define windows
                
                s_snr_window_time=[s_time-s_snr_window[0],s_time+s_snr_window[1]] 
                s_snr_w1=int(round((s_snr_window_time[0]-trace_start_time)*sampling_rate))
                s_snr_w2=int(round((s_snr_window_time[1]-trace_start_time)*sampling_rate))
                
                ### Make sure left side of the window is bigger than P+(S-P)/2
                if s_snr_w1<mid_samples:
                    s_snr_w1=mid_samples
                    
                if s_snr_w2>sw2:
                    s_snr_w2=sw2
                
                ### Compute
                
                snr_x=swm.SNR_pick(xy_array[:,0],s_samples,
                                   s_samples-s_snr_w1,s_snr_w2-s_samples,mode='mean',flag_plot=flag_plot)
                snr_y=swm.SNR_pick(xy_array[:,1],s_samples,
                                   s_samples-s_snr_w1,s_snr_w2-s_samples,mode='mean',flag_plot=flag_plot)
                
                ### Feed obs
                
                s_snr=np.mean((snr_x,snr_y))
                logging.info('SNR S: %f'%s_snr)
                
                if s_snr<=s_snr_thres:
                    logging.info('s_snr is %f < %f threshold, skip observation'%(s_snr,s_snr_thres))
                    continue
                
                Obs.s_snr=np.mean((snr_x,snr_y))
                
                #################### END COMPUTE SNR ####################
                
                ##################################
                ####### COMPUTE LAMBDA MIN #######
                ##################################
                
#                #### Save xy_array to pickle for testing (test)
#                
#                data_pickle=[xy_array,sw1,sw2]
#                pickle.dump(data_pickle,open('test/'+obs_id,'wb'))               
#                continue
                 
                #### Perform analysis, cutting of the data is done inside the function
                
                (LAMBDA1, LAMBDA2, LAGS,ANGLES)=sw.get_LAMBDAS(xy_array,
                min_lag-add_lag,max_lag+add_lag,Nlags=Nlags,Nangles=Nangles,
                cut_s1=sw1,cut_s2=sw2)
                       
                #### Process image
                
                logging.info('Process Lambda')
                (MinLambdas,ax_lambda2)=sw.process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                    min_lag_thres=min_lag,max_lag_thres=max_lag,
                                   min_thres=0.5,min_numbers=2,cont_step=0.01,quality_thres=0.2,zoom_factor=[2,6],
                                   flag_plot=flag_plot)
               
                
                #################### END COMPUTE LAMBDA MIN ####################
                
                ########################################
                ####### COMPUTE RMS FOR EACH MIN #######
                ########################################
                
                ### Compute RMS of unsplit and sort list based on RMS
                
                MinLambdas=sw.rms_MinLambdas(xy_array,sw1,sw2,MinLambdas,mode='norm',flag_plot=flag_plot)
                
                #### Feed obs
                
                Obs.MinLambdas=MinLambdas
                
                #################### END COMPUTE RMS ####################
                
                ########## Plot SWS diagnosis plot #########
    
                if flag_plot:

                    for kk in range(len(MinLambdas)):
                    
                        ### Initialize plotting but don't show
                        
                    
                        fig_sws=plt.figure(figsize=[ 8.1 ,7.8])

                        # Add a title to the whole figure so it is easier to identify
                        fig_sws.suptitle(f'{station_code} - {event_id}', fontsize=14, y=0.05)

                        [ax_ini,ax_un1,ax_un2,ax_lambda2,ax_polar,ax_polar_un]=sw.ax_sws_diagnosis()
                        (_,ax_lambda2)=sw.process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                        min_lag_thres=min_lag,max_lag_thres=max_lag,
                            min_thres=0.5,min_numbers=2,cont_step=0.01,quality_thres=0.2,zoom_factor=[2,6],
                            flag_plot=True,ax=ax_lambda2)
                        
                        ### Plot unsplitted if asked
                        lag=MinLambdas[kk].lag
                        angle=MinLambdas[kk].angle
                        (xy_unsplitted,_)=sw.unsplit(xy_array,lag,angle,flag_plot=True,ax_list=[ax_ini,ax_un1,ax_un2])
                        ax_un2.text(0.1, 0.1, 'RMS = %.3f'%MinLambdas[kk].rms, transform=ax_un2.transAxes)
                        ax_ini.set_title('Lag=%.1f, Angle=%.2f'%(lag,angle))
                        plot_window(ax_ini,sw1,sw2-sw1,facecolor='0.9',zorder=-1,label='S')
                   
                        ax_ini.axvline(x=s_samples,linestyle='-',color='k',lw=1,zorder=0)
                        
                        ### Plot particle motion if asked and unsplitted waveforms
                        xy_array_cut=sw.cut(xy_array,s0=sw1,s1=sw2)
                        xy_array_unsplit_cut=sw.cut(xy_unsplitted,s0=sw1,s1=sw2)
               
    
                        plot_particle_motion(xy_array_cut,time_array=None,ax=ax_polar,vmin=None,vmax=None,xlabel='X',ylabel='Y')
                        plot_particle_motion(xy_array_unsplit_cut,time_array=None,ax=ax_polar_un,vmin=None,vmax=None,xlabel='X',ylabel='Y')
                        
                        # save figure to output for direct comparison with my implementation
                        fig_sws.savefig(f'./figures/{station_code}_{event_id}_{kk}.png')
                        
                        fig_sws.show()
                   
                ########## END SWS diagnosis plot #########
        
                
                ###############################
                #### Store in pickle ##########
                ###############################

                pickle.dump(Obs,open(pickle_fullname,'wb'))
    
                
                ################################
                ###### END LOOP STATION ########
                ################################
                
            except ValueError:
                pass
             
#            except Exception as e: # Some memory leakage are associated to traceback 
#                exc_type, exc_obj, exc_tb = sys.exc_info()
#                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#                print(exc_type, fname, exc_tb.tb_lineno)
#                print(e)
#                traceback.clear_frames(e.__traceback__)
#                del e
#                pass
            

        #################################
        #### END LOOP EVENT ###########
        #################################







    


