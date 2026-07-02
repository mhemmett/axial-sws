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


from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.io.nlloc.core import read_nlloc_hyp
from obspy.signal.polarization import particle_motion_odr,vidale_adapt,eigval
from obspy.taup import TauPyModel
from general.util import smooth_curve

from general.plotwaveform import SDS2streams,plot_event,getPickForArrival,getPicks,getPicksAndArrivals,plot_picks,stream2data,jurkevics
from general.plotwaveform import plot_window,plot_particle_motion,plot_particle_motion_stream
import shearwavesplit as sw
import general.plotwaveform as pltwf


###################"
### Parameters

file_range=[1,7]
file_range=[8,15]
file_range=[16,22]
plt.close('all')
sds_root='/home/baillard/SDS/'
start_time=UTCDateTime(2015,2,1)
time_delay=5
network_code='OO'
channel_code="EH?,HH?"
p_window=[0.02,0.1] ### for P Polarization analysis
s_window=[0.02,0.3] ### for S Shearwavesplitting
stream_window=[1,1]  #### for extraction of the stream, before P and after S
freqmin=5
freqmax=40
flag_plot=False
inc_thres=20 # incidence threshold
rec_thres=0.7 # rectilinearity threshold
log_file='SWS_Final.log'+str(file_range[0])+'_'+str(file_range[1])
event_max=5000
flag_split=False

### Obspy Client parameter to get waveforms
network_code="OO"
center="IRIS"
channel_code="EH*,HH*"

###  SWS parameters

min_lag=0
max_lag=60
Nlags=60
Nangles=90

## Paths
#nlloc_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial_EQ/PROG/NLLOC_MODELS/NLLOC_3H/loc/AXIAL.sum'
nlloc_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/CATALOG/AXIAL.PHASE.FINAL_3D.nlloc'
station_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/STATIONS/stations_axial.xyz'
pickle_dir='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial_SWS/PROG/pickles_FREQ_60_1'

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

temp_dir='SWS_tmp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

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
    nlloc_list=sw.split_nlloc(nlloc_file,event_max,prefix='split',output_dir=temp_dir)
else:
    index_file=np.arange(file_range[0],int(file_range[1])+1)
    nlloc_list=['./'+temp_dir+'/'+'split_%03i.nlloc'%x for x in index_file]


#################################
#### START LOOP CATALOG #########
#################################

logging.info('Starting Loop over nlloc catalogs')

for nlloc_file in nlloc_list:
    logging.info('Processing catalog: %s'%(nlloc_file))

    Cat=read_nlloc_hyp(nlloc_file)
    
    #################################
    #### START LOOP EVENT ###########
    #################################

    #for i_event in range(3985,len(Cat)):
    for i_event in range(len(Cat)):
        gc.collect()
        Out.clear()
        In.clear()
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
        
        for i_station in range(len(PS_list)):
            plt.close('all')
            single_PS=PS_list[i_station]
            try:
                
                ##################################################
                ### Extract stream for station and process it ####
                ##################################################
                
                station_code=single_PS[0][0].waveform_id.station_code
                p_time=UTCDateTime(single_PS[0][0].time)
                s_time=UTCDateTime(single_PS[1][0].time)
                
                ps_delay=s_time-p_time
                time_window=ps_delay+np.sum(stream_window)
                
                ##### Initialize class for storing and create IDs
                
                Obs=sw.SWSobs()
                obs_id=station_code+'_'+p_time.strftime('%Y%m%d_%H%M%S')
                event_id=single_event.origins[0].time.strftime('%Y%m%d_%H%M%S')
                
                ##### Initialize storing name and skip iteration if file already exists
                
                pickle_name=obs_id
                pickle_fullname=pickle_dir+'/'+pickle_name+'.pickle'
                
                if os.path.exists(pickle_fullname):
                    continue
                
   
                #### Get Waveform
                
    #            st_raw=SDS2streams(sds_root,p_time-stream_window[0],time_window,
    #                            network_code=network_code,station_code=station_code,channel_code=channel_code) ### Make sure st is length 3 (UR)
    #            
                
                Retry=sw.retry(10,120)
                new_get_waveforms=Retry(client.get_waveforms)
                st_raw=new_get_waveforms(network=network_code,station=station_code,location='*',channel=channel_code,
                                            starttime=p_time-stream_window[0],endtime=s_time+stream_window[1])
                                            
                
        
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
                
                az,inc,rec=pltwf.jurkevics(st_p)
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
                
    
           
                ### Process stream and convert to data_array
                
                ### Filter SNR... (UR)
                
                ### Select the correct components
                
                st_x=st_filter.select(channel='??E')
                st_y=st_filter.select(channel='??N')
                st_xy=st_x+st_y
                xy_array=stream2data(st_xy)
                
                ### Get window start and end in samples sw0 and sw1
                
                sw_window_time=[s_time-s_window[0],s_time+s_window[1]] 
                trace_start_time=st_xy[0].stats.starttime
                sampling_rate=st_xy[0].stats.sampling_rate
                sw1=int((sw_window_time[0]-trace_start_time)*sampling_rate)
                sw2=int((sw_window_time[1]-trace_start_time)*sampling_rate)
                s_samples=int((s_time-trace_start_time)*sampling_rate) # in samples
                
                ### Compute Dominant frequency on E(X) and N(Y)
                
                dom_freq_x=sw.dominant_freq(xy_array[:,0],1/sampling_rate)
                dom_freq_y=sw.dominant_freq(xy_array[:,1],1/sampling_rate)
                
                Obs.dom_freq={'x':dom_freq_x,'y':dom_freq_y}
                
                pickle.dump(Obs,open(pickle_fullname,'wb'))
                
                continue
                #### Save xy_array to pickle for testing (test)
                
    #            data_pickle=[xy_array,sw1,sw2]
    #            pickle.dump(data_pickle,open('test/'+obs_id,'wb'))
    #            
    #            continue
                 
                #### Perform analysis, cutting of the data is done inside the function
                
                (LAMBDA1, LAMBDA2, LAGS,ANGLES)=sw.get_LAMBDAS(xy_array,min_lag,max_lag,Nlags=Nlags,Nangles=Nangles,
                cut_s1=sw1,cut_s2=sw2)
                       
                #### Process image
                
    
                (MinLambdas,ax_lambda2)=sw.process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                                   min_thres=0.5,min_numbers=2,cont_step=0.01,quality_thres=0.2,zoom_factor=[2,6],
                                   flag_plot=flag_plot)
               
          
    
                ########## Plot SWS diagnosis plot #########
    
                if flag_plot:
                    #
                    
                    for kk in range(len(MinLambdas)):
                    
                        ### Initialize plotting but don't show
                        
                        print(MinLambdas[kk].rec)
                    
                        plt.ioff()
                        fig_sws=plt.figure(figsize=[ 8.1 ,7.8])
                        [ax_ini,ax_un1,ax_un2,ax_lambda2,ax_polar,ax_polar_un]=sw.ax_sws_diagnosis()
                        (MinLambdas,ax_lambda2)=sw.process_LAMBDAS(LAMBDA2,LAMBDA1,LAGS,ANGLES,
                            min_thres=0.5,min_numbers=2,cont_step=0.01,quality_thres=0.2,zoom_factor=[2,6],
                            flag_plot=True,ax=ax_lambda2)
                        
                        ### Plot unsplitted if asked
                        lag=MinLambdas[kk].lag
                        angle=MinLambdas[kk].angle
                        (xy_unsplitted,_)=sw.unsplit(xy_array,lag,angle,flag_plot=True,ax_list=[ax_ini,ax_un1,ax_un2])
                        plot_window(ax_ini,start=sw1,length=sw2-sw1,facecolor='0.9',zorder=-1,label='S')
                   
                     
                        ax_ini.axvline(x=s_samples,linestyle='-',color='k',lw=1,zorder=0)
                        
                        ### Plot particle motion if asked and unsplitted waveforms
                        xy_array_cut=sw.cut(xy_array,s0=sw1,s1=sw2)
                        xy_array_unsplit_cut=sw.cut(xy_unsplitted,s0=sw1,s1=sw2)
               
    
                        plot_particle_motion(xy_array_cut,time_array=None,ax=ax_polar,vmin=None,vmax=None,xlabel='X',ylabel='Y')
                        plot_particle_motion(xy_array_unsplit_cut,time_array=None,ax=ax_polar_un,vmin=None,vmax=None,xlabel='X',ylabel='Y')
                             
                        fig_sws.show()
                   
              
                    sw.hold_plot()
                    plt.ion()
        
          
    
            
                        
                #### Compute rectilinearity of newly unsplitted waveforms
                
    #            #eig_vals,eig_vecs=cov_eig(xy_array_cut)
    #            #lambda1,lambda2=eig_vals
    #            #rec_unsplit=1-(lambda2)/(lambda1)
    #            
    
#                xy_array_cut=sw.cut(xy_array,s0=sw1,s1=sw2)
#                xy_array_unsplit_cut=sw.cut(xy_unsplitted,s0=sw1,s1=sw2)
#                eig_vals,eig_vecs=pltwf.cov_eig(xy_array_unsplit_cut)
#                lambda1,lambda2=eig_vals
#                rec_unsplit=1-(lambda2)/(lambda1)
#                
                #### Feed obs
                
                Obs.MinLambdas=MinLambdas
                
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
##                exc_type, exc_obj, exc_tb = sys.exc_info()
##                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
##                print(exc_type, fname, exc_tb.tb_lineno)
#                print(e)
#                traceback.clear_frames(e.__traceback__)
#                del e
#                pass
            

        #################################
        #### END LOOP EVENT ###########
        #################################







    


