#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:24:49 2018

@author: baillard
"""

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import pickle
import glob
import time
from obspy import UTCDateTime
import warnings
import scipy
import matplotlib


from scipy.ndimage import zoom,gaussian_filter
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import datetime as dt
import copy
import inspect
import datetime as dt
from matplotlib.ticker import FuncFormatter
import logging
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class,get_projection_names
import sws_methods as swm


import projection as gproj
import util as gutil
import GMT as ggmt
import shearwavesplit as sws
import sws_vertices as swv




##########################"
###### Functions

# select

# plot

# get_xyz

#############################
###### Classes
class SWScat():
    """
    Class made to store SWSobs classes. Please see obspy.catalog for details
    """
    
    def __init__(self,obs=None):
        if not obs:
            self.obs = []
        else:
            self.obs = obs
            
        self.station_list=[]
        
        self.update()
            
    def __getitem__(self, index):
        """
        __getitem__ method of the Catalog object to allow list accessibility
        :return: Event objects
        """

        if isinstance(index, slice):
            print(index)
            return self.__class__(obs=self.obs.__getitem__(index))
        else:
            return self.obs.__getitem__(index)
        
    def __str__(self):
        out= str(len(self.obs)) + ' SWS observation(s) in Catalog:\n'
            
        if len(self.obs)>=1:
            out+='station: %s time: %s' %(self.obs[0].station_code,self.obs[0].s_time.strftime('%Y-%m-%dT%H:%M:%S'))  
            out+='\n...\n'
            out+='station: %s time: %s' %(self.obs[-1].station_code,self.obs[-1].s_time.strftime('%Y-%m-%dT%H:%M:%S')) 
            
        return out
    
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
        
    #### Methods
    
    def update(self):
        self.update_stationlist()
        
    def compute_ini(self):
        self.compute_baz_trigo()
        self.compute_hyp_dist()
        
    def update_stationlist(self):
        """
        List all stations present in Catalog
        """
        station_list=list(set([x.station_code for x in self.obs]))
        self.station_list=station_list
        
    def remove_duplicates(self):
        """
        Function made to remove any duplicates based on obs_id,
        this is particularly useful when selecting events geographically for overlapping contours
        
        Outputs
        -------
            New_Cat: SWSCat object
        
        """
        
        obs=self.obs
        new_obs=list(set(obs))
        New_Cat=SWScat(obs=new_obs)
        
        ### Sort
        
        New_Cat=New_Cat.sort()
        
        return New_Cat
    
    def sort(self,param='obs_id'):
        """
        Sort SWScat object based on given parameter, default obs_id
        """
        values=[getattr(x,param) for x in self]
        ind_sorted=np.argsort(values)
        obs=self.obs
        new_obs=[obs[i] for i in ind_sorted]
        New_Cat=SWScat(obs=new_obs)
        
        return New_Cat
    
    def compute_baz_trigo(self,ini_lon=-130.1,ini_lat=45.9):
        """
        Compute baz 
        """
        station_dic=read_stationfile()

        for station in self.station_list:
            sta_lon=station_dic[station]['lon']
            sta_lat=station_dic[station]['lat']
        
            [x.get_baz_trigo(sta_lon,sta_lat,ini_lon,ini_lat) for x in self.obs if x.station_code==station]
    
    def compute_epi_dist(self,ini_lon=-130.1,ini_lat=45.9):
        """
        """
        station_dic=read_stationfile()

        for station in self.station_list:
            sta_lon=station_dic[station]['lon']
            sta_lat=station_dic[station]['lat']
        
            [x.get_epi_dist(sta_lon,sta_lat,ini_lon,ini_lat) for x in self.obs if x.station_code==station]
    
        
    def compute_hyp_dist(self,ini_lon=-130.1,ini_lat=45.9,z_ref=1.525):
        """
        Method made to compute the hypocenter distance (distance between the station and the event)
        
        Input
        -----
            ini_lon,ini_lat: float: ref lon,lat used for projection
            z_ref: float: reference depth used for hypocenter depth
        Output
        ------
            Update the catalog class
        UsedIn
        ------
            sws_methods.read_pickles
        """
        station_dic=read_stationfile()

        for station in self.station_list:
            sta_lon=station_dic[station]['lon']
            sta_lat=station_dic[station]['lat']
            sta_z=station_dic[station]['z']-z_ref
        
            [x.get_hyp_dist(sta_lon,sta_lat,sta_z,ini_lon,ini_lat) for x in self.obs if x.station_code==station]
    
    
    def write_pickle(self,pickle_file):
        """
        Method to write the SWScat into a pickle, the Cat is not pickled directly, 
        but the list of SWSobs inside that catalog (insure )
        """
        
        pickle_dir='/'.join(pickle_file.split('/')[:-1])+'/'
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        pickle.dump(self.obs,open(pickle_file,'wb'))
        
    def select(self,print_keys=False,**kwargs):
        """
        Defines a new SWSCat object based on given parameters.
        Please call SWSCat_obj.select(print_keys=True) to get the entire list of possible keys
        
    
        """
        
        if print_keys:
            print(list(get_funckeys(sws.SWSobs().match)))
            print(list(get_funckeys(sws.MinLambda().match)))
            print(['lambda_select','obs_contour_keys'])
            
        
        obss=self.obs
        #obss=self.obs

        
        ### Define arguments to be put into function
        
        obs_kwargs= {key:value for (key,value) in kwargs.items() if key in get_funckeys(sws.SWSobs().match)}
        lambda_kwargs= {key:value for (key,value) in kwargs.items() if key in get_funckeys(sws.MinLambda().match)}
        minlambda_arg = kwargs.get('lambda_select',None)
        contour_keys=kwargs.get('obs_contour_keys',None)
    
        ### Start selection of Obs
        
        sel_obs=[]
        
        for obs in obss:
            bool_obs=obs.match(**obs_kwargs)
     
            if not bool_obs:
                continue
            
            sel_lambda=[x for x in obs.MinLambdas if x.match(**lambda_kwargs)]
    
            if len(sel_lambda)==0:
                continue
            
            ### copy
            new_obs=sws.SWSobs()
            new_obs.__dict__=obs.__dict__.copy()
            new_obs.MinLambdas=sel_lambda
            
            if minlambda_arg is not None:
                sel_lambda=new_obs.minlambda_select(minlambda_arg[0],minlambda_arg[1])
            
            new_obs.MinLambdas=sel_lambda
            sel_obs.append(new_obs)
    
        ### Return
        
        new_cat=SWScat(obs=sel_obs)
        
        ### Geographic constrain in contours
        
        new_cat=new_cat.select_incontour(contour_keys=contour_keys)
        
        ### Update cat
        
        new_cat.update()
        
        return new_cat
    
    def select_incontour(self,contour_keys=None):
        """
        Method to select events that are geographically inside contours defined in the 
        contours.dat and vertices.dat files
        
        Inputs
        ------
            contour_keys: str,list: key defining each contour
            
        Output
        ------
            New_Cat: SWScat object
        """
        
        if contour_keys is None:
            return self
        
        contour_keys=sws.tolist(contour_keys)
        obs=self.obs
        
        #### Convert lon and lat into x,y
        
        lon=[x.event_lon for x in self.obs]
        lat=[x.event_lat for x in self.obs]
        
        x_ev,y_ev=gproj.ll2xy(lon,lat,-130.1,45.9) 
        
        #### Get polygon limits
        
        contour_list=swv.read_contours()
        
        sel_obs=[]
        for contour_key in contour_keys:
            try:
                xy=contour_list.get_xy(contour_key=contour_key)
            except:
                continue
            x_p,y_p=list(zip(*xy))
            
            #### Check whcih is inside polygon
            _,_,bool_in=gproj.is_in_polygon(x_ev,y_ev,x_p,y_p)
        
            ind_in=list(np.where(bool_in==True)[0])
            sel_obs+=[obs[i] for i in ind_in]
        
        New_Cat=SWScat(obs=sel_obs)
        New_Cat=New_Cat.remove_duplicates()
        
        return New_Cat
        
    def get_array(self,param,level=1):
        """
        Level is here so that we know if we have to look inside MinLambdas or not
        """
        
        if level==1:
            value=np.array([getattr(y,param) for y in self.obs for x in y.MinLambdas])
        elif level==2:
            value=np.array([getattr(x,param) for y in self.obs for x in y.MinLambdas])
            
        return value
    
    def get_dic(self,minlambda_select='min',ini_lon=-130.1,ini_lat=45.9):

        """
        Function made to convert the catalog into x,y,z,lags,fasts and s_times and into a dirctionnary
        
        Input:
            Cat: sws_method.Cat class: Containing SWS observations
            minlambda_select: str: defines if you want all min lambdas per obs or just the first one
                if 'all' then we take the values of all lambdas present in MinLambdas
        Output:
            xs,ys,zs,fasts,lags,s_times: np.array containing hypocenter coordinates and 
        """
        #############################
        ### Create arrays of data ###
        #############################
        
        lons,lats,zs,lags,fasts,s_times=[],[],[],[],[],[]
        baz_trigos,dom_periods,hyp_dists=[],[],[]
        
        
        for obs in self.obs:
            if len(obs.MinLambdas)==0:
                continue
            lon=obs.event_lon
            lat=obs.event_lat
            z=obs.event_depth
            hyp_dist=obs.hyp_dist
            baz_trigo=obs.baz_trigo
            dom_period=np.mean([obs.dom_period['x'],obs.dom_period['y']])
            
            ### If ALL
            if minlambda_select=='all':
                for MinLambda in obs.MinLambdas:
                    lag=MinLambda.lag # Only take the minimum lamba value
                    fast=MinLambda.angle
                    
                    ### Put into list
            
                    s_times.append(obs.s_time)
                    lons.append(lon)
                    lats.append(lat)
                    zs.append(z)
                    lags.append(lag)
                    fasts.append(fast)
                    hyp_dists.append(hyp_dist)
                    baz_trigos.append(baz_trigo)
                    dom_periods.append(dom_period)
                    
            ### I min
            else:
 
                lag=obs.MinLambdas[0].lag # Only take the minimum lamba value
                fast=obs.MinLambdas[0].angle
                
                ### Put into list
        
                s_times.append(obs.s_time)
                lons.append(lon)
                lats.append(lat)
                zs.append(z)
                lags.append(lag)
                fasts.append(fast)
                hyp_dists.append(hyp_dist)
                baz_trigos.append(baz_trigo)
                dom_periods.append(dom_period)
                    
        ### Convert to arrays
        
        lons=np.array(lons)
        lats=np.array(lats)
        ### Do conversion away from loop to gain some time
        xs,ys=gproj.ll2xy(lons,lats,ini_lon,ini_lat) # Project
        zs=np.array(zs)
        lags=np.array(lags)
        fasts=np.array(fasts)
        s_times=np.array(s_times)
        hyp_dists=np.array(hyp_dists)
        baz_trigos=np.array(baz_trigos)
        dom_periods=np.array(dom_periods)
        
        return {'x':xs,'y':ys,'z':zs,'lag':lags,'fast':fasts,'s_time':s_times,
                'baz_trigo':baz_trigos,'dom_period':dom_periods,
                'hyp_dist':hyp_dists}
        
    
    
    #### Plot methods ####
    
    def plot_contours(self,ax=None,contour_keys=None,**kwargs_contour):
  
        if ax is None:
            fig,ax=plt.subplots()
            
        contour_list=swv.read_contours()
        
        contour_list.plot_contours(ax=ax,contour_keys=contour_keys,**kwargs_contour)
        
        return ax
    
    def plot_polar_histofast(self,ax=None,facecolor=None,width_bin_deg=10,lw=1,edgecolor=None,
                             show_axis=True,
                             label_mode='trigo'):
        """
        Plot
        """
        

        ### Checks
        
        if label_mode not in ['trigo','azimuth']:
            raise ValueError('label mode should be either trigo or azimuth')
        
        if ax is None:
            fig,ax=plt.subplots(subplot_kw=dict(projection='polar'))
            
        if facecolor is None:
            facecolor=np.array([30,144,255])/255
            
        if edgecolor is None:
            edgecolor=facecolor
        ### Process
        
        elems_dic=self.get_dic()
        y=elems_dic['fast']*180/np.pi # in degrees
        y=np.hstack((y,y+180)) # Extend to
        
        ### Plot

        ax=sws.plot_polar_histo(y*np.pi/180,weights=None,width_bin_deg=width_bin_deg,
                                ax_polar=ax,facecolor=facecolor,edgecolor=edgecolor,alpha=1,lw=lw,text=False)
        
        ### Cosmetic

        ax.set_xticks(np.arange(0,360,30)*np.pi/180)
        if label_mode=='azimuth':
            trigo2azimuth_labels(ax,axis='x')
            

            
        if not show_axis:
            ax.tick_params(axis='both',labelbottom='off')
            
        ax.set_xlabel('$\Phi$ [°]') 
        
        ### Return
        
        return ax
    
    def plot_polar_histofast_multi(self,ax=None,num_ax=4,theta_rot=0,label_mode='trigo',
                                   show_axis=False,
                             width_bin_deg=10,facecolor=None,lw=1,edgecolor=None,norm=False,title=None):
        """
        Method made to plot multiple fast lag 
        """
        
        ### Get mutliple axes around center
        
        
        if title is None:
            title=','.join(self.station_list)
        (ax_main,ax_insets,theta_edges,r_edges,r_inner,r_width)=get_ax_polarinsets(ax_main=ax,num_ax=num_ax,theta_rot=theta_rot,title=title)
        section_theta_deg=(theta_edges[1]-theta_edges[0])*180/np.pi
        
        #### Find edge suitable for plotting the upper left polar plot (histo)
        
        ind_theta_inset=np.argmin(np.abs(theta_edges-135*np.pi/180))
        theta_inset=theta_edges[ind_theta_inset]
        
        r_center_inset=(r_edges+1)/2
        r_width_inset=(r_edges-1)*0.9
        (x_inset,y_inset)=pol2cart(r_center_inset, theta_inset)

        
        #### Select events based on baz_trigo
        
        y_lims=[]
        for kk,ax_inset in enumerate(ax_insets):
            Cat_select=self.select(obs_baz_trigo=[theta_edges[kk],theta_edges[kk+1]])
            ax_inset.get_yaxis().set_visible(True)
            ax_inset.get_xaxis().set_visible(True)
            Cat_select.plot_polar_histofast(ax=ax_inset,facecolor=facecolor,width_bin_deg=10,lw=lw,edgecolor=edgecolor,
                                            label_mode=label_mode,show_axis=show_axis)
            y_lims.append(ax_inset.get_ylim()[1])
            ax_inset.set_xlabel('')
        
            
        #### Normalize if asked
        
        if norm:
            y_lim_max=max(y_lims)
            for ax_inset in ax_insets:
                ax_inset.set_ylim([0,y_lim_max])
                
        #### Plot ax_middle
            
        ax_middle=get_ax_inset(ax_main,0,0,0.7*(r_inner*2),height=None,projection='polar',alpha=1,visible_axis=True)
        
        self.plot_polar_histofast(ax=ax_middle,facecolor=facecolor,width_bin_deg=10,lw=lw,edgecolor=edgecolor,
                                  label_mode=label_mode,show_axis=show_axis)
        
        ax_middle.set_xlabel('')
        ax_middle.tick_params(axis='x', which='major', pad=0)
        
        
        #### Plot histo on inset
        
        ax_top=get_ax_inset(ax_main,x_inset,y_inset,r_width_inset,height=None,projection='polar',alpha=1,visible_axis=False)
        trigo_angles=[x.baz_trigo for x in self.obs]
        sws.plot_polar_histo(trigo_angles,weights=None,width_bin_deg=section_theta_deg,ax_polar=ax_top,facecolor=[0,0,0.5],edgecolor='w',alpha=1,text=True)
        plt.setp(ax_top.spines.values(), color='w') # White color for polar frame
        ax_top.set_facecolor('w') # white background
        max_val=ax_top.get_ylim()[1]
        for theta in theta_edges:
            ax_top.plot([theta,theta],[0,max_val],'-k',lw=0.5)
    
        ### return
        
        return (ax_main,ax_insets,ax_middle,ax_top)
        
    
            
    def plot_movehisto2d_time(self,y_param,minlambda_select='min', 
                              x_label='X',y_label='Y',title='',ax=None,
                              angle_mode='trigo',lag_mode='sample',sampling_rate=200,
                              window_unit=None,window_bottom=None,window_top=None,
                              **movehisto2d_kwargs):
        """
        Function made to use movehist_2d and show variations of splitting parameters with time
        parameters are similat to the one for movehisto2d
        Plots showing angles used trigo convention by default (ie CCW from East) however angles can
        be shown using azimuth conention by specifyin angle_mode='azimuth', in that case all paramters
        should be given using that convention (ie. y_width, y_start,y_cycle...)
        
        parameter should be given in degrees
        
        Input
        ----
            angle_mode: str: ['trigo','azimuth']
            lag_mode: str: ['sample','ms','s']
            sampling_rate: float: sampling rate of the data to convert lags from samples to ms
        """

        
        ### Retrieve proper parameters
        
        elems_dic=self.get_dic(minlambda_select=minlambda_select)
        
        ### Select arrays 
        
        x=elems_dic['s_time']
        y=elems_dic[y_param]
        
        ### Convert raduians to degress
        if y_param=='fast':
            y=y*180/np.pi
            
        ### Change angle mode and lag_model if asked
        
        if y_param=='fast':
            if angle_mode=='azimuth':
                y=swm.trigo2azimuth(y)
        elif y_param=='lag':
            if lag_mode=='ms':
                y=y/sampling_rate*1000
            elif lag_mode=='s':
                y=y/sampling_rate
                
        ### Plot

        (ax,im,X,Y,Z,x_bins,x_diffs)=swm.plot_movehisto2d(x,y,
                x_label=x_label,y_label=y_label,title=title,ax=ax,
                **movehisto2d_kwargs)
        
        ##################################
        ### Plot time windows if asked ###
        ##################################
        
        if window_unit is not None:
            
            ### Modify seconds to proper unit
            
            if window_unit=='minute':
                x_diffs/=60
            elif window_unit=='hour':
                x_diffs/=3600
            elif window_unit=='day':
                x_diffs/=86400
            elif window_unit=='second':
                x_diffs=x_diffs
            else:
                raise ValueError('window unit must be either, day, hour, minute, second')
                
            window_label='Window size [%s]'%window_unit[0:3]
            
            ### Create new ax
            
            ax_win = ax.twinx()
            ax_win.set_yscale("log")
            
            ### Plot 
#            x_start=movehisto2d_kwargs.get('x_start',None)
#            x_end=movehisto2d_kwargs.get('x_end',None)
            
            lw_win=3
            ax_win.plot(x_bins, x_diffs, "k-",lw=lw_win,alpha=0.5)
            ax_win.plot(x_bins, x_diffs, "w-",lw=lw_win/3,alpha=0.5)
#            ax_win.set_xlim(swm.obspytime2matplotlib([x_start,x_end]))
            ### Cosmetic
            
            ax_win.set_ylim(bottom=window_bottom,top=window_top)
            ax_win.set_ylabel(window_label)

        
        ### Add vertical lines associated to start and end of eruption
        
        starteruption_time=UTCDateTime(2015,4,24,6) # Nooner and Chadwick 2016
        enderuption_time=UTCDateTime(2015,5,19)
        vline_times=[starteruption_time,enderuption_time]
        vline_times=swm.obspytime2matplotlib(vline_times)
             
        swm.plot_vlines(vline_times,ax,markercolor='w',markersize=15,
                        markeralpha=0.6,linecolor='w',linewidth=1.5)

        return ax
    
    def plot_obstime(self,day_width=10,facecolor='0.8'):
        ### Process
        time_array=self.get_array('s_time',level=1)
        
        datenum_array=obspytime2matplotlib(time_array)
        
        #datenum_array=[x.datetime for x in time_array]
        
        x_width=day_width
        x_start=datenum_array[0]
        x_end=datenum_array[-1]
        x_bins=np.arange(x_start,x_end,x_width)
        x_bins=np.append(x_bins,x_end)
        
        fig,ax=plt.subplots()
        ax.hist(datenum_array,x_bins,facecolor=facecolor,edgecolor='k')
        ax.xaxis_date()
        fig.autofmt_xdate()
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Number Obs.')
        fig.suptitle(','.join(self.station_list),y=0.95,weight='bold')
        ax.set_axisbelow(True)
        ax.xaxis.grid(lw=0.5,color='k',ls='--')
        
        return ax

    
    def plot_histolambda(self,ax=None,
                      alpha_bar=0.4,
                      bar_color=['r','b']):
        """
        Method made to plot the number of lambdas observations per pair for each station
        
        Input
        -----
            ax: plt.axes object
            alpha_bar: float: transparency level
            bar_color: list of str: color for each bar
            
        Output
        ------
            ax: plt.axes object
        """ 
        
        #### PROCESS
        ### Counter for MinLambdas
        
        counts=np.array([len(x.MinLambdas) for x in self.obs])
        station_list=np.array([x.station_code for x in self.obs])
        
        station_unique=np.unique(station_list)
        
        dict_stat={}
        
        for station in station_unique:
            elem_list=[]
            for i in [1,2]:
                elem_list.append(len(counts[(counts==i) & (station_list==station)]))
                
            dict_stat[station]=elem_list
        
        #### PLOT
        ### Prepare Plot
        
        if ax is None:
            fig,ax=plt.subplots()
        
        xticklabels=list(station_unique)
        x_bar=np.arange(1,len(station_unique)+1)
        
        ax.set_xlim([-0.5,len(station_unique)+1+0.5])
        ax.set_xticks(x_bar)
        ax.set_xticklabels(xticklabels)
        
        ### Start plot
        
        kk=-1
        for station in station_unique:
            kk+=1
            mm=-1
            width_bar=0.5/len(dict_stat[station])
            bar_left=kk+1-0.25
            for y_bar in dict_stat[station]:
                mm+=1
                bar_center=bar_left+width_bar/2
                bar_left+=width_bar
                ax.bar(bar_center,y_bar,width_bar,color=bar_color[mm],align='center',alpha=alpha_bar,label='%i'%(mm+1))
                if kk==0:
                    ax.legend(title=r"Number of $\lambda_{2}$ min.:")
                
        ax.set_ylabel('Number obs')
     
        ### Return
        return ax


    
    def plot_stations(self,**plot_stations_kwargs):
        
        ax=plot_stations(**plot_stations_kwargs)
        return ax

    def plot_datafile(self,data_file=['/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/caldera_smooth.ll',
                             '/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/axial_lavaflow_2015.ll'],
                  ax=None,color='k',lw=2,ls='-',ini_lon=-130.1,ini_lat=45.9):
    
        if ax is None:
            fig,ax=plt.subplots()
            
        for file in data_file:  
            data=read_datafile(file)
            lon_data,lat_data=data[:,0],data[:,1]
            
            x_data,y_data=gproj.ll2xy(lon_data,lat_data,ini_lon,ini_lat)
            
            ax.plot(x_data,y_data,color=color,lw=lw,ls=ls)
            ax.axis('equal')
            
        return ax


    def plot_events(self,ax=None,color='r',cmap=None,ini_lon=-130.1,ini_lat=45.9,vmin=None,vmax=None,
                    flag_rose=False,num_slice=4,theta_rot=0,r_edges=None,**kwargs_rose):
        """
        Plot all events present in the catalogs (No matter the station)
        """
    
        if ax is None:
            fig,ax=plt.subplots()
    
    
        ### Plot events
        
        lon_ev=np.array([x.event_lon for x in self.obs])
        lat_ev=np.array([x.event_lat for x in self.obs])
        z_ev=np.array([x.event_depth for x in self.obs])
        
        x_ev,y_ev=gproj.ll2xy(lon_ev,lat_ev,ini_lon,ini_lat)
    
        if cmap is not None:
            ax.scatter(x_ev,y_ev,s=0.05,c=z_ev,cmap=cmap,alpha=1,vmin=vmin,vmax=vmax)
        else:
            ax.plot(x_ev,y_ev,'o',mfc=color,mec='none',markersize=0.5)
            
        ### Plot rose on stations if asked
        
        if flag_rose:
            ini_lon=-130.1
            ini_lat=45.9
            station_dic=read_stationfile()
            
            for station in self.station_list:
                x_sta,y_sta=gproj.ll2xy(station_dic[station]['lon'],station_dic[station]['lat'],ini_lon,ini_lat)
                ax=plot_rose(ax=ax,x0=x_sta,y0=y_sta,
                             num_slice=num_slice,theta_rot=theta_rot,r_edges=r_edges,**kwargs_rose)
            
        ### Plot stations and caldera
        
        station_list=self.station_list
        self.plot_stations(station_list=station_list,ax=ax)
        self.plot_datafile(ax=ax)
        
        return ax
    
    def plot_polar_fastlag(self,x_width=2,y_width=4,y_start=-90,y_end=270,y_cycle=[-90,270],
                           ax=None,show_axis=True,cmap=plt.cm.get_cmap('jet'),label_mode='trigo',
                           lag_mode='sample',sampling_rate=200,
                           title=None,colorbar=True,lagticks_step=2,plot_mode='pcolormesh',
                           **movehisto2d_kwargs):
    
        """
        Method made to plot the polar histogram plot showing distribution of fast direciton and lags
        The function uses movehisto2d_bin to bin the data, the y_cycle is important to specify cyclicty of 
        fast directions
        
        Also, very important, fast directions are computed in radian using trigo convetion (CCW from x),
        but in most paper the convention is degree azimuth (CW from Norh)
        
        For the plots we always convert the radians to degree, so the input parametrers should be 
        specified in degrees not in radians
        x being the lag and y being the direction
        
        Inputs
        -----
            Same as for movehisto2d_bin
            label_mode: ['trigo','azimuth']: transform labels in between azimith or trigo convetion

        Outputs
        -------
            (ax_polar,cax): tuple of plt.axes
            
        TODO
        -----
            - Adapt fontsize to polar size
            
        """
        ### Checks
        

        if label_mode not in ['trigo','azimuth']:
            raise ValueError('label mode should be either trigo or azimuth')
       ### Retrieve proper parameters
        
        elems_dic=self.get_dic()
        
        ### Select arrays 
        
        x=elems_dic['lag']
        y=elems_dic['fast']*180/np.pi # in degrees
        
        ### Convert lag to ms or samples
       
        if lag_mode=='ms':
            x=x/sampling_rate*1000
        elif lag_mode=='s':
            x=x/sampling_rate

        ### Duplicate by adding +np.pi
         
#        x_left=x
#        y_left=y-np.pi
#        
#        x_left=x_left[(y_left>=-3*np.pi/4) & (y_left<-np.pi/2)] ## Add so that movehisto_2d bin works for 0
#        y_left=y_left[(y_left>=-3*np.pi/4) & (y_left<-np.pi/2)]
#        
#        x=np.concatenate((x_left,x,x))
#        y=np.concatenate((y_left,y,y+np.pi))
        
        x=np.hstack((x,x))
        y=np.hstack((y,y+180))
        
        #### Count
        
        (X,Y,Z,_,_)=movehisto2d_bin(x,y,x_width=x_width,y_width=y_width,
        y_start=y_start,y_end=y_end,y_cycle=y_cycle,
        **movehisto2d_kwargs)
        
        ### Plot
        
        ax_polar=ax
        if ax_polar is None:
            fig,ax_polar=plt.subplots(subplot_kw=dict(projection='polar'))
        
        ### When using polar the degrees have to be specfied in radians
        
        if plot_mode=='pcolormesh':
            ax_polar.pcolormesh(Y*np.pi/180,X,Z,cmap=cmap,rasterized=True)
        else:
            levels=np.linspace(0.5,np.nanmax(np.log10(Z)),15)
            imf=ax_polar.contourf(Y*np.pi/180,X,np.log10(Z),levels=levels,cmap=cmap)
            imc=ax_polar.contour(Y*np.pi/180,X,np.log10(Z),levels=levels,linewidths=0.5,colors='none')
            #for (cf,cc) in zip(imf.collections,imc.collections):
            #    cf.set_rasterized(True)
            #    cc.set_rasterized(True)
            # Since .collections zip is deprecated as of Python 3.8, 
            # access separately to avoid
            for cf in imf.collections:
                cf.set_rasterized(True)
            for cc in imc.collections:
                cc.set_rasterized(True)
            
#        fig,ax=plt.subplots()
#        ax.pcolormesh(X,Y,Z,cmap=cmap,rasterized=True)
 #       sys.exit()
        
        ### Cosmetics
        
        ax_polar.set_rlim([0,movehisto2d_kwargs.get('x_end',None)])
        ax_polar.set_rorigin(-20)
        ax_polar.set_rlabel_position(10)  # Move radial labels away from plotted line
        ax_polar.set_title(title,y=1.08,weight='bold')
        ax_polar.set_xlabel('$\Phi$ [°]') 
        ax_polar.grid(True,color='w',alpha=0.2)
        ax_polar.tick_params(axis='y', colors='white')
    
        ### Add colorbar if asked
        
        ax_polar.set_xticks(np.arange(0,360,30)*np.pi/180)
        
        if lagticks_step is not None:
            lag_start=0
            lag_end=movehisto2d_kwargs.get('x_end',None)
            lagticks=np.arange(lag_start,lag_end,lagticks_step)
            ax_polar.set_yticks(lagticks)
            

        ### Convert labels if asked
        
        if label_mode=='azimuth':
            trigo2azimuth_labels(ax_polar,axis='x')
        
        if not show_axis:
            ax_polar.tick_params(axis='both',labelbottom='off')
            ax_polar.set_title('')
#    
        if colorbar:
            cax = ggmt.get_cax(ax_polar,cax_width=0.03,pad=-5)    
            vmin=np.min(Z)
            vmax=np.max(Z)
            
            cmap = cmap
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                            norm=norm,
                                            orientation='vertical')
            
            cb.set_label('N Obs.')
            
            return (ax_polar,cax)
       
        else:
            return (ax_polar)
        
    def plot_polar_fastlag_multi(self,ax=None,num_ax=4,theta_rot=0,
                                 x_width=2,y_width=4,
                           show_axis=False,cmap=plt.cm.get_cmap('jet'),label_mode='trigo',
                           title=None,colorbar=False,lagticks=None,
                           **movehisto2d_kwargs):
#    def plot_polar_fastlag_multi(self,ax_main=None,num_ax=4,theta_rot=0,
#                         x_start=0,x_end=None,x_over=0,y_over=0,x_width=2,y_width=np.pi/20,smooth=False,**fastlagargs):
#        
        """
        Method made to plot multiple fast lag 
        """
            
        ### Get mutliple axes around center
        
        if title is None:
            title=','.join(self.station_list)
        (ax_main,ax_insets,theta_edges,r_edges,r_inner,r_width)=get_ax_polarinsets(ax_main=ax,num_ax=num_ax,theta_rot=theta_rot,title=title)
        section_theta_deg=(theta_edges[1]-theta_edges[0])*180/np.pi
        
        #### Find edge suitable for plotting the upper left polar plot (histo)
        
        ind_theta_inset=np.argmin(np.abs(theta_edges-135*np.pi/180))
        theta_inset=theta_edges[ind_theta_inset]
        
        r_center_inset=(r_edges+1)/2
        r_width_inset=(r_edges-1)*0.9
        (x_inset,y_inset)=pol2cart(r_center_inset, theta_inset)
        
        
        #### Select events based on baz_trigo
        
        for kk,ax_inset in enumerate(ax_insets):
            Cat_select=self.select(obs_baz_trigo=[theta_edges[kk],theta_edges[kk+1]])
            ax_inset.get_yaxis().set_visible(True)
            ax_inset.get_xaxis().set_visible(True)
            Cat_select.plot_polar_fastlag(ax=ax_inset,colorbar=colorbar,
                                          show_axis=show_axis,x_width=x_width,y_width=y_width,
                                          cmap=cmap,label_mode=label_mode,lagticks=lagticks,
                                          **movehisto2d_kwargs)
            ax_inset.set_xlabel('')

            
        #### Plot ax_middle
            
        ax_middle=get_ax_inset(ax_main,0,0,0.7*(r_inner*2),height=None,projection='polar',alpha=1,visible_axis=True)
        
        
        self.plot_polar_fastlag(ax=ax_middle,colorbar=colorbar,
                                          show_axis=show_axis,x_width=x_width,y_width=y_width,
                                          cmap=cmap,label_mode=label_mode,lagticks=lagticks,
                                          **movehisto2d_kwargs)
        
        ax_middle.set_xlabel('')
        ax_middle.tick_params(axis='x', which='major', pad=0)
        
        #### Plot histo on inset
        
        ax_top=get_ax_inset(ax_main,x_inset,y_inset,r_width_inset,height=None,projection='polar',alpha=1,visible_axis=False)
        trigo_angles=[x.baz_trigo for x in self.obs]
        sws.plot_polar_histo(trigo_angles,weights=None,width_bin_deg=section_theta_deg,ax_polar=ax_top,facecolor=[0,0,0.5],edgecolor='w',alpha=1,text=True)
        plt.setp(ax_top.spines.values(), color='w') # White color for polar frame
        ax_top.set_facecolor('w') # white background
        max_val=ax_top.get_ylim()[1]
        for theta in theta_edges:
            ax_top.plot([theta,theta],[0,max_val],'-k',lw=0.5)
    
        ### return
        
        return (ax_main,ax_insets,ax_middle,ax_top)


        
    def plot_polar_events(self,ax=None,title=None,ylim=None):
        """
        Method made to plot events and distribution in a polat plot view
        """
        
        ### Check if ax exist
        if ax is None:
            fig,ax_1=plt.subplots(subplot_kw=dict(projection='polar'))
            ax_2 = ax_1.figure.add_axes(ax_1.get_position(), projection='polar',frameon=False,
                                     theta_direction=ax_1.get_theta_direction(),
                                     theta_offset=ax_1.get_theta_offset())
        else:
            ax_1,ax_2=ax
            
        if title is None:
            title=','.join(self.station_list)
            
        ### Retrieve values
        baz_trigo=[x.baz_trigo for x in self.obs]
        epi_dist=[x.epi_dist for x in self.obs]
        
        #### Parameters
        
        bar_label_angle=-67.5
        ev_label_angle=22.5
        bar_color='b'
        label_offset=10
    
        ### First plot
        #ax_polar.figure.canvas.draw()
        ax_1=sws.plot_polar_histo(baz_trigo,weights=None,width_bin_deg=10,ax_polar=ax_1,facecolor=bar_color,edgecolor='none',alpha=0.2)
        ax_2.plot(baz_trigo,epi_dist,'ok',ms=1.5,mec='none',alpha=0.1)

        
        #### Cosmectic bar
        ax_1.spines["start"].set_visible(False)
        ax_1.grid('off')
        ax_1.get_xaxis().set_visible(False)
        ax_1.set_rlabel_position(bar_label_angle)
        ax_1.tick_params(axis='y', colors=bar_color)
        
        y_bar_min,y_bar_max=ax_1.get_ylim()
        y_bar_diff=np.diff(ax_1.get_ylim())[0]
        ax_1.set_rorigin(-(y_bar_min+0.1*y_bar_diff))
        ax_1.text(bar_label_angle*np.pi/180, y_bar_max*(1+label_offset/100),'N obs.', va="center", ha='left',rotation=90+bar_label_angle,color=bar_color)
        
        
        #### Cosmectic events
        if ylim is not None:
            ax_2.set_ylim([0,ylim])
        
        ax_2.set_rorigin(0)
        ax_2.patch.set_alpha(0)    
        y_ev_min,y_ev_max=ax_2.get_ylim()
        ax_2.text(ev_label_angle*np.pi/180, y_ev_max*(1+label_offset/100),'Dist', va="center", ha='left',rotation=-(90-ev_label_angle))
        
        ax_2.set_title(title,y=1.08,weight='bold')
        ax_2.set_xlabel('Back Azimuth') 
        
        return (ax_1,ax_2)


### Other functions
        
def zscores(data,thres=1,mode='original',flag_plot=False):
    """
    Function made to remove outliers based on the zscore value,
    also known as the standard value
    
    http://colingorrie.github.io/outlier-detection.html
    
    It will not remove any outliers if std==0
    
    Input
    -----
        data: np.array: 1D numpy array containing samples
        thres: float: threshold to remove outliers, 1 will remove ~ 30%
        of the data
        mode: str: 'original' or 'modified'
        
    Output
    ------
        data:np.array: the data with outliers removed
        
    """
      
    data=np.asarray(data)
    
    if mode=='original':
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val==0:
            z_scores=data*0
        else:
            z_scores = (data - mean_val) / std_val
        
    elif mode=='modified':
        median_val=np.median(data)
        MAD=np.median(np.abs(data - median_val))
        z_scores =  0.6745*(data - median_val) / MAD
        if MAD==0:
            z_scores=data*0
        else:
            z_scores =  0.6745*(data - median_val) / MAD
    else:
        raise ValueError('mode is either original or modified')
        
        
    if flag_plot:
        fig,ax=plt.subplots()
        x_bins=np.linspace(np.min(data),np.max(data),50)
        
        ax.hist(data,bins=x_bins,color='r')
        ax.hist(data[np.abs(z_scores) < thres],bins=x_bins,color='blue',alpha=0.5)
        
    return data[np.abs(z_scores) < thres]
    
def xyzd2mesh(x,y,z,d,
              x_start=None,x_end=None,
              y_start=None,y_end=None,
              z_start=None,z_end=None,
              x_step=None,y_step=None,z_step=None,
              dis_lim=0.3,num_lim=100,verbose=False):
    """
    Function made to return some statistics out of x,y,z,d array values
    Returning objects are meshes
    
    Method
    ------
    For each point of the builded mesh we select all elements that are located
    at a distance < dis lim. If there is more than num_lim elements we select the
    the closest num_lim elements to compute the statistics. Some bins will have less
    than num_lim elements of course but none of them will have more than num_lim
    
    Input
    -----
    x,y,z,d: np.array: x,y,z locations and d data
    ?_start,?_end: float: limits of the mesh
    ?_step: float: step seprating bins of the mesh
    dis_lim: float: maximum distance to look for events around bin center
    num_lim: float: maximum number of elements to be selected
    
    Output
    ------
    ?_MESH: np.arrays: 3D arrays containing requsted values, MEAN, STD...
    MEANZ is the mean for each bin but by removing outliers (above 2sigma)
    returns a dictionnary with proper mesh names
    
    """

    ### Check
    
    x_start=np.min(x) if x_start is None else x_start
    y_start=np.min(y) if y_start is None else y_start
    z_start=np.min(z) if z_start is None else z_start
    x_end=np.max(x) if x_end is None else x_end
    y_end=np.max(y) if y_end is None else y_end
    z_end=np.max(z) if z_end is None else z_end
    
    def_num=20 # number of elements in the mesh default
    x_step=(x_end-x_start)/def_num if x_step is None else x_step
    y_step=(y_end-y_start)/def_num if y_step is None else y_step
    z_step=(z_end-z_start)/def_num if z_step is None else z_step
    
    ### Build mesh
    
    x_mesh=np.linspace(x_start,x_end,int(round((x_end-x_start)/x_step))+1)
    y_mesh=np.linspace(y_start,y_end,int(round((y_end-y_start)/y_step))+1)
    z_mesh=np.linspace(z_start,z_end,int(round((z_end-z_start)/z_step))+1)
    
    X_mesh,Y_mesh,Z_mesh=np.meshgrid(x_mesh,y_mesh,z_mesh)
    
    ### Initialize
    
    MEAN_mesh=np.zeros_like(X_mesh)
    MEANZ_mesh=np.zeros_like(X_mesh)
    MEDIAN_mesh=np.zeros_like(X_mesh)
    STD_mesh=np.zeros_like(X_mesh)
    COUNT_mesh=np.zeros_like(X_mesh)

    ### START PROCESS
    
    for ia in range(X_mesh.shape[0]):
        if verbose:
            print('Processing row %i'%ia)
        for ib in range(X_mesh.shape[1]):
            for ic in range(X_mesh.shape[2]):
                x_bin=X_mesh[ia,ib,ic]
                y_bin=Y_mesh[ia,ib,ic]
                z_bin=Z_mesh[ia,ib,ic]
                
                # Compute distance
                
                dis=np.sqrt((x-x_bin)**2+(y-y_bin)**2+(z-z_bin)**2)
           
                ind=np.arange(len(dis))
                
                ### Select based on distance
                ind_dis=ind[dis<=dis_lim]
                dis_dis=dis[dis<=dis_lim]
                
                ### Sort
                
                ind_sort=np.argsort(dis_dis,kind='quicksort')
                ind_sel=ind_dis[ind_sort]
                
                ### Select max num_lim
                ind_sel=ind_sel[0:num_lim]
                d_sel=d[ind_sel]
                
                ### fill Meshes
                
                mean_val=np.mean(d_sel)
                std_val=np.std(d_sel)
                median_val=np.median(d_sel)
                
                ## Z score, compute mean but without oultiers
                d_zscore=zscores(d_sel,thres=1.5,mode='modified')
                meanz_val=np.mean(d_zscore)
                
                COUNT_mesh[ia,ib,ic]=len(d_sel)
                MEDIAN_mesh[ia,ib,ic]=median_val
                MEAN_mesh[ia,ib,ic]=mean_val
                STD_mesh[ia,ib,ic]=std_val
                MEANZ_mesh[ia,ib,ic]=meanz_val
                
    return {'X_mesh':X_mesh,'Y_mesh':Y_mesh,'Z_mesh':Z_mesh,
            'COUNT_mesh':COUNT_mesh,'MEDIAN_mesh':MEDIAN_mesh,
            'STD_mesh':STD_mesh,'MEAN_mesh':MEAN_mesh,
            'MEANZ_mesh':MEANZ_mesh}


    
def read_datafile(data_file):
    data=np.loadtxt(data_file)
    if data.shape[1]<2:
        raise ValueError('data file must have at least 2 columns to be loaded')
    
    return data

def read_stationfile(station_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/STATIONS/stations_axial.llz',
                     ini_lon=None,ini_lat=None):
    """
    Read station file having lon,lat,z,name as columns
    """
    
    print('Make sure coordinates are given in lon and lat')

    with open(station_file,'rt') as fic:
        lines=fic.readlines()
        station_dic={}
        for line in lines:
            lon,lat,z=[float(x) for x in line.split()[0:3]]

            station=line.split()[-1]
            if ini_lon is not None:
                x,y=gproj.ll2xy(lon,lat,ini_lon,ini_lat)
                station_dic[station]={'x':x,'y':y,'z':-z}
            else:
                station_dic[station]={'lon':lon,'lat':lat,'z':-z}
        
    return station_dic


def get_funckeys(f):

    return inspect.signature(f).parameters.keys()

def read_pickle(pickle_files):
    """
    Function made to read SWSobs class stored into pickle files
    the pickle files can only contain SWSobs class or a list of SWSobs class
    
    Input
    -----
        pickle_files: str,list: path to pickle files
    
    Output
    ------
        Cat: SWScat class: Catalog containing all the SWSobs
    """
    pickle_files=sws.tolist(pickle_files)

    obs=[]
    kk=0
    for pickle_file in pickle_files:
        kk+=1
        x=sws.tolist(pickle.load(open(pickle_file,'rb')))
        y_list=[]
        for element in x: # update with new methods
            y=sws.SWSobs()
            y.__dict__=element.__dict__.copy()
            y_list.append(y)
            
        obs.extend(y_list)

    Cat=SWScat(obs=obs)
    Cat.update()
    Cat.compute_ini() # Compute elements that are not in intial class (baz_trigo,hyp_dist..)
    
    return Cat


def cat_pickles(input_dir,output_dir,station_list=None):
    
    if station_list is None:
        pickle_files=glob.glob(input_dir+'/*.pickle')
        station_list=[x.split('/')[-1].split('_')[0] for x in pickle_files]
        station_list=list(set(station_list))
    
    station_list=sws.tolist(station_list)
    
    for station in station_list:
        pickle_files=glob.glob(input_dir+'/'+station+'_*_*.pickle')
        pickle_files.sort()
        print('Processing station %s\n'%(station))
        Cat=read_pickle(pickle_files)
        Cat.write_pickle(output_dir+'/'+station+'.cat.pickle')
        
            
def timestamp2matplotlib(values):

    values=sws.tolist(values)
    
    values=[mdates.date2num(UTCDateTime(x).datetime) for x in values]
    #values = list(mdates.date2num(values))
    if len(values)==1:
        values=values[0]
    return values

def timestamp2str(timestamp,time_format='%Y%m%dT%H:%M:%S'):
    return UTCDateTime(timestamp).strftime(time_format)

def obspytime2matplotlib(values):
    values=sws.tolist(values)
    out=[mdates.date2num(x.datetime) for x in values]
    return out

def zoom_1darray(in_array,factor):
    """
    Function made to multiply the array size by a factor 
    Works like scipy.zoom but actually performs interpolation inbetween points
    
    Inputs
    ------
        in_array: np.array: data to rescale (1xD)
        factor: int: zooming factor
    
    Outputs
    -------
        out_array: np.array: rescaled array (1x(NxD))
    """
    
    in_array=np.asarray(in_array)
    N=len(in_array)
    x = np.arange(N)
    x_new=np.linspace(x[0],x[-1],factor*N)
    out_array = np.interp(x_new, x, in_array) 
    return out_array

def get_ax_inset(ax_main,x_center,y_center,width,height=None,projection='rectilinear',alpha=1,visible_axis=True):
    """
    Function made to return the axes inset object
    Please use fig.canvas.draw() before calling the ax so the get_position() is right
    Please see stackoverflow inset_axes
    
    Inputs
    ------
        ax_main: plt.axes object: Main axes object to put the inset
        x_center,y_center: float: position of the inset
        width,height: float: must given in ax_main coordinates values
        projection: str: must be one of the get_projection_names or None
        alpha: float: bg alpha
        
    Outputs
    -------
        ax_inset: plt.axes object
    """
    
    ### Check

    axes_class=get_projection_class(projection)

    height=width if height is None else height
    
    new_width,_=(ax_main.transData.transform([width,0])-ax_main.transData.transform([0,0]))/ax_main.figure.dpi
    _,new_height=(ax_main.transData.transform([0,height])-ax_main.transData.transform([0,0]))/ax_main.figure.dpi

    ### Process and ensure height and width in data coordinates
    ax_val= inset_axes(ax_main, width=new_width, height=new_height, loc=10, 
                       bbox_to_anchor=(x_center,y_center),
                       bbox_transform=ax_main.transData, 
                       borderpad=0.0, axes_class=axes_class)
    
    #### Set background
    ax_val.patch.set_alpha(alpha)
    
    ### Axis visible
    if not visible_axis:
        ax_val.get_yaxis().set_visible(False)
        ax_val.get_xaxis().set_visible(False)

    #### Return
    return ax_val

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)



def get_ax_polarinsets(ax_main=None,num_ax=4,theta_rot=0,r_shift=1,x0=0,y0=0,r_width=0.8,title=''):
    """
    Function made to plot multiple polar inset axes
    
    Inputs
    -----
        ax_main: plt.axes: parent axes to put the inset
        num_ax: int: number of polar plots around center (N)
        theta_rot: float : radian for rotation
        r_shift: float: shift of the polar plot according to center
        x0,y0: float: center of the plot
        
    Outputs
    -------
        (ax_main,ax_insets,theta_edges)
        theta_edges: list: theta angles delimiting each sections (N+1)
        ax_insets: list: list plt.axes objects
        ax_main: plt.axes object: main obj
        
    """
    #### Define edges
    
    theta_edges=np.linspace(0,2*np.pi,num_ax+1)+theta_rot
    theta_centers=theta_edges[:-1]+np.diff(theta_edges)/2
    theta_diff=theta_centers[1]-theta_centers[0]
    r_width_max=(theta_diff*r_shift)*0.8 # width of the polar plots
    
    if r_width>r_width_max:
        r_width=r_width_max
    
    r_edges=r_shift+r_width/2
    
    xy_centers=[pol2cart(r_shift, theta_center) for theta_center in theta_centers]
    xy_edges=[pol2cart(r_edges, theta_edge) for theta_edge in theta_edges[:-1]]
    
    xy_centers=[(x+x0,y+y0) for x,y in xy_centers]
    xy_edges=[(x+x0,y+y0) for x,y in xy_edges]
    
    #### Define ax_main prop if not defined
    
    if ax_main is None:
        fig,ax_main=plt.subplots(figsize=(8, 8))
    ax_main.axis('equal')
    ax_main.set_xlim([-r_edges,r_edges])
    ax_main.set_ylim([-r_edges,r_edges])

    r_inner=r_shift-r_width/2
    ax_main.axis('off')
    
    ### Call insets_ax
    
    ax_insets=[get_ax_inset(ax_main,x_center,y_center,width=r_width,projection='polar',
                            visible_axis=False) for (x_center,y_center) in xy_centers ]
    
    ### Draw edges
    
    for x_edge,y_edge in xy_edges:
        ax_main.plot([x0,x_edge],[y0,y_edge],'-k',lw=0.5)
        
    #### plot title
    
    ax_main.text(x0,1.05*(y0+r_edges),title,fontsize=12,ha='center',weight='bold')
        
    return (ax_main,ax_insets,theta_edges,r_edges,r_inner,r_width)



#def movehisto2d_bin(x,y,x_width=None,y_width=None,
#                x_start=None,y_start=None,x_end=None,y_end=None,
#                x_over=0.9,y_over=0.9,
#                norm_y=False,smooth=False,gaussian_x_per=0.5,gaussian_y_per=0.5):
#    """
#    Function made to bin but using a moving window in both directionsn, this ensure better
#    consistency between neighbor bins.
#    The histogram can also work when data is an obspy.UTCDateTime array, then the x_start and x_end must
#    be given in UTCDateTime as well and the x_width should be given in days
#    
#    Inputs
#    ------
#        x,y: np.array: arrays containing the data to apply histogram on (x can be UTCDateTime)
#        x_width,y_width: float: width of the bins (in days for UTCDateTime)
#        [x,y]_[start,end]: float: start and end for histogram edges
#        [x,y]_over: float in [0,1]: overlap for windows [1 = full overlap]
#        norm_y: Boolean: True to normalize by the maximum in each column
#        smooth: Boolean: True for smoothin (Gaussian Filter)
#        gaussian_[x,y]_per: float in [0,100]: width percentage for smoothing (100= filter size equal to data range)
#
#        
#    Ouputs
#    ------
#         [X,Y,Z]: np.array: 2D arrays containing bining
#    
#    Comments:
#    ---------
#        To be added to the general functions (plot module?)
#        x_start and x_end needs to be modified for samples plots
#    """
#
#    #### Start process
#
#    x=np.asarray(x)
#    y=np.asarray(y)
#    
#    x_pixels=1000 # Number of elements in Z for imshow
#    y_pixels=1000
#    
#
#    ### Check
#    
#    gaussian_x=gaussian_x_per*x_pixels/100 # Convert percentage to number of pixels
#    gaussian_y=gaussian_y_per*y_pixels/100
#    
#    x_start=np.min(x) if x_start is None else x_start
#    y_start=np.min(y) if y_start is None else y_start
#    x_end=np.max(x) if x_end is None else x_end
#    y_end=np.max(y) if y_end is None else y_end
#    
#    x_width=(x_end-x_start)/20 if x_width is None else x_width
#    y_width=(y_end-y_start)/20 if y_width is None else y_width
#    
#    ### Define bin edges 
#    
#    y_step=(1-y_over)*y_width
#    x_step=(1-x_over)*x_width
#    
#    
#    (x_mesh,x_step)=smart_arange(x_start,x_end,x_step)
#    (y_mesh,y_step)=smart_arange(y_start,y_end,y_step)
#    
#    x_width=x_step/(1-x_over)
#    y_width=y_step/(1-y_over)
#    
#    print('New x_width and y_width: %f %f'%(x_width,y_width))
#    
#        ### For x
#    
#    x_rights=x_mesh+x_step/2
#    x_lefts=x_rights-x_width
#    
#        ### For y
#    
#    y_rights=y_mesh+y_step/2
#    y_lefts=y_rights-y_width
#    
#    ### Define meshes
#    
#    X,Y=np.meshgrid(x_mesh,y_mesh)
#    Z=np.zeros_like(X)
#        
#    ### Start Counting
#    
#    k_x=-1
#    for x_left,x_right in zip(x_lefts,x_rights):
#        k_x+=1
#        k_y=-1
#        y_select=y[(x>x_left) & (x<=x_right)] # select data 
#        
#        for y_left,y_right in zip(y_lefts,y_rights):
#            k_y+=1
#            counter=len(y_select[(y_select>y_left) & (y_select<=y_right)]) # Count numbers of elements  
#            Z[k_y,k_x]=counter # store
#        
#    #### Zoom and filter
#
#    if smooth:
#    
#        x_factor=int(x_pixels/Z.shape[1])
#        y_factor=int(y_pixels/Z.shape[0])
#        
#        x_factor=int(1) if x_factor<1 else x_factor
#        y_factor=int(1) if y_factor<1 else y_factor
#
#        Z_smooth=zoom(Z,[y_factor,x_factor],order=1) # row, column
#        X_smooth=zoom(X,[y_factor,x_factor],order=1) # row, column
#        Y_smooth=zoom(Y,[y_factor,x_factor],order=1) # row, column
#        
#        Z_smooth=gaussian_filter(Z_smooth,[gaussian_y,gaussian_x])
#        
#        ### Redefine x,y sizes
#        
#        x_mesh=zoom_1darray(x_mesh,x_factor)
#        y_mesh=zoom_1darray(y_mesh,y_factor)
#        
#        Z=Z_smooth
#        X=X_smooth
#        Y=Y_smooth
#    
#    #### Normalize by max
#    
#    if norm_y:
#        
#        max_y=np.max(Z,axis=0)[None,:]
#        max_y[max_y<=0]=1
#        Z=Z/max_y
#        
#    return (X,Y,Z)
    
def movehisto2d_bin(x,y,x_width=None,y_width=None,
                    x_mode='window',
                x_start=None,y_start=None,x_end=None,y_end=None,
                x_over=0.9,y_over=0.9,
                flag_y_norm=False,y_cycle=None,
                flag_resample=False,dx_resample=None,dy_resample=None,
                flag_filter=False,x_filter_per=0.5,y_filter_per=0.5,filter_mode='nearest'):
    """
    2019-05-16
    Function made to bin but using a moving window in both directions, this ensure better
    consistency between neighbor bins. 
    Two options for defining intervals in the x direction are possible.
    x_mode=['window','sample']. In 'window' mode the x bins are defined every x_width 
    (classical moving window). In 'sample' mode, the x bins are defined evrey x_width samples
    X_bins are centered, i.e. for each bin we look in the interval [x-x_width/2,x+x_width/2]
    
    Inputs
    ------
        x,y: np.array: arrays containing the data to apply histogram on
        x_width,y_width: float: width of the bins in x units or in number of samples
        [x,y]_[start,end]: float: start and end for histogram edges
        [x,y]_over: float in [0,1]: overlap for windows [1 = full overlap]
        flag_y_norm: bool: Should the ys be normalized by the max
        flag_resample: bool: Apply resampling (using scipy.griddata)
        dx_resample: float: resample every dx_resample
        flag_filter: bool: Apply gaussian filtering (using scipy.ndimage.gaussian_filter)
        x_filter_per: float: percentage of the range to be used as window size for the gaussian filter
            the bigger, the smoother
        filter_mode: str or sequence: define how the gaussian shoud behave 'nearest' or 'wrap' for cyclic
            can be ['nearest','wrap'] for different behavior in the two axis
        
    Ouputs
    ------
         [X,Y,Z]: np.array: 2D arrays containing bining
    
    Comment
    ------
        If you want to pcolormesh use sws_methods.XY2XY_mesh to increase the shape of X and Y, so that bin
        are centered
    """

    #### Check

    x=np.asarray(x)
    y=np.asarray(y)
    
    y=y[np.argsort(x)]
    x=x[np.argsort(x)]
    
    
    x_start=np.min(x) if x_start is None else x_start
    y_start=np.min(y) if y_start is None else y_start
    x_end=np.max(x) if x_end is None else x_end
    y_end=np.max(y) if y_end is None else y_end
    
    dx_resample=(x_end-x_start)/1000 if dx_resample is None else dx_resample
    dy_resample=(y_end-y_start)/1000 if dy_resample is None else dy_resample
    
    if x_mode=='sample':
        x_width=int(len(x)/100) if x_width is None else int(x_width) # Every n samples
    elif x_mode=='window':
        x_width=(x_end-x_start)/20 if x_width is None else x_width # Window size
    y_width=(y_end-y_start)/20 if y_width is None else y_width
    
    ### Define binining for X
    
    if x_mode=='sample':
        x_step=int(round((1-x_over)*x_width))
        if x_step==0:
            x_step=1
        ind_start=np.argmin(np.abs(x-x_start))-1
        ind_end=np.argmin(np.abs(x-x_end))+1
        #x_ind_bins=np.arange(0,len(x)-1,x_step)
#        x_ind_lefts=x_ind_bins-int(x_width/2)
#        x_ind_lefts[x_ind_lefts<=0]=0
#        x_ind_rights=x_ind_bins+int(x_width/2)
#        x_ind_rights[x_ind_rights>=len(x)-1]=len(x)-1
        x_ind_bins=np.arange(ind_start,ind_end-1,x_step)
        x_ind_lefts=x_ind_bins-int(x_width/2)
        x_ind_lefts[x_ind_lefts<=ind_start]=ind_start
        x_ind_rights=x_ind_bins+int(x_width/2)
        x_ind_rights[x_ind_rights>=ind_end-1]=ind_end-1
        
        x_lefts=x[x_ind_lefts]
        x_rights=x[x_ind_rights]
        x_bins=x[x_ind_bins]
        x_diffs=x_rights-x_lefts # to be used in plot_movehisto

    elif x_mode=='window':
        x_step=(1-x_over)*x_width
        #(x_bins,x_step)=swm.smart_arange(x_start,x_end,x_step)
        #x_width=x_step/(1-x_over)
        #x_rights=x_bins+x_width/2
        #x_lefts=x_bins-x_width/2

        first_center = x_start + x_width / 2
        last_center = x_end - x_width / 2
        (x_bins,x_step)=swm.smart_arange(first_center, last_center, x_step)
        x_width=x_step/(1-x_over)
        x_rights=x_bins+x_width/2
        x_lefts=x_bins-x_width/2

        x_diffs=x_rights-x_lefts
        
    ### Define binining for Y (is window by default)
    
    y_step=(1-y_over)*y_width
    (y_bins,y_step)=swm.smart_arange(y_start,y_end,y_step)
    y_width=y_step/(1-y_over)
    y_rights=y_bins+y_width/2
    y_lefts=y_bins-y_width/2

    ### Define meshes
    
    X,Y=np.meshgrid(x_bins,y_bins)
    Z=np.zeros_like(X)
        
    ##################
    ### Start Counting
    
    k_x=-1
    for x_left,x_right in zip(x_lefts,x_rights):
        k_x+=1
        k_y=-1
        y_select=y[(x>=x_left) & (x<x_right)] # select data along x
        
        ### Extend data if cyclic
            
        if y_cycle is not None:
            y_select=swm.extend_periodic_array(y_select,y_cycle,perc_ext=20)
        
        for y_left,y_right in zip(y_lefts,y_rights):
            k_y+=1
            counter=len(y_select[(y_select>=y_left) & (y_select<=y_right)]) # Count numbers of elements  
            Z[k_y,k_x]=counter # store
        
        
    ###################
    ### Resample
    
    if flag_resample:
        (y_resample,dy_resample)=swm.smart_arange(y_start,y_end,dy_resample)
        (x_resample,dx_resample)=swm.smart_arange(x_start,x_end,dx_resample)
        #print(dy_resample,dx_resample)
        X_resample,Y_resample=np.meshgrid(x_resample,y_resample)
        Z_resample=scipy.interpolate.griddata((X.ravel(),Y.ravel()), Z.ravel(), (X_resample, Y_resample),
                                              method='nearest',
                                              fill_value=0)
        ### fill_value is important in order to avoid nans
        Z=Z_resample
        
        #### Filter
    
        if flag_filter:
            
            gaussian_x=x_filter_per*X_resample.shape[0]/100
            gaussian_y=y_filter_per*X_resample.shape[1]/100
            #print(gaussian_x,gaussian_y)
            Z_gaussian=scipy.ndimage.gaussian_filter(Z,[gaussian_y,gaussian_x],mode=filter_mode)
            
            Z=Z_gaussian
            
        X=X_resample
        Y=Y_resample
        
    #### Normalize by max
    
    if flag_y_norm:
        
        max_y=np.max(Z,axis=0)[None,:]
        max_y[max_y<=0]=1
        Z=Z/max_y

    ### Return
    
    return (X,Y,Z,x_bins,x_diffs)


def plot_movehisto2d(x,y,
                x_label='X',y_label='Y',title='',ax=None,vmax=None,
                **movehisto2d_kwargs):


    """
    Function made to plot an histo2d but using a moving window in both directionsn, this ensure better
    consistency between neighbor bins. 
    The histogram can also work when data is an obspy.UTCDateTime array, then the x_start and x_end must
    be given in UTCDateTime as well and the x_width should be given in seconds.
    UTCDatetime are converted to timestamps (seconds since 1970)
    
    Inputs
    ------
        x,y: np.array: arrays containing the data to apply histogram on (x can be UTCDateTime)
        x_width,y_width: float: width of the bins (in seconds for UTCDateTime)
        [x,y]_[start,end]: float: start and end for histogram edges
        [x,y]_over: float in [0,1]: overlap for windows [1 = full overlap]
        norm_y: Boolean: True to normalize by the maximum in each column
        smooth: Boolean: True for smoothin (Gaussian Filter)
        gaussian_[x,y]_per: float in [0,100]: width percentage for smoothing (100= filter size equal to data range)
        [x,y]_label: str
        text_list: list,str: titles to be added to the right corner of the figure
        show_counts: bool: If True it will add subplots to shown counts
        
    Ouputs
    ------
        ax_list: plt.axes: object associated to the 4 plots (top,text,mesh,right)
        X,Y,Z: meshes :
        x_bins,x_diffs : x_diffs is in seconds, whereas x_bins in matplotlib time
            x_diffs is the time resolution (i.e. the size of the bins, should be constant
            if windows is used)
    
    Comments:
    ---------
        To be added to the general functions (plot module?)
        x_start and x_end needs to be modified for samples plots
        
    UsedIn
    ------
        SWSCat.plot_movehisto2d_time
    """
    
    ### Check if is made of UTCDateTimes
    
    time_flag=False
    if isinstance(x[0],UTCDateTime):
        time_flag=True
        print('X is in UTCDateTime')
        if movehisto2d_kwargs.get('x_mode','window')=='window':
            print('Remember width should be given in seconds, otherwise memory error')
    
    ### Modify x_start and x_end, and x if x is time and convert to timestamps
    x_start=movehisto2d_kwargs.get('x_start',None)
    x_end=movehisto2d_kwargs.get('x_end',None)
    y_start=movehisto2d_kwargs.get('y_start',None)
    y_end=movehisto2d_kwargs.get('y_end',None)
    
    if time_flag:
        if (x_start is not None) & (not isinstance(x_start,UTCDateTime)):
            raise ValueError('x_start must be given in obspy.UTCDateTime')
        if (x_end is not None) & (not isinstance(x_end,UTCDateTime)):
            raise ValueError('x_end must be given in obspy.UTCDateTime')
        x=np.array([value.timestamp for value in x]) # (seconds since 1970-01-01T00:00:00)
        x_start=x_start.timestamp if x_start is not None else None
        x_end=x_end.timestamp if x_end is not None else None
        movehisto2d_kwargs['x_start']=x_start
        movehisto2d_kwargs['x_end']=x_end

    ### Bin the data
    
    (X,Y,Z,x_bins,x_diffs)=swm.movehisto2d_bin(x,y,**movehisto2d_kwargs)
    
    # Added to fix edge artifacts issues - bins on edges have less data, highly sensitive to outliers
       # Drop first and last x-bins (edge time windows) before plotting
    if X.shape[1] > 2:          # only if we have at least 3 columns
        X = X[:, 1:-1]
        Y = Y[:, 1:-1]
        Z = Z[:, 1:-1]
        x_bins = x_bins[1:-1]
        x_diffs = x_diffs[1:-1]
        
    ########################
    #### Start plotting ####

    #### Grid spec
    
    bottom=0.15 if time_flag else 0.1
    
    ### Checks
    
    if ax is None:
        fig,ax = plt.subplots(gridspec_kw={'bottom':bottom,'left':0.15})
        
    if time_flag:
        X=np.array(swm.timestamp2matplotlib(X.ravel())).reshape(X.shape) # transform for plotting
        x_bins=swm.timestamp2matplotlib(x_bins)
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=30 ,ha='right')
        ax.set_xlim(swm.timestamp2matplotlib([x_start,x_end]))
    else:
        ax.set_xlim([x_start,x_end])
       
    (Xm,Ym)=swm.XY2XY_pcolormesh(X,Y) # To ensure Pcolormesh will be ceneterd on bins

    #im=ax.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'),rasterized=True)
    im=ax.pcolormesh(Xm,Ym,Z,cmap=plt.cm.get_cmap('jet'),rasterized=True,vmax=vmax)
    
    ax.set_ylim([y_start,y_end])
    
    ax.set_aspect('auto')
    if time_flag:
        ax.xaxis_date()
            
    ### Cosmetic
    
    ax.set_ylabel(y_label) 
    if not time_flag:
        ax.set_xlabel(x_label) 
   
    ###### Return
    
    return (ax,im,X,Y,Z,x_bins,x_diffs)


def plot_rose(ax=None,x0=0,y0=0,num_slice=4,theta_rot=0,r_edges=None,color='k',ls='-',lw=1):
    """
    Function made to plot a rose on the plot centered on x0,y0
    
    Inputs
    -----
        ax: plt.axes
        x0,y0: float: center
        num_slice: int
        theta_rot: float(radians): rotaton to apply
        r_edges: float: length of the arms
        color,ls,lw: plt.axes prop
        
    Outputs
    -------
        ax
        
    TODO
    ----
        - plot circle around edges if asked
    """

    if ax is None:
        fig,ax=plt.subplots()
        
    ### Compute angles
    theta_edges=np.linspace(0,2*np.pi,num_slice+1)+theta_rot
    
        ### Get length
    axis_old=ax.axis()
    
    if r_edges is None:
        r_edges=np.max([np.diff(ax.get_xlim()),np.diff(ax.get_ylim())])
    
    xy_edges=[pol2cart(r_edges, theta_edge) for theta_edge in theta_edges[:-1]]
    xy_edges=[(x+x0,y+y0) for x,y in xy_edges]
    
    ### Plot
    for x_edge,y_edge in xy_edges:
        ax.plot([x0,x_edge],[y0,y_edge],':k',color=color,ls=ls,lw=lw)
        
    ### Set lim
    ax.axis(axis_old)
            
    ### Return
    
    return ax

#
    
def smart_arange(x_start,x_end,x_step): 
    x_out=np.linspace(x_start,x_end,int(round((x_end-x_start)/x_step+1)))
    new_step=x_out[1]-x_out[0]
    return (x_out,new_step)


def check_index(ind,data):
    
    if ind<0:
        ind=0
    elif ind>=len(data):
        ind=len(data)-1
        
    return ind

def SNR_pick(data,ind_center,N_left,N_right,mode='mean',flag_plot=False):
    """
    Cmpute SNR around given pick based on absolute values
    """
    abs_data=np.abs(data)
    ind_left=check_index(ind_center-N_left,abs_data)
    ind_right=check_index(ind_center+N_right,abs_data)
    if mode=='mean':
        right=np.mean(abs_data[ind_center:ind_right])
    elif mode=='max':
        right=np.max(abs_data[ind_center:ind_right])
    else:
        raise ValueError('mode has to be mean or max')
    left=np.mean(abs_data[ind_left:ind_center+1])
    
    ratio=right/left
    
    if flag_plot:
        _,ax=plt.subplots(2,1,sharex=True)
        ax[0].plot(data,color='k',ls='-',lw=1)
        ax[1].plot(abs_data,color='k',ls='-',lw=1)
        ax[1].axvline(x=ind_center,color='green')
        ax[1].axvline(x=ind_center-N_left,color='r')
        ax[1].axvline(x=ind_center+N_right,color='r')
        ax[1].text(0.1, 0.9,'Ratio=%.1f'%ratio, horizontalalignment='center',
          verticalalignment='center', transform=ax[1].transAxes)
        
        ### Cosmetic
        ax[0].set_ylabel('Data')
        ax[1].set_ylabel('abs(Data)')
    
    return ratio

def XY2XY_pcolormesh(X,Y):
    """
    Function made to change the coordinates of X and Y so that
    bins of pcolormesh are properly centered on the values and not the edge
    can now plot pcolormesh(Xe,Ye,Z)
    """
    
    diff_x=np.diff(np.append(X[0,:],2*X[0,-1]-X[0,-2]))/2
    x_edges=X[0,:]+diff_x
    x_edges=np.insert(x_edges,0,X[0,0]-diff_x[0])
    
    diff_y=np.diff(np.append(Y[:,0],2*Y[-1,0]-Y[-2,0]))/2
    y_edges=Y[:,0]+diff_y
    y_edges=np.insert(y_edges,0,Y[0,0]-diff_y[0])
    
    X_e,Y_e=np.meshgrid(x_edges,y_edges)

    return (X_e,Y_e)

        
def plot_vlines(x_poss,ax,markercolor='w',markersize=15,
                markeralpha=1,linecolor='w',linewidth=1,**axvline_kwargs):
    """
    Function made to vertical lines with different possible colors
    if marker color='variable'
    In the middle of the line will be plotted a star
    """
    y_min,y_max=ax.get_ylim()
    y_center=(y_min+y_max)/2

    x_poss=sws.tolist(x_poss)
    
    if markercolor is 'variable':
        markercolor=ggmt.data2rgb(range(len(x_poss)))
    else:
        markercolor=[markercolor]*len(x_poss)
    
    if linecolor is 'variable':
        linecolor=ggmt.data2rgb(range(len(x_poss)))
    else:
        linecolor=[linecolor]*len(x_poss)
    
    for idx,x_pos in enumerate(x_poss):
        ax.vlines(x_pos,y_min,y_max,colors=linecolor[idx],linestyles='dotted',
                     linewidth=linewidth,**axvline_kwargs)
        ax.plot(x_pos,y_center,'*',mfc=markercolor[idx],mec='k',markersize=markersize,
                alpha=markeralpha)
    
    return ax

def extend_periodic_array(y,cycle_border,perc_ext=10):
    """
    Function made to extend periodic data so that we don't have border
    artifacts when are doing the histogram
    We replicate some of the samples to extend the range
    
    
    Usedin
    ------
    movehisto2d_bin
    """
    

    #Numpy converting range of angles from (-Pi, Pi) to (0, 2*Pi)
    #(angles + 2 * np.pi) % (2 * np.pi)
    T=np.diff(cycle_border)
    alpha=cycle_border[0]
    
    ### Clean values outside borders to avoid counting events twice
    
    y_in=((y-alpha) %T)+alpha
    
    ### Triplicate
    
    y_mid=y_in[(y_in>alpha) & (y_in<alpha+T)]
    
    y_tri=np.hstack((y_in-T,y_mid,y_in+T))
    
    y_fin=y_tri[(y_tri>alpha-T*perc_ext/100) & (y_tri<alpha+T+T*perc_ext/100)]
    
    return y_fin

def trigo2azimuth_labels(ax_polar,axis='x'):
    """
    Function meant to transform the labels in trigo to azimuth
    
    Input:
        axis: ['x','y']
    """
    
    ### Check
    
    if axis not in ['x','y']:
        raise ValueError('axis parameter should be either x or y')
        
    ### Process
    
    if axis=='x':
        rad_xticks=ax_polar.get_xticks()
        azi_xticklabels=[matplotlib.text.Text(x,0,'%.0f°'%(trigo2azimuth(x*180/np.pi))) for x in rad_xticks]
        ax_polar.set_xticklabels(azi_xticklabels)
    else:
        rad_xticks=ax_polar.get_yticks()
        azi_xticklabels=[matplotlib.text.Text(0,x,'%.0f°'%(trigo2azimuth(x*180/np.pi))) for x in rad_xticks]
        ax_polar.set_yticklabels(azi_xticklabels)
    
    return ax_polar
    
    
def trigo2azimuth(trigos):
    """
    Transform values from trigo convention (CCW from x) to azimuth convention
    (CW from N)
    """
    azimuths=(-trigos+90)%360
    
    return azimuths


def plot_stations(station_list=None,ax=None,mfc='w',mec='k',ms=10,alpha=1,
                  ini_lon=-130.1,ini_lat=45.9,name=True,**text_kwargs):
        
    if ax is None:
        fig,ax=plt.subplots()
        ax.axis('equal')
    station_dic=swm.read_stationfile()
    
    if station_list is None:
        station_list=list(station_dic.keys())

    for station in station_list:
        if not station_dic.get(station,False): # skip if station not in keys
            continue
        lon,lat=station_dic[station]['lon'],station_dic[station]['lat']
        x,y=gproj.ll2xy(lon,lat,ini_lon,ini_lat)
        ax.plot(x,y,'^',mfc=mfc,mec=mec,ms=ms,alpha=alpha)
        if name is True:
            ax.annotate(station,(x,y),xytext=(0, 5),fontsize=8, textcoords='offset points',
                        ha='center',va='bottom',**text_kwargs)
        
    
    return ax

#######################################
### Fit Gaussians to distribution


def multi_norm(x, *params):
    """
    Used in decompose gaussian to plot and fit a gaussian
    params should be [mean1,std1,scale1,mean2,std2,scale2...]
    """
    norm=np.zeros_like(x)
    for kk in range(0,len(params),3):

        mean=params[kk]
        std=params[kk+1]
        scale=params[kk+2]
        
        #norm += scale*scipy.stats.cauchy.pdf(x, loc=mean ,scale=std)
        norm += scale*scipy.stats.norm.pdf(x, loc=mean ,scale=std)
        #norm += scale * np.exp( -((x - mean)/std)**2)
    return norm

def find_extrema(y,mode='max',flag_plot=False):
    """
    Function made to locate local minima and maxima
    """
    if mode=='max':
        bool_exts=np.r_[True, y[1:] > y[:-1]] & np.r_[y[:-1] > y[1:], True]
    else:
        bool_exts=np.r_[True, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], True]
    
    inds=np.linspace(0,len(y)-1,len(y),dtype=int)

    ind_exts=inds[bool_exts]
    ind_exts=ind_exts[(ind_exts>0) & (ind_exts<len(y)-1)]
    y_exts=y[ind_exts]
    
    
    if flag_plot:
        fig,ax=plt.subplots()
        
        ax.plot(inds,y,'r')
        ax.plot(ind_exts,y_exts,'+k')
    
    return (y_exts,ind_exts)


### Plot

def decompose_gaussians(x,y,flag_sort=True,flag_plot=False,ax=None):
    """
    Function made to decompose a distribution (conatinung only 
    positive values, i.e. histogram) into multiple gaussians and 
    return the standad devation and mean of each gaussians.
    Gaussians are centered on each local maxima
    
    Input
    -----
        x,y: np.array: input data
        flaf_sort: boolean: if True will sort the output valus from max y to min y
        flag_plot: boolean: if True will plot the data
        
    Output:
    ------
        x_maxs: np.array: values of the max of the gaussian
        std_maxs: np.array: std the gaussian
    """
    
    ### Find local extrema
    
    (_,ind_maxs)=find_extrema(y,mode='max',flag_plot=False)
    (_,ind_mins)=find_extrema(y,mode='min',flag_plot=False)
    
    ### Pad ind_mins with 0 and len
    
    ind_mins=np.hstack((0,ind_mins,len(y)-1))
    
    if flag_plot:
        if ax is None:
            fig,ax=plt.subplots()
        ax.plot(x,y,'-k',lw=0.5)
        xx=np.linspace(x[0],x[-1],200)
        colors_curve=ggmt.data2rgb(list(range(len(ind_maxs))),cmap=plt.cm.get_cmap('rainbow'))
        
    x_maxs=[]
    std_maxs=[]
    y_maxs=[]
    
    ### For each ind_max find closest min to the left and to the rights
    
    for k_max,ind_max in enumerate(ind_maxs):
        diffs=ind_mins-ind_max
        ind_left=np.max(diffs[diffs<0])+ind_max
        ind_right=np.min(diffs[diffs>0])+ind_max
        
        ### Find closest ind
        
        diff_clos=np.min([np.abs(ind_max-ind_left),np.abs(ind_max-ind_right)])
        ind_left_c=ind_max-diff_clos
        ind_right_c=ind_max+diff_clos
        
        ### Select data
        
        x_sel=x[ind_left_c:ind_right_c]
        y_sel=y[ind_left_c:ind_right_c]
        
        ### fit
        
        x_max=x[ind_max]
        y_max=y[ind_max]
        param=[x_max,1,1]
        ini_bound_l=[x_max-np.abs(0.01*x_max),0,0]
        ini_bound_r=[x_max+np.abs(0.01*x_max),30,np.inf]
        bound=(ini_bound_l,ini_bound_r)
        
        try:
            fitted_params,pcov = scipy.optimize.curve_fit(multi_norm,x_sel, y_sel, p0=param,bounds=bound)
        except:
            continue
        keep_param=list(fitted_params)
        
        std_max=fitted_params[1]
        
        x_maxs.append(x_max)
        std_maxs.append(std_max)
        y_maxs.append(y_max)
        
        if flag_plot:
            ax.plot(x_sel,y_sel,'k',lw=3)
            ax.plot(x_sel,y_sel,'r',color=colors_curve[k_max],lw=1.5)
            ax.plot(xx,multi_norm(xx,*keep_param),'-k',lw=0.5)
            ax.plot(xx,multi_norm(xx,*keep_param),':k',color=colors_curve[k_max],lw=2)
            
            ax.text(x_max,y_max+np.max(y)*0.03,'$%.0f \pm %.0f \sigma$'%(x_max,std_max),ha='center',clip_on=True)
    
    ### Cosmetic 
        
    if flag_plot:
        ylim=ax.get_ylim()
        ax.set_ylim([ylim[0],ylim[1]*1.1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    ### Sort from biggets to lowest max instead of left to righ
    y_maxs=np.array(y_maxs)
    x_maxs=np.array(x_maxs)
    std_maxs=np.array(std_maxs)
    
    if flag_sort==True:
        ind_sorts=np.argsort(-y_maxs)
        y_maxs=y_maxs[ind_sorts]
        x_maxs=x_maxs[ind_sorts]
        std_maxs=std_maxs[ind_sorts]
            
    return (x_maxs,std_maxs)    

def centered_histo(x,x_bins=None,x_width=None,x_start=None,x_end=None,x_over=0,ax=None,flag_plot=False):
    """
    Like classical numpy histogram but with bins centered instead of edges given
    """
    
    
    x_step=(1-x_over)*x_width
    (x_bins,x_step)=swm.smart_arange(x_start,x_end,x_step)
    x_width=x_step/(1-x_over)
    x_rights=x_bins+x_width/2
    x_lefts=x_bins-x_width/2

    ### Define meshes
    
    counts=np.zeros_like(x_bins)
        
    ##################
    ### Start Counting
    
    k_x=-1
    for x_left,x_right in zip(x_lefts,x_rights):
        k_x+=1
       
        counter=len(x[(x>=x_left) & (x<=x_right)]) # Count numbers of elements  
        counts[k_x]=counter # store

    ### Plot if asked
    if flag_plot:
        if ax is None:
            fig,ax=plt.subplots()
        ax.plot(x_bins,counts,'or',mfc='r',mec='k')
        
    ### Return
    
    return (x_bins,x_lefts,x_rights,counts)

#
##D=Cat.select(obs_station=['AXAS2'],lambda_lag=[0,8])
##D.write_pickle('test.nlloc')
#
#A=Cat.get_array('lambda_value',level=2)
#B=Cat.get_array('rec',level=2)
#
#plt.plot(A,B,'ok',alpha=0.01)
#
#
#D=Cat.select(lambda_lag=[0,20],lambda_rec=[0.9,1],lambda_value=[0,0.01],lambda_quality=[0,1])
#
#plt.close('all')
#norm=True
#
#D.plot_movehist2d('lag',level=2,y_width=1,norm_y=norm,x_over=0.95,smooth=True)
#D.plot_movehist2d('angle',level=2,y_width=0.1,norm_y=norm)
#D.plot_movehist2d('rec',level=2,y_width=0.02,norm_y=norm)
##D.plot_movehist2d('lambda_value',level=2,y_width=0.1,y_start=-0.1,y_end=0.5,norm_y=True)
#
#
##time=[x.s_time for x in D.obs]
##time=[y.s_time for y in D.obs for x in y.MinLambdas]
##angles=[x.angle for y in D.obs for x in y.MinLambdas]
##lags=[x.lag for y in D.obs for x in y.MinLambdas]
##time_rel=[(x-time[0])/86400 for x in time] 
#
##pickle.dump([lags,time],open('movehisto_date.pickle','wb'))
#
#plt.close('all')
#movehisto2d(time,lags,20,1,y_end=20,norm_y=True,smooth=True)
#movehisto2d(time,angles,20,0.1,norm_y=True,smooth=True)
#
#sys.exit()
#
#plt.figure()
#plt.hexbin(time_rel,angles,gridsize=(50,20),cmap=plt.get_cmap('jet'))
#plt.colorbar()
#plt.figure()
#plt.hexbin(time_rel,lags,gridsize=(50,12),cmap=plt.get_cmap('jet'))
#plt.figure()
#plt.hexbin(time_rel,lags,gridsize=(50,12),cmap=plt.get_cmap('jet'),vmax=15)
#plt.colorbar()
##plt.plot(time_rel,lags,'ok',mfc='none')
##plt.xlim(0,200)
##plt.ylim(0,20)
#
##for i in range(1000):
##    a=2
##    
##gutil.full_path_list(directory_file,start_with='',end_with='')
## reed_pickles
## Transform list of pickles files or pickle files into catalag of SWS_obs
