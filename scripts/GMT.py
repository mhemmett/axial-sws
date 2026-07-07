
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:05:21 2018

@author: baillard

Utilities made to plot GMT grid file with python, main function is plot_netcdf
"""


import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import matplotlib
import os
import projection,math
import util as gutil
import projection as gproj

                

def netcdf2mesh(netcdf_file,flag_plot=False,ini_lon=None,ini_lat=None,lon_lim=None,lat_lim=None):
    """
    function made to convert a netcdf file into a mesh (2D) that can further be plotted using imshow
    netcdf_file: str
        grid file, it can be a GMT file
        
    Return:
    -------
    
    Z: numpy 2D array
        array containing terrain elevation
    extent: list
        range used by imshow to specify range of plot
    origin: str
        specify origin of grid
    """

    from netCDF4 import Dataset
    from general import projection
    dataset = Dataset(netcdf_file) 
    dict_keys=list(dataset.variables.keys())
    
    if dict_keys == ['x_range', 'y_range', 'z_range', 'spacing', 'dimension', 'z']:
        type_grd=0
    elif dict_keys==['lon','lat','z']:
        type_grd=2
    else:
        type_grd=1
        
    print(type_grd)
    
    if type_grd==1:
        z=dataset.variables['z'][:]
        Z=z
        x_range=list(dataset.variables['x'].actual_range)
        y_range=list(dataset.variables['y'].actual_range)
        origin='lower'
        
    elif type_grd==2:
        z=dataset.variables['z'][:]
        Z=z
        x_range=list(dataset.variables['lon'].actual_range)
        y_range=list(dataset.variables['lat'].actual_range)
        origin='lower'
        
    elif type_grd==0:
        x_range=list(dataset.variables['x_range'][:])
        y_range=list(dataset.variables['y_range'][:])
        dim=dataset.variables['dimension'][:]        
        z=dataset.variables['z'][:]
        #z=np.flipud(z)
        Z=z.reshape(np.flipud(dim),order='C')
        Z=np.flipud(Z)
        origin='lower'
    
    extent=x_range+y_range
    
    if lon_lim!=None and lat_lim!=None:
        cut_extent=lon_lim+lat_lim
        Z,extent=submesh(Z,extent,cut_extent)
    

    if ini_lon!=None and ini_lat!=None:
        lon=extent[0:2]
        lat=extent[2:4]

        
        x,y=projection.ll2xy(lon,lat,ini_lon,ini_lat)
        
        if np.diff(x)>50 or np.diff(y>50):
            warnings.warn('Conversion of lon/lat to km/km might be biased because region is bigger than 50km/50km')
        x=list(x)
        y=list(y)
        
        
        
        extent=x+y
        
    return Z,extent,origin,dataset

def submesh(Z,extent,cut_extent):
    """
    Extract submesh from a 2D array terrain mesh
    
    Z: numpy array
        2D with eleveation
    extent: list
        original extent of the form [lon_left,lon_right, lat_bot, lat_top] (like for imshow)
    cut_extent: list
        new extent, meant to be smaller than the original one
    """

    nx=Z.shape[1]
    ny=Z.shape[0]
    xlim=extent[0:2]
    ylim=extent[2:4]
    
    x_lin=np.linspace(xlim[0],xlim[1],nx)
    y_lin=np.linspace(ylim[0],ylim[1],ny)
    
    idx_left = (np.abs(x_lin-cut_extent[0])).argmin()
    idx_right = (np.abs(x_lin-cut_extent[1])).argmin()
    idx_bot = (np.abs(y_lin-cut_extent[2])).argmin()
    idx_top = (np.abs(y_lin-cut_extent[3])).argmin()
    
    new_Z=Z[idx_bot:idx_top,idx_left:idx_right]
    
    cut_extent=[x_lin[idx_left],x_lin[idx_right],y_lin[idx_bot],y_lin[idx_top]]
    
    return new_Z,cut_extent


def plot_topo(Z,extent,origin='lower',cmap=plt.cm.get_cmap('jet_r'),ax=None,vmin=None,vmax=None,shade=False,alpha=None):
    """
    Mean to plot the topo itself by adding shading if wanted
    Z: numpy array
        data with terrain
    origin: str
        imshow origin of the mesh
    cmap: cmap object
    ax: axes object
    shade: Boolean
        if True, shading will be applied to the map
        
    Return
    ------
    
    (ax,im): axes and imshow objects
    
    """
    from matplotlib.colors import LightSource

    if ax is None:
        f1, ax = plt.subplots()

    if shade==True:
        ls=LightSource(azdeg=270, altdeg=0)
        rgb=ls.shade(Z,cmap=cmap, blend_mode='overlay',vert_exag=0.1,vmin=vmin,vmax=vmax)
        im = ax.imshow(Z, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',extent=extent)
        im.remove()
        ax.imshow(rgb,cmap=cmap,interpolation='bilinear',origin=origin,extent=extent,alpha=alpha,
                  rasterized=True)
        
        
    
    else:
        im = ax.imshow(Z, cmap=cmap,vmin=vmin,vmax=vmax,origin='lower',extent=extent,zorder=1,alpha=alpha,
                       rasterized=True)
    
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:4])
    
    return (ax,im)

def plot_netcdf(netcdf_file,ini_lon=None,ini_lat=None,lon_lim=None,lat_lim=None,cmap=plt.cm.get_cmap('jet_r'),shade=False,ax=None,vmin=None,vmax=None,alpha=None):
    """
    function made to plot directly the netcdf files
    """
    Z,extent,origin,dataset=netcdf2mesh(netcdf_file,flag_plot=False,ini_lon=ini_lon,ini_lat=ini_lat,lon_lim=lon_lim,lat_lim=lat_lim)
    (ax,im)=plot_topo(Z,extent,origin='lower',cmap=cmap,ax=ax,vmin=vmin,vmax=vmax,shade=shade,alpha=alpha)
    
    return (ax,im)

def num2degreelabel(num,direction):
    if direction=='lon':
        if num>=0:
            suf='E'
        else:
            suf='W'
    elif direction=='lat':
        if num>=0:
            suf='N'
        else:
            suf='S'
    
    new_str=('%.2f°'%(abs(num)))+suf
    
    return new_str

def xy2ll_axes(ax,ini_lon,ini_lat,lon_inc=1,lat_inc=1):
    from general.projection import xy2ll,ll2xy
    """
    Function made to convert matplolib axes to lon lat axes
    
    ax: mpl object
        axes of the plot
    ini_lon,ini_lat: float
        origin used for conversion frm km to lon lat
    lon_inc,lat_inc: float
        increment     
    """
#        
#    lon_inc=0.1
#    lat_inc=0.1
#    
#    ini_lon=-130.1
#    ini_lat=45.9
    
    x_lim=ax.get_xlim()
    y_lim=ax.get_ylim()
 
    lon_lim,lat_lim=xy2ll(x_lim,y_lim,ini_lon,ini_lat)
    
    lon_ticks_lim=np.round(lon_lim/lon_inc)*lon_inc
    lat_ticks_lim=np.round(lat_lim/lat_inc)*lat_inc
    
    lon_ticks=np.arange(lon_ticks_lim[0],lon_ticks_lim[1],lon_inc)
    lon_ticks=np.append(lon_ticks,lon_ticks_lim[-1])
    lat_ticks=np.arange(lat_ticks_lim[0],lat_ticks_lim[1],lat_inc)
    lat_ticks=np.append(lat_ticks,lat_ticks_lim[-1])
    
    ## Reconvert to x and y ticks
    
    nx_ticks,_=ll2xy(lon_ticks,np.ones(lon_ticks.size)*lat_ticks[0],ini_lon,ini_lat)
    _,ny_ticks=ll2xy(np.ones(lat_ticks.size)*lon_ticks[0],lat_ticks,ini_lon,ini_lat)
    
    ### Get tick labels
    
    lon_ticklabels=[plt.text(x,0,num2degreelabel(x,'lon')) for x in lon_ticks]
    lat_ticklabels=[plt.text(0,x,num2degreelabel(x,'lat')) for x in lat_ticks]
    
    
    ax.set_xticks(nx_ticks)
    ax.set_yticks(ny_ticks)
    ax.set_xticklabels(lon_ticklabels)
    ax.set_yticklabels(lat_ticklabels)
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    return ax

def add_scale(ax,scale_length,scale_unit='km',scale_fontsize=12,scale_fontcolor='white'):
    """
    Function made to add scale to data
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
        
    fontprops = fm.FontProperties(size=12)
    scalebar = AnchoredSizeBar(ax.transData,
                               scale_length,'%.0f %s'%(scale_length,scale_unit), 'lower left', 
                               pad=1,
                               color=scale_fontcolor,
                               frameon=False,
                               size_vertical=0.02,
                               fontproperties=fontprops)
    
    ax.add_artist(scalebar)
    
    return ax
        

def plot_seismic_lines(file_in='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/VELOCITY/seismic_lines_axial_centered.txt',
                       ini_lon=-130.1,ini_lat=45.9,
                       ax=None,key='',fontsize=12,color='k',ls='--',lw=1,flag_cross=False):
    ### subfunctions
    
    def _file2list(file_in):
        """
        Function made to convert profile file into list, the file prof is of the form:
            
        num lon lat degree length_left length_right
        44  -129.8970556018282  45.9325296303469   -66.754078934341337  18     13.7375
        45  -130.0083610197549  45.8964387193036   -64.841653537197388   14   20.5
        
        see ../VELOCITY/seismic_lines_axial.txt for example
        """
    
        fic=open(file_in,'rt')
        
        lines=fic.readlines()
        
        A=[x.split() for x in lines[1:]]
         
        B=[]
        for list_single in A:
            new=[list_single[0]]+[float(x) for x in list_single[1:]]
            B.append(new)
        
        fic.close()
        
        return B
    
    #### Main Function
    
    list_prof=_file2list(file_in)
    
    if ax is None:
        fig,ax=plt.subplots()
        ax.set_aspect('equal')
    

    for list_single in list_prof:
        key,lon_prof,lat_prof,ang,llen,rlen=list_single
        
        x_prof,y_prof=gproj.ll2xy(lon_prof,lat_prof,ini_lon,ini_lat)

        plot_box([x_prof,y_prof],ang,[llen,rlen],ax=ax,key=key,flag_plot=True,
                 fontsize=fontsize,color=color,ls=ls,lw=lw,flag_cross=flag_cross)

        
    for line in ax.lines:
        line.set_color(color)
    return ax
        
    

def plot_box(center,angle_deg,len_prof,width_prof=None,ax=None,
             key='',flag_plot=True,fontsize=12,color='k',ls='-',lw=1,flag_cross=False,
             flag_label=True):
    """
    Made to plot profile box on a map (map view) (for profile for example)
    
    """
    
    ###################################
    ####### Functions needed ##########
    
    
    def axis2poly(axis_c):
    
        x0,x1,y0,y1=axis_c
    
        x_box=[x0,x1,x1,x0,x0]
        y_box=[y0,y0,y1,y1,y0]
        
        x_boxi,y_boxi=gutil.smooth_curve(x_box,y_box,periodic=False,num_points=500,smoothness=0,flag_plot=False,k=1)
    
        return np.array(list(zip(x_boxi,y_boxi)))
    
    def line2poly(A,B):
        
        x0,y0=A
        x1,y1=B
        x_lin=[x0,x1]
        y_lin=[y0,y1]
    
        x_lini,y_lini=gutil.smooth_curve(x_lin,y_lin,periodic=False,num_points=500,smoothness=0,flag_plot=False,k=1)
    
        return np.array(list(zip(x_lini,y_lini)))
    
    
    def find_cross(A,B,axis_c,flag_plot=False):
        """
        Function made to find the interesection between a line and a box
        """
        poly_axis=axis2poly(axis_c)
        poly_line=line2poly(A,B)
            
        
        x_sel,y_sel,bool_array=gproj.is_in_polygon(poly_line[:,0],poly_line[:,1],
                                                   poly_axis[:,0],poly_axis[:,1],flag_plot=flag_plot)
        
    
        if len(x_sel)==0:
            x_cross=None
            y_cross=None
        else:
            if (x_sel[0]==poly_line[0,0]) and (y_sel[0]==poly_line[0,1]):
                x_cross=None
                y_cross=None
            else:
                x_cross=x_sel[10]
                y_cross=y_sel[10]
            
    
        if flag_plot:
            ax=plt.gca()
            if x_cross is not None:
                ax.plot(x_cross,y_cross,'+g')
            
        return x_cross,y_cross

    
    ########## Main script
    flag_exist=True
    if flag_plot==True:
        if ax is None:
            flag_exist=False
            fig, ax = plt.subplots()
            
    ### Get axis limits
    
    axis_c=ax.axis()

    #### Define lenght and width
    L=np.sum(len_prof)
    if width_prof is None:
        width_prof=[0,0]
    else:
        W=np.sum(width_prof)/2
    angle=angle_deg*np.pi/180
    
    center=np.array(center)
    
    M=center+len_prof[0]*np.array([-np.cos(angle),-np.sin(angle)])
    N=M+L*np.array([np.cos(angle),np.sin(angle)])
    
    x_b=width_prof[0]*np.sin(angle)
    y_b=width_prof[0]*np.cos(angle)
    
    x_t=width_prof[1]*np.sin(angle)
    y_t=width_prof[1]*np.cos(angle)

    A=M+np.array([-x_t,y_t])
    B=M+np.array([x_b,-y_b])
    
    C=N+np.array([x_b,-y_b])
    D=N+np.array([-x_t,+y_t])
#    
#    x_s=W*np.sin(angle)
#    y_s=W*np.cos(angle)
    
#    C=N+np.array([x_s,-y_s])
#    D=N+np.array([-x_s,+y_s])
#    
#    A=M+np.array([-x_s,y_s])
#    B=M+np.array([x_s,-y_s])
    
    OUT=np.vstack((A,B,C,D,A))
    
    IN=np.vstack((M,N))
    
    if flag_plot==True:
        
        if width_prof!=[0,0]:
            ax.plot(OUT[:,0],OUT[:,1],color=color,linewidth=0.15)


        ax.plot(IN[:,0],IN[:,1],color=color,ls=ls,lw=lw)
        if flag_label:
            ax.plot(IN[0,0],IN[0,1],'ok',markersize=fontsize*1.6)
            ax.text(IN[0,0],IN[0,1],key,ha='center',va='center',color='w',fontsize=fontsize,weight='bold',clip_on=True)
                
        ### Find intersection
        if flag_cross:
            x_cross,y_cross=find_cross(M,N,axis_c,flag_plot=False) 
            if x_cross is not None:
                ax.text(x_cross,y_cross,key,ha='left',va='bottom',color='k',style='italic',
                         fontsize=fontsize,rotation=angle_deg,rotation_mode='anchor')
            
        if flag_exist is False:
            ax.set_aspect('equal')
            
    start_point=M
    end_point=N
    return (start_point,end_point)
    
    
def plot_circle(xo,yo,r,ax=None,color='white'):
    """
    Function made to plot circle
    """
    
    i=np.linspace(0,360,360)*np.pi/180
    
    x=r*np.cos(i)+xo
    y=r*np.sin(i)+yo
    
    if ax==None:
        _,ax=plt.subplots()
        
    ax.plot(x,y,color=color)
    ax.plot(xo,yo,'o',mec=color,mfc=color)
    
    return ax
    
def read_profileparam(profile_file):
    """
    keywords: profile, read, parameters
    
    Function made to read a profiles text file of the form:
    "center=[5.7,4.4]
    angle_deg=20
    len_prof=[0,5]
    width_prof=[0.5,0.5]
    
    center=[6,4.4]
    angle_deg=20
    len_prof=[0,5]
    width_prof=[0.5,0.5]"
    
    Return:
    --------
    
    dictionnary with all param for each set
           
    """
    
    fic=open(profile_file)
    lines=fic.read().splitlines()
    counter=0    
    k=-1
    dic_multiple={}
    while k < len(lines)-1:
        k+=1
        line=lines[k]
        try:
            val=eval(line.split('=')[1])
        except:
            continue
        if 'center' in line:
            tt=0
            dic_single={}
            dic_single['center']=val
            tt+=1
        elif 'angle_deg' in line:
            dic_single['angle_deg']=val
            tt+=1
        elif 'len_prof' in line:
            dic_single['len_prof']=val
            tt+=1
        elif 'width_prof' in line:
            dic_single['width_prof']=val
            tt+=1
            
        if tt==4:
            counter+=1
            dic_multiple[counter]=dic_single
                    
    fic.close()
    
    return dic_multiple
    
def get_profileparam(profile_file,key=None):
    """
    Function to retrieve the profile param specific to one set
    
    profile_file: str
    key: float
        order of profile in text file
        
    return:
    ------
    parameters used by plot_cross
    """

    dic_multiple=read_profileparam(profile_file)
    
  
    
    if key is None:
        center,angle_deg,len_prof,width_prof=[],[],[],[]
        for key_param in dic_multiple.keys():
            center.append(dic_multiple[key_param]['center'])
            angle_deg.append(dic_multiple[key_param]['angle_deg'])
            len_prof.append(dic_multiple[key_param]['len_prof'])
            width_prof.append(dic_multiple[key_param]['width_prof'])   
    else:
        center,angle_deg,len_prof,width_prof= dic_multiple[key]['center'],\
        dic_multiple[key]['angle_deg'],\
        dic_multiple[key]['len_prof'],\
        dic_multiple[key]['width_prof']
        
    return center,angle_deg,len_prof,width_prof

#def write_profileparam
#get_profileparam

def get_cax(ax,cax_x0=1.03,cax_y0=0,cax_width=0.05,cax_height=1):
    """
    Function made to return the cax positon of the colorbar, ax is the master ax
    ex: plt.colorbar(im,cax=cax), works also with polar plots
    
    Inputs:
        cax_values: float between 0 and 1
    
    Examples:
        cax_x0=0 and cax_y0=0 means lower left corner
    
    TODO
    -----
    Only works when appending right axes, implement also bottom and left axes
    """
    ax_positioner=ax.get_position()
    
    diff_x=ax_positioner.width
    diff_y=ax_positioner.height
    
    cax_x0=(ax_positioner.x0)+cax_x0*diff_x
    cax_y0=(ax_positioner.y0)+cax_y0*diff_y
    cax_width=ax_positioner.width*cax_width
    cax_height=ax_positioner.height*cax_height

    fig=ax.get_figure()
    cax = fig.add_axes([cax_x0, cax_y0, cax_width, cax_height])
    
    return cax

def plot_stations(station_list=None,ax=None,mfc='w',mec='k',ms=10,alpha=1,
                  ini_lon=-130.1,ini_lat=45.9,name=True,**text_kwargs):
        
    if ax is None:
        fig,ax=plt.subplots()
        ax.axis('equal')
    station_dic=read_stationfile()
    
    if station_list is None:
        station_list=list(station_dic.keys())

    for station in station_list:
        if not station_dic.get(station,False): # skip if station not in keys
            continue
        lon,lat=station_dic[station]['lon'],station_dic[station]['lat']
        x,y=gproj.ll2xy(lon,lat,ini_lon,ini_lat)
        ax.plot(x,y,'^',mfc=mfc,mec=mec,ms=ms,alpha=alpha)
        if name is True:
#            ax.annotate(station,(x,y),xytext=(0, 5),fontsize=8, textcoords='offset points',
#                        ha='center',va='bottom',**text_kwargs)
            ax.text(x,y*1.02,station,fontsize=8,ha='center',va='bottom',clip_on=True)
        
    
    return ax

def read_stationfile(station_file=None):
    """
    Read station file having lon,lat,z,name as columns

    Default station_file resolves to this repo's data/stations_axial.llz - was previously
    hardcoded to a path on Christian Baillard's own machine (/home/baillard/Dropbox/...),
    which never existed here.
    """

    if station_file is None:
        station_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'stations_axial.llz')

    print('Make sure coordinates are given in lon and lat')

    with open(station_file,'rt') as fic:
        lines=fic.readlines()
        station_dic={}
        for line in lines:
            lon,lat,z=[float(x) for x in line.split()[0:3]]
            station=line.split()[-1]
            station_dic[station]={'lon':lon,'lat':lat,'z':-z}
        
    return station_dic


    
def plot_lines(line_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/caldera_smooth.ll',
               ax=None,color='k',lw=1,ls='-',bg_color='k',bg_lw=None,
               ini_lon=-130.1,ini_lat=45.9):
    """
    Function made to plot lines from data file, the initial file should 
    have at least 2 columns specifying the lon and latitude of the data
    A file can have multiple contours specified, each contours separated 
    by a '>' line (GMT convention)
    
    ...
    lon lat
    lon lat
    >
    lon lat
    ...
    
    Input
    -----
        line_file: str
        ax: plt axe
        color,bg_color:
        lw,bg_lw:
        ini_lon,ini_lat:            
    """
    ### Check
    
    if ax is None:
        fig, ax = plt.subplots()
        
    ### Read file

    data_list=gutil.read_datafile(line_file)
    
    ### Plot
    
    h=[]
    for single_array in data_list:
        x,y=gproj.ll2xy(single_array[:,0],single_array[:,1],ini_lon,ini_lat)
        if bg_lw is not None:
            h2=ax.plot(x,y,color=bg_color,lw=bg_lw,ls='-')
            h.append(h2)
        h1=ax.plot(x,y,color=color,lw=lw,ls=ls)
        h.append(h1)


    
    return (ax,h)

def plot_cross_surface(center,angle_deg,len_prof,width_prof,
                       surface_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/AMC_clipped_1525.llzd',
                       ax=None,
                   mfc='r',mec='k',markersize=3,alpha=1,color_col_num=None,linestyle=None,vmin=None,vmax=None,cmap=plt.cm.get_cmap('gray'),
                   color='k',lw=2,num_points=100):
        """
        Made to plot cross sections of surface
        """

        ### Read surface file
        ini_lon=-130.1
        ini_lat=45.9
        surface=np.loadtxt(surface_file)
        x,y=projection.ll2xy(surface[:,0],surface[:,1],ini_lon,ini_lat)
        surface[:,0]=x
        surface[:,1]=y
        
        #### Check
 
        if ax is None:
            _,ax=plt.subplots()
            
        if color_col_num is None:
            color_scatter=mfc
        
        num_col=surface.shape[1]
        if color_col_num is not None and color_col_num > num_col:
            raise ValueError('color_col is bigger than surface size')
            
        ### Read surfaces and lines
    
        tmp_proj_surf,_=projection.project(surface,center,angle_deg,len_prof,width_prof)        
        
        #### Smooth
        
        x_surf,y_surf=math.average_data(tmp_proj_surf[:,0],tmp_proj_surf[:,2],num_points=num_points,mode='max',flag_plot=False)
        
        if color_col_num is not None:
            x_surf,color_scatter=math.average_data(tmp_proj_surf[:,0],tmp_proj_surf[:,color_col_num],num_points=num_points,mode='max',smooth_window=1,flag_plot=False)

        #### Project lines crossing as points
        
        if linestyle is None:
       
            h1=ax.scatter(x_surf,y_surf,s=markersize,c=color_scatter,vmin=vmin,vmax=vmax,cmap=cmap,zorder=10)
            ax.plot(x_surf,y_surf,':k',lw=1,zorder=11)
        else:
            h1=ax.plot(x_surf,y_surf,linestyle=linestyle,color=color,lw=lw)

    
        return (ax,h1)
    
def plot_cross_xlines(center,angle_deg,len_prof,width_prof,
                       line_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/GRIDS/caldera_smooth.ll',
                       ini_lon=-130.1,ini_lat=45.9,ax=None,
                       marker='v',mfc='k',alpha=1,markersize=10,**kwargs):
    """
    Function made to plot crossings of lines with profiles as points 
    
    Input:
        line_file: file containing at least two columns with coordinates of the line to be processed
    """
    
    from general import projection,math
    import numpy as np
    import matplotlib.pyplot as plt
    
    #### Check
 
    if ax is None:
        _,ax=plt.subplots()

    ### Process
    
    line=np.loadtxt(line_file)
    
    #### Convert to x,y if necessa
    
    x,y=projection.ll2xy(line[:,0],line[:,1],ini_lon,ini_lat)
    line[:,0]=x
    line[:,1]=y
    
    #### Project

    line=np.column_stack((line,np.zeros(line.shape[0])))
    tmp_proj_line,_=projection.project(line,center,angle_deg,len_prof,width_prof)
    
    if tmp_proj_line.size==0:
        return ax
    
    #### Cluster the points
    
    x_line=math.cluster_1d(tmp_proj_line[:,0],1,flag_plot=False)
    tmp_proj_line=np.column_stack((x_line,np.zeros(x_line.shape)))

    ### Plot
    
    ax.plot(tmp_proj_line[:,0],tmp_proj_line[:,1],'v',marker=marker,mfc=mfc,mec='none',markersize=markersize,**kwargs)
            
    return ax
 
def plot_cross_stations(center,angle_deg,len_prof,width_prof,z_shift=0,
                       station_file='/home/baillard/Dropbox/_Moi/Projects/Axial_EQ/DATA/STATIONS/stations_axial_1525m.llz',
                       ini_lon=-130.1,ini_lat=45.9,ax=None,
                       marker='^',mfc='w',mec='k',mew=1,alpha=1,ms=10,clip_on=False):
    """
    Function made to plot the stations 
    """
            
    ### Check
    if ax is None:
        _,ax=plt.subplots()
        
    ### Read station
    lon_sta,lat_sta,z_sta=np.loadtxt(station_file,unpack=True,usecols=[0,1,2])
    z_sta=-z_sta
    
    ### Convert
    x_sta,y_sta=projection.ll2xy(lon_sta,lat_sta,ini_lon,ini_lat)
    data=np.column_stack((x_sta,y_sta,z_sta+z_shift))            
    
    ### Project
    proj_data,_=projection.project(data,center,angle_deg,len_prof,width_prof)
    print(proj_data)
    ### Plot
    
    try:
        ax.plot(proj_data[:,0],proj_data[:,2],
                linestyle='none',marker=marker,mfc=mfc,mew=mew,alpha=alpha,ms=ms,clip_on=clip_on,mec=mec)
    except Exception as e:
        print(e)
        pass
    
    return ax

def clean_subplot_axes(ax_list,reshape=True):
    """
    Function made to remove labels and titles and only keep the ones in the bottom and the left edge
    ax_list is obtains from a reshdape
    """
                
    if reshape:
        ax_list=ax_list.reshape(-1)
    tot=len(ax_list)          
    i_xticklabel_rm=range(0,tot-2)
    i_yticklabel_rm=range(1,tot,2)
    i_ylabel_rm=range(1,tot,2)
    i_xlabel_rm=range(0,tot-2)
    for kk in range(len(ax_list)):
        ax_map=ax_list[kk]
        
        if kk in i_xticklabel_rm:
            ax_map.set_xticklabels([])
            
        if kk in i_yticklabel_rm:
            ax_map.set_yticklabels([])
    
        if kk in i_xlabel_rm:
            ax_map.set_xlabel('')
        
        if kk in i_ylabel_rm:
            ax_map.set_ylabel('')
            
    return ax_list

def round_labels(cax,precision=1,axis='y'):
    """
    Function made to give nice rounded labels
    
    Input:
        precision: float,int
            numnber of decimals
    """
    
    if axis=='y':
        list_text=cax.get_yticklabels()
    else:
        list_text=cax.get_xticklabels()
        
    new_text=[]
    for text_obj in list_text:
        string_val=text_obj.get_text()


        if ord(string_val[0])==8722:
            string_val='-'+string_val[1:]
        if precision<=0:
            val= int(   round(   float(string_val)  ,precision)   )
            format_str='%i'
        else:
            val=round(   float(string_val)  ,precision)
            format_str='%.'+str(precision)+'f'
            
        text_obj.set_text(format_str %(val))
        new_text.append(text_obj)
    
    if axis=='y':
        cax.set_yticklabels(new_text)
    else:
        cax.set_xticklabels(new_text)
    
    return cax
        
def get_Ncolors(num_colors=256,colormap_name='jet'):
    """
    Function made to retrieve N colors from a given colormap
    
    Inputs
    ------
    num_colors: int: number of desired colors
    colormap_name: str: matplotlib valid colormap
    
    Outputs:
    -------
    
    rgb_list: list: rgb list containnf rgb values
    
    """
    from matplotlib import cm

    kk=0
    
    rgb_list=[]
    
    while kk<num_colors:
        kk+=1
        ind=int(kk*256/(num_colors+1))
        cmd_str='cm.'+colormap_name+'(ind)'
        rgb=eval(cmd_str)
        rgb_list.append(rgb[0:-1])
        
    return rgb_list



def data2rgb(data,cmap=matplotlib.cm.jet,vmin=None,vmax=None):
    """
    Function made to get rgb array from data
    """
    
    range_data=np.max(data)-np.min(data)
    
    if vmin is None:
        vmin=np.min(data)-0.1*range_data
    if vmax is None:
        vmax=np.max(data)+0.1*range_data
        

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba=mapper.to_rgba(data)
    
    return rgba

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def create_plotgrid(nrows,ncols,height=2.5,width=2.5,
                    left=0.1,top=0.9,bottom=0.1,right=0.9,
                    hspace=0.1,wspace=0.1,clean_ticks=False):
    """
    Function made to create a set of subplots containing nrows and ncols
    
    Input
    -----
        nrows,ncols:int: number of desired rows and columns
        height,width: float: inches of single ax
    Ouput
    -----
        fig: mpl.figure object
        ax_list: np.array containing mpl.axes of the shape [nrows,ncol] 
    """
    
    ### Define size of the figure and GridSpec specificities
    
    nrows=int(nrows)
    ncols=int(ncols)
    fig_height=nrows*height
    fig_width=ncols*width
    fig=plt.figure(figsize=[fig_width,fig_height])
    
    gs_main = matplotlib.gridspec.GridSpec(nrows, ncols, figure=fig,
                       left=left, top=top, bottom=bottom,right=right,
                       hspace=hspace,wspace=wspace)

    
    ### Create axes list
    
    ax_list=[]
    for k_col in range(ncols):
        ax_col=[] 
        for k_row in range(nrows):
            ax = plt.Subplot(fig, gs_main[k_row,k_col]) 
            ax_col.append(ax)
            fig.add_subplot(ax)
            
            if clean_ticks:
                if k_row!=nrows-1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    ax.set_xlabel('')
                if k_col!=0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.set_ylabel('')
            
        ax_list.append(ax_col)
        

    ### Transform list to array
    
    ax_array=np.array(ax_list).transpose()
            
    #plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)     
    
    return (fig,ax_array)


#ini_lat=45.9
#
#from netCDF4 import Dataset
#from general import projection
#dataset = Dataset(netcdf_file) 
#
#lon_lim=[-130.1,-129.9]
#lat_lim=[45.9,46]
#Z,extent,origin,dataset=netcdf2mesh(netcdf_file,ini_lon=ini_lon,ini_lat=ini_lat,lon_lim=lon_lim,lat_lim=lat_lim)
#plot_topo(Z,extent,origin='lower',vmin=-1700,cmap=cmap,shade=False)
#
#lon_sta=-130.0089
#lat_sta=45.95468
#x_sta,y_sta=projection.ll2xy(lon_sta,lat_sta,ini_lon,ini_lat)
#plt.plot(x_sta,y_sta,'or')
#
#(ax,im)=plot_netcdf(netcdf_file,cmap=cmap,ax=None,alpha=0.8,vmin=-3000,vmax=None,shade=True,lon_lim=[-130.2,-129.7],lat_lim=[45.8,46.1],ini_lon=-130.1, ini_lat=45.9)
#
#plt.colorbar(im)
