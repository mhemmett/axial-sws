def plot_movehisto2d(x,y,
                x_label='X',y_label='Y',title='',ax=None,vmax=None,
                **movehisto2d_kwargs):
    """
    Function made to plot an histo2d but using a moving window in both directions, this ensure better
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
        flag_y_norm: Boolean: True to normalize by the maximum in each column
        flag_resample: Boolean: Enable resampling to uniform grid (required for smoothing)
        flag_filter: Boolean: Enable Gaussian filtering (requires flag_resample=True)
        x_filter_per, y_filter_per: float in [0,100]: width percentage for smoothing
        filter_mode: str or list: 'nearest' or 'wrap' for edge handling
        [x,y]_label: str
        
    Ouputs
    ------
        ax, im, X, Y, Z, x_bins, x_diffs
        
    Comments:
    ---------
        Uses Baillard's movehisto2d_bin for resampling and filtering
        
    UsedIn
    ------
        SWSCat.plot_movehisto2d_time, wrapper functions
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

    ### Bin the data (smoothing handled inside movehisto2d_bin via resampling + filtering)
    
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
       
    (Xm,Ym)=swm.XY2XY_pcolormesh(X,Y) # To ensure Pcolormesh will be centered on bins

    #im=ax.pcolormesh(X,Y,Z,cmap=plt.cm.get_cmap('jet'),rasterized=True)
    im=ax.pcolormesh(Xm,Ym,Z,cmap=plt.cm.get_cmap('inferno'),rasterized=True,vmax=vmax)
    
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


def plot_dt_timeseries_movehisto(results_df, time_column='event_datetime',
                                  x_width=5*24*3600, x_overlap=0.95,
                                  y_width=2, y_overlap=0.9,
                                  sampling_rate=200.0,
                                  figsize=(14, 6),
                                  station='AXAS2',
                                  flag_smooth=True,
                                  x_filter_per=10,
                                  y_filter_per=10):
    """
    Plot delay time (dt) over time using moving window 2D histogram with Baillard-style smoothing.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with splitting results
    time_column : str
        Name of time column (default 'event_datetime')
    x_width : float
        Width of time window in seconds (default 5 days = 5*24*3600)
    x_overlap : float
        Overlap fraction for time windows (0.95 = 95% overlap)
    y_width : float
        Bin width for dt in samples
    y_overlap : float
        Overlap for y-direction
    sampling_rate : float
        Sampling rate in Hz (default 200 Hz)
    figsize : tuple
        Figure size
    station : str
        Station name for title
    flag_smooth : bool
        Apply Baillard-style smoothing via resampling (default True)
    x_filter_per : float
        Gaussian filter width percentage in time direction (default 10)
    y_filter_per : float
        Gaussian filter width percentage in y direction (default 10)
    """
    import sws_methods as swm
    # Extract data
    df = results_df.copy()
    times = pd.to_datetime(df[time_column])
    dt_samples = df['dt'].values * sampling_rate
    
    # Convert to UTCDateTime for movehisto2d
    x = np.array([UTCDateTime(t) for t in times])
    y = dt_samples
    
    # Set time range
    x_start = UTCDateTime(times.min())
    x_end = UTCDateTime(times.max())
    y_start = 0
    y_end = 30

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with Baillard-style resampling and smoothing
    (ax, im, X, Y, Z, x_bins, x_diffs) = plot_movehisto2d(
        x, y,
        x_label='Date',
        y_label='Delay Time δt (samples)',
        ax=ax,
        x_width=x_width,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        x_over=x_overlap,
        y_over=y_overlap,
        y_width=y_width,
        flag_y_norm=True,  # Normalize by column (Baillard's norm_y)
        flag_resample=flag_smooth,  # Enable resampling for smoothing
        flag_filter=flag_smooth,  # Enable Gaussian filtering after resampling
        x_filter_per=x_filter_per,  # Smoothing percentage in time
        y_filter_per=y_filter_per,  # Smoothing percentage in y
        filter_mode='nearest',  # Non-cyclic edge handling
        vmax=None
    )
    
    # Add eruption line
    eruption_start_time = UTCDateTime(2015, 4, 24, 6)
    ax.axvline(eruption_start_time.matplotlib_date, color='white', 
               linestyle='--', alpha=0.9, linewidth=2, label='Eruption Onset')

    eruption_end_time = UTCDateTime(2015, 5, 19, 0)
    ax.axvline(eruption_end_time.matplotlib_date, color='white', 
               linestyle='--', alpha=0.9, linewidth=2, label='Eruption End')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Normalized Density', fontsize=10)
    
    # Add title with statistics
    dt_mean = df['dt'].mean()
    dt_std = df['dt'].std()
    title = (f"Delay Time Time Series - Station {station}\n"
             f"N = {len(df)} events | "
             f"δt: {dt_mean:.3f} ± {dt_std:.3f} s "
             f"({dt_mean*sampling_rate:.1f} ± {dt_std*sampling_rate:.1f} samples)")
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_phi_timeseries_movehisto(results_df, time_column='event_datetime',
                                   x_width=5*24*3600, x_overlap=0.95,
                                   y_width=9, y_overlap=0.9,
                                   figsize=(14, 6),
                                   station='AXAS2',
                                   flag_smooth=True,
                                   x_filter_per=10,
                                   y_filter_per=10):
    """
    Plot fast direction (phi) over time using moving window 2D histogram with Baillard-style smoothing.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with splitting results
    time_column : str
        Name of time column (default 'event_datetime')
    x_width : float
        Width of time window in seconds (default 5 days = 5*24*3600)
    x_overlap : float
        Overlap fraction for time windows (0.95 = 95% overlap)
    y_width : float
        Bin width for phi in degrees (default 9 = 180/20)
    y_overlap : float
        Overlap for y-direction
    figsize : tuple
        Figure size
    station : str
        Station name for title
    flag_smooth : bool
        Apply Baillard-style smoothing via resampling (default True)
    x_filter_per : float
        Gaussian filter width percentage in time direction (default 10)
    y_filter_per : float
        Gaussian filter width percentage in y direction (default 10)
    """
    import sws_methods as swm
    
    # Extract data
    df = results_df.copy()
    times = pd.to_datetime(df[time_column])
    phi_deg = df['phi'].values
    
    # Convert to UTCDateTime for movehisto2d
    x = np.array([UTCDateTime(t) for t in times])
    y = phi_deg
    
    # Set time range
    x_start = UTCDateTime(times.min())
    x_end = UTCDateTime(times.max())
    y_start = -90
    y_end = 90
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with Baillard-style resampling and smoothing
    (ax, im, X, Y, Z, x_bins, x_diffs) = plot_movehisto2d(
        x, y,
        x_label='Date',
        y_label='Fast Direction φ (°)',
        ax=ax,
        x_width=x_width,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        x_over=x_overlap,
        y_over=y_overlap,
        y_width=y_width,
        flag_y_norm=True,  # Normalize by column (Baillard's norm_y)
        flag_resample=flag_smooth,  # Enable resampling for smoothing
        flag_filter=flag_smooth,  # Enable Gaussian filtering after resampling
        x_filter_per=x_filter_per,  # Smoothing percentage in time
        y_filter_per=y_filter_per,  # Smoothing percentage in y
        #filter_mode=['nearest', 'wrap'],  # 'wrap' for cyclic phi, 'nearest' for time
        filter_mode = ['wrap', 'nearest'],
        vmax=None
    )
    
    # Add eruption line
    eruption_start_time = UTCDateTime(2015, 4, 24, 6)
    ax.axvline(eruption_start_time.matplotlib_date, color='white', 
            linestyle='--', alpha=0.9, linewidth=2, label='Eruption Onset')

    eruption_end_time = UTCDateTime(2015, 5, 19, 0)
    ax.axvline(eruption_end_time.matplotlib_date, color='white', 
               linestyle='--', alpha=0.9, linewidth=2, label='Eruption End')
    
    # Add horizontal line at 0
    ax.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Normalized Density', fontsize=10)
    
    # Add title with statistics
    phi_mean = df['phi'].mean()
    phi_std = df['phi'].std()
    title = (f"Fast Direction Time Series - Station {station}\n"
             f"N = {len(df)} events | "
             f"φ: {phi_mean:.1f}° ± {phi_std:.1f}° "
             f"({np.deg2rad(phi_mean):.2f} ± {np.deg2rad(phi_std):.2f} rad)")
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

## Do it for every AXAS2 file
def output_plots(filename, first_start, last_start, first_end, last_end, station):
    results_df = pd.read_csv(filename)

    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: UTCDateTime(x))

    # Convert results_df event_datetime to datetime for plotting
    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: x.datetime)

    fig, (ax1, ax2), (results_before, results_after) = plot_fast_direction_rose_eruption_comparison(
        results_df,
        title_prefix="Fast Direction Distribution, " + str(station) + ", SWSPy: " + str(first_start) + "-"+ str(last_start) + "σ, " + str(first_end) + "-" + str(last_end) + " Tmid End",
        nbins=36,  # 10° bins
        color='steelblue',
        figsize=(16, 7)
    )
    plt.show()

    results_df = pd.read_csv(filename)

    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: UTCDateTime(x))

    # Filter results_df to be +/- 24 hrs of the eruption onset
    eruption_time = UTCDateTime(2015, 4, 24, 6)
    time_window = 48 * 3600  # 48 hours in seconds
    start_time = eruption_time - time_window
    end_time = eruption_time + time_window
    mask = (results_df['event_datetime'] >= start_time) & (results_df['event_datetime'] <= end_time)
    results_df = results_df[mask]

    # Convert results_df event_datetime to datetime for plotting
    results_df['event_datetime'] = results_df['event_datetime'].apply(lambda x: x.datetime)

    results_df_baillard_phi = results_df.copy()
    # Conversion formula: phi_ccw_from_E = 90° - phi_cw_from_N
    results_df_baillard_phi['phi'] = 90 - results_df_baillard_phi['phi']

    # Handle wrapping: keep values in -90° to +90° range
    # If result > 90°, subtract 180°
    # If result < -90°, add 180°
    results_df_baillard_phi['phi'] = results_df_baillard_phi['phi'].apply(
        lambda x: x - 180 if x > 90 else (x + 180 if x < -90 else x)
    )

    # Now plot on AXEC2 data

    fig, ax = plot_dt_timeseries_movehisto(
        results_df_baillard_phi, 
        station=str(station) + ", SWSPy: " + str(first_start) + "-"+ str(last_start) + "σ, " + str(first_end) + "-" + str(last_end) + " Tmid End",
        x_width=200,  # 600 seconds
        x_overlap=0.9,
        y_width=1,          # 1 sample bins (finer)
        y_overlap=0.9,
        flag_smooth=True,
        x_filter_per=5,     # Less smoothing in time (5%)
        y_filter_per=3,      # Less smoothing in dt (5%)
        figsize=(6,8)
    )
    plt.show()

    fig, ax = plot_phi_timeseries_movehisto(
        results_df_baillard_phi, 
        station=str(station) + ", SWSPy: " + str(first_start) + "-"+ str(last_start) + "σ, " + str(first_end) + "-" + str(last_end) + " Tmid End",
        x_width=200,  # 600 seconds
        x_overlap=0.9,
        y_width=180/40,          # 5 degree bins (finer)
        y_overlap=0.9,
        flag_smooth=True,
        x_filter_per=5,     # Less smoothing in time (5%)
        y_filter_per=3,      # Less smoothing in phi (5%)
        figsize=(6,8)
    )
    plt.show()

    return results_df


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

def smart_arange(x_start,x_end,x_step): 
    x_out=np.linspace(x_start,x_end,int(round((x_end-x_start)/x_step+1)))
    new_step=x_out[1]-x_out[0]
    return (x_out,new_step)


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