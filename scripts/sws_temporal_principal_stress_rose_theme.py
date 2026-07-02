#!/usr/bin/env python3
"""
sws_temporal_principal_stress_rose_theme.py

Principal stress direction maps (phi + 90°) with Kidiwela source dots.
Sticks coloured by time period using the rose plot colour scheme
(purple → red → light-blue → purple gradient). No colorbar.
Source-binned only (no receiver_locations).

Naming convention: _principal_stress_kidiwela_source_rose_plot_theme

Produces (7-period and annual, filtered and unfiltered):
  sws_temporal_combined_phi_principal_stress_kidiwela_source_rose_plot_theme{suffix}.pdf
  sws_temporal_combined_phi_annual_principal_stress_kidiwela_source_rose_plot_theme{suffix}.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import tifffile
from PIL import Image as PILImage
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT_DIR      = BASE

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_PER_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_PER_DEG_LAT)

KIDIWELA_SOURCES = [
    dict(x=7.57, y=4.55, label='S1'),
    dict(x=7.53, y=6.60, label='S2'),
]

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

STATIONS    = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET_KM = {
    'AXAS1':(-0.20,-0.70),   # moved down
    'AXAS2':(-0.45,-0.70),   # moved down
    'AXCC1':(-0.75, 0.25),
    'AXEC1':( 0.15, 0.30),
    'AXEC2':( 0.15, 0.05),
    'AXEC3':( 0.30,-0.28),
}

X_START,X_END = 4.0,12.0; Y_START,Y_END = 0.0,12.0
STEP=0.1; DIS_LIM=0.3; NUM_LIM=100; COUNT_MIN=15
GAUSS_SIGMA=0.2/STEP; DEPTH_SLICES=[0.5,1.0,1.5]; Z_HALF_WIDTH=0.3
STICK_LEN=0.18; STICK_STEP=3; STICK_LW=0.8

FILES = {
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv',
             'splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv',
             'splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv',
             'splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv',
             'splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv',
             'splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv',
             'splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

# ── Rose plot colour scheme ───────────────────────────────────────────────────

def period_colors(n):
    """Same gradient as rose_plots_temporal.py: purple → red → blue → purple."""
    _start = np.array(mcolors.to_rgb('#ADD8E6'))   # light blue
    _end   = np.array(mcolors.to_rgb('#800080'))   # purple
    cols   = ['#800080', '#CC0000']                # pre, syn
    n_grad = max(n - 2, 1)
    for i in range(n_grad):
        t = i / max(n_grad - 1, 1)
        cols.append(mcolors.to_hex((1-t)*_start + t*_end))
    return cols[:n]

# ── Bathymetry ────────────────────────────────────────────────────────────────
BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
         'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS = None
_p=PILImage.open(BATHY); _t=_p.tag_v2
_olon,_olat=_t[33922][3],_t[33922][4]; _pl,_pb=_t[33550][0],_t[33550][1]
_nc,_nr=_p.size; _p.close()
_c0=max(0,int(((INI_LON+X_START/KM_PER_DEG_LON)-_olon)/_pl)-2)
_c1=min(_nc,int(((INI_LON+X_END/KM_PER_DEG_LON)-_olon)/_pl)+2)
_r0=max(0,int((_olat-(INI_LAT+Y_END/KM_PER_DEG_LAT))/_pb)-2)
_r1=min(_nr,int((_olat-(INI_LAT+Y_START/KM_PER_DEG_LAT))/_pb)+2)
_rgb=tifffile.imread(BATHY)[_r0:_r1,_c0:_c1]
_ds=max(1,max(_rgb.shape[:2])//1024); _rgb=_rgb[::_ds,::_ds]
_gray=np.dot(_rgb[...,:3].astype(np.float32),[0.299,0.587,0.114]).astype(np.uint8)
_ext=[(_olon+_c0*_pl-INI_LON)*KM_PER_DEG_LON,(_olon+_c1*_pl-INI_LON)*KM_PER_DEG_LON,
      (_olat-_r1*_pb-INI_LAT)*KM_PER_DEG_LAT,(_olat-_r0*_pb-INI_LAT)*KM_PER_DEG_LAT]

def _bathy(ax):
    ax.imshow(_gray,origin='upper',extent=_ext,aspect='auto',cmap='gray',zorder=0)

def _add_stations(ax, highlight=None):
    for sta,row in _sta.iterrows():
        mfc='#FFD700' if (highlight is None or sta==highlight) else 'white'
        ax.plot(row['x'],row['y'],'^',ms=6,mfc=mfc,mec='k',mew=0.7,zorder=12)
        dx,dy=LABEL_OFFSET_KM.get(sta,(0.12,0.12))
        ax.text(row['x']+dx,row['y']+dy,STA_DISPLAY.get(sta,sta),fontsize=5.5,zorder=13)

def _add_kidiwela(ax):
    for s in KIDIWELA_SOURCES:
        ax.plot(s['x'],s['y'],'o',ms=4,mfc='red',mec='k',mew=0.5,zorder=14)
        ax.text(s['x']+0.12,s['y']+0.12,s['label'],
                fontsize=5,color='red',fontweight='bold',zorder=15)

def _add_vectors(ax, Xg, Yg, PHI, CNT, color):
    """Sticks at phi+90° (principal stress), all one colour per time period."""
    segs = []
    ny, nx = PHI.shape
    for i in range(0, ny, STICK_STEP):
        for j in range(0, nx, STICK_STEP):
            if CNT[i,j] < COUNT_MIN: continue
            phi_rad = np.radians(PHI[i,j])
            x0, y0  = Xg[i,j], Yg[i,j]
            dx = STICK_LEN * np.sin(phi_rad)
            dy = STICK_LEN * np.cos(phi_rad)
            segs.append([(x0-dx, y0-dy), (x0+dx, y0+dy)])
    if segs:
        lc = LineCollection(segs, color=color,
                            linewidths=STICK_LW, zorder=6, alpha=0.9)
        ax.add_collection(lc)

# ── Time periods ──────────────────────────────────────────────────────────────

def build_7_periods(all_df):
    post=all_df[all_df['t']>=ERUPTION_END].sort_values('t').reset_index(drop=True)
    n=len(post); bounds=[ERUPTION_END]
    for i in range(1,5):
        idx=min(int(round(i*n/5)),n-1); bounds.append(post['t'].iloc[idx])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds=[('Pre-eruption',None,ERUPTION_START),('Syn-eruption',ERUPTION_START,ERUPTION_END)]
    for i in range(5): pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}',bounds[i],bounds[i+1]))
    return pds

def build_annual_periods():
    pds=[('Pre-eruption\n2015',None,ERUPTION_START),
         ('Syn-eruption\n2015',ERUPTION_START,ERUPTION_END),
         ('Post-eruption\n2015',ERUPTION_END,pd.Timestamp('2016-01-01',tz='UTC'))]
    for yr in range(2016,2027):
        t0=pd.Timestamp(f'{yr}-01-01',tz='UTC')
        t1=pd.Timestamp(f'{yr+1}-01-01',tz='UTC') if yr<2026 else None
        pds.append((str(yr),t0,t1))
    return pds

def subset(df,t0,t1):
    m=(df['t']>=t0) if t0 is not None else pd.Series(True,index=df.index)
    if t1 is not None: m=m&(df['t']<t1)
    return df[m]

# ── Binning ───────────────────────────────────────────────────────────────────

def xyzd2mesh(x,y,z,d_phi,z0):
    mz=np.abs(z-z0)<=Z_HALF_WIDTH
    xs,ys,phi_s=x[mz],y[mz],d_phi[mz]
    xn=np.arange(X_START,X_END+STEP*.5,STEP); yn=np.arange(Y_START,Y_END+STEP*.5,STEP)
    Xg,Yg=np.meshgrid(xn,yn); ny,nx=Xg.shape
    PHI=np.zeros((ny,nx)); CNT=np.zeros((ny,nx),dtype=int)
    if len(xs)<3: return Xg,Yg,PHI,CNT
    tree=cKDTree(np.column_stack([xs,ys]))
    gp=np.column_stack([Xg.ravel(),Yg.ravel()])
    nbrs=tree.query_ball_point(gp,DIS_LIM)
    for k,nb in enumerate(nbrs):
        if not nb: continue
        pts=np.column_stack([xs[nb],ys[nb]]); gpi=gp[k]
        dist=np.hypot(pts[:,0]-gpi[0],pts[:,1]-gpi[1])
        sel=np.array(nb)[np.argsort(dist)[:NUM_LIM]]
        CNT.ravel()[k]=len(sel)
        ang=2.0*np.radians(phi_s[sel])
        PHI.ravel()[k]=float(np.degrees(np.arctan2(np.mean(np.sin(ang)),
                                                    np.mean(np.cos(ang)))/2)%180)
    return Xg,Yg,PHI,CNT

# ── Figure builder ────────────────────────────────────────────────────────────

def make_page(df_or_dict, periods, highlight, title):
    colors = period_colors(len(periods))

    if isinstance(df_or_dict,dict):
        subs=[pd.concat([subset(df,t0,t1) for df in df_or_dict.values()],ignore_index=True)
              for _,t0,t1 in periods]
    else:
        subs=[subset(df_or_dict,t0,t1) for _,t0,t1 in periods]
    n_eqs=[s['event_datetime'].nunique() for s in subs]
    n_per=len(periods); n_dep=len(DEPTH_SLICES)

    panel_w=1.8 if n_per>7 else 2.2
    # No colorbar column
    fig=plt.figure(figsize=(n_per*panel_w+0.3, n_dep*3.0+0.3))
    gs=GridSpec(n_dep,n_per,hspace=0.04,wspace=0.04)

    for ri,z0 in enumerate(DEPTH_SLICES):
        for ci,((lbl,_,__),sub,neq,col) in enumerate(zip(periods,subs,n_eqs,colors)):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            if len(sub)>=COUNT_MIN:
                Xg,Yg,PHI,CNT=xyzd2mesh(
                    sub['x'].values,sub['y'].values,sub['z'].values,
                    sub['phi_az'].values,z0)
                _add_vectors(ax,Xg,Yg,PHI,CNT,color=col)
            else:
                ax.text(.5,.5,'–',ha='center',va='center',
                        transform=ax.transAxes,fontsize=8,color='gray')
            _add_stations(ax,highlight)
            _add_kidiwela(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{lbl}\nN={neq:,}',fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'z={z0} km',fontsize=8)

    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.97)
    return fig

# ── Run ───────────────────────────────────────────────────────────────────────

for dt_err_max,suffix in [(None,''),(0.1,'_filtered')]:
    label=f'(dt_error < {dt_err_max} s)' if dt_err_max else '(all data)'
    print(f'\nLoading {label}...')

    _sta=pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','elev_km','station'],
                     engine='python').set_index('station')
    _sta=_sta.loc[[s for s in STATIONS if s in _sta.index]]
    _sta['x'],_sta['y']=ll2xy(_sta['lat'].values,_sta['lon'].values)

    def _load(f):
        d=pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
        d=d[d['dt']>0]
        if dt_err_max: d=d[d['dt_error']<dt_err_max]
        return d

    dfs={}
    for sta,(f1,f2) in FILES.items():
        df=pd.concat([_load(f1),_load(f2)],ignore_index=True)
        df['x'],df['y']=ll2xy(df['event_lat'].values,df['event_lon'].values)
        df['z']=df['event_depth'].values
        df['t']=pd.to_datetime(df['event_datetime'],utc=True)
        df['phi_az']=df['phi'] % 180.0
        dfs[sta]=df
        print(f'  {sta}: {len(df):,} events')

    all_df=pd.concat(dfs.values(),ignore_index=True)

    for periods_fn,period_tag in [
        (lambda: build_7_periods(all_df), ''),
        (lambda: build_annual_periods(),  '_annual'),
    ]:
        periods=periods_fn()
        out=os.path.join(OUT_DIR,
            f'sws_temporal_combined_phi{period_tag}_principal_stress_kidiwela_source_rose_plot_theme{suffix}.pdf')
        with PdfPages(out) as pdf:
            for sta in STATIONS:
                fig=make_page(dfs[sta],periods,highlight=sta,
                              title=f'{sta} — σ₁ principal stress, rose plot theme {label}')
                pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
            fig=make_page(dfs,periods,highlight=None,
                          title=f'All stations — σ₁ principal stress, rose plot theme {label}')
            pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
        print(f'Saved {out}')

print('\nDone.')
