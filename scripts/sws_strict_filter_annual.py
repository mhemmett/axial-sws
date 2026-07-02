#!/usr/bin/env python3
"""
sws_strict_filter_annual.py

Generates ONLY these two PDFs with the strict filter
(dt_error < 0.1 s  AND phi_error <= 33.8°):

  sws_temporal_combined_dt_annual_kidiwela_source_filtered_strict_filter.pdf
  sws_temporal_combined_phi_annual_principal_stress_kidiwela_source_filtered_strict_filter.pdf

No other files are produced.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tifffile
from PIL import Image as PILImage
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT_DIR      = BASE
BATHY        = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
                'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')

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
    'AXAS1':(-0.20,-0.55),'AXAS2':(-0.45,-0.55),'AXCC1':(-0.75,0.25),
    'AXEC1':(0.15,0.30),'AXEC2':(0.15,0.05),'AXEC3':(0.30,-0.28),
}

X_START,X_END = 4.0,12.0; Y_START,Y_END = 0.0,12.0
STEP=0.1; DIS_LIM=0.3; NUM_LIM=100; COUNT_MIN=15
GAUSS_SIGMA=0.2/STEP; DEPTH_SLICES=[0.5,1.0,1.5]; Z_HALF_WIDTH=0.3
STICK_LEN=0.18; STICK_STEP=3; STICK_LW=0.7

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

# ── Strict filter ─────────────────────────────────────────────────────────────
DT_ERR_MAX  = 0.1
PHI_ERR_MAX = 33.8   # mean phi_error of standard filtered dataset

# ── Bathymetry ────────────────────────────────────────────────────────────────
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

# ── Time periods ──────────────────────────────────────────────────────────────
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
def xyzd2mesh(x,y,z,d_phi,d_dt,z0):
    mz=np.abs(z-z0)<=Z_HALF_WIDTH
    xs,ys,phi_s,dt_s=x[mz],y[mz],d_phi[mz],d_dt[mz]
    xn=np.arange(X_START,X_END+STEP*.5,STEP); yn=np.arange(Y_START,Y_END+STEP*.5,STEP)
    Xg,Yg=np.meshgrid(xn,yn); ny,nx=Xg.shape
    PHI=np.zeros((ny,nx)); DT=np.zeros((ny,nx)); CNT=np.zeros((ny,nx),dtype=int)
    if len(xs)<3: return Xg,Yg,PHI,DT,CNT
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
        DT.ravel()[k]=float(np.median(dt_s[sel]))
    PHI_sm=gaussian_filter(PHI,sigma=GAUSS_SIGMA)
    DT_sm =gaussian_filter(DT, sigma=GAUSS_SIGMA)
    return Xg,Yg,PHI_sm,DT_sm,CNT

# ── dt map page ───────────────────────────────────────────────────────────────
def make_dt_page(df_or_dict, periods, highlight, title, dt_vmax=None):
    cmap=plt.colormaps['Blues']
    vmin,vmax=0,dt_vmax or 0.3
    n_per=len(periods); n_dep=len(DEPTH_SLICES)
    if isinstance(df_or_dict,dict):
        subs=[pd.concat([subset(df,t0,t1) for df in df_or_dict.values()],ignore_index=True)
              for _,t0,t1 in periods]
    else:
        subs=[subset(df_or_dict,t0,t1) for _,t0,t1 in periods]
    n_eqs=[s['event_datetime'].nunique() for s in subs]
    panel_w=1.8 if n_per>7 else 2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.3))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    last_h=None
    for ri,z0 in enumerate(DEPTH_SLICES):
        for ci,((lbl,_,__),sub,neq) in enumerate(zip(periods,subs,n_eqs)):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            if len(sub)>=COUNT_MIN:
                Xg,Yg,PHI,DT,CNT=xyzd2mesh(sub['x'].values,sub['y'].values,
                                             sub['z'].values,sub['phi_az'].values,
                                             sub['dt'].values,z0)
                mask=CNT>=COUNT_MIN
                DT_plot=np.where(mask,DT,np.nan)
                im=ax.contourf(Xg,Yg,DT_plot,levels=np.linspace(vmin,vmax,15),
                               cmap=cmap,vmin=vmin,vmax=vmax,extend='neither')
                last_h=ScalarMappable(cmap=cmap,norm=Normalize(vmin=vmin,vmax=vmax))
                last_h.set_array([])
            _add_stations(ax,highlight); _add_kidiwela(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{lbl}\nN={neq:,}',fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'z={z0} km',fontsize=8)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        ticks=np.linspace(vmin,vmax,5)
        cb=plt.colorbar(last_h,cax=cax,ticks=ticks,format=FormatStrFormatter('%.2f'))
        cb.set_label('δt [s]',fontsize=9); cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.97)
    return fig

# ── phi principal stress page ─────────────────────────────────────────────────
def make_phi_page(df_or_dict, periods, highlight, title):
    cmap=plt.colormaps['hsv_r']
    vmin,vmax=0,180
    n_per=len(periods); n_dep=len(DEPTH_SLICES)
    if isinstance(df_or_dict,dict):
        subs=[pd.concat([subset(df,t0,t1) for df in df_or_dict.values()],ignore_index=True)
              for _,t0,t1 in periods]
    else:
        subs=[subset(df_or_dict,t0,t1) for _,t0,t1 in periods]
    n_eqs=[s['event_datetime'].nunique() for s in subs]
    panel_w=1.8 if n_per>7 else 2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.3))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    last_h=None
    for ri,z0 in enumerate(DEPTH_SLICES):
        for ci,((lbl,_,__),sub,neq) in enumerate(zip(periods,subs,n_eqs)):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            if len(sub)>=COUNT_MIN:
                Xg,Yg,PHI,DT,CNT=xyzd2mesh(sub['x'].values,sub['y'].values,
                                             sub['z'].values,sub['phi_az'].values,
                                             sub['dt'].values,z0)
                # Principal stress sticks coloured by phi
                segs,colors=[],[]
                ny,nx=PHI.shape
                for i in range(0,ny,STICK_STEP):
                    for j in range(0,nx,STICK_STEP):
                        if CNT[i,j]<COUNT_MIN: continue
                        phi_r=np.radians(PHI[i,j])
                        x0,y0=Xg[i,j],Yg[i,j]
                        dx=STICK_LEN*np.sin(phi_r); dy=STICK_LEN*np.cos(phi_r)
                        segs.append([(x0-dx,y0-dy),(x0+dx,y0+dy)])
                        colors.append(PHI[i,j]/180.)
                if segs:
                    lc=LineCollection(segs,colors=cmap(colors),
                                      linewidths=STICK_LW,zorder=6,alpha=0.9)
                    ax.add_collection(lc)
                last_h=ScalarMappable(cmap=cmap,norm=Normalize(vmin=vmin,vmax=vmax))
                last_h.set_array([])
            _add_stations(ax,highlight); _add_kidiwela(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{lbl}\nN={neq:,}',fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'z={z0} km',fontsize=8)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        ticks=np.arange(0,181,30)
        cb=plt.colorbar(last_h,cax=cax,ticks=ticks,format=FormatStrFormatter('%.0f'))
        cb.set_label('φ [° from N]',fontsize=9); cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.97)
    return fig

# ── Load data with strict filter ──────────────────────────────────────────────
print(f'Loading data (dt_error<{DT_ERR_MAX}s AND phi_error≤{PHI_ERR_MAX}°)...')
_sta=pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','elev_km','station'],
                 engine='python').set_index('station')
_sta=_sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'],_sta['y']=ll2xy(_sta['lat'].values,_sta['lon'].values)

dfs={}
for sta,(f1,f2) in FILES.items():
    def _load(f):
        d=pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
        d=d[d['dt']>0]
        d=d[d['dt_error']<DT_ERR_MAX]
        d=d[d['phi_error']<=PHI_ERR_MAX]
        return d
    df=pd.concat([_load(f1),_load(f2)],ignore_index=True)
    df['x'],df['y']=ll2xy(df['event_lat'].values,df['event_lon'].values)
    df['z']=df['event_depth'].values
    df['t']=pd.to_datetime(df['event_datetime'],utc=True)
    df['phi_az']=df['phi']%180.0
    dfs[sta]=df
    print(f'  {sta}: {len(df):,}')

all_df=pd.concat(dfs.values(),ignore_index=True)
dt_vmax_all=float(np.percentile(np.concatenate([df['dt'].values for df in dfs.values()]),99))
periods=build_annual_periods()
label=f'(dt_error<{DT_ERR_MAX}s, phi_error≤{PHI_ERR_MAX}°)'

# ── PDF 1: dt annual kidiwela strict filter ───────────────────────────────────
out1=os.path.join(OUT_DIR,
    'sws_temporal_combined_dt_annual_kidiwela_source_filtered_strict_filter.pdf')
print(f'\nWriting {os.path.basename(out1)}...')
with PdfPages(out1) as pdf:
    for sta in STATIONS:
        vmax=float(np.percentile(dfs[sta]['dt'].values,99))
        fig=make_dt_page(dfs[sta],periods,highlight=sta,
                         title=f'{sta} — δt {label}',dt_vmax=vmax)
        pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
        print(f'  {sta}')
    fig=make_dt_page(dfs,periods,highlight=None,
                     title=f'All stations — δt {label}',dt_vmax=dt_vmax_all)
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
    print('  All stations')
print(f'Saved {out1}')

# ── PDF 2: phi principal stress annual kidiwela strict filter ─────────────────
out2=os.path.join(OUT_DIR,
    'sws_temporal_combined_phi_annual_principal_stress_kidiwela_source_filtered_strict_filter.pdf')
print(f'\nWriting {os.path.basename(out2)}...')
with PdfPages(out2) as pdf:
    for sta in STATIONS:
        fig=make_phi_page(dfs[sta],periods,highlight=sta,
                          title=f'{sta} — φ principal stress {label}')
        pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
        print(f'  {sta}')
    fig=make_phi_page(dfs,periods,highlight=None,
                      title=f'All stations — φ principal stress {label}')
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
    print('  All stations')
print(f'Saved {out2}')
print('\nDone.')
