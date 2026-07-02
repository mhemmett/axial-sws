#!/usr/bin/env python3
"""
dt_norm_temporal.py

Ray-path-normalised delay time (dt / L) temporal figures.
dt/L in s/km separates genuine anisotropy from path-length accumulation.
Produces one combined PDF: 6 stations + all-stations on separate pages.
Runs both unfiltered and filtered (dt_error < 0.1 s) versions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.ma as ma
import os
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT_DIR      = BASE

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_PER_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_PER_DEG_LAT)

X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
STEP           = 0.1
DIS_LIM        = 0.3
NUM_LIM        = 100
COUNT_MIN      = 15
GAUSSIAN_SIGMA = 0.2 / STEP

DEPTH_SLICES = [0.5, 1.0, 1.5]
Z_HALF_WIDTH = 0.3

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}

LABEL_OFFSET_KM = {'AXAS1':(-0.20,-0.55),'AXAS2':(-0.45,-0.55),
                   'AXCC1':(-0.20, 0.25),'AXEC1':( 0.15, 0.30),
                   'AXEC2':( 0.15, 0.05),'AXEC3':( 0.30,-0.28)}

# ── AMC + ray paths ───────────────────────────────────────────────────────────
_AMC_C = np.array([8.0, 5.5, 2.0])
_AMC_R = 1.5

def _tangent(p1, p2):
    sl = float(np.linalg.norm(p2-p1))
    a2,b2,c2 = p1[:2],p2[:2],_AMC_C[:2]
    da,db = np.linalg.norm(a2-c2),np.linalg.norm(b2-c2)
    if da<=_AMC_R or db<=_AMC_R: return sl
    ta=np.sqrt(max(da**2-_AMC_R**2,0)); tb=np.sqrt(max(db**2-_AMC_R**2,0))
    ang=abs(np.arctan2(float(np.cross(a2-c2,b2-c2)),float(np.dot(a2-c2,b2-c2))))
    arc=_AMC_R*max(ang-np.arcsin(min(_AMC_R/da,1))-np.arcsin(min(_AMC_R/db,1)),0)
    s2=np.linalg.norm(b2-a2)
    return (ta+arc+tb)*(sl/s2) if s2>0 else sl

def ray_lengths(eq_x,eq_y,eq_z,sx,sy,sz=0.):
    p1=np.column_stack([eq_x,eq_y,eq_z]); p2=np.array([sx,sy,sz])
    d=p2-p1; f=p1-_AMC_C
    a=np.einsum('ij,ij->i',d,d); b=2*np.einsum('ij,ij->i',f,d)
    c_=np.einsum('ij,ij->i',f,f)-_AMC_R**2
    disc=np.clip(b**2-4*a*c_,0,None)
    t1=(-b-np.sqrt(disc))/(2*a+1e-30); t2=(-b+np.sqrt(disc))/(2*a+1e-30)
    hits=(disc>0)&((t1.clip(0)<=1)|(t2.clip(0)<=1))
    L=np.linalg.norm(d,axis=1)
    for i in np.where(hits)[0]: L[i]=_tangent(p1[i],p2)
    return L

# ── Spatial binning ───────────────────────────────────────────────────────────
def bin2d(x,y,z,d,x_start,x_end,y_start,y_end,z0,z_hw,step,dis_lim,num_lim):
    mz=np.abs(z-z0)<=z_hw
    xs,ys,ds=x[mz],y[mz],d[mz]
    xn=np.arange(x_start,x_end+step*.5,step); yn=np.arange(y_start,y_end+step*.5,step)
    X,Y=np.meshgrid(xn,yn); ny,nx=X.shape
    MED=np.zeros((ny,nx)); CNT=np.zeros((ny,nx),dtype=int)
    if len(xs)<3: return X,Y,MED,CNT
    tree=cKDTree(np.column_stack([xs,ys]))
    gp=np.column_stack([X.ravel(),Y.ravel()])
    nbrs=tree.query_ball_point(gp,dis_lim)
    for k,nb in enumerate(nbrs):
        if not nb: continue
        pts=np.column_stack([xs[nb],ys[nb]]); gpi=gp[k]
        dist=np.hypot(pts[:,0]-gpi[0],pts[:,1]-gpi[1])
        sel=np.array(nb)[np.argsort(dist)[:num_lim]]
        CNT.ravel()[k]=len(sel); MED.ravel()[k]=np.median(ds[sel])
    return X,Y,MED,CNT

# ── Bathymetry ────────────────────────────────────────────────────────────────
import tifffile
from PIL import Image as PILImage, ImageEnhance
BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
         'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS = None
_p=PILImage.open(BATHY); _t=_p.tag_v2
_olon,_olat=_t[33922][3],_t[33922][4]; _pl,_pb=_t[33550][0],_t[33550][1]
_nc,_nr=_p.size; _p.close()
_lon_min=INI_LON+X_START/KM_PER_DEG_LON; _lon_max=INI_LON+X_END/KM_PER_DEG_LON
_lat_min=INI_LAT+Y_START/KM_PER_DEG_LAT; _lat_max=INI_LAT+Y_END/KM_PER_DEG_LAT
_c0=max(0,int((_lon_min-_olon)/_pl)-2); _c1=min(_nc,int((_lon_max-_olon)/_pl)+2)
_r0=max(0,int((_olat-_lat_max)/_pb)-2); _r1=min(_nr,int((_olat-_lat_min)/_pb)+2)
_rgb=tifffile.imread(BATHY)[_r0:_r1,_c0:_c1]
_ds=max(1,max(_rgb.shape[:2])//1024); _rgb=_rgb[::_ds,::_ds]
_gray=np.dot(_rgb[...,:3].astype(np.float32),[0.299,0.587,0.114]).astype(np.uint8)
_ext=[(_olon+_c0*_pl-INI_LON)*KM_PER_DEG_LON,(_olon+_c1*_pl-INI_LON)*KM_PER_DEG_LON,
      (_olat-_r1*_pb-INI_LAT)*KM_PER_DEG_LAT,(_olat-_r0*_pb-INI_LAT)*KM_PER_DEG_LAT]

def _bathy(ax):
    ax.imshow(_gray,origin='upper',extent=_ext,aspect='auto',cmap='gray',zorder=0)

def _stations(ax,highlight=None):
    for sta,row in _sta.iterrows():
        mfc='#FFD700' if (highlight is None or sta==highlight) else 'white'
        ax.plot(row['x'],row['y'],'^',ms=6,mfc=mfc,mec='k',mew=0.7,zorder=12)
        dx,dy=LABEL_OFFSET_KM.get(sta,(0.12,0.12))
        ax.text(row['x']+dx,row['y']+dy,STA_DISPLAY.get(sta,sta),fontsize=5.5,zorder=13)

# ── Time periods ──────────────────────────────────────────────────────────────
def build_periods(all_df):
    post=all_df[all_df['t']>=ERUPTION_END].sort_values('t').reset_index(drop=True)
    n=len(post); bounds=[ERUPTION_END]
    for i in range(1,5):
        idx=min(int(round(i*n/5)),n-1); bounds.append(post['t'].iloc[idx])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds=[('Pre-eruption',None,ERUPTION_START),('Syn-eruption',ERUPTION_START,ERUPTION_END)]
    for i in range(5): pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}',bounds[i],bounds[i+1]))
    return pds

def subset(df,t0,t1):
    m=(df['t']>=t0) if t0 else pd.Series(True,index=df.index)
    if t1: m=m&(df['t']<t1)
    return df[m]

# ── Figure builder ────────────────────────────────────────────────────────────
def make_page(dfs_dict, periods, highlight, title, vmax, vstep):
    n_per=len(periods); n_dep=len(DEPTH_SLICES)
    cmap=plt.colormaps['Purples']; levels=np.linspace(0,vmax,15)
    fig=plt.figure(figsize=(n_per*2.2+0.6,n_dep*3.0+0.3))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    last_h=None
    # Pre-compute subsets
    subs=[]
    for _,t0,t1 in periods:
        if isinstance(dfs_dict,dict):
            frames=[subset(df,t0,t1) for df in dfs_dict.values()]
            subs.append(pd.concat(frames,ignore_index=True))
        else:
            subs.append(subset(dfs_dict,t0,t1))
    n_eqs=[s['event_datetime'].nunique() for s in subs]
    for ri,z0 in enumerate(DEPTH_SLICES):
        for ci,((lbl,_,__),sub,neq) in enumerate(zip(periods,subs,n_eqs)):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            if len(sub)>=COUNT_MIN:
                X,Y,MED,CNT=bin2d(sub['x'].values,sub['y'].values,sub['z'].values,
                                   sub['dt_norm'].values,
                                   X_START,X_END,Y_START,Y_END,
                                   z0,Z_HALF_WIDTH,STEP,DIS_LIM,NUM_LIM)
                D=gaussian_filter(MED,sigma=GAUSSIAN_SIGMA)
                D=ma.array(D,mask=CNT<COUNT_MIN)
                last_h=ax.contourf(X,Y,D,levels=levels,cmap=cmap,vmin=0,vmax=vmax,
                                   extend='neither',zorder=2,alpha=0.85)
            else:
                ax.text(.5,.5,'insufficient\ndata',ha='center',va='center',
                        transform=ax.transAxes,fontsize=6,color='gray')
            _stations(ax,highlight)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{lbl}\nN={neq:,}',fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'z={z0} km',fontsize=8)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        ticks=np.arange(0,vmax+vstep*.5,vstep)
        cb=plt.colorbar(last_h,cax=cax,ticks=ticks,format=FormatStrFormatter('%.4f'))
        cb.set_label('dt / L  [s km⁻¹]',fontsize=9); cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.97)
    return fig

# ── Run ───────────────────────────────────────────────────────────────────────
for dt_err_max, suffix in [(None,''), (0.1,'_filtered')]:
    label = f'(dt_error < {dt_err_max} s)' if dt_err_max else '(all data)'
    print(f'\n=== {label} ===')

    # Load data
    FILES = {
        'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv','splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
        'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv','splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
        'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv','splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
        'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv','splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
        'AXEC2':('axial-mldd-2015-2021-axec2.csv','splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
        'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv','splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
    }
    _sta=pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','elev_km','station'],
                     engine='python').set_index('station')
    _sta=_sta.loc[[s for s in STATIONS if s in _sta.index]]
    _sta['x'],_sta['y']=ll2xy(_sta['lat'].values,_sta['lon'].values)

    dfs={}
    for sta,(f1,f2) in FILES.items():
        df=pd.concat([pd.read_csv(BASE+f1),pd.read_csv(BASE+f2)]).loc[:,:'dt_error'].dropna()
        df=df[df['dt']>0]
        if dt_err_max: df=df[df['dt_error']<dt_err_max]
        df['x'],df['y']=ll2xy(df['event_lat'].values,df['event_lon'].values)
        df['z']=df['event_depth'].values
        df['t']=pd.to_datetime(df['event_datetime'],utc=True)
        srow=_sta.loc[sta]
        L=ray_lengths(df['x'].values,df['y'].values,df['z'].values,
                      float(srow['x']),float(srow['y']))
        df['ray_L_km']=L; df['dt_norm']=df['dt'].values/np.maximum(L,0.1)
        dfs[sta]=df
        print(f'  {sta}: {len(df):,} events  mean L={L.mean():.2f} km  '
              f'median dt/L={df["dt_norm"].median():.4f} s/km')

    all_df=pd.concat(dfs.values(),ignore_index=True)
    periods=build_periods(all_df)

    # Global vmax = 99th pct of all dt_norm
    vmax=float(np.percentile(all_df['dt_norm'].values,99))
    vstep=round(vmax/6,4)
    print(f'  dt/L vmax (99th pct): {vmax:.4f} s/km')

    out=os.path.join(OUT_DIR,f'sws_temporal_dt_norm{suffix}.pdf')
    with PdfPages(out) as pdf:
        for sta in STATIONS:
            fig=make_page(dfs[sta],periods,highlight=sta,
                          title=f'{sta} — median dt/L (s km⁻¹) {label}',
                          vmax=vmax,vstep=vstep)
            pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
            print(f'  Added {sta}')
        fig=make_page(dfs,periods,highlight=None,
                      title=f'All stations — median dt/L (s km⁻¹) {label}',
                      vmax=vmax,vstep=vstep)
        pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
        print(f'  Added ALL stations')
    print(f'Saved {out}')

print('\nDone.')
