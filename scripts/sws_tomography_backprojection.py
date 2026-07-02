#!/usr/bin/env python3
"""
sws_tomography_backprojection.py

A DISTINCT alternative to the regularized linear inversion in
sws_tomography_fractional_anisotropy_ultra_strict.py.

Both scripts share the same forward model: anisotropy accumulates LINEARLY
along the ray path, so for ray i with path-length kernel A_ij [km] in voxel j,

    dt_i * cos(2*phi_i) = sum_j A_ij * u_j
    dt_i * sin(2*phi_i) = sum_j A_ij * v_j         (u_j, v_j = m_j*cos/sin(2*theta_j))

The _ultra_strict script solves this as a single GLOBALLY COUPLED, REGULARIZED
inverse problem (damped least-squares + Laplacian smoothing via lsqr) -- voxel
values there are influenced by every ray and by their neighbors.

This script instead uses SIMPLE BACK-PROJECTION: each ray's own implied
average (u,v) over its full path -- u_ray = dt*cos(2phi)/L_ray, v_ray =
dt*sin(2phi)/L_ray, where L_ray is the ray's total path length in the grid --
is path-length-weighted-averaged over just the rays that sample each voxel.
There is NO damping, NO smoothing, and NO coupling between voxels: a voxel's
value depends only on the rays that physically pass through it. This is
noisier in low-coverage voxels (no regularization to suppress that) but
introduces no inversion artifacts (damping bias toward zero, smoothing
leakage between neighbors) and needs only sparse matrix-vector products, not
an iterative solve.

ΔVs/Vs conversion identical to the regularized-inversion script:
    R = 2/(m·Vs), a = ΔVs/Vs = -R + sqrt(4+R^2), Vs from the Baillard model.

Outputs:
  sws_tomography_dt_7period_dt004_backprojection_CORRECTED.pdf
  sws_tomography_dt_annual_dt004_backprojection_CORRECTED.pdf
  sws_tomography_dt_hourly48_dt004_backprojection_CORRECTED.pdf
  sws_tomography_phi_7period_dt004_backprojection_CORRECTED_contourf.pdf
  sws_tomography_phi_annual_dt004_backprojection_CORRECTED_contourf.pdf
  sws_tomography_phi_7period_dt004_backprojection_CORRECTED_stickvar.pdf
  sws_tomography_phi_annual_dt004_backprojection_CORRECTED_stickvar.pdf
  sws_tomography_phi_hourly48_dt004_backprojection_CORRECTED_stickvar.pdf
"""

import sys, warnings, os, time
sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import tifffile, math
from PIL import Image as PILImage
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
import matplotlib.lines as mlines

from pykonal_raytracer import BaillardRayTracer
from baillard_velocity import vs_at
from sws_forward_model import STATION_FILE, ll2xy

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
# FIX 6: harmonized QC cuts -- same as the other tomography scripts.
PHI_ERR_MAX  = 20.0   # keep splits with phi_error < PHI_ERR_MAX  [deg]
DT_ERR_MAX   = 0.04   # keep splits with dt_error  < DT_ERR_MAX   [s]
# Coverage mask threshold -- require >= MIN_PATH_KM cumulative ray path length
# per displayed voxel.
MIN_PATH_KM  = 0.5    # km
ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')
STATIONS    = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET = {'AXAS1':(-0.20,-0.55),'AXAS2':(-0.45,-0.55),
                'AXCC1':(-0.75,0.25),'AXEC1':(0.15,0.30),
                'AXEC2':(0.15,0.05),'AXEC3':(0.30,-0.28)}
KIDIWELA   = [dict(x=7.57,y=4.55,label='S1'),dict(x=7.53,y=6.60,label='S2')]

INI_LON,INI_LAT=-130.1,45.9
KM_PER_DEG_LAT=111.32; KM_PER_DEG_LON=111.32*np.cos(np.radians(INI_LAT))

# ── Voxel grid (identical to the inversion scripts, for direct comparability) ──
X_START,X_END=4.0,12.0; Y_START,Y_END=0.0,12.0; Z_MAX=4.0
VXY=0.30; VZ=0.25
xn=np.arange(X_START,X_END+VXY*.5,VXY); yn=np.arange(Y_START,Y_END+VXY*.5,VXY)
zn=np.arange(0.,Z_MAX+VZ*.5,VZ)
NX,NY,NZ=len(xn),len(yn),len(zn); J=NX*NY*NZ
def voxel_idx(ix,iy,iz): return ix*NY*NZ+iy*NZ+iz

DEPTH_SLICES=[0.3125,0.9375,1.5625,2.1875]; Z_HALF=0.3125
DEPTH_LABELS=['0.0–0.625 km','0.625–1.25 km','1.25–1.875 km','1.875–2.5 km']
COUNT_MIN=8; N_RAY=50

def depth_slice_idx(z0):
    centres=zn+VZ/2.
    sel=np.where((centres>=z0-Z_HALF)&(centres<z0+Z_HALF))[0]
    if len(sel)==0:
        k=int(np.argmin(np.abs(centres-z0))); return k,k+1
    return int(sel[0]),int(sel[-1])+1

# ── Pre-compute mean Vs per voxel from Baillard model ─────────────────────────
print('Querying Baillard Vs at each voxel centre...')
X3,Y3,Z3=np.meshgrid(xn,yn,zn,indexing='ij')
vs_voxels=vs_at(X3.ravel(),Y3.ravel(),Z3.ravel()).reshape(NX,NY,NZ)
vs_voxels=np.clip(vs_voxels,0.3,5.0)
print(f'  Vs range: {vs_voxels.min():.3f}–{vs_voxels.max():.3f} km/s')

def m_to_dvs_over_vs(m3d):
    """Same conversion as sws_tomography_fractional_anisotropy_ultra_strict.py:
    R = 2/(m·Vs), a = -R + sqrt(4+R^2), positive root, a in (0,2)."""
    R = 2.0 / (np.maximum(m3d * vs_voxels, 1e-6))
    a = -R + np.sqrt(4.0 + R**2)
    return np.clip(a, 0., 2.)

# ── Load + FMM (identical to the inversion scripts) ───────────────────────────
FILES={
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv','splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv','splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv','splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv','splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv','splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv','splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}
_sta_df=pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','e','s'],
                   engine='python').set_index('s')
_sta_df=_sta_df.loc[[s for s in STATIONS if s in _sta_df.index]]
_sta_df['x'],_sta_df['y']=((np.asarray(_sta_df['lon'])-INI_LON)*KM_PER_DEG_LON,
                            (np.asarray(_sta_df['lat'])-INI_LAT)*KM_PER_DEG_LAT)
_sta=_sta_df
sta_xy={s:(float(_sta_df.loc[s,'x']),float(_sta_df.loc[s,'y'])) for s in STATIONS if s in _sta_df.index}

print(f'Loading events (phi_error < {PHI_ERR_MAX}°, dt_error < {DT_ERR_MAX}s)...')
dfs={}
for sta,(f1,f2) in FILES.items():
    def _load(f):
        d=pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
        return d[(d['dt']>0)&(d['phi_error']<PHI_ERR_MAX)&(d['dt_error']<DT_ERR_MAX)]
    df=pd.concat([_load(f1),_load(f2)],ignore_index=True)
    df['x'],df['y']=((np.asarray(df['event_lon'])-INI_LON)*KM_PER_DEG_LON,
                     (np.asarray(df['event_lat'])-INI_LAT)*KM_PER_DEG_LAT)
    df['z']=df['event_depth'].values
    df['t']=pd.to_datetime(df['event_datetime'],utc=True)
    df['phi_az']=df['phi']%180.; dfs[sta]=df; print(f'  {sta}: {len(df):,}')
all_df=pd.concat(dfs.values(),ignore_index=True)
print(f'  TOTAL retained: {len(all_df):,}')

def build_7_periods(all_df):
    post=all_df[all_df['t']>=ERUPTION_END].sort_values('t').reset_index(drop=True)
    n=len(post); bounds=[ERUPTION_END]
    for i in range(1,5): bounds.append(post['t'].iloc[min(int(round(i*n/5)),n-1)])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds=[('Pre-eruption',None,ERUPTION_START),('Syn-eruption',ERUPTION_START,ERUPTION_END)]
    for i in range(5): pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}',bounds[i],bounds[i+1]))
    return pds

def build_annual():
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

def build_hourly_periods():
    """48 one-hour bins: 24h before eruption onset through the first 24h of
    the eruption (ERUPTION_START ± 24h)."""
    t_lo=ERUPTION_START-pd.Timedelta(hours=24)
    pds=[]
    for i in range(48):
        t0=t_lo+pd.Timedelta(hours=i); t1=t0+pd.Timedelta(hours=1)
        rel=i-24
        lbl=f'H{rel:+03d}\n{t0.strftime("%m/%d %H:%M")}'
        pds.append((lbl,t0,t1))
    return pds

tracer=BaillardRayTracer(stride=5)
for sta in STATIONS:
    if sta in sta_xy: tracer.precompute_station(sta,*sta_xy[sta])

print('\nBuilding global A matrix...')
t0_rt=time.time()
coo_rows,coo_cols,coo_vals=[],[],[]; ray_dt,ray_phi,ray_t=[],[],[]
row_idx=0; done=0; trace_fail=0; total=sum(len(dfs[s]) for s in STATIONS if s in sta_xy)
for sta in STATIONS:
    if sta not in sta_xy: continue
    for _,row in dfs[sta].iterrows():
        ex,ey,ez=float(row['x']),float(row['y']),float(row['z'])
        if ez<0 or ez>Z_MAX: done+=1; continue
        try:
            ray=tracer.trace(sta,ex,ey,ez,n_pts=N_RAY)
        except RuntimeError:
            trace_fail+=1; done+=1; continue
        vcols,vvals=tracer.ray_to_voxels(ray,xn,yn,zn)
        if len(vcols)==0: done+=1; continue
        coo_rows.extend([row_idx]*len(vcols)); coo_cols.extend(vcols.tolist())
        coo_vals.extend(vvals.tolist())
        ray_dt.append(float(row['dt'])); ray_phi.append(float(row['phi_az']))
        ray_t.append(row['t']); row_idx+=1; done+=1
        if done%10000==0:
            el=time.time()-t0_rt
            print(f'  {done:,}/{total:,}  {el:.0f}s',end='\r',flush=True)
N_RAYS=row_idx
print(f'\nDone in {time.time()-t0_rt:.0f}s  {N_RAYS:,} rays')
print(f'  Events skipped due to trace failure: {trace_fail:,}')
A=sp.csr_matrix((coo_vals,(coo_rows,coo_cols)),shape=(N_RAYS,J))
ray_dt=np.array(ray_dt); ray_phi=np.array(ray_phi); ray_t=np.array(ray_t)
phi_r=np.radians(2.*ray_phi); c_data=ray_dt*np.cos(phi_r); s_data=ray_dt*np.sin(phi_r)

# Per-ray total path length in the grid (fixed, independent of time period) --
# this is L_ray in u_ray = dt*cos(2phi)/L_ray, the ray's own implied AVERAGE
# anisotropy density under the linear-accumulation assumption.
L_ray=np.asarray(A.sum(axis=1)).ravel()
L_ray=np.maximum(L_ray,1e-9)
u_ray=c_data/L_ray; v_ray=s_data/L_ray

# Coverage thresholds used ONLY for the 48x1hr eruption-window periods.
COUNT_MIN_HOURLY=4; MIN_PATH_KM_HOURLY=1e-9; HOURLY_STD_MIN=2

def voxel_circ_std(A_sub, phi_sub, count_min=COUNT_MIN):
    """Per-voxel circular std-dev [deg, half-circle] of the RAW ray phi values
    that touch that voxel (unweighted), independent of the back-projected
    theta field. NaN where fewer than count_min rays touch the voxel."""
    A_csc=A_sub.tocsc()
    phi2=np.radians(2.*phi_sub); cosp=np.cos(phi2); sinp=np.sin(phi2)
    indptr,indices=A_csc.indptr,A_csc.indices
    std_flat=np.full(J,np.nan)
    for j in range(J):
        i0,i1=indptr[j],indptr[j+1]
        if i1-i0<count_min: continue
        rows=indices[i0:i1]
        Rbar=np.hypot(cosp[rows].mean(),sinp[rows].mean())
        Rbar=min(Rbar,1.-1e-9)
        std_flat[j]=np.degrees(np.sqrt(-2.*np.log(Rbar)))/2.
    return std_flat.reshape(NX,NY,NZ)

def solve_period(mask,label,count_min=COUNT_MIN,min_path_km=MIN_PATH_KM,std_count_min=None):
    """SIMPLE BACK-PROJECTION (no damping, no smoothing, no inter-voxel
    coupling): each voxel's (U,V) is the path-length-weighted mean of the
    sampling rays' own (u_ray,v_ray) -- i.e. each ray's average anisotropy
    over its full path, averaged over just the rays that touch that voxel."""
    if std_count_min is None: std_count_min=count_min
    A_sub=A[mask]
    col_path_km_p=np.asarray(A_sub.sum(axis=0)).ravel()
    cov_p=(col_path_km_p>=min_path_km).reshape(NX,NY,NZ)
    n=int(mask.sum())
    if n<count_min*5:
        nan3d=np.full((NX,NY,NZ),np.nan)
        return nan3d,nan3d.copy(),nan3d.copy(),cov_p
    numer_u=A_sub.T.dot(u_ray[mask])
    numer_v=A_sub.T.dot(v_ray[mask])
    denom=np.maximum(col_path_km_p,1e-9)
    U=(numer_u/denom).reshape(NX,NY,NZ)
    V=(numer_v/denom).reshape(NX,NY,NZ)
    U[~cov_p]=np.nan; V[~cov_p]=np.nan
    m3d=np.sqrt(U**2+V**2)
    a3d=m_to_dvs_over_vs(m3d)
    a3d[~cov_p]=np.nan
    theta=(0.5*np.degrees(np.arctan2(V,U))%180.)
    theta[~cov_p]=np.nan
    std3d=voxel_circ_std(A_sub,ray_phi[mask],count_min=std_count_min)
    std3d[~cov_p]=np.nan
    print(f'  {label:28s}: {n:,} rays  covered={cov_p.sum():,}  '
          f'a range 0–{np.nanpercentile(a3d,99):.3f}')
    return a3d, theta, std3d, cov_p

print('\nRunning back-projections...')
periods_7=build_7_periods(all_df); results_7=[]
for lbl,t0,t1 in periods_7:
    m=(ray_t>=t0) if t0 is not None else np.ones(N_RAYS,dtype=bool)
    if t1 is not None: m=m&(ray_t<t1)
    a3d,th3d,std3d,cov=solve_period(m,lbl.replace('\n',' '))
    results_7.append(dict(label=lbl,n=int(m.sum()),a=a3d,theta=th3d,std=std3d,cov=cov))

periods_a=build_annual(); results_a=[]
for lbl,t0,t1 in periods_a:
    m=(ray_t>=t0) if t0 is not None else np.ones(N_RAYS,dtype=bool)
    if t1 is not None: m=m&(ray_t<t1)
    a3d,th3d,std3d,cov=solve_period(m,lbl.replace('\n',' '))
    results_a.append(dict(label=lbl,n=int(m.sum()),a=a3d,theta=th3d,std=std3d,cov=cov))

periods_h=build_hourly_periods(); results_h=[]
for lbl,t0,t1 in periods_h:
    m=(ray_t>=t0)&(ray_t<t1)
    a3d,th3d,std3d,cov=solve_period(m,lbl.replace('\n',' '),
                                     count_min=COUNT_MIN_HOURLY,min_path_km=MIN_PATH_KM_HOURLY,
                                     std_count_min=HOURLY_STD_MIN)
    results_h.append(dict(label=lbl,n=int(m.sum()),a=a3d,theta=th3d,std=std3d,cov=cov))

# ── Bathymetry ────────────────────────────────────────────────────────────────
BATHY=('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
       'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS=None
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

def _bathy(ax): ax.imshow(_gray,origin='upper',extent=_ext,aspect='auto',cmap='gray',zorder=0)
def _sta(ax):
    for sta,row in _sta_df.iterrows():
        ax.plot(row['x'],row['y'],'^',ms=6,mfc='#FFD700',mec='k',mew=0.7,zorder=12)
        dx,dy=LABEL_OFFSET.get(sta,(0.12,0.12))
        ax.text(row['x']+dx,row['y']+dy,STA_DISPLAY.get(sta,sta),fontsize=5.5,zorder=13)
def _kid(ax):
    for s in KIDIWELA:
        ax.plot(s['x'],s['y'],'o',ms=4,mfc='red',mec='k',mew=0.5,zorder=14)
        ax.text(s['x']+0.12,s['y']+0.12,s['label'],fontsize=5,color='red',
                fontweight='bold',zorder=15)

COT66=math.cos(math.radians(66))/math.sin(math.radians(66)); DEPTH_TOP=0.1
strike_rad=math.radians(330.); L=8.
s_hat=np.array([math.sin(strike_rad),math.cos(strike_rad)])
def _faults(ax,z0):
    for cx,cy,dip_az in [(8.5,4.0,60.),(7.0,4.0,240.)]:
        horiz=(z0-DEPTH_TOP)*COT66; da=math.radians(dip_az)
        cx_z=cx+horiz*math.sin(da); cy_z=cy+horiz*math.cos(da)
        x1,y1=cx_z-L/2*s_hat[0],cy_z-L/2*s_hat[1]
        x2,y2=cx_z+L/2*s_hat[0],cy_z+L/2*s_hat[1]
        ax.plot([x1,x2],[y1,y2],'-',color='#cc0000',lw=0.9,alpha=0.75,zorder=11)

Xg,Yg=np.meshgrid(xn+VXY/2., yn+VXY/2., indexing='ij')

# ── Figure builders (same look-and-feel as the regularized-inversion script) ──
def make_page(results,title,count_min=COUNT_MIN,merge_depth=False):
    all_a=[r['a'][r['a']>0].ravel() for r in results if np.isfinite(r['a']).any()]
    a_vmax=float(np.nanpercentile(np.concatenate(all_a),99)) if all_a else 0.3
    cmap=plt.colormaps['Blues']; levels=np.linspace(0.,a_vmax,15)
    n_per=len(results)
    depth_rows=[(None,'All depths 0–4 km')] if merge_depth else list(zip(DEPTH_SLICES,DEPTH_LABELS))
    n_dep=len(depth_rows)
    panel_w=1.8 if n_per>7 else 2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.5))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    last_h=None
    for ri,(z0,zlbl) in enumerate(depth_rows):
        if z0 is None: iz0,iz1=0,NZ
        else: iz0,iz1=depth_slice_idx(z0)
        for ci,res in enumerate(results):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            sl=np.nanmean(res['a'][:,:,iz0:iz1],axis=2)
            msk=np.isfinite(sl)
            if msk.sum()>=count_min:
                sm=gaussian_filter(np.nan_to_num(sl),sigma=1.0)
                mp=np.ma.masked_where(~msk,sm)
                if not mp.mask.all():
                    ax.contourf(Xg,Yg,mp,levels=levels,cmap=cmap,
                                vmin=0,vmax=a_vmax,extend='neither')
                    last_h=ScalarMappable(cmap=cmap,norm=Normalize(0,a_vmax))
                    last_h.set_array([])
            _sta(ax); _kid(ax)
            if z0 is not None: _faults(ax,z0)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{res["label"]}\nN={res["n"]:,}',
                                    fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'{zlbl}',fontsize=7)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        cb=plt.colorbar(last_h,cax=cax,ticks=np.linspace(0,a_vmax,5),
                        format=FormatStrFormatter('%.2f'))
        cb.set_label('Shear anisotropy ΔVs/Vs (back-projection)',fontsize=9); cb.ax.tick_params(labelsize=7)
    legend_handles=[
        mlines.Line2D([],[],marker='o',color='w',mfc='red',mec='k',ms=5,label='Kidiwela'),
        mlines.Line2D([],[],marker='^',color='w',mfc='#FFD700',mec='k',ms=6,label='OBS station'),
        mlines.Line2D([],[],color='#cc0000',lw=1.2,label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles,loc='lower center',ncol=3,
               fontsize=7,framealpha=0.9,bbox_to_anchor=(0.45,0.0))
    fig.suptitle(title,fontsize=10,fontweight='bold',y=0.99)
    fig.subplots_adjust(bottom=0.06)
    return fig

def make_phi_contourf_page(results,title,count_min=COUNT_MIN):
    cmap=plt.colormaps['hsv_r']; levels=np.linspace(0.,180.,19)
    n_per=len(results); n_dep=len(DEPTH_SLICES)
    panel_w=1.8 if n_per>7 else 2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.5))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    last_h=None
    for ri,(z0,zlbl) in enumerate(zip(DEPTH_SLICES,DEPTH_LABELS)):
        iz0,iz1=depth_slice_idx(z0)
        for ci,res in enumerate(results):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            th_sl=np.nanmean(res['theta'][:,:,iz0:iz1],axis=2)
            msk=np.isfinite(th_sl)
            if msk.sum()>=count_min:
                c2=np.cos(np.radians(2.*np.nan_to_num(th_sl)))
                s2=np.sin(np.radians(2.*np.nan_to_num(th_sl)))
                c2=gaussian_filter(c2,sigma=1.0); s2=gaussian_filter(s2,sigma=1.0)
                th_sm=(0.5*np.degrees(np.arctan2(s2,c2)))%180.
                mp=np.ma.masked_where(~msk,th_sm)
                if not mp.mask.all():
                    ax.contourf(Xg,Yg,mp,levels=levels,cmap=cmap,
                                vmin=0,vmax=180,extend='neither')
                    last_h=ScalarMappable(cmap=cmap,norm=Normalize(0,180))
                    last_h.set_array([])
            _sta(ax); _kid(ax); _faults(ax,z0)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{res["label"]}\nN={res["n"]:,}',
                                    fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'{zlbl}',fontsize=7)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        cb=plt.colorbar(last_h,cax=cax,ticks=np.arange(0,181,30),
                        format=FormatStrFormatter('%.0f'))
        cb.set_label('φ [° from N]',fontsize=9); cb.ax.tick_params(labelsize=7)
    legend_handles=[
        mlines.Line2D([],[],marker='o',color='w',mfc='red',mec='k',ms=5,label='Kidiwela'),
        mlines.Line2D([],[],marker='^',color='w',mfc='#FFD700',mec='k',ms=6,label='OBS station'),
        mlines.Line2D([],[],color='#cc0000',lw=1.2,label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles,loc='lower center',ncol=3,
               fontsize=7,framealpha=0.9,bbox_to_anchor=(0.45,0.0))
    fig.suptitle(title,fontsize=10,fontweight='bold',y=0.99)
    fig.subplots_adjust(bottom=0.06)
    return fig

def make_phi_stickvar_page(results,title,std_vmax=None,merge_depth=False):
    cmap_std=plt.colormaps['viridis']
    STEP=1; SLEN=0.16; LW=0.6
    if std_vmax is None:
        all_std=[r['std'][np.isfinite(r['std'])].ravel() for r in results if np.isfinite(r['std']).any()]
        std_vmax=float(np.nanpercentile(np.concatenate(all_std),95)) if all_std else 30.
    n_per=len(results)
    depth_rows=[(None,'All depths 0–4 km')] if merge_depth else list(zip(DEPTH_SLICES,DEPTH_LABELS))
    n_dep=len(depth_rows)
    panel_w=1.8 if n_per>7 else 2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.5))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    last_h=None
    for ri,(z0,zlbl) in enumerate(depth_rows):
        if z0 is not None: iz0,iz1=depth_slice_idx(z0)
        for ci,res in enumerate(results):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            if z0 is None:
                c2=np.nanmean(np.cos(np.radians(2.*res['theta'])),axis=2)
                s2=np.nanmean(np.sin(np.radians(2.*res['theta'])),axis=2)
                th_sl=(0.5*np.degrees(np.arctan2(s2,c2)))%180.
                sd_sl=np.nanmean(res['std'],axis=2)
            else:
                th_sl=np.nanmean(res['theta'][:,:,iz0:iz1],axis=2)
                sd_sl=np.nanmean(res['std'][:,:,iz0:iz1],axis=2)
            msk=np.isfinite(th_sl)&np.isfinite(sd_sl)
            segs,colors=[],[]
            nx_,ny_=th_sl.shape
            for i in range(0,nx_,STEP):
                for j in range(0,ny_,STEP):
                    if not msk[i,j]: continue
                    r=np.radians(th_sl[i,j])
                    x0,y0=Xg[i,j],Yg[i,j]
                    dx=SLEN*np.sin(r); dy=SLEN*np.cos(r)
                    segs.append([(x0-dx,y0-dy),(x0+dx,y0+dy)])
                    colors.append(min(sd_sl[i,j]/std_vmax,1.0))
            if segs:
                lc=LineCollection(segs,colors=cmap_std(colors),
                                  linewidths=LW,zorder=6,alpha=0.9)
                ax.add_collection(lc)
                last_h=ScalarMappable(cmap=cmap_std,norm=Normalize(0,std_vmax))
                last_h.set_array([])
            _sta(ax); _kid(ax)
            if z0 is not None: _faults(ax,z0)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{res["label"]}\nN={res["n"]:,}',
                                    fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'{zlbl}',fontsize=7)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        cb=plt.colorbar(last_h,cax=cax,format=FormatStrFormatter('%.0f'))
        cb.set_label('circular std-dev of φ within voxel [°]',fontsize=9)
        cb.ax.tick_params(labelsize=7)
    legend_handles=[
        mlines.Line2D([],[],marker='o',color='w',mfc='red',mec='k',ms=5,label='Kidiwela'),
        mlines.Line2D([],[],marker='^',color='w',mfc='#FFD700',mec='k',ms=6,label='OBS station'),
        mlines.Line2D([],[],color='#cc0000',lw=1.2,label='Ring faults (depth-corrected)'),
    ]
    fig.legend(handles=legend_handles,loc='lower center',ncol=3,
               fontsize=7,framealpha=0.9,bbox_to_anchor=(0.45,0.0))
    fig.suptitle(title,fontsize=10,fontweight='bold',y=0.99)
    fig.subplots_adjust(bottom=0.06)
    return fig

def paginate_and_save(results_list,page_func,title,out_path,chunk=7,**kw):
    print(f'\nWriting {os.path.basename(out_path)}...')
    with PdfPages(out_path) as pdf:
        for start in range(0,len(results_list),chunk):
            fig=page_func(results_list[start:start+chunk],title,**kw)
            pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
    print(f'Saved {out_path}')

tag=f'(φ_err<{PHI_ERR_MAX}°, dt_err<{DT_ERR_MAX}s, SIMPLE BACK-PROJECTION -- no damping/smoothing, a=ΔVs/Vs=−R+√(4+R²), R=2/(m·Vs))'

out1=os.path.join(BASE,'sws_tomography_dt_7period_dt004_backprojection_CORRECTED.pdf')
print(f'\nWriting {os.path.basename(out1)}...')
with PdfPages(out1) as pdf:
    fig=make_page(results_7,f'SWS tomography — shear anisotropy ΔVs/Vs, back-projection (7 periods)  {tag}')
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
print(f'Saved {out1}')

out2=os.path.join(BASE,'sws_tomography_dt_annual_dt004_backprojection_CORRECTED.pdf')
print(f'Writing {os.path.basename(out2)}...')
with PdfPages(out2) as pdf:
    CHUNK=7
    for start in range(0,len(results_a),CHUNK):
        chunk=results_a[start:start+CHUNK]
        _orig=results_a[:]; results_a[:]=chunk
        fig=make_page(results_a,f'SWS tomography — shear anisotropy ΔVs/Vs, back-projection (annual)  {tag}')
        pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
        results_a[:]=_orig
print(f'Saved {out2}')

out3=os.path.join(BASE,'sws_tomography_phi_7period_dt004_backprojection_CORRECTED_contourf.pdf')
paginate_and_save(results_7,make_phi_contourf_page,
                   f'SWS tomography — fast direction φ, back-projection contourf (7 periods)  {tag}',out3,chunk=7)

out4=os.path.join(BASE,'sws_tomography_phi_annual_dt004_backprojection_CORRECTED_contourf.pdf')
paginate_and_save(results_a,make_phi_contourf_page,
                   f'SWS tomography — fast direction φ, back-projection contourf (annual)  {tag}',out4,chunk=7)

all_std_static=[r['std'][np.isfinite(r['std'])].ravel() for r in (results_7+results_a)
                 if np.isfinite(r['std']).any()]
std_vmax_static=float(np.nanpercentile(np.concatenate(all_std_static),95)) if all_std_static else 30.

out5=os.path.join(BASE,'sws_tomography_phi_7period_dt004_backprojection_CORRECTED_stickvar.pdf')
paginate_and_save(results_7,make_phi_stickvar_page,
                   f'SWS tomography — φ sticks colored by circular std-dev, back-projection (7 periods)  {tag}',
                   out5,chunk=7,std_vmax=std_vmax_static)

out6=os.path.join(BASE,'sws_tomography_phi_annual_dt004_backprojection_CORRECTED_stickvar.pdf')
paginate_and_save(results_a,make_phi_stickvar_page,
                   f'SWS tomography — φ sticks colored by circular std-dev, back-projection (annual)  {tag}',
                   out6,chunk=7,std_vmax=std_vmax_static)

out7=os.path.join(BASE,'sws_tomography_dt_hourly48_dt004_backprojection_CORRECTED.pdf')
paginate_and_save(results_h,make_page,
                   f'SWS tomography — shear anisotropy ΔVs/Vs, back-projection (hourly, eruption onset window, all depths)  {tag}',
                   out7,chunk=8,count_min=COUNT_MIN_HOURLY,merge_depth=True)

out8=os.path.join(BASE,'sws_tomography_phi_hourly48_dt004_backprojection_CORRECTED_stickvar.pdf')
all_std_h=[r['std'][np.isfinite(r['std'])].ravel() for r in results_h if np.isfinite(r['std']).any()]
std_vmax_h=float(np.nanpercentile(np.concatenate(all_std_h),95)) if all_std_h else std_vmax_static
paginate_and_save(results_h,make_phi_stickvar_page,
                   f'SWS tomography — φ sticks colored by circular std-dev, back-projection (hourly, eruption onset window, all depths)  {tag}',
                   out8,chunk=8,std_vmax=std_vmax_h,merge_depth=True)

print('\nDone.')
