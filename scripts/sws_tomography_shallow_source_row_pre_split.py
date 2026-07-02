#!/usr/bin/env python3
"""
sws_tomography_shallow_source_row_pre_split.py

Same as sws_tomography_shallow_source_row.py (single-row shear anisotropy
dVs/Vs tomography at 1.25-1.75 km depth) but additionally splits the
pre-eruption period into 5 equal-count (by number of events) bins instead
of leaving it as one block. Syn-eruption and the 5 post-eruption bins are
unchanged, giving 11 columns total instead of 7.

Output:
  results/shear_wave_anisotropy_shallow_source_pre_split.pdf
"""

import sys, warnings, os, time
sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import tifffile
from PIL import Image as PILImage
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import matplotlib.lines as mlines

from pykonal_raytracer import BaillardRayTracer
from baillard_velocity import vs_at
from sws_forward_model import STATION_FILE, ll2xy

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
PHI_ERR_MAX  = 20.0
DT_ERR_MAX   = 0.04
LAMBDA_DAMP  = 0.20; LAMBDA_SMOOTH = 0.10
MIN_PATH_KM  = 0.5
ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')
STATIONS    = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET = {'AXAS1':(-0.20,-0.55),'AXAS2':(-0.45,-0.55),
                'AXCC1':(-0.75,0.25),'AXEC1':(0.15,0.30),
                'AXEC2':(0.15,0.05),'AXEC3':(0.30,-0.28)}
INI_LON,INI_LAT=-130.1,45.9
KM_PER_DEG_LAT=111.32; KM_PER_DEG_LON=111.32*np.cos(np.radians(INI_LAT))

# ── Voxel grid ─────────────────────────────────────────────────────────────────
X_START,X_END=4.0,12.0; Y_START,Y_END=0.0,12.0; Z_MAX=4.0
VXY=0.30; VZ=0.25
xn=np.arange(X_START,X_END+VXY*.5,VXY); yn=np.arange(Y_START,Y_END+VXY*.5,VXY)
zn=np.arange(0.,Z_MAX+VZ*.5,VZ)
NX,NY,NZ=len(xn),len(yn),len(zn); J=NX*NY*NZ
def voxel_idx(ix,iy,iz): return ix*NY*NZ+iy*NZ+iz

# Single row: 1.25-1.75 km
DEPTH_SLICE=1.50; Z_HALF=0.25
DEPTH_LABEL='1.25–1.75 km'
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
    R = 2.0 / (np.maximum(m3d * vs_voxels, 1e-6))
    a = -R + np.sqrt(4.0 + R**2)
    return np.clip(a, 0., 2.)

# ── Laplacian ─────────────────────────────────────────────────────────────────
print('Building Laplacian...')
rows_,cols_,vals_=[],[],[]
for ix in range(NX):
 for iy in range(NY):
  for iz in range(NZ):
   j=voxel_idx(ix,iy,iz); nb=0
   for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
    ni,nj,nk=ix+di,iy+dj,iz+dk
    if 0<=ni<NX and 0<=nj<NY and 0<=nk<NZ:
     rows_.append(j); cols_.append(voxel_idx(ni,nj,nk)); vals_.append(1.); nb+=1
   rows_.append(j); cols_.append(j); vals_.append(-nb)
LAP=sp.csr_matrix((vals_,(rows_,cols_)),shape=(J,J))
Id2J=sp.eye(2*J); LAP2=sp.block_diag([LAP,LAP])

# ── Load + FMM ────────────────────────────────────────────────────────────────
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

def build_11_periods(all_df):
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'

    pre=all_df[all_df['t']<ERUPTION_START].sort_values('t').reset_index(drop=True)
    n_pre=len(pre); pre_bounds=[None]
    for i in range(1,5): pre_bounds.append(pre['t'].iloc[min(int(round(i*n_pre/5)),n_pre-1)])
    pre_bounds.append(ERUPTION_START)

    post=all_df[all_df['t']>=ERUPTION_END].sort_values('t').reset_index(drop=True)
    n_post=len(post); post_bounds=[ERUPTION_END]
    for i in range(1,5): post_bounds.append(post['t'].iloc[min(int(round(i*n_post/5)),n_post-1)])
    post_bounds.append(None)

    pds=[]
    for i in range(5):
        t0,t1=pre_bounds[i],pre_bounds[i+1]
        lbl=f'{fmt(t0)}\n–{fmt(t1)}' if t0 is not None else f'Pre-eruption\n–{fmt(t1)}'
        pds.append((lbl,t0,t1))
    pds.append(('Syn-eruption',ERUPTION_START,ERUPTION_END))
    for i in range(5): pds.append((f'{fmt(post_bounds[i])}\n–{fmt(post_bounds[i+1])}',post_bounds[i],post_bounds[i+1]))
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

def solve_period(mask,label,count_min=COUNT_MIN,min_path_km=MIN_PATH_KM):
    A_sub=A[mask]
    col_path_km_p=np.asarray(A_sub.sum(axis=0)).ravel()
    cov_p=(col_path_km_p>=min_path_km).reshape(NX,NY,NZ)
    n=int(mask.sum())
    if n<count_min*5:
        nan3d=np.full((NX,NY,NZ),np.nan)
        return nan3d,cov_p
    d=np.concatenate([c_data[mask],s_data[mask],np.zeros(4*J)])
    G2=sp.bmat([[A_sub,None],[None,A_sub]],format='csr')
    G2_aug=sp.vstack([G2,np.sqrt(LAMBDA_DAMP)*Id2J,np.sqrt(LAMBDA_SMOOTH)*LAP2],format='csr')
    res=spla.lsqr(G2_aug,d,iter_lim=500,show=False)
    x=res[0]; u=x[:J]; v=x[J:]
    m3d=np.sqrt(u**2+v**2).reshape(NX,NY,NZ)
    a3d=m_to_dvs_over_vs(m3d)
    a3d[~cov_p]=np.nan
    print(f'  {label:28s}: {n:,} rays  covered={cov_p.sum():,}  '
          f'a range 0–{np.nanpercentile(a3d,99):.3f}')
    return a3d, cov_p

print('\nRunning inversions...')
periods_11=build_11_periods(all_df); results_11=[]
for lbl,t0,t1 in periods_11:
    m=(ray_t>=t0) if t0 is not None else np.ones(N_RAYS,dtype=bool)
    if t1 is not None: m=m&(ray_t<t1)
    a3d,cov=solve_period(m,lbl.replace('\n',' '))
    results_11.append(dict(label=lbl,n=int(m.sum()),a=a3d,cov=cov))

# ── Bathymetry ────────────────────────────────────────────────────────────────
BATHY=('/Users/mhemmett/Seismology/axial-splitting-ml/data/'
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
Xg,Yg=np.meshgrid(xn+VXY/2., yn+VXY/2., indexing='ij')

# ── Figure builder: single row, all 7 periods ──────────────────────────────────
def make_single_row_page(results,title,count_min=COUNT_MIN):
    all_a=[r['a'][r['a']>0].ravel() for r in results if np.isfinite(r['a']).any()]
    a_vmax=float(np.nanpercentile(np.concatenate(all_a),99)) if all_a else 0.3
    cmap=plt.colormaps['Blues']; levels=np.linspace(0.,a_vmax,15)
    n_per=len(results)
    iz0,iz1=depth_slice_idx(DEPTH_SLICE)
    panel_w=1.8 if n_per>7 else 2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,3.5))
    gs=GridSpec(1,n_per+1,width_ratios=[1]*n_per+[0.04],wspace=0.04)
    last_h=None
    for ci,res in enumerate(results):
        ax=fig.add_subplot(gs[0,ci]); _bathy(ax)
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
        _sta(ax)
        ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
        ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_title(f'{res["label"]}\nN={res["n"]:,}',fontsize=7.5,fontweight='bold',pad=1)
        if ci==0: ax.set_ylabel(f'{DEPTH_LABEL}',fontsize=7)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        cb=plt.colorbar(last_h,cax=cax,ticks=np.linspace(0,a_vmax,5),
                        format=FormatStrFormatter('%.2f'))
        cb.set_label('Shear anisotropy ΔVs/Vs',fontsize=9); cb.ax.tick_params(labelsize=7)
    legend_handles=[
        mlines.Line2D([],[],marker='^',color='w',mfc='#FFD700',mec='k',ms=6,label='OBS station'),
    ]
    fig.legend(handles=legend_handles,loc='lower center',ncol=3,
               fontsize=7,framealpha=0.9,bbox_to_anchor=(0.45,0.0))
    fig.suptitle(title,fontsize=10,fontweight='bold',y=1.02)
    fig.subplots_adjust(bottom=0.12)
    return fig

out_path=os.path.join(BASE,'shear_wave_anisotropy_shallow_source_pre_split.pdf')
print(f'\nWriting {os.path.basename(out_path)}...')
with PdfPages(out_path) as pdf:
    fig=make_single_row_page(results_11,'Shear Anisotropy at 1.25-1.75 km depth, Axial Seamount')
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
print(f'Saved {out_path}')
print('\nDone.')
