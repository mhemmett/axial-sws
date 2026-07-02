#!/usr/bin/env python3
"""
sws_tomography_equalN.py

Tomographic inversion with equal-N time periods — each bin contains
approximately the same number of events as the syn-eruption period
(~5,838 events), giving uniform statistical weight across time.

Pre-eruption (~41k events)  → ~7 equal-count bins
Syn-eruption (~5.8k events) → 1 bin (reference)
Post-eruption (~57k events) → ~10 equal-count bins
Total: ~18 periods

Outputs:
  sws_tomography_dt_equalN_phi15.pdf
  sws_tomography_phi_equalN_phi15.pdf
"""

import sys, warnings, time, os
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
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from pykonal_raytracer import BaillardRayTracer
from sws_forward_model import STATION_FILE, ll2xy

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
OUT_DIR      = BASE
# FIX 6: harmonized QC cuts — SINGLE place to change them. Same defaults across all
# four tomography scripts (was phi_error<=15 with no dt cut here).
PHI_ERR_MAX  = 10.0   # keep splits with phi_error < PHI_ERR_MAX  [deg]
DT_ERR_MAX   = 0.01   # keep splits with dt_error  < DT_ERR_MAX   [s]
LAMBDA_DAMP  = 0.20
LAMBDA_SMOOTH= 0.10
SMOOTH_RATIO = 0.5
LAMBDA_FIXED = LAMBDA_DAMP
# FIX 7: coverage mask threshold — require >= MIN_PATH_KM cumulative ray path length
# per displayed voxel (replaces the dimensionally-odd col_sq>=10*LAMBDA_DAMP test).
MIN_PATH_KM  = 0.5    # km

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))
ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')
STATIONS = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET = {'AXAS1':(-0.20,-0.55),'AXAS2':(-0.45,-0.55),
                'AXCC1':(-0.75,0.25),'AXEC1':(0.15,0.30),
                'AXEC2':(0.15,0.05),'AXEC3':(0.30,-0.28)}
KIDIWELA = [dict(x=7.57,y=4.55,label='S1'),dict(x=7.53,y=6.60,label='S2')]

# ── Voxel grid ─────────────────────────────────────────────────────────────────
X_START,X_END = 4.0,12.0; Y_START,Y_END = 0.0,12.0; Z_MAX = 4.0
VXY=0.30; VZ=0.25
xn=np.arange(X_START,X_END+VXY*.5,VXY); yn=np.arange(Y_START,Y_END+VXY*.5,VXY)
zn=np.arange(0.,Z_MAX+VZ*.5,VZ)
NX,NY,NZ=len(xn),len(yn),len(zn); J=NX*NY*NZ
def voxel_idx(ix,iy,iz): return ix*NY*NZ+iy*NZ+iz

DEPTH_SLICES = [0.3125,0.9375,1.5625,2.1875]; Z_HALF=0.3125
DEPTH_LABELS = ['0.0–0.625 km','0.625–1.25 km','1.25–1.875 km','1.875–2.5 km']
COUNT_MIN    = 8; GAUSS=1.0; N_RAY=50

def depth_slice_idx(z0):
    """FIX 3: select z-voxels whose CENTRE (zn[k]+VZ/2) lies in [z0-Z_HALF, z0+Z_HALF)."""
    centres=zn+VZ/2.
    sel=np.where((centres>=z0-Z_HALF)&(centres<z0+Z_HALF))[0]
    if len(sel)==0:
        k=int(np.argmin(np.abs(centres-z0))); return k,k+1
    return int(sel[0]),int(sel[-1])+1

# ── Laplacian + regularisation ─────────────────────────────────────────────────
print('Building Laplacian...')
rows_,cols_,vals_=[],[],[]
for ix in range(NX):
    for iy in range(NY):
        for iz in range(NZ):
            j=voxel_idx(ix,iy,iz); nb=0
            for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                ni,nj,nk=ix+di,iy+dj,iz+dk
                if 0<=ni<NX and 0<=nj<NY and 0<=nk<NZ:
                    rows_.append(j); cols_.append(voxel_idx(ni,nj,nk))
                    vals_.append(1.); nb+=1
            rows_.append(j); cols_.append(j); vals_.append(-nb)
LAP=sp.csr_matrix((vals_,(rows_,cols_)),shape=(J,J))
Id2J=sp.eye(2*J); LAP2=sp.block_diag([LAP,LAP])
print(f'  {LAP.shape}, {LAP.nnz:,} non-zeros')

def build_augmented(A_sub):
    G=sp.bmat([[A_sub,None],[None,A_sub]],format='csr')
    return G,sp.vstack([G,np.sqrt(LAMBDA_DAMP)*Id2J,
                        np.sqrt(LAMBDA_SMOOTH)*LAP2],format='csr')

# ── Load data ──────────────────────────────────────────────────────────────────
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
_sta_df['x'],_sta_df['y']=ll2xy(_sta_df['lat'].values,_sta_df['lon'].values)
sta_xy={s:(float(_sta_df.loc[s,'x']),float(_sta_df.loc[s,'y'])) for s in STATIONS if s in _sta_df.index}

print(f'Loading events (phi_error < {PHI_ERR_MAX}°, dt_error < {DT_ERR_MAX}s)...')
dfs={}
for sta,(f1,f2) in FILES.items():
    def _load(f):
        d=pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
        return d[(d['dt']>0)&(d['phi_error']<PHI_ERR_MAX)&(d['dt_error']<DT_ERR_MAX)]
    df=pd.concat([_load(f1),_load(f2)],ignore_index=True)
    df['x'],df['y']=ll2xy(df['event_lat'].values,df['event_lon'].values)
    df['z']=df['event_depth'].values
    df['t']=pd.to_datetime(df['event_datetime'],utc=True)
    # FIX 1 (already correct here): phi is the swspy fast azimuth, deg CW from N;
    # only wrap into the 0–180° orientation half-circle.
    df['phi_az']=df['phi']%180.
    dfs[sta]=df; print(f'  {sta}: {len(df):,}')
all_df=pd.concat(dfs.values(),ignore_index=True)
print(f'  TOTAL retained: {len(all_df):,}')   # FIX 6: report total N after QC

# ── Build equal-N time periods ─────────────────────────────────────────────────
syn_N = len(all_df[(all_df['t']>=ERUPTION_START)&(all_df['t']<ERUPTION_END)])
print(f'\nSyn-eruption N={syn_N:,} → target bin size')

def equal_N_periods(df_all, target_N):
    """Split pre and post eruption into equal-N bins matching target_N."""
    periods = []

    # Pre-eruption: split into equal-N bins
    pre = df_all[df_all['t']<ERUPTION_START].sort_values('t')
    n_pre_bins = max(1, round(len(pre)/target_N))
    pre_bounds = [None] + [pre['t'].iloc[min(int(round(i*len(pre)/n_pre_bins)),len(pre)-1)]
                           for i in range(1,n_pre_bins)] + [ERUPTION_START]
    for i in range(n_pre_bins):
        t0,t1=pre_bounds[i],pre_bounds[i+1]
        sub=pre if t0 is None else pre[pre['t']>=t0]
        if t1 is not None: sub=sub[sub['t']<t1]
        lbl=f'Pre {i+1}/{n_pre_bins}'
        if t0 is not None: lbl=t0.strftime('%b%Y')+'–'+t1.strftime('%b%Y')
        else: lbl='Pre–'+t1.strftime('%b%Y')
        periods.append((lbl,t0,t1))

    # Syn-eruption: one bin
    periods.append(('Syn-eruption',ERUPTION_START,ERUPTION_END))

    # Post-eruption: split into equal-N bins
    post=df_all[df_all['t']>=ERUPTION_END].sort_values('t')
    n_post_bins=max(1,round(len(post)/target_N))
    post_bounds=[ERUPTION_END]+[post['t'].iloc[min(int(round(i*len(post)/n_post_bins)),len(post)-1)]
                                for i in range(1,n_post_bins)]+[None]
    def fmt(ts): return ts.strftime('%b%Y') if ts else 'present'
    for i in range(n_post_bins):
        t0,t1=post_bounds[i],post_bounds[i+1]
        periods.append((f'{fmt(t0)}–{fmt(t1)}',t0,t1))

    return periods

periods=equal_N_periods(all_df, syn_N)
print(f'Total periods: {len(periods)}')
for lbl,t0,t1 in periods:
    sub=all_df
    if t0 is not None: sub=sub[sub['t']>=t0]
    if t1 is not None: sub=sub[sub['t']<t1]
    print(f'  {lbl:25s}: {len(sub):,} events')

# ── FMM ray tracer (pre-compute once) ──────────────────────────────────────────
print('\nInitialising PyKonal FMM...')
tracer=BaillardRayTracer(stride=5)
for sta in STATIONS:
    if sta in sta_xy: tracer.precompute_station(sta,*sta_xy[sta])

# ── Build global A matrix (all events) ────────────────────────────────────────
print(f'\nTracing all {len(all_df):,} events...')
t0_rt=time.time()
coo_rows,coo_cols,coo_vals=[],[],[]
ray_dt,ray_phi,ray_t=[],[],[]
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
        cols,vals=tracer.ray_to_voxels(ray,xn,yn,zn)
        if len(cols)==0: done+=1; continue
        coo_rows.extend([row_idx]*len(cols))
        coo_cols.extend(cols.tolist()); coo_vals.extend(vals.tolist())
        ray_dt.append(float(row['dt'])); ray_phi.append(float(row['phi_az']))
        ray_t.append(row['t']); row_idx+=1; done+=1
        if done%10000==0:
            el=time.time()-t0_rt
            print(f'  {done:,}/{total:,}  {el:.0f}s  ~{el/done*(total-done):.0f}s remaining',
                  end='\r',flush=True)
N_RAYS=row_idx
print(f'\nDone in {time.time()-t0_rt:.0f}s  —  {N_RAYS:,} rays, {len(coo_vals):,} non-zeros')
print(f'  Events skipped due to trace failure: {trace_fail:,}')
A=sp.csr_matrix((coo_vals,(coo_rows,coo_cols)),shape=(N_RAYS,J))
ray_dt=np.array(ray_dt); ray_phi=np.array(ray_phi); ray_t=np.array(ray_t)
phi_r=np.radians(2.*ray_phi)
c_data=ray_dt*np.cos(phi_r); s_data=ray_dt*np.sin(phi_r)

# ── Invert each period ─────────────────────────────────────────────────────────
def solve_period(mask,label):
    A_sub=A[mask]
    # FIX 7: path-length coverage mask (was col_sq>=10.*LAMBDA_DAMP).
    col_path_km=np.asarray(A_sub.sum(axis=0)).ravel()
    cov_mask=(col_path_km>=MIN_PATH_KM)
    n=int(mask.sum())
    if n<COUNT_MIN*5:
        print(f'  {label}: only {n} rays — skip')
        return np.zeros(J),np.zeros(J),cov_mask
    d=np.concatenate([c_data[mask],s_data[mask]])
    d_aug=np.concatenate([d,np.zeros(4*J)])
    G,G_aug=build_augmented(A_sub)
    res=spla.lsqr(G_aug,d_aug,iter_lim=500,show=False)
    x=res[0]
    print(f'  {label:28s}: {n:,} rays  misfit={res[3]:.3f}  covered={cov_mask.sum():,}')
    return x[:J],x[J:],cov_mask

print('\nRunning per-period inversions...')
results=[]
for lbl,t0,t1 in periods:
    mask=(ray_t>=t0) if t0 is not None else np.ones(N_RAYS,dtype=bool)
    if t1 is not None: mask=mask&(ray_t<t1)
    u,v,cov=solve_period(mask,lbl)
    m=np.sqrt(u**2+v**2).reshape(NX,NY,NZ)
    th=(0.5*np.degrees(np.arctan2(v,u))%180).reshape(NX,NY,NZ)
    cov3d=cov.reshape(NX,NY,NZ)
    m[~cov3d]=np.nan; th[~cov3d]=np.nan
    n_events=int(mask.sum())
    results.append(dict(label=lbl,n=n_events,m=m,theta=th,cov=cov3d))

# ── Bathymetry ─────────────────────────────────────────────────────────────────
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

# FIX 3: plot at voxel centres (sampler treats xn/yn as LEFT EDGES). Bathymetry and
# station markers stay in the absolute-km frame and are NOT shifted. 'ij' preserved.
Xg,Yg=np.meshgrid(xn+VXY/2., yn+VXY/2., indexing='ij')

# ── Figure builders ────────────────────────────────────────────────────────────
all_m=[r['m'][r['m']>0].ravel() for r in results if np.isfinite(r['m']).any()]
m_vmax=float(np.percentile(np.concatenate(all_m),99)) if all_m else 0.05
levels=np.linspace(0.,m_vmax,15)

def make_strength_page(title):
    n_per=len(results); n_dep=len(DEPTH_SLICES)
    panel_w=1.6 if n_per>12 else 2.0
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.3))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    cmap=plt.colormaps['Blues']; last_h=None
    for ri,(z0,zlbl) in enumerate(zip(DEPTH_SLICES,DEPTH_LABELS)):
        iz0,iz1=depth_slice_idx(z0)   # FIX 3: centre-based depth selection
        for ci,res in enumerate(results):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            sl=np.nanmean(res['m'][:,:,iz0:iz1],axis=2)
            msk=np.isfinite(sl)
            if msk.sum()>=COUNT_MIN:
                sm=gaussian_filter(np.nan_to_num(sl),sigma=GAUSS)
                mp=np.ma.masked_where(~msk,sm)
                if not mp.mask.all():
                    ax.contourf(Xg,Yg,mp,levels=levels,cmap=cmap,
                                vmin=0,vmax=m_vmax,extend='neither')
                    last_h=ScalarMappable(cmap=cmap,norm=Normalize(0,m_vmax))
                    last_h.set_array([])
            _sta(ax); _kid(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=3)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{res["label"]}\nN={res["n"]:,}',
                                    fontsize=5.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'{zlbl}',fontsize=6)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        cb=plt.colorbar(last_h,cax=cax,ticks=np.linspace(0,m_vmax,5),
                        format=FormatStrFormatter('%.3f'))
        cb.set_label('m [s/km]',fontsize=8); cb.ax.tick_params(labelsize=6)
    fig.suptitle(title,fontsize=9,fontweight='bold',y=0.97)
    return fig

def make_phi_page(title):
    n_per=len(results); n_dep=len(DEPTH_SLICES)
    panel_w=1.6 if n_per>12 else 2.0
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.3))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    cmap_phi=plt.colormaps['hsv_r']; STEP=3; SLEN=0.18; LW=0.7; last_h=None
    for ri,(z0,zlbl) in enumerate(zip(DEPTH_SLICES,DEPTH_LABELS)):
        iz0,iz1=depth_slice_idx(z0)   # FIX 3: centre-based depth selection
        for ci,res in enumerate(results):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            m_sl=np.nanmean(res['m'][:,:,iz0:iz1],axis=2)
            th_sl=np.nanmean(res['theta'][:,:,iz0:iz1],axis=2)
            msk=np.isfinite(m_sl)
            segs,colors=[],[]
            ny_,nx_=m_sl.shape
            for i in range(0,ny_,STEP):
                for j in range(0,nx_,STEP):
                    if not msk[i,j]: continue
                    r=np.radians(th_sl[i,j])
                    x0,y0=Xg[i,j],Yg[i,j]
                    dx=SLEN*np.sin(r); dy=SLEN*np.cos(r)
                    segs.append([(x0-dx,y0-dy),(x0+dx,y0+dy)])
                    colors.append(th_sl[i,j]/180.)
            if segs:
                lc=LineCollection(segs,colors=cmap_phi(colors),
                                  linewidths=LW,zorder=6,alpha=0.9)
                ax.add_collection(lc)
                last_h=ScalarMappable(cmap=cmap_phi,norm=Normalize(0,180))
                last_h.set_array([])
            _sta(ax); _kid(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=3)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{res["label"]}\nN={res["n"]:,}',
                                    fontsize=5.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel(f'{zlbl}',fontsize=6)
    if last_h:
        cax=fig.add_subplot(gs[:,n_per])
        cb=plt.colorbar(last_h,cax=cax,ticks=np.arange(0,181,30),
                        format=FormatStrFormatter('%.0f'))
        cb.set_label('θ [° from N]',fontsize=8); cb.ax.tick_params(labelsize=6)
    fig.suptitle(title,fontsize=9,fontweight='bold',y=0.97)
    return fig

# ── Write PDFs ─────────────────────────────────────────────────────────────────
tag=f'(φ_err≤{PHI_ERR_MAX}°, equal-N~{syn_N:,}, λ={LAMBDA_DAMP})'
OUT_DT =os.path.join(OUT_DIR,'sws_tomography_dt_equalN_phi15_CORRECTED.pdf')
OUT_PHI=os.path.join(OUT_DIR,'sws_tomography_phi_equalN_phi15_CORRECTED.pdf')

print(f'\nWriting {os.path.basename(OUT_DT)}...')
with PdfPages(OUT_DT) as pdf:
    for start in range(0,len(results),10):
        chunk=results[start:start+10]
        _orig=results[:]; results[:]=chunk
        fig=make_strength_page(f'SWS tomography — anisotropy m [s/km] — equal-N  {tag}')
        pdf.savefig(fig,dpi=250,bbox_inches='tight'); plt.close(fig)
        results[:]=_orig
print(f'Saved {OUT_DT}')

print(f'Writing {os.path.basename(OUT_PHI)}...')
with PdfPages(OUT_PHI) as pdf:
    for start in range(0,len(results),10):
        chunk=results[start:start+10]
        _orig=results[:]; results[:]=chunk
        fig=make_phi_page(f'SWS tomography — fast direction θ — equal-N  {tag}')
        pdf.savefig(fig,dpi=250,bbox_inches='tight'); plt.close(fig)
        results[:]=_orig
print(f'Saved {OUT_PHI}')
print('Done.')
