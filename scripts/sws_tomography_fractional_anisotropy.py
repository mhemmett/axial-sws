#!/usr/bin/env python3
"""
sws_tomography_fractional_anisotropy.py

Same linearised tomographic inversion as sws_tomography_inversion.py but
displays results as SHEAR ANISOTROPY ΔVs/Vs (dimensionless) rather than
anisotropy strength m [s/km].

NOTE on terminology (FIX 2): the displayed quantity is NOT the Browaeys &
Chevrot (2004) elastic-tensor "fractional anisotropy" invariant. It is, to first
order, ΔVs/Vs ≈ (percent S anisotropy)/100. The filename is kept as-is to avoid
breaking downstream references, but every human-facing label says "shear
anisotropy ΔVs/Vs".

Citation: this is the splitting-intensity / vectorial parameterisation
(Chevrot 2000, 2006); ΔVs/Vs ≈ percent S anisotropy (Crampin & Peacock 2005).

Conversion uses the Baillard 3D Vs model for the local velocity v_j:

    R  =  2 / (m_j · v_j)          m_j [s/km], v_j [km/s]

    a_j = −R + √(4 + R²)            positive root only, a ∈ (0, 2)

This accounts for the velocity-dependence of the delay time:
  weak-anisotropy limit: a_j ≈ m_j · v_j  (= ΔVs/Vs)
  general: a_j derived from  δt·v/s = a/(1 − a²/4); a crosses 1 at m·Vs = 1.

Outputs:
  sws_tomography_dt_7period_phi15_fractional_anisotropy.pdf
  sws_tomography_dt_annual_phi15_fractional_anisotropy.pdf
"""

import sys, warnings, os, time
sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import tifffile, math
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
# FIX 6: harmonized QC cuts — SINGLE place to change them. Same defaults across all
# four tomography scripts (was phi_error<=15 with no dt cut here).
PHI_ERR_MAX  = 10.0   # keep splits with phi_error < PHI_ERR_MAX  [deg]
DT_ERR_MAX   = 0.01   # keep splits with dt_error  < DT_ERR_MAX   [s]
LAMBDA_DAMP  = 0.20; LAMBDA_SMOOTH = 0.10
# FIX 7: coverage mask threshold — require >= MIN_PATH_KM cumulative ray path length
# per displayed voxel (replaces col_sq>=10*LAMBDA_DAMP).
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

# ── Voxel grid ─────────────────────────────────────────────────────────────────
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
    """FIX 3: select z-voxels whose CENTRE (zn[k]+VZ/2) lies in [z0-Z_HALF, z0+Z_HALF)."""
    centres=zn+VZ/2.
    sel=np.where((centres>=z0-Z_HALF)&(centres<z0+Z_HALF))[0]
    if len(sel)==0:
        k=int(np.argmin(np.abs(centres-z0))); return k,k+1
    return int(sel[0]),int(sel[-1])+1

# ── Pre-compute mean Vs per voxel from Baillard model ─────────────────────────
print('Querying Baillard Vs at each voxel centre...')
X3,Y3,Z3=np.meshgrid(xn,yn,zn,indexing='ij')
# vs_at expects x=East, y=North, z=depth [km]
vs_voxels=vs_at(X3.ravel(),Y3.ravel(),Z3.ravel()).reshape(NX,NY,NZ)  # [km/s]
vs_voxels=np.clip(vs_voxels,0.3,5.0)
print(f'  Vs range: {vs_voxels.min():.3f}–{vs_voxels.max():.3f} km/s')

def m_to_dvs_over_vs(m3d):
    """Convert m_j [s/km] to shear anisotropy a_j = ΔVs/Vs using the Baillard Vs.

    R   = 2 / (m·Vs)
    a   = −R + √(4 + R²)     positive root.
    True range is a ∈ (0, 2); a crosses 1 at m·Vs = 1.333 (a(m·Vs=1)=0.828; NOT bounded by 1).
    In the weak-anisotropy limit a ≈ m·Vs = ΔVs/Vs.

    TODO (human decision — see ray_to_voxels in pykonal_raytracer.py): the inversion
    kernel is purely geometric (path length in km), so m_j is anisotropy per km and
    this conversion re-divides by Vs. Whether Vs is being used twice (kernel +
    conversion) is a modelling decision flagged for the human; do not silently change.
    """
    R = 2.0 / (np.maximum(m3d * vs_voxels, 1e-6))
    a = -R + np.sqrt(4.0 + R**2)   # positive root, true range a ∈ (0, 2)
    return np.clip(a, 0., 2.)      # FIX 2: was clip(0,1), which saturated valid voxels

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
_sta=_sta_df   # alias for iterrows in _sta()
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
    # FIX 1 (already correct here): phi is swspy fast azimuth, deg CW from N; wrap only.
    df['phi_az']=df['phi']%180.; dfs[sta]=df; print(f'  {sta}: {len(df):,}')
all_df=pd.concat(dfs.values(),ignore_index=True)
print(f'  TOTAL retained: {len(all_df):,}')   # FIX 6: report total N after QC

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
# FIX 7: path-length coverage mask (was col_sq>=10.*LAMBDA_DAMP).
col_path_km=np.asarray(A.sum(axis=0)).ravel()
cov_mask=(col_path_km>=MIN_PATH_KM).reshape(NX,NY,NZ)

G=sp.bmat([[A,None],[None,A]],format='csr')
G_aug=sp.vstack([G,np.sqrt(LAMBDA_DAMP)*Id2J,np.sqrt(LAMBDA_SMOOTH)*LAP2],format='csr')

def solve_period(mask,label):
    A_sub=A[mask]
    # FIX 7: path-length coverage mask (was col_sq_p>=10.*LAMBDA_DAMP).
    col_path_km_p=np.asarray(A_sub.sum(axis=0)).ravel()
    cov_p=(col_path_km_p>=MIN_PATH_KM).reshape(NX,NY,NZ)
    n=int(mask.sum())
    if n<COUNT_MIN*5: return np.full((NX,NY,NZ),np.nan),cov_p
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
periods_7=build_7_periods(all_df); results_7=[]
for lbl,t0,t1 in periods_7:
    m=(ray_t>=t0) if t0 is not None else np.ones(N_RAYS,dtype=bool)
    if t1 is not None: m=m&(ray_t<t1)
    a3d,cov=solve_period(m,lbl.replace('\n',' '))
    results_7.append(dict(label=lbl,n=int(m.sum()),a=a3d,cov=cov))

periods_a=build_annual(); results_a=[]
for lbl,t0,t1 in periods_a:
    m=(ray_t>=t0) if t0 is not None else np.ones(N_RAYS,dtype=bool)
    if t1 is not None: m=m&(ray_t<t1)
    a3d,cov=solve_period(m,lbl.replace('\n',' '))
    results_a.append(dict(label=lbl,n=int(m.sum()),a=a3d,cov=cov))

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

# FIX 3: plot at voxel centres (sampler treats xn/yn as LEFT EDGES). Bathymetry and
# station markers stay in the absolute-km frame and are NOT shifted. 'ij' preserved.
Xg,Yg=np.meshgrid(xn+VXY/2., yn+VXY/2., indexing='ij')

# ── Figure builder ────────────────────────────────────────────────────────────
def make_page(results,title):
    all_a=[r['a'][r['a']>0].ravel() for r in results if np.isfinite(r['a']).any()]
    a_vmax=float(np.nanpercentile(np.concatenate(all_a),99)) if all_a else 0.3
    cmap=plt.colormaps['Blues']; levels=np.linspace(0.,a_vmax,15)
    n_per=len(results); n_dep=len(DEPTH_SLICES)
    panel_w=1.8 if n_per>7 else 2.2
    fig=plt.figure(figsize=(n_per*panel_w+0.6,n_dep*3.0+0.5))
    gs=GridSpec(n_dep,n_per+1,width_ratios=[1]*n_per+[0.04],hspace=0.04,wspace=0.04)
    last_h=None
    for ri,(z0,zlbl) in enumerate(zip(DEPTH_SLICES,DEPTH_LABELS)):
        iz0,iz1=depth_slice_idx(z0)   # FIX 3: centre-based depth selection
        for ci,res in enumerate(results):
            ax=fig.add_subplot(gs[ri,ci]); _bathy(ax)
            sl=np.nanmean(res['a'][:,:,iz0:iz1],axis=2)
            msk=np.isfinite(sl)
            if msk.sum()>=COUNT_MIN:
                sm=gaussian_filter(np.nan_to_num(sl),sigma=1.0)
                mp=np.ma.masked_where(~msk,sm)
                if not mp.mask.all():
                    ax.contourf(Xg,Yg,mp,levels=levels,cmap=cmap,
                                vmin=0,vmax=a_vmax,extend='neither')
                    last_h=ScalarMappable(cmap=cmap,norm=Normalize(0,a_vmax))
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
        cb=plt.colorbar(last_h,cax=cax,ticks=np.linspace(0,a_vmax,5),
                        format=FormatStrFormatter('%.2f'))
        cb.set_label('Shear anisotropy ΔVs/Vs',fontsize=9); cb.ax.tick_params(labelsize=7)
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

tag=f'(φ_err<{PHI_ERR_MAX}°, dt_err<{DT_ERR_MAX}s, λ={LAMBDA_DAMP}, a=ΔVs/Vs=−R+√(4+R²), R=2/(m·Vs))'

# NOTE: filename keeps the legacy 'phi15' token to avoid breaking downstream refs,
# even though the harmonized cut (FIX 6) is now phi_error<10 & dt_error<0.01.
out1=os.path.join(BASE,'sws_tomography_dt_7period_phi15_fractional_anisotropy_CORRECTED.pdf')
print(f'\nWriting {os.path.basename(out1)}...')
with PdfPages(out1) as pdf:
    fig=make_page(results_7,f'SWS tomography — shear anisotropy ΔVs/Vs (7 periods)  {tag}')
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
print(f'Saved {out1}')

out2=os.path.join(BASE,'sws_tomography_dt_annual_phi15_fractional_anisotropy_CORRECTED.pdf')
print(f'Writing {os.path.basename(out2)}...')
with PdfPages(out2) as pdf:
    CHUNK=7
    for start in range(0,len(results_a),CHUNK):
        chunk=results_a[start:start+CHUNK]
        _orig=results_a[:]; results_a[:]=chunk
        fig=make_page(results_a,f'SWS tomography — shear anisotropy ΔVs/Vs (annual)  {tag}')
        pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
        results_a[:]=_orig
print(f'Saved {out2}')
print('\nDone.')
