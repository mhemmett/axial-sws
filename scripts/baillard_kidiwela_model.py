#!/usr/bin/env python3
"""
baillard_kidiwela_model.py

Baillard-style stress model using the two Kidiwela spherical Mogi sources
instead of the prolate spheroid. Produces:
  baillard_kidiwela_geometry_stress.pdf      — σ₁ field (pre + syn)
  baillard_kidiwela_geometry_stress_with_obs.pdf — model σ₁ + observed φ

Source geometry:
  S1: x=7.57, y=4.55 km, depth=3.33 km, R=0.43 km  (deep)
  S2: x=7.53, y=6.60 km, depth=1.25 km, R=0.20 km  (shallow)
  Dike: (8.29, 6.27)→(8.19, 8.20) km, az=353°, 2 m, 0–0.5 km depth

Approach (identical to Baillard 2019):
  surface stress → eigendecompose 2×2 horizontal tensor → σ₁ azimuth = φ_pred
"""

import sys, warnings
sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
import tifffile
from PIL import Image as PILImage
import os

from sws_forward_model import (
    STATION_FILE, ll2xy, build_interpolators, eval_stress_at,
    okada_dike_stress, MU_BG, NU_FIX,
    load_data, build_time_periods, build_observations,
    ERUPTION_START, ERUPTION_END,
)

# ── Constants ─────────────────────────────────────────────────────────────────
INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))
BASE    = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
OUT_DIR = BASE

STATIONS    = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET = {'AXAS1':(-0.20,-0.55),'AXAS2':(-0.45,-0.55),
                'AXCC1':(-0.75,0.25),'AXEC1':(0.15,0.30),
                'AXEC2':(0.15,0.05),'AXEC3':(0.30,-0.28)}

X_START, X_END = 4.0, 12.0; Y_START, Y_END = 0.0, 12.0
DIS_LIM=0.3; NUM_LIM=100; COUNT_MIN=15
STEP_OBS=0.10   # observation binning grid [km]
STEP_MOD=0.20   # model arrow grid [km]
ARROW_STEP=2    # subsampling on model grid

# ── Kidiwela Mogi sources ─────────────────────────────────────────────────────
SPHERES = [
    dict(x0=7.57, y0=4.55, d=3.33, R=0.43, label='S1'),
    dict(x0=7.53, y0=6.60, d=1.25, R=0.20, label='S2'),
]
DP_INFLATE =  0.05e9   # Pa  pre-eruption inflation
DP_DEFLATE = -0.05e9   # Pa  syn-eruption deflation

# ── Dike ──────────────────────────────────────────────────────────────────────
DIKE_START   = (8.29, 6.27)
DIKE_END     = (8.19, 8.20)
DIKE_AZ      = 353.
DIKE_DTOP    = 0.0    # km
DIKE_DBOT    = 0.5    # km
DIKE_OPENING = 2.0    # m

# ── Mogi stress (2D surface) ──────────────────────────────────────────────────
def mogi_surface_stress(X_km, Y_km, xn, yn, spheres, dP_Pa, mu=MU_BG, nu=NU_FIX):
    """Combined 2D surface stress from list of Mogi spheres at given ΔP."""
    lam = 2*nu*mu/(1-2*nu)
    Ux  = np.zeros_like(X_km, dtype=float)
    Uy  = np.zeros_like(X_km, dtype=float)
    for sph in spheres:
        dV = np.pi * (sph['R']*1e3)**3 * dP_Pa / mu
        C  = dV*(1-nu)/np.pi
        dx = (X_km - sph['x0'])*1e3
        dy = (Y_km - sph['y0'])*1e3
        d  = sph['d']*1e3
        R3 = (dx**2 + dy**2 + d**2)**1.5
        Ux += C*dx/R3
        Uy += C*dy/R3
    dm  = (xn[1]-xn[0])*1e3; dm2 = (yn[1]-yn[0])*1e3
    ny_, nx_ = Ux.shape
    sig = np.zeros((ny_, nx_, 3))
    e11 = (Ux[1:-1,2:]-Ux[1:-1,:-2])/(2*dm)
    e22 = (Uy[2:,1:-1]-Uy[:-2,1:-1])/(2*dm2)
    e12 = 0.5*((Ux[2:,1:-1]-Ux[:-2,1:-1])/(2*dm2) +
               (Uy[1:-1,2:]-Uy[1:-1,:-2])/(2*dm))
    tr  = e11+e22
    sig[1:-1,1:-1,0] = lam*tr + 2*mu*e11
    sig[1:-1,1:-1,1] = lam*tr + 2*mu*e22
    sig[1:-1,1:-1,2] = 2*mu*e12
    return sig

# ── Okada dike (rotated) ──────────────────────────────────────────────────────
def okada_dike_az(X_km, Y_km, xn, yn, xs, ys, xe, ye, az_deg,
                  depth_top, depth_bot, opening, mu=MU_BG, nu=NU_FIX):
    lam = 2*nu*mu/(1-2*nu)
    L   = np.sqrt((xe-xs)**2+(ye-ys)**2)
    xm  = 0.5*(xs+xe); ym = 0.5*(ys+ye)
    az  = np.radians(az_deg)
    s   = np.array([np.sin(az), np.cos(az)])   # along-strike
    n   = np.array([np.cos(az),-np.sin(az)])   # perpendicular
    dX  = X_km-xm; dY = Y_km-ym
    Xr  = dX*n[0]+dY*n[1]; Yr = dX*s[0]+dY*s[1]
    def Rok(xi,e,p): return np.sqrt(xi**2+e**2+p**2)+1e-10
    def fux(xi,e,p):
        r=Rok(xi,e,p); rp=r+p+1e-10; re=r+e+1e-10
        return -(1-2*nu)*xi/re - xi*e/(r*rp)
    def fuy(xi,e,p):
        r=Rok(xi,e,p); rp=r+p+1e-10; re=r+e+1e-10
        th=np.arctan2(xi*e,e*r+1e-10)
        return xi**2/(r*rp)-(1-2*nu)/re+(1-2*nu)*th
    def chinnery(f,xi_m,q_m,d1,d2):
        e1,e2=d1*1e3,d2*1e3
        x1=xi_m-(-L/2)*1e3; x2=xi_m-(L/2)*1e3
        return f(x2,e2,q_m)-f(x2,e1,q_m)-f(x1,e2,q_m)+f(x1,e1,q_m)
    sc  = opening/(2*np.pi)
    xi_m= Yr*1e3; q_m = Xr*1e3
    Ux_df = sc*chinnery(fuy,xi_m,q_m,depth_top,depth_bot)
    Uy_df = sc*chinnery(fux,xi_m,q_m,depth_top,depth_bot)
    Ux_geo= Ux_df*n[0]+Uy_df*s[0]
    Uy_geo= Ux_df*n[1]+Uy_df*s[1]
    dm  = (xn[1]-xn[0])*1e3; dm2=(yn[1]-yn[0])*1e3
    ny_,nx_=Ux_geo.shape; sig=np.zeros((ny_,nx_,3))
    e11=(Ux_geo[1:-1,2:]-Ux_geo[1:-1,:-2])/(2*dm)
    e22=(Uy_geo[2:,1:-1]-Uy_geo[:-2,1:-1])/(2*dm2)
    e12=0.5*((Ux_geo[2:,1:-1]-Ux_geo[:-2,1:-1])/(2*dm2)+
             (Uy_geo[1:-1,2:]-Uy_geo[1:-1,:-2])/(2*dm))
    tr=e11+e22
    sig[1:-1,1:-1,0]=lam*tr+2*mu*e11
    sig[1:-1,1:-1,1]=lam*tr+2*mu*e22
    sig[1:-1,1:-1,2]=2*mu*e12
    return sig

# ── Compute stress fields ─────────────────────────────────────────────────────
xn = np.arange(X_START, X_END+0.05, 0.05)
yn = np.arange(Y_START, Y_END+0.05, 0.05)
X, Y = np.meshgrid(xn, yn)

print('Computing Mogi stress fields...')
sig_pre = mogi_surface_stress(X, Y, xn, yn, SPHERES, DP_INFLATE)
sig_syn_mogi = mogi_surface_stress(X, Y, xn, yn, SPHERES, DP_DEFLATE)
sig_dike = okada_dike_az(X, Y, xn, yn,
                          DIKE_START[0], DIKE_START[1],
                          DIKE_END[0],   DIKE_END[1],
                          DIKE_AZ, DIKE_DTOP, DIKE_DBOT, DIKE_OPENING)
sig_syn = sig_syn_mogi + sig_dike

interps_pre = build_interpolators(sig_pre, xn, yn)
interps_syn = build_interpolators(sig_syn, xn, yn)
print('  Done.')

# ── σ₁ azimuth from 2×2 stress ───────────────────────────────────────────────
def sigma1_az(s2d):
    S = np.array([[s2d[0],s2d[2]],[s2d[2],s2d[1]]])
    eigvals, eigvecs = np.linalg.eigh(S)   # ascending → eigvecs[:,0] = σ₁
    v = eigvecs[:,0]
    return float(np.degrees(np.arctan2(v[0],v[1])) % 180.)

# ── Stations ──────────────────────────────────────────────────────────────────
_sta = pd.read_csv(STATION_FILE, sep=r'\s+',
                   names=['lon','lat','elev_km','station'],
                   engine='python').set_index('station')
_sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'], _sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)

# ── Δφ table ──────────────────────────────────────────────────────────────────
print('\nLoading SWS observations...')
dfs = load_data(dt_error_max=0.1)
all_df = pd.concat(dfs.values(), ignore_index=True)
periods = build_time_periods(all_df)
obs_phi, _, _ = build_observations(dfs, periods)
phi_pre_obs = {STATIONS[i]: obs_phi[i,0] for i in range(len(STATIONS))}
phi_syn_obs = {STATIONS[i]: obs_phi[i,1] for i in range(len(STATIONS))}

phi_pre_mod = {sta: sigma1_az(eval_stress_at(interps_pre,
               _sta.loc[sta,'x'], _sta.loc[sta,'y']))
               for sta in STATIONS if sta in _sta.index}
phi_syn_mod = {sta: sigma1_az(eval_stress_at(interps_syn,
               _sta.loc[sta,'x'], _sta.loc[sta,'y']))
               for sta in STATIONS if sta in _sta.index}

def circ_dphi(a, b):
    return float(np.degrees(np.arctan2(
        np.sin(2*np.radians(a-b)), np.cos(2*np.radians(a-b))))/2.)

print(f'\n{"Station":8s} {"φ_pre mod":>10} {"φ_pre obs":>10} '
      f'{"φ_syn mod":>10} {"φ_syn obs":>10} {"Δφ mod":>9} {"Δφ obs":>9} {"diff":>8}')
print('─'*85)
dphi_m_all, dphi_o_all = [], []
for sta in STATIONS:
    if sta not in phi_pre_mod: continue
    pm = phi_pre_mod[sta]; sm = phi_syn_mod[sta]
    po = phi_pre_obs[sta]; so = phi_syn_obs[sta]
    dm = circ_dphi(sm, pm); do = circ_dphi(so, po)
    dphi_m_all.append(dm); dphi_o_all.append(do)
    print(f'{sta[2:]:8s} {pm:>10.1f}° {po:>10.1f}° '
          f'{sm:>10.1f}° {so:>10.1f}° {dm:>+9.1f}° {do:>+9.1f}° {dm-do:>+8.1f}°')
rms = float(np.sqrt(np.mean((np.array(dphi_m_all)-np.array(dphi_o_all))**2)))
print(f'\nΔφ RMS: {rms:.1f}°')

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

# ── Model σ₁ grid ─────────────────────────────────────────────────────────────
xn_m = np.arange(X_START, X_END+STEP_MOD*.5, STEP_MOD)
yn_m = np.arange(Y_START, Y_END+STEP_MOD*.5, STEP_MOD)
Xm, Ym = np.meshgrid(xn_m, yn_m)

print('Computing model σ₁ fields on grid...')
def sigma1_field(interps):
    phi = np.full(Xm.shape, np.nan)
    for iy in range(len(yn_m)):
        for ix in range(len(xn_m)):
            phi[iy,ix] = sigma1_az(eval_stress_at(interps, xn_m[ix], yn_m[iy]))
    return phi

phi_mod_pre = sigma1_field(interps_pre)
phi_mod_syn = sigma1_field(interps_syn)

# ── Observed φ on model grid ──────────────────────────────────────────────────
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

obs_dfs = {}
for sta,(f1,f2) in FILES.items():
    def _load(f):
        d=pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
        return d[(d['dt']>0)&(d['dt_error']<0.1)]
    df=pd.concat([_load(f1),_load(f2)],ignore_index=True)
    df['x'],df['y']=ll2xy(df['event_lat'].values,df['event_lon'].values)
    df['t']=pd.to_datetime(df['event_datetime'],utc=True)
    df['phi_az']=df['phi']%180.0
    obs_dfs[sta]=df
obs_all=pd.concat(obs_dfs.values(),ignore_index=True)

def period_subset(df, t0, t1):
    m=(df['t']>=t0) if t0 is not None else pd.Series(True,index=df.index)
    if t1 is not None: m=m&(df['t']<t1)
    return df[m]

sub_pre_obs = period_subset(obs_all, None, ERUPTION_START)
sub_syn_obs = period_subset(obs_all, ERUPTION_START, ERUPTION_END)

def bin_phi(x, y, d_phi):
    PHI=np.full(Xm.shape,np.nan); CNT=np.zeros(Xm.shape,dtype=int)
    if len(x)<3: return PHI,CNT
    tree=cKDTree(np.column_stack([x,y]))
    gp=np.column_stack([Xm.ravel(),Ym.ravel()])
    nbrs=tree.query_ball_point(gp,DIS_LIM)
    for k,nb in enumerate(nbrs):
        if not nb: continue
        pts=np.column_stack([x[nb],y[nb]]); gpi=gp[k]
        dist=np.hypot(pts[:,0]-gpi[0],pts[:,1]-gpi[1])
        sel=np.array(nb)[np.argsort(dist)[:NUM_LIM]]
        CNT.ravel()[k]=len(sel)
        ang=2.0*np.radians(d_phi[sel])
        PHI.ravel()[k]=float(np.degrees(np.arctan2(
            np.mean(np.sin(ang)),np.mean(np.cos(ang)))/2)%180)
    return PHI,CNT

print('Binning observations...')
PHI_pre_obs,CNT_pre_obs=bin_phi(sub_pre_obs['x'].values,
                                  sub_pre_obs['y'].values,
                                  sub_pre_obs['phi_az'].values)
PHI_syn_obs,CNT_syn_obs=bin_phi(sub_syn_obs['x'].values,
                                  sub_syn_obs['y'].values,
                                  sub_syn_obs['phi_az'].values)
print(f'  Pre: {sub_pre_obs["event_datetime"].nunique():,} events')
print(f'  Syn: {sub_syn_obs["event_datetime"].nunique():,} events')

# ── Drawing helpers ───────────────────────────────────────────────────────────
def add_sources(ax, is_syn):
    """Draw Kidiwela Mogi sources. Pre: circles; Syn: point sources + dike."""
    for sph in SPHERES:
        if is_syn:
            ax.plot(sph['x0'],sph['y0'],'o',color='#888888',ms=8,
                    mec='black',mew=1.3,zorder=-1)
        else:
            c=Circle((sph['x0'],sph['y0']),radius=sph['R'],
                     edgecolor='black',facecolor='#888888',
                     alpha=0.80,linewidth=1.3,zorder=-1)
            ax.add_patch(c)
        ax.text(sph['x0']+0.12,sph['y0']+0.12,sph['label'],
                fontsize=6,color='black',fontweight='bold',zorder=10)
    if is_syn:
        ax.plot([DIKE_START[0],DIKE_END[0]],[DIKE_START[1],DIKE_END[1]],
                '-',color='black',lw=2.0,zorder=-1)

def add_bathy(ax):
    ax.imshow(_gray,origin='upper',extent=_ext,
              aspect='auto',cmap='gray',alpha=0.6,zorder=0)

def add_stations(ax):
    for sta,row in _sta.iterrows():
        ax.plot(row['x'],row['y'],'^',ms=8,mfc='yellow',mec='black',
                mew=1.0,zorder=12)
        dx,dy=LABEL_OFFSET.get(sta,(0.12,0.12))
        ax.text(row['x']+dx,row['y']+dy,STA_DISPLAY.get(sta,sta),
                fontsize=7,fontweight='bold',zorder=13)

def add_model_arrows(ax, phi_grid):
    step=ARROW_STEP
    Xq=Xm[::step,::step]; Yq=Ym[::step,::step]
    r=np.radians(phi_grid[::step,::step])
    U=np.sin(r); V=np.cos(r)
    qkw=dict(scale=14,width=0.003,headlength=2.5,headaxislength=2,
             headwidth=3,pivot='middle',alpha=0.60,zorder=3,color='navy')
    ax.quiver(Xq,Yq, U, V,**qkw)
    ax.quiver(Xq,Yq,-U,-V,**qkw)

def add_obs_arrows(ax, PHI, CNT):
    step=ARROW_STEP
    Xq=Xm[::step,::step]; Yq=Ym[::step,::step]
    Pq=PHI[::step,::step]; Cq=CNT[::step,::step]
    mask=Cq>=COUNT_MIN
    if not mask.any(): return
    r=np.radians(Pq[mask])
    U=np.sin(r); V=np.cos(r)
    qkw=dict(scale=14,width=0.003,headlength=2.5,headaxislength=2,
             headwidth=3,pivot='middle',alpha=0.90,zorder=7,color='red')
    ax.quiver(Xq[mask],Yq[mask], U, V,**qkw)
    ax.quiver(Xq[mask],Yq[mask],-U,-V,**qkw)

def finalise(ax):
    ax.set_xlim(4.5,12.0); ax.set_ylim(1.5,10.5)
    ax.set_aspect('equal'); ax.grid(True,alpha=0.15,lw=0.4)
    ax.set_xlabel('East [km]',fontsize=10)

# ── Figure 1: model σ₁ field only ─────────────────────────────────────────────
print('\nBuilding geometry stress figure...')
fig,axes=plt.subplots(1,2,figsize=(16,8),sharex=True,sharey=True)
for ax_i,(ax,title,phi_mod,is_syn) in enumerate(zip(
    axes,['Pre-eruption','Syn-eruption'],
    [phi_mod_pre,phi_mod_syn],[False,True]
)):
    add_sources(ax,is_syn)
    add_bathy(ax)
    add_model_arrows(ax,phi_mod)
    add_stations(ax)
    ax.set_title(title,fontsize=12,fontweight='bold')
    finalise(ax)
axes[0].set_ylabel('North [km]',fontsize=10)

legend_elements=[
    mpatches.Patch(facecolor='#888888',edgecolor='black',
                   label='Kidiwela S1 & S2 (circles pre / dots syn)'),
    plt.Line2D([0],[0],color='black',lw=2.0,label='Dike (syn only)'),
    plt.Line2D([0],[0],color='navy',lw=1.5,alpha=0.7,
               label='Model σ₁ (compression)'),
]
leg=axes[1].legend(handles=legend_elements,fontsize=8,loc='lower right',
                   framealpha=0.95)
leg.set_zorder(20)
fig.suptitle('Baillard-style model: Kidiwela Mogi sources + Okada dike\n'
             'S1=(7.57,4.55,3.33km,R=0.43km)  S2=(7.53,6.60,1.25km,R=0.20km)',
             fontsize=10,fontweight='bold')
plt.subplots_adjust(wspace=0.05)
out1=os.path.join(OUT_DIR,'baillard_kidiwela_geometry_stress.pdf')
fig.savefig(out1,dpi=200,bbox_inches='tight'); plt.close()
print(f'Saved {out1}')

# ── Figure 2: model + observed ────────────────────────────────────────────────
print('Building model + obs figure...')
fig,axes=plt.subplots(1,2,figsize=(16,8),sharex=True,sharey=True)
for ax_i,(ax,title,phi_mod,PHI_obs,CNT_obs,is_syn) in enumerate(zip(
    axes,
    [f'Pre-eruption  (N={sub_pre_obs["event_datetime"].nunique():,})',
     f'Syn-eruption  (N={sub_syn_obs["event_datetime"].nunique():,})'],
    [phi_mod_pre,phi_mod_syn],
    [PHI_pre_obs,PHI_syn_obs],
    [CNT_pre_obs,CNT_syn_obs],
    [False,True]
)):
    add_sources(ax,is_syn)
    add_bathy(ax)
    add_model_arrows(ax,phi_mod)
    add_obs_arrows(ax,PHI_obs,CNT_obs)
    add_stations(ax)
    ax.set_title(title,fontsize=11,fontweight='bold')
    finalise(ax)
axes[0].set_ylabel('North [km]',fontsize=10)

legend_elements=[
    mpatches.Patch(facecolor='#888888',edgecolor='black',
                   label='Kidiwela S1 & S2'),
    plt.Line2D([0],[0],color='black',lw=2.0,label='Dike (syn only)'),
    plt.Line2D([0],[0],color='navy',lw=1.5,alpha=0.7,
               label='Model σ₁ (compression)'),
    plt.Line2D([0],[0],color='red',lw=2.0,
               label='Observed φ (all depths, dt_err<0.1s)'),
]
leg=axes[1].legend(handles=legend_elements,fontsize=8,loc='lower right',
                   framealpha=0.95)
leg.set_zorder(20)
fig.suptitle('Kidiwela Mogi sources: modelled σ₁ vs. observed φ\n'
             'Pre-eruption and syn-eruption — all stations combined',
             fontsize=10,fontweight='bold')
plt.subplots_adjust(wspace=0.05)
out2=os.path.join(OUT_DIR,'baillard_kidiwela_geometry_stress_with_obs.pdf')
fig.savefig(out2,dpi=200,bbox_inches='tight'); plt.close()
print(f'Saved {out2}')
print('\nDone.')
