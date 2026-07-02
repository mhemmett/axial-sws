#!/usr/bin/env python3
"""
mogi_stress_3d_compliance.py

Uses Baillard's 3D S-wave velocity model to build a spatially varying shear
modulus map μ(x,y) = ρ × <Vs²>_depth, then optimises a mixing parameter α:

    μ_eff(x,y) = (1-α)·μ_bg  +  α·μ_3D(x,y)

where α=0 → uniform (current model), α=1 → fully 3D compliance.

Because σ = 2μ(x,y)·ε, spatial variation in μ changes the σ₁ direction as
well as magnitude, so α is a genuine free parameter that affects φ predictions.

Minimises RMS(φ_pred_pre − φ_obs_pre) over α ∈ [0,1].
Shows table of pre-eruption and syn-eruption results.
No dike. Mogi only (S1=5%, S2=100% deflation).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.ndimage import map_coordinates
import os

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
DATA_DIR     = '/Users/mhemmett/Seismology/axial-splitting-ml/data/'
STATION_FILE = DATA_DIR + 'stations_axial.llz'
NLL_BUF      = DATA_DIR + 'AXIAL_MODEL_3P_VELOCITY.S.mod.buf'

# ── NLL model parameters ──────────────────────────────────────────────────────
NLL_NX,NLL_NY,NLL_NZ = 302,302,92
NLL_OX,NLL_OY,NLL_OZ = 0.,0.,-0.5
NLL_DX,NLL_DY,NLL_DZ = 0.05,0.05,0.05
MEAN_SEAFLOOR_KM = 1.660
RHO = 2750.0   # kg/m³ crustal density

INI_LON,INI_LAT = -130.1,45.9
KPD_LAT = 111.32
KPD_LON = 111.32*np.cos(np.radians(INI_LAT))
def ll2xy(lat,lon): return ((np.asarray(lon)-INI_LON)*KPD_LON,(np.asarray(lat)-INI_LAT)*KPD_LAT)

MU_BG,NU = 30e9,0.25   # background shear modulus (Pa)

SPHERES=[dict(x0=7.57,y0=4.55,d=3.33,R=0.43,dP=0.05e9,label='S1'),
         dict(x0=7.53,y0=6.60,d=1.25,R=0.20,dP=0.05e9,label='S2')]

ERUPTION_START=pd.Timestamp('2015-04-24 06:00',tz='UTC')
ERUPTION_END  =pd.Timestamp('2015-05-19 00:00',tz='UTC')
STATIONS=['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']

FILES={
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv','splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv','splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv','splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv','splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv','splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv','splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

# ── Load 3D Vs and build 2D μ map ─────────────────────────────────────────────

print('Loading 3D Vs model and building μ(x,y) map...')
vel_3d = np.frombuffer(open(NLL_BUF,'rb').read(),
                       dtype=np.float32).reshape(NLL_NX,NLL_NY,NLL_NZ).copy()

# Depth range for averaging: 0.5–2.0 km below seafloor
z_sf_min,z_sf_max = 0.5,2.0
iz_min = int(np.round((MEAN_SEAFLOOR_KM+z_sf_min-NLL_OZ)/NLL_DZ))
iz_max = int(np.round((MEAN_SEAFLOOR_KM+z_sf_max-NLL_OZ)/NLL_DZ))
iz_min,iz_max = max(0,iz_min),min(NLL_NZ-1,iz_max)

# Mean Vs² over earthquake depth range → μ_3D in Pa
vs_mean_sq = np.mean(vel_3d[:,:,iz_min:iz_max+1]**2, axis=2)  # (NX,NY) km²/s²
mu_3d_grid = (RHO * vs_mean_sq * 1e6).T  # (NY,NX) Pa  [km²/s² → m²/s²: ×1e6]

print(f'  μ_3D range: {mu_3d_grid.min()/1e9:.2f} – {mu_3d_grid.max()/1e9:.2f} GPa')
print(f'  μ_3D mean:  {mu_3d_grid.mean()/1e9:.2f} GPa  (vs background {MU_BG/1e9:.0f} GPa)')

# Grid for stress computation
xn=np.arange(4,12.01,0.1); yn=np.arange(0,12.01,0.1); X,Y=np.meshgrid(xn,yn)
ny_g,nx_g=X.shape

# Interpolate μ_3D onto our analysis grid
# NLL ix = x/NLL_DX, iy = y/NLL_DY
# mu_3d_grid is indexed [iy, ix] because we transposed
ix_query = X.ravel() / NLL_DX   # continuous NLL ix
iy_query = Y.ravel() / NLL_DY
mu_3d_on_grid = map_coordinates(
    mu_3d_grid, [iy_query, ix_query], order=1, mode='nearest'
).reshape(ny_g, nx_g)

print(f'  μ_3D on analysis grid: {mu_3d_on_grid.min()/1e9:.2f}–{mu_3d_on_grid.max()/1e9:.2f} GPa')

# ── Stress tensor with spatially varying μ ────────────────────────────────────

def T_sphere_mogi(xn,yn,X,Y,sph,sc=1.):
    """Mogi displacement using background μ (source volume change)."""
    dV=np.pi*(sph['R']*1e3)**3*sph['dP']*sc/MU_BG; C=dV*(1-NU)/np.pi
    dx=(X-sph['x0'])*1e3; dy=(Y-sph['y0'])*1e3; d=sph['d']*1e3
    R3=(dx**2+dy**2+d**2)**1.5
    return C*dx/R3, C*dy/R3

def stress_field_3d_mu(xn,yn,X,Y,sc1,sc2,mu_grid,nu=NU):
    """
    Stress tensor field using spatially varying μ(x,y) = mu_grid.
    Displacement field computed with background MU_BG for consistency.
    Stress: σ = 2μ(x,y)·ε + λ(x,y)·tr(ε)·I
    """
    Ux=np.zeros_like(X); Uy=np.zeros_like(X)
    for sph,sc in zip(SPHERES,[sc1,sc2]):
        ux,uy=T_sphere_mogi(xn,yn,X,Y,sph,sc)
        Ux+=ux; Uy+=uy
    ny_,nx_=Ux.shape; dm=(xn[1]-xn[0])*1e3; dm2=(yn[1]-yn[0])*1e3
    T=np.zeros((ny_,nx_,2,2))
    # Strain via central differences
    e11=(Ux[1:-1,2:]-Ux[1:-1,:-2])/(2*dm)
    e22=(Uy[2:,1:-1]-Uy[:-2,1:-1])/(2*dm2)
    e12=0.5*((Ux[2:,1:-1]-Ux[:-2,1:-1])/(2*dm2)
            +(Uy[1:-1,2:]-Uy[1:-1,:-2])/(2*dm))
    tr=e11+e22
    # Spatially varying λ = 2νμ/(1-2ν)
    mu_i=mu_grid[1:-1,1:-1]; lam_i=2*nu*mu_i/(1-2*nu)
    T[1:-1,1:-1,0,0]=lam_i*tr+2*mu_i*e11
    T[1:-1,1:-1,0,1]=T[1:-1,1:-1,1,0]=2*mu_i*e12
    T[1:-1,1:-1,1,1]=lam_i*tr+2*mu_i*e22
    return T

def phi_at(T,xn,yn,sx,sy):
    iy=int(np.argmin(abs(yn-sy))); ix=int(np.argmin(abs(xn-sx)))
    ev,evec=np.linalg.eigh(T[iy,ix]); v=abs(ev[1]-ev[0])*evec[:,0]
    return float(np.degrees(np.arctan2(v[0],v[1]))%180)

def dphi(a,b):
    if np.isnan(a) or np.isnan(b): return float('nan')
    return float((a-b+90)%180-90)

# ── Load observations ─────────────────────────────────────────────────────────

_sta=pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','e','s'],engine='python').set_index('s')
_sta=_sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'],_sta['y']=ll2xy(_sta['lat'].values,_sta['lon'].values)

obs_pre={}; obs_syn={}
for sta,(f1,f2) in FILES.items():
    df=pd.concat([pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna() for f in [f1,f2]],ignore_index=True)
    df=df[df['dt']>0]; df['t']=pd.to_datetime(df['event_datetime'],utc=True); df['phi_az']=df['phi']+90
    pre=df[df['t']<ERUPTION_START]; syn=df[(df['t']>=ERUPTION_START)&(df['t']<ERUPTION_END)]
    obs_pre[sta]=float(pre['phi_az'].median()) if len(pre)>=10 else float('nan')
    obs_syn[sta]=float(syn['phi_az'].median()) if len(syn)>=10 else float('nan')

sp=(3.5-1.)/2.2   # pre-eruption source scale
sp_syn=0.95*sp     # S1 5% deflated
sp_s2=0.0          # S2 fully deflated

STA_XY=[(float(_sta.loc[s,'x']),float(_sta.loc[s,'y'])) for s in STATIONS]

# ── Optimise α ────────────────────────────────────────────────────────────────

print('\nOptimising mixing parameter α (0=uniform μ, 1=full 3D μ)...')

def mu_mixed(alpha):
    return (1-alpha)*MU_BG + alpha*mu_3d_on_grid

def objective(alpha):
    mu_eff = mu_mixed(alpha)
    T_pre  = stress_field_3d_mu(xn,yn,X,Y,sp,sp,mu_eff)
    errs   = []
    for s,(sx,sy) in zip(STATIONS,STA_XY):
        if np.isnan(obs_pre[s]): continue
        errs.append(dphi(phi_at(T_pre,xn,yn,sx,sy), obs_pre[s]))
    return np.sqrt(np.mean(np.array(errs)**2))

# Scan α over [0, 1]
alphas = np.arange(0, 1.01, 0.05)
rms_scan = [objective(a) for a in alphas]
best_idx = np.argmin(rms_scan)
print(f'  Scan: best α = {alphas[best_idx]:.2f}  RMS = {rms_scan[best_idx]:.2f}°')

# Refine
res = minimize_scalar(objective, bounds=(
    max(0, alphas[best_idx]-0.05), min(1, alphas[best_idx]+0.05)),
    method='bounded')
alpha_opt = float(np.clip(res.x, 0, 1))
rms_opt   = float(res.fun)
print(f'  Optimised: α = {alpha_opt:.4f}  RMS = {rms_opt:.2f}°')
print(f'  (α=0 uniform μ RMS = {rms_scan[0]:.2f}°  |  α=1 full 3D RMS = {rms_scan[-1]:.2f}°)')

# ── Compute stress fields at optimal α ────────────────────────────────────────

mu_opt    = mu_mixed(alpha_opt)
T_pre_opt = stress_field_3d_mu(xn,yn,X,Y,sp,sp,mu_opt)
T_syn_opt = stress_field_3d_mu(xn,yn,X,Y,sp_syn,sp_s2,mu_opt)

# ── Results table ──────────────────────────────────────────────────────────────

sep='─'*90
print('\n'+sep)
print(f'  3D compliance model  |  α = {alpha_opt:.3f}  |  '
      f'μ_eff = (1-α)·{MU_BG/1e9:.0f}GPa + α·μ_3D(x,y)  |  Mogi only')
print(sep)
print(f"{'Station':>8} | {'phi_obs':^14} | {'Dphi_obs':>9} | {'phi_pred':^14} | {'Dphi_pred':>10} | {'misfit':>7}")
print(f"{'':>8} | {'PRE':>6} {'SYN':>6} | {'':>9} | {'PRE':>6} {'SYN':>6} | {'':>10} | {'':>7}")
print(sep)
misfits=[]
for s,(sx,sy) in zip(STATIONS,STA_XY):
    do=dphi(obs_syn[s],obs_pre[s])
    pp=phi_at(T_pre_opt,xn,yn,sx,sy)
    ps=phi_at(T_syn_opt,xn,yn,sx,sy)
    dp=dphi(ps,pp)
    mf=dphi(dp,do)
    misfits.append(mf)
    print(f"{s[2:]:>8} | {obs_pre[s]:>6.1f} {obs_syn[s]:>6.1f} | {do:>+9.1f} | "
          f"{pp:>6.1f} {ps:>6.1f} | {dp:>+10.1f} | {mf:>+7.1f}")
print(sep)
rms_final=np.sqrt(np.nanmean(np.array(misfits)**2))
print(f"{'RMS':>8}   {'':>6} {'':>6}   {'':>9}   {'':>6} {'':>6}   {'':>10}   {rms_final:>+7.2f}°")

# ── α scan plot ───────────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(7,4))
ax.plot(alphas,rms_scan,'k-o',ms=4)
ax.axvline(alpha_opt,color='r',ls='--',label=f'α_opt={alpha_opt:.3f}  RMS={rms_opt:.1f}°')
ax.set_xlabel('Mixing parameter α  (0=uniform μ, 1=full 3D)', fontsize=11)
ax.set_ylabel('RMS φ misfit (°)', fontsize=11)
ax.set_title('Pre-eruption φ RMS vs 3D compliance mixing', fontsize=11)
ax.legend(); ax.grid(True,alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(BASE,'mogi_3d_compliance_scan.pdf'),dpi=200,bbox_inches='tight')
plt.close(fig)
print(f'\nSaved mogi_3d_compliance_scan.pdf')
print('Done.')
