#!/usr/bin/env python3
"""
sws_raytraced_dt.py

3D ray-traced δt accumulation through Baillard S-wave velocity model.

For each event-station pair:
  1. Compute a bent ray path using iterative pseudo-bending (Um & Thurber 1987)
     through the Baillard 3D Vs model.
  2. Distribute the measured δt along the ray, weighted by segment length.
  3. Accumulate into a 3D voxel grid.

Display: coverage-weighted median δt per voxel, shown as depth-slice maps
for 7 time periods (pre, syn, 5 post-eruption quintiles).

Output: sws_temporal_combined_dt_raytraced_kidiwela_source_phi15.pdf
"""

import sys, warnings, time
sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, tifffile
from PIL import Image as PILImage
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

from baillard_velocity import vs_at

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT          = BASE + 'sws_temporal_combined_dt_raytraced_kidiwela_source_phi15.pdf'

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_PER_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_PER_DEG_LAT)

KIDIWELA_SOURCES = [dict(x=7.57,y=4.55,label='S1'),
                    dict(x=7.53,y=6.60,label='S2')]
ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')
STATIONS    = ['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET_KM = {
    'AXAS1':(-0.20,-0.55),'AXAS2':(-0.45,-0.55),'AXCC1':(-0.75,0.25),
    'AXEC1':(0.15,0.30),'AXEC2':(0.15,0.05),'AXEC3':(0.30,-0.28),
}
PHI_ERR_MAX = 15.0

# ── Voxel grid ────────────────────────────────────────────────────────────────
X_START,X_END = 4.0,12.0; Y_START,Y_END = 0.0,12.0; Z_MAX = 4.0
VOXEL_XY = 0.30   # km
VOXEL_Z  = 0.25   # km
xn = np.arange(X_START, X_END+VOXEL_XY*.5, VOXEL_XY)
yn = np.arange(Y_START, Y_END+VOXEL_XY*.5, VOXEL_XY)
zn = np.arange(0.,       Z_MAX+VOXEL_Z*.5,  VOXEL_Z)
NX,NY,NZ = len(xn),len(yn),len(zn)

# Display depth slices (4 rows, 0–2.5 km)
DEPTH_SLICES  = [0.3125, 0.9375, 1.5625, 2.1875]
Z_HALF        = 0.3125
DEPTH_LABELS  = ['0.0–0.625 km','0.625–1.25 km','1.25–1.875 km','1.875–2.5 km']
COUNT_MIN     = 5   # min ray-segments per voxel to display

# ── Ray bending parameters ────────────────────────────────────────────────────
N_RAY  = 25    # samples per ray
N_ITER = 5     # bending iterations
H_GRAD = 0.05  # km finite-difference step for gradient
BEND_ALPHA = 0.25  # bending step size (fraction of ds²)

def bent_ray(eq_x, eq_y, eq_z, sta_x, sta_y):
    """
    Pseudo-bending ray from (eq_x,eq_y,eq_z) to (sta_x,sta_y,0).
    Returns (rx,ry,rz) arrays of N_RAY points along the bent ray [km].
    """
    t  = np.linspace(0., 1., N_RAY)
    rx = eq_x + t*(sta_x - eq_x)
    ry = eq_y + t*(sta_y - eq_y)
    rz = eq_z * (1. - t)        # linear depth: eq_z → 0 at station

    for _ in range(N_ITER):
        # Velocity and slowness at each ray point
        vs  = vs_at(rx, ry, rz)
        sl  = 1.0 / np.maximum(vs, 0.1)

        # Slowness gradient via central finite differences
        dsdx = (1./np.maximum(vs_at(rx+H_GRAD,ry,rz),0.1)
               -1./np.maximum(vs_at(rx-H_GRAD,ry,rz),0.1)) / (2.*H_GRAD)
        dsdy = (1./np.maximum(vs_at(rx,ry+H_GRAD,rz),0.1)
               -1./np.maximum(vs_at(rx,ry-H_GRAD,rz),0.1)) / (2.*H_GRAD)
        dsdz = (1./np.maximum(vs_at(rx,ry,rz+H_GRAD),0.1)
               -1./np.maximum(vs_at(rx,ry,rz-H_GRAD),0.1)) / (2.*H_GRAD)

        # Ray tangent direction
        drx = np.gradient(rx); dry = np.gradient(ry); drz = np.gradient(rz)
        ds_len = np.sqrt(drx**2 + dry**2 + drz**2) + 1e-10
        tx,ty,tz = drx/ds_len, dry/ds_len, drz/ds_len

        # Perpendicular slowness gradient (Ray equation: d(n*T̂)/ds = ∇n)
        TdotG = tx*dsdx + ty*dsdy + tz*dsdz
        Fx = dsdx - TdotG*tx
        Fy = dsdy - TdotG*ty
        Fz = dsdz - TdotG*tz

        L   = float(np.sqrt(np.sum((np.diff(rx)**2+np.diff(ry)**2+np.diff(rz)**2))))
        ds2 = (L / (N_RAY-1))**2

        # Update interior points (endpoints fixed)
        rx[1:-1] += BEND_ALPHA * ds2 * Fx[1:-1]
        ry[1:-1] += BEND_ALPHA * ds2 * Fy[1:-1]
        rz[1:-1] += BEND_ALPHA * ds2 * Fz[1:-1]

        # Clamp to model bounds
        rx = np.clip(rx, 0., 15.)
        ry = np.clip(ry, 0., 15.)
        rz = np.clip(rz, 0., Z_MAX)

    return rx, ry, rz


def accumulate_ray(rx, ry, rz, dt_val, DT_num, COV_den):
    """
    Distribute dt_val along the ray (weighted by segment length / total length).
    Accumulates into DT_num (dt contribution) and COV_den (coverage) voxel arrays.
    """
    drx = np.diff(rx); dry = np.diff(ry); drz = np.diff(rz)
    ds  = np.sqrt(drx**2+dry**2+drz**2)   # segment lengths [km]
    L   = ds.sum() + 1e-10

    mx = 0.5*(rx[:-1]+rx[1:])   # segment midpoints
    my = 0.5*(ry[:-1]+ry[1:])
    mz = 0.5*(rz[:-1]+rz[1:])

    for k in range(len(ds)):
        ix = int((mx[k]-X_START)/VOXEL_XY)
        iy = int((my[k]-Y_START)/VOXEL_XY)
        iz = int( mz[k]          /VOXEL_Z )
        if 0<=ix<NX and 0<=iy<NY and 0<=iz<NZ:
            w = ds[k] / L
            DT_num[ix,iy,iz] += dt_val * w
            COV_den[ix,iy,iz] += w


# ── Files ──────────────────────────────────────────────────────────────────────
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

# ── Load station locations ────────────────────────────────────────────────────
_sta = pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','elev_km','station'],
                   engine='python').set_index('station')
_sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'],_sta['y'] = ll2xy(_sta['lat'].values,_sta['lon'].values)
sta_xy = {sta:(float(_sta.loc[sta,'x']),float(_sta.loc[sta,'y']))
          for sta in STATIONS if sta in _sta.index}

# ── Load events with phi15 filter ─────────────────────────────────────────────
print(f'Loading events (phi_error ≤ {PHI_ERR_MAX}°)...')
dfs = {}
for sta,(f1,f2) in FILES.items():
    def _load(f):
        d=pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
        return d[(d['dt']>0)&(d['phi_error']<=PHI_ERR_MAX)]
    df=pd.concat([_load(f1),_load(f2)],ignore_index=True)
    df['x'],df['y']=ll2xy(df['event_lat'].values,df['event_lon'].values)
    df['z']=df['event_depth'].values
    df['t']=pd.to_datetime(df['event_datetime'],utc=True)
    dfs[sta]=df
    print(f'  {sta}: {len(df):,}')

all_df = pd.concat(dfs.values(),ignore_index=True)

# ── Time periods (quintiles from ALL data) ────────────────────────────────────
def build_7_periods(all_df):
    post=all_df[all_df['t']>=ERUPTION_END].sort_values('t').reset_index(drop=True)
    n=len(post); bounds=[ERUPTION_END]
    for i in range(1,5):
        idx=min(int(round(i*n/5)),n-1); bounds.append(post['t'].iloc[idx])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds=[('Pre-eruption',None,ERUPTION_START),
         ('Syn-eruption',ERUPTION_START,ERUPTION_END)]
    for i in range(5):
        pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}',bounds[i],bounds[i+1]))
    return pds

def subset(df,t0,t1):
    m=(df['t']>=t0) if t0 is not None else pd.Series(True,index=df.index)
    if t1 is not None: m=m&(df['t']<t1)
    return df[m]

periods = build_7_periods(all_df)

# ── Ray-trace all events → accumulate into per-period voxel grids ─────────────
print(f'\nRay-tracing {len(all_df):,} events through Baillard Vs model...')
print(f'  Grid: {NX}×{NY}×{NZ} voxels ({VOXEL_XY}×{VOXEL_XY}×{VOXEL_Z} km)')

# Pre-compute bent rays and voxel contributions for all events
# Store as list of (t, dt, DT_num_contribution) to support any time period split
# Memory-efficient: compute per-period accumulators directly

n_per = len(periods)
# Per-station, per-period voxel accumulators
# DT_num[sta][period] and COV_den[sta][period]
STA_LIST = [sta for sta in STATIONS if sta in sta_xy]
DT_num  = {sta: [np.zeros((NX,NY,NZ)) for _ in range(n_per)] for sta in STA_LIST}
COV_den = {sta: [np.zeros((NX,NY,NZ)) for _ in range(n_per)] for sta in STA_LIST}
N_rays  = {sta: [0]*n_per for sta in STA_LIST}

t_start = time.time()
total   = sum(len(dfs[sta]) for sta in STATIONS if sta in sta_xy)
done    = 0

for sta in STATIONS:
    if sta not in sta_xy: continue
    sx,sy = sta_xy[sta]
    df_s  = dfs[sta]

    for _, row in df_s.iterrows():
        eq_x,eq_y,eq_z = float(row['x']),float(row['y']),float(row['z'])
        dt_val = float(row['dt'])
        t_evt  = row['t']

        # Skip events above surface or too deep
        if eq_z < 0 or eq_z > Z_MAX: done+=1; continue

        # Determine which period this event belongs to
        per_idx = None
        for j,(lbl,t0,t1) in enumerate(periods):
            m=(t_evt>=t0) if t0 is not None else True
            if t1 is not None: m = m and (t_evt<t1)
            if m: per_idx=j; break
        if per_idx is None: done+=1; continue

        # Bent ray
        rx,ry,rz = bent_ray(eq_x,eq_y,eq_z,sx,sy)

        # Accumulate into THIS STATION's period voxel arrays
        accumulate_ray(rx,ry,rz,dt_val,DT_num[sta][per_idx],COV_den[sta][per_idx])
        N_rays[sta][per_idx]+=1
        done+=1

        if done%5000==0:
            el=time.time()-t_start
            print(f'  {done:,}/{total:,}  '
                  f'{el:.0f}s elapsed  '
                  f'~{el/done*(total-done):.0f}s remaining',end='\r',flush=True)

print(f'\nDone in {time.time()-t_start:.0f}s')
for sta in STA_LIST:
    total_sta = sum(N_rays[sta])
    print(f'  {sta}: {total_sta:,} rays total')

# ── Bathymetry ────────────────────────────────────────────────────────────────
BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
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

def _bathy(ax):
    ax.imshow(_gray,origin='upper',extent=_ext,aspect='auto',cmap='gray',zorder=0)

def _add_stations(ax,highlight=None):
    for sta,row in _sta.iterrows():
        mfc='#FFD700' if (highlight is None or sta==highlight) else 'white'
        ax.plot(row['x'],row['y'],'^',ms=6,mfc=mfc,mec='k',mew=0.7,zorder=12)
        dx,dy=LABEL_OFFSET_KM.get(sta,(0.12,0.12))
        ax.text(row['x']+dx,row['y']+dy,STA_DISPLAY.get(sta,sta),fontsize=5.5,zorder=13)

def _add_kidiwela(ax):
    for s in KIDIWELA_SOURCES:
        ax.plot(s['x'],s['y'],'o',ms=4,mfc='red',mec='k',mew=0.5,zorder=14)
        ax.text(s['x']+0.12,s['y']+0.12,s['label'],fontsize=5,
                color='red',fontweight='bold',zorder=15)

# ── Global dt vmax from 99th pct across all stations and periods ──────────────
all_dt_vals = []
for sta in STA_LIST:
    for j in range(n_per):
        cov = COV_den[sta][j]; num = DT_num[sta][j]
        mask = cov > 0
        if mask.any():
            all_dt_vals.extend((num[mask]/cov[mask]).ravel())
dt_vmax = float(np.percentile(all_dt_vals,99)) if all_dt_vals else 0.3
dt_vmin = 0.
print(f'Global δt range: 0 – {dt_vmax:.3f} s (99th pct)')

# ── Build figure page ─────────────────────────────────────────────────────────
cmap   = plt.colormaps['Blues']
levels = np.linspace(dt_vmin, dt_vmax, 15)
GAUSS  = 1.0   # smooth over ~1 voxel

Xg, Yg = np.meshgrid(xn, yn, indexing='ij')

def make_page(sta_key, highlight, title):
    """
    sta_key: station name for per-station page, or None for all-stations combined.
    """
    n_dep = len(DEPTH_SLICES)
    panel_w = 2.2
    fig = plt.figure(figsize=(n_per*panel_w+0.6, n_dep*3.0+0.3))
    gs  = GridSpec(n_dep, n_per+1,
                   width_ratios=[1]*n_per+[0.04], hspace=0.04, wspace=0.04)
    last_h = None

    for ri,(z0,zlbl) in enumerate(zip(DEPTH_SLICES,DEPTH_LABELS)):
        iz0 = max(0, int((z0-Z_HALF)/VOXEL_Z))
        iz1 = min(NZ, int((z0+Z_HALF)/VOXEL_Z)+1)

        for ci,(lbl,t0,t1) in enumerate(periods):
            ax = fig.add_subplot(gs[ri,ci]); _bathy(ax)

            # Select the right accumulator: per-station or combined
            if sta_key is not None:
                num_3d = DT_num[sta_key][ci]
                cov_3d = COV_den[sta_key][ci]
                n_rays_ci = N_rays[sta_key][ci]
            else:
                # Sum over all stations
                num_3d = sum(DT_num[s][ci] for s in STA_LIST)
                cov_3d = sum(COV_den[s][ci] for s in STA_LIST)
                n_rays_ci = sum(N_rays[s][ci] for s in STA_LIST)

            num_sl = num_3d[:,:,iz0:iz1].sum(axis=2)
            cov_sl = cov_3d[:,:,iz0:iz1].sum(axis=2)
            n_slice= int((cov_sl > 0).sum())

            if n_slice >= COUNT_MIN:
                with np.errstate(invalid='ignore',divide='ignore'):
                    dt_sl = np.where(cov_sl > 0, num_sl/cov_sl, np.nan)
                dt_sm = gaussian_filter(np.nan_to_num(dt_sl), sigma=GAUSS)
                mask  = (cov_sl >= COUNT_MIN/max(iz1-iz0,1))
                dt_pl = np.where(mask, dt_sm, np.nan)
                mdt   = np.ma.masked_invalid(dt_pl)
                if not mdt.mask.all():
                    ax.contourf(Xg,Yg,mdt,levels=levels,
                                cmap=cmap,vmin=dt_vmin,vmax=dt_vmax,
                                extend='neither')
                    last_h = ScalarMappable(cmap=cmap,
                                            norm=Normalize(dt_vmin,dt_vmax))
                    last_h.set_array([])

            _add_stations(ax,highlight); _add_kidiwela(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0:
                ax.set_title(f'{lbl}\nN={n_rays_ci:,}',
                             fontsize=7.5,fontweight='bold',pad=1)
            if ci==0:
                ax.set_ylabel(f'{zlbl}\n(n_vox={n_slice:,})',fontsize=6.5)

    if last_h:
        cax = fig.add_subplot(gs[:,n_per])
        ticks = np.linspace(dt_vmin,dt_vmax,5)
        cb = plt.colorbar(last_h,cax=cax,ticks=ticks,
                          format=FormatStrFormatter('%.2f'))
        cb.set_label('Ray-weighted δt [s]',fontsize=9)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.97)
    return fig

# ── Write PDF ──────────────────────────────────────────────────────────────────
print(f'\nWriting {os.path.basename(OUT)}...')
with PdfPages(OUT) as pdf:
    for sta in STATIONS:
        if sta not in sta_xy: continue
        fig = make_page(sta_key=sta, highlight=sta,
                        title=f'{sta} — ray-traced δt  (φ_err≤{PHI_ERR_MAX}°, '
                              f'rays from {sta} only)')
        pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
        print(f'  {sta}')
    fig = make_page(sta_key=None, highlight=None,
                    title=f'All stations — ray-traced δt  (φ_err≤{PHI_ERR_MAX}°, '
                          f'all rays combined)')
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
    print('  All stations')

print(f'Saved {OUT}')

# ── Second PDF: ray spatial coverage (COV_den only, Reds colormap) ─────────────
OUT2 = BASE + 'sws_temporal_combined_raycoverage_kidiwela_source_phi15.pdf'
print(f'\nWriting {os.path.basename(OUT2)}...')

# Global coverage vmax
all_cov_vals = []
for sta in STA_LIST:
    for j in range(n_per):
        c = COV_den[sta][j]
        if c.max() > 0:
            all_cov_vals.extend(c[c>0].ravel())
cov_vmax = float(np.percentile(all_cov_vals,99)) if all_cov_vals else 1.0
cov_vmin = 0.
cov_levels = np.linspace(cov_vmin, cov_vmax, 15)
cmap_cov   = plt.colormaps['Reds']
print(f'Coverage range: 0 – {cov_vmax:.4f} (99th pct)')

def make_coverage_page(sta_key, highlight, title):
    n_dep = len(DEPTH_SLICES)
    panel_w = 2.2
    fig = plt.figure(figsize=(n_per*panel_w+0.6, n_dep*3.0+0.3))
    gs  = GridSpec(n_dep, n_per+1,
                   width_ratios=[1]*n_per+[0.04], hspace=0.04, wspace=0.04)
    last_h = None

    for ri,(z0,zlbl) in enumerate(zip(DEPTH_SLICES,DEPTH_LABELS)):
        iz0 = max(0, int((z0-Z_HALF)/VOXEL_Z))
        iz1 = min(NZ, int((z0+Z_HALF)/VOXEL_Z)+1)

        for ci,(lbl,t0,t1) in enumerate(periods):
            ax = fig.add_subplot(gs[ri,ci]); _bathy(ax)

            if sta_key is not None:
                cov_3d = COV_den[sta_key][ci]
                n_rays_ci = N_rays[sta_key][ci]
            else:
                cov_3d = sum(COV_den[s][ci] for s in STA_LIST)
                n_rays_ci = sum(N_rays[s][ci] for s in STA_LIST)

            cov_sl = cov_3d[:,:,iz0:iz1].sum(axis=2)
            n_slice= int((cov_sl > 0).sum())

            if n_slice >= COUNT_MIN:
                cov_sm = gaussian_filter(cov_sl, sigma=GAUSS)
                cov_pl = np.where(cov_sl > 0, cov_sm, np.nan)
                mcov   = np.ma.masked_invalid(cov_pl)
                if not mcov.mask.all():
                    ax.contourf(Xg,Yg,mcov,levels=cov_levels,
                                cmap=cmap_cov,vmin=cov_vmin,vmax=cov_vmax,
                                extend='neither')
                    last_h = ScalarMappable(cmap=cmap_cov,
                                            norm=Normalize(cov_vmin,cov_vmax))
                    last_h.set_array([])

            _add_stations(ax,highlight); _add_kidiwela(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0:
                ax.set_title(f'{lbl}\nN={n_rays_ci:,}',
                             fontsize=7.5,fontweight='bold',pad=1)
            if ci==0:
                ax.set_ylabel(f'{zlbl}\n(n_vox={n_slice:,})',fontsize=6.5)

    if last_h:
        cax = fig.add_subplot(gs[:,n_per])
        ticks = np.linspace(cov_vmin,cov_vmax,5)
        cb = plt.colorbar(last_h,cax=cax,ticks=ticks,
                          format=FormatStrFormatter('%.3f'))
        cb.set_label('Ray path coverage\n(sum path length / total)',fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.97)
    return fig

with PdfPages(OUT2) as pdf:
    for sta in STATIONS:
        if sta not in sta_xy: continue
        fig = make_coverage_page(sta_key=sta, highlight=sta,
                                  title=f'{sta} — ray coverage density  '
                                        f'(φ_err≤{PHI_ERR_MAX}°, {sta} rays only)')
        pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
        print(f'  {sta}')
    fig = make_coverage_page(sta_key=None, highlight=None,
                              title=f'All stations — ray coverage density  '
                                    f'(φ_err≤{PHI_ERR_MAX}°, all rays combined)')
    pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
    print('  All stations')
print(f'Saved {OUT2}')
