#!/usr/bin/env python3
"""
sws_source_accumulated.py

Spatial φ (fast direction) and shear anisotropy ΔVs/Vs maps under the
END-MEMBER assumption that splitting accumulates entirely AT THE SOURCE
(within a small crack/damage zone around the hypocenter), rather than being
distributed along the whole ray path as in the tomographic inversion scripts
(sws_tomography_fractional_anisotropy*.py).

Binning is done directly on event hypocenter location (kNN in (x,y) at fixed
depth slice, à la sws_phi15_filter_plots.py's xyzd2mesh) -- there is no ray
tracing and no tomographic A-matrix here.

ΔVs/Vs conversion (MODELING ASSUMPTION -- flagged for human review):
  Because all of dt is assumed to accumulate over a short source-region
  length scale L_SOURCE_KM instead of the full ray path, the local
  anisotropy-per-km is

      m_source = dt / L_SOURCE_KM      [s/km]

  and then, exactly as in sws_tomography_fractional_anisotropy_ultra_strict.py:

      R = 2 / (m_source · Vs)          Vs from the Baillard 3D model at the
                                        hypocenter (vs_at)
      a = ΔVs/Vs = −R + √(4 + R²)      positive root, a ∈ (0, 2)

  L_SOURCE_KM is NOT measured -- it is an assumed characteristic source-zone
  thickness. Changing it rescales every ΔVs/Vs map by a fixed factor (smaller
  L_SOURCE_KM -> larger inferred ΔVs/Vs for the same observed dt). Default
  below (0.5 km) matches the MIN_PATH_KM coverage scale used elsewhere in this
  codebase; there is no independent constraint on it in this analysis.

Outputs:
  sws_source_accumulated_phi_7period_dt004_contourf.pdf
  sws_source_accumulated_dvsvs_7period_dt004_contourf.pdf
  sws_source_accumulated_phi_annual_dt004_contourf.pdf
  sws_source_accumulated_dvsvs_annual_dt004_contourf.pdf
  sws_source_accumulated_phi_hourly48_dt004_contourf.pdf
  sws_source_accumulated_dvsvs_hourly48_dt004_contourf.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from PIL import Image as PILImage
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.lines as mlines

from baillard_velocity import vs_at

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT_DIR      = BASE
BATHY        = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
                'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')

# FIX 6 harmonized QC cuts (same as the *_ultra_strict / dt001 tomography family).
PHI_ERR_MAX = 20.0   # deg
DT_ERR_MAX  = 0.04   # s

L_SOURCE_KM = 0.5     # ASSUMED source-region thickness -- see module docstring.

INI_LON, INI_LAT = -130.1, 45.9
KM_PER_DEG_LAT   = 111.32
KM_PER_DEG_LON   = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_PER_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_PER_DEG_LAT)

def xy2ll(x_km, y_km):
    return (INI_LON + np.asarray(x_km)/KM_PER_DEG_LON,
            INI_LAT + np.asarray(y_km)/KM_PER_DEG_LAT)

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
# For the 48x1hr eruption-window periods: include ANY grid node with at
# least one event ("any data"), not a minimum-count display cut.
COUNT_MIN_HOURLY=1
GAUSS_SIGMA=0.2/STEP
Z_HALF_WIDTH  = 0.3125
DEPTH_SLICES  = [0.3125, 0.9375, 1.5625, 2.1875]
DEPTH_SLICE_LABELS = [
    'z=0.0–0.625 km', 'z=0.625–1.25 km', 'z=1.25–1.875 km', 'z=1.875–2.5 km',
]

FILES = {
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv','splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv','splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv','splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv','splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv','splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv','splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

# ── Bathymetry ─────────────────────────────────────────────────────────────────
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

def _add_stations(ax):
    for sta,row in _sta.iterrows():
        ax.plot(row['x'],row['y'],'^',ms=6,mfc='#FFD700',mec='k',mew=0.7,zorder=12)
        dx,dy=LABEL_OFFSET_KM.get(sta,(0.12,0.12))
        ax.text(row['x']+dx,row['y']+dy,STA_DISPLAY.get(sta,sta),fontsize=5.5,zorder=13)

def _add_kidiwela(ax):
    for s in KIDIWELA_SOURCES:
        ax.plot(s['x'],s['y'],'o',ms=4,mfc='red',mec='k',mew=0.5,zorder=14)
        ax.text(s['x']+0.12,s['y']+0.12,s['label'],
                fontsize=5,color='red',fontweight='bold',zorder=15)

# ── Time periods ───────────────────────────────────────────────────────────────
def build_7_periods(all_df):
    post = all_df[all_df['t'] >= ERUPTION_END].sort_values('t').reset_index(drop=True)
    n = len(post); bounds = [ERUPTION_END]
    for i in range(1, 5):
        idx = min(int(round(i*n/5)), n-1)
        bounds.append(post['t'].iloc[idx])
    bounds.append(None)
    def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
    pds = [('Pre-eruption', None, ERUPTION_START),
           ('Syn-eruption', ERUPTION_START, ERUPTION_END)]
    for i in range(5):
        pds.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}', bounds[i], bounds[i+1]))
    return pds

def build_annual_periods():
    pds = [('Pre-eruption\n2015', None, ERUPTION_START),
           ('Syn-eruption\n2015', ERUPTION_START, ERUPTION_END),
           ('Post-eruption\n2015', ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC'))]
    for yr in range(2016, 2027):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr+1}-01-01', tz='UTC') if yr < 2026 else None
        pds.append((str(yr), t0, t1))
    return pds

def build_hourly_periods():
    """48 one-hour bins: 24h before eruption onset through the first 24h of
    the eruption (ERUPTION_START ± 24h)."""
    t_lo = ERUPTION_START - pd.Timedelta(hours=24)
    pds = []
    for i in range(48):
        t0 = t_lo + pd.Timedelta(hours=i); t1 = t0 + pd.Timedelta(hours=1)
        rel = i - 24
        pds.append((f'H{rel:+03d}\n{t0.strftime("%m/%d %H:%M")}', t0, t1))
    return pds

def subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None: m = m & (df['t'] < t1)
    return df[m]

# ── Hypocenter-direct binning ──────────────────────────────────────────────────
_xn = np.arange(X_START, X_END+STEP*.5, STEP)
_yn = np.arange(Y_START, Y_END+STEP*.5, STEP)
_Xg, _Yg = np.meshgrid(_xn, _yn)

def m_to_dvs_over_vs(m, vs):
    """Same conversion as sws_tomography_fractional_anisotropy_ultra_strict.py,
    applied here to the SOURCE-region anisotropy m = dt/L_SOURCE_KM."""
    R = 2.0 / np.maximum(m*vs, 1e-6)
    a = -R + np.sqrt(4.0 + R**2)
    return np.clip(a, 0., 2.)

def event_dvsvs(x, y, z, dt):
    """Per-event ΔVs/Vs using each event's OWN hypocenter Vs (not a shared
    grid-depth Vs) -- required so depth-merged (all-depths-in-one) panels are
    correct: each event's anisotropy is referenced to the velocity at its own
    depth before being spatially binned/medianed."""
    vs = np.clip(vs_at(x, y, z), 0.3, 5.0)
    return m_to_dvs_over_vs(dt/L_SOURCE_KM, vs)

def xyzd2mesh(x, y, z, d_phi, d_a, z0):
    """Bin observations directly at hypocenter (x,y). If z0 is not None,
    restrict to |z-z0|<=Z_HALF_WIDTH (single depth slice); if z0 is None, use
    ALL events regardless of depth (full-depth-column merge, for the hourly
    stacked panels). Returns circular-mean PHI [deg], median per-event ΔVs/Vs,
    and event count per grid node."""
    mz = np.ones(len(x), dtype=bool) if z0 is None else (np.abs(z-z0) <= Z_HALF_WIDTH)
    xs,ys,phi_s,a_s = x[mz],y[mz],d_phi[mz],d_a[mz]
    PHI = np.zeros(_Xg.shape); A = np.full(_Xg.shape, np.nan); CNT = np.zeros(_Xg.shape,dtype=int)
    if len(xs) < 3: return PHI,A,CNT
    tree = cKDTree(np.column_stack([xs,ys]))
    gp = np.column_stack([_Xg.ravel(),_Yg.ravel()])
    nbrs = tree.query_ball_point(gp, DIS_LIM)
    PHI_flat=PHI.ravel(); A_flat=A.ravel(); CNT_flat=CNT.ravel()
    for k,nb in enumerate(nbrs):
        if not nb: continue
        pts = np.column_stack([xs[nb],ys[nb]]); gpi = gp[k]
        dist = np.hypot(pts[:,0]-gpi[0], pts[:,1]-gpi[1])
        sel = np.array(nb)[np.argsort(dist)[:NUM_LIM]]
        CNT_flat[k] = len(sel)
        ang = 2.0*np.radians(phi_s[sel])
        PHI_flat[k] = float(np.degrees(np.arctan2(
            np.mean(np.sin(ang)),np.mean(np.cos(ang)))/2)%180)
        A_flat[k] = float(np.median(a_s[sel]))
    return gaussian_filter(PHI,GAUSS_SIGMA), A, CNT

# ── Page builders ───────────────────────────────────────────────────────────────
COT66 = None  # faults omitted here (kept minimal vs. the tomography scripts)

def make_phi_page(periods, df_or_dict, title, count_min=COUNT_MIN, merge_depth=False):
    cmap_phi = plt.colormaps['hsv_r']
    if isinstance(df_or_dict, dict):
        subs = [pd.concat([subset(df,t0,t1) for df in df_or_dict.values()],
                           ignore_index=True) for _,t0,t1 in periods]
    else:
        subs = [subset(df_or_dict,t0,t1) for _,t0,t1 in periods]
    n_eqs = [len(s) for s in subs]

    n_per = len(periods)
    depth_rows = [None] if merge_depth else list(DEPTH_SLICES)
    n_dep = len(depth_rows)
    panel_w = 1.8 if n_per > 7 else 2.2
    fig = plt.figure(figsize=(n_per*panel_w+0.6, n_dep*3.0+0.3))
    gs  = GridSpec(n_dep, n_per+1, width_ratios=[1]*n_per+[0.04], hspace=0.04, wspace=0.04)
    last_h = None
    for ri,z0 in enumerate(depth_rows):
        for ci,((lbl,_,__),sub,neq) in enumerate(zip(periods,subs,n_eqs)):
            ax = fig.add_subplot(gs[ri,ci]); _bathy(ax)
            if len(sub) >= count_min:
                PHI,A,CNT = xyzd2mesh(sub['x'].values,sub['y'].values,sub['z'].values,
                                       sub['phi_az'].values,sub['a_dvsvs'].values,z0)
                mask = CNT >= count_min
                mp = np.ma.masked_where(~mask, PHI)
                if not mp.mask.all():
                    ax.contourf(_Xg,_Yg,mp,levels=np.linspace(0.,180.,19),
                                cmap=cmap_phi,vmin=0,vmax=180,extend='neither')
                    last_h = ScalarMappable(cmap=cmap_phi,norm=Normalize(0,180))
                    last_h.set_array([])
            _add_stations(ax); _add_kidiwela(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{lbl}\nN={neq:,}',fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel('All depths 0–2.5 km' if z0 is None else DEPTH_SLICE_LABELS[ri],fontsize=7)
    if last_h:
        cax = fig.add_subplot(gs[:,n_per])
        cb = plt.colorbar(last_h,cax=cax,ticks=np.arange(0,181,30),
                          format=FormatStrFormatter('%.0f'))
        cb.set_label('φ [° from N]',fontsize=9); cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.97)
    return fig

def make_dvsvs_page(periods, df_or_dict, title, vmax=None, count_min=COUNT_MIN, merge_depth=False):
    cmap_a = plt.colormaps['Blues']
    if isinstance(df_or_dict, dict):
        subs = [pd.concat([subset(df,t0,t1) for df in df_or_dict.values()],
                           ignore_index=True) for _,t0,t1 in periods]
    else:
        subs = [subset(df_or_dict,t0,t1) for _,t0,t1 in periods]
    n_eqs = [len(s) for s in subs]

    n_per = len(periods)
    depth_rows = [None] if merge_depth else list(DEPTH_SLICES)
    n_dep = len(depth_rows)
    panel_w = 1.8 if n_per > 7 else 2.2
    fig = plt.figure(figsize=(n_per*panel_w+0.6, n_dep*3.0+0.3))
    gs  = GridSpec(n_dep, n_per+1, width_ratios=[1]*n_per+[0.04], hspace=0.04, wspace=0.04)
    last_h = None
    a_fields = {}
    for ri,z0 in enumerate(depth_rows):
        for ci,((lbl,_,__),sub,neq) in enumerate(zip(periods,subs,n_eqs)):
            if len(sub) >= count_min:
                PHI,A,CNT = xyzd2mesh(sub['x'].values,sub['y'].values,sub['z'].values,
                                       sub['phi_az'].values,sub['a_dvsvs'].values,z0)
                mask = CNT >= count_min
                a_fields[(ri,ci)] = (np.ma.masked_where(~mask,A), mask)
    all_a = [v[0].compressed() for v in a_fields.values() if v[1].any()]
    a_vmax = vmax if vmax is not None else (
        float(np.percentile(np.concatenate(all_a),99)) if all_a else 0.3)
    levels = np.linspace(0.,a_vmax,15)
    for ri,z0 in enumerate(depth_rows):
        for ci,((lbl,_,__),sub,neq) in enumerate(zip(periods,subs,n_eqs)):
            ax = fig.add_subplot(gs[ri,ci]); _bathy(ax)
            entry = a_fields.get((ri,ci))
            if entry is not None and entry[1].any():
                ax.contourf(_Xg,_Yg,entry[0],levels=levels,cmap=cmap_a,
                            vmin=0,vmax=a_vmax,extend='neither')
                last_h = ScalarMappable(cmap=cmap_a,norm=Normalize(0,a_vmax))
                last_h.set_array([])
            _add_stations(ax); _add_kidiwela(ax)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0: ax.set_title(f'{lbl}\nN={neq:,}',fontsize=7.5,fontweight='bold',pad=1)
            if ci==0: ax.set_ylabel('All depths 0–2.5 km' if z0 is None else DEPTH_SLICE_LABELS[ri],fontsize=7)
    if last_h:
        cax = fig.add_subplot(gs[:,n_per])
        cb = plt.colorbar(last_h,cax=cax,ticks=np.linspace(0,a_vmax,5),
                          format=FormatStrFormatter('%.2f'))
        cb.set_label('Shear anisotropy ΔVs/Vs (source-accumulated)',fontsize=9)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title,fontsize=11,fontweight='bold',y=0.97)
    return fig, a_vmax

def paginate_and_save(periods, df_or_dict, page_func, title, out_path, chunk=7, **kw):
    print(f'\nWriting {os.path.basename(out_path)}...')
    with PdfPages(out_path) as pdf:
        for start in range(0,len(periods),chunk):
            res = page_func(periods[start:start+chunk], df_or_dict, title, **kw)
            fig = res[0] if isinstance(res,tuple) else res
            pdf.savefig(fig,dpi=300,bbox_inches='tight'); plt.close(fig)
    print(f'Saved {out_path}')

# ── Load data ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Loading data (phi_error < {PHI_ERR_MAX}°, dt_error < {DT_ERR_MAX}s)...')
    _sta = pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','elev_km','station'],
                       engine='python').set_index('station')
    _sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
    _sta['x'],_sta['y'] = ll2xy(_sta['lat'].values,_sta['lon'].values)

    dfs = {}
    for sta,(f1,f2) in FILES.items():
        def _load(f):
            d = pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
            d = d[(d['dt']>0)&(d['phi_error']<PHI_ERR_MAX)&(d['dt_error']<DT_ERR_MAX)]
            return d
        df = pd.concat([_load(f1),_load(f2)],ignore_index=True)
        df['x'],df['y'] = ll2xy(df['event_lat'].values,df['event_lon'].values)
        df['z']      = df['event_depth'].values
        df['t']      = pd.to_datetime(df['event_datetime'],utc=True)
        df['phi_az'] = df['phi'] % 180.0
        df['a_dvsvs'] = event_dvsvs(df['x'].values,df['y'].values,df['z'].values,df['dt'].values)
        dfs[sta] = df
        print(f'  {sta}: {len(df):,}')

    all_df = pd.concat(dfs.values(), ignore_index=True)
    print(f'  TOTAL retained: {len(all_df):,}')

    periods_7      = build_7_periods(all_df)
    periods_annual = build_annual_periods()
    periods_hourly = build_hourly_periods()

    tag = f'(φ_err<{PHI_ERR_MAX}°, dt_err<{DT_ERR_MAX}s, source-accumulated, L={L_SOURCE_KM} km)'

    paginate_and_save(periods_7, all_df, make_phi_page,
        f'Source-accumulated φ (7 periods)  {tag}',
        os.path.join(OUT_DIR,'sws_source_accumulated_phi_7period_dt004_contourf.pdf'), chunk=7)

    paginate_and_save(periods_annual, all_df, make_phi_page,
        f'Source-accumulated φ (annual)  {tag}',
        os.path.join(OUT_DIR,'sws_source_accumulated_phi_annual_dt004_contourf.pdf'), chunk=7)

    paginate_and_save(periods_hourly, all_df, make_phi_page,
        f'Source-accumulated φ (hourly, eruption onset window, all depths)  {tag}',
        os.path.join(OUT_DIR,'sws_source_accumulated_phi_hourly48_dt004_contourf.pdf'),
        chunk=8, count_min=COUNT_MIN_HOURLY, merge_depth=True)

    # dt / ΔVs-Vs: fix a single vmax across 7-period + annual + hourly for comparability
    print('\nDetermining shared ΔVs/Vs color scale across 7-period and annual...')
    _, vmax_7 = make_dvsvs_page(periods_7, all_df,
        f'Source-accumulated shear anisotropy ΔVs/Vs (7 periods)  {tag}'); plt.close('all')
    _, vmax_a = make_dvsvs_page(periods_annual, all_df,
        f'Source-accumulated shear anisotropy ΔVs/Vs (annual)  {tag}'); plt.close('all')
    shared_vmax = max(vmax_7, vmax_a)

    paginate_and_save(periods_7, all_df, make_dvsvs_page,
        f'Source-accumulated shear anisotropy ΔVs/Vs (7 periods)  {tag}',
        os.path.join(OUT_DIR,'sws_source_accumulated_dvsvs_7period_dt004_contourf.pdf'),
        chunk=7, vmax=shared_vmax)

    paginate_and_save(periods_annual, all_df, make_dvsvs_page,
        f'Source-accumulated shear anisotropy ΔVs/Vs (annual)  {tag}',
        os.path.join(OUT_DIR,'sws_source_accumulated_dvsvs_annual_dt004_contourf.pdf'),
        chunk=7, vmax=shared_vmax)

    paginate_and_save(periods_hourly, all_df, make_dvsvs_page,
        f'Source-accumulated shear anisotropy ΔVs/Vs (hourly, eruption onset window, all depths)  {tag}',
        os.path.join(OUT_DIR,'sws_source_accumulated_dvsvs_hourly48_dt004_contourf.pdf'),
        chunk=8, vmax=shared_vmax, count_min=COUNT_MIN_HOURLY, merge_depth=True)

    print('\nDone.')
