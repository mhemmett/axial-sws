#!/usr/bin/env python3
"""
sws_temporal_annual.py

Annual-period version of the sws_temporal spatial maps.
Time periods: Pre-eruption | Syn-eruption | Post-eruption 2015 | 2016 | 2017 ... 2026

Produces:
  sws_temporal_combined_phi_annual.pdf
  sws_temporal_combined_dt_annual.pdf
  sws_temporal_combined_phi_annual_filtered.pdf   (dt_error < 0.1 s)
  sws_temporal_combined_dt_annual_filtered.pdf
Each PDF has one page per station + one all-stations page.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.ma as ma
import os
import tifffile
from PIL import Image as PILImage
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

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

STATIONS = ['AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3']
STA_DISPLAY = {'AXAS1':'AS1','AXAS2':'AS2','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET_KM = {
    'AXAS1':(-0.20,-0.55), 'AXAS2':(-0.45,-0.55), 'AXCC1':(-0.20, 0.25),
    'AXEC1':( 0.15, 0.30), 'AXEC2':( 0.15, 0.05), 'AXEC3':( 0.30,-0.28),
}

X_START, X_END = 4.0, 12.0
Y_START, Y_END = 0.0, 12.0
STEP           = 0.1
DIS_LIM        = 0.3
NUM_LIM        = 100
COUNT_MIN      = 15
GAUSS_SIGMA    = 0.2 / STEP
DEPTH_SLICES   = [0.5, 1.0, 1.5]
Z_HALF_WIDTH   = 0.3

DIC_PLOT_TEMPORAL = {
    'phi': dict(vmin=0,   vmax=180,  cmap='hsv',   label='φ [° from N]', step=30,    fmt='%.0f'),
    'dt':  dict(vmin=0,   vmax=0.15, cmap='Blues', label='dt [s]',       step=0.025, fmt='%.3f'),
}

# ── Annual time periods ───────────────────────────────────────────────────────

def build_annual_periods():
    periods = [
        ('Pre-eruption\n2015',     None,            ERUPTION_START),
        ('Syn-eruption\n2015',     ERUPTION_START,  ERUPTION_END),
        ('Post-eruption\n2015',    ERUPTION_END,    pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for yr in range(2016, 2027):
        t0 = pd.Timestamp(f'{yr}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{yr+1}-01-01', tz='UTC') if yr < 2026 else None
        periods.append((str(yr), t0, t1))
    return periods

def subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None:
        m = m & (df['t'] < t1)
    return df[m]

# ── Bathymetry (grayscale) ────────────────────────────────────────────────────

BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
         'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
PILImage.MAX_IMAGE_PIXELS = None
_p = PILImage.open(BATHY); _t = _p.tag_v2
_olon,_olat = _t[33922][3],_t[33922][4]
_pl,_pb     = _t[33550][0],_t[33550][1]
_nc,_nr = _p.size; _p.close()
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

def _add_stations(ax, highlight=None):
    for sta, row in _sta.iterrows():
        mfc = '#FFD700' if (highlight is None or sta == highlight) else 'white'
        ax.plot(row['x'],row['y'],'^',ms=6,mfc=mfc,mec='k',mew=0.7,zorder=12)
        dx,dy = LABEL_OFFSET_KM.get(sta,(0.12,0.12))
        ax.text(row['x']+dx,row['y']+dy,STA_DISPLAY.get(sta,sta),fontsize=5.5,zorder=13)

# ── Binning ───────────────────────────────────────────────────────────────────

def xyzd2mesh(x,y,z,d_phi,d_dt,z0):
    mz = np.abs(z-z0) <= Z_HALF_WIDTH
    xs,ys,phi_s,dt_s = x[mz],y[mz],d_phi[mz],d_dt[mz]
    xn=np.arange(X_START,X_END+STEP*.5,STEP); yn=np.arange(Y_START,Y_END+STEP*.5,STEP)
    X,Y=np.meshgrid(xn,yn); ny,nx=X.shape
    PHI=np.zeros((ny,nx)); DT=np.zeros((ny,nx)); CNT=np.zeros((ny,nx),dtype=int)
    if len(xs)<3: return X,Y,PHI,DT,CNT
    tree=cKDTree(np.column_stack([xs,ys]))
    gp=np.column_stack([X.ravel(),Y.ravel()])
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
    return X,Y,PHI,DT,CNT

# ── Figure builder ────────────────────────────────────────────────────────────

def make_annual_page(param, df_or_dict, periods, highlight, title, dt_vmax=None):
    dp    = dict(DIC_PLOT_TEMPORAL[param])
    if param == 'dt' and dt_vmax is not None:
        dp['vmax'] = dt_vmax
        dp['step'] = round(dt_vmax/6, 3)
    cmap   = plt.colormaps[dp['cmap']]
    vmin, vmax = dp['vmin'], dp['vmax']
    levels = np.linspace(vmin, vmax, 15)
    n_per  = len(periods)
    n_dep  = len(DEPTH_SLICES)

    # Pre-compute subsets and N counts
    subs, n_eqs = [], []
    for _, t0, t1 in periods:
        if isinstance(df_or_dict, dict):
            frames = [subset(df, t0, t1) for df in df_or_dict.values()]
            s = pd.concat(frames, ignore_index=True)
        else:
            s = subset(df_or_dict, t0, t1)
        subs.append(s)
        n_eqs.append(s['event_datetime'].nunique())

    panel_w = 1.8   # narrower panels for 14 columns
    panel_h = 3.0
    fig = plt.figure(figsize=(n_per*panel_w+0.6, n_dep*panel_h+0.3))
    gs  = GridSpec(n_dep, n_per+1,
                   width_ratios=[1]*n_per+[0.04],
                   hspace=0.04, wspace=0.04)

    last_h = None
    for ri, z0 in enumerate(DEPTH_SLICES):
        for ci, ((lbl,_,__), sub, neq) in enumerate(zip(periods, subs, n_eqs)):
            ax = fig.add_subplot(gs[ri, ci]); _bathy(ax)
            if len(sub) >= COUNT_MIN:
                Xg,Yg,PHI,DT,CNT = xyzd2mesh(
                    sub['x'].values, sub['y'].values, sub['z'].values,
                    sub['phi_az'].values, sub['dt'].values, z0)
                D = PHI if param=='phi' else DT
                D = gaussian_filter(D, sigma=GAUSS_SIGMA)
                D = ma.array(D, mask=CNT<COUNT_MIN)
                last_h = ax.contourf(Xg,Yg,D,levels=levels,cmap=cmap,
                                     vmin=vmin,vmax=vmax,
                                     extend='neither',zorder=2,alpha=0.85)
            else:
                ax.text(.5,.5,'–',ha='center',va='center',
                        transform=ax.transAxes,fontsize=8,color='gray')
            _add_stations(ax, highlight)
            ax.set_xlim(X_START,X_END); ax.set_ylim(Y_START,Y_END)
            ax.set_aspect('equal','box'); ax.tick_params(labelsize=4)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            if ri==0:
                ax.set_title(f'{lbl}\nN={neq:,}', fontsize=6.5,
                             fontweight='bold', pad=1)
            if ci==0: ax.set_ylabel(f'z={z0} km', fontsize=8)

    if last_h:
        cax   = fig.add_subplot(gs[:,n_per])
        ticks = np.arange(vmin, vmax+dp['step']*.5, dp['step'])
        cb    = plt.colorbar(last_h, cax=cax, ticks=ticks,
                             format=FormatStrFormatter(dp['fmt']))
        cb.set_label(dp['label'], fontsize=9); cb.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.97)
    return fig

# ── Run ───────────────────────────────────────────────────────────────────────

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

periods = build_annual_periods()
print(f'{len(periods)} annual time periods:')
for lbl,t0,t1 in periods:
    t0s = t0.strftime('%Y-%m-%d') if t0 else 'start'
    t1s = t1.strftime('%Y-%m-%d') if t1 else 'present'
    print(f'  {lbl.replace(chr(10)," "):25} {t0s} → {t1s}')

for dt_err_max, suffix in [(None, ''), (0.1, '_filtered')]:
    label = f'(dt_error < {dt_err_max} s)' if dt_err_max else '(all data)'
    print(f'\nLoading data {label}...')

    _sta = pd.read_csv(STATION_FILE, sep=r'\s+',
                       names=['lon','lat','elev_km','station'],
                       engine='python').set_index('station')
    _sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
    _sta['x'],_sta['y'] = ll2xy(_sta['lat'].values, _sta['lon'].values)

    def _load(f):
        d = pd.read_csv(BASE+f)
        d = d.loc[:, :'dt_error'].dropna()
        d = d[d['dt']>0]
        if dt_err_max: d = d[d['dt_error']<dt_err_max]
        return d

    dfs = {}
    for sta,(f1,f2) in FILES.items():
        df = pd.concat([_load(f1), _load(f2)], ignore_index=True)
        df['x'],df['y'] = ll2xy(df['event_lat'].values,df['event_lon'].values)
        df['z']      = df['event_depth'].values
        df['t']      = pd.to_datetime(df['event_datetime'],utc=True)
        df['phi_az'] = df['phi'] % 180.0
        dfs[sta]     = df
        print(f'  {sta}: {len(df):,} events')

    # dt vmax from data
    dt_vmax_all = float(np.percentile(
        np.concatenate([df['dt'].values for df in dfs.values()]), 99))
    print(f'  dt vmax (99th pct): {dt_vmax_all:.3f} s')

    for param in ['phi', 'dt']:
        dp = DIC_PLOT_TEMPORAL[param]
        out = os.path.join(OUT_DIR, f'sws_temporal_combined_{param}_annual{suffix}.pdf')
        with PdfPages(out) as pdf:
            for sta in STATIONS:
                vmax = 180 if param=='phi' else float(np.percentile(dfs[sta]['dt'].values,99))
                fig = make_annual_page(
                    param, dfs[sta], periods,
                    highlight=sta,
                    title=f'{sta} — {dp["label"]} annual {label}',
                    dt_vmax=vmax if param=='dt' else None,
                )
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig); print(f'  Added {sta}')

            # All-stations
            fig = make_annual_page(
                param, dfs, periods,
                highlight=None,
                title=f'All stations — {dp["label"]} annual {label}',
                dt_vmax=dt_vmax_all if param=='dt' else None,
            )
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig); print(f'  Added ALL stations')

        print(f'Saved {out}')

print('\nDone.')
