#!/usr/bin/env python3
"""
cosine_similarity_heatmap.py

Plots dt-weighted rose cosine similarity as coloured station markers
on the bathymetry, one panel per time period.

Produces (7-period and annual, vs pre and vs syn):
  cosine_similarity_7period_vs_pre.pdf
  cosine_similarity_7period_vs_syn.pdf
  cosine_similarity_annual_vs_pre.pdf
  cosine_similarity_annual_vs_syn.pdf
"""

import sys, warnings
sys.path.insert(0, '/Users/mhemmett/Seismology/axial-splitting-ml/scripts')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import tifffile
from PIL import Image as PILImage
import os

BASE         = '/Users/mhemmett/Seismology/axial-splitting-ml/results/'
STATION_FILE = '/Users/mhemmett/Seismology/axial-splitting-ml/data/stations_axial.llz'
OUT_DIR      = BASE

PHI_ERR_MAX = 15.0
NBINS       = 36
N_BOOT      = 300

ERUPTION_START = pd.Timestamp('2015-04-24 06:00', tz='UTC')
ERUPTION_END   = pd.Timestamp('2015-05-19 00:00', tz='UTC')

INI_LON, INI_LAT = -130.1, 45.9
KM_DEG_LAT = 111.32
KM_DEG_LON = 111.32 * np.cos(np.radians(INI_LAT))

def ll2xy(lat, lon):
    return ((np.asarray(lon)-INI_LON)*KM_DEG_LON,
            (np.asarray(lat)-INI_LAT)*KM_DEG_LAT)

STATIONS    = ['AXAS2','AXAS1','AXCC1','AXEC1','AXEC2','AXEC3']
STA_DISPLAY = {'AXAS2':'AS2','AXAS1':'AS1','AXCC1':'CC1',
               'AXEC1':'EC1','AXEC2':'EC2','AXEC3':'EC3'}
LABEL_OFFSET= {'AXAS2':(-0.45,-0.55),'AXAS1':(-0.20,-0.55),
               'AXCC1':(-0.75,0.25),'AXEC1':(0.15,0.30),
               'AXEC2':(0.15,0.05),'AXEC3':(0.30,-0.28)}

FILES = {
    'AXAS2':('splitting_results_mldd_2015_2021_axas2.csv','splitting_results_mldd_2022_2026_axas2_all_batches.csv'),
    'AXAS1':('splitting_results_mldd_2015_2021_axas1.csv','splitting_results_mldd_2022_2026_axas1_all_batches.csv'),
    'AXCC1':('splitting_results_mldd_2015_2021_axcc1_all_batches.csv','splitting_results_mldd_2022_2026_axcc1_all_batches.csv'),
    'AXEC1':('splitting_results_mldd_2015_2021_axec1_all_batches.csv','splitting_results_mldd_2022_2026_axec1_all_batches.csv'),
    'AXEC2':('axial-mldd-2015-2021-axec2.csv','splitting_results_mldd_2022_2026_axec2_all_batches.csv'),
    'AXEC3':('splitting_results_mldd_2015_2021_axec3_all_batches.csv','splitting_results_mldd_2022_2026_axec3_all_batches.csv'),
}

# ── Bathymetry ────────────────────────────────────────────────────────────────
BATHY = ('/Users/mhemmett/Downloads/MGDS_Download/AxialSeamount_MBARI/'
         'AxialSummit_MAUV_10Mar2021_All_OverEM302_Topo1m_histEq11.tif')
X_START,X_END,Y_START,Y_END = 4.,12.,0.,12.
PILImage.MAX_IMAGE_PIXELS = None
_p=PILImage.open(BATHY); _t=_p.tag_v2
_olon,_olat=_t[33922][3],_t[33922][4]; _pl,_pb=_t[33550][0],_t[33550][1]
_nc,_nr=_p.size; _p.close()
_c0=max(0,int(((INI_LON+X_START/KM_DEG_LON)-_olon)/_pl)-2)
_c1=min(_nc,int(((INI_LON+X_END/KM_DEG_LON)-_olon)/_pl)+2)
_r0=max(0,int((_olat-(INI_LAT+Y_END/KM_DEG_LAT))/_pb)-2)
_r1=min(_nr,int((_olat-(INI_LAT+Y_START/KM_DEG_LAT))/_pb)+2)
_rgb=tifffile.imread(BATHY)[_r0:_r1,_c0:_c1]
_ds=max(1,max(_rgb.shape[:2])//1024); _rgb=_rgb[::_ds,::_ds]
_gray=np.dot(_rgb[...,:3].astype(np.float32),[0.299,0.587,0.114]).astype(np.uint8)
_ext=[(_olon+_c0*_pl-INI_LON)*KM_DEG_LON,(_olon+_c1*_pl-INI_LON)*KM_DEG_LON,
      (_olat-_r1*_pb-INI_LAT)*KM_DEG_LAT,(_olat-_r0*_pb-INI_LAT)*KM_DEG_LAT]

# ── Stations ──────────────────────────────────────────────────────────────────
_sta = pd.read_csv(STATION_FILE,sep=r'\s+',names=['lon','lat','e','s'],
                   engine='python').set_index('s')
_sta = _sta.loc[[s for s in STATIONS if s in _sta.index]]
_sta['x'],_sta['y'] = ll2xy(_sta['lat'].values,_sta['lon'].values)

# ── Cosine similarity helpers ─────────────────────────────────────────────────
def _rose_hist(phi_vals, wts, nbins=NBINS):
    if len(phi_vals)==0: return np.zeros(nbins)
    p = phi_vals % 360.
    da = np.concatenate([np.deg2rad(p), np.deg2rad((p+180)%360)])
    dw = np.concatenate([wts, wts])
    c,_ = np.histogram(da,bins=np.linspace(0,2*np.pi,nbins+1),weights=dw)
    return c.astype(float)

def _cs(a,b):
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na>0 and nb>0 else float('nan')

def cos_sim_bootstrap(phi,dt,phi_err,dt_err,ref_phi,ref_dt,ref_pe,ref_de,
                      weighted=True,n_boot=N_BOOT,rng=None):
    if rng is None: rng=np.random.default_rng(42)
    N,Nr=len(phi),len(ref_phi)
    if N==0 or Nr==0: return float('nan'),float('nan'),N
    sims=np.empty(n_boot)
    for b in range(n_boot):
        idx=rng.integers(0,N,size=N)
        phi_b=phi[idx]+rng.normal(0,phi_err[idx])
        dt_b=np.abs(dt[idx]+rng.normal(0,dt_err[idx]))
        w_b=dt_b if weighted else np.ones(N)
        h_b=_rose_hist(phi_b,w_b)
        idx_r=rng.integers(0,Nr,size=Nr)
        phi_r=ref_phi[idx_r]+rng.normal(0,ref_pe[idx_r])
        dt_r=np.abs(ref_dt[idx_r]+rng.normal(0,ref_de[idx_r]))
        w_r=dt_r if weighted else np.ones(Nr)
        h_r=_rose_hist(phi_r,w_r)
        sims[b]=_cs(h_b,h_r)
    v=sims[~np.isnan(sims)]
    return (float(np.mean(v)),float(np.std(v)),N) if len(v) else (float('nan'),float('nan'),N)

def subset(df,t0,t1):
    m=(df['t']>=t0) if t0 is not None else pd.Series(True,index=df.index)
    if t1 is not None: m=m&(df['t']<t1)
    return df[m]

# ── Load data ─────────────────────────────────────────────────────────────────
print(f'Loading data (phi_error ≤ {PHI_ERR_MAX}°)...')
dfs={}
for sta,(f1,f2) in FILES.items():
    def _load(f):
        d=pd.read_csv(BASE+f).loc[:,:'dt_error'].dropna()
        return d[(d['dt']>0)&(d['phi_error']<=PHI_ERR_MAX)]
    df=pd.concat([_load(f1),_load(f2)],ignore_index=True)
    df['t']=pd.to_datetime(df['event_datetime'],utc=True)
    df['phi_az']=df['phi']%180.
    dfs[sta]=df
    print(f'  {sta}: {len(df):,}')

all_df=pd.concat(dfs.values(),ignore_index=True)

# ── Build periods ─────────────────────────────────────────────────────────────
post=all_df[all_df['t']>=ERUPTION_END].sort_values('t').reset_index(drop=True)
n=len(post); bounds=[ERUPTION_END]
for i in range(1,5): bounds.append(post['t'].iloc[min(int(round(i*n/5)),n-1)])
bounds.append(None)
def fmt(ts): return ts.strftime('%b %Y') if ts else 'present'
periods_7=[('Pre-eruption',None,ERUPTION_START),
           ('Syn-eruption',ERUPTION_START,ERUPTION_END)]
for i in range(5):
    periods_7.append((f'{fmt(bounds[i])}\n–{fmt(bounds[i+1])}',bounds[i],bounds[i+1]))

periods_a=[('Pre-eruption\n2015',None,ERUPTION_START),
           ('Syn-eruption\n2015',ERUPTION_START,ERUPTION_END),
           ('Post-eruption\n2015',ERUPTION_END,pd.Timestamp('2016-01-01',tz='UTC'))]
for yr in range(2016,2027):
    t0=pd.Timestamp(f'{yr}-01-01',tz='UTC')
    t1=pd.Timestamp(f'{yr+1}-01-01',tz='UTC') if yr<2026 else None
    periods_a.append((str(yr),t0,t1))

# ── Compute all cosine similarities ──────────────────────────────────────────
def compute_sims(periods, weighted, ref_idx):
    """Returns dict: sta -> list of (mu, sig, N) per period."""
    rng = np.random.default_rng(42)
    result = {sta: [] for sta in STATIONS}
    for sta in STATIONS:
        df = dfs[sta]
        period_data = []
        for _,t0,t1 in periods:
            sub = subset(df,t0,t1)
            period_data.append((sub['phi_az'].values.copy(),
                                 sub['dt'].values.copy(),
                                 sub['phi_error'].values.copy(),
                                 sub['dt_error'].values.copy()))
        ref_phi,ref_dt,ref_pe,ref_de = period_data[ref_idx]
        for phi,dt,pe,de in period_data:
            mu,sig,N = cos_sim_bootstrap(phi,dt,pe,de,ref_phi,ref_dt,ref_pe,ref_de,
                                          weighted=weighted,rng=rng)
            result[sta].append((mu,sig,N))
    return result

# ── Figure builder ────────────────────────────────────────────────────────────
CMAP = plt.colormaps['RdYlGn']   # red = low similarity, green = high
VMIN, VMAX = 0.4, 1.0
MARKER_SIZE = 220   # scatter marker size

def make_figure(sims_dict, periods, ref_label, weighted_label, title):
    n_per = len(periods)
    panel_w = 2.0
    fig = plt.figure(figsize=(n_per*panel_w+0.7, 4.5))
    gs  = GridSpec(1, n_per+1, width_ratios=[1]*n_per+[0.05],
                   hspace=0.04, wspace=0.04,
                   left=0.02, right=0.96, top=0.88, bottom=0.04)

    norm = Normalize(vmin=VMIN, vmax=VMAX)

    for ci,(lbl,t0,t1) in enumerate(periods):
        ax = fig.add_subplot(gs[0,ci])
        ax.imshow(_gray,origin='upper',extent=_ext,aspect='auto',
                  cmap='gray',alpha=0.6,zorder=0)

        # Plot each station coloured by cosine similarity
        for sta in STATIONS:
            mu,sig,N = sims_dict[sta][ci]
            sx,sy = float(_sta.loc[sta,'x']),float(_sta.loc[sta,'y'])
            if np.isnan(mu):
                color='gray'
            else:
                color=CMAP(norm(mu))
            ax.scatter([sx],[sy],s=MARKER_SIZE,c=[color],
                       marker='^',edgecolors='black',linewidths=0.8,
                       zorder=8,clip_on=False)
            # Uncertainty as error bar lines
            if not np.isnan(mu) and not np.isnan(sig) and N>0:
                # Map ±1σ in cosine similarity to a visual bar below the triangle
                bar_len = sig * 3.0   # scale to km for visibility
                ax.plot([sx,sx],[sy-0.1,sy-0.1-bar_len],
                        color='black',lw=1.2,zorder=9)
            # Station label
            dx,dy = LABEL_OFFSET.get(sta,(0.12,0.12))
            ax.text(sx+dx,sy+dy,STA_DISPLAY.get(sta,sta),
                    fontsize=5.5,fontweight='bold',zorder=10)
            # N annotation
            if N > 0:
                ax.text(sx,sy+0.35,f'N={N:,}',
                        fontsize=3.8,ha='center',color='#333333',zorder=10)

        ax.set_xlim(4.5,12.0); ax.set_ylim(1.5,10.5)
        ax.set_aspect('equal','box')
        ax.tick_params(labelsize=4,length=2)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_title(f'{lbl}',fontsize=6.5,fontweight='bold',pad=2)

    # Colorbar
    cax = fig.add_subplot(gs[0,n_per])
    sm  = ScalarMappable(cmap=CMAP,norm=norm); sm.set_array([])
    cb  = plt.colorbar(sm,cax=cax)
    cb.set_label('Cosine similarity',fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    fig.suptitle(f'{title}  |  {weighted_label}  |  {ref_label}  '
                 f'(φ_err≤{PHI_ERR_MAX}°, {N_BOOT} bootstraps)',
                 fontsize=8,fontweight='bold')
    return fig

# ── Run all combinations ──────────────────────────────────────────────────────
combos = [
    (periods_7, '7period', 'dt-weighted', True),
    (periods_a, 'annual',  'dt-weighted', True),
]

for ref_name,ref_idx in [('vs Pre-eruption',0),('vs Syn-eruption',1)]:
    ref_tag = 'vs_pre' if ref_idx==0 else 'vs_syn'
    for periods,ptag,wlabel,weighted in combos:
        print(f'Computing {ptag} {wlabel} {ref_name}...')
        sims = compute_sims(periods, weighted, ref_idx)
        fig  = make_figure(sims, periods, ref_name, wlabel,
                           f'Rose cosine similarity — {ptag.replace("_"," ")}')
        out  = os.path.join(OUT_DIR,
               f'cosine_similarity_{ptag}_{ref_tag}.pdf')
        fig.savefig(out,dpi=200,bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out}')

print('\nDone.')
