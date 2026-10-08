#!/usr/bin/env python3
"""
rose_map_inflation_periods_ccal_grade3.py

Five map panels, one per post-eruption EQUAL-INFLATION period defined on the Central Caldera
(CCAL) BOTPT de-tided uplift (bpr_inflation_periods_ccal.build_inflation_based_periods -- the
5 bins carry equal amounts of uplift, not equal time or equal event counts). Each panel shows the
full caldera rim on low-contrast grayscale bathymetry with a fast-direction rose at each station
(station name with a bold N= directly beneath it; no triangle, no rose outline circle), from the Grade-3
windowcheck dataset:

    SNR >= 2.0, Q_w >= 0.75, dt_err <= 0.05 s, dt <= T_dom/2, phi_err <= 20 deg

Roses use the same _draw_rose (36 bins, axial: each phi also plotted at phi+180, unit weights,
each rose scaled to its own maximum) in the same 5-step post-eruption period colours as the
rose-plot figures (_period_colors; period 1 darkened slightly for contrast on the
bathymetry), with a thin black outline on each wedge.

ROSE SIZE: one radius for every station, as large as the layout allows without overlap
(ROSE_RADIUS_KM, limited by AXAS1-AXAS2 at 1.15 km and AXEC1-AXEC2 at 1.19 km). AXEC2 and
AXEC3 are only 0.54 km apart, so at that radius their roses are pushed apart symmetrically along
the line joining them, just far enough to clear (relax_positions); each displaced rose gets a
thin leader line to a small dot at the station's true position.

Layout is 2 x 3; the sixth panel is the CCAL uplift curve with the five periods shaded in their
rose colours.

AXEC2's known-instrumental unleveled window (UNLEVEL_START..UNLEVEL_END, 2021-07 to 2022-09) is
dropped, as in the regional geodetic figures, so the drift does not bend one period's rose.

Output: rose_map_inflation_periods_ccal_grade3.pdf / .png
"""

import os
import itertools
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rose_7period_regions_windowcheck_grade3 import GRADE, load_station_raw, apply_grade, _subset
from rose_7period_6stations_newdata_snr_grades import STATION_ORDER, _draw_rose, _period_colors
from axec2_uplift_phi_cosine_vs_time import UNLEVEL_START, UNLEVEL_END
from animate_arctan_stress_vectors import load_bathy_gray, load_station_xy
import bpr_inflation_periods_ccal as ccal_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, 'rose_map_inflation_periods_ccal_grade3')

# Map window: the full caldera rim (rim spans ~4.5-10.5 km E, ~1.8-10.4 km N in this projection).
X_LIM = (3.8, 11.2)
Y_LIM = (1.5, 10.8)
BATHY_ALPHA = 0.45          # "lower intensity" grayscale
PERIOD1_COLOR = '#89BFDB'   # slightly darker than the ramp's #ADD8E6

ROSE_RADIUS_KM = 0.54       # < half of the closest undisplaced pairs (1.15 / 1.19 km)
ROSE_GAP_KM = 0.04          # clearance kept between displaced roses
LABEL_PAD_KM = 0.06
SIDE_LABELS = {'AXEC2': 'right', 'AXEC3': 'below'}  # displaced pair: AXEC2 labels outward,
                                                   # AXEC3 below (left would hit AXAS1)


def relax_positions(xy, r, gap, n_iter=200):
    """Push overlapping rose centres apart symmetrically along the line joining them."""
    pos = {k: np.array(v, dtype=float) for k, v in xy.items()}
    need = 2.0 * r + gap
    for _ in range(n_iter):
        moved = False
        for a, b in itertools.combinations(pos, 2):
            d = pos[b] - pos[a]
            dist = float(np.hypot(*d))
            if dist < need:
                push = (need - dist) / 2.0 * d / max(dist, 1e-9)
                pos[a] -= push
                pos[b] += push
                moved = True
        if not moved:
            break
    return pos


def main():
    periods = ccal_infl.build_inflation_based_periods()[2:]     # the 5 post-eruption bins
    _dd, _infl, inflation_roll, _rt, _rd = ccal_infl.load_daily_series()
    colors = _period_colors(7)[2:]
    # Period 1's #ADD8E6 washes out against the gray bathymetry: darken it slightly here only
    # (the shared ramp used by the other rose figures is left unchanged).
    colors[0] = PERIOD1_COLOR

    data = {}
    for sta in STATION_ORDER:
        df = apply_grade(load_station_raw(sta), GRADE)
        if sta == 'AXEC2':
            df = df[~((df['t'] >= UNLEVEL_START) & (df['t'] < UNLEVEL_END))]
        data[sta] = df

    xy = {sta: np.array(load_station_xy(sta)) for sta in STATION_ORDER}
    pos = relax_positions(xy, ROSE_RADIUS_KM, ROSE_GAP_KM)
    for sta in STATION_ORDER:
        shift = float(np.hypot(*(pos[sta] - xy[sta])))
        if shift > 0.01:
            print(f'  {sta} rose displaced {shift:.2f} km from the station')

    x0, y0 = np.mean(X_LIM), np.mean(Y_LIM)
    half = max(X_LIM[1] - X_LIM[0], Y_LIM[1] - Y_LIM[0]) / 2.0 + 0.1
    gray, extent = load_bathy_gray(x0, y0, half)

    fig, axes2d = plt.subplots(2, 3, figsize=(17, 14.2))
    axes = axes2d.ravel()
    u_all = inflation_roll.dropna()
    for k, (ax, (label, t0, t1)) in enumerate(zip(axes[:5], periods)):
        ax.imshow(gray, origin='upper', extent=extent, aspect='auto', cmap='gray',
                  alpha=BATHY_ALPHA, zorder=0)
        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        ax.set_aspect('equal')

        u_a = float(u_all.loc[t0:].iloc[0])
        u_b = float(u_all.loc[:t1].iloc[-1]) if t1 is not None else float(u_all.iloc[-1])

        for sta in STATION_ORDER:
            (sx, sy), (cx, cy) = xy[sta], pos[sta]
            if np.hypot(cx - sx, cy - sy) > 0.01:
                ax.plot([sx, cx], [sy, cy], color='0.25', lw=0.8, zorder=4)
                ax.plot(sx, sy, 'o', ms=3.5, color='0.15', zorder=4)
            sub = _subset(data[sta], t0, t1)
            rax = ax.inset_axes([cx - ROSE_RADIUS_KM, cy - ROSE_RADIUS_KM,
                                 2 * ROSE_RADIUS_KM, 2 * ROSE_RADIUS_KM],
                                transform=ax.transData, projection='polar', zorder=5)
            _draw_rose(rax, sub['phi_az'].values, np.ones(len(sub)), colors[k])
            rax.patch.set_alpha(0.0)
            rax.spines['polar'].set_visible(False)
            for bar in rax.patches:             # thin black outline on each wedge
                bar.set_edgecolor('black')
                bar.set_linewidth(0.3)

            side = SIDE_LABELS.get(sta)
            name_kw = dict(fontsize=9.5, fontweight='bold', zorder=6,
                           bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
            n_kw = dict(fontsize=8.5, fontweight='bold', color='0.15', zorder=6)
            if side == 'below':
                ax.text(cx, cy - ROSE_RADIUS_KM - LABEL_PAD_KM, sta, ha='center', va='top',
                        **name_kw)
                ax.text(cx, cy - ROSE_RADIUS_KM - LABEL_PAD_KM - 0.28, f'N={len(sub)}',
                        ha='center', va='top', **n_kw)
            elif side is None:
                # Name on top, N= directly beneath it, both above the rose.
                ax.text(cx, cy + ROSE_RADIUS_KM + LABEL_PAD_KM + 0.26, sta, ha='center',
                        va='bottom', **name_kw)
                ax.text(cx, cy + ROSE_RADIUS_KM + LABEL_PAD_KM, f'N={len(sub)}', ha='center',
                        va='bottom', **n_kw)
            else:
                sgn = 1.0 if side == 'right' else -1.0
                ha = 'left' if side == 'right' else 'right'
                lx = cx + sgn * (ROSE_RADIUS_KM + LABEL_PAD_KM)
                ax.text(lx, cy + 0.07, sta, ha=ha, va='bottom', **name_kw)
                ax.text(lx, cy - 0.07, f'N={len(sub)}', ha=ha, va='top', **n_kw)

        ax.set_title(f'Period {k + 1}: {label}\nuplift {u_a:.2f} → {u_b:.2f} m',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('East (km)')
        ax.set_ylabel('North (km)')

    # Sixth panel: the uplift record the periods are cut from, shaded in the rose colours.
    axu = axes[5]
    u = u_all.loc[periods[0][1]:]
    for k, (label, t0, t1) in enumerate(periods):
        t_end = t1 if t1 is not None else u.index[-1]
        axu.axvspan(t0, t_end, color=colors[k], alpha=0.35, lw=0, zorder=0)
        axu.text(t0 + (t_end - t0) / 2, 1.02, str(k + 1), transform=axu.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='0.25')
    axu.plot(u.index, u.values, color='black', lw=1.4, zorder=2)
    axu.set_ylabel('De-tided uplift (m, 30-day rolling mean)')
    axu.set_title('Central Caldera BOTPT uplift: 5 equal-inflation periods', fontsize=11,
                  fontweight='bold', pad=22)
    axu.grid(alpha=0.3)
    for side in ('top', 'right'):
        axu.spines[side].set_visible(False)

    fig.suptitle('Fast-direction roses by equal-inflation period (Central Caldera BOTPT uplift, '
                 'Grade 3)', fontsize=16, fontweight='bold', y=1.0)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT_BASE}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT_BASE}.pdf / .png')


if __name__ == '__main__':
    main()
