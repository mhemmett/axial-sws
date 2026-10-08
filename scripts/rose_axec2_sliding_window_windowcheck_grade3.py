#!/usr/bin/env python3
"""
rose_axec2_sliding_window_windowcheck_grade3.py

Remake of results/rose_axec2_sliding_window.gif (built by plot_rose_gif_sliding_window() in
nonlinloc_apr_14_jun_01_plots.ipynb, cell 183, from the old unfiltered MLdd results) using the
current production data chain: windowcheck SWSPy results loaded via
rose_7period_regions_windowcheck_grade3.load_station_raw('AXEC2') and cut with its Grade 3 filter

    SNR >= 2.0, Q_w >= 0.75, dt_err <= 0.05 s, dt <= T_dom/2, phi_err <= 20 deg

Frame layout, colours (purple before / red during / light-blue -> purple gradient after the
eruption), 36-bin axial rose, 1-month window and 7-day step are unchanged from the notebook
function. Frames are rendered at 12x12 in (vs. 36x36 in originally) with fonts scaled to match,
so the GIF is a manageable size; the look is otherwise identical.

Produces (new file, does not overwrite the original GIF):
    results/rose_axec2_sliding_window_windowcheck_grade3.gif

Run with:
    python3 rose_axec2_sliding_window_windowcheck_grade3.py
"""

import io
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from rose_7period_regions_windowcheck_grade3 import GRADE, load_station_raw, apply_grade

# The grade-3 module switches the global font to Arial on import; the original GIF used the
# matplotlib default, so restore it.
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(HERE, '..', 'results', 'rose_axec2_sliding_window_windowcheck_grade3.gif')

STATION = 'AXEC2'
TITLE_PREFIX = f'{STATION} Fast Direction (Grade 3)'
NBINS = 36
FIGSIZE = (12, 12)
LINEWIDTH = 1.5
EDGECOLOR = 'black'
WINDOW_MONTHS = 1
STEP_DAYS = 7
START_DATE = '2015-01-01'
END_DATE = '2026-06-01'
FPS = 6
ERUPTION_START = pd.Timestamp('2015-04-24 06:00:00')
ERUPTION_END = pd.Timestamp('2015-05-19 00:00:00')

RGB_BLUE = np.array(mcolors.to_rgb('#ADD8E6'))
RGB_PURPLE = np.array(mcolors.to_rgb('#800080'))
COL_RED = '#CC0000'


def fill_rose(ax, phi_values, color):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_facecolor('none')
    ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(LINEWIDTH * 1.2)
    ax.grid(False)

    if len(phi_values) == 0:
        ax.set_ylim(0, 1)
        return

    p = np.asarray(phi_values) % 360
    doubled = np.deg2rad(np.concatenate([p, (p + 180) % 360]))
    bins = np.linspace(0, 2 * np.pi, NBINS + 1)
    counts, edges = np.histogram(doubled, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, counts, width=2 * np.pi / NBINS, bottom=0,
           color=color, edgecolor=EDGECOLOR, linewidth=LINEWIDTH, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.2 if counts.max() > 0 else 1)


def main():
    df = apply_grade(load_station_raw(STATION), GRADE).dropna(subset=['phi'])
    df['_dt'] = df['t'].dt.tz_convert('UTC').dt.tz_localize(None)
    print(f'{STATION} grade-3 events: {len(df)}')

    windows = list(pd.date_range(START_DATE, END_DATE, freq=f'{STEP_DAYS}D'))
    post_windows = [w for w in windows if w >= ERUPTION_END]
    n_post = max(len(post_windows), 1)

    def frame_color(w_start):
        if w_start < ERUPTION_START:
            return '#800080'
        if w_start < ERUPTION_END:
            return COL_RED
        t = post_windows.index(w_start) / max(n_post - 1, 1)
        return mcolors.to_hex((1 - t) * RGB_BLUE + t * RGB_PURPLE)

    frames = []
    for k, w_start in enumerate(windows):
        w_end = w_start + pd.DateOffset(months=WINDOW_MONTHS)
        df_w = df[(df['_dt'] >= w_start) & (df['_dt'] < w_end)]
        if k % 50 == 0:
            print(f'  frame {k + 1}/{len(windows)}  [{w_start.date()} - {w_end.date()}]  N={len(df_w)}')

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, subplot_kw={'projection': 'polar'},
                               facecolor='white')
        fill_rose(ax, df_w['phi'].values, frame_color(w_start))
        ax.set_title(
            f'{TITLE_PREFIX}\n'
            f"{w_start.strftime('%Y-%m-%d')} – {w_end.strftime('%Y-%m-%d')}\n"
            f'N = {len(df_w)}',
            fontsize=max(10, int(36 * 1.4 * FIGSIZE[1] / 36)),
            fontweight='bold', pad=20,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB').quantize(colors=64))

    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=False)
    print(f'Saved: {os.path.abspath(OUT_GIF)}  ({len(frames)} frames @ {FPS} fps)')


if __name__ == '__main__':
    main()
