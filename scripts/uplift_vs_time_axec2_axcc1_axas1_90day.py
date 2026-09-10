#!/usr/bin/env python3
"""
uplift_vs_time_axec2_axcc1_axas1_90day.py

Companion to uplift_vs_phi_axec2_axcc1_axas1_90day.py: SAME x-axis quantity and SAME 90-day
centered rolling mean (applied locally to the raw daily de-tided uplift series returned by
each station's load_daily_series(), module-default ROLLING_DAYS=30 not overridden), but
plotted against TIME instead of against fast direction. Post-eruption window only, to match
the phi-vs-time companion (phi_vs_time_axec2_axcc1_axas1_90day.py) and the original
uplift-vs-phi comparison.

Produces (NEW file, 3 pages, one per station):
    uplift_vs_time_axec2_axcc1_axas1_90day.pdf

Run with:
    python3 uplift_vs_time_axec2_axcc1_axas1_90day.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from axec2_uplift_phi_cosine_vs_time import ERUPTION_END, UNLEVEL_START, UNLEVEL_END
import bpr_inflation_periods as ecal_infl
import bpr_inflation_periods_ccal as ccal_infl
import bpr_inflation_periods_ashes as ashes_infl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, 'uplift_vs_time_axec2_axcc1_axas1_90day.pdf')

UPLIFT_ROLLING_DAYS = 90   # matches uplift_vs_phi_axec2_axcc1_axas1_90day.py's
                          # UPLIFT_ROLLING_DAYS -- overrides the module-default 30-day
                          # geodetic smoothing locally, without touching the shared modules

STATIONS = [
    dict(name='AXEC2', infl_module=ecal_infl, geodetic_label='Eastern Caldera BOTPT',
        flag_unleveled=True),
    dict(name='AXCC1', infl_module=ccal_infl, geodetic_label='Central Caldera BOTPT',
        flag_unleveled=False),
    dict(name='AXAS1', infl_module=ashes_infl, geodetic_label='ASHES vent field BOTPT',
        flag_unleveled=False),
]


def make_page(station, inflation_roll):
    valid = inflation_roll.dropna()
    valid = valid[valid.index >= ERUPTION_END]

    t = valid.index
    y = valid.values

    fig, ax = plt.subplots(figsize=(9, 5))
    if station['flag_unleveled']:
        is_flagged = (t >= UNLEVEL_START) & (t < UNLEVEL_END)
        ax.scatter(t[~is_flagged], y[~is_flagged], s=18, color='#0072B2',
                  alpha=0.7, label='Post-eruption window')
        ax.scatter(t[is_flagged], y[is_flagged], s=22, color='#2ca02c', alpha=0.9,
                  marker='^', label='Seismometer potentially unleveled\n(2021-07 to 2022-09)')
    else:
        ax.scatter(t, y, s=18, color='#0072B2', alpha=0.7, label='Post-eruption window')

    ax.set_xlabel('Time')
    ax.set_ylabel(f'De-tided uplift (m, {UPLIFT_ROLLING_DAYS}-day rolling mean)')
    ax.set_title(f'{station["name"]}: De-Tided Uplift vs. Time\n'
                 f'({station["geodetic_label"]}, post-eruption, {UPLIFT_ROLLING_DAYS}-day rolling mean)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def main():
    figs = []
    for station in STATIONS:
        name = station['name']
        print(f'Loading {name} de-tided uplift series...')
        _dd, infl_raw, _infl_roll_30, _rt, _rd = station['infl_module'].load_daily_series()
        inflation_roll = infl_raw.rolling(f'{UPLIFT_ROLLING_DAYS}D', center=True,
                                          min_periods=UPLIFT_ROLLING_DAYS // 2).mean()
        n_valid = inflation_roll.dropna().shape[0]
        print(f'  {n_valid} {UPLIFT_ROLLING_DAYS}-day-rolled uplift points')

        figs.append(make_page(station, inflation_roll))
        print()

    with PdfPages(OUT_PDF) as pdf:
        for fig in figs:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
    for fig in figs:
        plt.close(fig)
    print(f'Saved {OUT_PDF} ({len(figs)} pages)')


if __name__ == '__main__':
    main()
