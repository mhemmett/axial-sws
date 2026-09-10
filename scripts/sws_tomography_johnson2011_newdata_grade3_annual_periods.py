#!/usr/bin/env python3
"""
sws_tomography_johnson2011_newdata_grade3_annual_periods.py

Same Johnson et al. (2011) 2-D delay-time SWS tomography re-run as
sws_tomography_johnson2011_newdata_grade3_7period.py (newdata 6-station
catalog, grade-3 filter, joint ray-id-offset merge, quad-tree gridding,
covariance-whitened bounded WLS, checkerboard resolution test) -- but with a
different period breakdown, per explicit user request:

    Pre-eruption | Syn-eruption | Post-eruption 2015 | 2016 | 2017 | 2018 |
    2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (partial, to present)

(14 panels total) instead of the 7 eruption-relative/equal-count periods used
by the other script. Pre-eruption/Syn-eruption boundaries are the SAME
ERUPTION_START/ERUPTION_END constants used throughout this project; "Post-
eruption 2015" runs from ERUPTION_END to 2016-01-01; each subsequent panel is
a plain calendar year; 2026 is left open-ended (t1=None) since the catalog's
2026 half-year is not yet complete.

Reuses sws_tomography_johnson2011_newdata_grade3_7period.py's cache-building,
per-period inversion, and BOTH figure functions (make_strength_figure,
make_phi_quiver_figure -- both already generic in the number of periods, and
both already apply the DISPLAY_COUNT_MIN=20 + SIGNIFICANCE_LEVEL=-5.5 display
gates) via import, not copy-paste. Also reuses
sws_tomography_johnson2011_newdata_grade3_7period_checkerboard.py's
make_checkerboard_figure the same way. No re-tracing (same ray cache).

Produces THREE new figures (does not touch any existing Johnson-et-al. output):
    sws_tomography_johnson2011_newdata_grade3_annual_periods_strength.pdf
    sws_tomography_johnson2011_newdata_grade3_annual_periods_phi_quiver.pdf
    sws_tomography_johnson2011_newdata_grade3_annual_periods_checkerboard.pdf

Run with:
    python3 sws_tomography_johnson2011_newdata_grade3_annual_periods.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import traveltime_anisotropy_7period_6stations_newdata_snr_grades as T
import sws_tomography_johnson2011_newdata_grade3_7period as S
import sws_tomography_johnson2011_newdata_grade3_7period_checkerboard as CB

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF_STRENGTH = os.path.join(HERE, 'sws_tomography_johnson2011_newdata_grade3_annual_periods_strength.pdf')
OUT_PDF_PHI = os.path.join(HERE, 'sws_tomography_johnson2011_newdata_grade3_annual_periods_phi_quiver.pdf')
OUT_PDF_CB = os.path.join(HERE, 'sws_tomography_johnson2011_newdata_grade3_annual_periods_checkerboard.pdf')


def build_annual_periods():
    periods = [
        ('Pre-eruption', None, T.ERUPTION_START),
        ('Syn-eruption', T.ERUPTION_START, T.ERUPTION_END),
        ('Post-eruption\n2015', T.ERUPTION_END, pd.Timestamp('2016-01-01', tz='UTC')),
    ]
    for year in range(2016, 2027):
        t0 = pd.Timestamp(f'{year}-01-01', tz='UTC')
        t1 = pd.Timestamp(f'{year + 1}-01-01', tz='UTC') if year < 2026 else None
        label = f'{year}' if year < 2026 else f'{year}\n(partial)'
        periods.append((label, t0, t1))
    return periods


def main():
    T._setup_environment(need_tracer=False)

    print('Loading JOINT six-station grade-3 ray cache (reusing existing cache, no re-tracing)...')
    joint_summ, joint_cells, station_index, sta_xy = S.build_joint_cache()
    print(f'Joint cache: {len(joint_summ):,} rays total across {len(S.STATIONS)} stations')

    periods = build_annual_periods()
    period_labels = [lbl for lbl, _t0, _t1 in periods]
    print(f'\n{len(periods)} annual periods: {[l.replace(chr(10), " ") for l in period_labels]}')

    print('\n=== Running Johnson et al. (2011) chain independently per period ===')
    results = []
    for lbl, t0, t1 in periods:
        print(f'\n-- {lbl.replace(chr(10), " ")} --')
        try:
            res = S.run_chain_grade3_period(joint_summ, joint_cells, t0, t1)
        except ValueError as exc:
            print(f'  [SKIP] {exc}')
            res = None
        results.append(res)

    print('\nBuilding figures...')
    fig1 = S.make_strength_figure(results, period_labels)
    with PdfPages(OUT_PDF_STRENGTH) as pdf:
        pdf.savefig(fig1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved {OUT_PDF_STRENGTH}')

    fig2 = S.make_phi_quiver_figure(results, period_labels)
    with PdfPages(OUT_PDF_PHI) as pdf:
        pdf.savefig(fig2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {OUT_PDF_PHI}')

    fig3 = CB.make_checkerboard_figure(results, period_labels)
    with PdfPages(OUT_PDF_CB) as pdf:
        pdf.savefig(fig3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved {OUT_PDF_CB}')


if __name__ == '__main__':
    main()
