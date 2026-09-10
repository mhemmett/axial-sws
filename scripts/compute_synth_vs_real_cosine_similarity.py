#!/usr/bin/env python3
"""
compute_synth_vs_real_cosine_similarity.py

One-off script (not part of the pipeline): quantifies how well the synthetic Hudson-crack
splitting model matches the real observations, per station, for pre- and syn-eruption
separately, using COSINE SIMILARITY between the two fast-direction (phi) rose histograms.

Per explicit user request, only the SINGLE-SILL and TWO-SILLS (shallow second sill, i.e. the
original Kidiwela-S2-location sill2 -- NOT the "sill2deep" variant, and NOT the "_v2" dike/sill1
geometry variants) synthetic models are used:
    synthetic_hudson_pre_syn_phidt.csv            (single-sill)
    synthetic_hudson_pre_syn_phidt_two_sills.csv  (two-sills, shallow sill2)

Method: for each (model, station, tier, period) combination, both the real tier-filtered phi_az
values and the synthetic phi_synth values are turned into the SAME symmetric rose histogram
used for plotting (rose_7period_6stations_newdata.py's _draw_rose: phi mod 180, doubled to
phi/phi+180 over 0-360 deg, NBINS=36 equal bins), then L2-normalized into unit vectors. Cosine
similarity = dot(v_real, v_synth) -- 1.0 means identical angular distribution shape, 0 means
orthogonal (no shared structure), and this metric is insensitive to the two samples having
different N (only the density SHAPE is compared, not magnitude).

Computed for BOTH tier 4 (quality>=0.5, dt<T_dom/2, phi_err<20deg, dt_err<0.05s) and tier 5
(+ phi_err<10deg, stricter) real-data QC, matching the two tiers already used in this session's
add_real_vs_synth_comparison_pages.py comparison figures.

Produces:
    synth_vs_real_cosine_similarity.csv (station, model, tier, period, N_real, N_synth, cosine_similarity)

Run with:
    python3 compute_synth_vs_real_cosine_similarity.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from rose_7period_6stations_newdata import (
    STATION_ORDER, load_station_raw, apply_tier, ERUPTION_START, ERUPTION_END,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'synth_vs_real_cosine_similarity.csv')

NBINS = 36   # matches rose_7period_6stations_newdata.py's _draw_rose

MODEL_FILES = [
    ('synthetic_hudson_pre_syn_phidt.csv', 'Single-sill model'),
    ('synthetic_hudson_pre_syn_phidt_two_sills.csv', 'Two-sills model (shallow sill2)'),
]

TIERS = [4, 5]
PERIODS = [
    ('Pre-eruption', None, ERUPTION_START),
    ('Syn-eruption', ERUPTION_START, ERUPTION_END),
]


def _subset(df, t0, t1):
    m = (df['t'] >= t0) if t0 is not None else pd.Series(True, index=df.index)
    if t1 is not None:
        m = m & (df['t'] < t1)
    return df[m]


def _rose_vector(phi_vals):
    """Same symmetric doubled-angle histogram as _draw_rose, L2-normalized to a unit vector.
    Returns None if there are no events (cosine similarity undefined)."""
    if len(phi_vals) == 0:
        return None
    doubled = []
    for phi in phi_vals:
        p = float(phi) % 360.0
        doubled.extend([p, (p + 180.0) % 360.0])
    doubled = np.array(doubled)
    bins = np.linspace(0, 360, NBINS + 1)
    counts, _ = np.histogram(doubled, bins=bins)
    counts = counts.astype(float)
    norm = np.linalg.norm(counts)
    if norm == 0:
        return None
    return counts / norm


def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return np.nan
    return float(np.dot(v1, v2))   # both already unit-normalized


def main():
    print('Loading real tier-4 and tier-5 events for all 6 stations...')
    real_raw = {sta: load_station_raw(sta) for sta in STATION_ORDER}
    real_tiered = {tier: {sta: apply_tier(real_raw[sta], tier) for sta in STATION_ORDER}
                   for tier in TIERS}

    rows = []
    for synth_csv, model_label in MODEL_FILES:
        synth_path = os.path.join(HERE, synth_csv)
        if not os.path.exists(synth_path):
            print(f'  Skipping {model_label}: {synth_csv} not found')
            continue
        synth_df = pd.read_csv(synth_path)
        print(f'\n=== {model_label} ({synth_csv}) ===')

        for sta in STATION_ORDER:
            for tier in TIERS:
                real_df = real_tiered[tier][sta]
                for period_label, t0, t1 in PERIODS:
                    real_sub = _subset(real_df, t0, t1)
                    synth_sub = synth_df[(synth_df['station'] == sta) &
                                         (synth_df['period'] == period_label)]

                    real_vec = _rose_vector(real_sub['phi_az'].values)
                    synth_phi_az = synth_sub['phi_synth'].values % 180.0
                    synth_vec = _rose_vector(synth_phi_az)

                    cos_sim = cosine_similarity(real_vec, synth_vec)
                    rows.append(dict(
                        model=model_label, station=sta, tier=tier, period=period_label,
                        N_real=len(real_sub), N_synth=len(synth_sub), cosine_similarity=cos_sim,
                    ))
                    print(f'  {sta:6s} tier{tier} {period_label:14s} N_real={len(real_sub):6,} '
                          f'N_synth={len(synth_sub):5,}  cos_sim={cos_sim:.4f}' if np.isfinite(cos_sim)
                          else f'  {sta:6s} tier{tier} {period_label:14s} N_real={len(real_sub):6,} '
                               f'N_synth={len(synth_sub):5,}  cos_sim=NaN (empty)')

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_CSV} ({len(out_df)} rows)')

    print('\n=== Mean cosine similarity by model x tier x period (across stations) ===')
    summary = out_df.groupby(['model', 'tier', 'period'])['cosine_similarity'].agg(['mean', 'std', 'count'])
    print(summary.to_string())


if __name__ == '__main__':
    main()
