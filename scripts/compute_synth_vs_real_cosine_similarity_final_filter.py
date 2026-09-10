#!/usr/bin/env python3
"""
compute_synth_vs_real_cosine_similarity_final_filter.py

Re-does compute_synth_vs_real_cosine_similarity.py's cosine-similarity comparison (synthetic
Hudson-crack fast-direction rose histograms vs. real observations, per station/period) under
the user's now-settled FINAL filter (2026-07-27 daily note):
    SNR >= 2.0, phi_err <= 20 deg, dt_err <= 0.05 s, dt <= T_dom/2, Q_w >= 0.75
(newdata_final_filter.py), replacing the old tier4/tier5 (Q_w>=0.5-based) comparison -- single
filter, not two tiers.

Same method as the original: symmetric doubled-angle rose histogram (phi mod 180, NBINS=36),
L2-normalized, cosine similarity between real and synthetic unit vectors. Same model scope
(single-sill and two-sills/shallow-sill2 only, per the original script's explicit user
request -- the "_v2"/"sill2deep" variants are still excluded from this computation).

Produces:
    synth_vs_real_cosine_similarity_final_filter.csv
        (station, model, period, N_real, N_synth, cosine_similarity)

Run with:
    python3 compute_synth_vs_real_cosine_similarity_final_filter.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from newdata_final_filter import (
    STATION_ORDER, load_station_final_filtered, ERUPTION_START, ERUPTION_END, FILTER_LABEL,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'synth_vs_real_cosine_similarity_final_filter.csv')

NBINS = 36

MODEL_FILES = [
    ('synthetic_hudson_pre_syn_phidt.csv', 'Single-sill model'),
    ('synthetic_hudson_pre_syn_phidt_two_sills.csv', 'Two-sills model (shallow sill2)'),
]

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
    return float(np.dot(v1, v2))


def main():
    print(f'Loading real data under the final filter ({FILTER_LABEL})...')
    real_filtered = {sta: load_station_final_filtered(sta) for sta in STATION_ORDER}
    for sta in STATION_ORDER:
        print(f'  {sta}: {len(real_filtered[sta]):,} events')

    rows = []
    for synth_csv, model_label in MODEL_FILES:
        synth_path = os.path.join(HERE, synth_csv)
        if not os.path.exists(synth_path):
            print(f'  Skipping {model_label}: {synth_csv} not found')
            continue
        synth_df = pd.read_csv(synth_path)
        print(f'\n=== {model_label} ({synth_csv}) ===')

        for sta in STATION_ORDER:
            real_df = real_filtered[sta]
            for period_label, t0, t1 in PERIODS:
                real_sub = _subset(real_df, t0, t1)
                synth_sub = synth_df[(synth_df['station'] == sta) &
                                     (synth_df['period'] == period_label)]

                real_vec = _rose_vector(real_sub['phi_az'].values)
                synth_phi_az = synth_sub['phi_synth'].values % 180.0
                synth_vec = _rose_vector(synth_phi_az)

                cos_sim = cosine_similarity(real_vec, synth_vec)
                rows.append(dict(
                    model=model_label, station=sta, period=period_label,
                    N_real=len(real_sub), N_synth=len(synth_sub), cosine_similarity=cos_sim,
                ))
                cos_str = f'{cos_sim:.4f}' if np.isfinite(cos_sim) else 'NaN (empty)'
                print(f'  {sta:6s} {period_label:14s} N_real={len(real_sub):6,} '
                      f'N_synth={len(synth_sub):5,}  cos_sim={cos_str}')

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_CSV} ({len(out_df)} rows)')

    print('\n=== Mean cosine similarity by model x period (across stations) ===')
    summary = out_df.groupby(['model', 'period'])['cosine_similarity'].agg(['mean', 'std', 'count'])
    print(summary.to_string())


if __name__ == '__main__':
    main()
