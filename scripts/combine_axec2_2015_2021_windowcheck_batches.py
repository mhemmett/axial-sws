"""
combine_axec2_2015_2021_windowcheck_batches.py

Concatenates every per-batch splitting-results CSV produced by
run_axec2_2015_2021_windowcheck_full_batches.py (AXEC2 2015-2021, mfast per-event bandpass,
max_t_shift_s=0.2s, window-check fix) into one combined CSV.

Reads all splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_batch_*.csv
files in production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results/,
sorted by batch number (not lexical filename order, so batch_2 sorts before batch_10), and
writes the combined result to
production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results/
splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_combined.csv
(a 'batch' column is added recording which batch each row came from). Prints a summary (total
rows, success count, per-batch row-count sanity check against the expected 1-497 range) so a
truncated/still-running batch set is caught rather than silently combined as if complete.

Run with:
    python3 combine_axec2_2015_2021_windowcheck_batches.py
"""

import glob
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'production_axec2_2015_2021_mfast_filters_maxdt02_windowcheck_full_lqt_pykonal_results')
FILENAME_RE = re.compile(r'_batch_(\d+)\.csv$')
OUT_CSV = os.path.join(
    RESULTS_DIR, 'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_combined.csv')

EXPECTED_BATCHES = set(range(1, 498))  # 1-497 inclusive


def main():
    pattern = os.path.join(RESULTS_DIR, 'splitting_results_mldd_2015_2021_axec2_mfast_filters_maxdt02_windowcheck_full_batch_*.csv')
    paths = glob.glob(pattern)
    if not paths:
        raise SystemExit(f"No batch CSVs found matching {pattern}")

    batch_paths = []
    for p in paths:
        m = FILENAME_RE.search(os.path.basename(p))
        if not m:
            continue
        batch_paths.append((int(m.group(1)), p))
    batch_paths.sort(key=lambda bp: bp[0])

    found_batches = {b for b, _ in batch_paths}
    missing = sorted(EXPECTED_BATCHES - found_batches)
    if missing:
        print(f"WARNING: {len(missing)} of {len(EXPECTED_BATCHES)} expected batches (1-497) are "
              f"missing/not yet written: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        print("Combining only the batches present -- re-run this script once the remaining "
              "batches finish if a complete combined CSV is needed.\n")
    else:
        print("All 497 expected batches present.\n")

    dfs = []
    total_rows = 0
    for batch_num, path in batch_paths:
        df = pd.read_csv(path)
        df['batch'] = batch_num
        dfs.append(df)
        total_rows += len(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)

    n_success = int(combined['success'].sum()) if 'success' in combined.columns else None
    print(f"Combined {len(batch_paths)} batch CSVs -> {len(combined)} rows total.")
    if n_success is not None:
        print(f"  success=True: {n_success} ({100*n_success/max(len(combined),1):.1f}%)")
    print(f"Wrote {OUT_CSV}")


if __name__ == '__main__':
    main()
