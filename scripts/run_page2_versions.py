"""
Run the two additional versions needed for page 2 of the rose-plot PDF:
  - ZNE / P-Jurkevics cut (old P-wave incidence angle, for comparison against the S-wave
    incidence methods used on page 1)
  - LQT / PyKonal-FMM, rerun to also capture phi_error/dt_error (page 1's run predates that
    column being added to the results CSV)

Appends/updates rows in lqt_zne_incidence_comparison_axec2_batch1_results.csv.

Run with:
    python3 run_page2_versions.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import run_lqt_zne_incidence_comparison_axec2_batch1 as mod

RESULTS_CSV = os.path.join(mod.HERE, 'lqt_zne_incidence_comparison_axec2_batch1_results.csv')

PAGE2_VERSIONS = [
    {'label': 'ZNE / P-Jurkevics cut', 'coord_system': 'ZNE', 'incidence_field': 'incidence_eigenvalue_jurkevics'},
    {'label': 'LQT / PyKonal-FMM',     'coord_system': 'LQT', 'incidence_field': 'incidence_pykonal_s'},
]


def main():
    print("Loading raw batch 1 metadata and building organized_waveforms...")
    batch1 = pd.read_csv(mod.RAW_METADATA_CSV)
    organized_all = mod.build_organized_waveforms(batch1)
    print(f"  {len(organized_all)} events loaded")

    existing = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()

    new_rows = []
    for version in PAGE2_VERSIONS:
        label = version['label']
        field = version['incidence_field']
        filtered = {eid: ed for eid, ed in organized_all.items()
                    if not np.isnan(ed[field]) and ed[field] <= mod.INCIDENCE_CUT_DEG}
        print(f"\n=== {label}: {len(filtered)}/{len(organized_all)} events pass <= {mod.INCIDENCE_CUT_DEG} deg cut ===")

        t0 = time.time()
        results = mod.perform_splitting_on_organized_waveforms(
            filtered, mod.FIRST_WINDOW_START, mod.LAST_WINDOW_START, mod.FIRST_WINDOW_END, mod.LAST_WINDOW_END,
            mod.N_WIN, mod.S_PICK_UNCERTAINTY, mode='swspy',
            coord_system=version['coord_system'], sws_method='EV_and_XC', incidence_field=field,
            plot_results=False
        )
        print(f"  Done in {time.time()-t0:.0f}s")

        for eid, r in results.items():
            res = r['result']
            new_rows.append({'version': label, 'event_id': eid, 'phi': res.get('phi'), 'dt': res.get('dt'),
                              'phi_error': res.get('phi_error'), 'dt_error': res.get('dt_error'),
                              'quality': res.get('quality'), 'success': res.get('success')})

    new_df = pd.DataFrame(new_rows)
    if not existing.empty:
        existing = existing[~existing['version'].isin(new_df['version'].unique())]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved {len(combined)} total rows to {RESULTS_CSV}")


if __name__ == '__main__':
    main()
