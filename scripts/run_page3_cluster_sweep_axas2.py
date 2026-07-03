"""
Page 3: clustering-parameter sweep. All 16 combinations of DBSCAN eps in
[0.05, 0.1, 0.15, 0.2] x min_samples in [5, 10, 15, 20], all using LQT + PyKonal-FMM
incidence (35 deg cut) - same base setup as page 1 row 4 / page 2 row 3, just varying
the clustering parameters instead. Only Q_w > 0.5 events are shown (16 single-panel rose
plots, one page).

Run with:
    python3 run_page3_cluster_sweep.py
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import run_lqt_zne_incidence_comparison_axas2_batch1 as mod

EPS_VALUES = [0.05, 0.1, 0.15, 0.2]
MIN_SAMPLES_VALUES = [5, 10, 15, 20]
INCIDENCE_FIELD = 'incidence_pykonal_s'

OUT_CSV = os.path.join(mod.HERE, 'page3_cluster_sweep_results_axas2.csv')


def main():
    print("Loading raw batch 1 metadata and building organized_waveforms...")
    batch1 = pd.read_csv(mod.RAW_METADATA_CSV)
    organized_all = mod.build_organized_waveforms(batch1)

    filtered = {eid: ed for eid, ed in organized_all.items()
                if not np.isnan(ed[INCIDENCE_FIELD]) and ed[INCIDENCE_FIELD] <= mod.INCIDENCE_CUT_DEG}
    print(f"LQT / PyKonal-FMM: {len(filtered)}/{len(organized_all)} events pass <= {mod.INCIDENCE_CUT_DEG} deg cut")

    rows = []
    combo_num = 0
    for eps in EPS_VALUES:
        for min_samples in MIN_SAMPLES_VALUES:
            combo_num += 1
            print(f"\n=== [{combo_num}/16] eps={eps}, min_samples={min_samples} ===")
            t0 = time.time()
            results = mod.perform_splitting_on_organized_waveforms(
                filtered, mod.FIRST_WINDOW_START, mod.LAST_WINDOW_START, mod.FIRST_WINDOW_END, mod.LAST_WINDOW_END,
                mod.N_WIN, mod.S_PICK_UNCERTAINTY, mode='swspy',
                coord_system='LQT', sws_method='EV_and_XC', incidence_field=INCIDENCE_FIELD,
                cluster_eps=eps, cluster_min_samples=min_samples, plot_results=False
            )
            print(f"  Done in {time.time()-t0:.0f}s")

            for eid, r in results.items():
                res = r['result']
                rows.append({'eps': eps, 'min_samples': min_samples, 'event_id': eid,
                             'phi': res.get('phi'), 'dt': res.get('dt'),
                             'phi_error': res.get('phi_error'), 'dt_error': res.get('dt_error'),
                             'quality': res.get('quality'), 'success': res.get('success')})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_CSV}")


if __name__ == '__main__':
    main()
