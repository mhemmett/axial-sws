"""
Combine shear-wave splitting results from AXEC2 into one file
"""
import glob
import os

import pandas as pd

import sws_methods as swm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'production_axec2_lqt_pykonal_results')

STATION = 'AXEC2'


def load_combined_results():
    result_files = glob.glob(os.path.join(RESULTS_DIR, 'splitting_results_mldd_2015_2021_axec2_batch_*.csv'))
    dfs = [pd.read_csv(f) for f in result_files]
    dfs = [d for d in dfs if len(d) > 0]
    results = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    results = results[results['success'] == True].copy()
    results = results[results['dt'] > 0]
    results['phi_az'] = results['phi'] % 180.0
    return results

results = load_combined_results()
results.to_csv('splitting_results_AXEC2_2015_2021_all_batches.csv', index=False)