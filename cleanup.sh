#!/usr/bin/env bash
# Repo cleanup: untrack legacy/orphan files, track active workflow notebooks,
# commit, and push. Generated 2026-04-27 to accompany the new .gitignore.
#
# Run from anywhere:   bash ~/Seismology/axial-splitting-ml/cleanup.sh
#
# Idempotent: re-running is safe — git rm --cached --ignore-unmatch swallows
# already-untracked files, and `git add` is a no-op for files that match
# .gitignore or are already up to date.

set -euo pipefail

REPO="$HOME/Seismology/axial-splitting-ml"
cd "$REPO"

echo "==> Removing any stale .git/index.lock"
rm -f .git/index.lock

echo "==> Untracking Baillard framework files (POSTER_*, ARTICLE_*, test_*, untitled*, vertices, general_stub)"
# shellcheck disable=SC2046
git rm --cached --ignore-unmatch -q -- \
  $(git ls-files 'scripts/POSTER_*.py' 'scripts/ARTICLE_*.py' 'scripts/test_*.py' 'scripts/untitled*.py' 2>/dev/null) \
  scripts/vertices.py \
  scripts/general_stub.py 2>/dev/null || true

echo "==> Untracking orphan utilities (no imports anywhere in active code)"
git rm --cached --ignore-unmatch -q -- \
  scripts/build_cmap.py \
  scripts/check_azimuth.py \
  scripts/check_pickles.py \
  scripts/clean_pickles.py \
  scripts/concatenate_pickles.py \
  scripts/create_small_catalog.py \
  scripts/earthquake_location.py \
  scripts/get_trace.py \
  scripts/movehisto.py \
  scripts/normalize_lag.py \
  scripts/obs_array.py \
  scripts/plot_polar_histo.py \
  scripts/plot_polar_on_cartesian.py \
  scripts/run_frequency.py \
  scripts/run_sws.py \
  scripts/run_sws_pool.py \
  scripts/slider_demo.py \
  scripts/split_contour.py \
  scripts/split_contour_test.py \
  scripts/split_nlloc.py \
  scripts/sws_frequency.py \
  scripts/sws_mesh_compute.py \
  scripts/sws_mesh_plot.py \
  scripts/shearwavesplit_plot.py \
  scripts/seismic_geometry.py 2>/dev/null || true

echo "==> Untracking superseded workflow notebooks"
git rm --cached --ignore-unmatch -q -- \
  scripts/master_shear_wave_splitting_workflow.ipynb \
  scripts/master_shear_wave_splitting_workflow_christian_data.ipynb \
  scripts/shear_wave_splitting_workflow.ipynb \
  scripts/dataframe-builder.ipynb \
  scripts/date_time.ipynb \
  scripts/filter_catalog.ipynb \
  scripts/get_waveforms_mini_filtered_catalogue.ipynb \
  scripts/pickle.ipynb \
  scripts/plot_movehisto2d_time.ipynb \
  scripts/read_events.ipynb \
  scripts/seismic_geometry_analysis.ipynb \
  scripts/splitting_sample.ipynb \
  scripts/taup-implementation.ipynb \
  scripts/waveform_plots.ipynb \
  scripts/waveforms.ipynb 2>/dev/null || true

echo "==> Untracking the old 'notebooks/' folder"
git rm --cached --ignore-unmatch -q -- \
  notebooks/.DS_Store \
  notebooks/shear_wave_splitting_workflow.ipynb \
  notebooks/shear_wave_splitting_workflow_christian.ipynb \
  notebooks/whale-investigation-christian.ipynb 2>/dev/null || true

echo "==> Recording deletion of stray ' copy.ipynb' (already gone from disk)"
git rm --cached --ignore-unmatch -q -- "scripts/axial_splitting_nonlinloc_swspy-mfast_AXEC2 copy.ipynb" 2>/dev/null || true

echo "==> Adding .gitignore"
git add -f .gitignore

echo "==> Adding active MLDD per-station processing notebooks"
for nb in \
  scripts/axial_splitting_mldd_AXAS1.ipynb \
  scripts/axial_splitting_mldd_AXAS2.ipynb \
  scripts/axial_splitting_mldd_AXCC1.ipynb \
  scripts/axial_splitting_mldd_AXEC1.ipynb \
  scripts/axial_splitting_mldd_AXEC2.ipynb \
  scripts/axial_splitting_mldd_AXEC2_batched.ipynb \
  scripts/axial_splitting_mldd_AXEC2_full_catalog.ipynb \
  scripts/axial_splitting_mldd_AXEC3.ipynb \
  scripts/axial_splitting_mldd_AXID1.ipynb; do
  [ -f "$nb" ] && git add -- "$nb"
done

echo "==> Adding active NonLinLoc-based processing notebooks"
for nb in \
  scripts/axial_splitting_nonlinloc_swspy_AXAS1.ipynb \
  scripts/axial_splitting_nonlinloc_swspy_AXAS1_batched.ipynb \
  scripts/axial_splitting_nonlinloc_swspy_AXAS2.ipynb \
  scripts/axial_splitting_nonlinloc_swspy_AXAS2_batched.ipynb \
  scripts/axial_splitting_nonlinloc_swspy_AXEC1.ipynb \
  scripts/axial_splitting_nonlinloc_swspy_AXEC1_batched.ipynb \
  scripts/axial_splitting_nonlinloc_swspy_AXEC2_batched.ipynb \
  scripts/axial_splitting_nonlinloc_swspy_AXEC3.ipynb; do
  [ -f "$nb" ] && git add -- "$nb"
done

echo "==> Adding plotting + catalog-prep notebooks"
for nb in \
  scripts/nonlinloc_apr_14_jun_01_plots.ipynb \
  scripts/mldd_apr_20_28_plots.ipynb \
  scripts/parse_mldd_catalog.ipynb; do
  [ -f "$nb" ] && git add -- "$nb"
done

echo "==> Staging modifications already in working tree (splitting_functions, sws_methods, tmid_analysis, etc.)"
git add -u -- scripts/ 2>/dev/null || true

echo "==> Final status preview"
git status --short

echo
echo "==> Committing"
git commit -m "Repo cleanup: untrack legacy/orphan files, track active workflow notebooks, add .gitignore" || {
  echo "(nothing to commit — likely already clean)"
}

push_attempt() {
  echo "==> Pushing to origin/main"
  git push origin main
}

if push_attempt; then
  echo "==> Done."
  exit 0
fi

echo
echo "==> Push failed. Clearing notebook outputs to shrink size and retrying..."
python3 - <<'PY'
import json, glob, os
patterns = [
  'scripts/axial_splitting_mldd_*.ipynb',
  'scripts/axial_splitting_nonlinloc_*.ipynb',
  'scripts/nonlinloc_*plots*.ipynb',
  'scripts/mldd_*plots*.ipynb',
  'scripts/parse_mldd_catalog.ipynb',
  'scripts/tmid_analysis.ipynb',
]
seen = set()
for pat in patterns:
  for nb in glob.glob(pat):
    if nb in seen: continue
    seen.add(nb)
    try:
      with open(nb) as f: data = json.load(f)
      cleared = 0
      for c in data.get('cells', []):
        if c.get('cell_type') == 'code':
          if c.get('outputs') or c.get('execution_count') is not None:
            cleared += 1
          c['outputs'] = []
          c['execution_count'] = None
      with open(nb, 'w') as f: json.dump(data, f, indent=1)
      print(f"cleared {cleared:>3} cells in {nb}")
    except Exception as e:
      print(f"skip {nb}: {e}")
PY

git add -u -- scripts/
git commit --amend --no-edit
push_attempt
echo "==> Done (with output-clearing retry)."
