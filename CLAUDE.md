# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Shear-wave splitting analysis of local seismicity at Axial Seamount (Juan de Fuca Ridge), tracking
fast-polarization direction (φ) and delay time (δt) across the April 2015 eruption and subsequent
inflation, as a proxy for crack-induced stress anisotropy. Full scientific background, catalogs, QC
thresholds, and method comparison live in `README.md` — read it for anything not covered here.

## Setup

```bash
bash env.sh                        # creates/reuses conda env "seismo", pip-installs requirements.txt, registers Jupyter kernel
conda activate seismo
jupyter notebook                   # select kernel "Python (seismo)"
```

`env.sh` is idempotent. No test suite / lint config for this repo's own code (`swspy/`'s `tests/` is
upstream, not the active surface). The vendored, modified SWSPy is not pip-installed — every
notebook/script loads it with:
```python
import sys
sys.path.insert(0, 'swspy')   # relative to repo root
import swspy
```

## Repository map

- `scripts/` — ~824 files; only a small subset is active (below), the rest is dormant/reference.
- `swspy/` — vendored SWSPy fork (DBSCAN clustering fix). Semi-external: changes here should track a
  specific upstream fix, not general refactors.
- `data/`, `pickles/`, `results/` — gitignored, local-only. Don't assume populated; ask rather than
  guess a path or fabricate data if something's missing.
- `docs/` — dated design notes on past refactors of the dynamic-parameter pipeline; useful for *why*
  current signatures look the way they do, but may describe an intermediate state — verify against code.
- `cleanup.sh` — one-off, already-run repo-cleanup script. Not part of routine workflow.
- `pylith_axial/` — separate PyLith stress-modeling component, see below.

## Splitting method (active code)

Production method is a **variant of SWSPy** (Silver & Chan 1991 + Teanby et al. 2004 clustering) that
draws on two other sources, not upstream SWSPy alone:
- **MFAST 2.2** (Wessel, Savage & Teanby 2017) for building the analysis window from each event's
  dominant period `T_dom`.
- **Christian Baillard's unpublished Python code** for pre-processing/filtering helpers (rotation,
  bandpass filtering, signal/noise windowing) ahead of the measurement.

**`scripts/splitting_functions.py`** (~9,300 lines) is the core workflow file, in rough pipeline order:
- QC helpers: `snr`, `calculate_incidence_angle*` (Jurkevics variants), `calculate_rectilinearity*`,
  each with a `_for_organized_waveforms` batch wrapper.
- Dynamic-parameter chain: `estimate_dominant_period` → `calculate_dynamic_parameters` →
  `create_splitting_analysis` → `perform_splitting_analysis` (MFAST 2.2 windowing + Baillard
  pre-processing, per above). See `docs/splitting_integration_update.md` for the call graph.
- Baillard's single-window method ported inline: functions suffixed `_baillard` (grid search over
  lags/angles, contour/minima extraction, RMS ranking) — kept as a standalone cross-validation method,
  not merged with the SWSPy path.

**Baillard-derived support modules**, imported transitively — load-bearing, not dead code:
`sws_methods.py`, `shearwavesplit.py`, `plotwaveform.py`, `GMT.py`, `projection.py`, `util.py`,
`sws_vertices.py`.

**`teanby_clustering.py`** — Calinski-Harabasz / Duda-Hart cluster-count selection helpers; a separate
path from the production SWSPy clustering (DBSCAN + Teanby (2004) representative-variance selection in
`swspy/swspy/splitting/split.py::_sws_win_clustering`) — not used by the production pipeline.

**Everything else in `scripts/`** (`ARTICLE_*.py`, `POSTER_*.py`, `test_*.py`, `untitled*.py`, orphan
`check_*`/`clean_*`/`concatenate_*` utilities, superseded notebooks) is reference-only, not active — see
README's "Files kept locally but not tracked". Don't extend these; confirm with the user before using
them for a task.

**No CLI entry points** — the workflow runs through per-station Jupyter notebooks in `scripts/`, e.g.
`axial_splitting_mldd_AXEC2_batched.ipynb` (production MLdd), `axial_splitting_nonlinloc_swspy_AXAS2.ipynb`
(NonLinLoc cross-validation), `nonlinloc_apr_14_jun_01_plots.ipynb` (plotting). To verify a change to
`splitting_functions.py`/`swspy/`, run/inspect the relevant notebook cells rather than writing a new
standalone test script. `_batched` notebooks write
`results/splitting_results_mldd_2015_2021_<station>_batch_<N>.csv` in ~100-event chunks — check
`results/` for the highest existing batch number before resuming a run.

Two splitting methods coexist by design (SWSPy variant = production, Baillard single-window =
validation) — don't collapse them; their agreement is itself a validation result. Notebooks/functions
also distinguish the **MLdd catalog** (production, Wang 2024) from the **NonLinLoc catalog**
(Wilcock/Baillard, cross-validation) — these are not interchangeable inputs, check which one a given
notebook targets before changing defaults.

## Stress modeling (`pylith_axial/`)

Independent PyLith 5.0.1 poroelastic model of the caldera (no shared imports/data with `scripts/`/`swspy/`).
- **Geometry**: two outward-dipping ring faults (east/west) + central spherical Mogi source at 3.33 km
  depth; flat topography, ~1.5 km lithostatic stress from seawater. Mesh: `mesh/generate_mesh.py` (Gmsh).
- **Physics**: full poroelasticity, inflation driven by prescribed fluid pressure at the Mogi cavity wall
  (stresses crust → loads ring faults). Config in `cfg/`; per-scenario overrides in `scenarios/`.
- **Driver**: `scripts/run_sweep.py` — two-pass sweep ("pre": no-fault inflation → resolve stress onto
  fault planes; "syn": convert to prescribed slip, rerun with `FaultCohesiveKin`), swept over fluid
  pressure (`output/pre_dP*`, `output/syn_dP*`) and pore pressure (`output/pore_Pf*`).
  `scripts/plot_stress_3d.py` / `visualize_sweep.py` for plotting; outputs are `.h5`/`.xmf` (ParaView).
- **Long-term plan**: cycle stress states across the 2015 eruption, then perturb penny-shaped saturated
  cracks with the resulting stress fields and forward-model splitting from synthetic seismograms — a
  physics-based counterpart to the observational φ/δt results.
- **Environment**: the PyLith solver itself (incl. `gmsh` for meshing) needs a separate local install,
  activated via `pylith_axial/activate_pylith.sh` (hardcoded to
  `~/Downloads/pylith-5.0.1-macOS-12.0-arm64`) — not covered by the `seismo` conda env. The Python-side
  orchestration/visualization scripts (`run_sweep.py`'s resolve step, `plot_stress_3d.py`,
  `visualize_sweep.py` — `h5py`, `pyvista`) run in `seismo` like everything else.
- `run_sweep.py` deliberately refuses to run until certain physical inputs (fluid-pressure sweep
  amplitudes, stress-to-slip relation) are filled in — intentional, not a bug. Ask the user for these
  values; don't guess placeholders.

## Session start: Obsidian check-in

At the start of every new session, before other work, check in against the user's Obsidian notes:
- `Daily To Do` folder: `/Users/mhemmett/Documents/Class-Notes/Daily To Do/`
- `Axial Seamount Literature Review` folder: `/Users/mhemmett/Documents/Class-Notes/Axial Seamount Literature Review/`

(Both live inside the `Class-Notes` vault — read only these two folders, not the rest of that vault.)

1. Read the most recent dated note in `Daily To Do` ("yesterday's note").
2. **First time only**: also read every note in `Daily To Do` and every note in
   `Axial Seamount Literature Review`, to build full context. Record in project memory that this
   initial ingest is complete (with the date and which notes were covered), so it is not repeated.
3. **Every time after that**: check both folders for notes dated after the last recorded check, and
   read only those.
4. From what was read, produce a summary covering: current project loose ends, the next goals stated
   in the previous daily note, and open questions about scientific interpretation.
5. Suggest 3 daily goals based on that summary and ask the user to confirm or adjust them.
6. Once confirmed, build a plan (via `/plan`) to accomplish the confirmed goals, and present it to the
   user.

## General

`data/`, `results/`, `pickles/` are gitignored — never assume a file under these is tracked, and don't
add them to git even implicitly (e.g. `git add -A`).
