# Axial Seamount Shear-Wave Splitting Analysis

Shear-wave splitting analysis of local seismicity at Axial Seamount on the Juan de Fuca Ridge, using OBS data from the OOI Regional Cabled Array. The goal is to track the temporal evolution of stress and crack-induced anisotropy across the **April 2015 eruption** and the long inflation cycle that has followed it.

## Project Overview

Splitting parameters — fast polarization direction (φ) and delay time (δt) — measure the orientation and density of aligned cracks in the shear-wave window. Tracking how φ and δt change in space and time should constrain how the volcanic stress field reorganizes around eruptive episodes at submarine ridge volcanoes, and feeds into hazard / forecasting work on the Juan de Fuca Ridge.

The long-term arc is to start just before the April 2015 eruption and run through to the present, currently re-inflated state of the caldera. As of this writing the analysis covers 2015–2021 from the OOI cabled array; future work extends it forward to today using 2022–2026 data from the **same caldera OBS network** (not a new deployment), sourced from the real-time (RT) version of the MLdd earthquake catalog.

### Research Objectives
- Track temporal changes in φ and δt across the 2015 eruption and the subsequent inflation cycle.
- Constrain the stress-field evolution within the caldera (and ultimately the rift zone).
- Validate a modified SWSPy pipeline against an independent Baillard implementation.
- Contribute to volcanic hazard assessment and eruption forecasting at submarine ridge volcanoes.

## Data

### Network and time period
- **Source**: OOI Regional Cabled Array (RCA), University of Washington
- **Period**: 2015–2021 (current); long-term goal is pre-2015 → present
- **Instrumentation**: short-period and broadband ocean-bottom seismometers

### Stations
Five caldera-floor stations carry the analysis through the eruption window:
- **AXAS1, AXAS2** (western caldera wall - ASHES vent field)
- **AXEC1, AXEC2, AXEC3** (eastern caldera wall)

**AXCC1** (central caldera) tipped over in March 2015 due to caldera volumetric inflation, but is reset after the eruption onset. This portion of data is excluded, but otherwise the station has good coupling and is included in the analysis.

### Earthquake catalogs

The primary catalog is the **MLdd (machine-learning + double-difference) catalog of Kaiwen Wang** at Axial Seamount, which provides high-quality picks and relocations across the eruption and the inflation cycle. **S-pick uncertainties from Wang (2024)** drive the per-event windowing in the splitting analysis.

Earlier method development used a **NonLinLoc 3D relocated catalog with kurtosis-based picks from Wilcock & Baillard**, primarily as a vehicle to develop the workflow and to cross-validate against Christian Baillard's original Python splitting code. Results are consistent between the two pipelines, so the NonLinLoc catalog now serves mostly as a regression / cross-check rather than the production input.

| Catalog | Role | File |
|---|---|---|
| Wang MLdd, 2015–2021 | **primary** — production runs | `data/Axial.MLDD.v202112.2`, `data/mldd_catalog_2015_2021.csv` |
| Wilcock/Baillard NonLinLoc | method dev + cross-validation | `data/AXIAL.PHASE.FINAL_3D_V2.nlloc` |
| Earlier DD pha (2022) | reference | `data/Axial.DD.pha.v20221129.1` |

Station metadata in `data/stations_axial.llz` and `data/AXIAL_stations.xml`. Per-station QC'd catalogs are pickled to `data/AX*.clean.cat.pickle`.

## Method

The production analysis is a **variant of SWSPy** (Silver & Chan 1991 eigenvalue minimization with Teanby et al. 2004 cluster analysis) that draws on two other sources rather than upstream SWSPy alone:
- **Christian Baillard's unpublished Python implementation** supplies the helper functions used for data pre-processing and filtering (component rotation, bandpass filtering, noise/signal windowing, P-coda handling) ahead of the splitting measurement itself.
- **MFAST 2.2** (Wessel, Savage & Teanby 2017) supplies the approach for building the shear-wave splitting analysis window from each event's dominant period `T_dom`, in place of upstream SWSPy's fixed windowing.

The repository also contains Christian Baillard's original single-window Python implementation as a standalone method, used both as the historical foundation of this work and as an ongoing cross-validation reference.

### Modified SWSPy (production)

Vendored in `swspy/` with corrections relative to upstream:
- **DBSCAN clustering** (`eps=0.15`, `min_samples=15`) in a circular-safe `(δt cos 2φ, δt sin 2φ)` coordinate space, matching MFAST.
- **Teanby (2004) representative variance** (Eq. 13–14) for cluster selection; best observation within the chosen cluster is the one with minimum observation variance.
- MFAST-compliant `Ncmin = 1`, `Mmax = 15`.
- **Dynamic windowing** based on the dominant period `T_dom` of each event, following the **MFAST 2.2** window-construction approach, rather than upstream SWSPy's fixed window lengths.
- **S-pick uncertainty** from Wang (2024) drives the analysis-window placement around the S arrival.
- **Pre-processing and filtering** (rotation, bandpass filtering, signal/noise windowing) reuse **Baillard's unpublished helper functions**, called from `splitting_functions.py` ahead of the SWSPy measurement step — see the Baillard-derived modules listed below.

Core implementation: `swspy/swspy/splitting/split.py::_sws_win_clustering` and the `splitting_functions.py` workflow in `scripts/`.

### Christian Baillard's single-window method (foundation + validation)

Much of the early method development, and a large fraction of the supporting scripts in this repo, draws on Christian Baillard's unpublished Python implementation. His core approach is:
- **Single adaptive window** sized to `T_dom × [0.5, 2.0]`.
- 100 lags × 100 angles exhaustive grid search per event.
- Multiple-minima detection on the eigenvalue surface via morphological filtering.
- RMS-based ranking of solutions; P-coda contamination handling on the noise window.

His scripts (notably `shearwavesplit.py`, `sws_methods.py`, `plotwaveform.py`, `GMT.py`, `projection.py`, `util.py`, `sws_vertices.py`) are still active dependencies of the workflow and are imported transitively by `splitting_functions.py`. Cross-validation against Baillard's reference results agrees to within ~5% on SNR, ≤1° on geometry, and consistent φ/δt — see `scripts/axial_splitting_nonlinloc_*` for the validation runs.

### Method comparison at a glance

| Feature | Modified SWSPy (production) | Baillard (validation) |
|---|---|---|
| Window strategy | multiple windows, cluster in window space | single adaptive window |
| Window length | dynamic, based on `T_dom` (MFAST-style) | `T_dom × [0.5, 2.0]` |
| S-arrival placement | Wang (2024) S-pick uncertainty | Baillard adaptive |
| Clustering | DBSCAN, circular-safe `(δt, φ)` | parameter-space minima |
| Quality metric | F-statistic 95% region | RMS difference |

### Quality control

Applied uniformly before splitting:
1. **SNR** (horizontal) > 2.0; signal window S → S+2.0 s; noise window adjusted to avoid P-coda.
2. **P-wave rectilinearity** > 0.7 over P ± 0.12 s (Jurkevics 1988).
3. **Incidence angle** < 35° from vertical — see [S-wave incidence angle & LQT rotation](#s-wave-incidence-angle--lqt-rotation) below; the QC cut moved from 30° (legacy P-wave incidence) to 35° when the incidence calculation itself was corrected.
4. **Magnitude** filter (default M > 0).

### S-wave incidence angle & LQT rotation

LQT ray-based rotation needs the **S-wave** incidence angle at the receiver, not the P-wave incidence angle — using the P-wave Jurkevics incidence (the original QC/rotation angle) for LQT was a bug, since P- and S-wave particle motion have different geometric relationships to the ray (P motion is along the ray → `arccos`; S/SV motion is transverse to it → `arcsin`).

Two corrected incidence-angle methods are implemented and cross-validated against each other and against the legacy P-Jurkevics/TauP approaches on AXEC2 data:
- **Eigenvalue S-incidence** (`calculate_incidence_angle_eigenvalue_jurkevics_s` in `splitting_functions.py`) — `arcsin(|v1_Z|)` from eigen-decomposition of the S-wave window, where the window is `[S − 0.02 s, S + T_dom]` and `T_dom` comes from SWSPy's own `get_dominant_period_baillard` (not the legacy, broken `estimate_dominant_period`).
- **PyKonal FMM ray-traced incidence** (`pykonal_raytracer.py::BaillardRayTracer.incidence_angle_at_station`) — eikonal fast-marching ray tracing through the Baillard 3D S-velocity model, angle of the final ray segment from vertical. Chosen over pseudo-bending (`bent_ray`, kept as `baillard_raytraced_incidence.py` for reference) as the more rigorous ray-tracing approach.

TauP-based incidence was tried and dropped (P-Jurkevics/Eigenvalue-S/PyKonal-FMM/TauP were compared directly; see `run_incidence_comparison_axec2_batch1.py`, `run_lqt_zne_incidence_comparison_ax*_batch1.py`). The legacy P-wave Jurkevics incidence function is kept in `splitting_functions.py` for reference only.

A `coord_system` parameter (`'ZNE'` or `'LQT'`) threads through `perform_splitting_on_organized_waveforms` → `perform_splitting_analysis` → `create_splitting_analysis`, controlling whether `0.0` (ZNE, no rotation) or the real incidence angle (LQT) is passed to `swspy.splitting.create_splitting_object`. Production runs use `coord_system='LQT'` with the PyKonal-FMM incidence at a 35° cut. `unit_test_incidence_angle_s.py` unit-tests the eigenvalue S-incidence function (0–90° range enforcement, geometry sanity checks, T_dom-window scaling).

## Repository Layout

```
axial-sws/
├── README.md                      # this file
├── env.sh                         # one-shot environment setup
├── requirements.txt               # pip dependencies
├── .gitignore                     # ignores data/, results/, pickles/, *.mseed, figures
├── cleanup.sh                     # one-off repo-cleanup helper (2026-04-27)
├── data/                          # catalogs, station metadata, per-station QC pickles  (untracked)
├── pickles/                       # cached intermediate objects                          (untracked)
├── results/                       # CSV outputs, rose plots, batched results             (untracked)
├── docs/
│   ├── splitting_workflow_update.md
│   ├── splitting_integration_update.md
│   └── package_updates.md
├── swspy/                         # vendored, modified SWSPy (Teanby clustering fix)
├── scripts/                       # active code — see below
├── Axial_Deformation/             # DMODELS (Okada dike + Yang spheroid) input/output grids (untracked)
└── pylith_axial/                  # PyLith poroelastic caldera stress model — see below
```

### Active code in `scripts/`

Core analysis library:
- `splitting_functions.py` — main workflow: QC, dynamic-parameter pipeline, both Baillard and SWSPy entry points.
- `teanby_clustering.py` — clustering helpers used by `splitting_functions.py`.
- `get_all_traces.py` — automated waveform retrieval from the cabled array.

Baillard-derived modules (still imported by the active workflow):
- `sws_methods.py`, `shearwavesplit.py`, `plotwaveform.py`, `GMT.py`, `projection.py`, `util.py`, `sws_vertices.py`.

Forward-modeling / ray tracing (crack-model side of the pipeline):
- `hudson_crack_model.py` — Hudson effective-medium theory for randomly-oriented, fluid-saturated penny cracks; stress tensor in, effective stiffness + predicted φ/δt out.
- `sws_forward_model.py` — MCMC forward model predicting splitting at OBS stations from analytic Mogi/dike stress, using `hudson_crack_model.py`.
- `sws_raytraced_dt.py`, `pykonal_raytracer.py` — ray tracing (pseudo-bending and eikonal) through the Baillard 3D S-velocity model; `pykonal_raytracer.py::BaillardRayTracer` is also the production S-wave incidence-angle ray tracer (see [S-wave incidence angle & LQT rotation](#s-wave-incidence-angle--lqt-rotation)).
- `baillard_raytraced_incidence.py` — standalone `bent_ray` pseudo-bending incidence-angle port, kept as a reference alternative to the PyKonal-FMM method (not used in production).
- `unit_test_incidence_angle_s.py` — unit tests for the eigenvalue S-wave incidence-angle function.
- `baillard_velocity.py`, `baillard_simple_model.py`, `baillard_kidiwela_model.py` — 3D S-velocity grid interpolation and simpler Mogi/Yang-source stress-to-splitting proxy models.

Production run + rose plots (AXEC2 full-catalog, LQT + PyKonal-FMM incidence):
- `run_production_axec2_all_batches.py` — production splitting run over the full 2015–2021 MLdd catalog (497 batches), `coord_system='LQT'`, PyKonal-FMM incidence, 35° cut, `Q_w` retained; writes to `production_axec2_lqt_pykonal_results/` (untracked).
- `build_production_rose_plots_axec2.py`, `build_production_rose_plots_axec2_qw05.py` — 7-panel (eruption-relative) and annual rose plots from the production run, all-data and `Q_w ≥ 0.5`/φ_err<20°/δt_err<0.04s filtered.
- `axec2_temporal_histogram_lqt_pykonal.py` — 2D moving-window density histogram of δt/φ vs. time for AXEC2, from the production dataset.
- `build_raw_ax*_batch1.py`, `build_raw_axec2_all_batches.py`, `fetch_raw_axas2_batch1.py` — raw-waveform QC rebuild (SNR + rectilinearity only, no incidence pre-filter) used to regenerate unbiased inputs for the above, since the pre-existing cached datasets were filtered with the old (buggy) incidence QC.
- `sws_percent_anisotropy.py` — spatial percent-anisotropy maps (`A = V_s,mean · dt · 100 / r`), per-voxel median (no tomographic inversion) along PyKonal-traced rays.

Production plotting / post-processing (consume `results/` CSVs, one script per figure family):
- `rose_plots_temporal.py`, `rose_plots_ultra_strict.py`, `rose_plots_strict_filter.py`, `rose_plots_baz.py`.
- `sws_temporal_*.py`, `sws_tomography_*.py`, `sws_histograms*.py`, `sws_mesh_*.py`, `sws_source_accumulated.py`, `sws_strict_filter_annual.py`, `sws_phi15_filter_plots.py`, `sws_gif_*.py`, `mogi_*.py`, `compute_pgv_*.py`, `plot_pgv_*.py`, `dt_norm_temporal.py`, `cosine_similarity_heatmap.py`, `temporal_histograms.py`, `map_sws_spatial.py`, `seismicity_map.py`, `visualization.py`, `funcs.py`.

Deformation modeling (DMODELS comparison, `Axial_Deformation/`):
- `deformation_util.py`, `deformation_analysis.py` — fixed/ported versions of Christian Baillard's 2019 `ARTICLE_deformation_util.py`/`ARTICLE_deformation.py`: broken imports repointed at this repo's flat modules, hardcoded paths made relative, matplotlib API updated for current versions (`plt.colormaps[...]`, dropped `savefig(quality=...)`/`frameon=`). Reads DMODELS (Okada dike + Yang spheroid) displacement grids from `Axial_Deformation/*.xyzuvw`, computes 2D stress/strain and the modeled principal-compression direction, and compares against Baillard's own hardcoded splitting/deformation observations (`get_station_obs`), for all 8 catalog stations including AXID1.
- `deformation_analysis_hemmett.py`, `deformation_geometry_stress_hemmett.py`, `deformation_change_pre1_syn6_hemmett.py`, `deformation_change_voxel_hemmett.py` — `_hemmett` variants that replace Baillard's hardcoded literals with this repo's own splitting-pipeline results (`get_station_obs_hemmett` in `deformation_util.py`), restricted to the 6 production stations (AXID1 excluded, matching `rose_plots_temporal.py`'s `STATION_ORDER`). `_geometry_stress_hemmett` additionally interpolates observed φ spatially along PyKonal-traced rays (per-voxel circular median, matching `sws_percent_anisotropy.py`'s method) rather than using one value per station; `_change_voxel_hemmett` extends the pre→syn linear-fit/RMS comparison from 6 station points to every ray-covered voxel.

Batch execution helpers:
- `run_ax*_notebook.sh`, `run_axec3_worker*.sh`, `run_axec3_single.sh` — retry-loop wrappers that run the corresponding `_batched.py` script under the `seismo` conda env; `memory_monitor.sh` watches system memory during long runs.

2022–2026 RT-catalog extension (same 6 caldera stations as the main analysis, **not** a new/different OBS deployment - "RT" refers to the real-time version of the MLdd earthquake catalog these events were originally sourced from):
- `axial_splitting_mldd_AX*_RT_batched.py` — per-station batched splitting runs against the RT catalog.
- `build_catalog_2022_2026.py` — builds the 2022–2026 RT catalog into this workflow's input format.
- `run_ax*_RT_notebook.sh` — retry-loop wrappers for the RT batched scripts, mirroring the main batch execution helpers above.
- Not yet run to completion; not part of the current decadal (2015–2021) production results.

Geodetic (BPR uplift) vs. fast-direction comparison:
- `process_bpr_detided_depth*.py` (`_ccal`, `_ashes`, and the undecorated Eastern Caldera/AXEC2
  variant) — de-tide the raw OOI BOTPT (bottom pressure recorder) seafloor-depth record for the
  Central Caldera (AXCC1), ASHES (AXAS1), and Eastern Caldera (AXEC2) sites into
  `data/bpr_detided_seafloor_depth*.csv`.
- `bpr_inflation_periods*.py` — turn a de-tided depth record into a daily-mean, rolling-window
  "uplift" series referenced to the post-eruption minimum (`load_daily_series()`).
- `axec2_uplift_phi_cosine_vs_time.py` — shared rolling fast-direction (φ) module; provides
  `rolling_phi_stats_daily_then_roll` (per-calendar-day circular mean, then a centered rolling
  mean — matches the uplift side's own daily-mean-first approach) used by the fit scripts below.
- `animate_arctan_stress_vectors.py` — defines the current production fit,
  `fit_atan2_vectorsum`/`invert_atan2_vectorsum`: φ as `C1 + C2·atan2(Y,X)` of a vector sum of a
  fixed-azimuth regional tectonic-stress vector and a growing inflation vector: physically
  motivated model derived from and superseding earlier cube-root/erfc/sigmoid fit attempts.
  Also provides the shared grayscale-bathymetry/station-xy helpers (`load_bathy_gray`,
  `load_station_xy`) and drives the animated GIF figures.
- `atan2_uplift_vs_phi_sixstations_ccal_30day.py` — production 6-station version: every station
  (AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3) fit against the **same** Central Caldera BOTPT
  uplift series (not each station's own local BOTPT), with matched 30-day rolling windows on
  both axes (φ and uplift) — the earlier 3-station version mixed 90-day φ against 90-day uplift,
  and before that 180-day φ against 30-day uplift. Per-station turnover (`u0`, the fixed uplift
  value the fitted curve's steepest transition is pinned to) is auto-estimated via a free-
  location logistic pre-fit, rather than hand-picked. Exposes `compute_all_station_fits()` for
  reuse by other scripts. Produces
  `atan2_uplift_vs_phi_sixstations_ccal_30day.pdf` (6 pages).
- `inflation_vector_map_sixstations_ccal.py` — single-page grayscale-bathymetry caldera map
  (reusing `animate_arctan_stress_vectors.py`'s bathymetry/station helpers), all 6 stations as
  yellow triangles, with a red arrow per station in the direction of that station's fitted
  inflation-vector azimuth (`beta_az`) from the fit above. **Caveat**: AXCC1's fit currently has
  an anomalously large `A` parameter, producing a near-discontinuous (rather than smooth
  S-curve) transition — its arrow direction is less trustworthy than the other 5 stations' until
  that fit is revisited.

### Active notebooks in `scripts/`

Per-station MLdd processing (production):
- `axial_splitting_mldd_AXAS1.ipynb`, `…_AXAS2.ipynb`, `…_AXCC1.ipynb`, `…_AXEC1.ipynb`, `…_AXEC2.ipynb` (+ `_batched`, `_full_catalog`), `…_AXEC3.ipynb`, `…_AXID1.ipynb`.

Per-station NonLinLoc processing (cross-validation):
- `axial_splitting_nonlinloc_swspy_AX*.ipynb` (+ `_batched` variants for chunked runs).
- `axial_splitting_nonlinloc_swspy-mfast*` notebooks for the MFAST-style parameter exploration.
- `axial_splitting_nonlinloc_AXAS2_parameter_comparison.ipynb`, `…_AXEC2_parameter_comparison.ipynb` for parameter-sensitivity sweeps.

Plotting and catalog prep:
- `nonlinloc_apr_14_jun_01_plots.ipynb` — current plotting notebook for combined cleaned per-station results.
- `mldd_apr_20_28_plots.ipynb` — MLdd 24-hr-window plots around the eruption.
- `parse_mldd_catalog.ipynb` — converts Wang's MLdd file into the workflow's catalog format.
- `tmid_analysis.ipynb` — sensitivity to the analysis-window center.

### Files kept locally but not tracked

- Most of Christian Baillard's plotting/figure framework: `scripts/ARTICLE_*.py`, `scripts/POSTER_*.py`, `scripts/test_*.py`, `scripts/untitled*.py`. Kept on disk for reference but not part of the active workflow, and confirmed not imported by anything that is. **Exception**: `ARTICLE_deformation.py`/`ARTICLE_deformation_util.py` are tracked — they're the source `deformation_util.py`/`deformation_analysis.py` were ported from (see [Deformation modeling](#deformation-modeling-dmodels-comparison)).
- Orphan utilities (`build_cmap.py`, `check_*`, `clean_*`, `concatenate_*`, etc.) and superseded notebooks (`master_shear_wave_splitting_workflow.ipynb`, the `notebooks/` folder, etc.).
- All data, results, pickles, waveforms (`*.mseed`), and figures (`*.png/pdf/jpg`) — see `.gitignore`.
- A couple of notebooks with large embedded-output JSON (`axial_splitting_mldd_AXEC1_batched.ipynb`, `axial_splitting_mldd_AXEC2_batched.ipynb`, `nonlinloc_apr_14_jun_01_plots.ipynb`) and two not-yet-classified notebooks (`fix_density_plots.ipynb`, `station_tilt.ipynb`).

## Getting Started

### Prerequisites
- conda (Miniconda or Anaconda)
- Python 3.11 (created by `env.sh` in the `seismo` conda environment)
- ObsPy ≥ 1.4.0
- NumPy, SciPy, Pandas, Matplotlib
- scikit-learn (clustering)
- `multitaper` (replaces deprecated `mtspec`)
- `pyproj`, `utm` (geodesy/projections used by the Baillard-derived modules)
- `cartopy` (mapping, `scripts/map_sws_spatial.py`)
- `h5py`, `pyvista` (reading/plotting PyLith `.h5` output in `pylith_axial/scripts/`)
- The local modified SWSPy in `swspy/` (already in this repo)
- For `pylith_axial/` only: a separate local **PyLith 5.0.1** install, activated via
  `pylith_axial/activate_pylith.sh` — not managed by `env.sh` (see [Stress modeling](#stress-modeling-pylith_axial)).

### Setup
```bash
git clone https://github.com/mhemmett/axial-sws.git
cd axial-sws
bash env.sh
```

`env.sh` creates (or reuses) the **`seismo`** conda environment, installs everything in `requirements.txt` into it via pip, registers a Jupyter kernel, and prints the activation command. Run it once after cloning; safe to re-run.

Activate it before working in the repo:
```bash
conda activate seismo
```

### Verifying the local SWSPy
```python
import sys
sys.path.insert(0, 'swspy')
import swspy
print(f"SWSPy loaded from: {swspy.__file__}")
```

## Usage

### Production splitting analysis (MLdd catalog)
Open the per-station notebook for the station of interest:
```bash
jupyter notebook scripts/axial_splitting_mldd_AXEC2_batched.ipynb
```
This runs the modified-SWSPy pipeline on Wang's MLdd catalog with dynamic windowing and Wang (2024) S-pick uncertainties. The `_batched` variant chunks events into ~100-event batches and writes `splitting_results_mldd_2015_2021_<station>_batch_<N>.csv` to `results/`.

### Cross-validation against Baillard's method (NonLinLoc catalog)
```bash
jupyter notebook scripts/axial_splitting_nonlinloc_swspy_AXAS2.ipynb
```
Runs the same modified SWSPy on the NonLinLoc/Wilcock–Baillard catalog. Pair with the corresponding `axial_splitting_nonlinloc_swspy-mfast_*.ipynb` to compare against Baillard's single-window output.

### Plotting
```bash
jupyter notebook scripts/nonlinloc_apr_14_jun_01_plots.ipynb
```
Combines per-station cleaned result CSVs and produces rose plots, density polar plots, and time-variation panels for the April 14 – June 1, 2015 window around the eruption.

### Programmatic
```python
from splitting_functions import perform_splitting_on_organized_waveforms

results = perform_splitting_on_organized_waveforms(
    passing_waveforms,
    use_dynamic_params=True,   # T_dom-based windowing
    method='swspy',            # or 'baillard' for the cross-check
)
```

## Status

- **S-wave incidence angle / LQT rotation fix**: complete — see [S-wave incidence angle & LQT rotation](#s-wave-incidence-angle--lqt-rotation). `coord_system` parameter added throughout the pipeline; production now runs LQT + PyKonal-FMM incidence at a 35° cut.
- **AXEC2 production run complete**: full 2015–2021 MLdd catalog, 497/497 batches, 88,649 successful LQT + PyKonal-FMM splitting measurements. Rose plots (7-panel + annual, all-data and `Q_w≥0.5` filtered) and a temporal histogram are built from this dataset — see `build_production_rose_plots_axec2*.py`, `axec2_temporal_histogram_lqt_pykonal.py`.
- **Next**: re-run AXAS1, AXAS2, AXCC1, AXEC1, AXEC3 with the same LQT + PyKonal-FMM pipeline (currently only AXEC2 has been fully re-run; the deformation-modeling comparison below still uses the older P-Jurkevics-incidence per-station results for the other 5 stations as an interim measure), then fold AXCC1 in once tilt-corrected, then complete the 2022–2026 RT-catalog extension (same caldera stations, real-time MLdd catalog variant - see [Data](#data)).
- **Deformation modeling started**: DMODELS (Okada dike + Yang spheroid) forward-model comparison against observed φ/δt, both replicating Baillard's original 2019 comparison and a new version using this repo's own splitting results — see [Deformation modeling (DMODELS comparison)](#deformation-modeling-dmodels-comparison) below. AXID1 is excluded from the new version (not part of the production catalog); the exact DMODELS source parameters behind most of the individual `.xyzuvw` files are not recoverable from the current `axial_comb_V0.m`/`axial_dike_V0.m` (only the most recent scenario in each is preserved).
- **Validation**: modified SWSPy and Baillard's single-window method agree to within ~5% on SNR, ≤1° on geometry, and consistent φ/δt across the NonLinLoc cross-check catalog.

## Deformation modeling (DMODELS comparison)

`Axial_Deformation/` (untracked — DMODELS output is data, not source) holds displacement-grid
output from Christian Baillard's 2019 MATLAB DMODELS tool (`axial_comb_V0.m`: Okada dike(s) +
Yang/Mogi spherical source superposition; `axial_dike_V0.m`: single-dike sandbox), 12 grids total
(`def_pre_1/2.xyzuvw`, `def_syn_1..10.xyzuvw`). `deformation_util.py`/`deformation_analysis.py`
read these grids, compute 2D isotropic-elastic stress/strain and the modeled principal
(most-compressive) stress direction, and compare it against observed splitting fast direction /
delay time at each station.

Two comparison paths exist:
- **Replicated Baillard comparison** (`deformation_analysis.py`) — uses Baillard's own hardcoded
  per-station splitting/deformation observations (`get_station_obs` in `deformation_util.py`),
  covering all 8 catalog stations including AXID1.
- **This repo's data** (`deformation_analysis_hemmett.py` and related `_hemmett` scripts) — replaces
  those hardcoded literals with median φ/δt computed from this repo's own splitting-pipeline
  results (`get_station_obs_hemmett`), restricted to the 6 production stations (AXID1 dropped, not
  part of the production catalog — see `rose_plots_temporal.py`'s `STATION_ORDER`). `dz`
  (vertical deformation) has no splitting-pipeline analog and is carried over from Baillard's
  original values for the 6 remaining stations.
  - `deformation_geometry_stress_hemmett.py` — map of modeled principal-stress-direction tick
    marks vs. observed φ, the latter spatially interpolated **along PyKonal-traced rays**
    (per-voxel circular median, ≥5 rays/voxel — same method as `sws_percent_anisotropy.py`)
    rather than one value per station.
  - `deformation_change_pre1_syn6_hemmett.py` / `deformation_change_voxel_hemmett.py` — pre→syn
    eruption change in modeled stress/orientation vs. change in observed δt/φ, with a linear fit
    + RMS; the `_voxel` version extends this from 6 station points to every ray-covered voxel.

**Caveat**: the DMODELS `.xyzuvw` grids' generating MATLAB scripts were edited in place run-to-run
(changing a `scenario` switch and hand-editing the output filename each time) rather than being
checked in per-config, so the exact dike/fault parameters behind most individual files (e.g.
`def_pre_1`, `def_syn_6`) are not recoverable from the current `axial_comb_V0.m`/`axial_dike_V0.m` —
only the last-run scenario in each script is preserved. The `_hemmett` comparisons currently also
use the **older, pre-LQT/PyKonal-FMM** per-station splitting results for 5 of 6 stations (only
AXEC2 has been fully re-run — see [Status](#status)); revisit once the full re-run is complete.

## Stress modeling (`pylith_axial/`)

Alongside the observational splitting analysis, `pylith_axial/` develops a companion **PyLith 5.0.1
poroelastic stress model** of the caldera, aimed at eventually connecting the modeled stress field to
the observed φ/δt evolution.

**Geometry**: two outward-dipping ring faults (east and west) and a spherical Mogi-like inflation
source centered in the caldera at 3.33 km depth. Topography is treated as flat, with ~1.5 km of
lithostatic stress from the overlying seawater column.

**Physics**: full poroelastic modeling (`pylith.materials.Poroelasticity` +
`IsotropicLinearPoroelasticity`) with depth-dependent properties (a highly fractured upper-edifice
layer over MORB crust). Inflation is driven by prescribing fluid pressure at the Mogi cavity wall
(rather than a mechanical traction), which poroelastically stresses the surrounding crust and, in turn,
the two ring faults — modeling how central-source inflation loads the fault system.

**Workflow**: `scripts/run_sweep.py` runs a two-pass sweep per scenario — a "pre" pass solves the
no-fault poroelastic inflation problem and records the Cauchy stress field, which is resolved onto each
fault plane to get shear traction and Coulomb stress change; a "syn" pass then converts that resolved
traction into a prescribed slip and reruns the full poroelastic + kinematic-fault model. Scenarios are
swept over injected fluid-pressure amplitude (`output/pre_dP*`, `output/syn_dP*`) and pore-pressure
level (`output/pore_Pf*`); mesh generation lives in `mesh/generate_mesh.py`, and per-scenario configs
are generated into `scenarios/`.

**Long-term plan**: run repeated inflation/stress cycles bracketing the April 2015 eruption (stress
build-up pre-eruption, relief post-eruption), then use the resulting stress fields to perturb
penny-shaped, fluid-saturated spherical cracks and forward-model shear-wave splitting on synthetic
seismograms — providing a physics-based prediction to compare against the observational φ/δt results
from the rest of this repository.

This component is independent of the observational pipeline (`scripts/`, `swspy/`) — it does not import
from or write into `data/`/`results/`, and requires a separate local PyLith installation
(`pylith_axial/activate_pylith.sh`, currently pointing at `~/Downloads/pylith-5.0.1-macOS-12.0-arm64`).

## Citation Pointers

- **Wang, K. (2024)** — MLdd catalog and S-pick uncertainty model used as the production input.
- Wilcock, W.S.D. & Baillard, C. — NonLinLoc / kurtosis-pick catalog used for method development and cross-validation.
- Hudson, T.S., Asplet, J., Walker, A.M. (2023). "Automated shear-wave splitting analysis for single- and multi-layer anisotropic media". *Seismica* — SWSPy.
- Silver, P.G., Chan, W.W. (1991). "Shear wave splitting and subcontinental mantle deformation". *JGR*, 96(B10), 16429–16454.
- Teanby, N.A., Kendall, J.M., Jones, R.H., Barkved, O. (2004). "Stress-induced temporal variations in seismic anisotropy observed in microseismic data". *GJI*, 156(3), 459–466.
- Wessel, A., Savage, M., Teanby, N. (2017). "Multiple Filter Automatic Splitting Technique (MFAST)", v2.2. DOI: 10.5281/zenodo.1042760.
- Jurkevics, A. (1988). "Polarization analysis of three-component array data". *BSSA*, 78(5), 1725–1743.

## Acknowledgements

This work builds substantially on **Christian Baillard's** earlier shear-wave-splitting code at Axial Seamount; many supporting scripts in `scripts/` are his and are still in use. The **MLdd catalog and S-pick uncertainty model** are from **Kaiwen Wang's** work on Axial seismicity. NonLinLoc relocations and kurtosis picks are from **William Wilcock and Christian Baillard**.

## Contact

Michael Hemmett
University of Washington
mhemmett@uw.edu

## License

MIT — see `LICENSE`.
