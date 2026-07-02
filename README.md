# Axial Seamount Shear-Wave Splitting Analysis

Shear-wave splitting analysis of local seismicity at Axial Seamount on the Juan de Fuca Ridge, using OBS data from the OOI Regional Cabled Array. The goal is to track the temporal evolution of stress and crack-induced anisotropy across the **April 2015 eruption** and the long inflation cycle that has followed it.

## Project Overview

Splitting parameters — fast polarization direction (φ) and delay time (δt) — measure the orientation and density of aligned cracks in the shear-wave window. Tracking how φ and δt change in space and time should constrain how the volcanic stress field reorganizes around eruptive episodes at submarine ridge volcanoes, and feeds into hazard / forecasting work on the Juan de Fuca Ridge.

The long-term arc is to start just before the April 2015 eruption and run through to the present, currently re-inflated state of the caldera. As of this writing the analysis covers 2015–2021 from the OOI cabled array; future work extends it forward to today and adds the 2022–2024 North Rift Zone OBS deployment.

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
- **AXAS1, AXAS2** (south caldera flank)
- **AXEC1, AXEC2, AXEC3** (eastern caldera floor)

**AXCC1** (central caldera) tipped over in March 2015 due to caldera volumetric inflation and is treated separately — it will be folded back in once orientation/tilt corrections are settled.

Future additions:
- **AXID1** for completeness of the cabled array
- The 15-station 2022–2024 temporary OBS deployment along the **North Rift Zone**

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
- **Hierarchical Ward clustering** (replacing upstream DBSCAN) in a circular-safe `(δt cos 2φ, δt sin 2φ)` coordinate space, matching MFAST.
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
| Clustering | Ward, circular-safe `(δt, φ)` | parameter-space minima |
| Quality metric | F-statistic 95% region | RMS difference |

### Quality control

Applied uniformly before splitting:
1. **SNR** (horizontal) > 2.0; signal window S → S+2.0 s; noise window adjusted to avoid P-coda.
2. **P-wave rectilinearity** > 0.7 over P ± 0.12 s (Jurkevics 1988).
3. **Incidence angle** < 30° from vertical (from P polarization) — keeps events inside the shear-wave window.
4. **Magnitude** filter (default M > 0).

## Repository Layout

```
axial-splitting-ml/
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
└── pylith_axial/                  # PyLith poroelastic caldera stress model — see below
```

### Active code in `scripts/`

Core analysis library:
- `splitting_functions.py` — main workflow: QC, dynamic-parameter pipeline, both Baillard and SWSPy entry points.
- `teanby_clustering.py` — clustering helpers used by `splitting_functions.py`.
- `get_all_traces.py` — automated waveform retrieval from the cabled array.

Baillard-derived modules (still imported by the active workflow):
- `sws_methods.py`, `shearwavesplit.py`, `plotwaveform.py`, `GMT.py`, `projection.py`, `util.py`, `sws_vertices.py`.

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

- All of Christian Baillard's plotting/figure framework: `scripts/ARTICLE_*.py`, `scripts/POSTER_*.py`, `scripts/test_*.py`, `scripts/untitled*.py`. These are kept on disk for reference but are not part of the active workflow.
- Orphan utilities (`build_cmap.py`, `check_*`, `clean_*`, `concatenate_*`, etc.) and superseded notebooks (`master_shear_wave_splitting_workflow.ipynb`, the `notebooks/` folder, etc.).
- All data, results, pickles, waveforms (`*.mseed`), and figures (`*.png/pdf/jpg`) — see `.gitignore`.

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
git clone https://github.com/mhemmett/axial-splitting-ml.git
cd axial-splitting-ml
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

- **Active production run**: AXEC2 across the full 2015–2021 MLdd catalog, batched, currently in the high-30s of ~100-event batches.
- **Validation**: modified SWSPy and Baillard's single-window method agree to within ~5% on SNR, ≤1° on geometry, and consistent φ/δt across the NonLinLoc cross-check catalog.
- **Next**: re-run AXAS1, AXEC1, AXEC3 with the same dynamic-parameter / Wang-S-pick pipeline, then fold AXCC1 in once tilt-corrected, then bring in the 2022–2024 North Rift Zone deployment.

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
