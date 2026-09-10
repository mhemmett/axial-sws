# Claude Code prompt — rebuild the SWS delay-time tomography to match Johnson et al. (2011)

Paste everything below the line into Claude Code, run from the `axial-sws` repo root.

---

Rewrite my shear-wave-splitting delay-time tomography as a new Python module that faithfully
reproduces the 2-D tomography method of Johnson, Savage & Townend (2011), *Distinguishing between
stress-induced and structural anisotropy at Mount Ruapehu volcano, New Zealand*, JGR 116, B12303,
Section 3.2–3.3. Match the repo's existing conventions (Python/NumPy/SciPy, the `swspy` + `scripts/`
pipeline, MLdd production catalog by default). Do **not** touch the Baillard or SWSPy measurement
code — this is a downstream step that consumes their φ/δt output.

## Before writing anything

1. **Read `README.md` and `CLAUDE.md`** for catalogs, QC thresholds, coordinate conventions, and where
   splitting results live (`results/splitting_results_mldd_2015_2021_<station>_batch_<N>.csv`). Ask me
   for a path rather than guessing if data isn't present — don't fabricate.
2. **Find and reuse my existing raypath / geometry code.** Search `scripts/splitting_functions.py` (and
   imported modules) for the incidence-angle and ray-geometry helpers — `calculate_incidence_angle*`
   (Jurkevics variants), the TauP/1-D-velocity-model usage, station and hypocenter handling. The new
   inversion must consume ray geometry from that existing code, not recompute raypaths from scratch.
   Also locate whatever current tomography/inversion script exists so you can see input/output formats
   and I/O conventions; show me what you found and how you plan to reuse it before proceeding.
3. **Confirm inputs with me:** per-ray records of station, event hypocenter, φ (fast polarization),
   δt (delay time), σ_δt (delay-time uncertainty / the 95% CI from the F-test), and whether nulls and
   the >35° incidence-angle rays are already filtered out (the paper excludes incidence >35° per Nuttli
   1961, and treats null results as no splitting — apply the same filters if my data doesn't already).

## Method to implement (follow the paper's equations)

### A. Core delay-time tomography (Section 3.2)

- **Model.** Delay time accumulates along the raypath (Crampin 1991; Zhang et al. 2007): the total δt of
  a ray is the sum over grid blocks it traverses,
  **δt_r = Σ_b (s_b · L_rb)**  (eq. 1),
  where `s_b` = strength of anisotropy per block (s/km, the model parameter) and `L_rb` = length of ray
  `r` within block `b`. This linear/additive assumption is a deliberate first-order approximation — keep it.
- **Gridding.** Quad-tree adaptive grid (Townend & Zoback 2001, 2004): start coarse and recursively
  subdivide any block traversed by **more than 65 rays**; **exclude** blocks with **fewer than 20 rays**;
  **minimum block size 5 km square**. Report how many blocks are created vs. used (paper got 149 created,
  121 used). Build the grid in a projected coordinate system (they used NZ Map Grid) — use the repo's
  existing projection for Axial rather than lat/lon degrees.
- **Design matrix `G`.** Row per ray, column per block, entries = `L_rb` (ray length in block). `d` = vector
  of measured δt; `m` = vector of block strengths `s_b`.
- **Weighted least-squares inversion** (eq. 2):
  **Gᵀ C⁻¹ d = (Gᵀ C⁻¹ G) m**,
  solved as a **bounded** linear least-squares problem (paper used Matlab `lsqlin` active-set; use
  `scipy.optimize.lsq_linear`, or `scipy.optimize.nnls` / a constrained solver, applying `C⁻¹` as a
  whitening transform on `G` and `d`). Constraints: **s_b ≥ 0** and **s_b ≤ δt_max / L_min(b)** where
  `L_min(b)` is the width of the smallest quad-tree block.
- **Error covariance `C`** (eqs. 3–4). Diagonal:
  **C_ii = σ_d²(i) + L_b² · σ̂_m² · n_b(i)/n**,
  off-diagonal (constant): **C_ij = σ̂_m² · L_b²/n²**, where `σ_d²(i)` = squared standard error of the i-th
  δt measurement, `L_b` = block length scale, `n_b(i)` = number of blocks ray i passes through, `σ̂_m²` =
  mean model variance when weighted as `1/σ_d(i)`, and `n` subdivides each ray's in-block segments. Let `n`
  be a parameter; the paper swept n = 1…1000 (used n = 100) and found results insensitive to it — reproduce
  that sensitivity check.
- **Output:** the per-block anisotropy-strength map (s/km) plus the model variance matrix.

### B. Spatial averaging of fast polarization φ (Sections 3.3, 4.2)

- On the same quad-tree grid, compute a **weighted circular mean φ per block** from every ray passing
  through it, using **1/d² weighting** (`d` = distance of the block from the recording station; Audoine et
  al. 2004) — support `none`, `1/d`, `1/d²` and default to `1/d²` (least scatter).
- Also implement the **tomography-based weighting** option (eqs. 6–7):
  **w_rb = s_b / δt_r** (anisotropy strength in the block normalized by the ray's total δt), and
  **φ̄_b = ½·atan2( Σ_r sinφ_r·w_rb / Σ w_rb , Σ_r cosφ_r·w_rb / Σ w_rb )** — mind the 180°-periodicity of φ
  (double-angle averaging), don't treat φ as a normal linear variable.
- **Plot/keep a block only if** its φ standard deviation < **30°** and standard error of the mean < **10°**
  (drops high-scatter / multimodal blocks). Output as rose diagrams centered on each block plus the mean
  bar, matching my existing plotting style.

### C. Resolution & significance tests (Section 3.2, end)

- **Checkerboard test** (eq. 5): build a synthetic model `m_CB` on a regular grid with blocks alternating
  strengths (paper used 0.01 and 0.02 s/km), forward-compute synthetic data **d_CB = G_CB · m_CB**, add
  standard-normal random noise, invert, and map recovery to show where structure is retrievable.
- **Resolution matrix** — compute it and confirm the over-constrained design (paper's was ≈ identity).
- **Statistical-significance limit** — from the model variance matrix, contour the region where
  **log₁₀ σ_m(ii) = −5.5** (their conservative cutoff); expose the threshold as a parameter and show a few
  contour choices so I can judge sensitivity.

## Deliverables

- A new, importable Python module (e.g. `scripts/sws_tomography.py`) with clearly separated functions for:
  quad-tree gridding, `G`/`C` construction, the constrained WLS inversion, spatial φ averaging, and the
  checkerboard/resolution/significance diagnostics — mirroring `splitting_functions.py` docstring and naming
  style.
- A short demo/driver (notebook cell or script) that runs the full chain on the MLdd results for one or more
  stations and produces: the δt-strength map, the φ spatial-average (rose) map, and the checkerboard figure.
- Inline citations in docstrings to the paper's equation numbers (eqs. 1–7) so the mapping from code to
  method is auditable.
- A brief `docs/` note explaining design choices and any place the Python solver departs from Matlab `lsqlin`.

## Verification (do this, don't skip)

- Unit-test the geometry: a single ray crossing a known set of blocks must give `Σ_b L_rb` = the ray's total
  in-grid length; a synthetic ray through a uniform-strength model must return δt = s·L.
- Confirm the checkerboard recovers input strengths within noise in well-sampled blocks.
- Sanity-check against the paper's numbers where my data overlaps in style (block counts of the same order,
  strengths ~0–0.025 s/km, φ means physically reasonable).
- Show me a diff/summary of what changed and run the demo before declaring done. Two splitting methods
  (SWSPy-variant vs Baillard) coexist by design — this tomography must work from either's φ/δt output; don't
  hard-wire it to one.

Ask me the open questions (data paths, whether filters are pre-applied, which station(s) to demo on) up front
rather than guessing.
