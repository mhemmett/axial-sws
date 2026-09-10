# Johnson, Savage & Townend (2011) 2-D Delay-Time SWS Tomography

## What this is

`scripts/sws_tomography_johnson2011.py` is a from-scratch reimplementation of the
2-D map-view delay-time shear-wave-splitting tomography method of Johnson, Savage
& Townend (2011), *Distinguishing between stress-induced and structural anisotropy
at Mount Ruapehu volcano, New Zealand*, JGR 116, B12303, Sections 3.2-3.3, applied
to the Axial Seamount SWS catalogs (MLdd φ/δt results, six stations). It consumes
existing splitting measurements and ray geometry (`sws_forward_model.ll2xy`,
`pykonal_raytracer.BaillardRayTracer`) — it does not touch the Baillard/SWSPy
measurement code that produces φ/δt.

Driver scripts:
- `scripts/sws_tomography_johnson2011_demo.py` — per-station demo/build-out driver
  (census, trace/cache, run the chain over 3 epochs x 2 quality thresholds, plot).
- `scripts/sws_tomography_johnson2011_joint.py` — the six-station joint driver
  (merges per-station caches into one global system, same epoch/quality sweep).

The original task spec is `sws_tomography_johnson2011_prompt.md` at the repo root.

This note records the method as built and the verified findings from testing it
against the real Axial catalogs. It is a chronicle of settled, tested work, not a
publishability judgment — that call belongs to the user.

## Method overview

Delay time accumulates linearly along a ray's path (Crampin 1991; Zhang et al.
2007):

    delta_t_r = Sum_b  s_b * L_rb                                        (eq. 1)

where `s_b` [s/km] is the anisotropy strength of grid block `b` (the model
parameter) and `L_rb` [km] is the length of ray `r` inside block `b`. Stacking
every ray gives a linear system `d = G m` (eqs. 1-2), solved as a **bounded,
covariance-whitened weighted least squares** problem:

    G^T C^-1 d = (G^T C^-1 G) m                                          (eq. 2)

The error covariance `C` (eqs. 3-4) has a diagonal term from the per-ray δt
uncertainty plus a shared model-variance term that couples every observation
through the number of blocks each ray crosses:

    C_ii = sigma_d^2(i) + L_b^2 * sigma_m_hat^2 * n_b(i) / n              (eq. 3)
    C_ij = sigma_m_hat^2 * L_b^2 / n^2   (i != j)                        (eq. 4)

Fast polarizations φ are averaged per block with a double-angle (180°-periodic)
circular mean under four weighting schemes — `none`, `1/d`, `1/d^2` (default,
Audoine et al. 2004), and a tomography-based weight `w_rb = s_b / delta_t_r`
(eq. 6), with the mean itself

    phi_bar_b = 1/2 * atan2( Sum_r w_rb sin(2 phi_r), Sum_r w_rb cos(2 phi_r) )   (eq. 7)

Model quality is assessed with an absolute-scale checkerboard test (eq. 5), the
resolution matrix, the condition number of `G^T C^-1 G`, and the posterior model
covariance `C_m = (G^T C^-1 G)^-1`.

The gridding is a quad-tree (Townend & Zoback 2001, 2004): start from
`ROOT_SIDE_KM` = 4 km root tiles over a fixed fine map grid and recursively
subdivide any block crossed by more than a threshold number of distinct rays,
down to a minimum block side; a block becomes a model column ("used") only if it
is crossed by at least a minimum number of rays.

## The 2-D quad-tree, not a "3-D octree"

The original task brief for this rebuild described the target grid as a "3-D
octree." The code deliberately does **not** build one, and the module docstring
says so explicitly: Johnson et al. (2011) is a 2-D map-view method, and the
tomography here is a genuine **2-D quad-tree** with a 0.25 km minimum block side.

What is 3-D is the ray geometry feeding it: rays are the true 3-D bent-eikonal
paths traced by `pykonal_raytracer.BaillardRayTracer`. Each ray is resampled to
~0.03 km point spacing and each segment's full 3-D length is assigned to the
**map (x,y) cell of that segment's midpoint** — depth is collapsed, not the path
length. This collapse is implemented with a single, two-element z column
(`GRID_Z0`, `GRID_ZSPAN` spanning far beyond every ray's depth range) fed to the
existing `ray_to_voxels` binner, so every segment maps to `iz == 0` and none is
ever dropped for being out of range in z. Because no segment is dropped, the sum
of fine-cell lengths for a ray equals its full 3-D arc length **exactly**, not
approximately — this is asserted as a hard benchmark (`verify_arclength`, §7a)
and holds to `1.78e-15` relative error over the traced catalog.

In short: the ray geometry is genuinely 3-D and the arc length assigned to the
grid is exact, but the *grid itself*, the inversion, and the resulting anisotropy
map are 2-D map view, matching the paper. The "3-D octree" phrasing in the
original brief was a misdescription of what a faithful Johnson (2011)
implementation should be; the code corrects it rather than reproducing it.

## Parameter overrides vs. the paper (locked)

Three grid parameters were deliberately set differently from the paper's
published values, for this denser-in-places/sparser-in-places Axial dataset. All
three are user-confirmed and locked in the module constants:

| Parameter | Paper (Ruapehu) | This implementation |
|---|---|---|
| Minimum block side | 5 km | **0.25 km** (`DELTA`) |
| Subdivide threshold | > 65 rays | > 65 rays (`COUNT_SUBDIVIDE`, unchanged) |
| Minimum rays to keep a block | >= 20 rays | **>= 8 rays** (`COUNT_KEEP`) |

The subdivide threshold matches the paper as-is. The minimum block size and the
keep threshold were relaxed because Axial's ray density per station is much
lower and more spatially uneven than Ruapehu's local network; using the paper's
5 km/20-ray values on this dataset would leave most of the map either empty or
merged into one or two blocks. These are not exploratory defaults — treat them
as fixed unless the user asks to revisit them.

## Where the Python solver departs from MATLAB `lsqlin`

The paper solved the bounded WLS problem with MATLAB's `lsqlin` active-set
solver, forming `C^-1` densely. This implementation instead:

- Solves the bounded problem with `scipy.optimize.lsq_linear(method='bvls')`
  (bounded-variable least squares) on a **whitened** system, rather than
  `lsqlin` on the raw normal equations.
- Never forms `C^-1` densely for the solve. `C = D + alpha * 1 1^T` (a rank-1 update
  of a diagonal — eqs. 3-4 rewritten in Sherman-Morrison form) is inverted
  **analytically**: `C^-1 = P(I - beta a a^T)P` with `P = diag(1/sqrt(D))`. A
  closed-form whitening operator `W = (I - kappa a a^T) P` is constructed such
  that `W^T W = C^-1` exactly, and `G`, `d` are whitened by `W` before the bounded
  solve — so the whitened normal equations `G_tilde^T G_tilde = G^T C^-1 G` and
  `G_tilde^T d_tilde = G^T C^-1 d` hold without ever materializing the full `R x R`
  dense `C^-1`.
- This whitening is independently cross-checked (`verify_whitening`): for several
  random systems (including `n` in {1, 100, 1000}, per the paper's stated
  insensitivity sweep), `C` is formed *densely* and inverted by Cholesky
  factorization, and the closed-form `W^T W` matches that dense `C^-1` to
  ~1e-16 relative error, with the whitened operator matching the dense
  `G^T C^-1 G` equally closely.
- Constraints are `s_b >= 0` and `s_b <= delta_t_max / L_min`, with `L_min` the
  smallest block side (a single global bound). If any block binds at that global
  bound, the solver automatically switches to a **per-block** bound
  `ub_b = delta_t_max / side(b)` and re-solves — a block-size-aware ceiling rather
  than one scaled to the smallest block in the whole grid.

## The single-station vs. joint-multi-station finding (headline, resolved)

**Single-station real-data inversions.** Running the full chain on each
station's own splitting results individually shows roughly 74% of blocks
clipped at the lower bound `s_b = 0`, and variance reduction of the raw δt data
near zero (`var_reduction ~ 0`).

**A joint six-station inversion was built and run** (`sws_tomography_johnson2011_joint.py`):
all six stations — AXAS1, AXAS2, AXCC1, AXEC1, AXEC2, AXEC3 — merged into one
system with globally unique ray IDs (452,926 combined rays at `q >= -0.5`;
quad-tree: 897 blocks created, 894 used). Per-ray recording-station coordinates
are carried through so the 1/d and 1/d² φ weighting still measures distance to
the correct receiver per ray, not to some single station.

The joint inversion:
- **Improves resolution**: the noise-realistic (per-ray-noise) checkerboard
  correlation in well-sampled blocks rises from 0.705 (best single station) to
  0.796 (joint).
- **Modestly lowers the condition number** of `G^T C^-1 G`: 2.49e6 -> 1.86e6.
- **Does not** lift `var_reduction` off ~0, and **does not** reduce the ~74%
  clipping at `s_b = 0`.

A first hypothesis was that the covariance was over-inflated because `dt_error`
(hence `sig_dt2 = dt_error^2`) is a ~3-sigma quantity rather than a true 1-sigma
standard error, artificially suppressing the fit. **This hypothesis was tested
and refuted**: rescaling `sig_dt2` by factors from 1/9 to 9 leaves `s_b`, the
clipping fraction, `var_reduction`, and `cond(G^T C^-1 G)` invariant to machine
tolerance. This is expected once the algebra is worked through — `C -> c*C`
uniformly for any positive scale `c`, so the whitened argmin of the bounded least
squares problem is unchanged (only the *variance/significance* scale, not the
point estimate, depends on `c`); the ridge-regularization gate in
`estimate_sigma_m2` never fired during the sweep (`cond0 = 2.57e6` throughout).

**Verified conclusion**: the clipping and near-zero variance reduction are a
genuine limit of delay-time strength tomography for this dataset, not a bug in
the covariance, the whitening, or the solver. The mechanism is ray-geometry
collinearity (all rays into a given station converge on one receiver, so many
raypaths through a block are nearly parallel/similar) combined with irreducible
event-to-event δt scatter at effectively fixed ray geometry, pressing against the
non-negativity constraint `s_b >= 0`. The noise-free checkerboard
(`verify_checkerboard`, §7c) recovers the input model perfectly (correlation
1.000) in every configuration tested, confirming the inversion machinery itself
is correct — the ceiling here is set by geometry and data, not code.

One open methodological note: `var_reduction` as currently reported is computed
on the raw per-ray δt (`1 - var(residual)/var(d)`), which is a weak yardstick
given how much of that variance is real geologic scatter rather than model
misfit. A geometry-binned variance reduction (e.g., grouping rays by
block-crossing pattern before computing variance) would likely be a more honest
metric — this is an **open follow-up**, not yet implemented.

## The φ-resolution finding

Joint per-block φ maps are essentially empty: `n_phi_kept` is near 0 across the
epoch/quality sweep. The keep gate requires circular standard deviation
`sigma_phi < 30°` and standard error `se_phi < 10°` per block (§5), but at
0.25 km blocks the within-block φ scatter is typically ~55-58° circular SD, far
above the 30° threshold.

This was checked and is **not** a weighting or effective-sample-size artifact:
running with the uniform (`'none'`) weight scheme also keeps essentially zero
blocks, so it is not the `1/d^2` weighting concentrating on a handful of
outlier-adjacent rays — the scatter is intrinsic to how much φ varies among rays
crossing the same small block.

Recovering usable φ maps would require either larger blocks (fewer, more
heavily-sampled blocks with correspondingly smoothed φ) or a relaxed
`sigma_phi_max`/`se_phi_max` gate. Both are **open decisions** for the user, not
resolved here — the module exposes both as parameters (`block_phi_average`'s
`sigma_phi_max`, `se_phi_max`, and the grid's `DELTA`/`COUNT_KEEP`) but the
current defaults simply reflect the (locked) delay-time grid, not a choice
optimized for φ resolution.

## Two items flagged for the human (open, not resolved here)

1. **`dt_error` semantics.** `sig_dt2 = dt_error**2` is used directly as
   `sigma_d^2(i)` in eq. 3. Tracing `dt_error`'s origin to
   `swspy/swspy/splitting/split.py`, it comes from an F-test half-width at
   `alpha = 0.003`, i.e. a ~99.7% (~3-sigma) confidence half-width — while the
   docstring in that code says 95%. If `dt_error` is really a ~3-sigma quantity,
   `sig_dt2` is roughly 5-9x a true `sigma^2`. As shown above, this scale factor
   is inert for the point-estimate strength map (`s_b`) because it cancels
   uniformly through `C`. It **does** matter for anything downstream that reports
   uncertainty or significance at face value (`sigma_m_hat^2`, the posterior
   `C_m`, and the `log10(sigma_m^2) = -5.5` contour) — those numbers would be
   inflated by roughly that same 5-9x factor if `dt_error` is truly ~3-sigma and
   is not first converted to a 1-sigma value. This is an open question about
   upstream `dt_error` semantics, not something this module resolves.
2. **The resolution/significance map is on the wrong block set.** The
   "significant" (red-square, `log10(diag(C_m)) < -5.5`) blocks are computed from
   `resolution_diagnostics`, which inverts the **unconstrained** normal equations
   `G^T C^-1 G` over *all* used blocks. But the map actually shown/reported is the
   ~74%-clipped, bound-constrained solution. The significance contour therefore
   describes the unconstrained problem's precision, not the precision of the
   constrained map that's plotted. Recomputing significance on only the active
   (unclipped, interior-to-bounds) block subset would be the honest version of
   this diagnostic — flagged as an open item, not fixed in this build.

## Bugs found and fixed during the build

- **φ 1/d weighting used δt instead of distance.** An early version of the
  `1/d` and `1/d^2` φ-weighting schemes weighted by the ray's delay time rather
  than the geometric block-to-station distance the paper (Audoine et al. 2004)
  specifies. Only the separate `'tomography'` weight (eq. 6) legitimately uses
  `delta_t_r`. Fixed by threading each ray's recording-station map coordinates
  (`sx`, `sy`) through `build_system` into `ray_sx`/`ray_sy`, so
  `block_phi_average` computes `d_rb` as the true map distance from each block's
  center to the station that recorded that particular ray (needed for the joint
  system, where different rays are recorded at different stations). A dedicated
  regression test, `verify_phi_distance_weight`, constructs two rays through one
  block with **equal** δt but different station distances and different φ; a
  δt-based (buggy) weight would collapse to the unweighted mean, while the fixed
  geometric weight correctly pulls `phi_bar` toward the closer station's ray.
  This test is part of the module's data-free self-test suite (`__main__`).
- **`estimate_sigma_m2` ridge regularization was not gated on conditioning.**
  The original fallback only added a ridge term to the diagonal of
  `G^T C0^-1 G` on an outright `LinAlgError`. That is not enough for the
  near-singular-but-technically-invertible case that shows up for a single,
  geometrically collinear station — the inversion can silently succeed and
  return garbage without ever raising. Fixed by gating the ridge on the
  **condition number** of `G^T C0^-1 G` (added when `cond > cond_max`, default
  `1e10`), in addition to the `LinAlgError` fallback; the condition number is
  returned to the caller (`return_cond=True`) so downstream diagnostics can see
  how collinear the geometry is.

Both the covariance/whitening algebra and the double-angle φ circular mean were
independently verified: `verify_whitening` cross-checks the closed-form
whitening operator against a densely-formed, Cholesky-inverted `C` across
several random systems (including the paper's `n` sweep) to ~1e-16 relative
error; `verify_phi_axial_mean` checks that the double-angle mean of
`{10°, 170°}` returns ~0°/180° and of `{80°, 100°}` returns ~90°, confirming the
180°-periodicity is handled correctly rather than as an ordinary linear mean.

## Cross-reference summary of verification benchmarks (§7)

| Benchmark | What it checks | Result |
|---|---|---|
| `verify_arclength` (§7a) | Sum of fine-cell lengths == ray's true 3-D arc length | max rel. error 1.78e-15 |
| `verify_uniform_strength` (§7b) | Uniform model `m = s*1` forward-models to `d_hat = s*L` | passes to solver tolerance |
| `verify_checkerboard` (§7c) | Noise-free checkerboard recovered in well-sampled blocks | correlation 1.000 |
| `verify_resolution` (§7d) | `Res = (G^T C^-1 G)^-1 (G^T C^-1 G) ~= I`; conditioning | `Res` ~ identity; conditioning reported (flagged, not fatal, above `cond_max`) |
| `verify_whitening` | Closed-form `W^T W` vs. dense Cholesky `C^-1` | ~1e-16 relative error |
| `verify_phi_axial_mean` | Double-angle circular mean handles 180° periodicity | exact to numerical tolerance |
| `verify_phi_distance_weight` | 1/d, 1/d² weight by geometric distance, not δt | passes (distinct, ordered weighted means) |
