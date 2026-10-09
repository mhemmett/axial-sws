# dmodels_axial

Self-contained MATLAB rebuild of Christian Baillard's DMODELS deformation grids
(`Axial_Deformation/`), for two pre-eruption states of the caldera:

| Scenario | Inflation | 2015 dike | Tectonic extension |
|---|---|---|---|
| `pre_2015` | uplift since the 2011 eruption | no | accumulated 2011 → 2015 |
| `pre_2026` | uplift since the 2015 eruption | yes | accumulated 2015 → 2026 |

Both are written for every extension rate in `p.extension.rates_per_yr`, because the
regional-to-volcanic strain ratio is the least constrained input and controls the
predicted stress directions.

No DMODELS install is needed; it runs in MATLAB or Octave:

```matlab
cd dmodels_axial
run_axial_dmodels
```

## Files

- `axial_dmodels_params.m`: every input, with provenance. Values marked ASSUMED are placeholders.
- `run_axial_dmodels.m`: builds the grids, station predictions and figures.
- `mogi_disp.m`: Mogi point source, written in terms of volume change.
- `okada85_tensile.m`: Okada (1985) opening dislocation, in the centroid convention of
  Beauducel's `okada85.m`. Checked against `cutde` (Thompson): relative error about 1e-11
  for vertical, dipping and horizontal sources.
- `axial_disp_components.m`: unit-amplitude fields for each source. Scenarios are linear combinations of these.
- `principal_compression.m`: the same 2-D stress and most-compressive-axis convention as
  `scripts/deformation_util.py::compute_sigma1_2d`.
- `ll2xy_axial.m`: the local km frame of `scripts/projection.py::ll2xy` (UTM 9, origin
  130.1°W 45.9°N). It matches pyproj to 0.1 m.
- `write_xyzuvw.m`: the grid layout read by `deformation_util.read_disp_file`.

## Outputs

Grids go to the gitignored `Axial_Deformation/remake_2015_2026/`:

- `def_pre_<year>_rate<r>.xyzuvw`: z = 0 grids at 100 m spacing. `read_disp_file` loads
  them and treats them as period `pre`.

Tables and figures go to the tracked `outputs/two_sphere/`:

- `manifest.csv`: the source amplitudes behind each grid.
- `station_predictions.csv`: azimuth of the most compressive horizontal stress (σHmax)
  and σ1 − σ2 at the six stations, using a 10 m stencil at the exact station position.
  The Python nearest-node method differs from it by ≤ 3° (largest at AXCC1, which sits
  next to the shallow source).
- `station_components.csv`: the same for inflation, dike and extension separately, plus
  the strain per unit source, so any mix of sources can be rebuilt by superposition.
- `maps_<scenario>.png`, `station_azimuth_vs_extension.png`.
- `dmodels_dike_comparison.csv` / `.png`: from `scripts/dmodels_dike_comparison.py`
  (see below).

## Comparison with the observed change

`scripts/dmodels_dike_comparison.py` compares the Grade-3 circular-mean fast direction before
the eruption (Jan–Apr 2015) with equal-inflation period 5 (from 2021-01-01) at each station,
against the modelled pre_2015 → pre_2026 change in σHmax azimuth. Without extension the
inflation sources only change in amplitude, so the modelled change is then the dike alone.

| Station | Observed Δφ | Modelled, 2 m dike, no extension |
|---|---|---|
| AXAS1 | +2.8 ± 0.9° | −2.1° |
| AXAS2 | +16.5 ± 1.3° | −5.1° |
| AXCC1 | +27.4 ± 2.9° | +13.5° |
| AXEC1 | +7.4 ± 0.4° | +16.4° |
| AXEC2 | +2.6 ± 0.4° | +12.9° |
| AXEC3 | +10.0 ± 0.5° | +10.3° |

- Fit: RMS misfit 12.1°, against 14.1° for "nothing changed". The best-fitting opening is
  1.5 m (11.9°).
- Direction: right at 4 of 6 stations, wrong at the two western ones.
- Size: about right at AXEC3, too large at AXEC1 and AXEC2, too small at AXCC1.
- Extension: at plausible rates (≤ 1e-5 /yr) it changes little. At ≥ 3e-5 /yr it makes the fit worse.
- Period-5 start: moving it ±3 months changes the observed Δφ by ≤ 0.5°.

## Modelling choices to check

- Uplift is calibrated at AXCC1 (the Central Caldera BOTPT node): 2.4 m for 2015 and 2.6 m for 2026.
- The inflation geometry is the Kidiwela two-sphere model from `scripts/mogi_stress_model.py`,
  kept fixed in both scenarios. The Yang prolate spheroid that Baillard used is not implemented.
- The 2015 dike uses the trace of Baillard's `dike_syn_1`, 0.1–2.0 km deep, with 2 m opening.
  Only this local segment is modelled; the full north-rift dike is not.
- The 2026 scenario starts from the post-2015 minimum, so the co-eruptive deflation is not
  included. The dike is included as permanent strain.
- Extension is uniaxial, ridge-normal at 110°. The repo's atan2 fits instead imply
  extension at 80°.
- Stresses are the 2-D surface field only, the same simplification Baillard made.
