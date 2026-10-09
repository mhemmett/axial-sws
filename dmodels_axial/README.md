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
run_axial_dmodels                   % two_sphere: Kidiwela two-sphere Mogi inflation
run_axial_dmodels('yang')           % Baillard's single Yang prolate spheroid
run_axial_dmodels('yang_reversed')  % the same spheroid plunging the other way
```

Then, from `scripts/`:

```bash
python3 dmodels_dike_comparison.py --model <model>   # one model against the data
python3 dmodels_inflation_model_comparison.py       # all three side by side
```

## Files

- `axial_dmodels_params.m`: every input, with provenance. Values marked ASSUMED are placeholders.
- `run_axial_dmodels.m`: builds the grids, station predictions and figures.
- `mogi_disp.m`: Mogi point source, written in terms of volume change.
- `yang_disp.m`: Yang et al. (1988) pressurized prolate spheroid, using the dMODELS algorithm
  as ported in uafgeotools/vmod, at the surface only. Checks:
  - matches `vmod` to about 1e-12;
  - tends to Mogi as a/b → 1 (0.3% difference at a/b = 1.01);
  - matches a `cutde` boundary-element model of the same cavity to within a few percent at
    60° plunge.

  The boundary-element model also confirms two features of the Yang solution. The axis
  plunges toward the strike azimuth. And a steep, elongated spheroid expands sideways like a
  dike: its uplift peak sits on the down-plunge side, with a low directly above it.
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

Grids go to the gitignored `Axial_Deformation/remake_2015_2026/<model>/`:

- `def_pre_<year>_rate<r>.xyzuvw`: z = 0 grids at 100 m spacing. `read_disp_file` loads
  them and treats them as period `pre`.

Tables and figures go to the tracked `outputs/<model>/` (`two_sphere`, `yang`,
`yang_reversed`):

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

`outputs/model_comparison/` holds the cross-model table (`station_comparison.csv`,
`summary.csv`) and figure (`model_comparison.png`) from
`scripts/dmodels_inflation_model_comparison.py`.

## Comparison with the observed change

`scripts/dmodels_dike_comparison.py` compares the Grade-3 circular-mean fast direction before
the eruption (Jan–Apr 2015) with equal-inflation period 5 (from 2021-01-01) at each station,
against the modelled pre_2015 → pre_2026 change in σHmax azimuth. Without extension the
inflation source only changes in amplitude, so the modelled change is then the dike alone.

Generated in `outputs/model_comparison/station_tables.md`. Values are circular means ± standard error, with a 2 m dike and no extension. Model cells show the modelled σHmax azimuth, with the misfit (model − observed, wrapped to ±90°) in brackets.

#### Change, present - pre-eruption 2015 (Δφ)

| Station | Observed | two_sphere | yang | yang_reversed |
|---|---|---|---|---|
| AXAS1 | +2.8° ± 0.9° | -2.1° (-4.8) | -3.8° (-6.5) | -0.9° (-3.6) |
| AXAS2 | +16.5° ± 1.3° | -5.1° (-21.6) | -5.5° (-22.0) | -4.9° (-21.4) |
| AXCC1 | +27.4° ± 2.9° | +13.5° (-14.0) | -14.9° (-42.4) | +2.3° (-25.2) |
| AXEC1 | +7.4° ± 0.4° | +16.4° (+9.0) | +22.6° (+15.2) | -41.8° (-49.2) |
| AXEC2 | +2.6° ± 0.4° | +12.9° (+10.4) | +32.4° (+29.8) | +3.3° (+0.7) |
| AXEC3 | +10.0° ± 0.5° | +10.3° (+0.3) | +18.7° (+8.7) | +1.3° (-8.7) |
| **RMS misfit** | | **12.1°** | **24.2°** | **24.5°** |

"Nothing changed" scores 14.1° RMS.

#### Pre-eruption 2015: observed φ vs modeled σHmax (pre_2015)

| Station | Observed | two_sphere | yang | yang_reversed |
|---|---|---|---|---|
| AXAS1 | 165.9° ± 0.8° | 169.0° (+3.1) | 18.9° (+33.0) | 48.3° (+62.4) |
| AXAS2 | 148.4° ± 1.0° | 30.7° (+62.3) | 48.9° (+80.5) | 62.4° (-86.1) |
| AXCC1 | 145.4° ± 2.7° | 20.4° (+55.0) | 109.8° (-35.6) | 74.6° (-70.8) |
| AXEC1 | 161.2° ± 0.3° | 94.4° (-66.7) | 98.1° (-63.0) | 9.6° (+28.4) |
| AXEC2 | 95.7° ± 0.3° | 107.0° (+11.3) | 96.3° (+0.6) | 144.5° (+48.9) |
| AXEC3 | 121.3° ± 0.5° | 119.0° (-2.3) | 119.7° (-1.6) | 166.9° (+45.6) |
| **RMS misfit** | | **43.8°** | **46.2°** | **60.0°** |

#### Present (period 5): observed φ vs modeled σHmax (pre_2026)

| Station | Observed | two_sphere | yang | yang_reversed |
|---|---|---|---|---|
| AXAS1 | 168.7° ± 0.5° | 166.9° (-1.8) | 15.2° (+26.5) | 47.4° (+58.8) |
| AXAS2 | 164.9° ± 0.9° | 25.6° (+40.7) | 43.4° (+58.4) | 57.5° (+72.6) |
| AXCC1 | 172.8° ± 0.9° | 33.9° (+41.1) | 94.9° (-78.0) | 76.9° (+84.1) |
| AXEC1 | 168.6° ± 0.3° | 110.8° (-57.8) | 120.7° (-47.9) | 147.7° (-20.8) |
| AXEC2 | 98.3° ± 0.3° | 119.9° (+21.7) | 128.6° (+30.4) | 147.8° (+49.6) |
| AXEC3 | 131.3° ± 0.3° | 129.2° (-2.0) | 138.4° (+7.1) | 168.2° (+36.9) |
| **RMS misfit** | | **34.5°** | **47.4°** | **57.8°** |

Best-fitting dike opening: two_sphere 1.5 m (11.9°), yang 0 m, yang_reversed 0.1 m.

- **The dike is only a partial explanation, and only with the two-sphere source.** There it
  brings the misfit from 14.1° to 12.1°, with the right sign at 4 of 6 stations. The two
  western stations rotate the wrong way.
- **With Baillard's single Yang spheroid, adding the dike makes things worse.** Its best
  opening is zero. The spheroid's dike-like sideways push dominates near the eastern
  stations, so the same dike rotates them by +19° to +32° and turns AXCC1 the wrong way.
- **The modelled change depends strongly on the inflation source.** The same dike gives
  different rotations because they depend on the background stress it adds to. Both Yang
  orientations fit the absolute fast directions worse than the two-sphere model does.
- **Extension:** at plausible rates (≤ 1e-5 /yr) it changes little, for every source.
- **Period-5 start:** moving it ±3 months changes the observed Δφ by ≤ 0.5°.

## Modelling choices to check

- Uplift is calibrated at AXCC1 (the Central Caldera BOTPT node): 2.4 m for 2015 and 2.6 m for 2026.
- Inflation geometry is kept fixed between the two states:
  - `two_sphere`: the Kidiwela model from `scripts/mogi_stress_model.py`.
  - `yang`: Baillard's spheroid from the `scripts/baillard_simple_model.py` header — centre
    (8.84, 5.38) km, 3.81 km deep, a = 2.2 km, b = 0.38 km, plunging 77° toward 286°.
  - `yang_reversed`: the same spheroid plunging toward 106°. That file's code contradicts its
    own header on the angle convention, so the reverse is kept as a check.
- The 2015 dike uses the trace of Baillard's `dike_syn_1`, 0.1–2.0 km deep, with 2 m opening.
  Only this local segment is modelled; the full north-rift dike is not.
- The 2026 scenario starts from the post-2015 minimum, so the co-eruptive deflation is not
  included. The dike is included as permanent strain.
- Extension is uniaxial, ridge-normal at 110°. The repo's atan2 fits instead imply
  extension at 80°.
- Stresses are the 2-D surface field only, the same simplification Baillard made.
