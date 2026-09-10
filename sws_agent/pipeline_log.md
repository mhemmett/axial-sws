# AXEC2 Agentified SWS Pipeline — Run Log

- **Task**: run one AXEC2 batch (~250 events) through the agentified SWS pipeline (per `sws_agent/RUN_REQUEST.md`)
- **Station**: AXEC2
- **Catalog**: MLdd 2015–2021 (production catalog, Wang 2024) — NOT NonLinLoc
- **Date**: 2026-07-07
- **Scope**: pilot run, ~250 events, using the settled production parameter set (LQT frame,
  PyKonal-FMM S-wave incidence at 35° cut, SNR>2.0, rectilinearity>0.7, DBSCAN + Teanby (2004)
  clustering, Q_w≥0.5 / φ_err<20° / δt_err<0.04s final filter)

---

## Stage 1 — Data Loading

**Agent**: sws-data-loading

### Correction to RUN_REQUEST.md's stated inputs

RUN_REQUEST.md names `data/axial_mldd_2015_2021_axec2_batch_{1,2,3}.mseed` as the input waveform
files. These per-batch combined mseed files **do not exist** for AXEC2 (verified: `ls data/` has
no such files for AXEC2; equivalent files do exist for AXEC1). The calling session had already
identified this and redirected data loading to the real, verified AXEC2 pilot dataset used below.

### Inputs used (exact paths)

- Per-event waveform files: `scripts/raw_axec2_all_batches_data/waveforms/batch{N}_event_<UTC
  timestamp>.mseed` (one file per event, 3-component)
- Companion metadata: `scripts/raw_axec2_all_batches_data/raw_axec2_all_batches_metadata.csv`
  (columns: batch, event_id, waveform_file, station, datetime, s_arrival_time, p_arrival_time,
  latitude, longitude, depth, back_azimuth, magnitude, snr_horizontal, rectilinearity_jurkevics,
  incidence_p_jurkevics, incidence_pykonal_s)
- Reference driver read for call pattern (not executed): `scripts/run_production_axec2_all_batches.py`
  (`build_organized_waveforms()`)
- Reference generator of the metadata CSV, read to establish provenance of the precomputed QC/
  incidence fields: `scripts/build_raw_axec2_all_batches.py`
- Core loader signatures read (not modified): `scripts/splitting_functions.py` —
  `create_extended_catalog(catalog_df, pre_p_time=1.0, post_s_time=1.0)`,
  `organize_waveform_data(traces_dict, extended_catalog)`, `organize_stream_by_events`,
  `calculate_snr_for_organized_waveforms`, `calculate_back_azimuth_for_organized_waveforms`,
  `calculate_rectilinearity_jurkevics_for_organized_waveforms`,
  `calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms`

### Pilot selection rule

Verified per-batch file counts on disk: batch1=210, batch2=218, batch3=225 event files
(matches metadata CSV row counts per batch exactly). Batch 1 alone (210) falls short of ~250, and
batch1+batch2 (428) overshoots by too much for a "small pilot." Chosen selection: **all 210
batch-1 events + the chronologically first 40 batch-2 events** (sorted by `datetime`), for a
clean, reproducible, deterministic pilot of exactly **250 events**, spanning
**2015-01-22T00:11:48.772Z to 2015-01-25T16:37:08.063Z**.

### Load result

All 250 selected events loaded successfully into the organized-waveforms dict structure (keyed by
`event_id`, each entry holding `traces` (obspy Stream, 3-component Z/N/E), `station`, `datetime`,
`s_arrival_time`, `p_arrival_time`, `back_azimuth`, `magnitude`, `latitude`, `longitude`, `depth`,
plus the carried-over QC/incidence fields — see below). **0 load failures**: no `obspy.read`
errors, no events with component count != 3, no unexpected channel naming (all resolved to
Z/N/E). 0 events had NaN `incidence_pykonal_s`.

### Metadata-reuse investigation (SNR / rectilinearity / P- and S-incidence)

**Question**: are `snr_horizontal`, `rectilinearity_jurkevics`, `incidence_p_jurkevics`,
`incidence_pykonal_s` in the metadata CSV legitimately reusable, or must they be recomputed fresh
for a faithful, self-contained pilot?

**Findings**:

1. **Provenance is the settled production pipeline itself.** `scripts/build_raw_axec2_all_batches.py`
   (the script that generated `raw_axec2_all_batches_metadata.csv`) calls exactly the primary
   loader chain named in this agent's own charter — `create_extended_catalog` →
   `organize_stream_by_events` / `organize_waveform_data` — followed by
   `calculate_snr_for_organized_waveforms`, `calculate_back_azimuth_for_organized_waveforms`,
   `calculate_rectilinearity_jurkevics_for_organized_waveforms`,
   `calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms`, and a PyKonal-FMM
   S-wave incidence computed via `pykonal_raytracer.BaillardRayTracer` — with the same QC
   thresholds (SNR≥2.0, rectilinearity≥0.7) and same 5–40 Hz bandpass/detrend/taper preprocessing
   that the production run (`run_production_axec2_all_batches.py`) and RUN_REQUEST.md's own
   parameter set specify. These are not ad hoc or stale values; they are one deterministic pass of
   the identical settled functions.

2. **(a) PyKonal velocity-model availability — checked explicitly.** The default velocity-model
   path hardcoded in `scripts/pykonal_raytracer.py` points outside this repo
   (`/Users/mhemmett/Seismology/axial-splitting-ml/data/AXIAL_MODEL_3P_VELOCITY.S.mod.buf`), but
   `build_raw_axec2_all_batches.py` overrides it with a repo-local copy at
   `data/AXIAL_MODEL_3P_VELOCITY.S.mod.buf`. **Verified present**: `data/AXIAL_MODEL_3P_VELOCITY.S.mod.buf`
   exists (33,563,072 bytes), along with `data/stations_axial.llz` and `data/mldd_catalog_2015_2021.csv`
   (all of build_raw_axec2_all_batches.py's other inputs). The `pykonal` package imports and the
   `BaillardRayTracer` instantiates/precomputes successfully in the `seismo` conda env (confirmed
   by direct import test). **So recomputation of `incidence_pykonal_s` is NOT blocked** — contrary
   to the a-priori concern in this agent's brief, all inputs needed to recompute it are present.
3. **(b) SNR/rectilinearity recompute functions exist and are usable.** `snr()`,
   `calculate_rectilinearity_jurkevics()`, and their `_for_organized_waveforms` batch wrappers are
   present and unmodified in `splitting_functions.py`.

**Recommendation: REUSE the precomputed metadata fields, do not recompute.** Reasoning:
   - Recomputation is *possible* (not blocked), but it is not *necessary* for faithfulness — the
     values were already produced by one deterministic pass of the exact same settled functions,
     same settled thresholds, and same settled preprocessing this pipeline would apply again. A
     second pass on identical inputs (same waveform files, same catalog rows, same velocity model)
     would be expected to reproduce the same numbers, modulo floating-point determinism, at the
     cost of a real-time PyKonal FMM precompute step per station.
   - Recomputing would NOT make the pilot more "self-contained" in any meaningful sense — the
     inputs (catalog CSV, velocity model, station file) it would depend on are the same repo-local
     `data/` files already consumed once to build this metadata; re-deriving from them is not a
     different, independent check, just redundant compute.
   - Reuse keeps this stage fast and reduces the chance of a divergence in this pilot due to
     library/version drift between environments, while still being traceable and swappable later
     (the generating script and its exact function calls are on record above; recomputation can be
     triggered later trivially if a genuine cross-check is wanted, since the blocker check came
     back negative).
   - This is not a case requiring escalation: the ambiguity is resolved with direct evidence, not
     assumed away.

### Organized-waveforms dict handed off

- 250 events, structure matches `build_organized_waveforms()` in
  `scripts/run_production_axec2_all_batches.py`: per event, `traces` (3-comp Stream),
  `station='AXEC2'`, `datetime`, `s_arrival_time`, `p_arrival_time`, `back_azimuth`,
  `snr_horizontal`, `rectilinearity_jurkevics`, `incidence_eigenvalue_jurkevics` (P, reference
  only), `incidence_pykonal_s` (drives QC/rotation per settled convention), `magnitude`,
  `latitude`, `longitude`, `depth`.
- Confirmed ready for pre-processing: all 250 events have 3 well-formed Z/N/E traces, non-null
  arrival times, non-null back-azimuth/location, and a numeric `incidence_pykonal_s` (0 NaNs) so
  the downstream 35° incidence QC cut can be applied without additional imputation.

**Auditor verdict (Stage 1, sws-auditor): PROCEED WITH NOTED CAVEATS — no Critical finding.
- MAJOR: Input metadata is already SNR>2.0 / rectilinearity>0.7 QC'd at build time (build_raw_axec2_all_batches.py L148-156); per-batch counts (210/218/225) are POST-SNR/rect counts. Only the 35° corrected-S incidence cut is a genuinely live in-pipeline QC gate; SNR/rect re-application will be ~100% pass (no-op). Faithfulness claim scoped accordingly; set QC pass-rate denominators to reflect this.
- MAJOR (caveat): the 250-event sample spans only ~2015-01-22 to 2015-01-25 (~3.5 days), entirely ~3 months PRE-eruption. Adequate for pipeline shakedown; cannot support temporal φ/δt anisotropy interpretation across the 2015 eruption — no temporal/eruption inference may be drawn from this pilot.
- MINOR (decisive test): reuse decision sound, but recommend recomputing SNR/rect/incidence_pykonal_s for 5-10 sample events and diffing vs CSV to convert deterministic-provenance argument into empirical evidence (recompute validates provenance; it cannot re-exercise the SNR/rect gate since failing events are already removed).
- OK: corrected-S PyKonal incidence (incidence_pykonal_s, 35° cut) is the field in use; legacy P-Jurkevics reference-only — documented 30°-P→35°-S bug NOT present.
- OK: organized-waveform dict structure matches downstream consumer field-for-field; 0 NaN incidence.
- MINOR (flag to windowing stage): arrival times are offsets from event origin; traces trimmed pre_p=1.0 s / post_s=2.0 s (differs from signature default post_s=1.0). Windowing stage must confirm reference frame and that 2.0 s post-S headroom suffices for MFAST dynamic windows (production LAST_WINDOW_END=2.5).**


---

## Stage 2 — Pre-processing & Dynamic Parameters

**Agent**: sws-preprocessing

### Dominant-period estimator used — confirmed correct

The production splitting chain (`create_splitting_analysis` → `swspy.splitting.create_splitting_object`
→ `perform_sws_analysis`, in `scripts/splitting_functions.py` / `swspy/swspy/splitting/split.py`) computes
T_dom (called `Tmid` in swspy) via **`swspy.splitting.get_dominant_period_baillard`** (scipy.signal.welch,
`nfft=256`, `num_wind=2`), run on the E and N components in a window `[S_pick - 0.1s, S_pick + 0.3s]`,
then `T_mid = mean(dom_period_E, dom_period_N) / 200` (division is by a hardcoded 200, i.e. it assumes
200 Hz sampling — verified true for all 250 AXEC2 traces here, sampling_rate == 200.0 Hz exactly, so this
is not a live bug for this dataset, but it is a fragile hardcode worth flagging for any future non-200 Hz
station).

The **legacy `estimate_dominant_period(trace, method='obspy')`** (PPSD-based) is used only inside
`calculate_dynamic_parameters()` — confirmed **NOT called anywhere in the production splitting call
path** (`create_splitting_analysis` / `perform_splitting_analysis` / `perform_splitting_on_organized_waveforms`
never call `calculate_dynamic_parameters`). This matches the module's own documentation at
`splitting_functions.py` line ~840, which explicitly calls the ObsPy/PPSD path "broken in the current
ObsPy version" and states the S-incidence-angle function deliberately uses
`swspy.splitting.get_dominant_period_baillard` instead "to keep the incidence-angle T_dom consistent with
the T_dom actually driving the splitting measurement's own window sizing." **Confirmed: this run uses the
correct Baillard estimator, not the legacy broken one, and this matches production exactly (not an
approximation).**

### Honesty note: what's actually dynamic vs fixed in production

Only the **window end-times** are T_dom-scaled in production. The **filter band is NOT dynamic** in the
production SWSPy path:
- `calculate_dynamic_parameters()` computes a T_dom-derived `optimal_freq_min`/`optimal_freq_max`, but
  this function is dead code for the production call chain — never invoked by `create_splitting_analysis`,
  `perform_splitting_analysis`, or `perform_splitting_on_organized_waveforms`.
- `create_splitting_analysis` itself has a bandpass-filter line present but **commented out**
  (`#stream_filtered.filter("bandpass", ...)`).
- `swspy/swspy/splitting/split.py` has no bandpass filter call anywhere in the splitting-object /
  `perform_sws_analysis` path (only a disabled/commented candidate frequency mask at line 914).
- The actual bandpass applied to these 250 events is a **fixed, non-dynamic 5–40 Hz** filter applied once
  upstream, at data-build time, in `scripts/build_raw_axec2_all_batches.py` (`tr.filter('bandpass',
  freqmin=5.0, freqmax=40.0)`), the same filter for every event regardless of T_dom.

So: **T_dom drives only the MFAST-style multi-window end-time scaling; it does not drive any filter band
in the actual production measurement** — the "filter band" side of `calculate_dynamic_parameters` is
unused/orphaned code.

### Production dynamic-window parameters used (reproduced exactly)

`FIRST_WINDOW_START=2, LAST_WINDOW_START=1, FIRST_WINDOW_END=1.5, LAST_WINDOW_END=2.5, N_WIN=7,
S_PICK_UNCERTAINTY=0.0395` (from `scripts/run_production_axec2_all_batches.py`), consumed inside
`swspy`'s splitting-object constructor as:
- pre-S-pick window bounds (function of `S_PICK_UNCERTAINTY=Terr`, NOT T_dom): earliest window start =
  `2·Terr` = 0.079 s before S; latest window start = `1·Terr` = 0.0395 s before S. Fixed across all events.
- post-S-pick window bounds (function of **T_dom**): earliest window end = `1.5·T_dom`; latest window end
  = `2.5·T_dom`. This is the genuinely T_dom-scaled part.

### T_dom / derived-window summary stats (250/250 events, computed with the correct Baillard estimator,
matching swspy's exact internal call)

- T_dom (Tmid): min = 0.0504 s, median = 0.0990 s, mean = 0.1013 s, max = 0.1867 s (n=250, 0 failures)
- Derived post-S window end range: earliest-end (1.5·T_dom) spans ~0.076–0.280 s; latest-end (2.5·T_dom,
  the binding constraint) spans ~0.126–0.467 s across the 250 events.
- Filter band: fixed 5–40 Hz for all events (not T_dom-derived — see honesty note above); the
  T_dom-derived `optimal_freq_min`/`optimal_freq_max` from `calculate_dynamic_parameters` was computed
  for completeness but confirmed unused by the production measurement.
- **0/250 events failed T_dom estimation** — all had ≥10 valid samples in the `[S-0.1s, S+0.3s]` window
  on both E and N components; no anomalous T_dom values observed (all within a tight, physically
  reasonable 0.05–0.19 s band, consistent with high-frequency local Axial microseismicity).

### Windowing reference-frame + headroom check (addresses Stage-1 auditor flag)

Stage 1's auditor flagged that `create_extended_catalog(pre_p_time=1.0, post_s_time=2.0)` differs from
the function's own default (`post_s_time=1.0`) and asked this stage to confirm the reference frame and
post-S headroom. **Empirically measured directly from the on-disk trace files (not from the documented
call parameters), for all 250 pilot events**:
- Concrete example — event `1_1` (`batch1_event_2015-01-22T00-11-48.772000Z.mseed`): event origin
  `2015-01-22T00:11:48.772Z`; `p_arrival_time=0.818s` → P at `00:11:49.590Z`; `s_arrival_time=1.648s` →
  S at `00:11:50.420Z`. Trace spans `00:11:48.090Z`–`00:11:52.920Z`. Trace start is exactly **1.5 s before
  P** (not 1.0 s), and trace end is exactly **2.5 s after S** (not 2.0 s).
- This −1.5 s / +2.5 s offset was found to be **exact and identical across all 250 events** (std = 0.0 s
  both ways) — i.e. the actual per-event mseed files on disk were pre-cut with a fixed P−1.5s/S+2.5s
  convention, upstream of and different from the `pre_p_time=1.0`/`post_s_time=2.0` values passed into
  `create_extended_catalog()` in `build_raw_axec2_all_batches.py`. Root cause: `organize_stream_by_events`
  only *matches* pre-existing trace boundaries from the raw per-event files by a time tolerance — it does
  not actually re-trim to `create_extended_catalog`'s computed `starttime`/`endtime`. So the
  `pre_p_sec`/`post_s_sec` values recorded in the extended catalog do not reflect the true on-disk trim;
  this stage's direct trace-timestamp measurement is the authoritative check.
- **S pick lands correctly inside the trace** (confirmed both via the worked example and via the
  aggregate check: `trace_start − p_time = −1.5s` for all 250 events, so `s_pick` always falls well
  inside the trimmed window, never at or past a boundary).
- **Post-S headroom is 2.5 s for every event, comfortably exceeding the maximum required
  `2.5·T_dom = 0.467 s`** (a >5x margin even at the single largest observed T_dom). **0/250 events have
  insufficient post-S headroom for the production `LAST_WINDOW_END=2.5` dynamic window.** This
  supersedes/corrects the Stage-1 auditor's documented-parameter-based concern (which assumed only 2.0 s
  of headroom) — the true on-disk headroom is more generous (2.5 s) than the code comments suggest.

### MFAST literature framing

The production windowing is a genuine MFAST-style multi-window scheme (Teanby et al. 2004 cluster-analysis
windowing; Savage et al. 2010 / MFAST 2.2, Wessel, Savage & Teanby 2017), scaling window end-times (and,
in MFAST's own canonical formulation, filter passband — though not in this repo's current production call
path, per the honesty note above) to each event's dominant period. Important distinction to keep explicit:
MFAST's canonical multi-window end-time span is ~0.5·T to 2.5·T, whereas the "T_dom × [0.5, 2.0]" bound
in the repo README describes the **Baillard single-window validation method** — do not conflate the two.
This pipeline's production multipliers (1.5·T_dom to 2.5·T_dom for window end, independently scaled
pre-S-pick bounds via S-pick uncertainty rather than T_dom) are a project-specific variant within that
MFAST-style family, not a literal reproduction of either canonical scheme. Confidence: high that the
windowing is genuine MFAST-style multi-window scaled to T_dom; medium on the exact multipliers (MFAST
v2.2 manual could not be re-fetched for this run).

### Event counts in/out

- 250 events in (rebuilt organized-waveforms dict, identical selection rule to Stage 1: all 210 batch-1
  events + chronologically-first 40 batch-2 events).
- 250/250 events produced a valid T_dom and valid derived dynamic-window parameters — 0 failures, 0
  anomalies, 0 headroom shortfalls.
- All 250 events handed off to sws-quality-control with `T_dom` (`Tmid`), the derived post-S window
  end-time bounds, and the (fixed, non-T_dom-dependent) 5–40 Hz filter band already baked into the
  on-disk waveform amplitudes from the upstream data-build step.

**Auditor verdict (Stage 2, sws-auditor): PROCEED — no Critical or Major finding.
- OK: correct Baillard/Welch dominant-period estimator (get_dominant_period_baillard, nfft=256, window [S-0.1,S+0.3], T_mid=mean(E,N)/200) is verifiably on the production path (swspy split.py L925-931); legacy estimate_dominant_period is dead code (calculate_dynamic_parameters uncalled). The /200 hardcode is valid only for fs=200 Hz (true for AXEC2).
- MINOR (wording tightening): the 'MFAST-style multi-window' framing must be scoped — ONLY the post-S window END-times scale with T_dom; there is NO per-event frequency-band scaling (bandpass inside create_splitting_analysis is commented out at splitting_functions.py:6059/6151; a fixed 5-40 Hz bandpass is applied once upstream at build_raw_axec2_all_batches.py:101), and the pre-S window bound scales with S-pick uncertainty (Terr), not T_dom. Fixed 5-40 Hz is defensible (T_dom range ~5.3-20 Hz sits inside the band).
- OK: post-S headroom ample — worst-case need 2.5·T_dom (~0.467s) + lag search (~0.30s) ~0.77s vs >=2.0s available; 0/250 shortfall robust under either trim reading (catalog params pre_p=1.0/post_s=2.0 vs measured on-disk P-1.5/S+2.5 are facts about different objects, not a conflict).
- OK: T_dom median ~10 Hz physically plausible for ML~0-2 local events; note T_dom is estimated on already-filtered traces so in-band placement is partly tautological, not independent confirmation. 0/250 T_dom failures expected (Welch always returns an argmax).
- FORWARD FLAG to Stage 3 (QC): create_splitting_analysis/perform_splitting_analysis DEFAULT incidence_field='incidence_eigenvalue_jurkevics' (legacy P, 30° regime). Production overrides to 'incidence_pykonal_s' with 35° cut (run_production L40,105,111). Stage 3 MUST explicitly pass incidence_field='incidence_pykonal_s' and confirm cut==35 — the legacy-P 30° bug reappears if any stage relies on the bare default.**
## Stage 3 — Quality Control

**Agent**: sws-quality-control

### Task and inputs

Received the 250-event AXEC2 pilot set (all 210 batch-1 events + chronologically-first 40
batch-2 events, per Stage 1's selection rule) from sws-preprocessing, with the metadata columns
already computed at build time by `scripts/build_raw_axec2_all_batches.py` and stored in
`scripts/raw_axec2_all_batches_data/raw_axec2_all_batches_metadata.csv`: `snr_horizontal`,
`rectilinearity_jurkevics`, `incidence_p_jurkevics` (legacy P, reference only),
`incidence_pykonal_s` (corrected S, PyKonal-FMM). Rebuilt the exact 250-event pilot subset
directly from this CSV (`batch==1` sorted by `datetime`, 210 rows, concatenated with the first
40 chronologically-sorted `batch==2` rows) — reproduces Stage 1/2's set exactly (250/250).

### QC criteria and thresholds (settled, per CLAUDE.md / RUN_REQUEST.md)

- **SNR (horizontal) > 2.0** — signal/noise RMS ratio averaged over E/N components
  (`compute_snr_for_event_baillard`, called by
  `calculate_snr_for_organized_waveforms`).
- **Rectilinearity (Jurkevics) > 0.7** — P-wave polarization linearity from the 3-component
  covariance-matrix eigenvalue decomposition (Jurkevics 1988), measured over P ± 0.12 s
  (`calculate_rectilinearity_jurkevics_for_organized_waveforms`,
  `p_arrival_variable='p_arrival_time'`, `analysis_window=0.12`).
- **Incidence < 35° from vertical, using the CORRECTED S-wave incidence** —
  `incidence_pykonal_s`, computed via PyKonal-FMM ray tracing
  (`scripts/pykonal_raytracer.py::BaillardRayTracer.incidence_angle_at_station`), NOT the
  legacy P-wave Jurkevics eigenvalue incidence (`incidence_p_jurkevics` /
  `incidence_eigenvalue_jurkevics`). This is the corrected-S/LQT-geometry field per the
  documented repo fix (P-incidence bug: 30° cut on legacy P → 35° cut on corrected S).
- Magnitude filter (M > 0): checked but **not enforced** — see Data Hygiene Note below.

### CRITICAL finding on `apply_quality_control()` — confirms and sharpens the Stage 2 forward flag

Read `scripts/splitting_functions.py::apply_quality_control()` (L3868-3983) directly. It is
**more broken than the Stage 2 flag stated**: it does not merely default to the legacy P field —
it has **no `incidence_field` parameter at all**. Line 3932 hardcodes
`incidence = event_data.get('incidence_eigenvalue_jurkevics', np.nan)` with no way to override
which key is read; `qc_thresholds` only accepts `min_snr`, `min_rectilinearity`,
`max_incidence`. Calling `apply_quality_control()` as documented/named would **always** apply
the legacy-P 30°-regime incidence to the QC gate, regardless of any caller intent — the bare
`apply_quality_control` API cannot drive the corrected-S cut.

Cross-checked against how the actual production pipeline handles this
(`scripts/run_production_axec2_all_batches.py` L39-41, 104-105): production does **not** call
`apply_quality_control()` for the incidence cut at all. It applies the incidence filter
manually and explicitly inline:
```python
INCIDENCE_FIELD = 'incidence_pykonal_s'
INCIDENCE_CUT_DEG = 35.0
...
filtered = {eid: ed for eid, ed in organized_all.items()
            if not np.isnan(ed[INCIDENCE_FIELD]) and ed[INCIDENCE_FIELD] <= INCIDENCE_CUT_DEG}
```
and relies on the upstream CSV already being SNR/rectilinearity-filtered at build time (see next
section). `incidence_field='incidence_pykonal_s'` is passed separately to
`perform_splitting_on_organized_waveforms` — that function *does* accept `incidence_field` (used
only for the LQT rotation inclination, not for QC).

**This stage followed the same pattern as production, not the (broken) `apply_quality_control`
driver**, to guarantee the corrected S-wave field drives the cut: SNR/rectilinearity/incidence
were each evaluated explicitly against `snr_horizontal`, `rectilinearity_jurkevics`, and
`incidence_pykonal_s` in a standalone filter, with `incidence_pykonal_s < 35.0` as the incidence
condition, exactly mirroring `run_production_axec2_all_batches.py`'s logic. **The corrected
S-wave field (`incidence_pykonal_s`), not legacy P (`incidence_p_jurkevics` /
`incidence_eigenvalue_jurkevics`), drove every incidence decision in this stage — confirmed by
code inspection of both the filter used here and of production's own bypass of
`apply_quality_control()` for the same reason.** No code in `splitting_functions.py` was
modified; this is a call-site choice, consistent with the "do not modify
`splitting_functions.py`" constraint. Downstream stages (splitting analysis) must pass
`incidence_field='incidence_pykonal_s'` explicitly to `perform_splitting_on_organized_waveforms`
for the LQT rotation — that parameter does exist and does work, unlike the QC gate.

### Decisive-test spot-check (Stage 1 auditor's requested test)

Recomputed `snr_horizontal`, `rectilinearity_jurkevics`, and `incidence_pykonal_s` fully from
scratch for 8 randomly sampled pilot events (`1_167, 1_9, 1_113, 1_67, 1_128, 1_213, 1_234,
1_216`, `random_state=42`), reading the on-disk waveform files directly
(`scripts/raw_axec2_all_batches_data/waveforms/`) and calling the exact same functions
`build_raw_axec2_all_batches.py` uses:
`calculate_snr_for_organized_waveforms`, `calculate_rectilinearity_jurkevics_for_organized_waveforms`,
and `BaillardRayTracer.incidence_angle_at_station` (PyKonal-FMM, `data/AXIAL_MODEL_3P_VELOCITY.S.mod.buf`,
same station precompute call: `STA_LAT, STA_LON = 45.93967, -129.9738`).

**Result: agreement to floating-point precision on all 8 events, all 3 metrics.**
- Max abs diff, SNR (horizontal): **4.44e-16**
- Max abs diff, rectilinearity (Jurkevics): **1.11e-16**
- Max abs diff, incidence (PyKonal-FMM S): **1.78e-15**

These are floating-point-noise-level differences (consistent with identical deterministic
recomputation, not agreement-by-coincidence). **Verdict: reuse of the precomputed CSV metadata
columns is empirically validated — no discrepancy, no escalation needed.** The spot-check also
incidentally re-confirms `calculate_snr_for_organized_waveforms` internally dispatches to
`compute_snr_for_event_baillard` (not the standalone `snr()` function earlier in the file) — the
same function build-time and this recompute both exercise.

### QC gate applied to all 250 pilot events

Per-criterion attrition (each criterion evaluated independently against all 250 events):

| Criterion | Threshold | Pass | Fail | Pass rate |
|---|---|---|---|---|
| SNR (horizontal) | > 2.0 | 250 | 0 | 100.0% |
| Rectilinearity (Jurkevics) | > 0.7 | 250 | 0 | 100.0% |
| Incidence (`incidence_pykonal_s`, corrected S) | < 35° | 243 | 7 | 97.2% |

**Combined (all 3 criteria, AND): 243/250 pass (97.2%).**

As flagged by the Stage 1 auditor: SNR and rectilinearity are effectively no-ops at this stage
(0 failures/250) because the pilot input was already pre-filtered at these same thresholds
inside `build_raw_axec2_all_batches.py` (`if snr_h < 2.0 or rect < 0.7: continue`, i.e. the CSV
by construction only contains events with `snr_horizontal >= 2.0` and
`rectilinearity_jurkevics >= 0.7`). Observed SNR range in the pilot set: 2.03–10.25 (median
4.31); rectilinearity range: 0.706–0.990 (median 0.958) — both comfortably clear of their
thresholds, consistent with a build-time pre-filter rather than a coincidence. **The genuinely
live, discriminating cut at this stage is the 35° corrected-S incidence cut.**

Incidence angle (`incidence_pykonal_s`) distribution over all 250 pilot events: **min = 9.78°,
median = 15.97°, max = 73.31°**; 0 missing/NaN values. **7 of 250 events (2.8%) exceed 35°** and
are cut:

| event_id | datetime | incidence_pykonal_s (°) |
|---|---|---|
| 1_21 | 2015-01-22T04:51:47.298Z | 47.55 |
| 1_30 | 2015-01-22T08:04:58.864Z | 56.40 |
| 1_165 | 2015-01-24T04:19:59.812Z | 73.31 |
| 1_184 | 2015-01-24T04:45:42.371Z | 60.99 |
| 1_202 | 2015-01-24T06:43:46.629Z | 56.93 |
| 2_250 | 2015-01-24T16:58:03.274Z | 61.89 |
| 2_298 | 2015-01-25T15:20:25.148Z | 60.69 |

No events had missing required fields (S-arrival, P-arrival, incidence) — 0 events dropped for
missing data; all 250 had complete metadata.

### Data hygiene note: magnitude field not usable for M>0 filter

Checked the magnitude filter (default M > 0, part of `qc_thresholds`). The `magnitude` column in
`raw_axec2_all_batches_metadata.csv` is **uniformly 0.0 across all 106,885 rows in the full
metadata CSV** (not just the pilot subset) — it is an unpopulated/placeholder field from the
catalog build, not a real per-event magnitude. Applying `M > 0` literally would incorrectly zero
out all 250 pilot events (and the entire catalog). This matches production's own behavior:
`run_production_axec2_all_batches.py` does not apply any magnitude filter either. **The
magnitude filter was checked and explicitly NOT enforced at this stage** (consistent with
production), pending the magnitude field being properly populated upstream — flagging this for
whoever owns the catalog build, not treating it as a QC failure here.

### QC-passed output

**243/250 events (97.2%)** pass all three settled QC criteria (SNR>2.0, rectilinearity>0.7,
`incidence_pykonal_s`<35°). Written to
`sws_agent/axec2_pilot_qc_passed.csv` (243 data rows + header), containing: `event_id`, `batch`,
`waveform_file`, `station`, `datetime`, `s_arrival_time`, `p_arrival_time`, `latitude`,
`longitude`, `depth`, `back_azimuth`, `magnitude`, `snr_horizontal`, `rectilinearity_jurkevics`,
`incidence_p_jurkevics` (legacy P, kept for reference only — NOT used for the cut),
`incidence_pykonal_s` (corrected S, the field that drove the cut). This is the exact set
handed to sws-splitting-analysis.

### Event counts in/out

- 250 events in (rebuilt pilot subset, identical to Stage 1/2's selection).
- 0 events dropped for missing required fields.
- 0 events failed SNR>2.0 or rectilinearity>0.7 (build-time pre-filter — expected no-op at this
  stage, not evidence the cuts are functioning as a live filter here).
- 7 events failed `incidence_pykonal_s`<35° (the live, discriminating cut).
- **243/250 events (97.2%) QC-passed**, handed to sws-splitting-analysis.

**Auditor verdict (Stage 3, sws-auditor): PROCEED — no Critical finding in this run.
- OK (most important): corrected-S incidence (incidence_pykonal_s, <35°) PROVABLY drove the QC cut — verified by cross-tabulating both incidence columns against pass/cut membership: 6 of 7 cut events have legacy-P incidence <20° (would have PASSED a P-cut) and were cut only because S>35; retained events 1_1 (P=34.6°) and 1_5 (P=69.7°) would FAIL a 30°-P cut yet are present. A legacy-P gate is arithmetically incapable of producing this output. 243/250 pass, 7 cut (all genuinely S>35°, 47.6-73.3°), clean gap between highest pass (34.82°, event 1_233) and lowest cut (47.55°).
- OK: reuse-validation spot-check — float-noise agreement (SNR 4.4e-16, rect 1.1e-16, incidence 1.8e-15) is the correct success criterion for a deterministic recompute; reuse sound. Minor scope gap: the 8-event sample were all within-threshold batch-1 events (no cut/boundary or batch-2 event recomputed); auditor independently cross-checked all 7 cut values vs source CSV (match).
- MINOR (boundary): stage used incidence < 35.0; production uses <= 35.0. Immaterial here (no event at exactly 35.0), but align to production's <= for future batches.
- OK (benign): magnitude uniformly 0.0 (placeholder); not enforced, matches production. Flagged for catalog-build owner.
- MAJOR LATENT DEFECT (flag for humans, NOT triggered here): apply_quality_control() (splitting_functions.py L3932 hardcodes incidence_eigenvalue_jurkevics [legacy P]; L3902 default max_incidence=30.0) has NO incidence_field parameter — calling it by name silently reinstates the 30°-legacy-P bug. Production and this stage BYPASS it (inline incidence_pykonal_s<=35 filter). Also perform_splitting_on_organized_waveforms DEFAULTS incidence_field='incidence_eigenvalue_jurkevics' (L3987) — downstream stages MUST pass incidence_field='incidence_pykonal_s' explicitly. Recommend: add incidence_field param + default max_incidence=35.0, or deprecate/guard the function.**

---

## Stage 4 — Splitting Analysis

**Agent**: sws-splitting-analysis

### Task and inputs

Received the 243 QC-passed events from sws-quality-control (`sws_agent/axec2_pilot_qc_passed.csv`).
Loaded each event's 3-component waveform from `scripts/raw_axec2_all_batches_data/waveforms/` and
built the organized-waveforms dict field-for-field per `build_organized_waveforms()` in
`scripts/run_production_axec2_all_batches.py` (mapping `incidence_p_jurkevics` →
`incidence_eigenvalue_jurkevics` reference key, and carrying `incidence_pykonal_s`). 243/243 events
loaded with no read failures.

### Exact production call used (mirrors run_production_axec2_all_batches.py L108-113)

```python
perform_splitting_on_organized_waveforms(
    organized_243, FIRST_WINDOW_START=2, LAST_WINDOW_START=1,
    FIRST_WINDOW_END=1.5, LAST_WINDOW_END=2.5, N_WIN=7, S_PICK_UNCERTAINTY=0.0395,
    mode='swspy', coord_system='LQT', sws_method='EV_and_XC',
    incidence_field='incidence_pykonal_s', plot_results=False)
```
`cluster_eps` and `cluster_min_samples` were left at the function defaults (0.15 / 15), exactly as
production does (production also passes neither), i.e. the settled SWSPy-fork operating point.

### incidence_field: explicit override confirmed (Stage 2+3 auditor flag addressed)

`perform_splitting_on_organized_waveforms` DEFAULTS `incidence_field='incidence_eigenvalue_jurkevics'`
(the legacy P-Jurkevics field) at `splitting_functions.py:3987` — verified by reading the signature's
default at runtime (`inspect.signature(...).parameters['incidence_field'].default ==
'incidence_eigenvalue_jurkevics'`). This stage passed `incidence_field='incidence_pykonal_s'`
EXPLICITLY, which overrides that default. The per-event run log confirms the corrected-S field
actually drove the geometry, e.g. `Incidence ('incidence_pykonal_s'): 18.37°, rotation inclination
(LQT): 18.37°` — i.e. `incidence_pykonal_s` sets the LQT rotation inclination angle AND keeps the
in-pipeline incidence handling consistent with the QC cut (both use the corrected PyKonal-FMM S-wave
incidence). Had the legacy-P default been relied on, both the LQT rotation and the incidence handling
would have been corrupted with P-wave inclinations. The bare legacy default was NOT used.

### Clustering path actually exercised — confirmed by code

Confirmed against `swspy/swspy/splitting/split.py::_sws_win_clustering` (called at split.py L1657/1659
with `method="dbscan"`):
- Per-window (φ, δt) measurements are transformed into the **circular-safe space**
  `((δt/max δt)·cos2φ, (δt/max δt)·sin2φ)` (L1240) — the doubled-angle transform that avoids the φ
  wraparound at 0°/180°.
- Clustering is **DBSCAN(eps=0.15, min_samples=15)** (L1259) — the code's live defaults, unchanged
  here. The `AgglomerativeClustering`/Ward line (L1256) is commented out; every live call uses DBSCAN.
- Representative cluster selected by **Teanby et al. (2004) Eq. 13-14 representative variance**
  (L1276-1288): per-cluster spread variance (Eq. 13) vs. error-derived data variance (Eq. 14), take
  the max as each cluster's representative variance, pick the minimum-variance cluster, then the
  best-error observation within it.
- This is the settled SWSPy-fork production clustering — NOT Ward, NOT k-means, and NOT the
  `scripts/teanby_clustering.py` Calinski-Harabasz/Duda-Hart path (unused by this pipeline).

### Method literature

- Silver & Chan (1991) eigenvalue minimization (`sws_method='EV_and_XC'` — EV eigenvalue + XC
  cross-correlation; EV_and_XC required for Q_w to be computed).
- MFAST 2.2 / Teanby et al. (2004) multi-window scheme (N_WIN=7; window end-times scaled to per-event
  T_dom, pre-S-pick bounds scaled by S-pick uncertainty).
- Wang (2024) S-pick uncertainty = 0.0395 s (sets the pre-S-pick window placement).
- DBSCAN(eps=0.15, min_samples=15) in circular-safe (δt·cos2φ, δt·sin2φ) space + Teanby (2004)
  Eq. 13-14 representative-cluster selection. Per sws-deep-research: the DBSCAN clustering is a
  documented ALGORITHM SUBSTITUTION (Wuestefeld-style / SWSPy fork) for Teanby (2004)'s original
  cluster-analysis scheme; the circular-safe transform avoids the φ wraparound; Eq. 13-14 pick the
  minimum-within-cluster-variance representative. Confidence HIGH on the substitution rationale; the
  specific eps/min_samples are TUNED operating points for this dataset, not literature constants.
- This modified-SWSPy production method is DISTINCT from the Baillard single-window validation method
  and is not collapsed with it (the two coexist by design; their agreement is itself a validation
  result). Only the SWSPy production method was run at this stage.

### Events in / out

- 243 QC-passed events in.
- **212/243 measured successfully (87.2%)**; **31/243 failed** (all `splitting_errors` in the SWSPy
  measurement — 0 missing-components, 0 missing-back-azimuth). Failed events are retained in the
  output CSV with `success=False` and null φ/δt.

### φ / δt / Q_w summary (212 successful measurements)

- φ (fast axis, wrapped to 0-180°): circular mean **92.6°**, circular spread **44.1°** (raw
  min/median/max 3.5° / 95.1° / 179.5°). Broad spread — expected for a raw, pre-final-filter set
  including low-Q_w measurements.
- δt (delay time): min **0.010 s**, median **0.045 s**, max **0.280 s**.
- Q_w (Wuestefeld 2010, stored in the `quality` column): min **-0.984**, median **0.075**,
  mean **0.217**, max **1.000**. Distribution: Q_w in [0.8,1.0] = 70; [0.5,0.8) = 20; [0.0,0.5) = 73;
  Q_w < 0 (Wuestefeld "null"-leaning) = 49. The downstream final filter (Q_w ≥ 0.5, φ_err < 20°,
  δt_err < 0.04 s) will retain the good subset — that cut is Stage 5's job, not applied here.

### Output

RAW (pre-final-filter) results written to **`sws_agent/axec2_pilot_splitting_raw.csv`** — one row per
input event (243 rows, including all 31 failures flagged `success=False`). Columns: `event_id`,
`datetime`, `phi`, `dt`, `phi_error`, `dt_error`, `quality` (the Q_w column name the Stage-5
final-filter `build_production_rose_plots_axec2_qw05.py` expects, verified — it filters on
`quality`/`success`/`dt`/`phi_error`/`dt_error`), `success`, plus per-event metadata `back_azimuth`,
`incidence_pykonal_s`, `latitude`, `longitude`, `depth`. Runner: `sws_agent/run_stage4_splitting.py`.

Auditor verdict (Stage 4, sws-auditor): PROCEED — no Critical finding.
- OK: incidence_field='incidence_pykonal_s' provably drove the LQT rotation inclination (L6064->L6075->L6098->swspy split.py:85; per-event print value 18.37° equals CSV incidence_pykonal_s for event 2_302 — genuine runtime evidence). Legacy-P default overridden.
- OK: phi = grid-search opt_phi + back_azimuth is the legitimate LQT->geographic rotation (swspy split.py:1336); (phi − back_azimuth) mod 180 is integer for all 212 successes — evidence the grid search + rotation ran, not fabrication.
- OK (standing clustering check): live path is DBSCAN(eps=0.15, min_samples=15) in circular-safe ((δt/maxδt)cos2phi,(δt/maxδt)sin2phi) space (split.py:1240,1259) + Teanby 2004 Eq.13-14 representative-variance selection (L1281-1297); Ward line (L1256) and alternate DBSCAN(0.25) genuinely commented out; not k-means, not teanby_clustering.py. NOTE: a separate multi-layer path (split.py:3297,3317) hardcodes DBSCAN(eps=0.25,min_samples=sqrt(n)) and ignores cluster_eps — NOT exercised by single-layer EV_and_XC.
- OK: 87.2% success (212/243) reasonable; 31 failures benign (DBSCAN found no >=15-window cluster, split.py:1315) — NOT concentrated at the 35° incidence edge (30/31 <26°) or any back-azimuth.
- MAJOR (Stage-5 flag, not a Stage-4 bug): δt tail ~0.25-0.28s on 0.4-0.5 km-deep events (~0.15 s/km apparent — implausible; likely cycle-skipping). NOT railing the 0.30s search bound (max 0.28s). 19 events δt>=0.25s, 6 δt>=0.27s (1_140,2_264,1_42,1_136,1_216,2_302). Several carry HIGH Q_w (0.71-0.94)+small phi_err, so Q_w>=0.5 will RETAIN them and inflate Stage-5 δt statistics/rose weighting. Decisive test: per-window δt histogram or Baillard single-window δt cross-check.
- MINOR: stored CSV phi is in (−90,90] (has negatives); the log's 0-180 φ figures (circ mean 92.6/spread 44.1) are the runner's %180 remap for summary only. Stage 5 must not double-wrap; confirm build_production_rose_plots_axec2_qw05.py applies its own %180.
- MINOR: prior archived AXEC2 production used n_windows=10; pilot uses N_WIN=7 (mirroring run_production_axec2_all_batches.py). φ/δt ranges broadly consistent. Confirm which N_WIN is the current settled production value.

## Stage 4 Addendum — Cycle-skip Diagnostic (auditor decisive test)

**Agent**: sws-splitting-analysis (diagnostic under Stage 4; NOT a re-run of the production measurement — `axec2_pilot_splitting_raw.csv` untouched).

### Decisive test executed

Per the Stage 4 auditor's named decisive test, ran the repo's settled Baillard single-window cross-validation method (`perform_splitting_analysis_baillard`, default grid `s_window=[0.02,0.3]`, `min_lag=0, max_lag=60, Nlags=60, Nangles=90`, adaptive window/maxlag ON) on every suspect event, using the SAME on-disk 3-component waveforms and the same event metadata (S/P picks, back-azimuth) as the swspy production measurement. This is the repo's standing cross-validation method, kept explicitly DISTINCT from the swspy production path (not collapsed): its independent, small-δt disagreement with a large swspy δt is the cycle-skip signature. swspy per-window δt spread was NOT cleanly extractable from the stored `perform_splitting_on_organized_waveforms` result (it exposes only the final clustered φ/δt, not the per-window grid), so — as the auditor allowed — the Baillard single-window cross-check alone is used for the verdict; it is decisive here.

### Suspect set (success=True AND δt ≥ 0.25 s)

**19 suspect events.** "survive" = would pass the settled Stage-5 final filter (quality/Q_w ≥ 0.5 AND φ_err < 20° AND δt_err < 0.04 s), i.e. would enter and bias Stage 5.

| event_id | depth (km) | Q_w | survives filter? | swspy δt (s) | Baillard δt (s) | δt ratio | T_dom (s) | (δt_sw−δt_B)/T_dom | swspy φ (°) | Baillard φ (°) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2_264 | 2.594 | 0.740 | YES | 0.280 | 0.026 | 10.8x | 0.103 | 2.48 | 19.9 | -22.0 |
| 1_140 | 0.986 | 0.715 | YES | 0.280 | 0.024 | 11.5x | 0.095 | 2.69 | 13.7 | -15.0 |
| 1_136 | 0.452 | 0.712 | YES | 0.270 | 0.025 | 10.7x | 0.099 | 2.47 | 12.7 | -15.0 |
| 1_42 | 0.435 | 0.712 | YES | 0.270 | 0.025 | 10.7x | 0.099 | 2.47 | 5.7 | -9.0 |
| 1_216 | 0.485 | 0.939 | YES | 0.270 | 0.025 | 10.8x | 0.092 | 2.67 | 31.8 | -36.0 |
| 2_302 | 0.492 | 0.939 | YES | 0.270 | 0.024 | 11.1x | 0.099 | 2.48 | 13.6 | -15.0 |
| 1_98 | 0.375 | 0.860 | YES | 0.260 | 0.024 | 10.7x | 0.107 | 2.21 | 22.6 | -25.0 |
| 1_116 | 0.477 | 0.958 | YES | 0.260 | 0.026 | 10.1x | 0.093 | 2.52 | 10.8 | -13.0 |
| 1_172 | 1.503 | 0.000 | no | 0.250 | 0.024 | 10.4x | 0.085 | 2.66 | 13.0 | -16.0 |
| 1_44 | 0.443 | 0.819 | YES | 0.250 | 0.023 | 10.7x | 0.089 | 2.54 | 25.3 | -26.0 |
| 1_247 | 1.408 | 0.000 | no | 0.250 | 0.023 | 11.1x | 0.089 | 2.56 | 33.4 | -34.0 |
| 1_59 | 0.185 | 0.744 | YES | 0.250 | 0.023 | 10.7x | 0.085 | 2.67 | 26.0 | -28.0 |
| 1_190 | 0.468 | 0.772 | YES | 0.250 | 0.026 | 9.7x | 0.087 | 2.57 | 10.5 | -14.0 |
| 1_91 | 0.367 | 0.883 | YES | 0.250 | 0.025 | 9.9x | 0.107 | 2.11 | 20.6 | -23.0 |
| 1_166 | 0.167 | 0.717 | no | 0.250 | 0.025 | 10.0x | 0.093 | 2.42 | 9.8 | -14.0 |
| 1_66 | 0.110 | 0.000 | no | 0.250 | 0.022 | 11.1x | 0.082 | 2.76 | 36.2 | -35.0 |
| 1_139 | 1.682 | 0.745 | YES | 0.250 | 0.026 | 9.7x | 0.092 | 2.44 | 14.6 | -16.0 |
| 1_107 | 0.381 | 0.830 | YES | 0.250 | 0.024 | 10.3x | 0.107 | 2.12 | 22.8 | -24.0 |
| 1_36 | 1.316 | -0.259 | no | 0.250 | 0.023 | 10.7x | 0.080 | 2.83 | 29.4 | -33.0 |

### Per-event swspy-vs-Baillard comparison and verdict

- **δt**: On ALL 19 suspects, swspy δt is 0.25–0.28 s while Baillard δt is 0.022–0.026 s — a ~10x disagreement (mean ratio ~10.5x). Baillard's ENTIRE minima set (both ranked solutions per event) stays below ~0.11 s on every suspect; Baillard finds NO minimum anywhere near swspy's 0.25–0.28 s, even though its lag grid extends well past it. The large swspy δt is uncorroborated by the independent method on every single suspect.
- **Integer-multiple (skip) relationship**: (δt_swspy − δt_Baillard)/T_dom clusters at **~2.5** (range 2.1–2.8) for all 19 — i.e. swspy's δt ≈ Baillard's δt plus ~2–3 dominant periods. This is the classic cycle-skip signature (the eigenvalue minimizer locked onto a lag offset by an integer-ish number of dominant periods).
- **φ**: Fast-axis MAGNITUDE is consistent between methods (|swspy φ| vs |Baillard φ| agree to within a few degrees on every event; the sign/offset differs by a fixed frame convention — Baillard reports the raw horizontal grid angle, swspy reports the LQT→geographic value). φ consistency alongside a 10x δt gap is itself textbook cycle-skip: the fast direction is recovered correctly while δt jumps by whole cycles.
- **Per-event verdict**: **all 19 suspects = CYCLE-SKIP** (Baillard disagrees, measures small δt; φ consistent; δt gap ≈ 2.5·T_dom). **0 genuine large-δt events confirmed** (none had Baillard corroborate a large δt).
- Caveat (does not change the verdict): Baillard's small δt is itself suspiciously uniform (~0.025 s ≈ 5 samples across all events); the diagnostic's load-bearing conclusion is not that 0.025 s is the true δt, but that swspy's large δt is NOT reproduced by the independent method on ANY suspect — the large values are unreliable.

### Counts

- Suspect events (δt ≥ 0.25 s, success=True): **19**
- Confirmed cycle-skips (Baillard small δt, φ consistent, δt gap ≈ 2–3·T_dom): **19**
- Genuine large δt (Baillard corroborates): **0**
- Confirmed skips that SURVIVE the settled Q_w≥0.5 / φ_err<20° / δt_err<0.04 s filter (→ would corrupt Stage 5 if not addressed): **14**

  The 14 corrupting survivors: 2_264, 1_140, 1_136, 1_42, 1_216, 2_302, 1_98, 1_116, 1_44, 1_59, 1_190, 1_91, 1_139, 1_107. Each carries high Q_w (0.71–0.96), φ_err ≤ 15°, and δt_err = 0.005 s — the final filter cannot distinguish them from good measurements; they would inflate the Stage-5 δt distribution and rose-diagram weighting. The remaining 5 suspects (1_66, 1_166, 1_172, 1_247, 1_36) are already excluded by the filter (Q_w<0.5 and/or δt_err=0.12 s / φ_err≈50°), so they do not reach Stage 5.

### Problem quantified (no cap value recommended here — human parameter decision for the orchestrator)

Of 19 success=True events with δt ≥ 0.25 s, **all 19 are cycle-skips**, and **14 of them survive the settled final filter** and would bias Stage 5 δt statistics upward. No δt-cap threshold is proposed here; this addendum only quantifies the contamination for orchestrator escalation.

**Summary**: Baillard cross-check confirms all 19 suspect δt≥0.25 s measurements are cycle-skips (δt ~10x / ~2.5·T_dom too large, φ consistent); **14 survive the Q_w≥0.5 filter and will corrupt Stage 5 unless the orchestrator addresses them.**

---

## Parameter Decisions (human-approved 2026-07-07) — Q_w threshold & cycle-skip guard

Two coupled parameter choices were escalated to the human (project owner) after the Stage 4 cycle-skip diagnostic and a Q_w-threshold literature review (sws-deep-research). Both are the human's decisions, recorded with grounding.

Decision 1 — Q_w headline threshold: KEEP Q_w >= 0.5 (settled production value).
Literature grounding (Wüstefeld et al. 2010 Q_w; formula confirmed verbatim in swspy/swspy/splitting/split.py::_calc_Q_w L1348-1374): Q_w is a cross-method (RC/XC vs EV/SC-Silver&Chan) AGREEMENT metric in [-1,1], NOT a per-measurement goodness-of-fit. Q_w->+1 = confident good split; Q_w->-1 = confident null; Q_w≈0 = maximally ambiguous (equidistant between good and null corners), essentially unclassifiable. Community accept threshold for a good split is 0.7-0.8; the project's 0.5 is already on the permissive end. Q_w>0 is below every published good-split threshold surveyed (SplitLab/Wüstefeld continuous convention |Q|>~0.7; Walpole et al. 2014 pairs Q with λ2/λ1; MFAST/Savage et al. 2010 does not use Q_w at all, grading A/B/C/D/N + δt<0.8·δt_max). Lowering to Q_w>0 would admit maximally-ambiguous near-null measurements with unstable φ — the exact quantity the project's ~5° stress-rotation hypothesis rests on. DECISION: retain Q_w>=0.5 as headline; no weighted/side-by-side variant for this pilot.

Decision 2 — cycle-skip guard: APPLY δt < 0.8·δt_max = 0.24 s (MFAST-grounded), removing the confirmed cycle-skip tail, before/alongside the Q_w+error final filter.
Evidence (Stage 4 Addendum diagnostic, Baillard cross-check): all 19 events with δt>=0.25s gave Baillard δt 0.022-0.026s vs swspy 0.25-0.28s (~10x), with (δt_swspy − δt_Baillard)/T_dom ≈ 2.1-2.8 — classic cycle-skip (φ recovered, δt jumped ~2-3 dominant periods). 19/19 confirmed skips, 0 genuine; 14 survive the Q_w>=0.5/err filter (high Q_w 0.71-0.96, φ_err<=15°, δt_err=0.005s). CRITICAL coupling (deep-research, confirmed against the Q_w formula): Q_w is ORTHOGONAL to cycle-skips — an internally-consistent skip earns Q_w≈+1 — so Q_w can never remove this tail; a max-lag/δt guard is the correct instrument (MFAST's δt<0.8·δt_max rule; here δt_max=0.30s → δt<0.24s). DECISION: apply δt<0.24s guard.

---

## Stage 5 — Final Filtering

**Agent**: sws-final-filtering

### Task and inputs

Received the 243-row raw splitting output from sws-splitting-analysis (`sws_agent/axec2_pilot_splitting_raw.csv`, 212 rows with `success=True`). Confirmed this file is the pre-final-filter product; it is left unmodified and referenced here as "pre-filter."

### Filter predicate — verified against `scripts/build_production_rose_plots_axec2_qw05.py::load_combined_results()`

Exact settled production sequence, verified by direct read of the source (`QW_MIN=0.5`, `PHI_ERR_MAX=20.0`, `DT_ERR_MAX=0.04`, all module-level constants):
1. `results['success'] == True`
2. `results['dt'] > 0` (drops null measurements)
3. `results.dropna(subset=['quality', 'phi_error', 'dt_error'])`
4. `(results['quality'] >= 0.5) & (results['phi_error'] < 20.0) & (results['dt_error'] < 0.04)`

Column names match the raw CSV exactly (`quality` = Q_w, `phi_error`, `dt_error`, `success`, `dt`) — no renaming needed. Confirmed operator on Q_w is `>=` (not `>`), `phi_error` and `dt_error` are strict `<`.

**Phi-wrap note (addresses Stage-4 auditor MINOR flag):** `load_combined_results()` applies `results['phi_az'] = results['phi'] % 180.0` itself, AFTER all filtering, purely for rose-plot binning — it does not read a pre-wrapped `phi`. The raw CSV's `phi` column is genuinely in `(−90, 90]` (has negatives) and was NOT pre-wrapped upstream. This stage applied the filter predicate directly to the raw `(−90,90]` `phi` values (the predicate itself never touches `phi`, only `phi_error`) and did **not** double-wrap `phi`. All φ summary statistics below are reported on the raw `(−90,90]` convention, stated explicitly, not the production script's post-filter `%180` convention — Stage 6 (plotting) should apply its own single `%180` wrap when it builds rose diagrams, exactly as `build_production_rose_plots_axec2_qw05.py` does, not inherit a pre-wrapped value from here.

### Human-approved parameter application (this run, additive to the settled production predicate)

Per the human decisions recorded above, this run applies the MFAST-grounded cycle-skip guard `dt < 0.8·dt_max = 0.24 s` **before** the Q_w/error predicate, in addition to (not instead of) the settled Q_w>=0.5 / phi_error<20° / dt_error<0.04s filter. This is a run-specific addition on top of the unchanged production defaults, not a change to the production script's own headline thresholds.

### Full funnel (243-row raw CSV)

| Step | Count | Dropped this step |
|---|---|---|
| 0. Input (raw CSV, all rows) | 243 | — |
| 1. `success == True` | 212 | 31 |
| 2. `dt > 0` | 212 | 0 (all 212 successes already have dt>0) |
| 3. dropna(quality, phi_error, dt_error) | 212 | 0 (no NaNs among the 212 successes) |
| 4. Cycle-skip guard: `dt < 0.24 s` (drop `dt >= 0.24`) | **N1 = 193** | **19** |
| 5. `quality (Q_w) >= 0.5` | **N2 = 75** | 118 |
| 6. `phi_error < 20.0` | **N3 = 46** | 29 |
| 7. `dt_error < 0.04` | **FINAL = 46** | 0 |

Funnel: 243 → 212 (measured) → 193 (after δt guard) → 75 (after Q_w≥0.5) → 46 (after φ_err<20°) → 46 (after δt_err<0.04, final).

**Removed by the cycle-skip guard (19 events, exactly matching the Stage 4 Addendum's confirmed-skip count):**
`1_36, 1_42, 1_44, 1_59, 1_66, 1_91, 1_98, 1_107, 1_116, 1_136, 1_139, 1_140, 1_166, 1_172, 1_190, 1_216, 1_247, 2_264, 2_302`

This is the exact same 19-event set the Stage 4 Addendum identified via Baillard cross-check as confirmed cycle-skips (δt ~10x too large, φ consistent, δt gap ≈2.5·T_dom). All 14 of the "corrupting survivors" the addendum flagged (`2_264, 1_140, 1_136, 1_42, 1_216, 2_302, 1_98, 1_116, 1_44, 1_59, 1_190, 1_91, 1_139, 1_107`) are confirmed removed by the guard before reaching the Q_w/error predicate — the guard did its job.

### Final-pass φ/δt/Q_w summary (46 passing measurements)

Convention: φ reported on the raw `(−90, 90]` convention (NOT `%180`-wrapped) — do not double-wrap downstream.

- **φ**: circular mean (doubled-angle, wrapped back to `(−90,90]`) = **−78.4°**; circular spread (doubled-angle circular std) = **20.6°**. Raw min/median/max: −85.5° / −73.4° / 89.3°.
- **δt**: min **0.010 s**, median **0.030 s**, max **0.110 s** (cycle-skip tail fully removed — max is now well below the pre-guard 0.28 s).
- **Q_w** (`quality`): min **0.584**, median **0.969**, max **1.000** — a much tighter, higher-confidence distribution than the pre-filter set (median Q_w jumped from 0.075 raw-successes to 0.969 final-pass), consistent with Decision 1's rationale that Q_w>=0.5 plus the guard yields genuinely well-constrained splits.
- **Final passing count: 46 / 243 (18.9% of QC-passed events, 21.7% of measured/success events).**

### Outputs

- `sws_agent/axec2_pilot_splitting_raw.csv` — unchanged, referenced as pre-filter (243 rows).
- `sws_agent/axec2_pilot_splitting_final.csv` — all 243 rows with added boolean columns `cycle_skip` (`dt>=0.24 & success`), `pass_qw` (`quality>=0.5`), `pass_phierr` (`phi_error<20`), `pass_dterr` (`dt_error<0.04`), `final_pass` (AND of `success`, `not cycle_skip`, `pass_qw`, `pass_phierr`, `pass_dterr`).
- `sws_agent/axec2_pilot_splitting_final_passed.csv` — the 46 `final_pass=True` rows only; this is the clean set for Stage 6 plotting.

### Thresholds applied (for the record)

- Q_w (quality) >= 0.5 — settled production value, Wüstefeld (2010); Decision 1, unchanged.
- φ_error < 20.0° — settled production error bound.
- δt_error < 0.04 s — settled production error bound.
- δt < 0.8·δt_max = 0.24 s (δt_max = 0.30 s, the swspy lag-search bound) — MFAST-grounded (Savage et al. 2010 / MFAST 2.2 δt<0.8·δt_max rule) cycle-skip guard; Decision 2, human-approved addition specific to this run (not previously part of the settled headline three-threshold set, and flagged here as such — this is a threshold ADDITION, not a change to the three headline production values, which remain exactly as stated).

No headline threshold value (Q_w>=0.5, φ_err<20°, δt_err<0.04s) was altered from production defaults in this run. The only change from a bare replay of `build_production_rose_plots_axec2_qw05.py`'s filter is the additive δt<0.24s cycle-skip guard, applied per explicit human decision and documented above.

Auditor verdict (Stage 5, sws-auditor): PROCEED — no Critical or Major finding.
- OK (decisive axial-statistics check, independently recomputed): φ circular mean −78.4°, spread 20.6° are AXIAL-correct (2φ-doubling: Σcos2φ=−32.63, Σsin2φ=−13.95 → mean 2φ=−156.84° → φ=−78.4°; R=0.771 → spread 20.6°). A naive linear mean would give ~−49.5° (wrong, dragged by the 8 positive-φ values); the near-boundary cases (+89.3°, −85.5°, ~5° apart as axes) correctly cluster. Stage 6 can build its rose on this.
- OK (funnel): 243 → success 212 → δt<0.24 guard 193 (exactly 19 removed, all dt 0.25-0.28, ids match Stage 4 Addendum confirmed skips) → Q_w>=0.5 75 → phi_err<20 46 → dt_err<0.04 46. final_pass==(success & ~cycle_skip & pass_qw & pass_phierr & pass_dterr) holds; count exactly 46.
- OK (predicate): matches build_production_rose_plots_axec2_qw05.py exactly — success==True, dt>0, dropna, quality>=0.5 & phi_error<20.0 & dt_error<0.04 (>= on Q_w, strict < on error bounds); phi%180 applied only after filtering to a new phi_az column; raw (−90,90] phi never overwritten, NOT double-wrapped.
- OK (parameters faithful): Q_w>=0.5 headline unchanged (not lowered to >0); δt<0.24s (0.8·δt_max, δt_max=0.30) cycle-skip guard applied additively, clearly flagged as sole run-specific addition.
- OK (plausibility, sanity not conclusion): φ~102° [0,180] (WNW-ESE/near E-W), spread ~20°, δt median 0.030s over ~1-4 km ⇒ ~3-5% apparent anisotropy — crack-induced range, physically sane. Pre-eruption 3.5-day snapshot: NO temporal/eruption inference licensed.
- MINOR (non-gating): dt_err<0.04 removed 0 (non-binding this pilot); raw min/median/max (−85.5/−73.4/+89.3) is a raw-number summary, correctly caveated as not axial.
- Forward to Stage 6: apply your own single phi%180 wrap and report axial mean by 2φ-doubling; do not inherit/re-wrap the (−90,90] phi.


## Stage 6 — Plotting & Interpretation

**Agent**: sws-plotting-interpretation

Input: `sws_agent/axec2_pilot_splitting_final_passed.csv` (N=46, final-pass measurements from Stage 5). Applied a single own `phi_az = phi % 180.0` wrap (mirroring `build_production_rose_plots_axec2_qw05.py` L58) for all binning/plotting; the raw `(−90,90]` `phi` column was never overwritten or re-wrapped.

### Figures produced

- `sws_agent/axec2_pilot_rose.png` — rose diagram of φ (axial, 0–180° mirrored to 0–360°), built with `build_production_rose_plots_axec2_qw05.py`'s doubled-angle histogram convention (36 bins). Titled with station AXEC2, MLdd catalog, N=46, Q_w>=0.5 + δt<0.24s guard, date span, and the axial mean/spread annotated on the figure.
- `sws_agent/axec2_pilot_polar_density.png` — polar density plot of φ (angle, axial-mirrored 0–360°) vs δt (radius), 2D-histogram density shading (36 angular bins × 14 radial bins) with individual points overlaid.
- `sws_agent/axec2_pilot_temporal.png` — two-panel φ-vs-time and δt-vs-time series, points colored by Q_w (plasma colormap) and sized by Q_w, error bars from phi_error/dt_error, axial mean φ line overlaid on the φ panel. Carries a mandatory red caveat title/subtitle stating this is a pre-eruption snapshot (~3.5 days, 2015-01-22 to 2015-01-25, ~3 months before the April 2015 eruption) with NO eruption crossing and NO temporal-anisotropy inference licensed.

Script used (ad hoc, not committed to `scripts/`, written for this pilot run and following the conventions of `scripts/build_production_rose_plots_axec2_qw05.py`): scratch script executed under `seismo` env; logic mirrors the referenced production script's `_draw_rose` doubled-angle histogram and its `phi_az = phi % 180.0` convention. No changes made to `swspy/`, `splitting_functions.py`, or any file outside `sws_agent/`.

### Axial statistics (2φ-doubling, independently recomputed here)

- Σcos(2φ) mean = −0.7092, Σsin(2φ) mean = −0.3033 → R = 0.7713 (mean resultant length).
- Axial circular mean φ = **−78.4°** (raw (−90,90] convention) ≡ **101.6°** in [0,180] — reproduces the Stage 5 auditor-confirmed value exactly.
- Axial spread (doubled-angle circular std) = **20.6°** — reproduces the auditor-confirmed value exactly.

### δt summary

- min 0.010 s, median 0.030 s, max 0.110 s (N=46; the one 0.110 s point is a single high-Q_w outlier, cycle-skip guard already removed the 0.24-0.30s tail upstream in Stage 5).

### Pattern flagged for human review (tentative, not a conclusion)

- The axial φ orientation (~102° in [0,180], i.e. roughly WNW-ESE / near E-W as a fast-polarization axis) together with δt on the order of ~0.03 s median over a source-receiver path of order 1-4 km gives an apparent shear-wave anisotropy of roughly ~3-5%. This magnitude and a broadly consistent, non-random azimuthal clustering (spread ~20°, not scattered over the full 0-180° range) is **a pattern consistent with, but not established as, crack-induced stress anisotropy** at Axial Seamount — this is only a qualitative flag for the human and for sws-auditor to weigh, not an assertion of that mechanism.
- **Mandatory caveat**: this 46-event set spans only ~3.5 days (2015-01-22 to 2015-01-25), entirely ~3 months *before* the April 2015 eruption. There is no eruption crossing and no pre/syn/post-eruption comparison in this pilot dataset. No statement about eruption-driven φ rotation, δt change through an inflation cycle, or any other temporal/eruption-cycle behavior is licensed by this data or these figures.
- **Pilot-scope caveat**: this entire run (N=46 out of a full multi-year, multi-station AXEC2 catalog) is a shakedown of the agentified 6-stage SWS pipeline, not a science result. A full-catalog re-run (multi-year, multi-station) is a separate, human-gated compute decision and was not performed or requested here.

N=46. Auditor verdict (Stage 6, sws-auditor): PROCEED — no Critical or Major finding.
- OK: rose plot is genuinely axial (symmetric, doubled-angle, 36-bin, N-up/clockwise per build_production_rose_plots_axec2_qw05.py _draw_rose L129-158); dominant petal ~101.6°≡281.6°, no naive single-lobe mis-orientation.
- OK: on-figure stats are axial 2φ values (mean φ=−78.4°≡101.6° [0,180], spread 20.6°), independently reproduced (mean cos2φ=−0.709, sin2φ=−0.303, R=0.771); no figure annotates the naive ~−49° linear mean.
- OK: polar density cluster ~95-105° (mirror ~275-285°), δt~0.02-0.03s, radial max ~0.11s (event 1_95) — matches CSV.
- OK: temporal figure carries the mandatory caveat (pre-eruption snapshot ~3.5 days 2015-01-22 to 25, ~3 months before April 2015 eruption, no eruption crossing, no temporal inference licensed) over a genuine 3.5-day axis (not stretched); Q_w color/size + error bars.
- OK: N=46 consistent across all figures/interpretation; filter columns hold on every row (Q_w min 0.584, φ_err max 19, δt 0.010/0.030/0.110).
- OK: interpretation stays within 'pattern consistent with, flag for human' — no temporal/eruption or stress-rotation claim asserted.
- MINOR (non-gating, RESOLVED in wrap-up): Stage 6 plotting script was ad hoc/unsaved — now saved to sws_agent/axec2_pilot_make_figures.py, regenerable.
- MINOR (non-gating): rose δt-weighting mode — stated below in wrap-up.
Overall: PROCEED — pipeline outputs faithful and audited.
---

## Pipeline Complete — Chain-of-Custody Summary (2026-07-08)

### Event funnel

- **250 loaded** (210 batch-1 + first 40 batch-2)
- → **250 valid T_dom** (dominant-period estimation)
- → **243 QC-passed** (35° corrected-S incidence angle cut; SNR/rectilinearity were no-ops, events were pre-QC'd at catalog build time)
- → **212 measured** / 31 splitting failures (swspy could not return a stable measurement)
- → **193** after δt < 0.24 s cycle-skip guard (19 removed; human-approved guard, Baillard cross-check confirmed 19/19 as genuine skips)
- → **75** with Q_w ≥ 0.5
- → **46** with φ_err < 20°
- → **46 final** (δt_err < 0.04 s checked but non-binding — no additional rows removed at this step)

### Final result

- N=46, station AXEC2, MLdd catalog (2015-01-22 to 2015-01-25 window)
- Axial φ mean = **101.6°** (≡ −78.4° in the raw (−90,90] convention), axial spread = **20.6°** (via 2φ-doubling; R=0.771)
- δt median = **0.030 s** (range 0.010–0.110 s)
- Q_w median = **0.969**

### Per-stage auditor outcome

All six stages returned **PROCEED**, zero Critical findings. The only **Major** finding across the whole run was Stage 4's δt cycle-skip tail, resolved by the human-approved δt < 0.24 s guard after a Baillard single-window cross-check independently confirmed all 19 flagged events as genuine cycle skips (not borderline/ambiguous). All other findings across all stages were Minor/non-gating and either resolved in this wrap-up or carried forward as open items below.

### Outputs (all in `sws_agent/`)

- `pipeline_log.md` — this run log (all 6 stages + audits + this summary)
- `axec2_pilot_qc_passed.csv` — 243 rows, post-QC
- `axec2_pilot_splitting_raw.csv` — 243 rows, pre-filter splitting measurements/failures
- `axec2_pilot_splitting_final.csv` — 243 rows, labelled with all filter-pass columns
- `axec2_pilot_splitting_final_passed.csv` — 46 rows, final_pass=True only
- `axec2_pilot_rose.png` — axial rose diagram
- `axec2_pilot_polar_density.png` — φ-vs-δt polar density plot
- `axec2_pilot_temporal.png` — φ/δt vs time, with mandatory pre-eruption-snapshot caveat
- `axec2_pilot_make_figures.py` — reproducible Stage 6 plotting script (regenerates the three PNGs above)
- `run_stage4_splitting.py` — Stage 4 splitting-measurement driver script

### Open items for human attention

1. **Latent `apply_quality_control()` legacy-P/30° hardcode** in `splitting_functions.py` — not triggered by this run (LQT/PyKonal path used instead), but should be fixed or deprecated so it cannot silently apply the wrong incidence convention in a future run.
2. **This sample is a 3.5-day pre-eruption snapshot** (2015-01-22 to 2015-01-25) — it licenses **no temporal or eruption-cycle result**. A full multi-year, multi-station AXEC2 (and other-station) re-run is a separate, human-gated compute decision, not performed here.
3. **Cycle-skips were ~9% of measurements** (19/212). Recommend promoting the δt < 0.8·δt_max guard from pilot-only to a settled pipeline step, and investigating why swspy cycle-skips on shallow events at this rate.
4. **N_WIN=7** in this pilot (per `run_production_axec2_all_batches.py`) vs archived production `n_windows=10` — confirm which is canonical before treating pilot output as directly comparable to prior production runs.
5. **Incidence-angle boundary**: this run used `<35°` strictly, vs production's `<=35°` — align the two before merging results.
6. **Magnitude column is a uniform-0.0 placeholder** in the source catalog — a catalog-build issue, not a splitting-pipeline issue, but should be fixed before magnitude is used for anything downstream.

### Close

Agentified 6-stage SWS pipeline completed end-to-end on the AXEC2 pilot, every stage independently audited, one human parameter gate (Q_w threshold + cycle-skip guard) resolved; result is a pipeline shakedown, not a science conclusion.
