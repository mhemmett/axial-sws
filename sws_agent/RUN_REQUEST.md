# Task: run one AXEC2 batch (~250 events) through the agentified SWS pipeline

Use the **sws-orchestrator** subagent (`.claude/agents/sws-orchestrator.md`) to drive this — do
not do it inline, and do not substitute a different agent.

## What to run

- Station: **AXEC2**
- Catalog: **MLdd 2015–2021** (production catalog, not NonLinLoc)
- Size: **~250 events** — use the existing per-batch waveform files
  `data/axial_mldd_2015_2021_axec2_batch_1.mseed`, `_batch_2.mseed`, `_batch_3.mseed`
  (verify actual per-batch event counts rather than assuming exactly 100 each)
- This is a small, pre-approved pilot run — the orchestrator does not need to escalate the run
  size itself, only genuinely ambiguous parameter choices.

## Environment / grounding

- Activate `seismo` conda env before running Python; vendored SWSPy loads via
  `sys.path.insert(0, 'swspy')` from the repo root (not pip-installed).
- Catalog/pickle: `data/AXEC2.clean.cat.pickle`, `data/mldd_catalog_2015_2021.csv`; station
  metadata in `data/stations_axial.llz` / `data/AXIAL_stations.xml`.
- Read `scripts/splitting_functions.py` for current call signatures of
  `organize_waveform_data`, `apply_quality_control`, `calculate_dynamic_parameters` /
  `get_dominant_period_baillard`, `create_splitting_analysis`,
  `perform_splitting_analysis` / `perform_splitting_on_organized_waveforms` — don't assume.
- This is an **already-settled production pipeline** — use it as the reference parameter set
  rather than re-deriving from zero: `coord_system='LQT'`, PyKonal-FMM S-wave incidence at a
  35° cut, SNR>2.0, rectilinearity>0.7 (Jurkevics 1988), DBSCAN(eps=0.15, min_samples=15) in
  circular-safe (δt·cos2φ, δt·sin2φ) space with Teanby (2004) Eq.13–14 cluster selection, final
  filter at Q_w≥0.5, φ_err<20°, δt_err<0.04s (see `scripts/build_production_rose_plots_axec2_qw05.py`
  for the exact filter and `scripts/run_production_axec2_all_batches.py` for the production call).
- Don't fabricate data/paths — if something needed is missing (e.g. a PyKonal velocity-model
  file), stop and report the exact blocker.

## Required output (firm requirement)

1. **A narrated, written account produced stage-by-stage** (not just a final summary) of the
   parameter choices at each pipeline stage and what literature informed them (SNR/rectilinearity
   thresholds — Jurkevics 1988; S-wave vs P-wave incidence and the 35° cut; T_dom dynamic
   windowing — MFAST 2.2 / Wessel, Savage & Teanby 2017; DBSCAN eps/min_samples + Teanby 2004
   Eq.13–14 cluster selection; Q_w/φ_err/δt_err final cut — Wüstefeld-style quality framing).
   Ground literature claims via **sws-deep-research**, don't assert from memory.
2. **Write this narration to `sws_agent/pipeline_log.md` as you go** — per stage: stage name,
   agent, parameters used, literature/justification, auditor verdict, event counts in/out.
3. **Final data**: splitting-results CSV(s) (φ, δt, errors, Q_w, per-event metadata) into
   `sws_agent/` — both pre-final-filter and post-filter results, clearly named or with a
   pass/fail column.
4. **Figures**: rose plot, polar density, temporal φ/δt plot for this batch, saved into
   `sws_agent/`.
5. Route every stage through **sws-auditor**; record its verdict (even "no issues found") in
   the log.

## Constraints

- Don't touch `swspy/`, don't refactor `splitting_functions.py`, don't modify any tracked file
  outside `sws_agent/` (README/CLAUDE.md clustering-method corrections are already done — don't
  revisit).
- `data/`, `results/`, `pickles/` are gitignored/local-only — read freely, don't write outputs
  there. Everything new goes in `sws_agent/` (already created).

## Deliverable

Report: event counts at each stage (loaded → QC-passed → measured → final-filter-passed),
where the output CSV(s)/figures landed, a pointer to `sws_agent/pipeline_log.md`, and any
blockers or auditor findings worth human attention.
