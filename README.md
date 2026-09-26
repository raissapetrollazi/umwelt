# Umwelt

Umwelt is an open-source computational laboratory for studying animal behavior through recorded observations and explicit computational models. It keeps recorded data, derived measurements, model state, and synthetic behavior as distinct information categories throughout the workflow.

## Status

Umwelt `0.4.0` is pre-alpha research software. The repository contains four narrow mouse experiments:

- **v0.1 temporal laboratory:** a complete recorded-to-synthetic sleep/activity workflow using the COMPASS dataset, including replicated temporal model comparison and a retained individual-variation development experiment;
- **v0.2 spatial laboratory:** a complete recorded-to-synthetic open-field movement workflow using the Roche pose dataset, with pinned control split, explicit image-space geometry, pre-declared spatial metrics, a seeded persistent reflecting random-walk baseline, replicated development comparison, provenance, reporting, and artifact hashes;
- **v0.3 behavioral-world laboratory:** a six-fold animal-level development comparison of a boundary-conditioned movement model, followed by paired synthetic arena-size interventions. The retained canonical run does not qualify the candidate for independent evaluation under its pre-declared rule;
- **v0.4 social-pair laboratory:** a frozen comparison of independent and directionally coupled pair movement using CalMS21 Task 1 poses, with a separate development split, once-run held-out test, and optional offline read-only result panel.

The v0.3 negative result remains final. The v0.4 candidate met its pre-declared preference rule on development and held-out pairs, but retains substantial absolute distance error and does not establish a biological social mechanism.

Umwelt is not a general animal simulator, a validated biological model, a veterinary or medical tool, or a finished research platform.

## Scientific question

Animal behavior is observable, while mechanisms capable of producing it are only partially observable. Umwelt makes small modeling assumptions executable so they can be generated from, perturbed, and compared against recorded observations. Agreement supports only the measured correspondence; it does not establish recovery of an underlying biological mechanism.

```text
recorded observations -> recorded measurements ---------+
          |                                              |
          +-> fit explicit model -> synthetic behavior -> comparison
```

## v0.1 temporal experiment

The v0.1 experiment uses the open COMPASS deposit (`10.5281/zenodo.160344`): 10-second PIR activity measurements and behaviorally defined sleep labels for 24 male C57BL/6J mice. The labels are based on immobility and are not EEG sleep-stage labels.

The implemented temporal laboratory compares state-only, phase-only, duration-only, and phase-plus-duration state dynamics while holding the activity-emission model constant. Twenty subjects are used for fitting and subjects 6, 12, 18, and 24 are retained as held-out development subjects.

A separate individual-variation experiment adds two sustained training-derived offsets per synthetic individual. It increased realistic between-subject spread in the retained canonical comparison but worsened several bout-distribution and circadian discrepancies, so it remains a diagnostic result rather than a promoted replacement model.

See `docs/v0.1-research-direction.md` and `runs/v0.1-individual-variation/report.md`.

## v0.2 spatial experiment

The v0.2 experiment uses the Roche/ETH open-field Zenodo deposit (`10.5281/zenodo.8188683`). The adapter validates 32 unique animals/recordings, keeps 13 mouse-body keypoints separate from four arena landmarks, and preserves treatment/dose only as source provenance.

The primary experiment uses exactly the eight `Control` / dose `0` animals. The outcome-blind split is pinned before trajectory results:

- fitting: `16459-67049`, `16459-67053`, `16459-67060`, `16459-67067`, `16459-67071`, `16459-67074`;
- development: `16459-67064`, `16459-67078`.

Recorded trajectory extraction uses `bodycentre`, preserves source likelihood without inventing a cutoff, applies no interpolation or smoothing, and does not bridge explicit gaps. The Roche coordinate frame remains `x-right-y-down` in pixels. The source does not establish sampling rate, pixel-to-centimeter calibration, or physical arena dimensions, so Umwelt does not invent speed or physical units.

The first spatial baseline is `persistent-reflecting-random-walk-v1`. It fits pooled stationary probability, positive-displacement magnitude, turning dispersion, and normalized initial position on the six fitting controls. Synthetic trajectories are seeded and constrained by the recorded image-space arena using an explicit reflection rule.

The frozen spatial evaluation compares exactly three registered dimensions: path length with valid-transition exposure, adjacent-displacement distributions including zero, and absolute-turning distributions. Synthetic evaluation copies the development recording's `bodycentre` availability mask but never its coordinate values. Arena bounds constrain generation; boundary distance, center/periphery, occupancy, and outside-arena metrics remain unregistered. No implicit pass/fail threshold is assigned.

The [canonical 32-replicate development run](runs/v0.2-spatial-baseline/report.md) is retained as reproducible evidence. The baseline systematically generated longer paths for both development animals and retained nonzero displacement- and turning-distribution discrepancies; this is model-development evidence, not validation or a confirmatory result.

See `docs/v0.2-research-direction.md` for the full protocol and interpretation boundaries.

## v0.3 behavioral-world experiment

Version 0.3 adds an explicit boundary band to the Roche image-space arena. A candidate model conditions positive movement magnitude on the zone at the start of each transition. Six former fitting controls form leave-one-animal-out folds; the two animals inspected in v0.2 remain a separate diagnostic and cannot select the candidate. The final candidate is also simulated with paired random seeds in compact, recorded-scale, and expanded arenas. These are model sensitivity analyses with no recorded intervention counterpart.

The [canonical 16-replicate run](runs/v0.3-behavioral-world/report.md) met two of five pre-declared preference criteria. It reduced median boundary-band occupancy error but increased absolute path-length discrepancy for every cross-validation animal, so it is **not eligible for independent evaluation** under this rule. This is a complete negative development experiment, not a validation result. See `docs/v0.3-research-direction.md` for the frozen protocol and interpretation boundaries.

## v0.4 social-pair experiment

CalMS21 Task 1 supplies resident and intruder poses with seven keypoints per animal. Umwelt preserves identities, recording/frame context, and confidence values; derives gap-aware neck-to-neck distance and adjacent signed distance change; and fits two explicit image-space generators. S0 moves the animals independently. S1 adds one fitted directional response by the resident to the intruder's current position. Both use the same recorded frame grid, availability mask, image limits, and paired random seeds.

The [frozen experiment](docs/v0.4-frozen-experiment.md) uses 56 training sequences for fitting, 14 for development, and the source's 19 held-out test sequences. In the [canonical result](docs/v0.4-results.md), S1 reduced equal-pair mean distance W1 from 144.27 to 100.57 pixels on held-out test and improved 13 of 19 pairs. Signed-distance-change W1 barely changed (2.135 to 2.131 pixels), and six pairs had worse distance error. These are model-comparison results in image coordinates, not evidence of recovered intention or biological mechanism. The optional local HTML panel summarizes both phases without embedding raw poses.

The CalMS21 data are subject to separate CC-BY-NC-SA terms reported by its [dataset paper](https://arxiv.org/html/2104.02710v4). The raw archive and generated experiment artifacts stay outside Git. A future commercial use of the data or its derivatives needs a separate rights decision or another dataset; registering the Umwelt software does not change the source terms.

## Quick start

Python 3.12 or newer is required.

```console
python -m pip install -e .

# v0.1
umwelt data download
umwelt data verify
umwelt replay --config examples/v0.1-compass.json --subject 24 --limit 12 --format csv
umwelt run --config examples/v0.1-compass.json
umwelt compare --config examples/v0.1-temporal-lab.json
umwelt individual --config examples/v0.1-temporal-lab.json --output runs/v0.1-individual-variation

# v0.2, after the Roche source files are prepared locally
umwelt spatial --config examples/v0.2-spatial-lab.json

# v0.3, using the same Roche source files
umwelt world --config examples/v0.3-world-lab.json

# v0.4, after obtaining the CalMS21 Task 1 archive as documented in data/README.md
python -m pip install -e '.[calms21]'
umwelt social-audit --archive data/raw/calms21/task1_classic_classification.zip --output runs/v0.4-source-audit/report.json
umwelt social-develop --archive data/raw/calms21/task1_classic_classification.zip --output runs/v0.4-social-development
umwelt social-test --archive data/raw/calms21/task1_classic_classification.zip --development runs/v0.4-social-development --output runs/v0.4-social-heldout-test
umwelt social-view --development runs/v0.4-social-development --test runs/v0.4-social-heldout-test --output runs/v0.4-social-dashboard.html
```

Downloaded source data remains under `data/raw/` and is excluded from Git. Roche and CalMS21 preparation are documented in `data/README.md`. Experiment commands require new output paths and never overwrite existing results.

## Spatial artifacts

A v0.2 spatial run writes compact reproducibility artifacts rather than retaining every synthetic trajectory:

- `resolved-config.json`
- `protocol.json`
- `model.json`
- `seeds.json`
- `replicate-metrics.jsonl`
- `provenance.json`
- `report.md`
- `artifact-manifest.json`

Existing run directories are never overwritten.

The v0.3 run adds `world.json`, `models.json`, `interventions.json`, `model-comparison.json`, and `intervention-comparison.json` to its compact artifact set. Its full artifact list and hashes are in `runs/v0.3-behavioral-world/artifact-manifest.json`.

## Scientific principles

- Keep observation, derivation, inference/model state, and generated behavior distinct.
- Record assumptions, configuration, provenance, code version, and random seeds.
- Prefer inspectable mechanisms and narrow questions before increasing complexity.
- Evaluate fidelity instead of assuming it.
- Preserve negative and null results.
- Keep claims within the evidence produced by data and model.
- Do not invent missing time, physical calibration, biological semantics, or causal interpretation.

## Planned direction

The v0.3 candidate's failed preference rule remains final. A [separate transfer audit](docs/v0.3-follow-up-evaluation.md) is proposed for a metadata-screened open-field cohort, pending source-compatibility checks. The v0.4 social experiment is complete as a narrow comparison; broader front-end experiment control is planned around v0.7. Later milestones may add neural or physiological modalities, latent-state experiments, general counterfactual branching, multimodal ethology, multiple species, and additional model families. See `docs/roadmap.md`.

## Non-goals

- Umwelt does not claim to reconstruct animal consciousness, true intentions, or subjective experience.
- Synthetic internal state is not evidence of a real animal's mental state.
- Umwelt is not a veterinary or medical tool.
- It is not a real-time monitoring platform or a general artificial-life system.
- It is not intended to replace biological experiments.
- Geometric similarity does not establish biological mechanism recovery.

## Documentation

- `docs/project-direction.md`
- `docs/v0.1-research-direction.md`
- `docs/v0.2-research-direction.md`
- `docs/v0.3-research-direction.md`
- `docs/v0.3-performance.md`
- `docs/v0.3-follow-up-evaluation.md`
- `docs/v0.4-research-direction.md`
- `docs/v0.4-source-audit.md`
- `docs/v0.4-frozen-experiment.md`
- `docs/v0.4-results.md`
- `docs/architecture.md`
- `docs/scientific-principles.md`
- `docs/data-and-ethics.md`
- `docs/scope.md`
- `docs/roadmap.md`

## License

Umwelt is licensed under the Apache License 2.0. The COMPASS source data has its own CC0 1.0 terms; the Roche open-field source data is CC BY 4.0; the CalMS21 paper reports CC-BY-NC-SA terms for its data. Source datasets remain governed by their original terms.
