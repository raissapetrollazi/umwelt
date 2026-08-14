# Umwelt

Umwelt is an open-source computational laboratory for studying animal behavior through recorded observations and explicit computational models. It keeps recorded data, derived measurements, model state, and synthetic behavior as distinct information categories throughout the workflow.

## Status

Umwelt `0.2.0` is pre-alpha research software with two narrow mouse experiments:

- **v0.1 temporal laboratory:** a complete recorded-to-synthetic sleep/activity workflow using the COMPASS dataset, including replicated temporal model comparison and a retained individual-variation development experiment;
- **v0.2 spatial laboratory:** a complete recorded-to-synthetic open-field movement workflow using the Roche pose dataset, with pinned control split, explicit image-space geometry, pre-declared spatial metrics, a seeded persistent reflecting random-walk baseline, replicated development comparison, provenance, reporting, and artifact hashes.

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

The frozen spatial evaluation compares path length, displacement and turning distributions, boundary-distance distributions, geometric center occupancy, a fixed `4 x 4` occupancy grid, and outside-arena fraction. Recorded and synthetic measurements are computed separately before descriptive comparison. No implicit pass/fail threshold is assigned.

See `docs/v0.2-research-direction.md` for the full protocol and interpretation boundaries.

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
```

Downloaded source data remains under `data/raw/` and is excluded from Git. Roche data uses the manual source workflow documented in `data/README.md`.

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

## Scientific principles

- Keep observation, derivation, inference/model state, and generated behavior distinct.
- Record assumptions, configuration, provenance, code version, and random seeds.
- Prefer inspectable mechanisms and narrow questions before increasing complexity.
- Evaluate fidelity instead of assuming it.
- Preserve negative and null results.
- Keep claims within the evidence produced by data and model.
- Do not invent missing time, physical calibration, biological semantics, or causal interpretation.

## Planned direction

Later milestones may add causal environmental interventions, social context, neural or physiological modalities, latent-state experiments, counterfactual branching, multimodal ethology, multiple species, and additional model families. Those capabilities remain planned unless repository evidence explicitly shows otherwise. See `docs/roadmap.md`.

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
- `docs/architecture.md`
- `docs/scientific-principles.md`
- `docs/data-and-ethics.md`
- `docs/scope.md`
- `docs/roadmap.md`

## License

Umwelt is licensed under the Apache License 2.0. The COMPASS source data has its own CC0 1.0 terms; the Roche open-field source data is CC BY 4.0. Source datasets remain governed by their original terms.
