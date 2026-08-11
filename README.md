# Umwelt

Umwelt is an open-source computational laboratory for studying animal behavior
through recorded observations and explicit computational models. It keeps
recorded data, scientific interpretation, model state, and synthetic behavior
as distinct information categories throughout the workflow.

## Status

Umwelt `0.1.0` is pre-alpha research software. It implements one deliberately
narrow recorded-to-synthetic experiment for mouse sleep/activity dynamics. The
repository also contains the in-progress v0.2 spatial foundation; that
foundation is not yet a complete recorded-to-synthetic spatial experiment.
Umwelt is not a general animal simulator, a validated biological model, or a
finished research platform.

The v0.1 source tree can download and verify a pinned open dataset, replay fixed
recorded observations, fit inspectable temporal models, generate seeded
synthetic sequences, and compare recorded and synthetic temporal statistics
across deterministic replicates. The v0.2 foundation adds explicit spatial
types, arena geometry, gap-aware trajectory measurements, and a validated
adapter for recorded Roche open-field pose data. It has no runtime dependencies
outside the Python 3.12 standard library.

## Scientific question

Animal behavior is observable, but mechanisms capable of producing it are only
partially observable. A computational model makes one set of assumptions
explicit so that it can be executed, perturbed, and tested against recorded
behavior. Agreement supports only the measured correspondence; it does not
establish that the model has recovered the underlying biological mechanism.

```text
real mouse -> recorded observations -> replay and recorded metrics
                         |
                         +-> fit explicit model
                                  |
                                  v
                       synthetic observations -> synthetic metrics
                                  |                       |
                                  +------ comparison -----+
```

## v0.1 experiment

The first experiment uses the open COMPASS deposit
([DOI 10.5281/zenodo.160344](https://doi.org/10.5281/zenodo.160344)). Its weekly
files contain 10-second PIR activity measurements and behaviorally defined
sleep labels for 24 male C57BL/6J mice under a documented 12-hour light / 12-hour
dark cycle. The sleep labels are based on at least 40 seconds of immobility;
they are not EEG labels or claims about subjective experience.

The original baseline is a pooled, phase-conditioned, two-state Markov model.
The expanded temporal laboratory compares state-only, phase-only,
duration-only, and phase-plus-duration state dynamics while holding the
activity-emission model constant. Twenty subjects are used for fitting. Subjects
6, 12, 18, and 24 are held out from fitting but are now described as held-out
development subjects because their original baseline results informed the
expanded model comparison.

The replicated comparison reports aggregate and per-subject occupancy,
activity, transitions, empirical bout-distribution distances, circadian
profiles, and autocorrelation. It exposes predictive simulation intervals and
does not assign an unvalidated pass/fail threshold.

A separate v0.1 development experiment asks whether the pooled model's "average
mouse" limitation can be reduced by a minimal, explicit population mechanism.
It fits two sustained logit offsets for each training mouse: sleep bias and
state-switching rate. These reconstruct state-specific leaving hazards, and one
fixed training-derived profile is resampled for each synthetic individual.
Development mice are not used to calibrate profiles, and activity emissions
remain pooled. In the 32-replicate canonical comparison, this mechanism
increased realistic between-subject spread but systematically worsened bout
distribution and phase-profile discrepancies. It is therefore retained as an
experimental diagnostic, not selected as a replacement for the pooled model.
The [canonical report](runs/v0.1-individual-variation/report.md) preserves the
full result and its limitations.

See [the v0.1 experiment document](docs/v0.1-research-direction.md) for the
design, assumptions, artifact contract, and limitations.

## v0.2 spatial foundation

The in-progress spatial milestone uses the Roche open-field Zenodo deposit
([DOI 10.5281/zenodo.8188683](https://doi.org/10.5281/zenodo.8188683)). Its
metadata describes 32 unique animals and recordings: eight controls at dose 0
and eight yohimbine recordings at each of doses 1, 3, and 6 mg/kg. Treatment is
retained as source provenance, not interpreted as a behavioral cause.

The adapter pins the published size and MD5 for both the metadata and pose
archive. Cataloging verifies the metadata; the archive has a separate verifier
because it may be removed after extraction. The extracted DeepLabCut CSVs are
validated against a 32-file manifest derived from the verified archive. Each
manifest entry pins the extracted byte size, SHA-256, and exact frame count;
those member hashes are derived integrity facts, not checksums published by
Zenodo. Each frame keeps 13 mouse keypoints separate from four arena landmarks
(`tl`, `tr`, `bl`, and `br`), with `(x, y)` coordinates and source likelihoods.

The spatial vocabulary represents coordinate frames with axis orientation,
recorded or synthetic series, rectangular bounds, and representative positions.
Derived trajectory samples retain the source point, acceptance/rejection
status, selected keypoint and likelihood rule, plus a `SpatialContext`
containing subject, recording, source category, and coordinate frame. Path
length retains that rule, context, and coordinate unit, while movement headings
are normalized as counterclockwise radians from `+x` for both Cartesian and
image coordinates.

All current Roche geometry remains in an `x-right-y-down` image frame measured
in pixels. The downloaded source does not establish a sampling rate,
pixel-to-centimeter calibration, or physical arena dimensions, so Umwelt does
not derive speed or invent physical units from it. No spatial generative
baseline, seeded synthetic trajectory workflow, or recorded-versus-synthetic
comparison is implemented yet. Pinned protocol facts now freeze the eight
control animals into an outcome-blind six-animal fitting and two-animal
development split before any trajectory outcomes are used. The executable
position policy selects `bodycentre`, retains source likelihood without an
unsupported cutoff, applies no interpolation or smoothing, and does not bridge
explicit gaps. See
[the v0.2 research direction](docs/v0.2-research-direction.md) and
[local data instructions](data/README.md).

## Quick start

Python 3.12 or newer is required.

```console
python -m pip install -e .
umwelt data download
umwelt data verify
umwelt replay --config examples/v0.1-compass.json --subject 24 --limit 12 --format csv
umwelt run --config examples/v0.1-compass.json
umwelt compare --config examples/v0.1-temporal-lab.json
umwelt individual --config examples/v0.1-temporal-lab.json --output runs/v0.1-individual-variation
```

Downloaded source data is stored under `data/raw/` and excluded from Git. The
commands above currently operate on the v0.1 COMPASS laboratory; Roche data is
downloaded manually and has no spatial run command yet. The original `run`
command retains its trajectory artifact contract. The `compare` command writes
compact replicate metrics, model-family summaries, subject-level comparisons,
SVG diagnostics, provenance, seeds, a report, and an artifact hash manifest.
The `individual` command performs a separate paired comparison of the pooled
phase+duration model and the training-population variation model. Both
replicated workflows intentionally discard full replicate trajectories.
Existing run directories are never overwritten.

## Scientific principles

- Keep observation, interpretation, model state, and generated behavior distinct.
- Record assumptions, configuration, provenance, code version, and random seeds.
- Prefer inspectable mechanisms and narrow questions before increasing complexity.
- Evaluate fidelity instead of assuming it.
- Preserve negative and null results.
- Keep claims within the evidence produced by data and model.

## Planned direction

The v0.2 spatial foundation is under active development. Its generative model
and experiment workflow remain to be built. Later milestones may add
environmental interventions, social context, neural or physiological
modalities, latent-state experiments, counterfactual branching, multiple
species, and additional model families. See the
[research roadmap](docs/roadmap.md).

## Non-goals

- Umwelt does not claim to reconstruct animal consciousness.
- It does not infer true intentions or subjective experiences from behavior alone.
- Synthetic internal state is not evidence of a real animal's mental state.
- Umwelt is not a veterinary or medical tool.
- It is not a real-time monitoring platform or a general artificial-life system.
- It is not intended to replace biological experiments.

## Documentation

- [Project direction](docs/project-direction.md)
- [v0.1 experiment](docs/v0.1-research-direction.md)
- [v0.2 spatial experiment](docs/v0.2-research-direction.md)
- [Architecture](docs/architecture.md)
- [Scientific principles](docs/scientific-principles.md)
- [Data and ethics](docs/data-and-ethics.md)
- [Scope](docs/scope.md)
- [Research roadmap](docs/roadmap.md)

## License

Umwelt is licensed under the Apache License 2.0. See [LICENSE](LICENSE). The
COMPASS source data has its own CC0 1.0 terms; the Roche open-field source data
is CC BY 4.0. Both remain governed by their source terms and attributed through
dataset provenance.
