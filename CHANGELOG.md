# Changelog

## Unreleased

## 0.2.0 - 2026-08-13

- Add an interpretable four-model temporal ladder separating phase and elapsed
  bout-duration dependence.
- Add deterministic replicated model comparison with shared activity emissions.
- Add aggregate and per-subject predictive summaries, empirical bout
  Wasserstein/KS distances, and fixed-lag temporal diagnostics.
- Add compact temporal artifacts, SVG diagnostics, and the `umwelt compare`
  command while preserving the original experiment schema and command.
- Reclassify the original held-out subjects as development evaluation after
  their baseline results informed model development.
- Add a separate `umwelt individual` experiment comparing pooled phase+duration
  dynamics with a minimal training-population individual-variation mechanism.
- Preserve paired state streams, isolated activity randomness, training-only
  profile provenance, between-subject dispersion diagnostics, and compact
  individual-variation artifacts without changing activity emissions.
- Record the canonical individual-variation comparison, including its improved
  population spread and worsened bout-distribution and circadian trade-offs.
- Define the v0.2 spatial research question while explicitly deferring
  unresolved v0.1 validation and model-fidelity questions.
- Add axis-aware coordinate frames, two-dimensional pose observations,
  frame-indexed spatial series, rectangular arena geometry, and gap-aware
  trajectory measurements.
- Add the pinned Roche open-field adapter for 32 metadata records and extracted
  DeepLabCut pose files, preserving source likelihoods and treatment provenance.
- Verify metadata and archive against their published byte sizes and MD5s, then
  validate all 32 extracted pose files against exact frame counts, sizes, and
  SHA-256s derived from the verified archive.
- Keep 13 mouse keypoints separate from the four arena landmarks and keep
  sampling rate, physical calibration, and arena dimensions explicitly unknown
  while declaring Roche image axes as `x-right-y-down`.
- Preserve subject, recording, source category, coordinate unit, and axis
  orientation through trajectory derivation; preserve source coordinates and
  explicit likelihood-rejection reasons; normalize headings counterclockwise
  from `+x`; and return path length with its selection rule, spatial context,
  and unit.
- Document the manual Roche download, extraction layout, published archive and
  metadata checksums, and current spatial experiment boundary.
- Add GitHub Actions unit-test coverage for Python 3.12 and 3.13.
- Freeze an outcome-blind six-control fitting and two-control development split
  for the first v0.2 spatial experiment before inspecting trajectory outcomes.
- Add an executable `bodycentre` position policy that preserves source
  likelihood, assumes no unsupported cutoff, applies no interpolation or
  smoothing, and does not bridge explicit gaps.
- Freeze the first spatial metric contract to pixel path length with exposure,
  adjacent displacement, and absolute turning while keeping quality control
  separate and arena-dependent metrics unregistered.
- Add the seeded persistent reflecting random-walk baseline, synthetic
  availability masking, recorded-versus-synthetic evaluation, compact reports,
  dataset/software provenance, and artifact hashes.

## 0.1.0 - 2026-08-08

- Add pinned, resumable, checksum-verified COMPASS dataset downloads.
- Preserve recorded and synthetic observations as explicit source categories.
- Add fixed recorded-data replay through a headless command-line interface.
- Add a seeded phase-conditioned Markov sleep/activity baseline.
- Add held-out metrics for occupancy, activity, transitions, bouts, daily phase,
  and temporal dependence.
- Add reproducible experiment configuration, provenance, model, seed, synthetic
  observation, comparison, report, and artifact checksum outputs.
- Document the first experiment, limitations, data ethics, architecture, and
  future research direction.
