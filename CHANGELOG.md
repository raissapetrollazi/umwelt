# Changelog

## Unreleased

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
- Preserve paired stochastic streams, training-only profile provenance,
  between-subject dispersion diagnostics, and compact individual-variation
  artifacts without changing activity emissions.

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
