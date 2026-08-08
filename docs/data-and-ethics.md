# Data and Ethics

Initial Umwelt development should prioritize public or open datasets that were previously collected. The project does not authorize, prescribe, or require new animal experimentation.

## Dataset Use

Dataset licenses and access conditions must be respected. Original provenance, authorship, versioning, and access conditions should be preserved in future dataset integrations and derived artifacts.

Original experimental conditions matter. When recorded data is used, future documentation should preserve relevant context such as recording conditions, preprocessing steps, annotations, exclusions, synchronization assumptions, missing-data handling, and known limitations.

## Information Categories

Recorded, derived, inferred, and synthetic information must remain explicitly distinguishable throughout preprocessing, modeling, training, validation, evaluation, and reporting.

Where relevant, Umwelt should also distinguish annotations, derived measurements or features, inferred variables, model parameters, synthetic trajectories, synthetic behavioral states, and synthetic internal model variables.

A result should preserve enough provenance to determine whether information was observed, annotated, derived, inferred, or generated.

## Traceability

Derived outputs should remain traceable to their source data and transformation steps. Recorded animal data must not silently become synthetic data, inferred state, training data, or evaluation data without documentation of its role and transformations.

Synthetic results must be labeled as synthetic. Generated behavior and synthetic internal variables must not be presented as direct evidence about real animals' subjective experience, true intentions, emotions, consciousness, or complete biological state.

## Reproducibility

Reproducibility is part of the laboratory infrastructure. Future experiments should preserve enough information to reproduce computational runs, including where applicable dataset provenance, configuration, model version, preprocessing choices, random seed, initial conditions, environment configuration, interventions, outputs, and evaluation artifacts.

This principle does not freeze exact storage formats or classes.

## Future Animal Data

Any future use involving newly collected animal data would require appropriate institutional, ethical, and legal processes outside the scope of this software.

Umwelt documentation and software should avoid implying that running the software is sufficient authorization for animal experimentation. The repository should not fabricate or generalize legal requirements beyond the requirements applicable to a specific future context.
