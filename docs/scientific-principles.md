# Scientific Principles

Umwelt should support careful computational experiments without overstating what recorded behavior or simulation can show.

## 1. Observation Is Not Interpretation

Recorded observations describe measured or annotated events. Interpretations are claims derived from those observations. Umwelt preserves this distinction in data structures, analysis outputs, and documentation.

## 2. A Model State Is Not an Animal Mental State

A synthetic animal is an explicit computational model. Internal model variables must not be presented as evidence about the true subjective state, cognition, intention, emotion, or consciousness of a real animal.

## 3. Real and Synthetic Data Must Remain Distinguishable

Recorded animal data and model-generated behavior are labeled and stored in ways that prevent accidental conflation. Evaluation compares these categories without erasing their different origins.

## 4. Assumptions Should Be Explicit

Model assumptions, preprocessing choices, excluded variables, and simplifications should be documented near the experiment configuration and resulting artifacts.

## 5. Experiments Should Be Reproducible

Experiments should be defined by inspectable configuration, stable inputs, and repeatable execution procedures. Version 0.1 records a resolved configuration and refuses to overwrite an existing run directory.

## 6. Seeds and Configuration Should Be Recorded When Applicable

When stochastic procedures are used, deterministic seeds should be recorded. Version 0.1 stores both the master seed and each derived synthetic-series seed.

## 7. Provenance Matters

Datasets, transformations, intermediate artifacts, model versions, and generated outputs should remain traceable to their sources. Source and artifact checksums are part of the current run record.

## 8. Negative or Null Results Must Not Be Hidden

Experiments that fail to reproduce a behavioral pattern, produce weak effects, or show no meaningful difference are scientifically relevant and should remain reportable.

## 9. Claims Must Remain Within the Evidence

Claims should be limited to what the recorded data, model assumptions, simulation outputs, and validation procedures support.

## 10. Simulation Fidelity Must Be Evaluated

Similarity between simulated and recorded behavior should be measured rather than assumed. A useful simulation must be evaluated against relevant recorded observations and the limits of that evaluation must be stated.
