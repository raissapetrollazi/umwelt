# Umwelt

Umwelt is intended to become an open-source computational laboratory for studying animal behavior through recorded data and explicit computational models.

## Status

Umwelt is currently in an early design and pre-alpha stage.

This repository contains the initial architectural and scientific foundation for the project. It does not yet contain a working behavioral simulator, dataset integration, command-line interface, machine learning system, or behavioral model.

## Motivation

Animal behavior is observable, but the mechanisms capable of producing it are only partially observable. Computational models can make hypotheses about those mechanisms explicit enough to simulate, perturb, inspect, and compare against recorded behavior.

Umwelt is planned as a framework for expressing those hypotheses carefully, with conservative claims and clear separation between observation, interpretation, model state, and generated behavior.

## Concept

Recorded data and synthetic behavior should remain conceptually distinct.

```text
real animal
    -> recorded observations
    -> replay / analysis

recorded observations
    -> explicit behavioral model
    -> synthetic animal
    -> simulated environment
    -> controlled intervention
    -> comparison with recorded observations
```

A synthetic animal in Umwelt is an explicit computational model. Its internal state must not be presented as evidence about the true subjective state, cognition, intention, emotion, or consciousness of a real animal.

## Intended Direction

Future work is planned to explore narrow, reproducible workflows for:

- behavioral dataset replay;
- explicit environments;
- spatial trajectories;
- behavioral state models;
- generative simulation;
- controlled computational interventions;
- reproducible experiments;
- comparison between recorded and simulated behavior;
- possible future integration with neural or physiological data.

These are planned directions, not current capabilities.

## Scientific Principles

Umwelt should favor reproducibility, explicit assumptions, provenance, inspectable mechanisms, deterministic seeds where applicable, and modest claims. Real observations, derived interpretations, internal model variables, and synthetic behavior should remain distinguishable throughout the workflow.

## Initial Scope

The first research milestone has not yet been selected.

That milestone should be deliberately narrow, likely involving one species, one open dataset, a small number of behavioral variables or states, one simple environment or recorded context, and one clearly defined scientific question.

## Non-Goals

- Umwelt does not claim to reconstruct animal consciousness.
- Umwelt does not infer true intentions or subjective experiences merely from observed behavior.
- Umwelt is not a veterinary or medical tool.
- Umwelt is not currently a real-time monitoring platform.
- Umwelt is not currently a general-purpose artificial life simulator.
- Umwelt is not intended to replace biological experiments.

## License

Umwelt is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
