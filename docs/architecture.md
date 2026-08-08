# Architecture Direction

This document describes planned conceptual boundaries for Umwelt. It does not describe an implemented architecture.

Umwelt is expected to grow from small, reproducible experiments rather than from a broad framework designed in advance. Future implementation should keep recorded observations, model assumptions, simulation state, interventions, and outputs distinct.

## Conceptual Boundaries

Future Umwelt work may include the following concepts:

- recorded data adapters for loading specific open datasets;
- observations representing measured or annotated behavior from real animals;
- environments representing explicit spatial or contextual state where relevant;
- animals or agents representing modeled entities within a simulation;
- behavioral models that generate synthetic behavior from explicit assumptions;
- a simulation engine for advancing model and environment state;
- interventions for controlled computational perturbations;
- experiment configuration for reproducible runs;
- provenance records for inputs, transformations, model settings, and outputs;
- artifacts produced by replay, simulation, and evaluation workflows;
- replay tools for inspecting recorded observations;
- evaluation tools for comparing recorded and synthetic behavior.

These are planned boundaries and vocabulary, not current modules, classes, or APIs.

## Planned Flow

```text
recorded animal data
    -> data adapter
    -> observations
    -> replay / analysis

observations
    -> explicit model assumptions
    -> synthetic animal
    -> explicit environment
    -> simulation
    -> evaluation against recorded observations
```

## Design Constraints

The early architecture should remain headless-first and CLI-first. A graphical interface may be considered later, but it should not be required for running experiments or inspecting core outputs.

Future implementation should favor plain, inspectable artifacts; deterministic seeds where applicable; explicit configuration; and clear provenance. Dependencies should be selected only when they support a defined scientific milestone.

## Avoiding Premature Structure

Umwelt does not yet define dataset formats, model APIs, simulation loops, evaluation metrics, or command-line commands. Those choices should follow from the first selected research question and dataset.
