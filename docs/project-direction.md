# Project Direction

Umwelt is an open-source computational laboratory in development for studying animal behavior through recorded observations and explicit computational models.

Umwelt is intentionally ambitious in scope but conservative in scientific claims. Large scientific goals are pursued through narrow, independently testable milestones.

## Core Identity

Umwelt is intended to connect recorded animal observations to explicit computational hypotheses that can be inspected, executed, perturbed, and quantitatively evaluated.

Its long-term scientific loop is:

```text
recorded animal observations
    -> replay and analysis
    -> explicit behavioral model
    -> synthetic animal
    -> explicit simulated environment
    -> controlled intervention
    -> quantitative comparison with recorded behavior
```

Umwelt is not merely a behavior classifier, visualization package, generic agent simulator, general-purpose artificial-life platform, or animal-control interface.

## Executable Behavioral Hypotheses

A central idea is that behavioral hypotheses should become executable.

Models should eventually be capable of generating behavior, not only assigning labels to recorded behavior. Generated behavior can then be compared with recorded observations so assumptions, successes, and failures are testable.

A recurring research question is:

> What minimal mechanisms are sufficient to reproduce specific properties of observed animal behavior?

This is an important recurring question, not the only question Umwelt may study.

## Synthetic Animals Are Not Puppets

Researchers should not directly command synthetic animal actions such as "go to food," "turn left," or "sleep."

Instead, researchers manipulate modeled experimental conditions—such as food placement, lighting, stimuli, resource availability, or environmental state—and observe how an explicit behavioral model responds as simulation advances.

This is a scientific interaction principle, not a frozen CLI or API specification.

## Explicit Worlds Without Required Graphics

A scientifically explicit world does not require graphical rendering.

A headless environment may still contain position, orientation, velocity, trajectories, distances, arena geometry, objects, zones, resources, environmental state, and other animals.

Rendering and simulation are separate concerns. Future 2D or 3D visualization may exist, but graphics must not become a requirement for scientific simulation.

## Replay and Simulation

**Replay** represents recorded observations from a real experiment. Its history is fixed. Researchers may inspect, query, advance through, and analyze recorded observations, but they cannot intervene retroactively in what actually happened.

**Simulation** represents behavior produced by an explicit computational model. Researchers may eventually manipulate modeled environmental conditions, perform interventions, branch runs, generate synthetic trajectories, and explore counterfactual conditions.

Recorded and simulated data must never be silently conflated.

## Observer and Model/Debug Views

An **observer view** contains only information plausibly available to an experimental observer or measurement system: for example position, movement, orientation, observable behavioral labels, environmental state, interactions, and recorded physiological or neural measurements when available.

A **model/debug view** may expose variables that belong to a synthetic model itself, including hidden model state, transition variables, modeled drives, candidate action scores, probabilities, and decision variables.

This separation enables a future research pattern in which latent-state inference using observer-accessible data can be evaluated against known synthetic-model internal state.

Synthetic model state is a property of the model. It is not evidence that the corresponding mental state, intention, emotion, drive, or subjective experience exists in a real animal.

## Modeling Philosophy

When scientifically reasonable, Umwelt should prefer models that are explicit, generative, interpretable, inspectable, reproducible, stochastic where appropriate, and quantitatively evaluable against recorded behavior.

This does not permanently prohibit machine learning or deep learning. Model complexity should require scientific justification, and more complex models may be introduced when a scientific question or the empirical failure of simpler models warrants them.

## Provenance and Information Categories

Umwelt should distinguish recorded observations, annotations, derived measurements or features, inferred variables, model parameters, synthetic trajectories, synthetic behavioral states, and synthetic internal model variables.

Results should preserve enough provenance to determine whether information was observed, annotated, derived, inferred, or generated. Terminology should remain consistent across the project.

## Reproducibility as Laboratory Infrastructure

Reproducibility is part of Umwelt's scientific architecture, not merely a reporting feature.

Future computational experiments should preserve enough information to reproduce runs, including where applicable dataset provenance, configuration, model version, preprocessing choices, random seed, initial conditions, environment configuration, interventions, outputs, and evaluation artifacts.

Exact storage formats and classes are intentionally not frozen yet.

## Scientific Ambition and Restraint

Umwelt may be technically ambitious. Scientific claims must remain bounded by evidence, explicit assumptions, and quantitative evaluation.

Negative results are valid results. A model that fails in an informative and reproducible way can still produce a scientifically useful experiment.

## Relationship to the Roadmap

The foundation stage establishes project principles and boundaries. The primary product and research roadmap is the versioned sequence from v0.1 through v1.0.

See the [Umwelt Research Roadmap](roadmap.md) for the complete milestone sequence and [v0.1 Research Direction](v0.1-research-direction.md) for the first planned scientific milestone.

## Permanent Scientific Boundaries

Umwelt must not claim to reconstruct consciousness or subjective experience, discover true intention from behavior alone, equate computational state with biological psychology, simulate a complete biological brain as a scientific claim, replace biological experimentation, or provide veterinary or medical diagnosis.

These are constraints on claims and purpose, not reasons to reduce the project's computational ambition.
