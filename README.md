# Umwelt

Umwelt is an open-source computational laboratory in development for studying animal behavior through recorded observations and explicit computational models.

## Status

Umwelt is currently pre-alpha, foundation-stage software.

This repository contains the architectural and scientific foundation for the project. It does not yet contain a working behavioral simulator, dataset integration, replay system, functional command-line interface, machine learning system, or behavioral model.

## Scientific Direction

Umwelt is intended to make behavioral hypotheses executable. Recorded observations should be replayable and analyzable; explicit behavioral models should eventually generate synthetic behavior that can be compared quantitatively with recorded behavior.

The project is intentionally ambitious in scope but conservative in scientific claims. Large scientific goals are pursued through narrow, independently testable milestones.

The first planned research milestone is **v0.1 — Sleep / Activity Mouse**, which asks whether a compact, explicit generative model can reproduce key temporal properties of recorded mouse sleep/activity dynamics. This is a planned milestone, not an implemented capability.

## Documentation

- [Project Direction](docs/project-direction.md)
- [Research Roadmap](docs/roadmap.md)
- [v0.1 Research Direction](docs/v0.1-research-direction.md)
- [Scientific Principles](docs/scientific-principles.md)
- [Scope](docs/scope.md)
- [Architecture Direction](docs/architecture.md)
- [Data and Ethics](docs/data-and-ethics.md)

## Scientific Boundaries

A synthetic animal is an explicit computational model. Synthetic internal state must not be presented as evidence of a real animal's subjective experience, true intention, emotion, consciousness, or complete biological state.

Umwelt is not intended to replace biological experiments or provide veterinary or medical diagnosis, and it is not a general-purpose artificial-life platform.

## License

Umwelt is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
