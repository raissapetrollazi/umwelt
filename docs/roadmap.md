# Umwelt Research Roadmap

Umwelt is intentionally ambitious in scope but conservative in scientific claims. Large scientific goals are pursued through narrow, independently testable milestones.

The historical foundation stage established project identity, scientific principles, architectural direction, licensing, package metadata, and initial documentation. Version 0.1 now implements the first narrow recorded-to-synthetic baseline.

Version 0.1 is implemented as pre-alpha research software. Every later milestone remains planned unless repository evidence explicitly shows otherwise.

## v0.1 - Sleep / Activity Mouse

**Status:** implemented baseline in Umwelt 0.1.0.

**Purpose:** establish the first complete recorded-to-synthetic scientific loop using a deliberately simple temporal behavioral problem.

**Core research question:**

> Can a compact, explicit generative model reproduce key temporal properties of recorded mouse sleep/activity dynamics?

Implemented capabilities:

- download, verify, and ingest the open COMPASS mouse dataset;
- preserve dataset provenance and published checksums;
- replay recorded observations reproducibly;
- represent a small number of behavioral or sleep/activity states;
- fit a compact, interpretable, phase-conditioned Markov model;
- generate synthetic state or activity sequences;
- reproduce synthetic runs from configuration and random seeds;
- quantitatively compare recorded and synthetic behavior;
- explicitly report model failures and mismatches;
- expose the workflow through a minimal headless CLI.

Implemented comparison dimensions include state occupancy, bout-duration summaries, transition frequencies, activity distributions, daily phase profiles, and sleep-state autocorrelation.

v0.1 is not merely a classification milestone. The synthetic animal must generate new behavioral sequences.

The selected weekly COMPASS labels are based on PIR immobility and remain explicitly distinct from the separate EEG scoring files in the deposit. EEG integration is not part of v0.1.

See [v0.1 Research Direction](v0.1-research-direction.md).

## v0.2 - Spatial Mouse

Introduce an explicit body and spatial world.

Planned concepts include:

- `(x, y)` position or equivalent spatial coordinates;
- orientation;
- velocity or speed;
- trajectories;
- arena geometry;
- objects;
- distances;
- spatial relationships;
- potentially another animal when supported by data.

Recorded pose and movement datasets should connect to explicit spatial representations. At this stage Umwelt should become recognizably a computational world rather than only a temporal sequence generator.

## v0.3 - Behavioral World

Give the environment causal structure.

Potential modeled concepts include food, shelter or nest, environmental objects, light, zones, stimuli, resources, and changing environmental conditions.

Synthetic animals should perceive some subset of environmental state and respond according to explicit behavioral models. Researchers manipulate modeled environmental conditions rather than directly puppeteering animals.

This is the first milestone where controlled environmental intervention becomes central.

## v0.4 - Social Animals

Support multiple individuals and explicit social context.

Potential areas include:

- proximity;
- approach;
- avoidance;
- interactions;
- recent encounter history;
- individual differences;
- simple modeled social variables.

This milestone must not claim true social cognition or intention. It should remain distinct from abstract population-game or general artificial-life systems: Umwelt's emphasis is empirical animal behavior and experimentally grounded behavioral models.

## v0.5 - Neural Mouse

Introduce neural or electrophysiological modalities when supported by appropriate open datasets.

Possible modalities include EEG, LFP, spike data, and related electrophysiological or physiological signals.

A major scientific direction is to evaluate whether neural information contributes predictive or explanatory information about behavioral transitions beyond past behavior and environmental history.

This milestone does not mean simulating a biological brain, and electrophysiological signals must not be equated with direct access to subjective experience.

## v0.6 - Latent State Inference

Make observer/model-state separation an experimental capability.

Synthetic organisms may contain explicit hidden or internal computational variables. Observer mode hides those variables. Inference methods may attempt to estimate latent state using only observable behavior, movement, environment, and other permitted measurements.

Because the internal state of the synthetic model is known, Umwelt can provide computational ground truth for evaluating latent-state inference methods.

Synthetic latent state is not equivalent to real animal mental state.

## v0.7 - Counterfactual Laboratory

Support controlled branching from reproducible initial conditions.

Examples include:

- same seed and initial state, different food location;
- same model, different lighting condition;
- same environment, different stimulus;
- isolated versus modeled social condition.

Runs should make interventions explicit and reproducible so counterfactual computational experiments can be compared systematically.

## v0.8 - Multimodal Ethology

Support synchronized combinations of modalities such as movement, pose, behavioral labels, environmental variables, physiological measurements, and neural signals.

Preserve synchronization assumptions, modality provenance, preprocessing provenance, missing-data handling, and distinctions between observed, derived, inferred, and generated information.

Do not assume every dataset or species provides every modality.

## v0.9 - Multiple Species and Model Families

Remove implicit mouse-specific assumptions from the general infrastructure.

Support multiple species, environments, observational modalities, and behavioral model families while keeping species-specific assumptions explicit and shared infrastructure reusable.

This milestone does not promise universal animal modeling.

## v1.0 - Umwelt

The long-term target is a computational laboratory capable of:

- replaying recorded animal observations;
- representing explicit environments;
- constructing and evaluating explicit behavioral models;
- instantiating synthetic organisms;
- generating synthetic behavior;
- performing controlled computational interventions;
- branching counterfactual experiments;
- integrating multiple observational modalities where appropriate;
- quantitatively evaluating correspondence and divergence between generated and recorded behavior.

v1.0 must preserve the project's core principles: reproducibility, headless-first operation, inspectability, explicit assumptions, explicit provenance, conservative scientific claims, and a clear boundary between observation, inference, and simulation.

It must not claim consciousness reconstruction, genuine intention reconstruction, subjective-experience reconstruction, complete biological realism, or replacement of biological experimentation.
