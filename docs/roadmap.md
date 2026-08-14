# Umwelt Research Roadmap

Umwelt is intentionally ambitious in scope but conservative in scientific claims. Large scientific goals are pursued through narrow, independently testable milestones.

The historical foundation stage established project identity, scientific principles, architectural direction, licensing, package metadata, and initial documentation. Versions 0.1 and 0.2 implement the first narrow temporal and spatial recorded-to-synthetic development laboratories.

Versions 0.1 and 0.2 remain pre-alpha research software whose models are not scientifically validated. Every later milestone remains planned unless repository evidence explicitly shows otherwise.

## v0.1 - Sleep / Activity Mouse

**Status:** implemented pre-alpha research laboratory. The original baseline,
replicated temporal model comparison, and individual-variation experiment are
implemented. The models are not scientifically validated, and prospective
evaluation remains unresolved.

**Purpose:** establish the first complete recorded-to-synthetic scientific loop using a deliberately simple temporal behavioral problem.

**Core research question:**

> Can a compact, explicit generative model reproduce key temporal properties of recorded mouse sleep/activity dynamics?

Implemented capabilities:

- download, verify, and ingest the open COMPASS mouse dataset;
- preserve dataset provenance and published checksums;
- replay recorded observations reproducibly;
- represent a small number of behavioral or sleep/activity states;
- preserve the compact phase-conditioned Markov baseline;
- compare state-only, phase-only, duration-only, and phase-plus-duration models
  with shared activity emissions;
- generate synthetic state or activity sequences;
- summarize deterministic synthetic replicate distributions without retaining
  every full trajectory;
- reproduce synthetic runs from configuration and random seeds;
- quantitatively compare recorded and synthetic behavior globally and by mouse;
- explicitly report model failures and mismatches;
- expose the workflow through a minimal headless CLI.

Implemented comparison dimensions include state occupancy, empirical bout
distributions, transition frequencies, activity statistics, daily phase
profiles, sleep-state autocorrelation, individual-subject results, and
predictive simulation intervals.

The canonical evidence supports beginning v0.2 planning. This decision reflects
the maturity of the reproducible laboratory, not acceptance of a biological
model. Remaining bout-distribution and long-lag discrepancies, pooled activity
emissions, and prospective evaluation are explicitly deferred and must remain
visible in future work.

v0.1 is not merely a classification milestone. The synthetic animal must generate new behavioral sequences.

The selected weekly COMPASS labels are based on PIR immobility and remain explicitly distinct from the separate EEG scoring files in the deposit. EEG integration is not part of v0.1.

See [v0.1 Research Direction](v0.1-research-direction.md).

## v0.2 - Spatial Mouse

**Status:** implemented pre-alpha recorded-to-synthetic development laboratory.
The model is not scientifically validated, and the two development animals do
not provide confirmatory population evidence.

**Purpose:** introduce an explicit body and headless spatial world, then ask
whether a compact generative movement model can reproduce selected geometric
and kinematic properties of recorded open-field trajectories.

Implemented capabilities:

- finite two-dimensional points, named coordinate frames with units and axis
  orientation, pose keypoints, frame-indexed spatial series, source categories,
  and explicit gaps;
- axis-aligned rectangular arena geometry, containment, boundary distance, and
  deterministic center-region construction;
- representative-position extraction plus gap-aware displacement, normalized
  counterclockwise movement heading, turning, and a unit-bearing path-length
  result, all preserving subject, recording, source, and coordinate context;
- a pinned adapter for the Roche open-field Zenodo deposit, mapping 32 unique
  animals and recordings from metadata to their DeepLabCut CSVs;
- verification of the metadata and archive against their published sizes and
  MD5s, followed by exact extracted-file sizes, frame counts, and SHA-256s from
  a 32-entry manifest derived from the verified archive;
- separation of 13 mouse-body keypoints from the four `tl`, `tr`, `bl`, and `br`
  arena landmarks;
- preservation of treatment and dosage as provenance rather than inferred
  behavioral cause;
- explicit unknown values for sampling rate, pixel-to-centimeter calibration,
  and physical arena dimensions, while retaining the declared Roche image-axis
  convention (`x-right-y-down`);
- machine-readable, leakage-safe eight-control protocol facts with a pinned
  six-animal fitting and two-animal development split derived outcome-blind from
  animal identifiers and the verified metadata checksum;
- an executable `bodycentre` position protocol with no unsupported likelihood
  cutoff; it applies no interpolation or smoothing and does not bridge explicit
  gaps;
- a frozen per-animal metric contract covering pixel path length with exposure,
  adjacent displacement, and absolute turning while keeping quality control
  separate and arena-dependent metrics unregistered;
- a seeded persistent reflecting random-walk baseline fitted only on six control
  animals and evaluated descriptively on two designated development animals;
- generation on the development frame grid with only the recorded availability
  mask shared, never recorded coordinate values;
- reproducible fitting, generation, evaluation, reporting, dataset/software
  provenance, seed, and artifact-hash workflows exposed through `umwelt spatial`.

The canonical spatial run remains development evidence, not model validation.
Later spatial claims require a prospectively declared question and independent
evaluation rather than further tuning on the two development animals.

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
