# Data and Ethics

Umwelt prioritizes public or open data that was collected previously. The
software does not authorize, prescribe, or require new animal experimentation.

## Dataset use in v0.1

Version 0.1 integrates two files from the COMPASS Zenodo deposit
[`10.5281/zenodo.160344`](https://doi.org/10.5281/zenodo.160344). The deposit
identifies its data license as CC0 1.0. Umwelt preserves the dataset title,
authors, DOI, related article DOI, license, selected filenames, published sizes,
published MD5 checksums, local paths, adapter name, and known experimental
context in run provenance.

The data is downloaded locally and is not redistributed in this repository.
CC0 permits broad reuse, but provenance and authorship remain scientifically
important even where attribution is not a license condition.

## Original experimental context

The selected weekly files describe 24 male C57BL/6J mice, individually housed
in four groups of six cages, under a 12-hour light / 12-hour dark cycle. Activity
was measured by passive infrared sensors at 10-second intervals. The weekly
sleep labels were derived from extended immobility.

That context constrains interpretation. Umwelt does not generalize the resulting
baseline to other sexes, strains, housing conditions, measurement systems,
protocols, species, or sleep definitions.

## Information categories

The v0.1 workflow distinguishes:

- recorded activity measurements;
- recorded, behaviorally derived sleep labels;
- missing source measurements;
- model parameters fitted from selected recorded subjects;
- synthetic model states;
- synthetic activity emissions;
- metrics derived separately from recorded or synthetic series;
- comparisons derived from those metric sets.

The held-out time grid and missingness mask may structure synthetic generation,
but held-out state and activity values do not become model inputs. Every exported
synthetic row includes the `synthetic` source label.

## Traceability

Derived outputs remain traceable to source files and transformations. A run
records source checksums, resolved configuration, subject roles, model
parameters, random seeds, software version, Git revision when available,
platform, and hashes of completed artifacts.

Recorded data must not silently become synthetic training or evaluation data.
Synthetic output must not be described as observed behavior. A metric computed
from one category must retain that category in its artifact.

## Scientific claims

A close numerical match between recorded and synthetic summaries does not prove
that the model has recovered a biological mechanism. A mismatch does not by
itself invalidate the source data. Claims must remain bounded by the selected
observations, preprocessing supplied by the deposit, explicit model assumptions,
held-out design, and implemented metrics.

Synthetic states and internal variables must not be presented as evidence of a
real animal's subjective experience, true intention, emotion, consciousness, or
complete biological state.

## Future animal data

Any future use of newly collected animal data would require the institutional,
ethical, and legal processes applicable to that specific work, outside the
scope of Umwelt. Running the software is not authorization for animal
experimentation. Project documentation must not fabricate or generalize
regulatory requirements beyond a known future context.
