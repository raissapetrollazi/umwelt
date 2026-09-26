# Experiment report: v0.3-roche-behavioral-world

## Interpretation boundary

This is a model-development comparison and a synthetic arena-size sensitivity analysis. Neither the model preference rule nor a synthetic intervention response validates a biological mechanism or predicts how a real animal would respond to an arena-size change.

## Protocol

- Dataset: pinned Roche open-field Control dose 0 recordings
- Cross-validation: six leave-one-animal-out folds, each fitted on five controls
- Legacy diagnostic: two previously inspected controls, excluded from preference
- Master seed: 1730; replicates per animal/model: 16
- Position: source `bodycentre`, no invented likelihood cutoff or smoothing
- Units: image pixels and radians; no physical speed or arena dimension claims
- Recorded positions outside landmark-derived bounds remain in spatial metrics and quality control; starts outside have no zone assignment

## Recorded-scale model preference

M1 eligible for later independent evaluation: **no**. This is a pre-declared development rule, not a validation threshold.

| Criterion | Result |
| --- | --- |
| `lower-median-absolute-relative-path-discrepancy` | not met |
| `lower-absolute-relative-path-discrepancy-in-at-least-four-animals` | not met |
| `no-higher-median-adjacent-displacement-distance` | met |
| `no-higher-median-absolute-turning-distance` | not met |
| `lower-median-boundary-band-occupancy-error` | met |

Animals with lower absolute relative path discrepancy: 0 of 6.

| Animal | M0 absolute relative path error | M1 absolute relative path error |
| --- | ---: | ---: |
| `16459-67049` | 0.297718 | 0.423644 |
| `16459-67053` | 0.0807757 | 0.166757 |
| `16459-67060` | 0.0215824 | 0.106545 |
| `16459-67067` | 0.19621 | 0.288013 |
| `16459-67071` | 0.698674 | 0.822326 |
| `16459-67074` | 0.194098 | 0.302418 |

The machine-readable model comparison reports all registered metric distributions and 5th/50th/95th replicate percentiles, including the separate legacy diagnostic.

## Paired arena-size interventions

The final M1 model was fitted on the six development-pool animals. Their arenas and observation masks supplied templates for paired compact, recorded-scale, and expanded synthetic conditions.

| Condition vs recorded-scale simulation | Median animal path change (px) | Median animal boundary occupancy change | Median animal reflection change |
| --- | ---: | ---: | ---: |
| `compact` | -1737.98 | +0.00448453 | +229.25 |
| `expanded` | +1054.72 | -0.00249354 | -114 |

The intervention artifact also retains paired changes in valid zone transitions and positive-displacement summaries. Reflection counts are model/debug values from the full generated path before observation masking. No recorded arena-size intervention exists here.

## Reproducibility

Resolved configuration, frozen protocol, source checksums, fitted fold and final models, paired seeds, compact replicate metrics, software provenance, and artifact hashes accompany this report. Full synthetic trajectories were discarded after measurement.
