# Experiment report: v0.1-compass-temporal-lab-individual-variation

## Interpretation boundary

This experiment asks whether a minimal training-population variation mechanism can reproduce stable differences among synthetic individuals. The fitted offsets are computational parameters, not evidence of personality, physiology, intention, subjective state, or a biological random-effects mechanism.

## Evaluation protocol

- Source: COMPASS weekly PIR activity and behaviorally defined sleep data
- Dataset DOI: `10.5281/zenodo.160344`
- Observation interval: 10 seconds
- Sleep labels: behaviorally defined from immobility; not EEG-defined sleep
- Training subjects: 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23
- Held-out development subjects: 6, 12, 18, 24
- Exposure caveat: these development subjects were inspected in earlier v0.1 experiments, so this remains model-development evidence rather than a pristine confirmatory test.
- Development-subject behavior is not used to fit or select individual profiles.

## Scientific question

Is training-population variation in a small set of state-dynamics parameters sufficient to reproduce the between-mouse heterogeneity missed by the pooled phase+duration model?

## Population mechanism

The control is the existing pooled `phase-duration-hazard-v1` model. The population variant keeps its phase conditioning, duration bins, smoothing, and shared activity emissions unchanged. Each synthetic individual instead samples one fixed profile from the training population.

Each training profile contains two sustained smoothed logit offsets:

- sleep bias, which increases wake-to-sleep hazard and decreases sleep-to-wake hazard when positive;
- switching rate, which shifts both leaving hazards in the same direction.

Profiles are sampled uniformly with replacement. Their effects remain active throughout pooled phase+duration dynamics; the model does not memorize a training mouse's trajectory and does not calibrate to a development mouse.

## Reproducibility

- Git revision: `6e2e97d57c7b0388806565f5e010cbfa3c80f217`
- Configuration SHA-256: `c39e4379be241a99ed1b512286a5f172938719dd96c9af2fe127aa730343c016`
- Master seed: 1729
- Replicates per model: 32
- Pooled and population variants use paired state random streams for each subject/replicate.
- Activity uses a separate matched seed; draw consumption may diverge after state trajectories diverge.
- Population-profile selection uses a separate deterministic SHA-256-derived seed.
- Source checksums are preserved in `provenance.json`.
- Artifact checksums are preserved in `artifact-manifest.json`.
- Full replicate trajectories are intentionally discarded after compact evaluation.

Synthetic values below are medians followed by the 5th-95th predictive simulation interval.

## Aggregate fidelity

| Metric | Recorded | Phase + duration (pooled) | Phase + duration + population variation |
| --- | ---: | ---: | ---: |
| Sleep fraction | 0.5901 | 0.6105 [0.5991, 0.6205] | 0.6161 [0.5773, 0.6429] |
| Mean activity | 15.6262 | 14.7215 [14.3922, 15.2054] | 14.5323 [13.6226, 15.8369] |
| Nonzero activity fraction | 0.3403 | 0.3198 [0.3124, 0.3303] | 0.3159 [0.2946, 0.3456] |
| State changes per hour | 9.7286 | 9.5698 [9.3016, 9.8665] | 9.5899 [7.3627, 11.8507] |
| Wake bout median (s) | 70.0000 | 80.0000 [70.0000, 80.0000] | 80.0000 [70.0000, 80.0000] |
| Sleep bout median (s) | 120.0000 | 100.0000 [90.0000, 110.0000] | 95.0000 [80.0000, 139.0000] |

Aggregate distribution and circadian discrepancies:

| Model | Wake Wasserstein (s) | Sleep Wasserstein (s) | Sleep phase RMSE | Activity phase RMSE |
| --- | ---: | ---: | ---: | ---: |
| Phase + duration (pooled) | 22.2483 [14.9223, 27.9888] | 37.0237 [26.5492, 48.4924] | 0.0503 [0.0441, 0.0606] | 2.3355 [2.0401, 2.7474] |
| Phase + duration + population variation | 46.6155 [14.6485, 71.1021] | 49.9270 [28.7199, 157.3195] | 0.0539 [0.0444, 0.0690] | 2.4569 [2.0291, 2.9111] |

## Temporal preservation

| Metric | Recorded | Phase + duration (pooled) | Phase + duration + population variation |
| --- | ---: | ---: | ---: |
| Sleep autocorrelation, lag 1 | 0.9441 | 0.9443 [0.9421, 0.9456] | 0.9444 [0.9300, 0.9562] |
| Sleep autocorrelation, lag 6 | 0.7701 | 0.7720 [0.7619, 0.7766] | 0.7708 [0.7299, 0.8107] |
| Sleep autocorrelation, lag 60 | 0.5061 | 0.4130 [0.3931, 0.4322] | 0.4184 [0.3735, 0.4600] |
| Sleep autocorrelation, lag 360 | 0.1272 | 0.2208 [0.2077, 0.2520] | 0.2256 [0.2006, 0.2517] |

## Between-subject heterogeneity

The statistics below are descriptive spread across the development mice inside each weekly replicate population.

| Spread metric | Recorded | Pooled | Population variation |
| --- | ---: | ---: | ---: |
| Sleep fraction SD | 0.0468 | 0.0093 [0.0046, 0.0147] | 0.0288 [0.0071, 0.0662] |
| Sleep fraction range | 0.1273 | 0.0241 [0.0121, 0.0391] | 0.0735 [0.0185, 0.1687] |
| Mean activity SD | 2.3475 | 0.3719 [0.1865, 0.5882] | 1.0583 [0.2170, 2.2533] |
| State changes/hour SD | 2.8765 | 0.3815 [0.1532, 0.6419] | 2.0670 [1.0358, 3.7429] |
| Wake bout median SD (s) | 29.4746 | 4.3301 [0.0000, 5.0000] | 11.5723 [4.3301, 25.9808] |
| Sleep bout median SD (s) | 12.2474 | 8.2916 [4.3301, 11.6605] | 26.1304 [12.1517, 68.8180] |

See `individual-variation.svg` for per-subject sleep-fraction predictive intervals.

## Individual mice

| Subject | Metric | Recorded | Pooled | Population variation |
| --- | --- | ---: | ---: | ---: |
| 6 | Sleep fraction | 0.5156 | 0.6069 [0.5917, 0.6319] | 0.6146 [0.5069, 0.6635] |
| 6 | Changes/hour | 14.0293 | 9.6940 [9.1489, 10.2946] | 9.2703 [5.8498, 14.8069] |
| 6 | Wake bout median (s) | 60.0000 | 80.0000 [70.0000, 80.0000] | 80.0000 [60.0000, 130.0000] |
| 6 | Sleep bout median (s) | 110.0000 | 100.0000 [82.7500, 110.0000] | 100.0000 [70.0000, 182.5000] |
| 12 | Sleep fraction | 0.6429 | 0.6094 [0.5897, 0.6271] | 0.6310 [0.5732, 0.6848] |
| 12 | Changes/hour | 10.5851 | 9.7816 [9.1251, 10.3700] | 9.4889 [5.5972, 13.3959] |
| 12 | Wake bout median (s) | 60.0000 | 80.0000 [70.0000, 80.0000] | 75.0000 [65.5000, 110.0000] |
| 12 | Sleep bout median (s) | 120.0000 | 100.0000 [85.5000, 120.0000] | 110.0000 [80.0000, 244.5000] |
| 18 | Sleep fraction | 0.5908 | 0.6102 [0.5934, 0.6308] | 0.6119 [0.5566, 0.6803] |
| 18 | Changes/hour | 7.6963 | 9.6248 [9.1128, 10.0953] | 9.7950 [7.1562, 15.0570] |
| 18 | Wake bout median (s) | 130.0000 | 80.0000 [70.0000, 80.0000] | 70.0000 [60.0000, 100.0000] |
| 18 | Sleep bout median (s) | 120.0000 | 100.0000 [90.0000, 117.2500] | 90.0000 [65.5000, 154.5000] |
| 24 | Sleep fraction | 0.6110 | 0.6116 [0.5906, 0.6296] | 0.6076 [0.5370, 0.6547] |
| 24 | Changes/hour | 6.6082 | 9.3015 [8.3883, 9.9087] | 9.0691 [5.6661, 15.5343] |
| 24 | Wake bout median (s) | 100.0000 | 80.0000 [70.0000, 80.0000] | 70.0000 [60.0000, 114.5000] |
| 24 | Sleep bout median (s) | 90.0000 | 100.0000 [80.0000, 120.0000] | 95.0000 [60.0000, 166.2500] |

## Descriptive interpretation

Across the 6 declared heterogeneity summaries, the recorded value fell inside the pooled predictive interval for 0 and inside the population-variation interval for 4. This count is a diagnostic summary, not an acceptance score or hypothesis test.

The population mechanism should be considered useful only where it increases realistic between-subject spread without materially damaging the temporal and circadian structure already captured by the phase+duration model.

## Remaining limitations

- The empirical population contains only the training mice in this COMPASS experiment.
- Only two global state-dynamics offsets vary between synthetic individuals.
- The profile distribution is resampled rather than estimated as a continuous biological population distribution.
- Activity-emission parameters remain pooled and conditionally independent across epochs.
- Development subjects were previously inspected and cannot support pristine confirmatory claims.
- Behaviorally defined immobility is not sleep-stage physiology.

## Next experiment

If training-population offsets improve heterogeneity while preserving temporal fidelity, the next v0.1 question is whether the effect survives prospective subject-level validation restricted to training animals and whether activity emissions require their own explicit individual or serial dynamics. If not, the failure itself should determine which minimal population mechanism is tested next.
