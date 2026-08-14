# Experiment report: v0.2-roche-spatial-baseline

## Interpretation boundary

This development experiment asks whether a compact persistent reflecting random walk reproduces three frozen image-space movement summaries. Similarity does not establish a biological navigation mechanism, intention, anxiety, motivation, or subjective state.

## Protocol

- Dataset: `roche-open-field-zenodo-8188683-v1`
- Fitting animals: 16459-67049, 16459-67053, 16459-67060, 16459-67067, 16459-67071, 16459-67074
- Development animals: 16459-67064, 16459-67078
- Replicates: 32
- Master seed: 1729
- Position: recorded `bodycentre`, no invented likelihood cutoff, interpolation, or smoothing
- Units: image pixels; no speed or physical-distance claims
- Registered metrics: path length with transition exposure, adjacent displacement, and absolute turning
- Arena bounds constrain the model but boundary, center, and occupancy metrics remain unregistered
- Synthetic evaluation copies only the recorded bodycentre availability mask, never recorded coordinate values
- Synthetic trajectories are discarded after compact evaluation.

## Model

```json
{
  "boundary_rule": "repeated-axis-reflection",
  "initial_position_fraction": [
    0.3482307434903927,
    0.2897374483092097
  ],
  "model_id": "persistent-reflecting-random-walk-v1",
  "positive_displacement": {
    "family": "lognormal",
    "log_mean": 0.5585528829820319,
    "log_std": 1.388594066511102
  },
  "stationary_probability": 0.0,
  "turning": {
    "family": "zero-mean-normal-radians",
    "std": 1.0657326937028495
  }
}
```

## Development comparison

- `path_length_signed_relative_difference` (ratio: (synthetic - recorded) / recorded): `16459-67064`=0.261665, `16459-67078`=0.167543
  - Median of the 2 animal medians: 0.214604
- `adjacent_displacement_wasserstein_px` (px): `16459-67064`=1.42491, `16459-67078`=1.69581
  - Median of the 2 animal medians: 1.56036
- `absolute_turning_wasserstein_radians` (radian): `16459-67064`=0.249071, `16459-67078`=0.279404
  - Median of the 2 animal medians: 0.264238

These summaries are descriptive model-development evidence from two designated development animals. Replicate frames are not treated as independent animals. No pass/fail threshold or confirmatory population claim is assigned.
