# Tests

Automated tests cover the COMPASS adapter, source-category invariants,
configuration validation, model fitting and seeded generation, evaluation,
artifact production, and command-line behavior. Temporal-lab coverage includes
duration hazards, phase wrapping, missingness, template isolation, empirical
distribution metrics, subject-level summaries, replicate seeds, deterministic
artifacts, and manifest integrity.

The test suite uses small generated fixtures and does not require network access
or the downloaded research dataset.

Install the optional CalMS21 reader to run its archive-adapter tests:

```console
python -m pip install --editable '.[dev,calms21]'
```

```console
python -m unittest discover -s tests -v
```
