# Tests

Automated tests cover the COMPASS adapter, source-category invariants,
configuration validation, model fitting and seeded generation, evaluation,
artifact production, and command-line behavior.

The test suite uses small generated fixtures and does not require network access
or the downloaded research dataset.

```console
python -m unittest discover -s tests -v
```
