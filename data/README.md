# Local data

Downloaded datasets live under `data/raw/` and are intentionally excluded from Git.

Umwelt v0.1 uses the COMPASS dataset deposited at Zenodo under DOI
[`10.5281/zenodo.160344`](https://doi.org/10.5281/zenodo.160344). The deposited
data is dedicated under CC0 1.0. Dataset authorship, experimental context, file
roles, and published checksums remain part of Umwelt's provenance records.

The v0.1 workflow uses the paired weekly files containing 10-second PIR activity
measurements and behaviorally defined sleep labels for 24 mice. These sleep
labels are based on extended immobility; they are not silently relabeled as EEG
measurements or subjective states.

The command-line downloader is documented in the project README. It verifies
the exact file sizes and MD5 checksums published by Zenodo.
