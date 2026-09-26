# Local data

Downloaded datasets live under `data/raw/` and are intentionally excluded from
Git. Source licenses and attribution still apply to local copies.

## COMPASS (v0.1)

Umwelt v0.1 uses the COMPASS dataset deposited at Zenodo under DOI
[`10.5281/zenodo.160344`](https://doi.org/10.5281/zenodo.160344). The deposited
data is dedicated under CC0 1.0. Dataset authorship, experimental context, file
roles, and published checksums remain part of Umwelt's provenance records.

The v0.1 workflow uses the paired weekly files containing 10-second PIR activity
measurements and behaviorally defined sleep labels for 24 mice. These sleep
labels are based on extended immobility; they are not silently relabeled as EEG
measurements or subjective states.

The `umwelt data download` command downloads the pinned COMPASS files. `umwelt
data verify` checks their exact file sizes and MD5 checksums against the Zenodo
record.

## Roche open field (v0.2 foundation)

The v0.2 foundation uses the pose and metadata files from Zenodo record
[`10.5281/zenodo.8188683`](https://doi.org/10.5281/zenodo.8188683), released
under CC BY 4.0. Raw videos are not needed for the trajectory-first foundation
and should not be downloaded by default.

There is no Roche download command in the CLI yet. From the repository root,
download the two required source files and extract the pose archive:

```console
mkdir -p data/raw/roche-open-field
curl --fail --location --continue-at - \
  --output data/raw/roche-open-field/data.zip \
  "https://zenodo.org/records/8188683/files/data.zip?download=1"
curl --fail --location \
  --output data/raw/roche-open-field/METADATA_ROCHE.csv \
  "https://zenodo.org/records/8188683/files/METADATA_ROCHE.csv?download=1"
unzip data/raw/roche-open-field/data.zip -d data/raw/roche-open-field
```

The expected local layout is:

```text
data/raw/roche-open-field/
├── METADATA_ROCHE.csv
├── data.zip
└── data/
    └── Yohimbine_Roche/
        └── 32 DeepLabCut CSV files
```

`data.zip` may be removed after its published size/MD5 and extracted members
have been verified; cataloging the extracted recordings does not require the
archive to remain present.

Zenodo publishes an exact byte size and MD5 for each required source file:

```text
data.zip            514028268 bytes  MD5 8b48f2060146c03d578d5d6971aa70e9
METADATA_ROCHE.csv       6266 bytes  MD5 096e21e4d319130370aa4bb244b670f8
```

Both properties can be checked locally with:

```console
stat --format='%n %s bytes' data/raw/roche-open-field/data.zip
stat --format='%n %s bytes' data/raw/roche-open-field/METADATA_ROCHE.csv
md5sum data/raw/roche-open-field/data.zip
md5sum data/raw/roche-open-field/METADATA_ROCHE.csv
```

Zenodo does not publish checksums for the 32 individual CSV members inside
`data.zip`. Umwelt therefore carries a
[32-entry extracted-file manifest](../src/umwelt/datasets/roche_open_field.py)
derived locally only after the archive matched Zenodo's published size and MD5.
Each entry pins the animal identifier, canonical member filename, extracted
size, exact individual frame count, and SHA-256. These SHA-256 values are Umwelt
integrity facts derived from the verified archive; they must not be attributed
to Zenodo.

The individual counts sum to 1,726,595 frames and range from 53,914 to 53,964.
That range is descriptive only: validation compares each CSV with its own
manifest count, size, and SHA-256, so a truncated file cannot pass merely by
remaining inside the global range.

`METADATA_ROCHE.csv` is semicolon-delimited UTF-8 with a byte-order mark. Its
pinned columns are `Animal ID`, `DLC file`, `Group`, `Dosage`, and `Video`. It
maps 32 unique animals to 32 pose files: eight `Control` recordings at dose `0`
and eight `Yohimbine` recordings at each of doses `1`, `3`, and `6` mg/kg.

Each extracted pose CSV uses the three-row DeepLabCut header and contains `(x,
y, likelihood)` triples for 13 mouse keypoints:

```text
nose, headcentre, neck, earl, earr, bodycentre, bcl, bcr, hipl, hipr,
tailbase, tailcentre, tailtip
```

Four additional tracked points describe arena landmarks and are not part of the
mouse pose: `tl`, `tr`, `bl`, and `br`.

The validated rows establish sequential frame indices. Umwelt represents their
DeepLabCut coordinates in the conventional image-pixel frame, with `+x` to the
right and `+y` downward, and records that orientation explicitly. Derived
movement headings are normalized to counterclockwise radians from `+x`. The
files do not establish sampling
frequency, pixel-to-centimeter calibration, or physical arena dimensions. Keep
those values unknown; do not derive physical speed or dimensions from
filenames, frame counts, or videos that were not included in the validated
source boundary.

## CalMS21 Task 1 candidate (v0.4)

The [CalMS21 deposit](https://data.caltech.edu/records/s0vdx-0k302) has DOI
`10.22002/D1.1991`. Its Task 1 archive includes pose and annotation JSON for
training and test sequences, plus derived task-programming features. Umwelt's
first social experiment uses pose coordinates and source confidence only;
expert action annotations and derived features are separate from geometry.

The [dataset paper](https://arxiv.org/html/2104.02710v4) declares CC-BY-NC-SA
for the data and links to the 2.0 license text. Store this archive only under
ignored `data/raw/` and retain the source attribution and license with any
local analysis. Future commercial use needs its own rights decision or a
different source. The Umwelt code's Apache-2.0 license does not relicense the
dataset.

```console
python -m pip install --editable '.[calms21]'
mkdir -p data/raw/calms21
curl --fail --location --retry 12 --retry-all-errors --continue-at - \
  --output data/raw/calms21/task1_classic_classification.zip \
  'https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1'
md5sum data/raw/calms21/task1_classic_classification.zip
umwelt social-audit \
  --archive data/raw/calms21/task1_classic_classification.zip \
  --output runs/v0.4-source-audit/report.json
```

The Caltech-published Task 1 MD5 is
`8a02654fddae28614ee24a6a082261b8`. Its verified byte size is
457,357,950. The audit command rejects a mismatched MD5, reports training
pose quality and pair geometry, and scans test sequence identifiers without
calculating test pose outcomes. The audit output is ignored by Git.
