"""Pre-result protocol facts for the first Umwelt v0.2 spatial experiment."""

from __future__ import annotations

import hashlib

from umwelt.datasets.roche_open_field import ROCHE_METADATA_MD5

ROCHE_CONTROL_SUBJECT_IDS = (
    "16459-67049",
    "16459-67053",
    "16459-67060",
    "16459-67064",
    "16459-67067",
    "16459-67071",
    "16459-67074",
    "16459-67078",
)
ROCHE_CONTROL_SPLIT_METHOD = "sha256-pinned-metadata-ranking-v1"
ROCHE_CONTROL_SPLIT_HASH_PREFIX = f"umwelt-v0.2-control-split-v1|{ROCHE_METADATA_MD5}"
ROCHE_CONTROL_DEVELOPMENT_COUNT = 2


def _split_digest(subject_id: str) -> str:
    payload = f"{ROCHE_CONTROL_SPLIT_HASH_PREFIX}|{subject_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_roche_control_split() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the outcome-blind fitting/development assignment.

    The two smallest SHA-256 digests become development animals. Role tuples
    retain source-metadata order so later iteration order is reproducible.
    """

    ranked = sorted(
        ROCHE_CONTROL_SUBJECT_IDS,
        key=lambda subject_id: (_split_digest(subject_id), subject_id),
    )
    development_set = set(ranked[:ROCHE_CONTROL_DEVELOPMENT_COUNT])
    fitting = tuple(
        subject_id
        for subject_id in ROCHE_CONTROL_SUBJECT_IDS
        if subject_id not in development_set
    )
    development = tuple(
        subject_id
        for subject_id in ROCHE_CONTROL_SUBJECT_IDS
        if subject_id in development_set
    )
    return fitting, development


ROCHE_CONTROL_FITTING_SUBJECTS, ROCHE_CONTROL_DEVELOPMENT_SUBJECTS = (
    derive_roche_control_split()
)
