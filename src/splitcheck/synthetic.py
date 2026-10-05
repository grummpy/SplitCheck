"""Synthetic clinic visits, plus hand-planted train/test fixtures.

The clinic table is the experiment. The fixtures are separate, smaller frames
with a known number of leaked rows so the detectors can be checked exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from splitcheck.constants import (
    DEFAULT_SEED,
    FINGERPRINT_NOISE,
    GROUP_COLUMN,
    N_FINGERPRINTS,
    N_PATIENTS,
    NEAR_OFFSET,
    NOISE,
    PLANTED_EXACT_COPIES,
    PLANTED_NEAR_COPIES,
    PLANTED_SHARED_GROUPS,
    SIGNAL,
    VISIT_NOISE,
    VISITS_PER_PATIENT,
)

FIXTURE_FEATURES = ("x0", "x1", "x2", "site")


@dataclass(frozen=True)
class Fixture:
    """One train/test pair with the leakage we expect a detector to report."""

    name: str
    kind: str
    summary: str
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: tuple[str, ...]
    group_column: str
    expect_exact: int
    expect_near: int
    expect_groups: tuple[str, ...]


def make_clinic_dataset(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Build the repeated-measures table used by the leaky-versus-clean lab.

    Each patient has one label, one billing code, and one fingerprint. Visits
    share those values and add noise to the clinical measurements. ``billing_code``
    is a patient key, not a diagnosis. The fingerprint is independent of the label.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for i in range(N_PATIENTS):
        risk = float(rng.normal())
        label = int(risk > 0)
        age = 55.0 + SIGNAL * risk + float(rng.normal(0.0, NOISE))
        bmi = 28.0 + 0.5 * SIGNAL * risk + float(rng.normal(0.0, NOISE * 0.5))
        systolic = 128.0 + SIGNAL * risk + float(rng.normal(0.0, NOISE * 1.4))
        fingerprint = rng.normal(size=N_FINGERPRINTS)
        patient_id = f"PT{i:04d}"
        for visit in range(VISITS_PER_PATIENT):
            row: dict[str, object] = {
                "patient_id": patient_id,
                "visit_id": f"{patient_id}-V{visit}",
                "age": age + float(rng.normal(0.0, VISIT_NOISE["age"])),
                "bmi": bmi + float(rng.normal(0.0, VISIT_NOISE["bmi"])),
                "systolic_bp": systolic + float(rng.normal(0.0, VISIT_NOISE["systolic_bp"])),
                "billing_code": f"B{i:04d}",
                "y": label,
            }
            for k in range(N_FINGERPRINTS):
                row[f"fp{k}"] = float(fingerprint[k] + rng.normal(0.0, FINGERPRINT_NOISE))
            rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values(["patient_id", "visit_id"]).reset_index(drop=True)


def build_fixtures(seed: int = DEFAULT_SEED) -> tuple[Fixture, ...]:
    """Return every planted case, including the clean controls.

    Counts are constants, not something the detector computes and then checks
    against itself. ``all_three_together`` plants the three leaks in one pair
    so a hit on one kind cannot hide a miss on another.
    """
    streams = np.random.SeedSequence(seed).spawn(6)
    exact = _copies_fixture(
        name="exact_row_copies",
        kind="leak",
        summary="Test rows copied from train with no change in the features.",
        rng=np.random.default_rng(streams[0]),
        mode="exact",
        n_copies=PLANTED_EXACT_COPIES,
    )
    near = _copies_fixture(
        name="near_row_copies",
        kind="leak",
        summary=f"Test rows copied from train with {NEAR_OFFSET:.0e} added to each numeric feature.",
        rng=np.random.default_rng(streams[1]),
        mode="near",
        n_copies=PLANTED_NEAR_COPIES,
    )
    shared = _shared_patients_fixture(np.random.default_rng(streams[2]))
    combined = _combined_fixture(np.random.default_rng(streams[3]))
    separated = _clean_separated_fixture(np.random.default_rng(streams[4]))
    repeated = _clean_repeated_measures_fixture(np.random.default_rng(streams[5]))
    return (exact, near, shared, combined, separated, repeated)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows).reset_index(drop=True)


def _random_row(rng: np.random.Generator, patient_id: str, loc: float) -> dict[str, object]:
    draws = rng.normal(loc=loc, scale=1.0, size=3)
    site = "east" if rng.random() < 0.5 else "west"
    return {
        "patient_id": patient_id,
        "site": site,
        "x0": float(draws[0]),
        "x1": float(draws[1]),
        "x2": float(draws[2]),
    }


def _block(rng: np.random.Generator, n: int, prefix: str, loc: float) -> list[dict[str, object]]:
    return [_random_row(rng, f"{prefix}{i:03d}", loc) for i in range(n)]


def _copy_row(source: dict[str, object], patient_id: str, jitter: float) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "site": source["site"],
        "x0": float(source["x0"]) + jitter,
        "x1": float(source["x1"]) + jitter,
        "x2": float(source["x2"]) + jitter,
    }


def _copies_fixture(
    name: str,
    kind: str,
    summary: str,
    rng: np.random.Generator,
    mode: str,
    n_copies: int,
) -> Fixture:
    train_rows = _block(rng, 40, "TR", loc=0.0)
    test_rows = _block(rng, 20, "TE", loc=20.0)
    jitter = 0.0 if mode == "exact" else NEAR_OFFSET
    prefix = "EX" if mode == "exact" else "NR"
    for i in range(n_copies):
        test_rows.append(_copy_row(train_rows[i], f"{prefix}{i:03d}", jitter))
    expect_exact = n_copies if mode == "exact" else 0
    expect_near = n_copies if mode == "near" else 0
    return Fixture(
        name=name,
        kind=kind,
        summary=summary,
        train=_frame(train_rows),
        test=_frame(test_rows),
        feature_columns=FIXTURE_FEATURES,
        group_column=GROUP_COLUMN,
        expect_exact=expect_exact,
        expect_near=expect_near,
        expect_groups=(),
    )


def _shared_visit(
    rng: np.random.Generator, patient_id: str, loc: float, site: str
) -> dict[str, object]:
    draws = rng.normal(loc=loc, scale=0.4, size=3)
    return {
        "patient_id": patient_id,
        "site": site,
        "x0": float(draws[0]),
        "x1": float(draws[1]),
        "x2": float(draws[2]),
    }


def _shared_patients_fixture(rng: np.random.Generator) -> Fixture:
    train_rows = _block(rng, 16, "TR", loc=0.0)
    test_rows = _block(rng, 16, "TE", loc=20.0)
    for group in PLANTED_SHARED_GROUPS:
        site = "east"
        train_rows.append(_shared_visit(rng, group, loc=0.0, site=site))
        train_rows.append(_shared_visit(rng, group, loc=0.0, site=site))
        test_rows.append(_shared_visit(rng, group, loc=20.0, site=site))
        test_rows.append(_shared_visit(rng, group, loc=20.0, site=site))
    return Fixture(
        name="shared_patient_ids",
        kind="leak",
        summary="The same patient ids appear in train and test, on rows that are not copies.",
        train=_frame(train_rows),
        test=_frame(test_rows),
        feature_columns=FIXTURE_FEATURES,
        group_column=GROUP_COLUMN,
        expect_exact=0,
        expect_near=0,
        expect_groups=PLANTED_SHARED_GROUPS,
    )


def _combined_fixture(rng: np.random.Generator) -> Fixture:
    train_rows = _block(rng, 40, "TR", loc=0.0)
    test_rows = _block(rng, 16, "TE", loc=20.0)
    for i in range(PLANTED_EXACT_COPIES):
        test_rows.append(_copy_row(train_rows[i], f"EX{i:03d}", 0.0))
    for i in range(PLANTED_NEAR_COPIES):
        source = train_rows[PLANTED_EXACT_COPIES + i]
        test_rows.append(_copy_row(source, f"NR{i:03d}", NEAR_OFFSET))
    for group in PLANTED_SHARED_GROUPS:
        train_rows.append(_shared_visit(rng, group, loc=0.0, site="west"))
        test_rows.append(_shared_visit(rng, group, loc=20.0, site="west"))
    return Fixture(
        name="all_three_together",
        kind="leak",
        summary="Exact copies, near-copies, and shared patient ids in one split.",
        train=_frame(train_rows),
        test=_frame(test_rows),
        feature_columns=FIXTURE_FEATURES,
        group_column=GROUP_COLUMN,
        expect_exact=PLANTED_EXACT_COPIES,
        expect_near=PLANTED_NEAR_COPIES,
        expect_groups=PLANTED_SHARED_GROUPS,
    )


def _clean_separated_fixture(rng: np.random.Generator) -> Fixture:
    return Fixture(
        name="clean_separated_rows",
        kind="clean",
        summary="Independent rows, disjoint ids, test features shifted far from train.",
        train=_frame(_block(rng, 30, "TR", loc=0.0)),
        test=_frame(_block(rng, 24, "TE", loc=20.0)),
        feature_columns=FIXTURE_FEATURES,
        group_column=GROUP_COLUMN,
        expect_exact=0,
        expect_near=0,
        expect_groups=(),
    )


def _clean_repeated_measures_fixture(rng: np.random.Generator) -> Fixture:
    """Several visits per patient, and every patient kept on one side of the split."""
    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for i in range(8):
        center = rng.normal(loc=0.0, scale=1.0, size=3)
        for visit in range(3):
            noise = rng.normal(scale=0.5, size=3)
            train_rows.append(
                {
                    "patient_id": f"A{i:02d}",
                    "site": "east",
                    "x0": float(center[0] + noise[0]),
                    "x1": float(center[1] + noise[1]),
                    "x2": float(center[2] + noise[2]),
                }
            )
    for i in range(6):
        center = rng.normal(loc=15.0, scale=1.0, size=3)
        for visit in range(3):
            noise = rng.normal(scale=0.5, size=3)
            test_rows.append(
                {
                    "patient_id": f"B{i:02d}",
                    "site": "west",
                    "x0": float(center[0] + noise[0]),
                    "x1": float(center[1] + noise[1]),
                    "x2": float(center[2] + noise[2]),
                }
            )
    return Fixture(
        name="clean_repeated_measures",
        kind="clean",
        summary="Repeated visits, split so no patient id is on both sides.",
        train=_frame(train_rows),
        test=_frame(test_rows),
        feature_columns=FIXTURE_FEATURES,
        group_column=GROUP_COLUMN,
        expect_exact=0,
        expect_near=0,
        expect_groups=(),
    )
