"""The planted cases the lab claims to catch, with counts fixed in constants."""

import pandas as pd

from splitcheck.constants import (
    PLANTED_EXACT_COPIES,
    PLANTED_NEAR_COPIES,
    PLANTED_SHARED_GROUPS,
)
from splitcheck.detectors import audit_split
from splitcheck.synthetic import build_fixtures


LEAK_NAMES = {
    "exact_row_copies",
    "near_row_copies",
    "shared_patient_ids",
    "all_three_together",
}
CLEAN_NAMES = {
    "clean_separated_rows",
    "clean_repeated_measures",
}


def _by_name():
    return {spec.name: spec for spec in build_fixtures()}


def test_fixture_catalog_covers_every_planted_case_and_both_controls():
    specs = _by_name()
    assert set(specs) == LEAK_NAMES | CLEAN_NAMES
    assert PLANTED_EXACT_COPIES == 11
    assert PLANTED_NEAR_COPIES == 9
    assert PLANTED_SHARED_GROUPS == ("SHARE0", "SHARE1", "SHARE2", "SHARE3", "SHARE4")


def test_detector_finds_every_planted_leak():
    specs = _by_name()
    exact = specs["exact_row_copies"]
    near = specs["near_row_copies"]
    shared = specs["shared_patient_ids"]
    combined = specs["all_three_together"]

    assert exact.expect_exact == PLANTED_EXACT_COPIES
    assert near.expect_near == PLANTED_NEAR_COPIES
    assert shared.expect_groups == PLANTED_SHARED_GROUPS
    assert combined.expect_exact == PLANTED_EXACT_COPIES
    assert combined.expect_near == PLANTED_NEAR_COPIES
    assert combined.expect_groups == PLANTED_SHARED_GROUPS

    for spec in (exact, near, shared, combined):
        audit = audit_split(
            spec.train,
            spec.test,
            feature_columns=spec.feature_columns,
            group_column=spec.group_column,
        )
        assert audit.exact_duplicate_rows == spec.expect_exact
        assert audit.near_duplicate_rows == spec.expect_near
        assert audit.overlapping_groups == spec.expect_groups
        assert spec.kind == "leak"
        assert audit.has_leakage


def test_clean_controls_raise_no_alarm():
    for spec in build_fixtures():
        if spec.kind != "clean":
            continue
        assert spec.name in CLEAN_NAMES
        audit = audit_split(
            spec.train,
            spec.test,
            feature_columns=spec.feature_columns,
            group_column=spec.group_column,
        )
        assert audit.exact_duplicate_rows == 0
        assert audit.near_duplicate_rows == 0
        assert audit.overlapping_groups == ()
        assert audit.has_leakage is False


def test_fixtures_are_reproducible():
    first = build_fixtures(42)
    second = build_fixtures(42)
    for left, right in zip(first, second, strict=True):
        assert left.name == right.name
        pd_equal(left.train, right.train)
        pd_equal(left.test, right.test)


def pd_equal(left, right):
    pd.testing.assert_frame_equal(left, right)
