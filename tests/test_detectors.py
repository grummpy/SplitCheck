"""Hand-built frames. These do not go through the fixture generator."""

import numpy as np
import pandas as pd
import pytest

from splitcheck.detectors import audit_split


def test_exact_copies_are_found_and_a_changed_label_does_not_hide_them():
    train = pd.DataFrame(
        {"x0": [0.0, 1.0, 2.0], "x1": [0.0, 1.0, 2.0], "y": [0, 1, 0]}
    )
    test = pd.DataFrame(
        {"x0": [2.0, 9.0], "x1": [2.0, 9.0], "y": [1, 1]}
    )
    audit = audit_split(train, test, feature_columns=["x0", "x1"])
    assert audit.exact_duplicate_rows == 1
    assert audit.near_duplicate_rows == 0
    assert audit.exact_pairs == ((0, 2),)


def test_near_copies_are_not_counted_as_exact():
    train = pd.DataFrame({"x0": [0.0, 4.0], "x1": [0.0, 4.0], "site": ["east", "east"]})
    test = pd.DataFrame(
        {"x0": [0.0 + 5e-4, 4.0], "x1": [0.0 + 5e-4, 3.0], "site": ["east", "west"]}
    )
    audit = audit_split(train, test, feature_columns=["x0", "x1", "site"])
    assert audit.exact_duplicate_rows == 0
    assert audit.near_duplicate_rows == 1
    assert audit.near_pairs[0][0] == 0
    assert audit.near_pairs[0][1] == 0


def test_categorical_mismatch_blocks_an_otherwise_exact_row():
    train = pd.DataFrame({"x0": [0.0], "site": ["east"]})
    test = pd.DataFrame({"x0": [0.0], "site": ["west"]})
    audit = audit_split(train, test, feature_columns=["x0", "site"])
    assert audit.exact_duplicate_rows == 0
    assert audit.near_duplicate_rows == 0
    assert audit.has_leakage is False


def test_categorical_keys_preserve_types_and_column_boundaries():
    train = pd.DataFrame({
        "value": [1.0, 2.0, 3.0, 4.0],
        "category": pd.Series([1, True, "1", "alpha\0beta"], dtype=object),
        "second": pd.Series(["x", "x", "x", "gamma"], dtype=object),
    })
    test = pd.DataFrame({
        "value": [1.0, 1.0, 2.0, 4.0, 4.0],
        "category": pd.Series(["1", 1, True, "alpha", "alpha\0beta"], dtype=object),
        "second": pd.Series(["x", "x", "x", "beta\0gamma", "gamma"], dtype=object),
    })

    audit = audit_split(train, test, feature_columns=["value", "category", "second"])

    # "1", 1, and True are different categorical values; embedded NUL bytes
    # cannot make two columns look like one serialized string key.
    assert audit.exact_pairs == ((1, 0), (2, 1), (4, 3))


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_tolerance_is_an_error(tolerance):
    frame = pd.DataFrame({"x0": [1.0, 2.0]})
    with pytest.raises(ValueError, match="finite"):
        audit_split(frame, frame, feature_columns=["x0"], near_atol=tolerance)


def test_nonfinite_numeric_feature_is_an_error():
    train = pd.DataFrame({"x0": [1.0, np.inf]})
    test = pd.DataFrame({"x0": [1.0, 2.0]})
    with pytest.raises(ValueError, match="finite values"):
        audit_split(train, test, feature_columns=["x0"])


def test_shared_groups_are_reported_when_rows_are_not_copies():
    train = pd.DataFrame({"x0": [0.0, 1.0, 2.0], "patient_id": ["A", "B", "C"]})
    test = pd.DataFrame({"x0": [10.0, 11.0, 12.0], "patient_id": ["B", "C", "D"]})
    audit = audit_split(train, test, feature_columns=["x0"], group_column="patient_id")
    assert audit.exact_duplicate_rows == 0
    assert audit.near_duplicate_rows == 0
    assert audit.overlapping_groups == ("B", "C")
    assert audit.test_rows_with_overlapping_group == 2
    assert audit.train_rows_with_overlapping_group == 2


def test_integer_group_ids_are_compared_as_strings():
    train = pd.DataFrame({"x0": [0.0], "patient_id": [7]})
    test = pd.DataFrame({"x0": [5.0], "patient_id": [7]})
    audit = audit_split(train, test, feature_columns=["x0"], group_column="patient_id")
    assert audit.overlapping_groups == ("7",)


def test_separated_rows_raise_no_alarm():
    train = pd.DataFrame({"x0": [0.0, 1.0], "x1": [0.0, 1.0], "patient_id": ["A", "B"]})
    test = pd.DataFrame({"x0": [20.0, 21.0], "x1": [20.0, 21.0], "patient_id": ["C", "D"]})
    audit = audit_split(train, test, feature_columns=["x0", "x1"], group_column="patient_id")
    assert audit.has_leakage is False
    assert audit.overlapping_groups == ()


def test_missing_column_is_an_error():
    frame = pd.DataFrame({"x0": [1.0]})
    with pytest.raises(ValueError, match="not in both frames"):
        audit_split(frame, frame, feature_columns=["missing"])


def test_empty_frame_is_an_error():
    train = pd.DataFrame({"x0": [1.0]})
    test = pd.DataFrame({"x0": []})
    with pytest.raises(ValueError, match="at least one row"):
        audit_split(train, test, feature_columns=["x0"])


def test_negative_tolerance_is_an_error():
    frame = pd.DataFrame({"x0": [1.0, 2.0]})
    with pytest.raises(ValueError, match="non-negative"):
        audit_split(frame, frame, feature_columns=["x0"], near_atol=-1.0)


def test_missing_feature_values_are_an_error():
    train = pd.DataFrame({"x0": [1.0, np.nan]})
    test = pd.DataFrame({"x0": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        audit_split(train, test, feature_columns=["x0"])
