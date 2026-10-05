"""A second run with the documented seed must match the first."""

import pandas as pd

from splitcheck.constants import DEFAULT_SEED
from splitcheck.pipelines import dataset_fingerprint, run_lab
from splitcheck.report import render_report
from splitcheck.synthetic import make_clinic_dataset


def test_documented_seed_is_42():
    assert DEFAULT_SEED == 42


def test_same_seed_rebuilds_the_same_table():
    first = make_clinic_dataset(DEFAULT_SEED)
    second = make_clinic_dataset(DEFAULT_SEED)
    other = make_clinic_dataset(0)
    pd.testing.assert_frame_equal(first, second)
    assert dataset_fingerprint(first) == dataset_fingerprint(second)
    assert not first.equals(other)
    assert dataset_fingerprint(first) != dataset_fingerprint(other)


def test_same_seed_reprints_metrics_and_the_report():
    first = run_lab(DEFAULT_SEED)
    second = run_lab(DEFAULT_SEED)
    assert first.dataset_sha256 == second.dataset_sha256
    assert len(first.scores) == len(second.scores)
    for left, right in zip(first.scores, second.scores, strict=True):
        assert left.key == right.key
        assert left.accuracy == right.accuracy
        assert left.f1 == right.f1
        assert left.roc_auc == right.roc_auc
        assert left.average_precision == right.average_precision
    assert first.scaler_only_auc_delta == second.scaler_only_auc_delta
    assert first.random_audit.overlapping_groups == second.random_audit.overlapping_groups
    assert render_report(first) == render_report(second)


def test_a_different_seed_changes_the_table():
    assert dataset_fingerprint(make_clinic_dataset(DEFAULT_SEED)) != dataset_fingerprint(
        make_clinic_dataset(7)
    )
