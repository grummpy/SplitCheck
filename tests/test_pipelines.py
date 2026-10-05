"""Leaky scores should clear the clean scores, and the clean split should audit clean."""

from splitcheck.constants import DEFAULT_SEED
from splitcheck.report import render_report


def test_leaky_logistic_score_is_inflated_versus_clean(seeded_lab):
    result = seeded_lab
    leaky = result.score("leaky")
    clean = result.score("clean")
    group_leak = result.score("group_leak")
    preprocess = result.score("preprocess_leak")
    clinical = result.score("clinical_only")

    assert leaky.roc_auc >= 0.99
    assert group_leak.roc_auc >= 0.97
    assert preprocess.roc_auc >= 0.99
    assert clean.roc_auc <= 0.90
    assert leaky.roc_auc >= clean.roc_auc + 0.15
    assert abs(clean.roc_auc - clinical.roc_auc) < 0.05
    assert abs(result.scaler_only_auc_delta) < 0.02
    assert leaky.model == clean.model == "logistic regression"


def test_forest_random_split_beats_the_group_split(seeded_lab):
    result = seeded_lab
    leaked = result.score("forest_random")
    clean = result.score("forest_group")
    assert leaked.roc_auc >= clean.roc_auc + 0.15
    assert clean.roc_auc <= 0.90
    assert leaked.fit == clean.fit == "train only"


def test_random_split_is_flagged_and_group_split_is_not(seeded_lab):
    result = seeded_lab
    assert result.random_audit.overlapping_groups
    assert result.random_audit.test_rows_with_overlapping_group > 0
    assert result.random_audit.exact_duplicate_rows == 0
    assert result.random_audit.near_duplicate_rows == 0
    assert result.group_audit.overlapping_groups == ()
    assert result.group_audit.exact_duplicate_rows == 0
    assert result.group_audit.near_duplicate_rows == 0
    assert result.group_audit.has_leakage is False


def test_report_explains_the_gap_and_passes_every_fixture(seeded_lab):
    result = seeded_lab
    assert all(item.passed for item in result.fixtures)
    text = render_report(result)
    assert f"Seed: {DEFAULT_SEED}" in text
    assert "Why the leaky score is inflated" in text
    assert "Gap (leaky minus clean):" in text
    assert "FAIL" not in text
    assert text.count("PASS") == len(result.fixtures)
    assert result.dataset_sha256 in text
