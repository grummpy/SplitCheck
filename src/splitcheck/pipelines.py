"""Fit the same estimators under a leaky protocol and a correct one.

The logistic-regression pair uses age, BMI, systolic blood pressure, and a
target encoding of ``billing_code``. The leaky run encodes and scales on every
row, then draws a random split. The clean run holds out whole patients and fits
both transformers on the training patients only.

The forest pair leaves ``billing_code`` out. It uses the clinical features plus
the patient fingerprint, and the only protocol difference is the split. That
shows group leakage without target encoding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from splitcheck.constants import (
    AUDIT_FEATURES,
    CLINICAL_FEATURES,
    DEFAULT_SEED,
    FINGERPRINT_FEATURES,
    GROUP_COLUMN,
    LOGREG_MAX_ITER,
    N_TREES,
    TARGET_COLUMN,
    TARGET_ENCODING_SMOOTHING,
    TEST_SIZE,
    VISITS_PER_PATIENT,
)
from splitcheck.detectors import SplitAudit, audit_split
from splitcheck.synthetic import Fixture, build_fixtures, make_clinic_dataset


@dataclass(frozen=True)
class Score:
    key: str
    title: str
    split: str
    fit: str
    model: str
    accuracy: float
    f1: float
    roc_auc: float
    average_precision: float
    n_train: int
    n_test: int


@dataclass(frozen=True)
class FixtureCheck:
    name: str
    kind: str
    summary: str
    expect_exact: int
    expect_near: int
    expect_groups: int
    exact_duplicate_rows: int
    near_duplicate_rows: int
    overlapping_groups: int
    passed: bool


@dataclass(frozen=True)
class LabResult:
    seed: int
    dataset_rows: int
    dataset_patients: int
    visits_per_patient: int
    positive_rate: float
    dataset_sha256: str
    random_audit: SplitAudit
    group_audit: SplitAudit
    scores: tuple[Score, ...]
    scaler_only_auc_delta: float
    fixtures: tuple[FixtureCheck, ...]

    def score(self, key: str) -> Score:
        for item in self.scores:
            if item.key == key:
                return item
        raise KeyError(key)


def run_lab(seed: int = DEFAULT_SEED) -> LabResult:
    """Generate the clinic table from ``seed`` and score every protocol."""
    frame = make_clinic_dataset(seed)
    random_idx = _random_indices(frame, seed)
    group_idx = _group_indices(frame, seed)
    random_audit = _audit_indices(frame, random_idx)
    group_audit = _audit_indices(frame, group_idx)

    scores = (
        _logreg_score(frame, random_idx, "full", "leaky", "Leaky: random split, encode and scale on all rows"),
        _logreg_score(
            frame,
            random_idx,
            "train",
            "group_leak",
            "Group leak only: random split, encode and scale on train",
        ),
        _logreg_score(
            frame,
            group_idx,
            "full",
            "preprocess_leak",
            "Preprocess leak only: group split, encode and scale on all rows",
        ),
        _logreg_score(frame, group_idx, "train", "clean", "Clean: group split, encode and scale on train"),
        _clinical_score(frame, group_idx),
        _forest_score(frame, random_idx, "forest_random", "Forest: random split, clinical features and fingerprint", seed),
        _forest_score(frame, group_idx, "forest_group", "Forest: group split, clinical features and fingerprint", seed),
    )
    return LabResult(
        seed=seed,
        dataset_rows=int(len(frame)),
        dataset_patients=int(frame[GROUP_COLUMN].nunique()),
        visits_per_patient=VISITS_PER_PATIENT,
        positive_rate=float(frame[TARGET_COLUMN].mean()),
        dataset_sha256=dataset_fingerprint(frame),
        random_audit=random_audit,
        group_audit=group_audit,
        scores=scores,
        scaler_only_auc_delta=_scaler_only_delta(frame, group_idx),
        fixtures=tuple(_check_fixture(spec) for spec in build_fixtures(seed)),
    )


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Hash column names and values. Same seed, same hash, no Python ``hash()``."""
    hasher = hashlib.sha256()
    for name in frame.columns:
        hasher.update(str(name).encode())
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column):
            values = np.ascontiguousarray(column.to_numpy(dtype=np.float64))
            hasher.update(values.tobytes())
        else:
            hasher.update("\0".join(column.astype(str)).encode())
    return hasher.hexdigest()[:16]


def fit_target_encoding(
    codes: pd.Series, target: pd.Series, smoothing: float = TARGET_ENCODING_SMOOTHING
) -> tuple[pd.Series, float]:
    """Map each code to a shrunk mean of ``target``. Returns the map and the global mean."""
    labels = np.asarray(target, dtype=float)
    global_mean = float(labels.mean())
    grouped = pd.DataFrame({"code": codes.astype(str).to_numpy(), "y": labels}).groupby("code", sort=True)["y"]
    stats = grouped.agg(["sum", "count"])
    encoded = (stats["sum"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return encoded, global_mean


def transform_target_encoding(codes: pd.Series, encoded: pd.Series, global_mean: float) -> np.ndarray:
    """Apply a fitted encoding. Unseen codes become ``global_mean``."""
    mapped = pd.Series(codes.astype(str).to_numpy()).map(encoded)
    return mapped.fillna(global_mean).to_numpy(dtype=float)


def _random_indices(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=frame[TARGET_COLUMN].to_numpy(),
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _group_indices(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(frame, frame[TARGET_COLUMN], groups=frame[GROUP_COLUMN])
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _audit_indices(frame: pd.DataFrame, indices: tuple[np.ndarray, np.ndarray]) -> SplitAudit:
    train = frame.iloc[indices[0]].reset_index(drop=True)
    test = frame.iloc[indices[1]].reset_index(drop=True)
    return audit_split(train, test, feature_columns=AUDIT_FEATURES, group_column=GROUP_COLUMN)


def _design_matrix(frame: pd.DataFrame, encoded: np.ndarray) -> np.ndarray:
    clinical = frame[list(CLINICAL_FEATURES)].to_numpy(dtype=float)
    return np.column_stack([clinical, encoded])


def _logreg_score(
    frame: pd.DataFrame,
    indices: tuple[np.ndarray, np.ndarray],
    fit_on: str,
    key: str,
    title: str,
) -> Score:
    train_idx, test_idx = indices
    if fit_on == "full":
        encoding, global_mean = fit_target_encoding(frame["billing_code"], frame[TARGET_COLUMN])
        encoded = transform_target_encoding(frame["billing_code"], encoding, global_mean)
        raw = _design_matrix(frame, encoded)
        scaled = StandardScaler().fit(raw).transform(raw)
    elif fit_on == "train":
        encoding, global_mean = fit_target_encoding(
            frame.iloc[train_idx]["billing_code"], frame.iloc[train_idx][TARGET_COLUMN]
        )
        encoded = transform_target_encoding(frame["billing_code"], encoding, global_mean)
        raw = _design_matrix(frame, encoded)
        scaled = StandardScaler().fit(raw[train_idx]).transform(raw)
    else:
        raise ValueError(f"unknown fit population: {fit_on}")
    return _make_score(
        key=key,
        title=title,
        split="random" if key in {"leaky", "group_leak"} else "group",
        fit="all rows" if fit_on == "full" else "train only",
        model="logistic regression",
        model_factory=lambda: LogisticRegression(C=1.0, solver="lbfgs", max_iter=LOGREG_MAX_ITER),
        x_train=scaled[train_idx],
        y_train=frame[TARGET_COLUMN].to_numpy()[train_idx],
        x_test=scaled[test_idx],
        y_test=frame[TARGET_COLUMN].to_numpy()[test_idx],
    )


def _clinical_score(frame: pd.DataFrame, indices: tuple[np.ndarray, np.ndarray]) -> Score:
    train_idx, test_idx = indices
    raw = frame[list(CLINICAL_FEATURES)].to_numpy(dtype=float)
    scaled = StandardScaler().fit(raw[train_idx]).transform(raw)
    return _make_score(
        key="clinical_only",
        title="Reference: group split, clinical features only",
        split="group",
        fit="train only",
        model="logistic regression",
        model_factory=lambda: LogisticRegression(C=1.0, solver="lbfgs", max_iter=LOGREG_MAX_ITER),
        x_train=scaled[train_idx],
        y_train=frame[TARGET_COLUMN].to_numpy()[train_idx],
        x_test=scaled[test_idx],
        y_test=frame[TARGET_COLUMN].to_numpy()[test_idx],
    )


def _forest_score(
    frame: pd.DataFrame,
    indices: tuple[np.ndarray, np.ndarray],
    key: str,
    title: str,
    seed: int,
) -> Score:
    train_idx, test_idx = indices
    columns = list(CLINICAL_FEATURES + FINGERPRINT_FEATURES)
    raw = frame[columns].to_numpy(dtype=float)
    scaled = StandardScaler().fit(raw[train_idx]).transform(raw)
    return _make_score(
        key=key,
        title=title,
        split="random" if key == "forest_random" else "group",
        fit="train only",
        model="random forest",
        model_factory=lambda: RandomForestClassifier(
            n_estimators=N_TREES, random_state=seed, n_jobs=1
        ),
        x_train=scaled[train_idx],
        y_train=frame[TARGET_COLUMN].to_numpy()[train_idx],
        x_test=scaled[test_idx],
        y_test=frame[TARGET_COLUMN].to_numpy()[test_idx],
    )


def _scaler_only_delta(frame: pd.DataFrame, indices: tuple[np.ndarray, np.ndarray]) -> float:
    """ROC-AUC change from fitting the scaler on all rows, with an honest split and no encoding."""
    train_idx, test_idx = indices
    raw = frame[list(CLINICAL_FEATURES)].to_numpy(dtype=float)
    y_train = frame[TARGET_COLUMN].to_numpy()[train_idx]
    y_test = frame[TARGET_COLUMN].to_numpy()[test_idx]
    full = StandardScaler().fit(raw).transform(raw)
    train_only = StandardScaler().fit(raw[train_idx]).transform(raw)

    def auc(scaled: np.ndarray) -> float:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=LOGREG_MAX_ITER)
        model.fit(scaled[train_idx], y_train)
        proba = model.predict_proba(scaled[test_idx])[:, 1]
        return float(roc_auc_score(y_test, proba))

    return auc(full) - auc(train_only)


def _make_score(
    key: str,
    title: str,
    split: str,
    fit: str,
    model: str,
    model_factory,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Score:
    estimator = model_factory()
    estimator.fit(x_train, y_train)
    proba = estimator.predict_proba(x_test)[:, 1]
    predicted = (proba >= 0.5).astype(int)
    return Score(
        key=key,
        title=title,
        split=split,
        fit=fit,
        model=model,
        accuracy=float(accuracy_score(y_test, predicted)),
        f1=float(f1_score(y_test, predicted, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        average_precision=float(average_precision_score(y_test, proba)),
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
    )


def _check_fixture(spec: Fixture) -> FixtureCheck:
    audit = audit_split(
        spec.train,
        spec.test,
        feature_columns=spec.feature_columns,
        group_column=spec.group_column,
    )
    passed = (
        audit.exact_duplicate_rows == spec.expect_exact
        and audit.near_duplicate_rows == spec.expect_near
        and audit.overlapping_groups == spec.expect_groups
    )
    return FixtureCheck(
        name=spec.name,
        kind=spec.kind,
        summary=spec.summary,
        expect_exact=spec.expect_exact,
        expect_near=spec.expect_near,
        expect_groups=len(spec.expect_groups),
        exact_duplicate_rows=audit.exact_duplicate_rows,
        near_duplicate_rows=audit.near_duplicate_rows,
        overlapping_groups=len(audit.overlapping_groups),
        passed=passed,
    )
