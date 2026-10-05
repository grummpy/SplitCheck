"""Find duplicate rows and shared groups between a train frame and a test frame.

Duplicate checks use the feature columns only. The group check uses the group
column only. A patient who shows up on both sides is group leakage even when
none of their visits are copies.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from splitcheck.constants import EXACT_ATOL, NEAR_ATOL


@dataclass(frozen=True)
class SplitAudit:
    """What leaked between one train frame and one test frame."""

    n_train: int
    n_test: int
    feature_columns: tuple[str, ...]
    group_column: str | None
    exact_atol: float
    near_atol: float
    exact_duplicate_rows: int
    near_duplicate_rows: int
    exact_pairs: tuple[tuple[int, int], ...]
    near_pairs: tuple[tuple[int, int, float], ...]
    overlapping_groups: tuple[str, ...]
    train_rows_with_overlapping_group: int
    test_rows_with_overlapping_group: int

    @property
    def has_leakage(self) -> bool:
        return (
            self.exact_duplicate_rows > 0
            or self.near_duplicate_rows > 0
            or len(self.overlapping_groups) > 0
        )


def audit_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    group_column: str | None = None,
    near_atol: float = NEAR_ATOL,
    exact_atol: float = EXACT_ATOL,
) -> SplitAudit:
    """Audit ``test`` against ``train``.

    A test row is an exact duplicate when some train row matches every
    categorical feature and lies within ``exact_atol`` (Chebyshev) on the
    numeric features. It is a near-duplicate when it is not exact and the
    numeric distance is within ``near_atol``. Categorical features must match
    exactly in both cases. Group ids are compared as strings.
    """
    columns = tuple(feature_columns)
    _validate_inputs(train, test, columns, group_column, near_atol, exact_atol)
    numeric, categorical = _column_kinds(train, columns)
    _require_same_kinds(test, numeric, categorical)

    exact_pairs, near_pairs = _duplicate_pairs(
        train, test, numeric, categorical, exact_atol, near_atol
    )
    overlap, train_overlap_rows, test_overlap_rows = _group_overlap(train, test, group_column)
    return SplitAudit(
        n_train=int(len(train)),
        n_test=int(len(test)),
        feature_columns=columns,
        group_column=group_column,
        exact_atol=exact_atol,
        near_atol=near_atol,
        exact_duplicate_rows=len(exact_pairs),
        near_duplicate_rows=len(near_pairs),
        exact_pairs=tuple(exact_pairs),
        near_pairs=tuple(near_pairs),
        overlapping_groups=tuple(overlap),
        train_rows_with_overlapping_group=train_overlap_rows,
        test_rows_with_overlapping_group=test_overlap_rows,
    )


def _validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
    group_column: str | None,
    near_atol: float,
    exact_atol: float,
) -> None:
    if len(train) == 0 or len(test) == 0:
        raise ValueError("train and test must each contain at least one row")
    if not columns:
        raise ValueError("feature_columns must name at least one column")
    if not math.isfinite(exact_atol) or not math.isfinite(near_atol):
        raise ValueError("tolerances must be finite")
    if exact_atol < 0 or near_atol < 0:
        raise ValueError("tolerances must be non-negative")
    if exact_atol > near_atol:
        raise ValueError("exact_atol must be less than or equal to near_atol")
    missing = [name for name in columns if name not in train.columns or name not in test.columns]
    if missing:
        raise ValueError(f"feature column not in both frames: {missing[0]}")
    if group_column is not None and (group_column not in train.columns or group_column not in test.columns):
        raise ValueError(f"group column not in both frames: {group_column}")
    if train[list(columns)].isna().any().any() or test[list(columns)].isna().any().any():
        raise ValueError("feature columns must not contain missing values")


def _column_kinds(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for name in columns:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            categorical.append(name)
        else:
            numeric.append(name)
    return numeric, categorical


def _require_same_kinds(test: pd.DataFrame, numeric: list[str], categorical: list[str]) -> None:
    for name in numeric:
        if not pd.api.types.is_numeric_dtype(test[name]):
            raise ValueError(f"column {name} is numeric in train and non-numeric in test")
    for name in categorical:
        series = test[name]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            raise ValueError(f"column {name} is non-numeric in train and numeric in test")


def _typed_categorical_value(value: object) -> Hashable:
    """Make categorical equality explicit without flattening types to strings."""
    if isinstance(value, np.generic):
        value = value.item()
    value_type = (type(value).__module__, type(value).__qualname__)
    try:
        hash(value)
    except TypeError:
        raise ValueError("categorical feature values must be hashable") from None
    return (*value_type, value)  # type: ignore[return-value]


def _keys(frame: pd.DataFrame, categorical: list[str]) -> list[tuple[Hashable, ...]]:
    if not categorical:
        return [tuple()] * len(frame)
    return [
        tuple(_typed_categorical_value(value) for value in row)
        for row in frame[categorical].itertuples(index=False, name=None)
    ]


def _positions_by_key(keys: list[tuple[Hashable, ...]]) -> dict[tuple[Hashable, ...], list[int]]:
    positions: dict[tuple[Hashable, ...], list[int]] = {}
    for position, key in enumerate(keys):
        positions.setdefault(key, []).append(position)
    return positions


def _duplicate_pairs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    exact_atol: float,
    near_atol: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, float]]]:
    train_positions_by_key = _positions_by_key(_keys(train, categorical))
    test_positions_by_key = _positions_by_key(_keys(test, categorical))
    train_num = train[numeric].to_numpy(dtype=np.float64) if numeric else None
    test_num = test[numeric].to_numpy(dtype=np.float64) if numeric else None
    if train_num is not None and (not np.isfinite(train_num).all() or not np.isfinite(test_num).all()):
        raise ValueError("numeric feature columns must contain only finite values")
    exact: list[tuple[int, int]] = []
    near: list[tuple[int, int, float]] = []
    for key, test_pos in test_positions_by_key.items():
        train_pos = train_positions_by_key.get(key)
        if not train_pos:
            continue
        if train_num is None or test_num is None:
            for test_index in test_pos:
                exact.append((int(test_index), int(train_pos[0])))
            continue
        neighbors = NearestNeighbors(n_neighbors=1, metric="chebyshev", algorithm="brute")
        neighbors.fit(train_num[np.asarray(train_pos)])
        distances, indices = neighbors.kneighbors(test_num[np.asarray(test_pos)], return_distance=True)
        for local_i, test_index in enumerate(test_pos):
            distance = float(distances[local_i, 0])
            train_index = int(train_pos[int(indices[local_i, 0])])
            if distance <= exact_atol:
                exact.append((int(test_index), train_index))
            elif distance <= near_atol:
                near.append((int(test_index), train_index, distance))
    exact.sort()
    near.sort(key=lambda item: (item[0], item[1]))
    return exact, near


def _group_overlap(
    train: pd.DataFrame, test: pd.DataFrame, group_column: str | None
) -> tuple[list[str], int, int]:
    if group_column is None:
        return [], 0, 0
    train_groups = train[group_column].astype(str)
    test_groups = test[group_column].astype(str)
    overlap = sorted(set(train_groups) & set(test_groups))
    overlap_set = set(overlap)
    return (
        overlap,
        int(train_groups.isin(overlap_set).sum()),
        int(test_groups.isin(overlap_set).sum()),
    )
