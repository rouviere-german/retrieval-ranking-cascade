"""The pandera data contracts on the impression log, the catalog and ingested feedback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from retrieval_ranking_cascade.ml.contracts import (
    DataContractError,
    catalog_schema,
    feedback_schema,
    impressions_schema,
    validate,
)

pytestmark = pytest.mark.unit

N_USERS = 5
N_ITEMS = 12
SLOTS = 3


def _impressions(**overrides: Any) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "session_id": [0, 0, 0, 1, 1, 1],
            "user_id": [0, 0, 0, 1, 1, 1],
            "item_id": [3, 4, 5, 6, 7, 8],
            "position": [1, 2, 3, 1, 2, 3],
            "clicked": [1, 0, 0, 0, 1, 0],
            "randomized": [False] * 6,
            "day": [0, 0, 0, 2, 2, 2],
            "period": [0, 0, 0, 1, 1, 1],
        }
    ).astype(
        {
            "session_id": "int64",
            "user_id": "int64",
            "item_id": "int64",
            "position": "int64",
            "clicked": "int64",
            "day": "int64",
            "period": "int64",
        }
    )
    for name, value in overrides.items():
        frame[name] = value
    return frame


def _catalog(n_items: int = N_ITEMS) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": np.arange(n_items, dtype=np.int64),
            "category": (np.arange(n_items) % 3).astype(np.int64),
            "launch_day": np.zeros(n_items, dtype=np.int64),
        }
    )


def _feedback(**overrides: Any) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [2, 3],
            "position": [1, 2],
            "clicked": [1, 0],
            "source": ["0123456789ab", "0123456789ab"],
        }
    ).astype({"user_id": "int64", "item_id": "int64", "position": "int64", "clicked": "int64"})
    for name, value in overrides.items():
        frame[name] = value
    return frame


def _schema() -> pa.DataFrameSchema:
    return impressions_schema(n_users=N_USERS, n_items=N_ITEMS, slots=SLOTS)


# --- the happy path ---


def test_a_well_formed_log_and_catalog_pass() -> None:
    validate(_impressions(), _schema())
    validate(_catalog(), catalog_schema(n_items=N_ITEMS, n_categories=3))
    validate(_feedback(), feedback_schema(n_items=N_ITEMS, slots=SLOTS))


def test_validate_returns_the_frame_it_was_given() -> None:
    frame = _impressions()
    assert validate(frame, _schema()).shape == frame.shape


# --- impression log ---


@pytest.mark.parametrize(
    ("overrides", "column"),
    [
        ({"user_id": [0, 0, 0, 99, 99, 99]}, "user_id"),
        ({"item_id": [3, 4, 5, 6, 7, 99]}, "item_id"),
        ({"position": [0, 1, 2, 1, 2, 3]}, "position"),
        ({"clicked": [2, 0, 0, 0, 1, 0]}, "clicked"),
        ({"period": [0, 0, 0, 9, 9, 9]}, "period"),
        ({"day": [-1, 0, 0, 2, 2, 2]}, "day"),
    ],
)
def test_out_of_range_columns_are_rejected(overrides: dict[str, list[Any]], column: str) -> None:
    with pytest.raises(DataContractError) as info:
        validate(_impressions(**overrides), _schema())
    assert info.value.schema == "impressions"
    assert any(failure["column"] == column for failure in info.value.failures)


def test_a_session_that_spans_two_users_is_rejected() -> None:
    """Sessions are the ranking query unit, so one session must belong to one user."""
    with pytest.raises(DataContractError, match="impressions"):
        validate(_impressions(user_id=[0, 1, 0, 1, 1, 1]), _schema())


def test_non_contiguous_positions_are_rejected() -> None:
    with pytest.raises(DataContractError, match="impressions"):
        validate(_impressions(position=[1, 2, 5, 1, 2, 3]), _schema())


def test_the_same_item_twice_in_one_session_is_rejected() -> None:
    with pytest.raises(DataContractError, match="impressions"):
        validate(_impressions(item_id=[3, 3, 5, 6, 7, 8]), _schema())


def test_an_extra_or_missing_column_is_rejected() -> None:
    extra = _impressions()
    extra["surprise"] = 1
    with pytest.raises(DataContractError, match="impressions"):
        validate(extra, _schema())
    with pytest.raises(DataContractError, match="impressions"):
        validate(_impressions().drop(columns=["day"]), _schema())


def test_a_float_column_is_not_silently_coerced() -> None:
    """Coercion off is deliberate: a float position means the producer is broken."""
    frame = _impressions()
    frame["position"] = frame["position"].astype("float64")
    with pytest.raises(DataContractError, match="impressions"):
        validate(frame, _schema())


# --- catalog ---


def test_a_sparse_or_duplicated_catalog_is_rejected() -> None:
    schema = catalog_schema(n_items=N_ITEMS, n_categories=3)
    with pytest.raises(DataContractError, match="catalog"):
        validate(_catalog(N_ITEMS - 1), schema)
    duplicated = _catalog()
    duplicated.loc[0, "item_id"] = 1
    with pytest.raises(DataContractError, match="catalog"):
        validate(duplicated, schema)


def test_an_unknown_category_is_rejected() -> None:
    frame = _catalog()
    frame.loc[0, "category"] = 99
    with pytest.raises(DataContractError, match="catalog"):
        validate(frame, catalog_schema(n_items=N_ITEMS, n_categories=3))


# --- feedback ---


@pytest.mark.parametrize(
    "overrides",
    [
        {"item_id": [2, 999]},
        {"position": [0, 2]},
        {"position": [1, SLOTS + 5]},
        {"clicked": [1, 7]},
        {"source": ["not-hex-at-all", "0123456789ab"]},
    ],
)
def test_malformed_feedback_is_rejected(overrides: dict[str, list[Any]]) -> None:
    with pytest.raises(DataContractError) as info:
        validate(_feedback(**overrides), feedback_schema(n_items=N_ITEMS, slots=SLOTS))
    assert info.value.schema == "feedback"


# --- the error itself ---


def test_the_error_summarizes_failures_without_copying_any_row_values() -> None:
    """A contract error must not become a second copy of the data it rejected."""
    frame = _impressions(item_id=[3, 4, 5, 6, 7, 424_242])
    with pytest.raises(DataContractError) as info:
        validate(frame, _schema())
    error = info.value
    assert "424242" not in str(error)
    assert all(set(failure) == {"column", "check", "count"} for failure in error.failures)
    assert all(isinstance(failure["count"], int) for failure in error.failures)
    assert "424242" not in repr(error.failures)


def test_every_failing_check_is_reported_not_just_the_first() -> None:
    """Lazy validation: one run has to tell the producer everything that is wrong."""
    frame = _impressions(item_id=[3, 4, 5, 6, 7, 999], clicked=[5, 0, 0, 0, 1, 0])
    with pytest.raises(DataContractError) as info:
        validate(frame, _schema())
    columns = {failure["column"] for failure in info.value.failures}
    assert {"item_id", "clicked"} <= columns
    assert sum(int(failure["count"]) for failure in info.value.failures) >= 2
