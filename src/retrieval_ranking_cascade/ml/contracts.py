"""pandera data contracts for the impression log, the item catalog and ingested feedback.

Bounds depend on the world (number of users, items, slots), so the schemas are built by
functions. Validation is lazy (all failures at once) and failures are summarized as
(column, check, count) only: the failing values are never copied into the exception, so a contract
error cannot leak row contents into logs.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

__all__ = [
    "DataContractError",
    "catalog_schema",
    "feedback_schema",
    "impressions_schema",
    "validate",
]


class DataContractError(ValueError):
    """A frame violated its contract. ``failures`` holds (column, check, count) summaries."""

    def __init__(self, schema: str, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract {schema!r} violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.schema = schema
        self.failures = failures


def _int_column(low: int, high: int) -> pa.Column:
    return pa.Column("int64", pa.Check.in_range(low, high), nullable=False, coerce=False)


def impressions_schema(*, n_users: int, n_items: int, slots: int) -> pa.DataFrameSchema:
    """Contract for the impression log produced by the logging policy (one row per shown item)."""

    def one_user_and_period_per_session(frame: pd.DataFrame) -> bool:
        grouped = frame.groupby("session_id", sort=False)[["user_id", "period"]].nunique()
        return bool((grouped.to_numpy() == 1).all())

    def positions_are_contiguous(frame: pd.DataFrame) -> bool:
        grouped = frame.groupby("session_id", sort=False)["position"].agg(["min", "max", "size"])
        return bool(((grouped["min"] == 1) & (grouped["max"] == grouped["size"])).all())

    return pa.DataFrameSchema(
        {
            "session_id": _int_column(0, 2**31 - 1),
            "user_id": _int_column(0, n_users - 1),
            "item_id": _int_column(0, n_items - 1),
            "position": _int_column(1, slots),
            "clicked": pa.Column("int64", pa.Check.isin([0, 1]), nullable=False),
            "randomized": pa.Column("bool", nullable=False),
            "day": _int_column(0, 10_000),
            "period": pa.Column("int64", pa.Check.isin([0, 1, 2]), nullable=False),
        },
        checks=[
            pa.Check(one_user_and_period_per_session, name="one_user_and_period_per_session"),
            pa.Check(positions_are_contiguous, name="positions_are_contiguous"),
        ],
        unique=["session_id", "item_id"],
        strict=True,
        name="impressions",
    )


def catalog_schema(*, n_items: int, n_categories: int) -> pa.DataFrameSchema:
    """Contract for the item catalog: dense ids, known categories, non-negative launch days."""
    return pa.DataFrameSchema(
        {
            "item_id": pa.Column("int64", pa.Check.in_range(0, n_items - 1), unique=True),
            "category": _int_column(0, n_categories - 1),
            "launch_day": _int_column(0, 10_000),
        },
        checks=[pa.Check(lambda frame: len(frame) == n_items, name="dense_item_ids")],
        strict=True,
        name="catalog",
    )


def feedback_schema(*, n_items: int, slots: int) -> pa.DataFrameSchema:
    """Contract for feedback events accepted by the ingestion endpoint."""
    return pa.DataFrameSchema(
        {
            "user_id": _int_column(0, 2**31 - 1),
            "item_id": _int_column(0, n_items - 1),
            "position": _int_column(1, slots),
            "clicked": pa.Column("int64", pa.Check.isin([0, 1]), nullable=False),
            "source": pa.Column("str", pa.Check.str_matches(r"^[0-9a-f]{12}$")),
        },
        strict=True,
        name="feedback",
    )


def validate(frame: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate lazily and raise a summarized :class:`DataContractError` on any failure."""
    try:
        return schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases[["column", "check"]].astype(str)
        counts = cases.value_counts().sort_index()
        failures: list[dict[str, str | int]] = [
            {"column": column, "check": check, "count": int(count)}
            for (column, check), count in counts.items()
        ]
        raise DataContractError(str(schema.name), failures) from None
